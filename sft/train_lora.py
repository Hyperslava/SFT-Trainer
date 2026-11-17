#!/usr/bin/env python3
"""LoRA / QLoRA SFT for CodeLLMExp on a single consumer GPU.

Unlike koboldcpp (inference only, GGUF), training runs on the original
HuggingFace checkpoint with transformers + peft.

Example (RTX 4060 8 GB, 3-4B model, 4-bit QLoRA):
    python sft/train_lora.py \
        --model Qwen/Qwen3-4B-Instruct-2507 \
        --data-dir sft/data \
        --output-dir sft/out/qwen3-4b-codellmexp-lora \
        --load-4bit --merge

Loss is computed only on the assistant answer: the prompt tokens are masked
with -100. No trl dependency, so it works on any recent transformers version.
"""

from __future__ import annotations

import argparse
import gc
import inspect
import json
import math
import os
import random
import re
from pathlib import Path

import torch
import torch.nn.functional as F
import transformers
from torch.utils.checkpoint import checkpoint
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

try:  # transformers >= 4.45
    from transformers import AutoModelForImageTextToText
except ImportError:  # pragma: no cover
    AutoModelForImageTextToText = None

IGNORE_INDEX = -100
TEXT_PROJ_SUFFIXES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
VISION_MODULE_RE = re.compile(
    r"(^|\.)(visual|vision|vpm|image|video|multi_modal_projector|mm_projector|"
    r"merger|connector|projector)([._]|$)"
)


def detect_text_targets(model) -> str | None:
    """Regex for LoRA modules of the language tower only (vision-language models).

    Qwen3-VL -> model.language_model.layers.N..., SmolVLM -> model.text_model...,
    Llama-3.2-Vision -> language_model.model... . Returns None if the text tower
    cannot be told apart from the vision tower.
    """
    all_proj = [n for n, _ in model.named_modules() if n.endswith(TEXT_PROJ_SUFFIXES)]
    text_proj = [n for n in all_proj if not VISION_MODULE_RE.search(n)]
    if not text_proj:
        return None
    m = re.match(r"^(.*)\.\w+\.\d+\..+$", text_proj[0])
    if not m:
        return None
    suffix = "|".join(TEXT_PROJ_SUFFIXES)
    return rf"{re.escape(m.group(1))}\.\w+\.\d+\..*\.({suffix})"


class ChatSFTDataset(torch.utils.data.Dataset):
    """prompt/completion records -> token ids with a masked prompt.

    Everything is tokenized once up front: `lengths` feeds the length-grouped
    sampler and __getitem__ stays a list lookup.
    """

    def __init__(self, path: str, tokenizer, max_len: int, enable_thinking: bool | None,
                 limit: int = 0, seed: int = 0):
        with open(path, encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
        if limit and len(rows) > limit:
            rows = random.Random(seed).sample(rows, limit)
        self.tok = tokenizer
        self.max_len = max_len
        self.enable_thinking = enable_thinking
        self.eos_id = tokenizer.eos_token_id
        self.items = [self._encode(row) for row in rows]
        self.lengths = [len(it["input_ids"]) for it in self.items]

    def __len__(self) -> int:
        return len(self.items)

    def _render_prompt(self, row: dict) -> str:
        messages = [
            {"role": "system", "content": row["system"]},
            {"role": "user", "content": row["user"]},
        ]
        kwargs = dict(tokenize=False, add_generation_prompt=True)
        if self.enable_thinking is not None:
            kwargs["enable_thinking"] = self.enable_thinking
        try:
            return self.tok.apply_chat_template(messages, **kwargs)
        except TypeError:
            kwargs.pop("enable_thinking", None)
            return self.tok.apply_chat_template(messages, **kwargs)

    def _encode(self, row: dict) -> dict:
        prompt_ids = self.tok(self._render_prompt(row), add_special_tokens=False)["input_ids"]
        answer_ids = self.tok(row["assistant"], add_special_tokens=False)["input_ids"]
        if self.eos_id is not None:
            answer_ids = answer_ids + [self.eos_id]

        if len(answer_ids) >= self.max_len:
            answer_ids = answer_ids[: self.max_len - 1]
            if self.eos_id is not None:
                answer_ids.append(self.eos_id)

        room = self.max_len - len(answer_ids)
        if len(prompt_ids) > room:
            # keep the tail of the header: the code snippet matters most
            prompt_ids = prompt_ids[-room:] if room > 0 else []

        input_ids = prompt_ids + answer_ids
        labels = [IGNORE_INDEX] * len(prompt_ids) + answer_ids
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": [1] * len(input_ids),
        }

    def __getitem__(self, idx: int) -> dict:
        return self.items[idx]


class PadCollator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, features: list[dict]) -> dict:
        max_len = max(len(f["input_ids"]) for f in features)
        batch = {"input_ids": [], "labels": [], "attention_mask": []}
        for f in features:
            pad = max_len - len(f["input_ids"])
            batch["input_ids"].append(f["input_ids"] + [self.pad_token_id] * pad)
            batch["attention_mask"].append(f["attention_mask"] + [0] * pad)
            batch["labels"].append(f["labels"] + [IGNORE_INDEX] * pad)
        return {k: torch.tensor(v, dtype=torch.long) for k, v in batch.items()}


class LengthGroupedSampler(torch.utils.data.Sampler):
    """Shuffle, then sort by length inside chunks of `chunk_batches` batches.

    Neighbouring examples have similar length, so padding in a batch is small;
    batches themselves are shuffled so the order stays random.
    """

    def __init__(self, lengths: list[int], batch_size: int, seed: int, chunk_batches: int = 32):
        self.lengths = lengths
        self.bs = batch_size
        self.seed = seed
        self.chunk = batch_size * chunk_batches
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.lengths)

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1
        idx = list(range(len(self.lengths)))
        rng.shuffle(idx)
        batches = []
        for i in range(0, len(idx), self.chunk):
            part = sorted(idx[i : i + self.chunk], key=self.lengths.__getitem__, reverse=True)
            batches += [part[j : j + self.bs] for j in range(0, len(part), self.bs)]
        rng.shuffle(batches)
        return iter([i for b in batches for i in b])


LOSS_CHUNK = 1024  # answer tokens per lm_head + cross-entropy block


def _ce_sum(head, hidden, labels):
    return F.cross_entropy(head(hidden).float(), labels, reduction="sum")


class FastSFTTrainer(Trainer):
    """Trainer with length-grouped batches and an answer-only lm_head.

    The prompt (system + code) is masked, yet the stock loss still runs the
    lm_head over every position. With a ~250k vocabulary that is a large share
    of the step, so here the lm_head is applied only to positions that have a
    label, in checkpointed chunks (full logits are never kept in memory).
    """

    def __init__(self, *args, group_by_length: bool = True, answer_only_loss: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.group_by_length = group_by_length
        self.answer_only_loss = answer_only_loss

    def _get_train_sampler(self, *args, **kwargs):
        ds = self.train_dataset
        if not self.group_by_length or not hasattr(ds, "lengths"):
            return super()._get_train_sampler(*args, **kwargs)
        return LengthGroupedSampler(ds.lengths, self.args.per_device_train_batch_size, self.args.seed)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        base = model.get_base_model() if hasattr(model, "get_base_model") else model
        inner, head = getattr(base, "model", None), getattr(base, "lm_head", None)
        if not self.answer_only_loss or inner is None or head is None:
            return super().compute_loss(
                model, inputs, return_outputs=return_outputs, num_items_in_batch=num_items_in_batch
            )

        out = inner(
            input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], use_cache=False
        )
        labels = inputs["labels"][:, 1:]
        mask = labels != IGNORE_INDEX
        hidden, targets = out.last_hidden_state[:, :-1][mask], labels[mask]

        total = hidden.new_zeros((), dtype=torch.float32)
        for i in range(0, hidden.size(0), LOSS_CHUNK):
            total = total + checkpoint(
                _ce_sum, head, hidden[i : i + LOSS_CHUNK], targets[i : i + LOSS_CHUNK],
                use_reentrant=False,
            )
        if num_items_in_batch is not None and self.model_accepts_loss_kwargs:
            loss = total / num_items_in_batch
        else:
            loss = total / mask.sum().clamp(min=1)
        return (loss, {"loss": loss}) if return_outputs else loss


def estimate_params(config) -> float:
    """Rough parameter count of the language model (dense transformer formula)."""
    t = getattr(config, "text_config", None) or config
    h = getattr(t, "hidden_size", 0)
    inter = getattr(t, "intermediate_size", 0) or 4 * h
    layers = getattr(t, "num_hidden_layers", 0)
    return layers * (4 * h * h + 3 * h * inter) + getattr(t, "vocab_size", 0) * h


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="HF id or local dir of the base model")
    p.add_argument("--arch", default="auto", choices=["auto", "causal", "vl"],
                   help="'vl' for vision-language checkpoints (Qwen3-VL, Qwen2.5-VL, ...). "
                        "auto detects a vision_config and trains the language tower only")
    p.add_argument("--data-dir", default="sft/data")
    p.add_argument("--output-dir", default="sft/out/lora")
    p.add_argument("--epochs", type=float, default=2.0)
    p.add_argument("--max-steps", type=int, default=-1, help="debug: stop after N optimizer steps")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--bs", type=int, default=4, help="per-device train batch size")
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=2, help="dataloader workers (raise on a fast GPU)")
    p.add_argument("--max-len", type=int, default=2048)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--target-modules", default="all-linear",
                   help="'all-linear' or a comma separated list (q_proj,k_proj,...)")
    p.add_argument("--load-4bit", action="store_true", help="force QLoRA (needs bitsandbytes)")
    p.add_argument("--no-4bit", action="store_true", help="force bf16 LoRA")
    p.add_argument("--no-gradient-checkpointing", action="store_true", help="force checkpointing off")
    p.add_argument("--gradient-checkpointing", action="store_true", help="force checkpointing on")
    p.add_argument("--no-group-by-length", action="store_true",
                   help="plain random batches instead of length-grouped ones")
    p.add_argument("--no-answer-only-loss", action="store_true",
                   help="stock HF loss (lm_head over all positions)")
    p.add_argument("--eval-samples", type=int, default=256, help="val subset size (0 = all)")
    p.add_argument("--eval-bs", type=int, default=8)
    p.add_argument("--merge", action="store_true", help="also save the merged fp16/bf16 model")
    p.add_argument("--merge-dir", default=None, help="where to save the merged model")
    p.add_argument("--eval-steps", type=int, default=200)
    p.add_argument("--save-steps", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-thinking", action="store_true", default=True,
                   help="pass enable_thinking=False to the chat template (Qwen3)")
    p.add_argument("--allow-thinking", dest="no_thinking", action="store_false")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    use_cuda = torch.cuda.is_available()
    bf16 = use_cuda and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if bf16 else (torch.float16 if use_cuda else torch.float32)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    is_vl = args.arch == "vl" or (
        args.arch == "auto"
        and (hasattr(config, "vision_config") or hasattr(config, "vision_tower"))
    )
    if is_vl and AutoModelForImageTextToText is None:
        raise SystemExit(
            f"{config.model_type} is a vision-language model but this transformers build has no "
            "AutoModelForImageTextToText; upgrade transformers or pick a text-only base."
        )
    model_cls = AutoModelForImageTextToText if is_vl else AutoModelForCausalLM

    # QLoRA only pays off when bf16 weights do not fit; on small models the nf4
    # dequantisation just slows every step. Same for gradient checkpointing.
    n_params = estimate_params(config)
    if args.load_4bit and args.no_4bit:
        raise SystemExit("--load-4bit and --no-4bit are mutually exclusive")
    use_4bit = args.load_4bit or (not args.no_4bit and use_cuda and n_params > 3.0e9)
    if args.no_gradient_checkpointing and args.gradient_checkpointing:
        raise SystemExit("--gradient-checkpointing and --no-gradient-checkpointing are mutually exclusive")
    use_ckpt = args.gradient_checkpointing or (
        not args.no_gradient_checkpointing and (use_4bit or n_params > 1.2e9)
    )
    print(f"~{n_params / 1e9:.1f}B params -> 4-bit: {use_4bit}, gradient checkpointing: {use_ckpt}")
    if is_vl:
        print(f"vision-language model ({config.model_type}): only the language tower is trained")

    dtype_key = "dtype" if int(transformers.__version__.split(".")[0]) >= 5 else "torch_dtype"
    model_kwargs: dict = dict(trust_remote_code=True, **{dtype_key: dtype})
    if use_cuda:
        model_kwargs["device_map"] = {"": 0}
    if use_4bit:
        from transformers import BitsAndBytesConfig

        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
        )
        model_kwargs[dtype_key] = dtype

    model = model_cls.from_pretrained(args.model, **model_kwargs)
    model.config.use_cache = False

    if use_4bit:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=use_ckpt)
    elif use_ckpt:
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable()

    targets = args.target_modules
    if targets == "all-linear" and is_vl:
        detected = detect_text_targets(model)
        if detected:
            targets = detected
            print(f"LoRA target modules (language tower only): {targets}")
        else:
            print("WARN: could not isolate the language tower, falling back to all-linear")
    elif targets != "all-linear":
        targets = [t.strip() for t in targets.split(",") if t.strip()]

    lora = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=targets,
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    thinking = False if args.no_thinking else None
    train_path = Path(args.data_dir) / "train.jsonl"
    if not train_path.exists():
        raise SystemExit(
            f"{train_path} not found.\n"
            "run `make sft-data` (or `python sft/prepare_data.py --outdir sft/data`) first"
        )
    train_ds = ChatSFTDataset(train_path, tokenizer, args.max_len, thinking)
    val_path = Path(args.data_dir) / "val.jsonl"
    eval_ds = (
        ChatSFTDataset(val_path, tokenizer, args.max_len, thinking,
                       limit=args.eval_samples, seed=args.seed)
        if val_path.exists() else None
    )
    print(f"train examples: {len(train_ds)}  eval examples: {len(eval_ds) if eval_ds else 0}")

    args_dict = inspect.signature(TrainingArguments.__init__).parameters
    eval_key = "eval_strategy" if "eval_strategy" in args_dict else "evaluation_strategy"

    steps_per_epoch = max(1, math.ceil(len(train_ds) / (args.bs * args.grad_accum)))
    total_steps = args.max_steps if args.max_steps > 0 else int(steps_per_epoch * args.epochs)
    warmup = max(1, int(total_steps * 0.03))
    warmup_kwargs = (
        {"warmup_ratio": 0.03} if "warmup_ratio" in args_dict else {"warmup_steps": warmup}
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.bs,
        per_device_eval_batch_size=args.eval_bs,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        **warmup_kwargs,
        logging_steps=10,
        save_steps=args.save_steps,
        save_total_limit=2,
        bf16=bf16,
        fp16=use_cuda and not bf16,
        gradient_checkpointing=use_ckpt,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_8bit" if use_4bit else ("adamw_torch_fused" if use_cuda else "adamw_torch"),
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=args.num_workers,
        seed=args.seed,
        **{eval_key: "steps" if eval_ds else "no"},
        eval_steps=args.eval_steps if eval_ds else None,
        **({"include_num_input_tokens_seen": "non_padding"} if "include_num_input_tokens_seen" in args_dict else {}),
    )

    trainer = FastSFTTrainer(
        group_by_length=not args.no_group_by_length,
        answer_only_loss=not args.no_answer_only_loss,
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=PadCollator(tokenizer.pad_token_id),
    )
    trainer.train()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    tokenizer.save_pretrained(out)
    print(f"adapter saved to {out.resolve()}")

    if args.merge:
        merge_dir = Path(args.merge_dir or f"{args.output_dir}-merged")
        merge_dir.mkdir(parents=True, exist_ok=True)
        print("merging adapter into the base weights ...")
        merged = model.merge_and_unload()
        merged.save_pretrained(merge_dir, safe_serialization=True, max_shard_size="4GB")
        tokenizer.save_pretrained(merge_dir)
        del merged
        gc.collect()
        torch.cuda.empty_cache()
        print(f"merged model saved to {merge_dir.resolve()} (fp16/bf16, ready for GGUF conversion)")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
