#!/usr/bin/env python3
"""Одно окно для запуска SFT: заполнили параметры -> окно закрылось -> обучение в терминале.

Запуск (из активированного .venv или через make gui):

    .venv/bin/python sft/gui.py
    make gui

Кнопка «Начать обучение» собирает команду `sft/train_lora.py`, сохраняет
последние значения в ~/.config/codellmexp-sft/gui.json и заменяет процесс GUI
процессом обучения (os.execvp). Поэтому прогресс-бар и логи Trainer идут в том
же терминале, из которого запущено окно.

Если окно запущено не из терминала (двойным кликом), скрипт попробует открыть
эмулятор терминала (x-terminal-emulator, gnome-terminal, konsole, xfce4-terminal,
xterm) и запустить обучение в нём.
"""

from __future__ import annotations

import json
import math
import os
import shlex
import subprocess
import sys
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from shutil import which

try:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
except ModuleNotFoundError:  # system python without python3-tk
    sys.exit(
        "tkinter не найден в этом интерпретаторе.\n"
        "Запустите через venv:  make gui  (или .venv/bin/python sft/gui.py)\n"
        "На Debian/Ubuntu при необходимости: sudo apt install python3-tk"
    )

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = (
    Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    / "codellmexp-sft"
    / "gui.json"
)

MODEL_PRESETS = [
    "Qwen/Qwen3-4B-Instruct-2507",
    "Qwen/Qwen3-VL-2B-Instruct",
    "Qwen/Qwen2.5-Coder-3B-Instruct",
    "Qwen/Qwen2.5-Coder-1.5B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
]


def fmt_num(value: float) -> str:
    """1e-4 -> '0.0001', 2.0 -> '2'."""
    return f"{value:g}"


@dataclass
class TrainConfig:
    model: str = "Qwen/Qwen3-4B-Instruct-2507"
    arch: str = "auto"
    data_dir: str = "sft/data"
    output_dir: str = "sft/out/gui-run"
    epochs: float = 2.0
    bs: int = 4
    grad_accum: int = 4
    lr: float = 1e-4
    max_len: int = 2048
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    num_workers: int = 2
    eval_steps: int = 200
    save_steps: int = 200
    seed: int = 42
    quant: str = "auto"  # auto | 4bit | bf16
    grad_ckpt: str = "auto"  # auto | on | off
    eval_samples: int = 256
    group_by_length: bool = True
    answer_only_loss: bool = True
    merge: bool = True
    no_thinking: bool = True
    prepare_data: bool = False
    tasks: str = "audit,fix"

    def train_argv(self) -> list[str]:
        argv = [
            sys.executable,
            "sft/train_lora.py",
            "--model", self.model,
            "--arch", self.arch,
            "--data-dir", self.data_dir,
            "--output-dir", self.output_dir,
            "--epochs", fmt_num(self.epochs),
            "--max-len", str(self.max_len),
            "--bs", str(self.bs),
            "--grad-accum", str(self.grad_accum),
            "--lr", fmt_num(self.lr),
            "--lora-r", str(self.lora_r),
            "--lora-alpha", str(self.lora_alpha),
            "--lora-dropout", fmt_num(self.lora_dropout),
            "--num-workers", str(self.num_workers),
            "--eval-steps", str(self.eval_steps),
            "--save-steps", str(self.save_steps),
            "--seed", str(self.seed),
            "--eval-samples", str(self.eval_samples),
        ]
        if self.quant == "4bit":
            argv.append("--load-4bit")
        elif self.quant == "bf16":
            argv.append("--no-4bit")
        if self.grad_ckpt == "on":
            argv.append("--gradient-checkpointing")
        elif self.grad_ckpt == "off":
            argv.append("--no-gradient-checkpointing")
        if not self.group_by_length:
            argv.append("--no-group-by-length")
        if not self.answer_only_loss:
            argv.append("--no-answer-only-loss")
        if self.merge:
            argv.append("--merge")
        if not self.no_thinking:
            argv.append("--allow-thinking")
        return argv

    def prepare_argv(self) -> list[str]:
        return [
            sys.executable,
            "sft/prepare_data.py",
            "--outdir", self.data_dir,
            "--tasks", self.tasks,
        ]

    def shell_command(self) -> str:
        parts = []
        if self.prepare_data:
            parts.append(shlex.join(self.prepare_argv()))
        parts.append("exec " + shlex.join(self.train_argv()))
        return " && ".join(parts)


def load_state() -> TrainConfig:
    cfg = TrainConfig()
    try:
        data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return cfg
    known = {f.name for f in fields(TrainConfig)}
    for key, value in data.items():
        if key in known:
            setattr(cfg, key, value)
    return cfg


def save_state(cfg: TrainConfig) -> None:
    try:
        CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        CONFIG_PATH.write_text(
            json.dumps(asdict(cfg), ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except OSError:
        pass


def gpu_summary() -> str:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=3, check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return "GPU: nvidia-smi недоступен"
    lines = (out.stdout or "").strip().splitlines()
    if not lines:
        return "GPU: не найден"
    parts = [p.strip() for p in lines[0].split(",")]
    if len(parts) >= 3:
        name, total, used = parts[0], parts[1], parts[2]
        try:
            return f"GPU: {name} · свободно {int(total) - int(used)} / {total} MiB"
        except ValueError:
            pass
    return f"GPU: {lines[0]}"


def count_lines(path: Path) -> int | None:
    try:
        with path.open("rb") as f:
            return sum(chunk.count(b"\n") for chunk in iter(lambda: f.read(1 << 20), b""))
    except OSError:
        return None


def parse_int(raw: str, name: str, lo: int, hi: int, errors: list[str]) -> int | None:
    try:
        value = int(raw.strip())
    except ValueError:
        errors.append(f"{name}: ожидается целое число, получено «{raw}»")
        return None
    if not lo <= value <= hi:
        errors.append(f"{name}: должно быть в диапазоне {lo}…{hi}")
        return None
    return value


def parse_float(raw: str, name: str, lo: float, hi: float, errors: list[str]) -> float | None:
    try:
        value = float(raw.strip().replace(",", "."))
    except ValueError:
        errors.append(f"{name}: ожидается число, получено «{raw}»")
        return None
    if not lo <= value <= hi:
        errors.append(f"{name}: должно быть в диапазоне {lo:g}…{hi:g}")
        return None
    return value


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


class App:
    def __init__(self, root: tk.Tk, cfg: TrainConfig):
        self.root = root
        self.cfg = cfg
        self.vars: dict[str, tk.Variable] = {}
        self._preview_job: str | None = None
        self._examples_job: str | None = None
        self._examples: int | None = None

        root.title("CodeLLMExp — обучение LoRA")
        root.geometry("900x780")
        root.minsize(820, 700)

        self._build()
        self._refresh_examples()
        self._update_preview()

    # ---------- построение окна ----------
    def _var(self, key: str, value) -> tk.Variable:
        if isinstance(value, bool):
            var: tk.Variable = tk.BooleanVar(value=value)
        else:
            var = tk.StringVar(value=str(value))
        var.trace_add("write", lambda *_: self._schedule_preview())
        self.vars[key] = var
        return var

    def _entry(self, parent, row: int, col: int, label: str, key: str, width: int = 12) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=col, sticky="w", padx=(8, 6), pady=4)
        entry = ttk.Entry(parent, textvariable=self._var(key, getattr(self.cfg, key)), width=width)
        entry.grid(row=row, column=col + 1, sticky="we", padx=(0, 12), pady=4)

    def _build(self) -> None:
        pad = {"padx": 10, "pady": 6}

        header = ttk.Frame(self.root)
        header.pack(fill="x", **pad)
        ttk.Label(header, text="Параметры SFT", font=("TkDefaultFont", 13, "bold")).pack(side="left")
        ttk.Label(header, text=gpu_summary(), foreground="#555").pack(side="right")

        top = ttk.LabelFrame(self.root, text="Модель и данные")
        top.pack(fill="x", **pad)
        top.columnconfigure(1, weight=1)

        ttk.Label(top, text="Модель").grid(row=0, column=0, sticky="w", padx=(8, 6), pady=4)
        ttk.Combobox(
            top, textvariable=self._var("model", self.cfg.model), values=MODEL_PRESETS
        ).grid(row=0, column=1, sticky="we", padx=(0, 4), pady=4)
        ttk.Button(top, text="…", width=3, command=self._pick_model).grid(row=0, column=2, padx=(0, 8))

        ttk.Label(top, text="Данные (train.jsonl)").grid(row=1, column=0, sticky="w", padx=(8, 6), pady=4)
        data_row = ttk.Frame(top)
        data_row.grid(row=1, column=1, columnspan=2, sticky="we", padx=(0, 8), pady=4)
        data_row.columnconfigure(0, weight=1)
        ttk.Entry(data_row, textvariable=self._var("data_dir", self.cfg.data_dir)).grid(
            row=0, column=0, sticky="we"
        )
        ttk.Button(data_row, text="…", width=3, command=self._pick_data).grid(row=0, column=1, padx=(4, 0))

        ttk.Label(top, text="Результат").grid(row=2, column=0, sticky="w", padx=(8, 6), pady=4)
        out_row = ttk.Frame(top)
        out_row.grid(row=2, column=1, columnspan=2, sticky="we", padx=(0, 8), pady=4)
        out_row.columnconfigure(0, weight=1)
        ttk.Entry(out_row, textvariable=self._var("output_dir", self.cfg.output_dir)).grid(
            row=0, column=0, sticky="we"
        )
        ttk.Button(out_row, text="…", width=3, command=self._pick_output).grid(row=0, column=1, padx=(4, 0))

        params = ttk.LabelFrame(self.root, text="Гиперпараметры")
        params.pack(fill="x", **pad)
        params.columnconfigure(1, weight=1)
        params.columnconfigure(3, weight=1)

        self._entry(params, 0, 0, "Эпохи", "epochs")
        self._entry(params, 0, 2, "Learning rate", "lr")
        self._entry(params, 1, 0, "Batch size", "bs")
        self._entry(params, 1, 2, "Grad accumulation", "grad_accum")
        self._entry(params, 2, 0, "Max length", "max_len")
        self._entry(params, 2, 2, "Dataloader workers", "num_workers")
        self._entry(params, 3, 0, "LoRA r", "lora_r")
        self._entry(params, 3, 2, "LoRA alpha", "lora_alpha")
        self._entry(params, 4, 0, "LoRA dropout", "lora_dropout")
        self._entry(params, 4, 2, "Seed", "seed")
        self._entry(params, 5, 0, "Eval каждые N шагов", "eval_steps")
        self._entry(params, 5, 2, "Save каждые N шагов", "save_steps")
        self._entry(params, 6, 0, "Eval: примеров из val (0 = все)", "eval_samples")

        options = ttk.LabelFrame(self.root, text="Опции")
        options.pack(fill="x", **pad)
        left = ttk.Frame(options)
        left.pack(side="left", fill="x", expand=True, padx=6, pady=2)
        right = ttk.Frame(options)
        right.pack(side="left", fill="x", expand=True, padx=6, pady=2)

        for label, key, values in (
            ("Квантизация", "quant", ["auto", "4bit", "bf16"]),
            ("Gradient checkpointing", "grad_ckpt", ["auto", "on", "off"]),
        ):
            row = ttk.Frame(left)
            row.pack(anchor="w", pady=2)
            ttk.Label(row, text=label).pack(side="left")
            ttk.Combobox(
                row, textvariable=self._var(key, getattr(self.cfg, key)),
                values=values, state="readonly", width=8,
            ).pack(side="left", padx=(6, 0))
        ttk.Checkbutton(
            left, text="Merge после обучения", variable=self._var("merge", self.cfg.merge)
        ).pack(anchor="w", pady=2)
        ttk.Checkbutton(
            left, text="Батчи по близкой длине (меньше padding)",
            variable=self._var("group_by_length", self.cfg.group_by_length),
        ).pack(anchor="w", pady=2)
        ttk.Checkbutton(
            left, text="Loss/lm_head только по токенам ответа",
            variable=self._var("answer_only_loss", self.cfg.answer_only_loss),
        ).pack(anchor="w", pady=2)

        ttk.Checkbutton(
            right, text="Отключить thinking (Qwen3)", variable=self._var("no_thinking", self.cfg.no_thinking)
        ).pack(anchor="w", pady=2)
        ttk.Checkbutton(
            right, text="Подготовить данные перед обучением",
            variable=self._var("prepare_data", self.cfg.prepare_data),
        ).pack(anchor="w", pady=2)

        arch_row = ttk.Frame(right)
        arch_row.pack(anchor="w", pady=2)
        ttk.Label(arch_row, text="Архитектура").pack(side="left")
        ttk.Combobox(
            arch_row, textvariable=self._var("arch", self.cfg.arch),
            values=["auto", "causal", "vl"], state="readonly", width=8,
        ).pack(side="left", padx=(6, 0))

        preview = ttk.LabelFrame(self.root, text="Команда (её выполнит терминал)")
        preview.pack(fill="both", expand=True, **pad)
        self.preview = tk.Text(preview, height=6, wrap="word", font=("TkFixedFont", 9))
        self.preview.pack(side="left", fill="both", expand=True, padx=(6, 0), pady=6)
        scroll = ttk.Scrollbar(preview, command=self.preview.yview)
        scroll.pack(side="right", fill="y", pady=6, padx=(0, 6))
        self.preview.configure(yscrollcommand=scroll.set, state="disabled")

        bottom = ttk.Frame(self.root)
        bottom.pack(fill="x", **pad)
        self.status = ttk.Label(bottom, text="", foreground="#555")
        self.status.pack(side="left")
        ttk.Button(bottom, text="Отмена", command=self.root.destroy).pack(side="right")
        start = ttk.Button(bottom, text="Начать обучение", command=self.start)
        start.pack(side="right", padx=(0, 8))
        self.root.bind("<Control-Return>", lambda _e: self.start())

    # ---------- вспомогательное ----------
    def _pick_model(self) -> None:
        path = filedialog.askdirectory(title="Каталог с HF-чекпоинтом", initialdir=str(REPO_ROOT))
        if path:
            self.vars["model"].set(path)

    def _pick_data(self) -> None:
        path = filedialog.askdirectory(title="Каталог с train.jsonl", initialdir=str(REPO_ROOT / "sft"))
        if path:
            self.vars["data_dir"].set(path)
            self._refresh_examples()

    def _pick_output(self) -> None:
        path = filedialog.askdirectory(title="Куда сохранять чекпоинты", initialdir=str(REPO_ROOT / "sft"))
        if path:
            self.vars["output_dir"].set(path)

    def _schedule_preview(self) -> None:
        if self._preview_job:
            try:
                self.root.after_cancel(self._preview_job)
            except tk.TclError:
                pass
        if self._examples_job:
            try:
                self.root.after_cancel(self._examples_job)
            except tk.TclError:
                pass
        self._preview_job = self.root.after(150, self._update_preview)
        self._examples_job = self.root.after(400, self._refresh_examples)

    def _values(self) -> dict:
        return {key: var.get() for key, var in self.vars.items()}

    def _cfg_from_vars(self) -> TrainConfig:
        v = self._values()
        return TrainConfig(
            model=str(v["model"]).strip(),
            arch=str(v["arch"]),
            data_dir=str(v["data_dir"]).strip(),
            output_dir=str(v["output_dir"]).strip(),
            epochs=float(v["epochs"]),
            bs=int(v["bs"]),
            grad_accum=int(v["grad_accum"]),
            lr=float(v["lr"]),
            max_len=int(v["max_len"]),
            lora_r=int(v["lora_r"]),
            lora_alpha=int(v["lora_alpha"]),
            lora_dropout=float(v["lora_dropout"]),
            num_workers=int(v["num_workers"]),
            eval_steps=int(v["eval_steps"]),
            save_steps=int(v["save_steps"]),
            seed=int(v["seed"]),
            quant=str(v["quant"]),
            grad_ckpt=str(v["grad_ckpt"]),
            eval_samples=int(v["eval_samples"]),
            group_by_length=bool(v["group_by_length"]),
            answer_only_loss=bool(v["answer_only_loss"]),
            merge=bool(v["merge"]),
            no_thinking=bool(v["no_thinking"]),
            prepare_data=bool(v["prepare_data"]),
        )

    def _update_preview(self) -> None:
        try:
            text = f"cd {shlex.quote(str(REPO_ROOT))}\n{self._cfg_from_vars().shell_command()}"
        except (ValueError, KeyError):
            text = "…"
        self.preview.configure(state="normal")
        self.preview.delete("1.0", "end")
        self.preview.insert("1.0", text)
        self.preview.configure(state="disabled")

    def _refresh_examples(self) -> None:
        try:
            data_dir = resolve_path(str(self.vars["data_dir"].get()))
        except KeyError:
            return
        self._examples = count_lines(data_dir / "train.jsonl")
        self._update_status()

    def _update_status(self) -> None:
        try:
            bs = int(self.vars["bs"].get())
            accum = int(self.vars["grad_accum"].get())
            epochs = float(self.vars["epochs"].get())
        except ValueError:
            self.status.configure(text="Проверьте числовые поля")
            return
        if not self._examples:
            self.status.configure(
                text="train.jsonl не найден — включите «Подготовить данные» или укажите каталог"
            )
            return
        steps = max(1, math.ceil(self._examples / (bs * accum)))
        self.status.configure(
            text=f"{self._examples} примеров · ~{steps} шагов/эпоха · всего ~{int(steps * epochs)}"
        )

    # ---------- запуск ----------
    def start(self) -> None:
        errors: list[str] = []
        model = str(self.vars["model"].get()).strip()
        data_dir = str(self.vars["data_dir"].get()).strip().rstrip("/")
        output_dir = str(self.vars["output_dir"].get()).strip().rstrip("/")

        if not model:
            errors.append("Модель: укажите HF id или локальный каталог")
        if not data_dir:
            errors.append("Данные: укажите каталог")
        if not output_dir:
            errors.append("Результат: укажите каталог")

        epochs = parse_float(str(self.vars["epochs"].get()), "Эпохи", 0.01, 20, errors)
        bs = parse_int(str(self.vars["bs"].get()), "Batch size", 1, 512, errors)
        accum = parse_int(str(self.vars["grad_accum"].get()), "Grad accumulation", 1, 4096, errors)
        lr = parse_float(str(self.vars["lr"].get()), "Learning rate", 1e-7, 1e-1, errors)
        max_len = parse_int(str(self.vars["max_len"].get()), "Max length", 64, 32768, errors)
        lora_r = parse_int(str(self.vars["lora_r"].get()), "LoRA r", 1, 512, errors)
        lora_alpha = parse_int(str(self.vars["lora_alpha"].get()), "LoRA alpha", 1, 2048, errors)
        dropout = parse_float(str(self.vars["lora_dropout"].get()), "LoRA dropout", 0.0, 1.0, errors)
        workers = parse_int(str(self.vars["num_workers"].get()), "Dataloader workers", 0, 64, errors)
        eval_steps = parse_int(str(self.vars["eval_steps"].get()), "Eval steps", 1, 1_000_000, errors)
        save_steps = parse_int(str(self.vars["save_steps"].get()), "Save steps", 1, 1_000_000, errors)
        seed = parse_int(str(self.vars["seed"].get()), "Seed", 0, 2**31 - 1, errors)
        eval_samples = parse_int(str(self.vars["eval_samples"].get()), "Eval samples", 0, 1_000_000, errors)

        if errors:
            messagebox.showerror("Проверьте параметры", "\n".join(f"• {e}" for e in errors))
            return

        cfg = TrainConfig(
            model=model,
            arch=str(self.vars["arch"].get()),
            data_dir=data_dir,
            output_dir=output_dir,
            epochs=epochs, bs=bs, grad_accum=accum, lr=lr, max_len=max_len,
            lora_r=lora_r, lora_alpha=lora_alpha, lora_dropout=dropout,
            num_workers=workers, eval_steps=eval_steps, save_steps=save_steps, seed=seed,
            quant=str(self.vars["quant"].get()),
            grad_ckpt=str(self.vars["grad_ckpt"].get()),
            eval_samples=eval_samples,
            group_by_length=bool(self.vars["group_by_length"].get()),
            answer_only_loss=bool(self.vars["answer_only_loss"].get()),
            merge=bool(self.vars["merge"].get()),
            no_thinking=bool(self.vars["no_thinking"].get()),
            prepare_data=bool(self.vars["prepare_data"].get()),
        )

        train_file = resolve_path(cfg.data_dir) / "train.jsonl"
        if not train_file.exists() and not cfg.prepare_data:
            messagebox.showerror(
                "Нет данных",
                f"{train_file} не найден.\n\n"
                "Включите «Подготовить данные перед обучением» или выполните `make sft-data`.",
            )
            return

        out_path = resolve_path(cfg.output_dir)
        if out_path.exists() and out_path.is_dir() and any(out_path.iterdir()):
            if not messagebox.askyesno(
                "Каталог не пуст",
                f"{out_path}\n\nВ нём уже есть файлы (чекпоинты прошлого запуска).\n"
                "Продолжить и писать поверх?",
            ):
                return

        save_state(cfg)
        command = cfg.shell_command()
        self.root.destroy()
        launch_in_terminal(command)


def terminal_command(command: str) -> str:
    """Обучение + пауза, чтобы окно терминала не закрылось сразу после финиша."""
    return f"{command}; echo; read -r -p 'Готово. Enter, чтобы закрыть…' _"


def launch_in_terminal(command: str) -> None:
    os.chdir(REPO_ROOT)
    os.environ["PYTHONUNBUFFERED"] = "1"
    command = f"cd {shlex.quote(str(REPO_ROOT))} && {command}"

    if sys.stdout.isatty():
        # заменяем процесс GUI процессом обучения в текущем терминале
        os.execvp("bash", ["bash", "-c", command])

    shell_cmd = terminal_command(command)
    terminals = [
        ("x-terminal-emulator", ["-e", "bash", "-c", shell_cmd]),
        ("gnome-terminal", ["--", "bash", "-c", shell_cmd]),
        ("konsole", ["-e", "bash", "-c", shell_cmd]),
        ("xfce4-terminal", ["-x", "bash", "-c", shell_cmd]),
        ("xterm", ["-e", "bash", "-c", shell_cmd]),
    ]
    for name, args in terminals:
        if which(name):
            os.execvp(name, [name, *args])

    log = REPO_ROOT / "sft" / "out" / "gui-run.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w") as handle:
        code = subprocess.call(["bash", "-c", command], stdout=handle, stderr=subprocess.STDOUT)
    print(f"Терминал не найден. Лог: {log} (exit {code})")


def main() -> None:
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        sys.exit(
            f"Не удалось открыть окно: {exc}\n"
            "Нужна графическая сессия (DISPLAY). Без неё используйте make train …"
        )
    try:
        style = ttk.Style(root)
        if "clam" in style.theme_names():
            style.theme_use("clam")
    except tk.TclError:
        pass
    App(root, load_state())
    root.mainloop()


if __name__ == "__main__":
    main()
