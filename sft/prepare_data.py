#!/usr/bin/env python3
"""Convert CodeLLMExp.jsonl into SFT chat records (prompt/completion per task).

Fixes applied to the raw dataset:
  * markdown code fences are stripped;
  * `vulnerable_lines` are re-based: in the raw file the numbers include the
    opening ``` line, so fenced snippets need -1;
  * records with invalid line ranges (start=end=0) are dropped;
  * explanations with `potential_impact` are normalised to `impact`;
  * language labels are lower-cased (the raw file mixes "Java"/"java");
  * train/val/test split is grouped by snippet hash, so the many augmented
    variants of the same code never leak across splits.

Output: <outdir>/{train,val,test}.jsonl with fields
  id, task, language, system, user, assistant, meta
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

FENCE_RE = re.compile(r"^\s*```[A-Za-z0-9_+\-]*\r?\n(.*?)\r?\n?\s*```\s*$", re.S)

DEFAULT_INPUTS = ("data/codellmexp/CodeLLMExp.jsonl", "CodeLLMExp.jsonl")

LANG_NAME = {"python": "python", "java": "java", "c": "c"}

SYSTEM_AUDIT = (
    "You are a security code auditor specialised in the CWE Top 25. "
    "Given one code snippet, identify the vulnerability, localise it and explain it. "
    "Reply with a single JSON object and nothing else, with exactly these keys:\n"
    '{"cwe": "CWE-XXX", "lines": [start, end], "root_cause": "...", '
    '"impact": "...", "mitigation": "..."}\n'
    "Line numbers are 1-based and refer to the snippet itself, not counting the "
    "markdown fences."
)

SYSTEM_FIX = (
    "You are a secure coding assistant. Rewrite the given vulnerable code so that "
    "the vulnerability is removed while the intended functionality is preserved. "
    "Return only the corrected code in a single fenced code block and nothing else."
)


def strip_fences(text: str) -> tuple[str, bool]:
    """Return (code, was_fenced)."""
    m = FENCE_RE.match(text or "")
    if m:
        return m.group(1).rstrip("\n"), True
    return (text or "").strip("\n"), False


def explanation_fields(exp: dict) -> tuple[str, str, str]:
    root = (exp.get("root_cause") or "").strip()
    impact = (exp.get("impact") or exp.get("potential_impact") or "").strip()
    mitigation = (exp.get("mitigation") or "").strip()
    extra = (exp.get("additional_info") or "").strip()
    if extra:
        root = f"{root} {extra}".strip()
    return root, impact, mitigation


def build_examples(rec: dict, tasks: list[str]) -> list[dict]:
    lang = str(rec.get("language", "")).lower()
    if lang not in LANG_NAME:
        return []

    code, fenced = strip_fences(rec["code"])
    if not code.strip():
        return []

    cwe_list = rec.get("cwe") or []
    if isinstance(cwe_list, str):
        cwe_list = [cwe_list]
    cwe = cwe_list[0] if cwe_list else "CWE-000"

    vl = rec.get("vulnerable_lines") or {}
    start, end = int(vl.get("start", 0)), int(vl.get("end", 0))
    if fenced:  # the raw numbering starts at the ``` line
        start, end = start - 1, end - 1

    n_lines = len(code.split("\n"))
    lines_ok = 1 <= start <= end <= n_lines
    if lines_ok:
        # snap the range to the snippet (some annotations overshoot by 1 line)
        end = min(end, n_lines)

    root, impact, mitigation = explanation_fields(rec.get("explanation") or {})
    fixed, _ = strip_fences(rec.get("fixed_code") or "")
    fixed_ok = bool(fixed.strip()) and fixed.strip() != code.strip()

    out: list[dict] = []
    common = {"id": rec.get("id"), "language": lang, "source": rec.get("source_dataset")}

    if "audit" in tasks and lines_ok and root and impact and mitigation:
        target = {
            "cwe": cwe,
            "lines": [start, end],
            "root_cause": root,
            "impact": impact,
            "mitigation": mitigation,
        }
        out.append(
            {
                **common,
                "task": "audit",
                "system": SYSTEM_AUDIT,
                "user": (
                    f"Audit the following {lang} code for security vulnerabilities:\n\n"
                    f"```{lang}\n{code}\n```"
                ),
                "assistant": json.dumps(target, ensure_ascii=False),
                "meta": {"cwe": cwe, "lines": [start, end], "rate": rec.get("rate")},
            }
        )

    if "fix" in tasks and fixed_ok:
        out.append(
            {
                **common,
                "task": "fix",
                "system": SYSTEM_FIX,
                "user": (
                    f"Fix the security vulnerability in the following {lang} code:\n\n"
                    f"```{lang}\n{code}\n```"
                ),
                "assistant": f"```{lang}\n{fixed}\n```",
                "meta": {"cwe": cwe, "rate": rec.get("rate")},
            }
        )

    return out


def group_key(rec: dict) -> str:
    code, _ = strip_fences(rec["code"])
    cwe_list = rec.get("cwe") or ["?"]
    cwe = cwe_list[0] if isinstance(cwe_list, list) else cwe_list
    return hashlib.sha1(f"{code}\x00{cwe}".encode()).hexdigest()


def split_groups(keys: list[str], val: float, test: float, seed: int) -> dict[str, str]:
    keys = sorted(keys)
    random.Random(seed).shuffle(keys)
    n = len(keys)
    n_test = int(n * test)
    n_val = int(n * val)
    assign = {}
    for i, k in enumerate(keys):
        if i < n_test:
            assign[k] = "test"
        elif i < n_test + n_val:
            assign[k] = "val"
        else:
            assign[k] = "train"
    return assign


def resolve_input(value: str | None) -> Path:
    if value:
        path = Path(value)
        if not path.exists():
            raise SystemExit(
                f"dataset not found: {path}\n"
                "run `make data` to unpack assets/codellmexp.zip, or pass --input"
            )
        return path
    for candidate in DEFAULT_INPUTS:
        if Path(candidate).exists():
            return Path(candidate)
    raise SystemExit(
        "CodeLLMExp.jsonl not found.\n"
        "run `make data` to unpack assets/codellmexp.zip into data/codellmexp/,\n"
        "or pass --input /path/to/CodeLLMExp.jsonl"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", default=None,
                    help="path to CodeLLMExp.jsonl (default: data/codellmexp/CodeLLMExp.jsonl)")
    ap.add_argument("--outdir", default="sft/data")
    ap.add_argument("--tasks", default="audit,fix", help="comma separated: audit,fix")
    ap.add_argument("--min-rate", type=int, default=4, help="drop records rated below this")
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--test-frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=0, help="debug: use only N first records")
    args = ap.parse_args()

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    unknown = set(tasks) - {"audit", "fix"}
    if unknown:
        raise SystemExit(f"unknown tasks: {sorted(unknown)}")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    records = []
    input_path = resolve_input(args.input)
    print(f"input: {input_path}")
    with open(input_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
            if args.limit and len(records) >= args.limit:
                break

    stats = Counter()
    by_group: dict[str, list[dict]] = defaultdict(list)
    seen = set()

    for rec in records:
        stats["raw"] += 1
        if (rec.get("rate") or 0) < args.min_rate:
            stats["dropped_rate"] += 1
            continue

        code, fenced = strip_fences(rec["code"])
        vl = rec.get("vulnerable_lines") or {}
        s, e = int(vl.get("start", 0)), int(vl.get("end", 0))
        if fenced:
            s, e = s - 1, e - 1
        if not (1 <= s <= e <= len(code.split("\n"))):
            stats["dropped_bad_lines"] += 1

        examples = build_examples(rec, tasks)
        if not examples:
            stats["dropped_no_example"] += 1
            continue

        key = group_key(rec)
        for ex in examples:
            sig = (ex["task"], ex["user"], ex["assistant"])
            if sig in seen:
                stats["dropped_duplicate"] += 1
                continue
            seen.add(sig)
            by_group[key].append(ex)

    assign = split_groups(list(by_group), args.val_frac, args.test_frac, args.seed)
    files = {name: open(outdir / f"{name}.jsonl", "w", encoding="utf-8") for name in ("train", "val", "test")}
    per_split = {name: Counter() for name in files}
    try:
        for key, examples in by_group.items():
            split = assign[key]
            for ex in examples:
                files[split].write(json.dumps(ex, ensure_ascii=False) + "\n")
                per_split[split][ex["task"]] += 1
                stats[f"kept_{ex['task']}"] += 1
    finally:
        for f in files.values():
            f.close()

    print(f"groups={len(by_group)}  records={dict(stats)}")
    for name in ("train", "val", "test"):
        print(f"  {name}: {sum(per_split[name].values())} examples {dict(per_split[name])}")
    print(f"written to {outdir.resolve()}")


if __name__ == "__main__":
    main()
