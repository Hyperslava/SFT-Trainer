#!/usr/bin/env python3
"""Score a model served by koboldcpp (OpenAI-compatible API) on the holdout split.

    python sft/evaluate.py --data sft/data/test.jsonl --limit 100
    python sft/evaluate.py --endpoint http://127.0.0.1:5001/v1/chat/completions

Metrics for the `audit` task: JSON parse rate, CWE exact match, line-range
overlap / exact match. For `fix`: share of answers that contain a fenced block.
"""

from __future__ import annotations

import argparse
import json
import re
import urllib.request
from pathlib import Path

FENCE = re.compile(r"```[A-Za-z0-9_+\-]*\s*\n(.*?)```", re.S)


def chat(endpoint: str, system: str, user: str, max_tokens: int, model: str) -> str:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": max_tokens,
        "stream": False,
    }
    req = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        data = json.loads(resp.read())
    return data["choices"][0]["message"]["content"]


def parse_audit(text: str) -> dict | None:
    text = text.strip()
    m = FENCE.search(text)
    if m:
        text = m.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                return None
    return None


def overlap(a: list[int], b: list[int]) -> float:
    lo, hi = max(a[0], b[0]), min(a[1], b[1])
    inter = max(0, hi - lo + 1)
    union = max(a[1], b[1]) - min(a[0], b[0]) + 1
    return inter / union if union > 0 else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", default="http://127.0.0.1:5001/v1/chat/completions")
    ap.add_argument("--model", default="koboldcpp")
    ap.add_argument("--data", default="sft/data/test.jsonl")
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--task", default="audit", choices=["audit", "fix"])
    args = ap.parse_args()

    rows = [json.loads(l) for l in Path(args.data).read_text(encoding="utf-8").splitlines() if l.strip()]
    rows = [r for r in rows if r["task"] == args.task][: args.limit]
    print(f"evaluating {len(rows)} `{args.task}` examples against {args.endpoint}")

    parsed = cwe_hits = exact = 0
    iou_sum = 0.0
    fenced = 0
    for i, row in enumerate(rows, 1):
        answer = chat(args.endpoint, row["system"], row["user"], args.max_tokens, args.model)
        if args.task == "fix":
            if FENCE.search(answer):
                fenced += 1
        else:
            pred = parse_audit(answer)
            if pred is not None:
                parsed += 1
                cwe_hits += pred.get("cwe") == row["meta"]["cwe"]
                got = pred.get("lines")
                if isinstance(got, list) and len(got) == 2 and all(isinstance(x, int) for x in got):
                    iou = overlap(sorted(got), sorted(row["meta"]["lines"]))
                    iou_sum += iou
                    exact += iou == 1.0
        if i % 10 == 0:
            print(f"  {i}/{len(rows)}")

    n = max(len(rows), 1)
    if args.task == "fix":
        print(f"answers containing a code block: {fenced}/{n} = {fenced / n:.1%}")
    else:
        print(f"JSON parsed:     {parsed}/{n} = {parsed / n:.1%}")
        print(f"CWE exact match: {cwe_hits}/{n} = {cwe_hits / n:.1%}")
        print(f"line IoU (mean): {iou_sum / n:.3f}")
        print(f"lines exact:     {exact}/{n} = {exact / n:.1%}")


if __name__ == "__main__":
    main()
