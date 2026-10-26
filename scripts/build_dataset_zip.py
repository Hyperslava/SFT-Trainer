#!/usr/bin/env python3
"""Pack the CodeLLMExp dataset into a reproducible zip committed to the repo.

The archive layout is fixed, so `unzip` always lands in the same place:

    codellmexp/CodeLLMExp.jsonl
    codellmexp/README.md          # copy of docs/DATASET.md
    codellmexp/source/...
    codellmexp/seed/...
    codellmexp/cwe_seeds/...

Windows artefacts (*:Zone.Identifier, .DS_Store, Thumbs.db) are skipped.
Timestamps and permissions are pinned, so rebuilding the same input gives the
same sha256 and no diff noise in git.

    python3 scripts/build_dataset_zip.py
    python3 scripts/build_dataset_zip.py --out assets/codellmexp.zip
"""

from __future__ import annotations

import argparse
import hashlib
import zipfile
from pathlib import Path

PREFIX = "codellmexp"
ROOT_FILES = [(Path("CodeLLMExp.jsonl"), "CodeLLMExp.jsonl"), (Path("docs/DATASET.md"), "README.md")]
ROOT_DIRS = ["source", "seed", "cwe_seeds"]
SKIP_NAMES = {".DS_Store", "Thumbs.db", "desktop.ini", "README.md.orig"}
SKIP_SUFFIXES = (":Zone.Identifier", ".tmp", ".log")
FIXED_DATE = (2026, 1, 1, 0, 0, 0)


def iter_dir(root: Path):
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.name in SKIP_NAMES or path.name.endswith(SKIP_SUFFIXES):
            continue
        yield path


def add_bytes(zf: zipfile.ZipFile, arcname: str, data: bytes) -> None:
    info = zipfile.ZipInfo(arcname, date_time=FIXED_DATE)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o644 << 16
    zf.writestr(info, data)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=".", help="repository root (default: .)")
    ap.add_argument("--out", default="assets/codellmexp.zip")
    ap.add_argument("--sums", default="assets/SHA256SUMS")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    out = root / args.out
    out.parent.mkdir(parents=True, exist_ok=True)

    files = 0
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for src, arc in ROOT_FILES:
            path = root / src
            if not path.exists():
                raise SystemExit(f"missing input file: {path}")
            add_bytes(zf, f"{PREFIX}/{arc}", path.read_bytes())
            files += 1
        for dirname in ROOT_DIRS:
            directory = root / dirname
            if not directory.exists():
                raise SystemExit(f"missing input directory: {directory}")
            for path in iter_dir(directory):
                arc = f"{PREFIX}/{dirname}/{path.relative_to(directory).as_posix()}"
                add_bytes(zf, arc, path.read_bytes())
                files += 1

    digest = sha256(out)
    sums = root / args.sums
    sums.parent.mkdir(parents=True, exist_ok=True)
    sums.write_text(f"{digest}  {out.name}\n", encoding="utf-8")

    size_mb = out.stat().st_size / 1024 / 1024
    print(f"wrote {out.relative_to(root)}: {files} files, {size_mb:.1f} MiB")
    print(f"sha256 {digest}")
    print(f"wrote {sums.relative_to(root)}")


if __name__ == "__main__":
    main()
