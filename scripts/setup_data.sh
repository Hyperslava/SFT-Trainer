#!/usr/bin/env bash
# Unpack the dataset archive (assets/codellmexp.zip) into data/.
#
#   bash scripts/setup_data.sh                  # unpack ./assets/codellmexp.zip -> ./data/codellmexp/
#   bash scripts/setup_data.sh path/to.zip      # custom archive
#   DATASET_ZIP_URL=https://... bash scripts/setup_data.sh   # download first (e.g. GitHub Release asset)
#
# The archive is verified against assets/SHA256SUMS and the unpack is
# idempotent: a marker with the archive checksum is kept in data/.
set -euo pipefail

ZIP="${1:-assets/codellmexp.zip}"
DEST="${2:-data}"
URL="${DATASET_ZIP_URL:-}"
SUMS="$(dirname "$ZIP")/SHA256SUMS"
MARKER="$DEST/.codellmexp.unpacked"

if [[ ! -f "$ZIP" && -n "$URL" ]]; then
  echo ">>> downloading dataset archive"
  mkdir -p "$(dirname "$ZIP")"
  curl -fL --retry 3 --retry-delay 2 -o "$ZIP" "$URL"
fi

if [[ ! -f "$ZIP" ]]; then
  echo "error: dataset archive not found: $ZIP" >&2
  echo "       expected it in the repo (assets/codellmexp.zip) or set DATASET_ZIP_URL" >&2
  exit 1
fi

SHA="$(python3 - "$ZIP" <<'PY'
import hashlib, sys
h = hashlib.sha256()
with open(sys.argv[1], "rb") as f:
    for chunk in iter(lambda: f.read(1 << 20), b""):
        h.update(chunk)
print(h.hexdigest())
PY
)"

if [[ -f "$SUMS" ]]; then
  EXPECTED="$(awk -v name="$(basename "$ZIP")" '$2 == name { print $1 }' "$SUMS" | head -1)"
  if [[ -z "$EXPECTED" ]]; then
    echo "warning: $(basename "$ZIP") is not listed in $SUMS, skipping checksum check" >&2
  elif [[ "$EXPECTED" != "$SHA" ]]; then
    echo "error: checksum mismatch for $ZIP" >&2
    echo "       expected $EXPECTED" >&2
    echo "       actual   $SHA" >&2
    exit 1
  else
    echo ">>> checksum ok ($SHA)"
  fi
fi

if [[ -f "$MARKER" ]] && grep -qx "$SHA" "$MARKER" && [[ -f "$DEST/codellmexp/CodeLLMExp.jsonl" ]]; then
  echo ">>> dataset already unpacked in $DEST/codellmexp"
  exit 0
fi

echo ">>> unpacking $ZIP -> $DEST/"
mkdir -p "$DEST"
python3 - "$ZIP" "$DEST" <<'PY'
import sys, zipfile
with zipfile.ZipFile(sys.argv[1]) as zf:
    zf.extractall(sys.argv[2])
PY
echo "$SHA" > "$MARKER"

COUNT="$(find "$DEST/codellmexp" -type f | wc -l | tr -d ' ')"
SIZE="$(du -sh "$DEST/codellmexp" | cut -f1)"
echo ">>> done: $COUNT files, $SIZE in $DEST/codellmexp"
echo "    next: make sft-data   (or: python sft/prepare_data.py --outdir sft/data)"
