#!/usr/bin/env bash
# Merge-ready HF model -> GGUF -> Q4_K_M for koboldcpp.
#
# koboldcpp itself only runs GGUF, it cannot train and (in the CLI build) cannot
# convert or quantize. llama.cpp does the conversion, koboldcpp then just loads
# the resulting file.
#
# Usage:
#   bash sft/export_gguf.sh sft/out/qwen3-4b-codellmexp-lora-merged codellmexp-q4
set -euo pipefail

MERGED_DIR="${1:?usage: export_gguf.sh <merged-hf-dir> [out-name] [work-dir]}"
OUT_NAME="${2:-codellmexp}"
WORK_DIR="${3:-sft/out}"
LLAMA_DIR="${LLAMA_CPP_DIR:-sft/third_party/llama.cpp}"
PYTHON="${PYTHON:-}"

if [[ -z "$PYTHON" ]]; then
  if [[ -x ".venv/bin/python" ]]; then PYTHON=".venv/bin/python"; else PYTHON="python3"; fi
fi
mkdir -p "$WORK_DIR"

# 1. llama.cpp (converter + quantizer)
if [[ ! -d "$LLAMA_DIR" ]]; then
  echo ">>> cloning llama.cpp into $LLAMA_DIR"
  git clone --depth 1 https://github.com/ggml-org/llama.cpp "$LLAMA_DIR"
fi

if ! command -v cmake >/dev/null 2>&1; then
  echo ">>> installing cmake into the python env"
  uv pip install --python "$PYTHON" cmake >/dev/null
  export PATH="$(dirname "$("$PYTHON" -c 'import sys;print(sys.executable)')"):$PATH"
fi

if [[ ! -x "$LLAMA_DIR/build/bin/llama-quantize" ]]; then
  echo ">>> building llama-quantize (once, ~2-4 min)"
  cmake -S "$LLAMA_DIR" -B "$LLAMA_DIR/build" -DGGML_NATIVE=ON -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release
  cmake --build "$LLAMA_DIR/build" --target llama-quantize -j "$(nproc)"
fi

# 2. converter dependencies
if ! "$PYTHON" -c "import gguf, sentencepiece, numpy" >/dev/null 2>&1; then
  echo ">>> installing converter dependencies"
  uv pip install --python "$PYTHON" gguf sentencepiece numpy
fi

# 3. HF -> GGUF (bf16)
F16="$WORK_DIR/$OUT_NAME-f16.gguf"
echo ">>> converting $MERGED_DIR -> $F16"
"$PYTHON" "$LLAMA_DIR/convert_hf_to_gguf.py" "$MERGED_DIR" --outfile "$F16" --outtype bf16

MMPROJ=""
if "$PYTHON" -c "import json,sys; c=json.load(open(sys.argv[1])); sys.exit(0 if ('vision_config' in c or 'vision_tower' in c) else 1)" "$MERGED_DIR/config.json"; then
  echo ">>> vision-language checkpoint: exporting mmproj"
  MMPROJ="$WORK_DIR/mmproj-$OUT_NAME.gguf"
  "$PYTHON" "$LLAMA_DIR/convert_hf_to_gguf.py" "$MERGED_DIR" --mmproj \
    --outfile "$MMPROJ" --outtype bf16 \
    || { echo "WARN: mmproj export failed, the text GGUF is still usable"; MMPROJ=""; }
fi

# 4. quantize
OUT="$WORK_DIR/$OUT_NAME-Q4_K_M.gguf"
echo ">>> quantizing -> $OUT"
"$LLAMA_DIR/build/bin/llama-quantize" "$F16" "$OUT" Q4_K_M

echo
echo "done: $OUT"
echo "run it with koboldcpp:"
echo "  ./koboldcpp-linux-x64 --model $OUT --usecuda 0 --gpulayers 99 \\"
if [[ -n "$MMPROJ" ]]; then
  echo "      --mmproj $MMPROJ --contextsize 8192 --jinja --host 127.0.0.1 --port 5001"
else
  echo "      --contextsize 8192 --jinja --host 127.0.0.1 --port 5001"
fi
