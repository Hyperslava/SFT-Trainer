#!/usr/bin/env bash
# Create .venv and install the training stack (uv if available, pip otherwise).
set -euo pipefail

cd "$(dirname "$0")/.."
PYVER="${PYTHON_VERSION:-3.12}"

if command -v uv >/dev/null 2>&1; then
  echo ">>> using uv (python $PYVER)"
  uv venv --python "$PYVER" .venv
  uv pip install --python .venv/bin/python -r sft/requirements.txt
else
  echo ">>> uv not found, using python3 -m venv"
  python3 -m venv .venv
  .venv/bin/python -m pip install --upgrade pip
  .venv/bin/python -m pip install -r sft/requirements.txt
fi

echo
echo ">>> installed. activate with: source .venv/bin/activate"
echo ">>> check CUDA: .venv/bin/python -c 'import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))'"
