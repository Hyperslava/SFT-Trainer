SHELL := /bin/bash
VENV  ?= .venv
PY    := $(if $(wildcard $(VENV)/bin/python),$(VENV)/bin/python,python3)
ZIP   ?= assets/codellmexp.zip
MODEL ?= Qwen/Qwen3-4B-Instruct-2507
RUN   ?= sft/out/run
TASKS ?= audit,fix

.DEFAULT_GOAL := help
.PHONY: help setup check-env data sft-data train export eval zip clean

help:
	@echo "make setup      создает .venv и ставит зависимости"
	@echo "make data       распаковывает $(ZIP) в data/codellmexp/"
	@echo "make sft-data   собирает sft/data/{train,val,test}.jsonl из датасета"
	@echo "make train      QLoRA SFT:  make train MODEL=Qwen/Qwen3-4B-Instruct-2507 RUN=sft/out/my-run"
	@echo "make export     merge -> GGUF Q4_K_M для koboldcpp"
	@echo "make eval       оценка модели через koboldcpp API"
	@echo "make zip        пересобрать assets/codellmexp.zip (для мейнтейнера)"
	@echo "make clean      удалить сгенерированные данные и чекпоинты"

setup:
	bash scripts/setup_env.sh

check-env:
	@test -x "$(VENV)/bin/python" || { echo "нет $(VENV): запустите 'make setup'"; exit 1; }
	@$(VENV)/bin/python -c "import torch, transformers, peft" 2>/dev/null \
		|| { echo "зависимости не установлены: запустите 'make setup'"; exit 1; }

data:
	bash scripts/setup_data.sh "$(ZIP)" data

sft-data: data
	$(PY) sft/prepare_data.py --outdir sft/data --tasks $(TASKS)

train: sft-data check-env
	$(VENV)/bin/python sft/train_lora.py \
		--model $(MODEL) \
		--data-dir sft/data \
		--output-dir $(RUN) \
		--load-4bit --epochs 2 --max-len 2048 \
		--bs 2 --grad-accum 16 --lr 1e-4 --merge

export: check-env
	@test -d "$(RUN)-merged" || { echo "нет $(RUN)-merged: сначала запустите make train или укажите RUN=..."; exit 1; }
	bash sft/export_gguf.sh "$(RUN)-merged" codellmexp-q4 sft/out

eval:
	$(PY) sft/evaluate.py --data sft/data/test.jsonl --limit 100

zip:
	python3 scripts/build_dataset_zip.py --out "$(ZIP)"

clean:
	rm -rf sft/data sft/out data/codellmexp data/.codellmexp.unpacked
