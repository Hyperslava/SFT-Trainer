# CodeLLMExp SFT

Пайплайн дообучения локальной LLM на датасете **CodeLLMExp** (поиск и объяснение
уязвимостей в коде на Python / Java / C): QLoRA SFT → merge → GGUF → запуск в
**koboldcpp**.

Код лежит в git, датасет — одним архивом `assets/codellmexp.zip`, который
распаковывается перед обучением. В репозитории нет ни распакованных 10 тысяч
файлов датасета, ни весов моделей.

## Структура

```
.
├── Makefile                  # make setup / data / sft-data / train / export / eval
├── assets/
│   ├── codellmexp.zip        # датасет, 7.1 MiB, CC BY 4.0
│   └── SHA256SUMS            # контрольная сумма архива
├── docs/
│   └── DATASET.md            # описание датасета (копия README из архива)
├── scripts/
│   ├── setup_env.sh          # .venv + зависимости (uv или pip)
│   ├── setup_data.sh         # проверка sha256 + распаковка в data/
│   └── build_dataset_zip.py  # пересборка архива (для мейнтейнера)
├── sft/
│   ├── prepare_data.py       # CodeLLMExp.jsonl -> sft/data/*.jsonl
│   ├── train_lora.py         # LoRA/QLoRA SFT (transformers + peft)
│   ├── export_gguf.sh        # merge -> GGUF bf16 -> Q4_K_M
│   ├── evaluate.py           # оценка через koboldcpp API
│   ├── requirements.txt
│   └── README.md             # подробная инструкция по обучению
├── .gitignore
└── .gitattributes
```

## Быстрый старт

```bash
git clone <repo-url> codellmexp-sft
cd codellmexp-sft

make setup        # .venv (python 3.12) + sft/requirements.txt
make data         # sha256 + распаковка assets/codellmexp.zip -> data/codellmexp/
make sft-data     # data/codellmexp/CodeLLMExp.jsonl -> sft/data/{train,val,test}.jsonl

make train MODEL=Qwen/Qwen3-4B-Instruct-2507 RUN=sft/out/qwen3-4b
make export RUN=sft/out/qwen3-4b          # sft/out/codellmexp-q4-Q4_K_M.gguf

~/koboldcpp-linux-x64 --model sft/out/codellmexp-q4-Q4_K_M.gguf \
  --usecuda 0 --gpulayers 99 --contextsize 8192 --jinja --port 5001
```

После `make sft-data` получается 16 960 train / 920 val / 955 test примеров
(задачи `audit` и `fix`). Подробности, гиперпараметры под 8 ГБ VRAM, VL-модели,
конвертация в GGUF и метрики — в [`sft/README.md`](sft/README.md).

## Датасет

`assets/codellmexp.zip` (7.1 MiB) содержит:

```
codellmexp/
├── CodeLLMExp.jsonl     # 10 403 записи: код, CWE, строки, объяснение, fixed_code
├── source/              # сниппеты из внешних бенчмарков (CodeLMSec, SecurityEval, ...)
├── seed/                # сниппеты из канонических seed-примеров
├── cwe_seeds/           # исходные уязвимые примеры по CWE и языкам
└── README.md            # описание датасета (docs/DATASET.md)
```

`make data` распаковывает его в `data/codellmexp/`:

- сверяет sha256 с `assets/SHA256SUMS` и падает при несовпадении;
- идемпотентен: повторный запуск ничего не делает (маркер `data/.codellmexp.unpacked`);
- каталог `data/` в git не попадает (см. `.gitignore`).

Другой источник архива (например, asset GitHub Release) можно подставить так:

```bash
DATASET_ZIP_URL=https://github.com/<user>/<repo>/releases/download/v1/codellmexp.zip make data
# или напрямую:
bash scripts/setup_data.sh assets/codellmexp.zip
```

Обновление датасета (мейнтейнер): положить новые файлы в корень/подкаталоги,
затем

```bash
make zip    # пересоберёт assets/codellmexp.zip + SHA256SUMS детерминированно
git add assets/ && git commit -m "data: update CodeLLMExp"
```

Сборка детерминированная: фиксированные даты и права в zip, файлы
`*:Zone.Identifier`, `.DS_Store` и `Thumbs.db` выкидываются, поэтому одинаковый
вход даёт одинаковый sha256 и в git не появляется шум.

### Почему zip, а не файлы в git

- 10 452 файла и ~68 МиБ распакованного датасета превратили бы `git status`,
  диффы и клонирование в мусор; архив версионируется атомарно.
- 7.1 MiB — сильно ниже лимита GitHub в 100 МиБ на файл, LFS не нужен.
- Если датасет вырастет за 100 МиБ: включите Git LFS (`git lfs track "*.zip"`)
  или вынесите архив в GitHub Release и используйте `DATASET_ZIP_URL`.

## Требования

- Python 3.12 (для `bitsandbytes` надёжнее, чем 3.13), `uv` опционально;
- GPU ≥ 8 ГБ VRAM для 3–4B QLoRA; `cmake`/`g++` понадобятся только для сборки
  `llama-quantize` при экспорте в GGUF (скрипт поставит `cmake` в venv сам);
- диски: ~4 ГБ на базовую модель, ~4 ГБ на merged-чекпоинт, ~2.5 ГБ на GGUF.

## Лицензии и атрибуция

- **Датасет** — CodeLLMExp, [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
  При распространении и публикации результатов нужна атрибуция; см.
  [`NOTICE.md`](NOTICE.md) и [`docs/DATASET.md`](docs/DATASET.md).
- **Код репозитория** — лицензия не задана: добавьте `LICENSE` перед публикацией.
