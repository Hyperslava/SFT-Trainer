# SFT на CodeLLMExp → GGUF → koboldcpp

## Главное

**koboldcpp обучать не умеет.** Это инференс-сервер для GGUF (llama.cpp): он не
делает backward pass, не читает датасеты и не сохраняет веса. В этой сборке
(`~/koboldcpp-linux-x64`) нет даже флагов конвертации/квантизации — только
`--lora` для применения уже готового GGUF-адаптера.

Поэтому пайплайн такой:

```
CodeLLMExp.jsonl
      │  sft/prepare_data.py      (чистка, шаблоны задач, grouped split)
      ▼
sft/data/{train,val,test}.jsonl
      │  sft/train_lora.py        (transformers + peft, QLoRA на GPU)
      ▼
LoRA-адаптер  ──merge──►  HF-чекпоинт (bf16)
      │  sft/export_gguf.sh       (llama.cpp: convert_hf_to_gguf + llama-quantize)
      ▼
codellmexp-Q4_K_M.gguf
      │
      ▼
koboldcpp --model ... --jinja      (инференс, /v1/chat/completions)
```

Важно: обучать можно только исходный HF-чекпоинт, а не GGUF. GGUF-файл
`Qwen3.5-4B-Q4_K_M.gguf` для обучения не годится — нужны веса той же базовой
модели в формате HuggingFace (`safetensors`).

## 0. Окружение

RTX 4060 Laptop (8 ГБ VRAM), 15 ГБ RAM, 18 ядер — 3–4B модель с QLoRA влезает
спокойно.

```bash
make setup        # .venv (python 3.12) + зависимости; uv, если установлен
source .venv/bin/activate
```

`bitsandbytes` нужен только для `--load-4bit`. Если он не ставится — обучайте
1.5B модель в bf16 без 4-бит (см. ниже).

## 1. Данные

Датасет поставляется архивом `assets/codellmexp.zip` и распаковывается в
`data/codellmexp/`:

```bash
make data         # sha256 + unzip -> data/codellmexp/
make sft-data     # -> sft/data/{train,val,test}.jsonl
```

`prepare_data.py` сам находит `data/codellmexp/CodeLLMExp.jsonl` (фолбэк —
`./CodeLLMExp.jsonl`). Если датасета нет, скрипт подскажет запустить `make data`.

Скрипт делает то, чего в «сыром» jsonl нет:

| проблема в датасете | что делает скрипт |
|---|---|
| 5 284 записи обёрнуты в ` ```python … ``` ` | снимает fences |
| `vulnerable_lines` считаются **включая** строку с fence | вычитает 1 для fenced-записей |
| 232 записи с `vulnerable_lines = {0,0}` | выбрасывает (нечему учить локализацию) |
| 14 записей с `potential_impact` вместо `impact` | нормализует ключ |
| `language`: `Python`/`python`, `Java`/`java` | приводит к нижнему регистру |
| 7 438 уникальных сниппетов на 10 403 записи (аугментации одного кода) | split по хешу сниппета, чтобы варианты не утекли в test |
| 180 записей с `rate = 3` | отбрасывает (`--min-rate 4`) |
| дублирующиеся пары prompt/answer | дедуплицирует |

Итог на вашем файле: **9 985 `audit` + 8 850 `fix`** примеров,
train/val/test = 16 960 / 920 / 955.

Две задачи:

* `audit` — по коду вернуть JSON `{"cwe", "lines", "root_cause", "impact", "mitigation"}`;
* `fix` — по уязвимому коду вернуть исправленный код в fence-блоке.

Хотите только одну — `--tasks audit` или `--tasks fix`.

## 2. Базовая модель

Под 8 ГБ берите 3–4B instruct:

| модель | режим | VRAM |
|---|---|---|
| `Qwen/Qwen2.5-Coder-3B-Instruct` | QLoRA 4-bit, seq 2048 | ~5 ГБ |
| `Qwen/Qwen3-4B-Instruct-2507` | QLoRA 4-bit, seq 2048 | ~6 ГБ |
| `meta-llama/Llama-3.2-3B-Instruct` | QLoRA 4-bit | ~5.5 ГБ |
| `Qwen/Qwen2.5-Coder-1.5B-Instruct` | LoRA bf16 (без 4-bit) | ~5 ГБ |
| 7–8B | QLoRA, seq 1024 | ~7.5 ГБ, на грани — не рекомендую |

Если у `Qwen3.5-4B` из вашего GGUF есть HF-веса — берите её, иначе чекпоинт не
совпадёт с тем, что вы потом запускаете.

### Vision-language базы (Qwen3-VL, Qwen2.5-VL, SmolVLM)

`Qwen/Qwen3-VL-2B-Instruct` тоже подходит, но это VL-чекпоинт: `AutoModelForCausalLM`
его не принимает (`Unrecognized configuration class Qwen3VLConfig`). Скрипт
определяет `vision_config` и грузит модель через `AutoModelForImageTextToText`,
а LoRA ставит только в языковой декодер (`model.language_model.*`) — vision-башня
не обучается и не раздувает адаптер.

```bash
python sft/train_lora.py \
  --model Qwen/Qwen3-VL-2B-Instruct \
  --data-dir sft/data --output-dir sft/out/qwen3-vl-2b-codellmexp-lora \
  --load-4bit --epochs 2 --max-len 2048 --bs 1 --grad-accum 16 --lr 1e-4 --merge
```

Датасет текстовый, картинки не нужны — обучается фактически только языковая
часть, и в koboldcpp такой GGUF работает без `--mmproj`. Для текстовых задач
проще брать обычную текстовую модель той же величины: на неё уйдёт весь бюджет
VRAM, а не половина. VL берите, если хотите переиспользовать ровно ту базу,
которая у вас уже крутится в koboldcpp.

## 3. Обучение

Через make (подставит data-dir, epochs, bs, merge):

```bash
make train MODEL=Qwen/Qwen3-4B-Instruct-2507 RUN=sft/out/qwen3-4b
```

Или напрямую, с полным контролем параметров:

```bash
python sft/train_lora.py \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --data-dir sft/data \
  --output-dir sft/out/qwen3-4b-codellmexp-lora \
  --load-4bit \
  --epochs 2 \
  --max-len 2048 \
  --bs 1 --grad-accum 16 \
  --lr 1e-4 \
  --merge
```

Что внутри:

* loss только по токенам ответа (prompt замаскирован `-100`) — модель не учится
  пересказывать инструкцию;
* чат-шаблон токенизатора, `enable_thinking=False` для Qwen3;
* LoRA `r=16, alpha=32`, `target_modules="all-linear"` (работает для любой
  архитектуры), gradient checkpointing, `paged_adamw_8bit`;
* чекпоинты и eval каждые 200 шагов, адаптер + слитая модель в `--output-dir`.

Ориентиры: ~1 060 шагов на эпоху, на 4060 Laptop это порядка 30–50 минут на
эпоху. Если хочется быстрее — `--max-len 1536` (90-й процентиль примеров —
~600 токенов, так что качество почти не пострадает).

Быстрая проверка, что всё запускается:

```bash
python sft/prepare_data.py --limit 300 --outdir /tmp/sft-smoke
python sft/train_lora.py --model Qwen/Qwen3-4B-Instruct-2507 \
  --data-dir /tmp/sft-smoke --output-dir /tmp/sft-smoke/out \
  --load-4bit --max-steps 20 --bs 1 --grad-accum 2 --max-len 1024
```

## 4. Merge (уже сделан флагом `--merge`)

Появляется `sft/out/qwen3-4b-codellmexp-lora-merged/` — обычный HF-чекпоинт в
bf16. Именно его конвертируем в GGUF.

## 5. GGUF и квантизация

```bash
bash sft/export_gguf.sh \
  sft/out/qwen3-4b-codellmexp-lora-merged \
  codellmexp-q4 \
  sft/out
```

Скрипт при первом запуске клонирует llama.cpp в `sft/third_party/llama.cpp`,
ставит `cmake` в venv, собирает `llama-quantize`, конвертирует в bf16-GGUF и
квантует в `Q4_K_M` → `sft/out/codellmexp-q4-Q4_K_M.gguf` (~2.5 ГБ для 4B).
`cmake` в системе не установлен — это учтено.

Для VL-чекпоинта скрипт дополнительно пытается собрать `mmproj-*.gguf` — он нужен
только для распознавания картинок. У `Qwen3-VL` в HF-репозитории нет
`preprocessor_config.json`, поэтому этот шаг падает с `KeyError: image_mean` и
печатает WARN; текстовый GGUF при этом собирается и запускается нормально.

## 6. Запуск в koboldcpp

```bash
~/koboldcpp-linux-x64 \
  --model sft/out/codellmexp-q4-Q4_K_M.gguf \
  --usecuda 0 --gpulayers 99 \
  --contextsize 8192 \
  --jinja \
  --host 127.0.0.1 --port 5001
```

`--jinja` обязателен: иначе koboldcpp возьмёт собственный формат промпта, а
модель обучалась на chat-шаблоне Qwen, и качество просядет.

Проверка через OpenAI-совместимый API:

```bash
curl -s http://127.0.0.1:5001/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "koboldcpp",
    "temperature": 0,
    "messages": [
      {"role": "system", "content": "You are a security code auditor specialised in the CWE Top 25. Given one code snippet, identify the vulnerability, localise it and explain it. Reply with a single JSON object and nothing else, with exactly these keys:\n{\"cwe\": \"CWE-XXX\", \"lines\": [start, end], \"root_cause\": \"...\", \"impact\": \"...\", \"mitigation\": \"...\"}\nLine numbers are 1-based and refer to the snippet itself, not counting the markdown fences."},
      {"role": "user", "content": "Audit the following python code for security vulnerabilities:\n\n```python\nimport subprocess\ndef run(cmd):\n    return subprocess.check_output(cmd, shell=True)\n```"}
    ]
  }' | python3 -m json.tool
```

## 7. Оценка на holdout

```bash
python sft/evaluate.py --data sft/data/test.jsonl --limit 100 --task audit
python sft/evaluate.py --data sft/data/test.jsonl --limit 50  --task fix
```

Считает: долю валидного JSON, точность CWE, IoU диапазона строк, точное
совпадение строк; для `fix` — долю ответов с код-блоком. Тест собран по
хешам сниппетов, поэтому метрики честные (без утечки аугментаций).

## 8. Альтернатива: LoRA вместо merge

koboldcpp умеет `--lora`, но адаптер должен быть в GGUF. Тогда:

```bash
python sft/third_party/llama.cpp/convert_lora_to_gguf.py \
  sft/out/qwen3-4b-codellmexp-lora \
  --base sft/out/qwen3-4b-codellmexp-lora-merged   # или исходная база
# затем: koboldcpp ... --model base-Q4_K_M.gguf --lora adapter.gguf
```

Минус: адаптер должен точно соответствовать базе, а квантизация базы и адаптера
по отдельности даёт чуть больше ошибок, чем merge → quantize. Для одноразового
результата проще merge.

## 9. Подводные камни

* **Не делайте random split по строкам.** В датасете 7 438 уникальных кода на
  10 403 записи: почти одинаковые сниппеты с разными объяснениями попадут в
  train и test, и метрики будут завышены. `prepare_data.py` группирует по
  `sha1(code + cwe)`.
* **Не учите `severity`.** У 4 219 записей там `Unknown` — модель выучит
  бесполезный ярлык. В таргете его нет.
* **232 записи с `lines = [0,0]`** бессмысленны для локализации; они отброшены,
  но если нужны только объяснения — уберите проверку `lines_ok` в
  `build_examples`.
* **Таргеты на английском** (как в датасете). Если нужен русский — переведите
  `explanation.*` и поменяйте `SYSTEM_*`/`user` в `prepare_data.py`, иначе
  получится смесь языков.
* **Catastrophic forgetting.** 2 эпохи на 17k примерах узкой задачи — модель
  начнёт отвечать JSON-ом на любой вопрос. Если это мешает, уменьшите `--lr` до
  5e-5, оставьте 1 эпоху или подмешайте 5–10 % обычных instruct-данных.
* **Проверяйте `loss` на val, а не на train.** При r=16 и 2 эпохах переобучение
  начинается быстро; `save_total_limit=2` оставит два последних чекпоинта.
* **Лицензия CC BY 4.0** — при публикации модели/результатов нужна атрибуция
  датасета.

## 10. Если не хочется писать код

Те же данные (`sft/data/*.jsonl`) едят готовые тренеры, если привести их к
формату `messages`:

* **Unsloth** — самый экономный по VRAM, но проверьте поддержку вашей версии
  Python (на 3.11/3.12 надёжнее);
* **LLaMA-Factory** — `llamafactory-cli train ...` с `dataset: alpaca`/`sharegpt`;
* **axolotl** — YAML-конфиг, QLoRA из коробки.

Формат наших файлов (`system`/`user`/`assistant`) конвертируется в ShareGPT
одной строкой кода, но `train_lora.py` уже делает всё нужное и не тянет лишних
зависимостей.
