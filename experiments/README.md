# Эксперименты

Правила: [`AGENTS.md`](AGENTS.md). Общий judge: [`llm.py`](llm.py). Первый эксперимент: [`baseline/`](baseline/).

Сводная таблица описаний и метрик: **[`RESULTS.md`](RESULTS.md)**. Второй эксперимент: [`baseline_gemma/`](baseline_gemma/) — тот же baseline с Gemma 4 через Gemini API. После полной оценки таблица обновляется автоматически; вручную без API: `.venv/bin/python experiments/update_results.py`.

[`exp1/`](exp1/README.md) — адаптация на train: Flash разбирает кейсы, Argon синтезирует общий промпт и советы по четырём темам, Flash оценивает test с routing по `topic`.

[`exp2/`](exp2/README.md) — те же разборы train переиспользуются без новых вызовов, а работа добавляется в момент синтеза: Argon получает фиксированный блок DEFINITIONS (D1–D15) и рубрику из пяти критериев, поэтому базовый промпт обязан содержать явные критерии, приоритеты, ссылки на train-примеры и границы.

[`exp3/`](exp3/README.md) — seed-промпт из полезных частей exp1 и exp2, ссылка на train-примеры у каждого критерия и два артефакта: промпт со ссылками для объяснения решения и его обрезанная Argon'ом версия без ссылок для инференса.

В таблицах train и test указаны отдельно. Оценка на train использует тот же zero-shot промпт без обучения и без gold в запросе; все 150 строк идут только как вход для предсказаний. Для выбора файла поддерживается `--dataset` (старый `--test` остаётся алиасом). `--request-interval` ограничивает частоту запросов, не меняя параметры модели.

Judge inference in baseline, baseline_gemma and exp1–exp3 now defaults to **3 concurrent requests**. Use `--concurrency 1` for sequential execution or `--concurrency 2` for a lower load. `--request-interval SECONDS` applies globally across all workers; the default is 0. Concurrency and spacing are recorded in the manifest. Prompts, model parameters, schemas and retry policy are unchanged. Three concurrent requests target roughly 3× throughput; actual gains depend on latency and provider quotas.

Each request owns its model state. Results are saved individually in dataset order, even when API responses arrive out of order. On failure, no new requests start; requests already sent are allowed to finish and successful responses are saved before the error is reported. Completed responses are recovered from the journal on resume, including in baseline. Use a **new `--run-dir`** for this code version; existing manifests and published results are preserved. `--limit` counts new requests, so a smoke test and its continuation can share one run directory.

```sh
# One real smoke request, then up to 11 more in the same partial run (12 total).
.venv/bin/python experiments/exp3/infer.py --run-dir experiments/exp3/runs/concurrency-smoke --limit 1
.venv/bin/python experiments/exp3/infer.py --run-dir experiments/exp3/runs/concurrency-smoke --limit 11

# Offline checks for ID/label alignment, budgets, errors, cancellation and resume.
.venv/bin/python -m unittest experiments.test_inference experiments.test_llm
```

```text
experiments/
  AGENTS.md                 # бюджет, секреты, формат и воспроизводимость
  llm.py                    # LiteLLM + Pydantic-AI, native schema, учёт USD
  requirements.txt          # прямые зависимости
  requirements.lock.txt     # все версии проверенного окружения Python 3.12
  test_llm.py               # проверки без API
  test_results.py           # полные/частичные прогоны и копии результатов
  update_results.py         # пересборка общей таблицы без API
  RESULTS.md                # описание и метрики завершённых экспериментов
  baseline/
    README.md               # идея, запуск, результат
    prompt.md               # фиксированный промпт
    infer.py                # инференс + официальный scorer
    runs/                   # локальные промежуточные данные, исключены из git
  baseline_gemma/            # тот же baseline с Gemma 4 через Gemini API
  exp1/                      # разбор train + синтез промптов Argon + routing по topic
  exp2/
    definitions.md           # фиксированный словарь схемы D1–D15, подставляется дословно
    synthesis_prompt.md      # рубрика пяти критериев и обязательная структура базового промпта
    train.py                 # переиспользование разборов exp1 + один синтез Argon
    infer.py                 # инференс по зафиксированным промптам + официальный scorer
```

Каждую новую идею помещать в отдельную подпапку. В `infer.py` переиспользовать `make_judge`/`judge_one` и отдельный бюджетный журнал эксперимента. Если нужен этап адаптации, добавить `train.py` и явно сохранить его артефакты. Модель выбирается через `MODEL`, ключи — через `.env`.

Установка проверенного окружения:

```sh
uv venv --python python3.12 .venv
uv pip install --python .venv/bin/python -r experiments/requirements.lock.txt
.venv/bin/python -m unittest experiments.test_llm
```
