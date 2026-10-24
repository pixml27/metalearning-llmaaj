# Baseline: zero-shot judge

**Идея:** без обучения, примеров в промпте и извлечения экспертных правил попросить LLM оценить весь диалог по корректности фактов, чисел, дат и использованию доступной информации. Одно нарушение → `bad`. Это исходная точка для сравнения будущих промптов.

Модель получает только `context` и `conversation` одного public test примера. `item_id`, `topic`, `label` и альтернативные переводы в запрос не попадают. `original` сохраняет исходный смешанный язык. Отдельный запрос на каждый из 80 диалогов, без истории других примеров. Выход — native JSON schema с единственным полем `label: Literal["good", "bad"]`, затем валидация Pydantic-AI. Train-скрипта нет.

Промпт: [`prompt.md`](prompt.md). Инференс и оценка: [`infer.py`](infer.py). Обвязка: [`../llm.py`](../llm.py). Параметры: температура 0, до 2048 выходных токенов, таймаут 120 секунд, автоматические повторы отключены. Лимит учитывает и внутренние thinking tokens, видимый ответ — только метка. Алиас `latest` и серверная генерация не гарантируют детерминированность.

Из корня репозитория, Python 3.12:

```sh
uv venv --python python3.12 .venv
uv pip install --python .venv/bin/python -r experiments/requirements.lock.txt
# Ключ уже можно хранить в ../.env; либо создать .env по .env.example.
.venv/bin/python experiments/baseline/infer.py --dry-run --run-dir experiments/baseline/runs/first
.venv/bin/python experiments/baseline/infer.py --limit 1 --run-dir experiments/baseline/runs/first
.venv/bin/python experiments/baseline/infer.py --run-dir experiments/baseline/runs/first
```

`MODEL` и ключи читаются из окружения, затем локального `.env` эксперимента (если есть), `.env` репозитория, затем существующего `../.env`; ранее заданное значение не перезаписывается. Общий бюджет эксперимента — максимум $10, задаётся `EXPERIMENT_BUDGET_USD`. Каждый запрос резервирует оценку стоимости до отправки, затем заменяет её оценкой по usage. Ошибочные/оборванные попытки без измеренного расхода сохраняют резерв. `.budget.jsonl` общий для всех запусков baseline. Это защита по оценке тарифов LiteLLM, а не гарантия суммы в счёте провайдера. При неизвестной цене новые запросы запрещены.

В папке запуска: `manifest.json` с параметрами, хешами и версиями; копия `prompt.md`; `calls.jsonl` с ответами и расходом; `predictions.csv`; `summary.json`; после всех 80 ответов — `score.json` штатного scorer. Все файлы сохраняются локально в игнорируемом `runs/`. Повторный запуск с тем же `--run-dir` продолжает сохранённые предсказания. Изменение кода/промпта/модели/данных требует новой папки. Частичный прогон не получает official score.

Public test используется для измерения, промпт зафиксирован до первого запроса и по результатам не подбирался.

После полной оценки автоматически обновляется общая таблица [`../RESULTS.md`](../RESULTS.md). [`../baseline_gemma/`](../baseline_gemma/) использует тот же runner и промпт с другой моделью и отдельным бюджетом; сохранённые исходные результаты baseline остаются без изменений.

Техническая заметка: адаптер `pydantic-ai-litellm==0.2.8` сам не передаёт native JSON schema; тонкий `JudgmentModel` добавляет её в `litellm.acompletion`, явно передаёт температуру 0, сохраняет стоимость до парсинга и формирует `RequestUsage` для текущего Pydantic-AI.

Документация: [адаптер](https://github.com/mochow13/pydantic-ai-litellm), [Gemini в LiteLLM](https://docs.litellm.ai/docs/providers/gemini), [structured output Gemini](https://ai.google.dev/gemini-api/docs/structured-output).

## Результат: 24 октября 2026

80/80 валидных ответов, без API-повторов и ошибок. `MODEL=gemini/gemini-flash-latest`; ответившая модель — `gemini-3.8-flash`; язык входа — `original`.

**macro_topic_accuracy = 0.797250**, accuracy = 0.7875 (63/80). «Всегда good»: 0.625889; разница +17.14 п.п.

| Тема | Правильно / n | Accuracy |
|---|---:|---:|
| credit | 13/15 | 86.67% |
| financial_position | 19/22 | 86.36% |
| savings | 17/20 | 85.00% |
| spending | 14/23 | 60.87% |

Матрица ошибок (строки gold, столбцы prediction; порядок good, bad): `[[34, 16], [1, 29]]`.

Оценка расходов LiteLLM: **$0.178971**, 46,913 входных и 38,343 выходных токенов (включая thinking). Пробный запрос входит в эти 80 запросов. Это тарифная оценка, не выписка провайдера.

Сохранены [предсказания](results/20261024-public-original/predictions.csv), [полный score](results/20261024-public-original/score.json), [17 ошибок с ID](results/20261024-public-original/errors.csv), [параметры и хеши](results/20261024-public-original/manifest.json), [журнал ответов и usage](results/20261024-public-original/calls.jsonl).

16 из 17 ошибок — `good → bad`: judge слишком строг относительно экспертной разметки. В `spending` это 9 из 12 настоящих `good`; при этом 29 из 30 настоящих `bad` найдены. Это наблюдение по меткам, без запроса объяснений модели.

Вывод: общий промпт уже превосходит константный baseline, но не воспроизводит все критерии эксперта. Это один public прогон без подбора промпта; качество на private и статистическое преимущество из него не следуют.

## Оценка на train: тот же zero-shot подход

Отдельный прогон на всех 150 строках `data/public_train.csv`, язык `original`. Обучения и адаптации нет: gold исключается из каждого запроса, промпт и параметры модели совпадают с test-прогоном. Train-метрики записываются отдельно от test и не являются метриками на отложенной выборке.

```sh
.venv/bin/python experiments/baseline/infer.py --dataset data/public_train.csv --dry-run --run-dir experiments/baseline/results/20261024-public-train-original
.venv/bin/python experiments/baseline/infer.py --dataset data/public_train.csv --limit 1 --run-dir experiments/baseline/results/20261024-public-train-original
.venv/bin/python experiments/baseline/infer.py --dataset data/public_train.csv --run-dir experiments/baseline/results/20261024-public-train-original
```

Артефакты этого прогона: `results/20261024-public-train-original/`. В manifest явно указан `dataset_split=train`; legacy поле `test_sha256` хранит SHA256 выбранного CSV, включая train. По завершении автоматически обновляется общая таблица с отдельной колонкой выборки.

### Результат train: 24 октября 2026

150/150 валидных ответов, 150 API-вызовов без ошибок и повторов. Модель в ответе: `gemini-3.8-flash`. **macro_topic_accuracy = 0.738748**, accuracy = 0.733333 (110/150), kappa = 0.487179, macro topic kappa = 0.492236.

| Тема | Правильно / n | Accuracy |
|---|---:|---:|
| credit | 28/36 | 77.78% |
| financial_position | 29/36 | 80.56% |
| savings | 25/34 | 73.53% |
| spending | 28/44 | 63.64% |

Матрица ошибок (строки gold, столбцы prediction; порядок good, bad): `[[55, 35], [5, 55]]`. Ошибок: 40.

Оценка расходов этого train-прогона: **$0.355635**; 86,785 входных и 77,479 выходных токенов. Общий бюджетный журнал baseline включает и предыдущий test-прогон: $0.534606.

[Предсказания](results/20261024-public-train-original/predictions.csv), [score](results/20261024-public-train-original/score.json), [ошибки](results/20261024-public-train-original/errors.csv), [manifest](results/20261024-public-train-original/manifest.json), [usage и стоимость](results/20261024-public-train-original/summary.json), [ответы API](results/20261024-public-train-original/calls.jsonl), [бюджетный журнал](results/20261024-public-train-original/budget.jsonl).

Вывод: это диагностическая zero-shot оценка на train без обучения. Промпт и параметры не менялись по ходу прогона; результат train отдельно от test и не заменяет оценку на private.
