# Baseline Gemma: тот же zero-shot judge

**Идея:** повторить baseline без изменения промпта и параметров, заменив только Gemini Flash на Gemma 4.

Модель: **`gemma-4-26b-a4b-it` через Gemini API**, строка LiteLLM — `gemini/gemma-4-26b-a4b-it`. Интеграция: [`infer.py`](infer.py) → общий [`../baseline/infer.py`](../baseline/infer.py) → Pydantic-AI + [`../llm.py`](../llm.py) → `litellm.acompletion` с провайдером `gemini/` и `GEMINI_API_KEY`.

[`prompt.md`](prompt.md) — побайтовая копия baseline. Тот же public test (80 диалогов), язык `original`, без train; в запросе только `context` и `conversation` одного примера. Температура 0, до 2048 выходных токенов (включая thinking), таймаут 120 секунд; автоматические повторы отключены. Native JSON schema `{"label":"good"}` / `{"label":"bad"}` и валидация Pydantic. Thinking отдельно не задаётся, как и в baseline; серверные настройки моделей могут различаться. Ошибки API/парсинга не превращаются в метки.

## Запуск

Из корня репозитория, с тем же Python 3.12 и зависимостями, что у baseline:

```sh
uv venv --python python3.12 .venv
uv pip install --python .venv/bin/python -r experiments/requirements.lock.txt
cp experiments/baseline_gemma/.env.example experiments/baseline_gemma/.env
.venv/bin/python -m unittest experiments.test_llm experiments.test_results
.venv/bin/python experiments/baseline_gemma/infer.py --dry-run --run-dir experiments/baseline_gemma/results/20261024-public-original
.venv/bin/python experiments/baseline_gemma/infer.py --limit 1 --run-dir experiments/baseline_gemma/results/20261024-public-original
.venv/bin/python experiments/baseline_gemma/infer.py --run-dir experiments/baseline_gemma/results/20261024-public-original
```

Ключ берётся из окружения / `.env` репозитория / `../.env`; копировать ключ в эксперимент не требуется. Приоритет настроек: окружение процесса → `baseline_gemma/.env` → `.env` репозитория → `../.env`. Если `MODEL` в окружении перекрывает Gemma другой моделью, запуск останавливается. Для нового независимого прогона нужна новая папка; готовые предсказания в той же папке продолжаются без повторных запросов.

Отдельные `.budget.jsonl` и `.run.lock` относятся только к baseline_gemma. По [тарифу Google](https://ai.google.dev/gemini-api/docs/pricing#gemma-4), проверенному 24 октября 2026, вход и выход Gemma 4 через Gemini API бесплатны; в установленном LiteLLM цена равна 0. Общая обвязка разрешает известную нулевую цену только для двух проверенных Gemma 4 ID, сохраняя запрет неизвестных тарифов. Квоты запросов/токенов сохраняются.

Полный запуск сохраняет prompt, manifest с параметрами/хешами/версиями, calls с usage и фактической моделью, predictions, summary и score штатного scorer. Затем автоматически обновляется [`../RESULTS.md`](../RESULTS.md). Частичный прогон в таблицу метрик не попадает.

Документация: [Gemma в Gemini API](https://ai.google.dev/gemma/docs/core/gemma_on_gemini_api), [провайдер Gemini в LiteLLM](https://docs.litellm.ai/docs/providers/gemini).

## Результат: 24 октября 2026

80/80 валидных ответов, 80 успешных API-вызовов. Запрошенная и фактически ответившая модель — `gemma-4-26b-a4b-it` через Gemini API; язык `original`. Пробный запрос входит в эти 80. До него была сетевая ошибка DNS в песочнице; после 26 готовых ответов API вернул один `429`. Прогон продолжен порциями с паузами, без повторной отправки успешных примеров и без изменения параметров/промпта; автоматические повторы отключены.

**macro_topic_accuracy = 0.792819**, accuracy = 0.7750 (62/80), kappa = 0.470588, macro topic kappa = 0.526037.

| Тема | Правильно / n | Accuracy |
|---|---:|---:|
| credit | 14/15 | 93.33% |
| financial_position | 17/22 | 77.27% |
| savings | 18/20 | 90.00% |
| spending | 13/23 | 56.52% |

Матрица ошибок (строки gold, столбцы prediction; порядок good, bad): `[[48, 2], [16, 14]]`.

Оценка расходов LiteLLM: **$0**, 46,913 входных и 475 выходных токенов. Отрицательной/неизвестной цены нет: в карте явно записаны нулевые тарифы.

Сохранены [предсказания](results/20261024-public-original/predictions.csv), [полный score](results/20261024-public-original/score.json), [18 ошибок с ID](results/20261024-public-original/errors.csv), [manifest](results/20261024-public-original/manifest.json), [ответы и usage](results/20261024-public-original/calls.jsonl), [summary](results/20261024-public-original/summary.json), [все бюджетные резервы и расчёты](results/20261024-public-original/budget.jsonl).

По сравнению с Gemini Flash: разница macro topic accuracy -0.443 п.п., 62 против 63 верных ответов. Gemma пропустила 16 из 30 настоящих `bad` и ошибочно отклонила 2 из 50 `good`; у Flash соответственно 1 и 16. Это различие поведения на одном public-прогоне; статистическое преимущество и качество на private из него не следуют.

Вывод: при том же zero-shot промпте Gemma дала близкий основной score бесплатно, но чаще пропускала нарушения. Промпт по тестовым результатам не менялся.

## Оценка на train: тот же zero-shot подход

Отдельный прогон на всех 150 строках `data/public_train.csv`, язык `original`. Обучения и адаптации нет: gold исключается из каждого запроса, промпт и параметры модели совпадают с test-прогоном. Train-метрики записываются отдельно от test и не являются метриками на отложенной выборке.

```sh
.venv/bin/python experiments/baseline_gemma/infer.py --dataset data/public_train.csv --request-interval 3 --dry-run --run-dir experiments/baseline_gemma/results/20261024-public-train-original
.venv/bin/python experiments/baseline_gemma/infer.py --dataset data/public_train.csv --request-interval 3 --limit 1 --run-dir experiments/baseline_gemma/results/20261024-public-train-original
.venv/bin/python experiments/baseline_gemma/infer.py --dataset data/public_train.csv --request-interval 3 --run-dir experiments/baseline_gemma/results/20261024-public-train-original
```

Артефакты этого прогона: `results/20261024-public-train-original/`. В manifest явно указан `dataset_split=train`; legacy поле `test_sha256` хранит SHA256 выбранного CSV, включая train. По завершении автоматически обновляется общая таблица с отдельной колонкой выборки.

Запросы запускаются с интервалом не менее 3 секунд для соблюдения бесплатной квоты. Температура, лимит выходных токенов, таймаут и отключённые автоматические повторы не меняются.

### Результат train: 24 октября 2026

150/150 валидных ответов, 150 API-вызовов без ошибок, 429 и повторов. Модель в ответе: `gemma-4-26b-a4b-it`. Прогон начат после завершения платного Gemini Flash. **macro_topic_accuracy = 0.767343**, accuracy = 0.760000 (114/150), kappa = 0.457831, macro topic kappa = 0.481336.

| Тема | Правильно / n | Accuracy |
|---|---:|---:|
| credit | 34/36 | 94.44% |
| financial_position | 25/36 | 69.44% |
| savings | 27/34 | 79.41% |
| spending | 28/44 | 63.64% |

Матрица ошибок (строки gold, столбцы prediction; порядок good, bad): `[[86, 4], [32, 28]]`. Ошибок: 36.

Оценка расходов этого train-прогона: **$0**; 86,785 входных и 893 выходных токенов.

[Предсказания](results/20261024-public-train-original/predictions.csv), [score](results/20261024-public-train-original/score.json), [ошибки](results/20261024-public-train-original/errors.csv), [manifest](results/20261024-public-train-original/manifest.json), [usage и стоимость](results/20261024-public-train-original/summary.json), [ответы API](results/20261024-public-train-original/calls.jsonl), [бюджетный журнал](results/20261024-public-train-original/budget.jsonl).

Вывод: это диагностическая zero-shot оценка на train без обучения. Промпт и параметры не менялись по ходу прогона; результат train отдельно от test и не заменяет оценку на private.
