# Exp1: разбор train и синтез промптов

**Идея:** Flash анализирует каждый размеченный train-кейс вместе с сохранённым предсказанием baseline; Argon превращает все комментарии в исправленный общий промпт и четыре совета по темам; Flash оценивает test с соответствующим советом.

1. `train.py` использует готовые 150 train-предсказаний Flash, не повторяя baseline-инференс. Для каждого кейса запрашивает **А:** почему baseline прав/неправ относительно gold; **Б:** какой условный критерий специфичен для темы. Результат: `analyses.jsonl` и `analyses.csv`, строки с ID, темой, gold, baseline, флагом совпадения и комментариями А/Б. Причины — реконструкция по данным, не записанные рассуждения baseline и не официальные экспертные rationale.
2. Один запрос `gemini/gemini-4-argon` получает полный baseline-промпт и **всю таблицу**. Возвращает `artifact.json`: `base_prompt` и `topic_advice` для четырёх тем. Промпты компилируются и фиксируются до test.
3. `infer.py` выбирает совет по готовому `topic`, передаёт Flash общий промпт + один совет и только `context`/`conversation`. Ответ — native structured `{"label":"good"}` / `{"label":"bad"}`. После всех предсказаний официальный scorer читает gold и считает метрики.

Test не участвует в адаптации. Это один обученный артефакт и один test-прогон, без подбора по test. Комментарии могут ошибаться; синтезу явно задано учитывать поддержку gold и не превращать догадки или частные суммы в универсальные правила.

Из корня репозитория, существующая `.venv`:

```sh
cp experiments/exp1/.env.example experiments/exp1/.env
# GEMINI_API_KEY наследуется из существующего ../.env.
.venv/bin/python experiments/exp1/train.py --dry-run
.venv/bin/python experiments/exp1/train.py --limit 1 --analysis-only
.venv/bin/python experiments/exp1/train.py
.venv/bin/python experiments/exp1/infer.py --limit 1
.venv/bin/python experiments/exp1/infer.py
```

Этапы имеют один бюджетный журнал exp1 и общий лимит $10. Успешные структурированные ответы восстанавливаются из журнала; повторный запуск не повторяет готовые запросы. При изменении кода, моделей или входов нужна новая папка артефактов/результатов. Анализ: до 3072 выходных токенов, три одновременных запроса, старты не чаще раза в 4.1 с. Синтез: до 32768 токенов, таймаут 900 с. Judge: те же 2048 токенов и Flash, что в baseline. Сохраняются реальные имена моделей, usage, хеши и промпты всех этапов. Короткие А/Б являются требуемым артефактом обучения; test-ответ содержит только метку.

API и тариф Argon проверены по [Google](https://ai.google.dev/gemini-api/docs/models/gemini-4-argon/) и [описанию модели](https://ai.google.dev/gemini-api/docs/latest-model): модель доступна по ключу; общий лимит вывода включает thinking. Стоимость в отчёте exp1 включает анализ, синтез и test, исходные baseline-предсказания переиспользуются как готовый вход.

Таблица текущего запуска: [JSONL](artifacts/20261024-public-original/analyses.jsonl), [CSV](artifacts/20261024-public-original/analyses.csv). Завершённый этап адаптации сохраняет в этой же папке `synthesis_input.json`, `artifact.json`, `routed_prompts.json`, читаемый `prompt.md` и журналы всех API-ответов. Хеши защищают от незаметного изменения артефакта между адаптацией и test.

## Результат: 24 октября 2026

150 разборов `gemini-3.8-flash`, один синтез `gemini-4-argon`, 80 test-ответов `gemini-3.8-flash`. Все 231 ответа валидны, API-повторов не было. Все 150 train-строк переданы в одном synthesis-запросе. Артефакт зафиксирован до test, вручную не редактировался.

**macro_topic_accuracy: 0.797250 → 0.922925 (+12.57 п.п.)**. Accuracy: 0.7875 → 0.9125, 73/80 правильных. Kappa: 0.814570; macro topic kappa: 0.839128.

| Тема | Baseline | Exp1 | Правильно exp1 |
|---|---:|---:|---:|
| credit | 86.67% | 100.00% | 15/15 |
| financial_position | 86.36% | 90.91% | 20/22 |
| savings | 85.00% | 100.00% | 20/20 |
| spending | 60.87% | 78.26% | 18/23 |

Исправлено ошибок baseline: **12**; новых ошибок: **2**; общих ошибок: 5. Confusion matrix exp1 (gold × prediction, good/bad): `[[46, 4], [3, 27]]`.

Оценка LiteLLM: анализ **$0.582956**, синтез **$0.093224**, test **$0.190955**, всего **$0.867136**. Пробные запросы включены, baseline-предсказания повторно не генерировались. Это оценка по тарифам, не выписка провайдера.

Артефакт: [JSON](artifacts/20261024-public-original/artifact.json), [читаемые промпты](artifacts/20261024-public-original/prompt.md), [точный вход Argon со всеми А/Б](artifacts/20261024-public-original/synthesis_input.json). Test: [score](results/20261024-public-original/score.json), [предсказания](results/20261024-public-original/predictions.csv), [сравнение всех кейсов](results/20261024-public-original/comparison.csv), [изменения по темам](results/20261024-public-original/comparison.json), [расход](results/20261024-public-original/summary.json).

Вывод: адаптация снизила число ошибок на этом public test. Мы одновременно меняли общий промпт и добавляли routing, поэтому отдельно приписать эффект одному из компонентов нельзя. Качество на private и статистическая устойчивость одним прогоном не установлены.

## Оценка на train тем же артефактом

```sh
.venv/bin/python experiments/exp1/infer.py --dataset data/public_train.csv --run-dir experiments/exp1/runs/20261024-public-train-original
```

Используются те же артефакт, промпты, Flash, параметры и язык, что для test. В judge-запросах нет gold; отдельный прогон содержит только новые предсказания на 150 train-кейсах. Это **in-sample оценка**, поскольку артефакт построен по этому train. `dataset_split` определяется по имени публичного CSV либо задаётся через `--split`. Новый расход — только эти judge-запросы; адаптация переиспользуется без новых вызовов Argon или разборов А/Б.

Результат полного train-прогона: **macro_topic_accuracy = 0.963384**, accuracy = 96.00% (144/150). Kappa = 0.918033; macro topic kappa = 0.925368.

| Тема | Train baseline | Train exp1 | Test exp1 |
|---|---:|---:|---:|
| credit | 77.78% | 97.22% (35/36) | 100.00% |
| financial_position | 80.56% | 97.22% (35/36) | 90.91% |
| savings | 73.53% | 100.00% (34/34) | 100.00% |
| spending | 63.64% | 90.91% (40/44) | 78.26% |

Train macro_topic_accuracy: 0.738748 → 0.963384 (+22.46 п.п.). Все 60 bad найдены; 6 ошибок — good → bad. Матрица (gold × prediction, good/bad): `[[84, 6], [0, 60]]`.

Новый расход train-инференса: **$0.363956**; суммарный учтённый расход exp1 (адаптация + test + train): **$1.231092**. Адаптация повторно не запускалась.

Сохранены [train score](results/20261024-public-train-original/score.json), [предсказания](results/20261024-public-train-original/predictions.csv), [шесть ошибок](results/20261024-public-train-original/errors.csv), [сравнение с baseline](results/20261024-public-train-original/comparison.csv) и [расход](results/20261024-public-train-original/summary.json).
