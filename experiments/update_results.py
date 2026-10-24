"""Rebuild RESULTS.md from completed experiment artifacts; no model/API calls."""
from pathlib import Path
import json
import re


def cell(value) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def dataset_split(manifest: dict) -> str:
    return manifest.get("dataset_split") or {"public_train.csv": "train", "public_test.csv": "test"}.get(
        Path(manifest.get("dataset_path", "")).name, "evaluation")


def completed_runs(root: Path) -> list[tuple]:
    # Prefer published results to identical copies under ignored runs/.
    paths = sorted(root.glob("*/results/*/score.json")) + sorted(root.glob("*/runs/*/score.json"))
    runs, seen = [], set()
    for path in paths:
        directory = path.parent
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
        report = json.loads(path.read_text(encoding="utf-8"))
        if summary["status"] != "complete" or summary["completed"] != manifest["n"] or report["n"] != manifest["n"]:
            continue
        # Same configuration can have independent reruns; deduplicate only copies
        # of the exact manifest (including its original creation timestamp).
        identity = (manifest["experiment"], manifest["config_sha256"], manifest["created_at"])
        if identity not in seen:
            seen.add(identity)
            runs.append((directory, manifest, summary, report))
    return sorted(runs, key=lambda r: (r[1]["created_at"], r[1]["experiment"]))


def update_results(root: Path) -> Path:
    root = root.resolve()
    runs = completed_runs(root)
    lines = ["# Результаты экспериментов", "",
             "Только полные прогоны; метрики рассчитаны штатным `src/evaluate.py`. Основная метрика — `macro_topic_accuracy` (выше лучше).",
             "Таблица обновляется автоматически после полной оценки эксперимента. Ручное обновление без API: `.venv/bin/python experiments/update_results.py`.", "",
             "Train-строки baseline — отдельная zero-shot оценка без обучения: gold исключается из judge-запросов. В адаптивных экспериментах train может использоваться для обучения; это указывается в описании. Train и test показаны раздельно.", "",
             "| Эксперимент / запуск | Выборка | Описание | Модель в ответе | Язык | n | macro_topic_accuracy | Accuracy | Kappa | Macro topic kappa | USD, оценка |",
             "|---|---|---|---|---|---:|---:|---:|---:|---:|---:|"]
    for directory, manifest, summary, report in runs:
        experiment = manifest["experiment"]
        readme = root / experiment / "README.md"
        match = re.search(r"\*\*Идея:\*\*\s*(.+)", readme.read_text(encoding="utf-8")) if readme.exists() else None
        description = match.group(1) if match else experiment
        link = directory.relative_to(root).as_posix()
        lines.append(f"| [{cell(experiment)}]({experiment}/README.md) / [{cell(directory.name)}]({link}/score.json) "
                     f"| {cell(dataset_split(manifest))} | {cell(description)} | {cell(', '.join(summary['response_models']))} | {cell(manifest['language'])} "
                     f"| {report['n']} | **{report['macro_topic_accuracy']:.6f}** | {report['accuracy']:.6f} "
                     f"| {report['kappa']:.6f} | {report['macro_topic_kappa']:.6f} | {summary['cost_usd_estimated_this_run']:.6f} |")
    lines += ["", "## Accuracy по темам", "",
              "| Эксперимент / запуск | Выборка | credit | financial_position | savings | spending |",
              "|---|---|---:|---:|---:|---:|"]
    for directory, manifest, summary, report in runs:
        values = " | ".join(f"{report['by_topic'][topic]['accuracy']:.6f} ({report['by_topic'][topic]['correct']}/{report['by_topic'][topic]['n']})"
                            for topic in ["credit", "financial_position", "savings", "spending"])
        lines.append(f"| {cell(manifest['experiment'])} / {cell(directory.name)} | {cell(dataset_split(manifest))} | {values} |")
    lines += ["", "## Условия и артефакты", "",
              "Baseline и baseline_gemma: один zero-shot промпт, без обучения и примеров из train в промпте; весь диалог, только context и conversation; температура 0, max_output_tokens 2048, таймаут 120 с, без автоматических повторов; Pydantic-AI + LiteLLM, native JSON schema, метка good/bad.",
              "Exp1: разбор всех train-кейсов Flash с gold и сохранённой baseline-меткой → один синтез Argon → общий промпт и совет по topic → judge Flash. Gold оцениваемой выборки не передаётся judge. USD для exp1 показывает стоимость исходной адаптации плюс соответствующий инференс; при повторном использовании артефакта новый расход — только инференс. Исходные baseline-ответы переиспользуются.",
              "Exp2: те же 150 разборов train, что в exp1, переиспользованы без новых вызовов → один синтез Argon под рубрику из пяти критериев (ясные критерии, приоритеты, ссылки на train-примеры, границы, соответствие реализации) поверх фиксированного блока DEFINITIONS → базовый промпт с секциями TASK/CRITERIA/PRIORITY/BOUNDARIES/EXAMPLES + совет по topic → judge Flash. DEFINITIONS подставляется в каждый промпт без изменений и проверяется по хешу. USD в строке — эквивалентная стоимость pipeline, включая переиспользованные разборы; новый расход exp2 — синтез и инференс.",
              "Exp3: seed-промпт собран из синтезированных промптов exp1 и exp2 (без воды и без неиспользованного определения) → первый запрос Argon добавляет к каждому критерию ссылку на train-кейсы в формате [[ev: id | метка | фрагмент | критерий]] → второй запрос того же Argon вырезает только эти ссылки, и его ответ идёт в инференс. Промпт со ссылками сохраняется отдельно как объяснение решения; отсутствие ссылок и train-ID в промпте судьи проверяется офлайн.",
              "Стоимость — оценка LiteLLM по usage, не выписка провайдера. У Gemma 4 через Gemini API вход и выход бесплатны по [тарифу Google](https://ai.google.dev/gemini-api/docs/pricing#gemma-4); действуют квоты API.",
              "Это локальные public-метрики; близкие результаты сами по себе не доказывают статистическое преимущество.", ""]
    for directory, manifest, summary, report in runs:
        link = directory.relative_to(root).as_posix()
        lines += [f"- `{manifest['experiment']}` / `{directory.name}`: запрошена `{manifest['settings']['model']}`; "
                  f"[manifest]({link}/manifest.json), [prompt]({link}/prompt.md), [predictions]({link}/predictions.csv), "
                  f"[score]({link}/score.json), [usage]({link}/summary.json). "
                  f"Выборка: `{dataset_split(manifest)}`; SHA256 dataset (legacy поле test_sha256): `{manifest['test_sha256']}`; prompt: `{manifest['prompt_sha256']}`."]
        if "adaptation" in summary:
            adaptation = summary["adaptation"]
            inference = summary["inference"]
            synthesizers = ", ".join(adaptation["synthesis"]["response_models"])
            lines += [f"  Адаптация: ${adaptation['cost_usd_estimated']:.6f}; {dataset_split(manifest)}-инференс: ${inference['cost_usd_estimated']:.6f}; "
                      f"синтезировавшая модель: `{synthesizers}`. Хеш артефакта: `{manifest['artifact_sha256']}`."]
            if summary.get("adaptation_reused"):
                lines += ["  Артефакт адаптации переиспользован без новых API-запросов. USD в строке — стоимость эквивалентного полного pipeline; новый расход этого прогона — только инференс."]
            if dataset_split(manifest) == "train":
                lines += ["  Это оценка на train, использованном для построения артефакта (in-sample), а не независимая валидация."]
    destination = root / "RESULTS.md"
    temporary = destination.with_suffix(".md.tmp")
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    temporary.replace(destination)
    return destination


if __name__ == "__main__":
    print(update_results(Path(__file__).resolve().parent))
