"""Zero-shot model inference + official scoring. Run from any working directory."""
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
import argparse
import asyncio
import csv
import fcntl
import json
import os
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))
from common import load_dataset, load_labels
from evaluate import score
from prepare_inputs import model_rows
from experiments.llm import (Budget, Judgment, Settings, digest, judge_one, make_judge,
                             prices_for, request_ceiling, utc_now, write_json)
from experiments.update_results import update_results

HERE = Path(__file__).resolve().parent


def save_predictions(path: Path, rows: list[dict], predictions: dict) -> None:
    temp = path.with_suffix(".csv.tmp")
    with temp.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["item_id", "label"])
        writer.writeheader()
        writer.writerows({"item_id": r["item_id"], "label": predictions[r["item_id"]]}
                         for r in rows if r["item_id"] in predictions)
    temp.replace(path)


def summarize(run_dir: Path, budget: Budget, total: int, completed: int, report=None) -> dict:
    calls = [json.loads(line) for line in (run_dir / "calls.jsonl").read_text().splitlines()] if (run_dir / "calls.jsonl").exists() else []
    result = {"status": "complete" if completed == total else "partial", "completed": completed, "total": total,
              "api_responses_this_run": len(calls),
              "cost_usd_estimated_this_run": sum(c["cost_usd_estimated"] for c in calls),
              "accounted_usd_entire_experiment": budget.used(), "experiment_budget_usd": budget.limit,
              "cost_note": "LiteLLM price estimate; failed requests retain conservative reservations",
              "response_models": sorted({c["response_model"] for c in calls}),
              "input_tokens": sum(c["usage"].get("prompt_tokens", 0) for c in calls),
              "output_tokens": sum(c["usage"].get("completion_tokens", 0) for c in calls),
              "updated_at": utc_now()}
    if report is not None:
        result["macro_topic_accuracy"] = report["score"]
        result["accuracy"] = report["accuracy"]
    write_json(run_dir / "summary.json", result)
    return result


async def run(args) -> None:
    here = args.experiment_dir
    settings = Settings.from_env(here / ".env")
    if args.expected_model is not None and settings.model != args.expected_model:
        raise ValueError("MODEL does not match this experiment; copy its .env.example to .env and check environment overrides")
    dataset = args.test.resolve()
    gold = load_dataset(dataset, require_labels=True)
    # Gold is excluded even when evaluating the train split; no adaptation.
    inputs = model_rows(gold, train=False, language=args.language)
    prompt = (here / "prompt.md").read_text(encoding="utf-8").strip()
    prices = prices_for(settings.model)
    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    budget = Budget(here / ".budget.jsonl", settings.budget_usd)
    config = {"experiment": args.experiment_name, "settings": asdict(settings), "language": args.language,
              "dataset_split": args.split or {"public_train.csv": "train", "public_test.csv": "test"}.get(dataset.name, "evaluation"),
              "request_interval_seconds": args.request_interval,
              "test_sha256": digest(dataset.read_bytes()), "prompt_sha256": digest(prompt),
              "source_sha256": {str(p.relative_to(REPO)): digest(p.read_bytes()) for p in
                                dict.fromkeys([Path(__file__).resolve(), args.entrypoint, REPO / "experiments/llm.py",
                                               REPO / "src/evaluate.py", REPO / "src/common.py", REPO / "src/prepare_inputs.py"])},
              "packages": {p: version(p) for p in ["litellm", "pydantic-ai-litellm", "pydantic-ai-slim", "pydantic", "python-dotenv"]},
              "prices": prices, "structured_output": Judgment.model_json_schema(), "n": len(inputs)}
    config_hash = digest(json.dumps(config, sort_keys=True))
    manifest_path = run_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest["config_sha256"] != config_hash:
            raise ValueError("Run configuration changed; choose a new --run-dir")
    else:
        if any(run_dir.iterdir()):
            raise ValueError("New run directory must be empty")
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True, check=True).stdout.strip()
        manifest = {**config, "config_sha256": config_hash, "created_at": utc_now(),
                    "git_commit": commit, "dataset_path": str(dataset), "python": sys.version.split()[0],
                    "note": "Hashes capture working files; commit may predate uncommitted experiment code"}
        write_json(manifest_path, manifest)
        (run_dir / "prompt.md").write_text(prompt + "\n", encoding="utf-8")
    prediction_path = run_dir / "predictions.csv"
    existing = load_labels(prediction_path) if prediction_path.exists() else []
    predictions = {p["item_id"]: p["label"] for p in existing}
    expected = {r["item_id"] for r in inputs}
    if not set(predictions) <= expected:
        raise ValueError("Saved predictions contain unknown IDs")
    pending = [r for r in inputs if r["item_id"] not in predictions]
    selected = pending[:args.limit] if args.limit is not None else pending
    estimates = [request_ceiling([
        {"role": "system", "content": prompt},
        {"role": "user", "content": json.dumps({"context": r["context"], "conversation": r["conversation"]},
                                                 ensure_ascii=False, separators=(",", ":"))}],
        Judgment.model_json_schema(), settings, prices) for r in selected]
    print(json.dumps({"model": settings.model, "language": args.language, "saved": len(predictions),
                      "planned_requests": len(selected), "estimated_upper_usd": sum(estimates),
                      "experiment_accounted_usd": budget.used(), "run_dir": str(run_dir)}, ensure_ascii=False), flush=True)
    if args.dry_run:
        return
    budget.require(sum(estimates))
    if selected:
        if settings.model.startswith("gemini/") and not os.getenv("GEMINI_API_KEY"):
            raise ValueError("GEMINI_API_KEY is missing from the environment/.env")
        agent, model = make_judge(prompt, settings, budget, run_dir / "calls.jsonl", run_dir.name, prices)
        last_started = None
        for row in selected:
            if last_started is not None:
                await asyncio.sleep(max(0, args.request_interval - (time.monotonic() - last_started)))
            last_started = time.monotonic()
            result = await judge_one(agent, model, row)
            predictions[row["item_id"]] = result.label
            save_predictions(prediction_path, inputs, predictions)
            summarize(run_dir, budget, len(inputs), len(predictions))
            print(f"{len(predictions)}/{len(inputs)} {row['item_id']} {result.label} experiment_usd={budget.used():.6f}", flush=True)
    report = None
    if len(predictions) == len(inputs):
        # Reload gold only for scoring; all model calls have already finished.
        report = score(load_dataset(dataset, require_labels=True), load_labels(prediction_path))
        write_json(run_dir / "score.json", report)
    summary = summarize(run_dir, budget, len(inputs), len(predictions), report)
    if report is not None:
        update_results(REPO / "experiments")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main(*, experiment_dir: Path = HERE, experiment_name: str = "baseline",
         expected_model: str | None = None, entrypoint: Path = Path(__file__).resolve()) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", "--test", dest="test", type=Path, default=REPO / "data/public_test.csv",
                        help="Labeled evaluation dataset; gold is never sent to the model")
    parser.add_argument("--split", choices=["train", "test"], help="Table label; inferred for public_train.csv/public_test.csv")
    parser.add_argument("--request-interval", type=float, default=0, help="Minimum seconds between request starts; does not change model parameters")
    parser.add_argument("--language", choices=["original", "en", "es"], default="original")
    parser.add_argument("--run-dir", type=Path, default=experiment_dir / "runs" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    parser.add_argument("--limit", type=int, help="New API requests this invocation; same --run-dir resumes")
    parser.add_argument("--dry-run", action="store_true", help="Save configuration and estimate costs without API calls")
    args = parser.parse_args()
    args.experiment_dir, args.experiment_name = experiment_dir, experiment_name
    args.expected_model, args.entrypoint = expected_model, entrypoint
    if args.limit is not None and args.limit < 0:
        parser.error("--limit must be nonnegative")
    if not 0 <= args.request_interval < float("inf"):
        parser.error("--request-interval must be finite and nonnegative")
    try:
        with (experiment_dir / ".run.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            asyncio.run(run(args))
    except Exception as exc:
        # Provider exception text can contain a credential-bearing URL or request.
        failure = {"error_type": type(exc).__name__, "status_code": getattr(exc, "status_code", None), "time": utc_now()}
        if args.run_dir.is_dir():
            write_json(args.run_dir / "failure.json", failure)
        print(json.dumps(failure), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
