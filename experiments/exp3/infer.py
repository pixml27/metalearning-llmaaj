"""Apply the frozen exp3 link-free prompts (rules + definitions + topic advice) with Flash."""
from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
import argparse
import asyncio
import csv
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.exp3.common import *
from experiments.llm import Judgment
from experiments.inference import add_inference_arguments, run_judgments
from experiments.update_results import update_results


def save_predictions(path, inputs, predictions):
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["item_id", "label"])
        writer.writeheader()
        writer.writerows({"item_id": r["item_id"], "label": predictions[r["item_id"]]}
                         for r in inputs if r["item_id"] in predictions)
    temporary.replace(path)


async def run(args):
    settings = stage_settings()["inference"]
    artifact, prompts, definitions, adaptation = load_artifact(args.artifacts)
    dataset = args.dataset.resolve()
    split = args.split or {"public_train.csv": "train", "public_test.csv": "test"}.get(dataset.name, "evaluation")
    # The prediction input is an allowlist without gold; no train files are needed.
    inputs = model_rows(load_dataset(dataset), train=False, language=args.language)
    output = args.run_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    bundle = (args.artifacts / "prompt.md").read_text().strip()
    prices = prices_for(settings.model)
    config = {"experiment": "exp3", "dataset_split": split, "n": len(inputs), "language": args.language,
              "settings": asdict(settings), "prices": prices, "packages": packages(),
              "concurrency": args.concurrency, "request_interval_seconds": args.request_interval,
              "dataset_path": str(dataset), "test_sha256": digest(dataset.read_bytes()),
              "prompt_sha256": digest(bundle), "artifact_directory": str(args.artifacts.resolve()),
              "artifact_sha256": digest((args.artifacts / "artifact.json").read_bytes()),
              "adaptation_manifest_sha256": digest((args.artifacts / "manifest.json").read_bytes()),
              "analysis_table_sha256": digest((args.artifacts / "analyses.jsonl").read_bytes()),
              "linked_artifact_sha256": digest((args.artifacts / "linked_artifact.json").read_bytes()),
              "definitions_sha256": digest(definitions),
              "routed_prompts_sha256": digest(json.dumps(prompts, sort_keys=True)),
              "routing": "provided topic -> base prompt + fixed definitions + advice for exactly that topic",
              "source_sha256": source_hashes()}
    frozen_manifest(output, config)
    (output / "prompt.md").write_text(bundle + "\n")
    write_json(output / "routed_prompts.json", prompts)
    budget = Budget(HERE / ".budget.jsonl", settings.budget_usd)
    calls_path = output / "calls.jsonl"
    predictions_path = output / "predictions.csv"
    predictions = {r["item_id"]: r["label"] for r in load_labels(predictions_path)} if predictions_path.exists() else {}
    recovered = cached_outputs(calls_path, Judgment)
    predictions.update({iid: value.label for iid, value in recovered.items()})
    expected = {r["item_id"] for r in inputs}
    if not set(predictions) <= expected:
        raise ValueError("Prediction cache contains unknown IDs")
    pending = [r for r in inputs if r["item_id"] not in predictions]
    selected = pending[:args.limit] if args.limit is not None else pending
    upper = sum(estimate(prompts[r["topic"]], {"context": r["context"], "conversation": r["conversation"]},
                         Judgment, settings, prices) for r in selected)
    print(json.dumps({"stage": "inference", "cached": len(predictions), "planned_requests": len(selected),
                      "estimated_upper_usd": upper, "accounted_entire_experiment_usd": budget.used()}), flush=True)
    if args.dry_run:
        return
    budget.require(upper)

    def summarize(report=None):
        inference = cost_summary([calls_path])
        summary = {"status": "complete" if len(predictions) == len(inputs) else "partial",
                   "completed": len(predictions), "total": len(inputs),
                   "response_models": inference["response_models"], "inference": inference,
                   "adaptation": adaptation, "api_responses_this_run": inference["api_responses"],
                   "cost_usd_estimated_this_run": adaptation["cost_usd_estimated"] + inference["cost_usd_estimated"],
                   "adaptation_reused": True, "dataset_split": split,
                   "evaluation_note": "in-sample: artifact learned from this train set" if split == "train" else "fixed artifact evaluation",
                   "cost_note": "Equivalent pipeline cost includes the exp1 train analyses reused without new calls, both exp3 Argon requests and this evaluation; LiteLLM estimate, not invoice",
                   "accounted_usd_entire_experiment": budget.used(), "experiment_budget_usd": budget.limit,
                   "input_tokens": inference["input_tokens"], "output_tokens": inference["output_tokens"],
                   "updated_at": utc_now()}
        if report is not None:
            summary.update(macro_topic_accuracy=report["score"], accuracy=report["accuracy"])
        write_json(output / "summary.json", summary)
        return summary

    if predictions:
        save_predictions(predictions_path, inputs, predictions)
    def save_result(row, result):
        predictions[row["item_id"]] = result.label
        save_predictions(predictions_path, inputs, predictions)
        summarize()
        print(f"inference {len(predictions)}/{len(inputs)} {row['topic']} {row['item_id']} {result.label} experiment_usd={budget.used():.6f}", flush=True)

    await run_judgments(selected, prompt_for=lambda row: prompts[row["topic"]], settings=settings,
                        budget=budget, calls_path=calls_path, run_id=output.name, prices=prices,
                        on_result=save_result, concurrency=args.concurrency, interval=args.request_interval)
    report = None
    if len(predictions) == len(inputs):
        gold = load_dataset(dataset, require_labels=True)
        report = score(gold, load_labels(predictions_path))
        write_json(output / "score.json", report)
        errors = [{"item_id": r["item_id"], "topic": r["topic"], "gold": r["label"], "prediction": predictions[r["item_id"]]}
                  for r in gold if predictions[r["item_id"]] != r["label"]]
        with (output / "errors.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=["item_id", "topic", "gold", "prediction"])
            writer.writeheader()
            writer.writerows(errors)
    summary = summarize(report)
    if report is not None:
        update_results(REPO / "experiments")
    print(json.dumps(summary, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=HERE / "artifacts/20261024-public-original")
    parser.add_argument("--dataset", type=Path, default=REPO / "data/public_test.csv")
    parser.add_argument("--split", choices=["train", "test"], help="Inferred for public_train.csv and public_test.csv")
    parser.add_argument("--run-dir", type=Path, default=HERE / "runs/20261024-public-original")
    parser.add_argument("--language", choices=["original", "en", "es"], default="original")
    add_inference_arguments(parser)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 0:
        parser.error("--limit must be nonnegative")
    try:
        with experiment_lock():
            asyncio.run(run(args))
    except Exception as exc:
        failure = {"error_type": type(exc).__name__, "status_code": getattr(exc, "status_code", None), "time": utc_now()}
        if args.run_dir.is_dir():
            write_json(args.run_dir / "failure.json", failure)
        print(json.dumps(failure), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
