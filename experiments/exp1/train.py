"""Audit all train cases, then synthesize one base prompt and four topic advices."""
from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
import argparse
import asyncio
import json
import os
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.exp1.common import *


async def run(args):
    stages = stage_settings()
    concurrency = int(os.getenv("ANALYSIS_CONCURRENCY", "3"))
    interval = float(os.getenv("ANALYSIS_REQUEST_INTERVAL", "4.1"))
    if not 1 <= concurrency <= 8 or not 0 <= interval < float("inf"):
        raise ValueError("Invalid analysis concurrency/request interval")
    gold, payloads, baseline_prompt, baseline_manifest = analysis_inputs(args.train, args.baseline, args.language)
    analysis_prompt = (HERE / "analysis_prompt.md").read_text().strip()
    synthesis_prompt = (HERE / "synthesis_prompt.md").read_text().strip()
    output = args.artifacts.resolve()
    output.mkdir(parents=True, exist_ok=True)
    prices = {stage: prices_for(cfg.model) for stage, cfg in stages.items()}
    config = {"experiment": "exp1", "dataset_split": "train", "n": len(gold), "language": args.language,
              "train_sha256": digest(args.train.read_bytes()),
              "baseline_predictions_sha256": digest((args.baseline / "predictions.csv").read_bytes()),
              "baseline_manifest_sha256": digest((args.baseline / "manifest.json").read_bytes()),
              "baseline_prompt_sha256": digest(baseline_prompt),
              "analysis_prompt_sha256": digest(analysis_prompt), "synthesis_prompt_sha256": digest(synthesis_prompt),
              "settings": {k: asdict(v) for k, v in stages.items()}, "prices": prices,
              "concurrency": concurrency, "request_interval_seconds": interval,
              "source_sha256": source_hashes(), "packages": packages()}
    manifest = frozen_manifest(output, config)
    for name, text in [("baseline_prompt.md", baseline_prompt), ("analysis_prompt.md", analysis_prompt), ("synthesis_prompt.md", synthesis_prompt)]:
        (output / name).write_text(text + "\n")
    budget = Budget(HERE / ".budget.jsonl", stages["analysis"].budget_usd)
    analysis_calls, synthesis_calls = output / "analysis_calls.jsonl", output / "synthesis_calls.jsonl"
    cached = cached_outputs(analysis_calls, CaseAnalysis)
    expected = {p["item_id"] for p in payloads}
    if not set(cached) <= expected:
        raise ValueError("Analysis cache contains unknown IDs")
    pending = [p for p in payloads if p["item_id"] not in cached]
    selected = pending[:args.limit] if args.limit is not None else pending
    upper = sum(estimate(analysis_prompt, p, CaseAnalysis, stages["analysis"], prices["analysis"]) for p in selected)
    print(json.dumps({"stage": "analysis", "cached": len(cached), "planned_requests": len(selected),
                      "estimated_upper_usd": upper, "reserved_for_later_stages_estimate_usd": 2,
                      "accounted_entire_experiment_usd": budget.used(), "models": {k:v.model for k,v in stages.items()}}, indent=2), flush=True)
    if args.dry_run:
        return
    budget.require(upper + (2 if selected else 0))
    semaphore, gate = asyncio.Semaphore(concurrency), StartGate(interval)
    failures = []

    def table():
        result = []
        for p in payloads:
            if p["item_id"] in cached:
                result.append({k: p[k] for k in ["item_id", "topic", "gold_label", "baseline_label", "baseline_correct"]}
                              | cached[p["item_id"]].model_dump())
        return result

    def save_table():
        values = table()
        temporary = output / "analyses.jsonl.tmp"
        temporary.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in values))
        temporary.replace(output / "analyses.jsonl")
        write_analyses_csv(output / "analyses.csv", values)
        write_json(output / "summary.json", {"status": "analysing", "completed": len(values), "total": len(payloads),
                   "analysis": cost_summary([analysis_calls]), "accounted_entire_experiment_usd": budget.used(), "updated_at": utc_now()})

    save_table()
    async def audit(payload):
        async with semaphore:
            if failures:
                return
            await gate.wait()
            if failures:
                return
            try:
                result = await structured_call(analysis_prompt, payload, CaseAnalysis, stages["analysis"],
                                               budget, analysis_calls, output.name, payload["item_id"])
                cached[payload["item_id"]] = result
                save_table()
                print(f"analysis {len(cached)}/{len(payloads)} {payload['item_id']} experiment_usd={budget.used():.6f}", flush=True)
            except Exception as exc:
                failures.append(exc)
    await asyncio.gather(*(audit(p) for p in selected))
    if failures:
        raise failures[0]
    if len(cached) < len(payloads) or args.analysis_only:
        return
    values = table()
    synth_input = synthesis_payload(values, baseline_prompt)
    write_json(output / "synthesis_input.json", synth_input)
    synth_cache = cached_outputs(synthesis_calls, PromptArtifact)
    if "synthesis" in synth_cache:
        artifact = synth_cache["synthesis"]
    else:
        ceiling = estimate(synthesis_prompt, synth_input, PromptArtifact, stages["synthesis"], prices["synthesis"])
        budget.require(ceiling)
        print(json.dumps({"stage": "synthesis", "n_train_cases": len(values), "estimated_upper_usd": ceiling,
                          "input_utf8_bytes": len(json.dumps(synth_input, ensure_ascii=False).encode())}), flush=True)
        artifact = await structured_call(synthesis_prompt, synth_input, PromptArtifact, stages["synthesis"],
                                         budget, synthesis_calls, output.name, "synthesis")
    write_json(output / "artifact.json", artifact.model_dump())
    prompts = compiled_prompts(artifact)
    write_json(output / "routed_prompts.json", prompts)
    (output / "prompt.md").write_text(artifact.base_prompt + "\n\n" + "\n\n".join(
        f"## {topic}\n{text}" for topic, text in artifact.topic_advice.model_dump().items()) + "\n")
    summary = {"status": "complete", "completed": len(values), "total": len(payloads),
               "analysis": cost_summary([analysis_calls]), "synthesis": cost_summary([synthesis_calls]),
               "cost_usd_estimated": cost_summary([analysis_calls, synthesis_calls])["cost_usd_estimated"],
               "artifact_sha256": digest((output / "artifact.json").read_bytes()),
               "routed_prompts_sha256": digest((output / "routed_prompts.json").read_bytes()),
               "accounted_entire_experiment_usd": budget.used(), "updated_at": utc_now()}
    write_json(output / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=REPO / "data/public_train.csv")
    parser.add_argument("--baseline", type=Path, default=REPO / "experiments/baseline/results/20261024-public-train-original")
    parser.add_argument("--artifacts", type=Path, default=HERE / "artifacts/20261024-public-original")
    parser.add_argument("--language", choices=["original", "en", "es"], default="original")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--analysis-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 0:
        parser.error("--limit must be nonnegative")
    try:
        with experiment_lock():
            asyncio.run(run(args))
    except Exception as exc:
        failure = {"error_type": type(exc).__name__, "status_code": getattr(exc, "status_code", None), "time": utc_now()}
        if args.artifacts.is_dir():
            write_json(args.artifacts / "failure.json", failure)
        print(json.dumps(failure), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
