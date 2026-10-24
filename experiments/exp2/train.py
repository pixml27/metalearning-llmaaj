"""Reuse the frozen exp1 train analyses; synthesize one rubric-shaped prompt over fixed definitions."""
from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
import argparse
import asyncio
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.exp2.common import *


async def run(args):
    stages = stage_settings()
    gold, rows, baseline_prompt, provenance = reused_analyses(args.train, args.baseline, args.analyses, args.language)
    definitions = (HERE / "definitions.md").read_text().strip()
    synthesis_prompt = (HERE / "synthesis_prompt.md").read_text().strip()
    output = args.artifacts.resolve()
    output.mkdir(parents=True, exist_ok=True)
    prices = {stage: prices_for(cfg.model) for stage, cfg in stages.items()}
    config = {"experiment": "exp2", "dataset_split": "train", "n": len(gold), "language": args.language,
              "train_sha256": digest(args.train.read_bytes()),
              "baseline_predictions_sha256": digest((args.baseline / "predictions.csv").read_bytes()),
              "baseline_manifest_sha256": digest((args.baseline / "manifest.json").read_bytes()),
              "baseline_prompt_sha256": digest(baseline_prompt),
              "definitions_sha256": digest(definitions),
              "synthesis_prompt_sha256": digest(synthesis_prompt),
              "reused_analyses": provenance,
              "settings": {k: asdict(v) for k, v in stages.items()}, "prices": prices,
              "source_sha256": source_hashes(), "packages": packages()}
    frozen_manifest(output, config)
    for name, text in [("baseline_prompt.md", baseline_prompt), ("definitions.md", definitions),
                       ("synthesis_prompt.md", synthesis_prompt)]:
        (output / name).write_text(text + "\n")
    (output / "analyses.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    write_analyses_csv(output / "analyses.csv", rows)
    payload = synthesis_payload(rows, baseline_prompt, definitions)
    write_json(output / "synthesis_input.json", payload)
    budget = Budget(HERE / ".budget.jsonl", stages["synthesis"].budget_usd)
    calls_path = output / "synthesis_calls.jsonl"
    cached = cached_outputs(calls_path, PromptArtifact)
    ceiling = estimate(synthesis_prompt, payload, PromptArtifact, stages["synthesis"], prices["synthesis"])
    print(json.dumps({"stage": "synthesis", "n_train_cases": len(rows), "analysis_calls_reused": True,
                      "cached_synthesis": "synthesis" in cached, "estimated_upper_usd": ceiling,
                      "input_utf8_bytes": len(json.dumps(payload, ensure_ascii=False).encode()),
                      "reserved_for_inference_estimate_usd": 1,
                      "accounted_entire_experiment_usd": budget.used(),
                      "models": {k: v.model for k, v in stages.items()}}, indent=2), flush=True)
    if args.dry_run:
        return
    if "synthesis" in cached:
        artifact = cached["synthesis"]
    else:
        budget.require(ceiling + 1)
        artifact = await structured_call(synthesis_prompt, payload, PromptArtifact, stages["synthesis"],
                                         budget, calls_path, output.name, "synthesis")
    write_json(output / "artifact.json", artifact.model_dump())
    prompts = compiled_prompts(artifact, definitions)
    write_json(output / "routed_prompts.json", prompts)
    (output / "prompt.md").write_text(artifact.base_prompt.strip() + "\n\n" + definitions + "\n\n" + "\n\n".join(
        f"## {topic}\n{text}" for topic, text in artifact.topic_advice.model_dump().items()) + "\n")
    checks = structure_check(artifact, rows, definitions)
    synthesis = cost_summary([calls_path])
    summary = {"status": "complete", "completed": len(rows), "total": len(rows),
               "analysis": provenance["analysis"], "analysis_reused_from": provenance["analyses_directory"],
               "analysis_new_api_calls": 0, "synthesis": synthesis,
               "cost_usd_estimated": synthesis["cost_usd_estimated"] + provenance["analysis"]["cost_usd_estimated"],
               "cost_usd_estimated_new": synthesis["cost_usd_estimated"],
               "cost_note": "Equivalent adaptation cost: exp1 train analyses reused without new API calls plus this synthesis",
               "artifact_sha256": digest((output / "artifact.json").read_bytes()),
               "routed_prompts_sha256": digest((output / "routed_prompts.json").read_bytes()),
               "definitions_sha256": digest(definitions), "structure_check": checks,
               "rubric_selfcheck": artifact.rubric_selfcheck.model_dump(),
               "accounted_entire_experiment_usd": budget.used(), "updated_at": utc_now()}
    write_json(output / "summary.json", summary)
    print(json.dumps({k: summary[k] for k in ["status", "synthesis", "structure_check",
                                              "accounted_entire_experiment_usd"]}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=REPO / "data/public_train.csv")
    parser.add_argument("--baseline", type=Path, default=REPO / "experiments/baseline/results/20261024-public-train-original")
    parser.add_argument("--analyses", type=Path, default=REPO / "experiments/exp1/artifacts/20261024-public-original",
                        help="Completed exp1 adaptation directory supplying the frozen A/B table")
    parser.add_argument("--artifacts", type=Path, default=HERE / "artifacts/20261024-public-original")
    parser.add_argument("--language", choices=["original", "en", "es"], default="original")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
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
