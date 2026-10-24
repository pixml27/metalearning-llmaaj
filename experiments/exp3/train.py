"""Two Argon requests: a prompt with training evidence, then the same prompt with the evidence cut."""
from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
import argparse
import asyncio
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.exp3.common import *


async def run(args):
    stages = stage_settings()
    gold, rows, baseline_prompt, provenance = reused_analyses(args.train, args.baseline, args.analyses, args.language)
    definitions = (HERE / "definitions.md").read_text().strip()
    seed = (HERE / "seed_prompt.md").read_text().strip()
    seed_base, seed_advice = parse_bundle(seed)
    synthesis_prompt = (HERE / "synthesis_prompt.md").read_text().strip()
    strip_prompt = (HERE / "strip_prompt.md").read_text().strip()
    output = args.artifacts.resolve()
    output.mkdir(parents=True, exist_ok=True)
    prices = {stage: prices_for(cfg.model) for stage, cfg in stages.items()}
    config = {"experiment": "exp3", "dataset_split": "train", "n": len(gold), "language": args.language,
              "train_sha256": digest(args.train.read_bytes()),
              "baseline_predictions_sha256": digest((args.baseline / "predictions.csv").read_bytes()),
              "baseline_manifest_sha256": digest((args.baseline / "manifest.json").read_bytes()),
              "baseline_prompt_sha256": digest(baseline_prompt),
              "definitions_sha256": digest(definitions), "seed_prompt_sha256": digest(seed),
              "seed_sources": ["experiments/exp1/artifacts/20261024-public-original/prompt.md",
                               "experiments/exp2/artifacts/20261024-public-original/prompt.md"],
              "synthesis_prompt_sha256": digest(synthesis_prompt), "strip_prompt_sha256": digest(strip_prompt),
              "reused_analyses": provenance,
              "settings": {k: asdict(v) for k, v in stages.items()}, "prices": prices,
              "source_sha256": source_hashes(), "packages": packages()}
    frozen_manifest(output, config)
    for name, text in [("baseline_prompt.md", baseline_prompt), ("definitions.md", definitions),
                       ("seed_prompt.md", seed), ("synthesis_prompt.md", synthesis_prompt),
                       ("strip_prompt.md", strip_prompt)]:
        (output / name).write_text(text + "\n")
    (output / "analyses.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    write_analyses_csv(output / "analyses.csv", rows)
    payload = synthesis_payload(rows, seed, definitions)
    write_json(output / "synthesis_input.json", payload)
    budget = Budget(HERE / ".budget.jsonl", stages["synthesis"].budget_usd)
    calls_path = output / "synthesis_calls.jsonl"
    linked_cache = cached_outputs(calls_path, LinkedArtifact, "synthesis")
    ceiling = estimate(synthesis_prompt, payload, LinkedArtifact, stages["synthesis"], prices["synthesis"])
    print(json.dumps({"stage": "synthesis", "n_train_cases": len(rows), "analysis_calls_reused": True,
                      "seed_base_words": len(seed_base.split()),
                      "seed_advice_words": {t: len(v.split()) for t, v in seed_advice.items()},
                      "cached_synthesis": bool(linked_cache), "estimated_upper_usd": ceiling,
                      "input_utf8_bytes": len(json.dumps(payload, ensure_ascii=False).encode()),
                      "reserved_for_strip_and_inference_estimate_usd": 1.3,
                      "accounted_entire_experiment_usd": budget.used(),
                      "models": {k: v.model for k, v in stages.items()}}, indent=2), flush=True)
    if args.dry_run:
        return
    if linked_cache:
        linked = linked_cache["synthesis"]
    else:
        budget.require(ceiling + 1.3)
        linked = await structured_call(synthesis_prompt, payload, LinkedArtifact, stages["synthesis"],
                                       budget, calls_path, output.name, "synthesis")
    write_json(output / "linked_artifact.json", linked.model_dump())
    (output / "prompt_with_links.md").write_text(bundle_text(linked.base_prompt_with_links,
                                                             linked.topic_advice_with_links.model_dump()))
    links = link_report(linked, rows)
    print(json.dumps({"stage": "links", **links}, indent=2), flush=True)
    strip_input = strip_payload(linked)
    write_json(output / "strip_input.json", strip_input)
    strip_cache = {} if args.restrip else cached_outputs(calls_path, PromptArtifact, "strip")
    if strip_cache:
        artifact = strip_cache["strip"]
    else:
        strip_ceiling = estimate(strip_prompt, strip_input, PromptArtifact, stages["synthesis"], prices["synthesis"])
        budget.require(strip_ceiling)
        print(json.dumps({"stage": "strip", "estimated_upper_usd": strip_ceiling}), flush=True)
        artifact = await structured_call(strip_prompt, strip_input, PromptArtifact, stages["synthesis"],
                                         budget, calls_path, output.name, "strip")
    checks = strip_report(linked, artifact, rows, definitions)
    synthesis = cost_summary([calls_path])
    summary_payload = {"status": "complete", "completed": len(rows), "total": len(rows),
                       "analysis": provenance["analysis"], "analysis_reused_from": provenance["analyses_directory"],
                       "analysis_new_api_calls": 0, "synthesis": synthesis,
                       "cost_usd_estimated": synthesis["cost_usd_estimated"] + provenance["analysis"]["cost_usd_estimated"],
                       "cost_usd_estimated_new": synthesis["cost_usd_estimated"],
                       "cost_note": "Equivalent adaptation cost: exp1 train analyses reused without new API calls plus both exp3 Argon requests",
                       "definitions_sha256": digest(definitions), "link_report": links, "strip_report": checks,
                       "rubric_selfcheck": linked.rubric_selfcheck.model_dump(),
                       "accounted_entire_experiment_usd": budget.used(), "updated_at": utc_now()}
    print(json.dumps({"stage": "strip_report", **checks}, indent=2), flush=True)
    if checks["evidence_markers_left"] or checks["training_cases_left"]:
        write_json(output / "strip_failure.json", summary_payload)
        raise ValueError("Inference prompt still carries training evidence; rerun with --restrip")
    write_json(output / "artifact.json", artifact.model_dump())
    prompts = compiled_prompts(artifact, definitions)
    write_json(output / "routed_prompts.json", prompts)
    (output / "prompt.md").write_text(artifact.base_prompt.strip() + "\n\n" + definitions + "\n\n" + "\n\n".join(
        f"## {topic}\n{text}" for topic, text in artifact.topic_advice.model_dump().items()) + "\n")
    summary = {**summary_payload,
               "artifact_sha256": digest((output / "artifact.json").read_bytes()),
               "linked_artifact_sha256": digest((output / "linked_artifact.json").read_bytes()),
               "routed_prompts_sha256": digest((output / "routed_prompts.json").read_bytes())}
    write_json(output / "summary.json", summary)
    print(json.dumps({k: summary[k] for k in ["status", "synthesis", "accounted_entire_experiment_usd"]}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=REPO / "data/public_train.csv")
    parser.add_argument("--baseline", type=Path, default=REPO / "experiments/baseline/results/20261024-public-train-original")
    parser.add_argument("--analyses", type=Path, default=REPO / "experiments/exp1/artifacts/20261024-public-original",
                        help="Completed exp1 adaptation directory supplying the frozen A/B table")
    parser.add_argument("--artifacts", type=Path, default=HERE / "artifacts/20261024-public-original")
    parser.add_argument("--language", choices=["original", "en", "es"], default="original")
    parser.add_argument("--restrip", action="store_true", help="Request the link-free prompt again instead of reusing the saved answer")
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
