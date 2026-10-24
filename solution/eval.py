"""Predict every unlabeled test ID using saved prompts; no train access or scorer."""
from dataclasses import asdict

from .common import SolutionError
from .common import (ARTIFACT_VERSION, Judgment, canonical, digest, directory_lock, frozen_manifest,
                     judge_payload, load_artifact, read_inputs, runtime_fingerprint, save_predictions,
                     utc_now, write_json)
from .llm import Runner, estimate, prices_for, usage_summary


async def run(test_path, artifacts, output, config, *, dry_run=False):
    if output == artifacts or artifacts in output.parents or output in artifacts.parents:
        raise SolutionError("Output and adaptation artifacts must be separate directories")
    rows = read_inputs(test_path, labeled=False)
    with directory_lock(artifacts, shared=True):
        _, prompts, adaptation = load_artifact(artifacts)
        if adaptation["inference_settings"] != asdict(config.inference):
            raise SolutionError("Judge model/parameters must match the adaptation run")
        # Include all adaptation reservations in the pipeline cost, even after a crash.
        from .llm import Budget
        adaptation_used = Budget(artifacts / "budget.jsonl", config.budget_usd).used()
        artifact_sha256 = digest((artifacts / "artifact.json").read_bytes())
        adaptation_sha256 = digest((artifacts / "summary.json").read_bytes())
    prices = prices_for(config.inference.model)
    jobs = [(row["item_id"], prompts[row["topic"]], judge_payload(row)) for row in rows]
    if dry_run:
        result = {"mode": "dry-run", "test_examples": len(rows), "model": config.inference.model,
                  "estimated_upper_usd_before_cache": sum(estimate(prompt, payload, Judgment, config.inference, prices)
                                                          for _, prompt, payload in jobs),
                  "note": "No API calls or files written."}
        print(canonical(result), flush=True)
        return result
    output.mkdir(parents=True, exist_ok=True)
    with directory_lock(output):
        frozen_manifest(output, {"command": "eval", "test_sha256": digest(test_path.read_bytes()), "n": len(rows),
                                 "artifact_sha256": artifact_sha256, "adaptation_sha256": adaptation_sha256,
                                 "settings": asdict(config.inference), "budget_usd": config.budget_usd,
                                 "concurrency": config.concurrency, "request_interval_seconds": config.request_interval_seconds,
                                 "prices": prices, "runtime": runtime_fingerprint()})
        runner = Runner(output, config)
        # The configured cap covers adapt + this eval. Artifacts remain read-only.
        runner.budget.limit = config.budget_usd - adaptation_used
        runner.budget.require(0)
        write_json(output / "summary.json", {"status": "running", "artifact_version": ARTIFACT_VERSION})
        results = await runner.stage("inference", jobs, Judgment, config.inference, prices)
        save_predictions(output / "predictions.csv", rows, {item_id: value.label for item_id, value in results.items()})
        summary = {"status": "complete", "artifact_version": ARTIFACT_VERSION, "test_examples": len(rows),
                   "artifact_sha256": artifact_sha256, "stages": usage_summary(output),
                   "inference_accounted_usd": runner.budget.used(), "adaptation_accounted_usd": adaptation_used,
                   "pipeline_accounted_usd": adaptation_used + runner.budget.used(), "budget_usd": config.budget_usd,
                   "predictions_sha256": digest((output / "predictions.csv").read_bytes()), "completed_at": utc_now()}
        write_json(output / "summary.json", summary)
        (output / "failure.json").unlink(missing_ok=True)
        print(canonical(summary), flush=True)
        return summary
