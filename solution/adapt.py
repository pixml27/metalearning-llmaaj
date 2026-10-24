"""Audit train cases, synthesize exp3 rules with evidence, strip it, then explain."""
from dataclasses import asdict
import json

from .common import SolutionError
from .common import (ARTIFACT_VERSION, CaseAnalysis, Judgment, PromptArtifact, atomic_text,
                     canonical, compiled_prompts, digest, directory_lock, frozen_manifest,
                     judge_payload, load_artifact, prompt_text, read_inputs, runtime_fingerprint,
                     save_predictions, utc_now, write_json)
from .explanation import Explanation, render_explanation, validate_explanation
from .llm import Runner, estimate, prices_for, usage_summary
from .synthesis import LinkedArtifact, link_report, strip_payload, strip_report, synthesis_payload


async def run(train_path, directory, config, *, dry_run=False):
    rows = read_inputs(train_path, labeled=True)
    baseline_prompt = prompt_text("baseline.md")
    settings = {"baseline": config.inference, "analysis": config.analysis,
                "synthesis": config.synthesis, "strip": config.synthesis, "explanation": config.explanation}
    prices = {stage: prices_for(value.model) for stage, value in settings.items()}
    if dry_run:
        # Later stages depend on model outputs; their exact estimates are printed before those requests.
        baseline_upper = sum(estimate(baseline_prompt, judge_payload(row), Judgment, config.inference,
                                      prices["baseline"]) for row in rows)
        result = {"mode": "dry-run", "train_examples": len(rows),
                  "request_counts": {"baseline": len(rows), "analysis": len(rows), "synthesis": 1, "strip": 1, "explanation": 1},
                  "baseline_upper_usd_before_cache": baseline_upper, "budget_usd": config.budget_usd,
                  "models": {name: value.model for name, value in settings.items()},
                  "note": "No API calls or files written. Later stages are priced from their actual inputs before execution."}
        print(canonical(result), flush=True)
        return result
    directory.mkdir(parents=True, exist_ok=True)
    with directory_lock(directory):
        frozen_manifest(directory, {"command": "adapt", "train_sha256": digest(train_path.read_bytes()),
                                    "n": len(rows), "config": asdict(config), "prices": prices,
                                    "runtime": runtime_fingerprint()})
        if (directory / "summary.json").exists():
            state = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
            if state.get("status") == "complete":
                load_artifact(directory)
                print(canonical({"status": "complete", "adaptation_reused": True}), flush=True)
                return state
        write_json(directory / "summary.json", {"status": "running", "artifact_version": ARTIFACT_VERSION})
        runner = Runner(directory, config)
        for name in ("baseline", "analysis", "synthesis", "strip", "explanation"):
            atomic_text(directory / f"{name}_prompt.md", prompt_text(f"{name}.md") + "\n")
        definitions = prompt_text("definitions.md")
        atomic_text(directory / "definitions.md", definitions + "\n")
        atomic_text(directory / "seed_prompt.md", prompt_text("seed.md") + "\n")
        baseline = await runner.stage("baseline", [(row["item_id"], baseline_prompt, judge_payload(row)) for row in rows],
                                      Judgment, config.inference, prices["baseline"])
        save_predictions(directory / "baseline_predictions.csv", rows, {item_id: value.label for item_id, value in baseline.items()})
        analysis_jobs = []
        for row in rows:
            label = baseline[row["item_id"]].label
            payload = {"baseline_prompt": baseline_prompt, "item_id": row["item_id"], "topic": row["topic"],
                       **judge_payload(row), "gold_label": row["label"], "baseline_label": label,
                       "baseline_correct": label == row["label"]}
            analysis_jobs.append((row["item_id"], prompt_text("analysis.md"), payload))
        audits = await runner.stage("analysis", analysis_jobs, CaseAnalysis, config.analysis, prices["analysis"])
        analyses = [{"item_id": row["item_id"], "topic": row["topic"], "gold_label": row["label"],
                     "baseline_label": baseline[row["item_id"]].label,
                     "baseline_correct": baseline[row["item_id"]].label == row["label"],
                     **audits[row["item_id"]].model_dump()} for row in rows]
        atomic_text(directory / "analyses.jsonl", "".join(canonical(row) + "\n" for row in analyses))
        synthesis_input = synthesis_payload(analyses, prompt_text("seed.md"), definitions)
        write_json(directory / "synthesis_input.json", synthesis_input)
        synthesized = await runner.stage("synthesis", [("synthesis", prompt_text("synthesis.md"), synthesis_input)],
                                         LinkedArtifact, config.synthesis, prices["synthesis"])
        linked = synthesized["synthesis"]
        write_json(directory / "linked_artifact.json", linked.model_dump())
        links = link_report(linked, analyses)
        write_json(directory / "link_report.json", links)
        strip_input = strip_payload(linked)
        write_json(directory / "strip_input.json", strip_input)

        def validate_stripped(value):
            checks = strip_report(linked, value, analyses, definitions)
            write_json(directory / "strip_report.json", checks)
            if checks["evidence_markers_left"] or checks["training_cases_left"]:
                write_json(directory / "strip_failure.json", checks)
                raise SolutionError("Inference prompt still carries training evidence; resume to retry stripping")

        stripped = await runner.stage("strip", [("strip", prompt_text("strip.md"), strip_input)],
                                      PromptArtifact, config.synthesis, prices["strip"],
                                      validate=validate_stripped)
        artifact = stripped["strip"]
        (directory / "strip_failure.json").unlink(missing_ok=True)
        write_json(directory / "artifact.json", artifact.model_dump())
        write_json(directory / "routed_prompts.json", compiled_prompts(artifact))
        explanation_input = {"artifact": artifact.model_dump(), "fixed_definitions_appended_verbatim": definitions,
                             "n_train_cases": len(rows), "train_analyses": analyses}
        write_json(directory / "explanation_input.json", explanation_input)
        explained = await runner.stage("explanation", [("explanation", prompt_text("explanation.md"), explanation_input)],
                                       Explanation, config.explanation, prices["explanation"],
                                       validate=lambda value: validate_explanation(value, artifact, rows))
        report = explained["explanation"]
        explanation = render_explanation(report, artifact, rows, analyses)
        write_json(directory / "explanation.json", report.model_dump())
        atomic_text(directory / "explanation.md", explanation)
        names = ["artifact.json", "routed_prompts.json", "explanation.json", "explanation.md", "manifest.json",
                 "analyses.jsonl", "baseline_predictions.csv", "budget.jsonl", "linked_artifact.json", "definitions.md",
                 "link_report.json", "strip_report.json", "seed_prompt.md", "synthesis_prompt.md", "strip_prompt.md"]
        summary = {"status": "complete", "artifact_version": ARTIFACT_VERSION, "train_examples": len(rows),
                   "inference_settings": asdict(config.inference), "stages": usage_summary(directory),
                   "accounted_usd": runner.budget.used(), "budget_usd": config.budget_usd,
                   "files_sha256": {name: digest((directory / name).read_bytes()) for name in names},
                   "completed_at": utc_now()}
        write_json(directory / "summary.json", summary)
        (directory / "failure.json").unlink(missing_ok=True)
        print(canonical(summary), flush=True)
        return summary
