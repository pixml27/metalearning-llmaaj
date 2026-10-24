"""Exp2 schemas, reused train analyses, fixed definitions, and the synthesis call."""
from __future__ import annotations

from dataclasses import replace
from importlib.metadata import version
from pathlib import Path
import csv
import fcntl
import json
import os
import re
import sys

REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))
from common import TOPICS, load_dataset, load_labels
from evaluate import score
from prepare_inputs import model_rows
from experiments.llm import (Budget, JudgmentModel, Settings, digest, prices_for, request_ceiling,
                             utc_now, write_json)
from pydantic import BaseModel, ConfigDict, Field
from pydantic_ai import Agent, NativeOutput
from pydantic_ai.usage import UsageLimits

SECTIONS = ("TASK", "CRITERIA", "PRIORITY", "BOUNDARIES", "EXAMPLES")
ID_PREFIX = re.compile(r"\b[0-9a-f]{8,32}\b")
DEFINITION_CODE = re.compile(r"\bD(?:1[0-5]|[1-9])\b")


class TopicAdvice(BaseModel):
    model_config = ConfigDict(extra="forbid")
    credit: str = Field(min_length=1, max_length=3000)
    financial_position: str = Field(min_length=1, max_length=3000)
    savings: str = Field(min_length=1, max_length=3000)
    spending: str = Field(min_length=1, max_length=3000)


class RubricSelfCheck(BaseModel):
    """Where each rubric row is satisfied; documentation only, never sent to the judge."""
    model_config = ConfigDict(extra="forbid")
    clear_criteria: str = Field(min_length=1, max_length=900)
    clear_priorities: str = Field(min_length=1, max_length=900)
    links_to_training_examples: str = Field(min_length=1, max_length=900)
    clear_boundaries: str = Field(min_length=1, max_length=900)
    consistency_with_implementation: str = Field(min_length=1, max_length=900)


class PromptArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid")
    base_prompt: str = Field(min_length=1, max_length=14000)
    topic_advice: TopicAdvice
    rubric_selfcheck: RubricSelfCheck


def stage_settings() -> dict[str, Settings]:
    base = Settings.from_env(HERE / ".env")
    if base.model != "gemini/gemini-flash-latest":
        raise ValueError("Exp2 inference must use MODEL=gemini/gemini-flash-latest")
    return {
        "synthesis": replace(base, model=os.getenv("SYNTHESIS_MODEL", "gemini/gemini-4-argon"),
                             max_output_tokens=int(os.getenv("SYNTHESIS_MAX_OUTPUT_TOKENS", "32768")),
                             timeout_seconds=float(os.getenv("SYNTHESIS_TIMEOUT_SECONDS", "900"))),
        "inference": base,
    }


def source_hashes() -> dict:
    paths = [*HERE.glob("*.py"), REPO / "experiments/llm.py", REPO / "experiments/inference.py", REPO / "src/common.py",
             REPO / "src/evaluate.py", REPO / "src/prepare_inputs.py"]
    return {str(p.relative_to(REPO)): digest(p.read_bytes()) for p in sorted(paths)}


def packages() -> dict:
    return {p: version(p) for p in ["litellm", "pydantic-ai-litellm", "pydantic-ai-slim", "pydantic", "python-dotenv"]}


def frozen_manifest(directory: Path, config: dict) -> dict:
    config_hash = digest(json.dumps(config, sort_keys=True))
    path = directory / "manifest.json"
    if path.exists():
        manifest = json.loads(path.read_text())
        if manifest["config_sha256"] != config_hash:
            raise ValueError("Inputs/code/settings changed; use a new output directory")
    else:
        if any(directory.iterdir()):
            raise ValueError("New output directory must be empty")
        manifest = {**config, "config_sha256": config_hash, "created_at": utc_now()}
        write_json(path, manifest)
    return manifest


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def reused_analyses(train_path: Path, baseline_dir: Path, analyses_dir: Path, language: str):
    """Validate the frozen exp1 train analyses and reuse them without new analysis calls."""
    gold = load_dataset(train_path, require_labels=True)
    predictions = load_labels(baseline_dir / "predictions.csv")
    baseline_manifest = json.loads((baseline_dir / "manifest.json").read_text())
    prompt = (baseline_dir / "prompt.md").read_text().strip()
    train_hash = digest(train_path.read_bytes())
    if baseline_manifest["test_sha256"] != train_hash:
        raise ValueError("Baseline predictions are not from this training dataset")
    if baseline_manifest["language"] != language or baseline_manifest["prompt_sha256"] != digest(prompt):
        raise ValueError("Baseline language/prompt does not match its manifest")
    if baseline_manifest.get("dataset_split") != "train":
        raise ValueError("Baseline source must be the train run, never a test run")
    report = score(gold, predictions)  # validates exact coverage and all topics
    if report != json.loads((baseline_dir / "score.json").read_text()):
        raise ValueError("Baseline score disagrees with saved predictions")
    summary = json.loads((analyses_dir / "summary.json").read_text())
    manifest = json.loads((analyses_dir / "manifest.json").read_text())
    if summary["status"] != "complete" or summary["completed"] != summary["total"]:
        raise ValueError("Reused analyses come from an incomplete adaptation")
    if manifest["dataset_split"] != "train" or manifest["language"] != language:
        raise ValueError("Reused analyses are not this train split/language")
    if manifest["train_sha256"] != train_hash:
        raise ValueError("Reused analyses were built from another train dataset")
    if manifest["baseline_prompt_sha256"] != digest(prompt):
        raise ValueError("Reused analyses were built on another baseline prompt")
    rows = read_jsonl(analyses_dir / "analyses.jsonl")
    labels = {r["item_id"]: r["label"] for r in gold}
    predicted = {r["item_id"]: r["label"] for r in predictions}
    if len(rows) != len(gold) or {r["item_id"] for r in rows} != set(labels):
        raise ValueError("Reused analyses do not cover exactly this train set")
    for row in rows:
        if row["gold_label"] != labels[row["item_id"]] or row["baseline_label"] != predicted[row["item_id"]]:
            raise ValueError("Reused analyses disagree with gold/baseline labels")
        if row["topic"] not in TOPICS or not row["comment_a"].strip() or not row["comment_b"].strip():
            raise ValueError("Reused analyses contain an unusable row")
    provenance = {"analyses_directory": str(analyses_dir.resolve()),
                  "analyses_sha256": digest((analyses_dir / "analyses.jsonl").read_bytes()),
                  "analyses_manifest_sha256": digest((analyses_dir / "manifest.json").read_bytes()),
                  "analysis_prompt_sha256": manifest["analysis_prompt_sha256"],
                  "analysis_settings": manifest["settings"]["analysis"],
                  "analysis": summary["analysis"]}
    return gold, rows, prompt, provenance


def compiled_prompts(artifact: PromptArtifact, definitions: str) -> dict[str, str]:
    return {t: artifact.base_prompt.strip() + "\n\n" + definitions.strip()
            + f"\n\nTopic: {t}\nTopic-specific advice:\n" + getattr(artifact.topic_advice, t).strip()
            for t in TOPICS}


def load_artifact(directory: Path):
    summary = json.loads((directory / "summary.json").read_text())
    if summary["status"] != "complete":
        raise ValueError("Adaptation is not complete")
    if summary["artifact_sha256"] != digest((directory / "artifact.json").read_bytes()):
        raise ValueError("Adaptation artifact was modified")
    if summary["routed_prompts_sha256"] != digest((directory / "routed_prompts.json").read_bytes()):
        raise ValueError("Routed prompts were modified")
    definitions = (directory / "definitions.md").read_text().strip()
    if summary["definitions_sha256"] != digest(definitions):
        raise ValueError("Definitions block was modified")
    artifact = PromptArtifact.model_validate_json((directory / "artifact.json").read_text())
    prompts = json.loads((directory / "routed_prompts.json").read_text())
    if prompts != compiled_prompts(artifact, definitions):
        raise ValueError("Compiled prompts disagree with artifact")
    return artifact, prompts, definitions, summary


def synthesis_payload(analyses: list[dict], baseline_prompt: str, definitions: str) -> dict:
    return {"task": "Infer the expert's conversation-level good/bad rubric from labeled TRAIN cases; rewrite the base prompt so its criteria, priorities, cited examples and boundaries are the rules actually applied at inference.",
            "baseline_prompt": baseline_prompt, "fixed_definitions_appended_verbatim": definitions,
            "n_train_cases": len(analyses), "train_analyses": analyses}


def structure_check(artifact: PromptArtifact, rows: list[dict], definitions: str) -> dict:
    """Record, without new API calls, how far the artifact follows the rubric structure."""
    text = artifact.base_prompt
    known = {r["item_id"] for r in rows}
    cited = {token for token in ID_PREFIX.findall(text.lower())}
    matched = {token for token in cited if any(i.startswith(token) for i in known)}
    advice = artifact.topic_advice.model_dump()
    numbered = [line.strip() for line in definitions.splitlines() if DEFINITION_CODE.match(line.strip())]
    return {"sections_present": {name: bool(re.search(rf"(?m)^\W*{name}\b", text)) for name in SECTIONS},
            "cited_case_prefixes": len(matched), "unmatched_id_like_tokens": sorted(cited - matched),
            "definition_codes_in_base_prompt": sorted(set(DEFINITION_CODE.findall(text)),
                                                      key=lambda c: int(c[1:])),
            "definition_lines_restated_verbatim": [line.split(maxsplit=1)[0] for line in numbered if line in text],
            "base_prompt_words": len(text.split()),
            "topic_advice_words": {t: len(v.split()) for t, v in advice.items()}}


def estimate(prompt: str, payload: dict, schema, settings: Settings, prices: dict) -> float:
    return request_ceiling([{"role": "system", "content": prompt},
                            {"role": "user", "content": json.dumps(payload, ensure_ascii=False, separators=(",", ":"))}],
                           schema.model_json_schema(), settings, prices)


async def structured_call(prompt, payload, schema, settings, budget, calls, run_id, item_id):
    model = JudgmentModel(settings, budget, calls, run_id, prices_for(settings.model))
    agent = Agent(model=model, system_prompt=prompt, output_type=NativeOutput(schema), retries=0,
                  name="exp2_" + schema.__name__)
    model.item_id = item_id
    result = await agent.run(json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
                             usage_limits=UsageLimits(request_limit=1))
    return result.output


def cached_outputs(calls_path: Path, schema) -> dict:
    valid = {}
    for call in read_jsonl(calls_path):
        if call.get("finish_reason") != "stop" or not call.get("raw_output"):
            continue
        try:
            valid[call["item_id"]] = schema.model_validate_json(call["raw_output"])
        except ValueError:
            pass
    return valid


def experiment_lock():
    stream = (HERE / ".run.lock").open("a")
    try:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BaseException:
        stream.close()
        raise
    return stream


def write_analyses_csv(path: Path, rows: list[dict]):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["item_id", "topic", "gold_label", "baseline_label",
                                                    "baseline_correct", "comment_a", "comment_b"])
        writer.writeheader()
        writer.writerows(rows)


def cost_summary(paths: list[Path]) -> dict:
    calls = [call for path in paths for call in read_jsonl(path)]
    return {"api_responses": len(calls), "cost_usd_estimated": sum(c["cost_usd_estimated"] for c in calls),
            "response_models": sorted({c["response_model"] for c in calls}),
            "input_tokens": sum(c["usage"].get("prompt_tokens", 0) for c in calls),
            "output_tokens": sum(c["usage"].get("completion_tokens", 0) for c in calls)}
