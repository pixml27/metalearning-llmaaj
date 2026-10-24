"""Exp1 schemas, frozen inputs, stage settings, and reusable structured calls."""
from __future__ import annotations

from dataclasses import asdict, replace
from importlib.metadata import version
from pathlib import Path
from typing import Literal
import asyncio
import csv
import fcntl
import json
import os
import sys
import time

REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))
from common import TOPICS, load_dataset, load_labels
from evaluate import score
from prepare_inputs import model_rows
from experiments.llm import (Budget, JudgmentModel, Settings, append_jsonl, digest,
                             prices_for, request_ceiling, utc_now, write_json)
from pydantic import BaseModel, ConfigDict, Field
from pydantic_ai import Agent, NativeOutput
from pydantic_ai.usage import UsageLimits


class CaseAnalysis(BaseModel):
    model_config = ConfigDict(extra="forbid")
    comment_a: str = Field(min_length=1, max_length=1800)
    comment_b: str = Field(min_length=1, max_length=1500)


class TopicAdvice(BaseModel):
    model_config = ConfigDict(extra="forbid")
    credit: str = Field(min_length=1, max_length=2400)
    financial_position: str = Field(min_length=1, max_length=2400)
    savings: str = Field(min_length=1, max_length=2400)
    spending: str = Field(min_length=1, max_length=2400)


class PromptArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid")
    base_prompt: str = Field(min_length=1, max_length=4000)
    topic_advice: TopicAdvice


def stage_settings() -> dict[str, Settings]:
    base = Settings.from_env(HERE / ".env")
    if base.model != "gemini/gemini-flash-latest":
        raise ValueError("Exp1 inference must use MODEL=gemini/gemini-flash-latest")
    return {
        "analysis": replace(base, model=os.getenv("ANALYSIS_MODEL", base.model),
                            max_output_tokens=int(os.getenv("ANALYSIS_MAX_OUTPUT_TOKENS", "3072"))),
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


def analysis_inputs(train_path: Path, baseline_dir: Path, language: str) -> tuple[list, list, str, dict]:
    gold = load_dataset(train_path, require_labels=True)
    inputs = model_rows(gold, train=False, language=language)
    predictions = load_labels(baseline_dir / "predictions.csv")
    baseline_manifest = json.loads((baseline_dir / "manifest.json").read_text())
    prompt = (baseline_dir / "prompt.md").read_text().strip()
    if baseline_manifest["test_sha256"] != digest(train_path.read_bytes()):
        raise ValueError("Baseline predictions are not from this training dataset")
    if baseline_manifest["language"] != language or baseline_manifest["prompt_sha256"] != digest(prompt):
        raise ValueError("Baseline language/prompt does not match its manifest")
    if baseline_manifest.get("dataset_split") != "train":
        raise ValueError("Baseline source must be the train run, never a test run")
    report = score(gold, predictions)  # validates exact coverage and all topics
    if report != json.loads((baseline_dir / "score.json").read_text()):
        raise ValueError("Baseline score disagrees with saved predictions")
    pred = {r["item_id"]: r["label"] for r in predictions}
    labels = {r["item_id"]: r["label"] for r in gold}
    payloads = [{"baseline_prompt": prompt, "item_id": r["item_id"], "topic": r["topic"],
                 "context": r["context"], "conversation": r["conversation"],
                 "gold_label": labels[r["item_id"]], "baseline_label": pred[r["item_id"]],
                 "baseline_correct": labels[r["item_id"]] == pred[r["item_id"]]} for r in inputs]
    return gold, payloads, prompt, baseline_manifest


def compiled_prompts(artifact: PromptArtifact) -> dict[str, str]:
    return {t: artifact.base_prompt.strip() + f"\n\nTopic: {t}\nTopic-specific advice:\n"
            + getattr(artifact.topic_advice, t).strip() for t in TOPICS}


def load_artifact(directory: Path):
    summary = json.loads((directory / "summary.json").read_text())
    if summary["status"] != "complete" or summary["completed"] != summary["total"]:
        raise ValueError("Adaptation is not complete")
    if summary["artifact_sha256"] != digest((directory / "artifact.json").read_bytes()):
        raise ValueError("Adaptation artifact was modified")
    if summary["routed_prompts_sha256"] != digest((directory / "routed_prompts.json").read_bytes()):
        raise ValueError("Routed prompts were modified")
    artifact = PromptArtifact.model_validate_json((directory / "artifact.json").read_text())
    prompts = json.loads((directory / "routed_prompts.json").read_text())
    if prompts != compiled_prompts(artifact):
        raise ValueError("Compiled prompts disagree with artifact")
    return artifact, prompts, summary


def synthesis_payload(analyses: list[dict], baseline_prompt: str) -> dict:
    return {"task": "Infer the expert's conversation-level good/bad rubric from labeled TRAIN cases; improve the baseline and route by supplied topic.",
            "baseline_prompt": baseline_prompt, "n_train_cases": len(analyses), "train_analyses": analyses}


def estimate(prompt: str, payload: dict, schema, settings: Settings, prices: dict) -> float:
    return request_ceiling([{"role": "system", "content": prompt},
                            {"role": "user", "content": json.dumps(payload, ensure_ascii=False, separators=(",", ":"))}],
                           schema.model_json_schema(), settings, prices)


def structured_agent(prompt, schema, settings, budget, calls, run_id):
    model = JudgmentModel(settings, budget, calls, run_id, prices_for(settings.model))
    agent = Agent(model=model, system_prompt=prompt, output_type=NativeOutput(schema), retries=0,
                  name="exp1_" + schema.__name__)
    return agent, model


async def structured_call(prompt, payload, schema, settings, budget, calls, run_id, item_id):
    agent, model = structured_agent(prompt, schema, settings, budget, calls, run_id)
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


class StartGate:
    def __init__(self, interval):
        self.interval, self.last = interval, 0.0
        self.lock = asyncio.Lock()

    async def wait(self):
        async with self.lock:
            await asyncio.sleep(max(0, self.interval - (time.monotonic() - self.last)))
            self.last = time.monotonic()


def write_analyses_csv(path: Path, rows: list[dict]):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["item_id", "topic", "gold_label", "baseline_label", "baseline_correct", "comment_a", "comment_b"])
        writer.writeheader()
        writer.writerows(rows)


def cost_summary(paths: list[Path]) -> dict:
    calls = [call for path in paths for call in read_jsonl(path)]
    return {"api_responses": len(calls), "cost_usd_estimated": sum(c["cost_usd_estimated"] for c in calls),
            "response_models": sorted({c["response_model"] for c in calls}),
            "input_tokens": sum(c["usage"].get("prompt_tokens", 0) for c in calls),
            "output_tokens": sum(c["usage"].get("completion_tokens", 0) for c in calls)}
