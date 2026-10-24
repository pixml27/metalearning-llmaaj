"""Exp3 schemas: seed bundle, linked synthesis, link-free inference prompt, offline checks."""
from __future__ import annotations

from dataclasses import replace
from difflib import SequenceMatcher
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

SECTIONS = ("TASK", "CRITERIA", "PRIORITY", "BOUNDARIES")
EVIDENCE = re.compile(r"\[\[\s*ev:(.*?)\]\]", re.DOTALL)
CRITERION = re.compile(r"\bC\d+\b")
HEX_TOKEN = re.compile(r"\b[0-9a-f]{8,32}\b")
DEFINITION_CODE = re.compile(r"\bD(?:1[0-4]|[1-9])\b")


class TopicAdvice(BaseModel):
    model_config = ConfigDict(extra="forbid")
    credit: str = Field(min_length=1, max_length=5000)
    financial_position: str = Field(min_length=1, max_length=5000)
    savings: str = Field(min_length=1, max_length=5000)
    spending: str = Field(min_length=1, max_length=5000)


class RubricSelfCheck(BaseModel):
    """Where each rubric row is satisfied; documentation only, never sent to the judge."""
    model_config = ConfigDict(extra="forbid")
    clear_criteria: str = Field(min_length=1, max_length=900)
    clear_priorities: str = Field(min_length=1, max_length=900)
    links_to_training_examples: str = Field(min_length=1, max_length=900)
    clear_boundaries: str = Field(min_length=1, max_length=900)
    consistency_with_implementation: str = Field(min_length=1, max_length=900)


class LinkedArtifact(BaseModel):
    """First Argon answer: the documented prompt, every rule carrying its training evidence."""
    model_config = ConfigDict(extra="forbid")
    base_prompt_with_links: str = Field(min_length=1, max_length=24000)
    topic_advice_with_links: TopicAdvice
    rubric_selfcheck: RubricSelfCheck


class PromptArtifact(BaseModel):
    """Second Argon answer: the same prompt with the evidence spans removed, used at inference."""
    model_config = ConfigDict(extra="forbid")
    base_prompt: str = Field(min_length=1, max_length=24000)
    topic_advice: TopicAdvice


def stage_settings() -> dict[str, Settings]:
    base = Settings.from_env(HERE / ".env")
    if base.model != "gemini/gemini-flash-latest":
        raise ValueError("Exp3 inference must use MODEL=gemini/gemini-flash-latest")
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


def parse_bundle(text: str) -> tuple[str, dict[str, str]]:
    """Split a prompt file into the topic-independent base and one advice per topic."""
    parts = re.split(r"(?m)^##\s+(\w+)\s*$", text.strip())
    base, rest = parts[0].strip(), parts[1:]
    advice = {rest[i]: rest[i + 1].strip() for i in range(0, len(rest), 2)}
    if set(advice) != set(TOPICS) or not base:
        raise ValueError("Seed prompt needs a base section and exactly the four topic sections")
    return base, advice


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


def bundle_text(base: str, advice: dict[str, str]) -> str:
    return base.strip() + "\n\n" + "\n\n".join(f"## {t}\n{advice[t].strip()}" for t in TOPICS) + "\n"


def load_artifact(directory: Path):
    summary = json.loads((directory / "summary.json").read_text())
    if summary["status"] != "complete":
        raise ValueError("Adaptation is not complete")
    if summary["artifact_sha256"] != digest((directory / "artifact.json").read_bytes()):
        raise ValueError("Adaptation artifact was modified")
    if summary["linked_artifact_sha256"] != digest((directory / "linked_artifact.json").read_bytes()):
        raise ValueError("Documented artifact was modified")
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


def synthesis_payload(analyses: list[dict], seed_prompt: str, definitions: str) -> dict:
    return {"task": "Improve the seed judge prompt from labeled TRAIN cases; attach training evidence to every criterion and topic bullet.",
            "seed_prompt": seed_prompt, "fixed_definitions_appended_verbatim": definitions,
            "n_train_cases": len(analyses), "train_analyses": analyses}


def strip_payload(linked: LinkedArtifact) -> dict:
    return {"task": "Remove the evidence spans only; keep every rule byte-identical.",
            "base_prompt_with_links": linked.base_prompt_with_links,
            "topic_advice_with_links": linked.topic_advice_with_links.model_dump()}


def evidence_links(text: str) -> list[dict]:
    links = []
    for raw in EVIDENCE.findall(text):
        fields = [part.strip() for part in raw.split("|")]
        links.append({"case": fields[0].lower() if fields else "", "label": fields[1].lower() if len(fields) > 1 else "",
                      "text": fields[2] if len(fields) > 2 else "", "criterion": fields[3] if len(fields) > 3 else "",
                      "fields": len(fields)})
    return links


def link_report(linked: LinkedArtifact, rows: list[dict]) -> dict:
    """Check every link against the training table: real case, real gold label, placed on a criterion."""
    labels = {r["item_id"]: r["gold_label"] for r in rows}
    texts = {"base": linked.base_prompt_with_links, **linked.topic_advice_with_links.model_dump()}
    links = {part: evidence_links(text) for part, text in texts.items()}
    resolved, unknown, wrong_label, malformed = {}, [], [], 0
    for part, found in links.items():
        for link in found:
            if link["fields"] != 4 or not link["case"]:
                malformed += 1
                continue
            matches = [i for i in labels if i.startswith(link["case"])]
            if len(matches) != 1:
                unknown.append(link["case"])
                continue
            resolved.setdefault(part, []).append(matches[0])
            if link["label"] not in {"good", "bad"} or link["label"] != labels[matches[0]]:
                wrong_label.append({"case": link["case"], "claimed": link["label"], "gold": labels[matches[0]]})
    base = linked.base_prompt_with_links
    criteria = sorted(set(CRITERION.findall(base)), key=lambda c: int(c[1:]))
    covered = {link["criterion"] for link in links["base"] if link["criterion"] in criteria}
    linked_cases = sorted({case for cases in resolved.values() for case in cases})
    return {"links_total": sum(len(v) for v in links.values()),
            "links_per_part": {part: len(found) for part, found in links.items()},
            "distinct_cases_cited": len(linked_cases),
            "cases_with_disagreeing_baseline": sum(1 for r in rows if r["item_id"] in linked_cases and not r["baseline_correct"]),
            "criteria": criteria, "criteria_without_link": [c for c in criteria if c not in covered],
            "malformed_links": malformed, "unknown_cases": sorted(set(unknown)), "wrong_labels": wrong_label}


def normalized(text: str) -> list[str]:
    return re.sub(r"[^0-9a-z ]+", " ", text.lower()).split()


def strip_report(linked: LinkedArtifact, stripped: PromptArtifact, rows: list[dict], definitions: str) -> dict:
    """Verify the second answer only removed evidence and left the rules intact."""
    known = {r["item_id"] for r in rows}
    before = {"base": linked.base_prompt_with_links, **linked.topic_advice_with_links.model_dump()}
    after = {"base": stripped.base_prompt, **stripped.topic_advice.model_dump()}
    leaked, similarity = {}, {}
    for part, text in after.items():
        leaked[part] = sorted({token for token in HEX_TOKEN.findall(text.lower())
                               if any(i.startswith(token) for i in known)})
        mechanical = normalized(EVIDENCE.sub(" ", before[part]))
        similarity[part] = round(SequenceMatcher(None, mechanical, normalized(text)).ratio(), 4)
    return {"evidence_markers_left": sum(len(evidence_links(text)) for text in after.values()),
            "training_cases_left": {part: value for part, value in leaked.items() if value},
            "similarity_to_mechanical_strip": similarity,
            "criteria_preserved": sorted(set(CRITERION.findall(before["base"]))) == sorted(set(CRITERION.findall(after["base"]))),
            "sections_present": {name: bool(re.search(rf"(?m)^\W*{name}\b", after["base"])) for name in SECTIONS},
            "definition_codes": sorted(set(DEFINITION_CODE.findall(after["base"])), key=lambda c: int(c[1:])),
            "definition_lines_restated_verbatim": [line.split(maxsplit=1)[0] for line in
                                                   (raw.strip() for raw in definitions.splitlines())
                                                   if DEFINITION_CODE.match(line) and line in after["base"]],
            "words": {part: len(text.split()) for part, text in after.items()},
            "words_with_links": {part: len(text.split()) for part, text in before.items()}}


def estimate(prompt: str, payload: dict, schema, settings: Settings, prices: dict) -> float:
    return request_ceiling([{"role": "system", "content": prompt},
                            {"role": "user", "content": json.dumps(payload, ensure_ascii=False, separators=(",", ":"))}],
                           schema.model_json_schema(), settings, prices)


async def structured_call(prompt, payload, schema, settings, budget, calls, run_id, item_id):
    model = JudgmentModel(settings, budget, calls, run_id, prices_for(settings.model))
    agent = Agent(model=model, system_prompt=prompt, output_type=NativeOutput(schema), retries=0,
                  name="exp3_" + schema.__name__)
    model.item_id = item_id
    result = await agent.run(json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
                             usage_limits=UsageLimits(request_limit=1))
    return result.output


def cached_outputs(calls_path: Path, schema, item_id: str | None = None) -> dict:
    valid = {}
    for call in read_jsonl(calls_path):
        if call.get("finish_reason") != "stop" or not call.get("raw_output"):
            continue
        if item_id is not None and call["item_id"] != item_id:
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
