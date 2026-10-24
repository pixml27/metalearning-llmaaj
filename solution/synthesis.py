"""Exp3 synthesis schemas and offline checks; independent of experiment artifacts."""
from difflib import SequenceMatcher
import re

from pydantic import Field
from .common import PromptArtifact, StrictModel, TopicAdvice

SECTIONS = ("TASK", "CRITERIA", "PRIORITY", "BOUNDARIES")
EVIDENCE = re.compile(r"\[\[\s*ev:(.*?)\]\]", re.DOTALL)
CRITERION = re.compile(r"\bC\d+\b")
DEFINITION_CODE = re.compile(r"\bD(?:1[0-4]|[1-9])\b")


class RubricSelfCheck(StrictModel):
    clear_criteria: str = Field(min_length=1, max_length=900)
    clear_priorities: str = Field(min_length=1, max_length=900)
    links_to_training_examples: str = Field(min_length=1, max_length=900)
    clear_boundaries: str = Field(min_length=1, max_length=900)
    consistency_with_implementation: str = Field(min_length=1, max_length=900)


class LinkedArtifact(StrictModel):
    base_prompt_with_links: str = Field(min_length=1, max_length=24000)
    topic_advice_with_links: TopicAdvice
    rubric_selfcheck: RubricSelfCheck


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
    known = {r["item_id"].lower() for r in rows}
    before = {"base": linked.base_prompt_with_links, **linked.topic_advice_with_links.model_dump()}
    after = {"base": stripped.base_prompt, **stripped.topic_advice.model_dump()}
    candidates = known | {item_id[:8] for item_id in known if len(item_id) >= 8}
    leaked, similarity = {}, {}
    for part, text in after.items():
        leaked[part] = sorted(token for token in candidates
                              if re.search(rf"(?<!\w){re.escape(token)}(?!\w)", text.lower()))
        mechanical = normalized(EVIDENCE.sub(" ", before[part]))
        similarity[part] = round(SequenceMatcher(None, mechanical, normalized(text)).ratio(), 4)
    return {"evidence_markers_left": sum(len(re.findall(r"\[\[\s*ev:", text, re.IGNORECASE)) for text in after.values()),
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
