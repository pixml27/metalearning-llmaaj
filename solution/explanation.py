"""Ground the jury explanation in exact deployed prompt text and real train cases."""
from typing import Literal
import json

from pydantic import Field
from .common import SolutionError
from .common import StrictModel, TOPICS, digest, prompt_text

Scope = Literal["shared", "credit", "financial_position", "savings", "spending"]


class Criterion(StrictModel):
    name: str = Field(min_length=1, max_length=200)
    condition: str = Field(min_length=1, max_length=1200)
    priority: str = Field(min_length=1, max_length=800)
    prompt_scope: Scope
    prompt_quote: str = Field(min_length=12, max_length=2400)
    example_ids: list[str] = Field(min_length=1, max_length=6)


class Boundary(StrictModel):
    description: str = Field(min_length=1, max_length=1600)
    example_ids: list[str] = Field(min_length=1, max_length=6)


class Explanation(StrictModel):
    decision_policy: str = Field(min_length=1, max_length=2000)
    criteria: list[Criterion] = Field(min_length=1, max_length=16)
    boundaries: list[Boundary] = Field(min_length=1, max_length=8)


def validate_explanation(report, artifact, rows):
    topics = {row["item_id"]: row["topic"] for row in rows}
    for criterion in report.criteria:
        source = artifact.base_prompt if criterion.prompt_scope == "shared" else getattr(artifact.topic_advice, criterion.prompt_scope)
        if criterion.prompt_quote not in source:
            raise SolutionError("Explanation must quote exact text from its declared prompt scope")
        if any(item_id not in topics or (criterion.prompt_scope != "shared" and topics[item_id] != criterion.prompt_scope)
               for item_id in criterion.example_ids):
            raise SolutionError("Explanation criterion cites unknown or mismatched training IDs")
    for boundary in report.boundaries:
        if any(item_id not in topics for item_id in boundary.example_ids):
            raise SolutionError("Explanation boundary cites unknown training IDs")


def render_explanation(report, artifact, rows, analyses):
    validate_explanation(report, artifact, rows)
    selected = {item_id for value in [*report.criteria, *report.boundaries] for item_id in value.example_ids}
    analysis_by_id = {row["item_id"]: row for row in analyses}
    parts = ["# Learned conversation judge", "Automatically generated from this adaptation run's labeled training set.",
             "The analyses reconstruct possible reasons for the expert labels; they are not official expert rationales.",
             "## Decision and priorities", report.decision_policy,
             "## Criteria used for prediction"]
    for index, criterion in enumerate(report.criteria, 1):
        parts.extend([f"### {index}. {criterion.name}", criterion.condition,
                      f"Priority: {criterion.priority}", f"Prompt scope: `{criterion.prompt_scope}`.",
                      "Exact deployed prompt text:", "```text\n" + criterion.prompt_quote + "\n```",
                      "Training evidence: " + ", ".join(f"`{item_id}`" for item_id in criterion.example_ids) + "."])
    parts.append("## Boundaries and uncertainty")
    for boundary in report.boundaries:
        parts.extend([boundary.description, "Training evidence: " + ", ".join(f"`{item_id}`" for item_id in boundary.example_ids) + "."])
    parts.extend(["## Referenced training examples",
                  "The following context, complete conversations and expert labels are copied from the current train input."])
    for row in rows:
        if row["item_id"] not in selected:
            continue
        analysis = analysis_by_id[row["item_id"]]
        parts.extend([f"### {row['item_id']} — {row['topic']}",
                      f"Expert label: **{row['label']}**; baseline label: **{analysis['baseline_label']}**.",
                      "```json\n" + json.dumps({"context": row["context"], "conversation": row["conversation"]},
                                               ensure_ascii=False, indent=2) + "\n```",
                      "A: " + analysis["comment_a"], "B: " + analysis["comment_b"]])
    parts.extend(["## Connection to the implementation",
                  "eval loads artifact.json, compiles the base prompt plus fixed definitions and exactly one advice selected by the supplied topic, "
                  "and sends only context and the complete conversation to the configured judge model. "
                  "Its structured response contains only good/bad. This explanation does not change the prediction prompts.",
                  "## Exact prediction prompts", "### Shared base prompt", "```text\n" + artifact.base_prompt + "\n```"])
    parts.extend(["### Fixed definitions", "```text\n" + prompt_text("definitions.md") + "\n```"])
    for topic in TOPICS:
        parts.extend([f"### {topic}", "```text\n" + getattr(artifact.topic_advice, topic) + "\n```"])
    parts.append("Canonical artifact SHA256: `" + digest(artifact.model_dump_json()) + "`.")
    return "\n\n".join(parts) + "\n"
