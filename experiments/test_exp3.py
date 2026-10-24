"""Offline checks for the seed bundle, evidence links and the link-free inference prompt."""
from pathlib import Path
from tempfile import TemporaryDirectory
import json
import re
import unittest

from pydantic import ValidationError
from experiments.exp3.common import (LinkedArtifact, PromptArtifact, SECTIONS, TOPICS, bundle_text,
                                     compiled_prompts, digest, evidence_links, link_report,
                                     load_artifact, parse_bundle, strip_payload, strip_report)

REPO = Path(__file__).resolve().parents[1]
DEFINITIONS = (REPO / "experiments/exp3/definitions.md").read_text().strip()
SEED = (REPO / "experiments/exp3/seed_prompt.md").read_text().strip()
CASES = ["04b71730fc4e3c4fdf19e0888108a8a5", "06c311a69434996e8d65c5ea27e85da3"]
ROWS = [{"item_id": CASES[0], "gold_label": "good", "baseline_correct": False},
        {"item_id": CASES[1], "gold_label": "bad", "baseline_correct": True}]
LINKED_BASE = ("TASK\nLabel the conversation.\n"
               "CRITERIA\nC1 reads D4 and D5. [[ev: 04b71730 | good | averaged complete months | C1]]\n"
               "C2 deducts on_hold (D8). [[ev: 06c311a6 | bad | pooled credit limit | C2]]\n"
               "PRIORITY\nC1 before C2.\n"
               "BOUNDARIES\nB1 currency default tolerated. [[ev: 04b71730 | good | MXN assumed | C1]] "
               "[[ev: 06c311a6 | bad | MXN plus scope error | C2]]")
STRIPPED_BASE = ("TASK\nLabel the conversation.\n"
                 "CRITERIA\nC1 reads D4 and D5.\nC2 deducts on_hold (D8).\n"
                 "PRIORITY\nC1 before C2.\nBOUNDARIES\nB1 currency default tolerated.")


def linked_artifact(base: str = LINKED_BASE) -> LinkedArtifact:
    return LinkedArtifact.model_validate({
        "base_prompt_with_links": base,
        "topic_advice_with_links": {t: f"Advice for {t}. [[ev: 04b71730 | good | complete months | C1]]" for t in TOPICS},
        "rubric_selfcheck": {"clear_criteria": "C1-C2.", "clear_priorities": "PRIORITY.",
                             "links_to_training_examples": "Inline spans.", "clear_boundaries": "B1.",
                             "consistency_with_implementation": "Compiled verbatim."}})


def stripped_artifact(base: str = STRIPPED_BASE) -> PromptArtifact:
    return PromptArtifact.model_validate({"base_prompt": base,
                                          "topic_advice": {t: f"Advice for {t}." for t in TOPICS}})


class Exp3Tests(unittest.TestCase):
    def test_seed_bundle_splits_into_base_and_four_topics(self):
        base, advice = parse_bundle(SEED)
        self.assertEqual(set(advice), set(TOPICS))
        for name in ("TASK", "CRITERIA", "PRIORITY"):
            self.assertIn(name, base)
        self.assertNotIn("##", base)
        self.assertIn("transfers_own is never spending", advice["spending"])
        self.assertEqual(parse_bundle(bundle_text(base, advice))[1], advice)
        with self.assertRaises(ValueError):
            parse_bundle("base only, no topics")

    def test_seed_drops_the_definition_that_nothing_referenced(self):
        codes = [line.split()[0] for line in DEFINITIONS.splitlines() if re.match(r"D\d", line)]
        self.assertEqual(codes, [f"D{i}" for i in range(1, 15)])
        self.assertIn("CURRENT_PARTIAL = the month containing today", DEFINITIONS)

    def test_links_are_checked_against_the_training_table(self):
        report = link_report(linked_artifact(), ROWS)
        self.assertEqual(report["links_total"], 8)
        self.assertEqual(report["criteria"], ["C1", "C2"])
        self.assertEqual(report["criteria_without_link"], [])
        self.assertEqual((report["malformed_links"], report["unknown_cases"], report["wrong_labels"]), (0, [], []))
        self.assertEqual(report["distinct_cases_cited"], 2)
        self.assertEqual(report["cases_with_disagreeing_baseline"], 1)
        broken = linked_artifact(LINKED_BASE.replace("[[ev: 06c311a6 | bad | pooled credit limit | C2]]",
                                                     "[[ev: 06c311a6 | good | pooled credit limit | C2]]")
                                .replace("[[ev: 04b71730 | good | MXN assumed | C1]]",
                                         "[[ev: deadbeef | good | invented | C1]] [[ev: 04b71730 | bad]]"))
        report = link_report(broken, ROWS)
        self.assertEqual(report["unknown_cases"], ["deadbeef"])
        self.assertEqual(report["malformed_links"], 1)
        self.assertEqual([w["case"] for w in report["wrong_labels"]], ["06c311a6"])

    def test_strip_report_detects_leftover_evidence_and_rule_drift(self):
        clean = strip_report(linked_artifact(), stripped_artifact(), ROWS, DEFINITIONS)
        self.assertEqual(clean["evidence_markers_left"], 0)
        self.assertEqual(clean["training_cases_left"], {})
        self.assertTrue(clean["criteria_preserved"])
        self.assertEqual(clean["sections_present"], {name: True for name in SECTIONS})
        self.assertEqual(clean["definition_codes"], ["D4", "D5", "D8"])
        self.assertGreater(clean["similarity_to_mechanical_strip"]["base"], 0.95)
        leaky = strip_report(linked_artifact(), stripped_artifact(
            STRIPPED_BASE + f"\nSee case {CASES[0]} and [[ev: 06c311a6 | bad | x | C2]]"), ROWS, DEFINITIONS)
        self.assertEqual(leaky["evidence_markers_left"], 1)
        self.assertEqual(list(leaky["training_cases_left"]), ["base"])
        drifted = strip_report(linked_artifact(), stripped_artifact(
            "TASK\nRewritten from scratch with different rules and no criteria."), ROWS, DEFINITIONS)
        self.assertFalse(drifted["criteria_preserved"])
        self.assertLess(drifted["similarity_to_mechanical_strip"]["base"], 0.5)

    def test_strip_payload_carries_only_the_linked_texts(self):
        payload = strip_payload(linked_artifact())
        self.assertEqual(payload["base_prompt_with_links"], LINKED_BASE)
        self.assertEqual(set(payload["topic_advice_with_links"]), set(TOPICS))
        self.assertNotIn("rubric_selfcheck", payload)
        self.assertEqual(len(evidence_links(payload["base_prompt_with_links"])), 4)

    def test_inference_loads_only_a_complete_unmodified_adaptation(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            artifact, linked = stripped_artifact(), linked_artifact()
            (root / "artifact.json").write_text(artifact.model_dump_json())
            (root / "linked_artifact.json").write_text(linked.model_dump_json())
            (root / "routed_prompts.json").write_text(json.dumps(compiled_prompts(artifact, DEFINITIONS)))
            (root / "definitions.md").write_text(DEFINITIONS + "\n")
            summary = {"status": "complete",
                       "artifact_sha256": digest((root / "artifact.json").read_bytes()),
                       "linked_artifact_sha256": digest((root / "linked_artifact.json").read_bytes()),
                       "routed_prompts_sha256": digest((root / "routed_prompts.json").read_bytes()),
                       "definitions_sha256": digest(DEFINITIONS)}
            (root / "summary.json").write_text(json.dumps(summary))
            prompts = load_artifact(root)[1]
            self.assertEqual(prompts, compiled_prompts(artifact, DEFINITIONS))
            for topic in TOPICS:
                self.assertIn(DEFINITIONS, prompts[topic])
                self.assertNotIn("[[ev:", prompts[topic])
            (root / "linked_artifact.json").write_text(linked.model_dump_json() + " ")
            with self.assertRaises(ValueError):
                load_artifact(root)

    def test_schemas_require_four_topics_and_reject_extra_fields(self):
        value = {"base_prompt": "Rules.", "topic_advice": {t: "Advice." for t in TOPICS}, "extra": 1}
        with self.assertRaises(ValidationError):
            PromptArtifact.model_validate(value)
        del value["extra"]
        del value["topic_advice"]["savings"]
        with self.assertRaises(ValidationError):
            PromptArtifact.model_validate(value)


if __name__ == "__main__":
    unittest.main()
