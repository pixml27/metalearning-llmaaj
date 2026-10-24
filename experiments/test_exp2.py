"""Offline checks for reused train analyses, fixed definitions and rubric structure reporting."""
from pathlib import Path
from tempfile import TemporaryDirectory
import json
import unittest

from pydantic import ValidationError
from experiments.exp2.common import (PromptArtifact, SECTIONS, TOPICS, compiled_prompts, digest,
                                     load_artifact, reused_analyses, structure_check,
                                     synthesis_payload)

REPO = Path(__file__).resolve().parents[1]
DEFINITIONS = (REPO / "experiments/exp2/definitions.md").read_text().strip()


def artifact_value(base_prompt: str = "Shared rubric.") -> dict:
    return {"base_prompt": base_prompt, "topic_advice": {t: f"Unique advice for {t}." for t in TOPICS},
            "rubric_selfcheck": {"clear_criteria": "C1-C3.", "clear_priorities": "PRIORITY.",
                                 "links_to_training_examples": "EXAMPLES.", "clear_boundaries": "BOUNDARIES.",
                                 "consistency_with_implementation": "Compiled verbatim."}}


class Exp2Tests(unittest.TestCase):
    def test_frozen_exp1_analyses_are_reused_and_mismatched_sources_rejected(self):
        gold, rows, prompt, provenance = reused_analyses(
            REPO / "data/public_train.csv",
            REPO / "experiments/baseline/results/20261024-public-train-original",
            REPO / "experiments/exp1/artifacts/20261024-public-original", "original")
        self.assertEqual((len(gold), len(rows)), (150, 150))
        self.assertEqual(sum(not r["baseline_correct"] for r in rows), 40)
        self.assertTrue(all(r["comment_a"] and r["comment_b"] for r in rows))
        self.assertEqual(provenance["analysis"]["api_responses"], 150)
        with self.assertRaises(ValueError):  # test-split baseline must never be the source
            reused_analyses(REPO / "data/public_train.csv",
                            REPO / "experiments/baseline/results/20261024-public-original",
                            REPO / "experiments/exp1/artifacts/20261024-public-original", "original")

    def test_definitions_are_appended_verbatim_with_only_the_selected_advice(self):
        artifact = PromptArtifact.model_validate(artifact_value())
        routes = compiled_prompts(artifact, DEFINITIONS)
        for topic in TOPICS:
            self.assertIn("Shared rubric.", routes[topic])
            self.assertIn(DEFINITIONS, routes[topic])
            self.assertIn("D15 no_interest_payment", routes[topic])
            for other in TOPICS:
                self.assertEqual(f"Unique advice for {other}." in routes[topic], topic == other)
        value = artifact_value()
        del value["topic_advice"]["credit"]
        with self.assertRaises(ValidationError):
            PromptArtifact.model_validate(value)
        del value["rubric_selfcheck"]
        with self.assertRaises(ValidationError):
            PromptArtifact.model_validate(value)

    def test_synthesis_payload_carries_every_row_definitions_and_baseline(self):
        rows = [{"item_id": str(i), "comment_a": "A", "comment_b": "B"} for i in range(150)]
        payload = synthesis_payload(rows, "baseline", DEFINITIONS)
        self.assertEqual(payload["n_train_cases"], 150)
        self.assertEqual(payload["train_analyses"], rows)
        self.assertEqual(payload["baseline_prompt"], "baseline")
        self.assertEqual(payload["fixed_definitions_appended_verbatim"], DEFINITIONS)

    def test_structure_check_reports_sections_cited_cases_and_invented_ids(self):
        rows = [{"item_id": "04b71730fc4e3c4fdf19e0888108a8a5"}, {"item_id": "06c311a69434996e8d65c5ea27e85da3"}]
        text = ("TASK\nLabel the conversation.\nCRITERIA\nC1 uses D5 and D6.\nPRIORITY\nC1 first.\n"
                "BOUNDARIES\n04b71730 is thin evidence.\nEXAMPLES\n06c311a6 bad, illustrates C1; deadbeef12 invented.")
        checks = structure_check(PromptArtifact.model_validate(artifact_value(text)), rows, DEFINITIONS)
        self.assertEqual(checks["sections_present"], {name: True for name in SECTIONS})
        self.assertEqual(checks["cited_case_prefixes"], 2)
        self.assertEqual(checks["unmatched_id_like_tokens"], ["deadbeef12"])
        self.assertEqual(checks["definition_codes_in_base_prompt"], ["D5", "D6"])
        self.assertEqual(checks["definition_lines_restated_verbatim"], [])
        bare = structure_check(PromptArtifact.model_validate(artifact_value(
            "Shared rubric.\n" + DEFINITIONS.splitlines()[3])), rows, DEFINITIONS)
        self.assertEqual(bare["sections_present"], {name: False for name in SECTIONS})
        self.assertEqual(bare["definition_lines_restated_verbatim"], ["D1"])

    def test_inference_rejects_modified_definitions_artifact_or_incomplete_adaptation(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = PromptArtifact.model_validate(artifact_value())
            (root / "artifact.json").write_text(artifact.model_dump_json())
            (root / "routed_prompts.json").write_text(json.dumps(compiled_prompts(artifact, DEFINITIONS)))
            (root / "definitions.md").write_text(DEFINITIONS + "\n")
            summary = {"status": "complete", "completed": 150, "total": 150,
                       "artifact_sha256": digest((root / "artifact.json").read_bytes()),
                       "routed_prompts_sha256": digest((root / "routed_prompts.json").read_bytes()),
                       "definitions_sha256": digest(DEFINITIONS)}
            (root / "summary.json").write_text(json.dumps(summary))
            self.assertEqual(load_artifact(root)[1], compiled_prompts(artifact, DEFINITIONS))
            (root / "definitions.md").write_text(DEFINITIONS.replace("D5 net_purchases(M) = purchases − refunds.",
                                                                     "D5 net_purchases(M) = purchases.") + "\n")
            with self.assertRaises(ValueError):
                load_artifact(root)
            (root / "definitions.md").write_text(DEFINITIONS + "\n")
            (root / "artifact.json").write_text(artifact.model_dump_json() + " ")
            with self.assertRaises(ValueError):
                load_artifact(root)
            (root / "artifact.json").write_text(artifact.model_dump_json())
            summary["status"] = "synthesising"
            (root / "summary.json").write_text(json.dumps(summary))
            with self.assertRaises(ValueError):
                load_artifact(root)


if __name__ == "__main__":
    unittest.main()
