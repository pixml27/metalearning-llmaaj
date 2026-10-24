"""Offline checks for train-only adaptation and exact topic routing."""
from pathlib import Path
from tempfile import TemporaryDirectory
import json
import unittest

from pydantic import ValidationError
from experiments.exp1.common import (CaseAnalysis, PromptArtifact, TOPICS, analysis_inputs,
                                    cached_outputs, compiled_prompts, synthesis_payload,
                                    load_artifact, digest)


class Exp1Tests(unittest.TestCase):
    def test_saved_train_predictions_align_and_no_test_is_read(self):
        repo = Path(__file__).resolve().parents[1]
        gold, payloads, prompt, manifest = analysis_inputs(repo / "data/public_train.csv",
            repo / "experiments/baseline/results/20261024-public-train-original", "original")
        self.assertEqual(len(gold), 150)
        self.assertEqual(len(payloads), 150)
        self.assertEqual(sum(not p["baseline_correct"] for p in payloads), 40)
        self.assertTrue(all(p["gold_label"] in {"good", "bad"} for p in payloads))
        with self.assertRaises(ValueError):
            analysis_inputs(repo / "data/public_train.csv",
                repo / "experiments/baseline/results/20261024-public-original", "original")

    def test_exactly_four_topic_advices_and_only_selected_tip(self):
        value = {"base_prompt": "Shared rubric.", "topic_advice": {t: f"Unique advice for {t}." for t in TOPICS}}
        artifact = PromptArtifact.model_validate(value)
        routes = compiled_prompts(artifact)
        for topic in TOPICS:
            self.assertIn("Shared rubric.", routes[topic])
            for other in TOPICS:
                self.assertEqual(f"Unique advice for {other}." in routes[topic], topic == other)
        del value["topic_advice"]["credit"]
        with self.assertRaises(ValidationError):
            PromptArtifact.model_validate(value)

    def test_synthesis_gets_all_rows_and_baseline(self):
        rows = [{"item_id": str(i), "comment_a": "A", "comment_b": "B"} for i in range(150)]
        payload = synthesis_payload(rows, "baseline")
        self.assertEqual(payload["n_train_cases"], 150)
        self.assertEqual(payload["train_analyses"], rows)
        self.assertEqual(payload["baseline_prompt"], "baseline")

    def test_resume_recovers_valid_structured_outputs_only(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "calls.jsonl"
            entries = [
                {"item_id": "ok", "finish_reason": "stop", "raw_output": '{"comment_a":"a","comment_b":"b"}'},
                {"item_id": "invalid", "finish_reason": "stop", "raw_output": '{"label":"bad"}'},
                {"item_id": "truncated", "finish_reason": "length", "raw_output": '{"comment_a":"a","comment_b":"b"}'},
            ]
            path.write_text("".join(json.dumps(e) + "\n" for e in entries))
            cache = cached_outputs(path, CaseAnalysis)
            self.assertEqual(set(cache), {"ok"})

    def test_inference_rejects_modified_or_incomplete_adaptation(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = PromptArtifact.model_validate({"base_prompt": "Shared rubric.",
                "topic_advice": {t: f"Advice for {t}." for t in TOPICS}})
            (root / "artifact.json").write_text(artifact.model_dump_json())
            (root / "routed_prompts.json").write_text(json.dumps(compiled_prompts(artifact)))
            summary = {"status": "complete", "completed": 150, "total": 150,
                       "artifact_sha256": digest((root / "artifact.json").read_bytes()),
                       "routed_prompts_sha256": digest((root / "routed_prompts.json").read_bytes())}
            (root / "summary.json").write_text(json.dumps(summary))
            self.assertEqual(load_artifact(root)[1], compiled_prompts(artifact))
            (root / "artifact.json").write_text(artifact.model_dump_json() + " ")
            with self.assertRaises(ValueError):
                load_artifact(root)
            summary["status"] = "analysing"
            (root / "summary.json").write_text(json.dumps(summary))
            with self.assertRaises(ValueError):
                load_artifact(root)


if __name__ == "__main__":
    unittest.main()
