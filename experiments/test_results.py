"""Offline regression checks for the experiment comparison table."""
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import json
import unittest

from experiments.update_results import update_results, dataset_split


class ResultsTests(unittest.TestCase):
    def test_legacy_and_explicit_dataset_split(self):
        self.assertEqual(dataset_split({"dataset_path": "/old/path/public_test.csv"}), "test")
        self.assertEqual(dataset_split({"dataset_path": "/old/path/public_train.csv"}), "train")
        self.assertEqual(dataset_split({"dataset_path": "custom.csv", "dataset_split": "train"}), "train")

    def test_completed_runs_are_included_copies_deduplicated_and_partial_runs_excluded(self):
        source = Path(__file__).resolve().parent / "baseline/results/20261024-public-original"
        original = {name: json.loads((source / f"{name}.json").read_text())
                    for name in ["manifest", "summary", "score"]}
        with TemporaryDirectory() as temporary:
            root = Path(temporary)

            def save(experiment, storage, name, data):
                directory = root / experiment / storage / name
                directory.mkdir(parents=True)
                for kind, value in data.items():
                    (directory / f"{kind}.json").write_text(json.dumps(value))
                (root / experiment / "README.md").write_text(f"**Идея:** Description for {experiment}.\n")

            save("baseline", "results", "first", original)
            save("baseline", "runs", "copy", original)
            gemma = deepcopy(original)
            gemma["manifest"].update(experiment="baseline_gemma", config_sha256="gemma")
            gemma["manifest"]["settings"]["model"] = "gemini/gemma-4-26b-a4b-it"
            gemma["summary"].update(response_models=["gemma-4-26b-a4b-it"], cost_usd_estimated_this_run=0)
            save("baseline_gemma", "results", "second", gemma)
            partial = deepcopy(gemma)
            partial["manifest"]["config_sha256"] = "partial"
            partial["summary"].update(status="partial", completed=1)
            save("baseline_gemma", "runs", "unfinished", partial)
            train = deepcopy(original)
            train["manifest"].update(dataset_split="train", config_sha256="train", dataset_path="public_train.csv")
            save("baseline", "results", "train", train)
            destination = update_results(root)
            text = destination.read_text()
            self.assertEqual(text.count("| [baseline]"), 2)
            self.assertEqual(text.count("| [baseline_gemma]"), 1)
            self.assertNotIn("unfinished", text)
            self.assertNotIn("/copy/", text)
            self.assertIn("0.797250", text)
            self.assertIn("gemini/gemma-4-26b-a4b-it", text)
            self.assertIn("| 0.000000 |", text)
            self.assertIn("| train |", text)
            self.assertIn("| test |", text)
            update_results(root)
            self.assertEqual(destination.read_text(), text)

    def test_adaptive_run_reports_training_and_test_costs(self):
        source = Path(__file__).resolve().parent / "baseline/results/20261024-public-original"
        values = {name: json.loads((source / f"{name}.json").read_text())
                  for name in ["manifest", "summary", "score"]}
        values["manifest"].update(experiment="exp1", artifact_sha256="artifact-hash")
        values["summary"].update(cost_usd_estimated_this_run=1.0,
            adaptation={"cost_usd_estimated": 0.8, "synthesis": {"response_models": ["gemini-4-argon"]}},
            inference={"cost_usd_estimated": 0.2})
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory = root / "exp1/results/first"
            directory.mkdir(parents=True)
            for name, value in values.items():
                (directory / f"{name}.json").write_text(json.dumps(value))
            (root / "exp1/README.md").write_text("**Идея:** Adaptive prompt.\n")
            text = update_results(root).read_text()
            self.assertIn("| 1.000000 |", text)
            self.assertIn("$0.800000", text)
            self.assertIn("$0.200000", text)
            self.assertIn("gemini-4-argon", text)
            self.assertIn("artifact-hash", text)


if __name__ == "__main__":
    unittest.main()
