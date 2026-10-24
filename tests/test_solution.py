"""Exercise the real pipeline and native schema bridge with a fake API, offline."""
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
import asyncio
import csv
import io
import json
import os
import shutil
import unittest

from solution import adapt, eval
from solution.__main__ import main
from solution.common import (CaseAnalysis, Judgment, PromptArtifact, TOPICS, compiled_prompts,
                             load_artifact, read_inputs, write_json)
from solution.config import Config, Settings
from solution.explanation import Explanation, validate_explanation
from solution.llm import Budget, BudgetExceeded, prices_for
from litellm import Usage


PRICES = {"input_usd_per_token_upper": 0.0000001, "output_usd_per_token_upper": 0.0000001}


def artifact_value():
    return {"base_prompt": "Evaluate all assistant responses. One supported violation makes the conversation bad.",
            "topic_advice": {topic: f"Apply only the {topic} criterion." for topic in TOPICS}}


def explanation_value():
    return {"decision_policy": "Any supported violation makes the entire conversation bad.",
            "criteria": [{"name": "Evaluate every response", "condition": "Check the complete conversation.",
                          "priority": "A supported violation overrides completeness.", "prompt_scope": "shared",
                          "prompt_quote": "Evaluate all assistant responses.", "example_ids": ["train-0"]}],
            "boundaries": [{"description": "The accepted example alone does not establish a universal threshold.",
                            "example_ids": ["train-0"]}]}


def dataset(prefix, labeled):
    rows = [{"item_id": f"{prefix}-{i}", "topic": topic, "context": {"balance": 10 + i},
             "conversation": [{"role": "user", "content": "What is the balance?"},
                              {"role": "assistant", "content": str(10 + i)},
                              {"role": "user", "content": "Check the previous answer too."}]}
            for i, topic in enumerate(TOPICS)]
    if labeled:
        for i, row in enumerate(rows):
            row["label"] = "good" if i % 2 == 0 else "bad"
    return rows


def write_rows(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


class FakeProvider:
    def __init__(self):
        self.requests = []
        self.fail_once_at = None
        self.invalid_judge = False
        self.invalid_explanation = False

    async def complete(self, **kwargs):
        self.requests.append(kwargs)
        if len(self.requests) == self.fail_once_at:
            raise RuntimeError("provider error with a secret that must never be printed")
        fields = kwargs["response_format"]["json_schema"]["schema"]["properties"]
        if "label" in fields:
            value = {"label": "invalid" if self.invalid_judge else "good"}
        elif "comment_a" in fields:
            value = {"comment_a": "Compare the observable claim with gold; baseline reasoning is unavailable.",
                     "comment_b": "Check only supported topic-specific conditions."}
        elif "base_prompt" in fields:
            value = artifact_value()
        else:
            value = explanation_value()
            if self.invalid_explanation:
                value["criteria"][0]["prompt_quote"] = "An invented rule absent from the artifact."
        return SimpleNamespace(usage=Usage(prompt_tokens=20, completion_tokens=30), model="resolved-test-model",
                               id=f"response-{len(self.requests)}", choices=[SimpleNamespace(
                                   message=SimpleNamespace(content=json.dumps(value)), finish_reason="stop")])


class PipelineTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.train, self.test = self.root / "train.jsonl", self.root / "test.jsonl"
        self.artifacts, self.output = self.root / "artifacts", self.root / "output"
        write_rows(self.train, dataset("train", True))
        write_rows(self.test, dataset("test", False))
        inference = Settings("gemini/gemini-flash-latest", max_output_tokens=2048)
        meta = Settings("gemini/gemini-4-argon", max_output_tokens=32768)
        self.config = Config(inference, inference, meta, meta, concurrency=1, request_interval_seconds=0)
        self.provider = FakeProvider()
        self.api = AsyncMock(side_effect=self.provider.complete)
        for target, value in [("solution.llm.litellm.acompletion", self.api),
                              ("solution.llm.litellm.completion_cost", 0.000001),
                              ("solution.adapt.prices_for", PRICES), ("solution.eval.prices_for", PRICES)]:
            mocked = patch(target, value) if target.endswith("acompletion") else patch(target, return_value=value)
            mocked.start()
            self.addCleanup(mocked.stop)
        self.silence = redirect_stdout(io.StringIO())
        self.silence.__enter__()
        self.addCleanup(self.silence.__exit__, None, None, None)

    async def adapt(self):
        return await adapt.run(self.train, self.artifacts, self.config)

    async def predict(self, artifacts=None, output=None, config=None):
        return await eval.run(self.test, artifacts or self.artifacts, output or self.output, config or self.config)

    async def test_complete_pipeline_with_isolated_read_only_artifacts_and_exact_routes(self):
        result = await self.adapt()
        self.assertEqual(result["status"], "complete")
        self.assertEqual(self.api.await_count, 10)  # 4 baseline + 4 audits + synthesis + explanation
        analyses = [json.loads(line) for line in (self.artifacts / "analyses.jsonl").read_text().splitlines()]
        self.assertEqual(len(analyses), 4)
        self.assertEqual(sum(not row["baseline_correct"] for row in analyses), 2)
        synthesis_input = json.loads((self.artifacts / "synthesis_input.json").read_text())
        self.assertEqual(synthesis_input["train_analyses"], analyses)
        self.assertEqual(synthesis_input["n_train_cases"], 4)
        report = (self.artifacts / "explanation.md").read_text()
        self.assertIn('"balance": 10', report)
        self.assertIn("Expert label: **good**", report)
        self.assertIn(artifact_value()["base_prompt"], report)
        # Simulate another container with just the required artifacts and no train/caches.
        isolated = self.root / "isolated"
        isolated.mkdir()
        for name in ("artifact.json", "routed_prompts.json", "explanation.json", "explanation.md",
                     "manifest.json", "summary.json", "budget.jsonl", ".run.lock"):
            shutil.copyfile(self.artifacts / name, isolated / name)
        self.train.unlink()
        before = {path.name: path.read_bytes() for path in isolated.iterdir()}
        for path in isolated.iterdir():
            path.chmod(0o444)
        isolated.chmod(0o555)
        try:
            await self.predict(artifacts=isolated)
            self.assertEqual(before, {path.name: path.read_bytes() for path in isolated.iterdir()})
        finally:
            isolated.chmod(0o755)
            for path in isolated.iterdir():
                path.chmod(0o644)
        with (self.output / "predictions.csv").open(newline="") as stream:
            reader = csv.DictReader(stream)
            self.assertEqual(reader.fieldnames, ["item_id", "label"])
            predictions = list(reader)
        self.assertEqual({row["item_id"] for row in predictions}, {f"test-{i}" for i in range(4)})
        for row, request in zip(dataset("test", False), self.provider.requests[-4:]):
            messages = request["messages"]
            self.assertEqual(json.loads(messages[-1]["content"]), {"context": row["context"], "conversation": row["conversation"]})
            prompt = messages[0]["content"]
            for topic in TOPICS:
                self.assertEqual(f"Apply only the {topic} criterion." in prompt, topic == row["topic"])
            self.assertEqual(request["num_retries"], 0)
            schema = request["response_format"]["json_schema"]["schema"]
            self.assertEqual(set(schema["properties"]), {"label"})
            self.assertFalse(schema["additionalProperties"])
        count = self.api.await_count
        await self.predict(artifacts=isolated)
        self.assertEqual(self.api.await_count, count)

    async def test_interrupted_adaptation_resumes_without_repaying_successful_rows(self):
        self.provider.fail_once_at = 2
        with self.assertRaises(RuntimeError):
            await self.adapt()
        self.assertFalse((self.artifacts / "artifact.json").exists())
        used = Budget(self.artifacts / "budget.jsonl", 10).used()
        self.assertGreater(used, 0.000001)  # the failed attempt retains its reservation
        await self.adapt()
        self.assertEqual(self.api.await_count, 11)  # only the failed attempt is repeated
        count = self.api.await_count
        await self.adapt()
        self.assertEqual(self.api.await_count, count)

    async def test_eval_resumes_and_never_emits_a_partial_submission(self):
        await self.adapt()
        self.provider.fail_once_at = self.api.await_count + 2
        with self.assertRaises(RuntimeError):
            await self.predict()
        self.assertFalse((self.output / "predictions.csv").exists())
        await self.predict()
        self.assertEqual(self.api.await_count, 15)
        self.assertEqual(json.loads((self.output / "summary.json").read_text())["status"], "complete")

    async def test_changed_inputs_and_config_reject_old_caches_before_requests(self):
        await self.adapt()
        count = self.api.await_count
        with self.assertRaisesRegex(ValueError, "fresh output directory"):
            await adapt.run(self.train, self.artifacts, replace(self.config, analysis=replace(self.config.analysis, max_output_tokens=123)))
        await self.predict()
        count = self.api.await_count
        rows = dataset("test", False)
        rows[0]["context"]["balance"] = 999
        write_rows(self.test, rows)
        with self.assertRaisesRegex(ValueError, "fresh output directory"):
            await self.predict()
        self.assertEqual(self.api.await_count, count)

    async def test_modified_artifact_and_labeled_test_fail_before_api(self):
        await self.adapt()
        count = self.api.await_count
        write_rows(self.test, dataset("test", True))
        with self.assertRaisesRegex(ValueError, "unlabeled"):
            await self.predict()
        write_rows(self.test, dataset("test", False))
        with (self.artifacts / "artifact.json").open("a") as stream:
            stream.write(" ")
        with self.assertRaisesRegex(ValueError, "modified"):
            await self.predict()
        self.assertEqual(self.api.await_count, count)

    async def test_incomplete_adaptation_and_different_judge_are_rejected(self):
        await self.adapt()
        changed = replace(self.config, inference=replace(self.config.inference, model="gemini/gemma-4-26b-a4b-it"))
        with self.assertRaisesRegex(ValueError, "must match"):
            await self.predict(config=changed)
        write_json(self.artifacts / "summary.json", {"status": "running"})
        with self.assertRaisesRegex(ValueError, "incomplete"):
            await self.predict()

    async def test_invalid_judge_reply_fails_and_is_still_accounted(self):
        await self.adapt()
        self.provider.invalid_judge = True
        with self.assertRaises(Exception):
            await self.predict()
        self.assertFalse((self.output / "predictions.csv").exists())
        self.assertEqual(Budget(self.output / "budget.jsonl", 10).used(), 0.000001)

    async def test_invalid_explanation_is_not_completed_and_can_be_retried_alone(self):
        self.provider.invalid_explanation = True
        with self.assertRaisesRegex(ValueError, "exact text"):
            await self.adapt()
        with self.assertRaises(ValueError):
            load_artifact(self.artifacts)
        count = self.api.await_count
        self.provider.invalid_explanation = False
        await self.adapt()
        self.assertEqual(self.api.await_count, count + 1)

    async def test_dry_run_does_not_write_or_call_api(self):
        await adapt.run(self.train, self.artifacts, self.config, dry_run=True)
        self.assertFalse(self.artifacts.exists())
        self.assertEqual(self.api.await_count, 0)
        await self.adapt()
        count = self.api.await_count
        await eval.run(self.test, self.artifacts, self.output, self.config, dry_run=True)
        self.assertFalse(self.output.exists())
        self.assertEqual(self.api.await_count, count)

    async def test_pipeline_budget_includes_adaptation_in_read_only_artifacts(self):
        await self.adapt()
        count = self.api.await_count
        used = Budget(self.artifacts / "budget.jsonl", 10).used()
        with self.assertRaises(BudgetExceeded):
            await self.predict(config=replace(self.config, budget_usd=used + 0.00000001))
        self.assertEqual(self.api.await_count, count)

    async def test_concurrent_requests_and_missing_usage_fail_without_default_labels(self):
        self.config = replace(self.config, concurrency=3)
        await self.adapt()
        self.assertEqual(self.api.await_count, 10)
        original = self.provider.complete

        async def missing_usage(**kwargs):
            response = await original(**kwargs)
            await asyncio.sleep(0)
            response.usage = None
            return response

        self.api.side_effect = missing_usage
        with self.assertRaisesRegex(ValueError, "usage/cost"):
            await self.predict()
        self.assertFalse((self.output / "predictions.csv").exists())
        self.assertGreater(Budget(self.output / "budget.jsonl", 10).used(), 0)


class ContractTests(unittest.TestCase):
    def test_explanation_requires_real_ids_and_exact_prompt_quotes(self):
        artifact = PromptArtifact.model_validate(artifact_value())
        value = explanation_value()
        value["criteria"][0]["example_ids"] = ["invented-id"]
        with self.assertRaises(ValueError):
            validate_explanation(Explanation.model_validate(value), artifact, dataset("train", True))
        value = explanation_value()
        value["criteria"][0]["prompt_scope"] = "savings"
        value["criteria"][0]["prompt_quote"] = artifact.topic_advice.savings
        with self.assertRaises(ValueError):
            validate_explanation(Explanation.model_validate(value), artifact, dataset("train", True))

    def test_input_contract_duplicate_ids_and_missing_gold(self):
        with TemporaryDirectory() as temp:
            path = Path(temp) / "train.jsonl"
            rows = dataset("train", True)
            rows[1]["item_id"] = rows[0]["item_id"]
            write_rows(path, rows)
            with self.assertRaises(ValueError):
                read_inputs(path, labeled=True)
            write_rows(path, dataset("train", False))
            with self.assertRaises(ValueError):
                read_inputs(path, labeled=True)

    def test_unknown_prices_and_nonfinite_settings_are_rejected(self):
        with patch("solution.llm.litellm.get_model_info", return_value={"supports_response_schema": True,
                   "input_cost_per_token": 0, "output_cost_per_token": 0}):
            with self.assertRaises(ValueError):
                prices_for("unknown-model")
        with self.assertRaises(ValueError):
            Settings("model", timeout_seconds=float("nan"))

    def test_corrupt_budget_cannot_disable_the_spending_limit(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "budget.jsonl"
            for amount in (float("nan"), float("inf"), -1):
                path.write_text(json.dumps({"event": "reserved", "attempt_id": "x", "accounted_usd": amount}) + "\n")
                with self.assertRaises(ValueError):
                    Budget(path, 10).require(0.1)

    def test_cli_masks_provider_value_errors(self):
        stderr = io.StringIO()
        with patch("solution.__main__.Config.from_env", side_effect=ValueError("SECRET_API_KEY")), redirect_stderr(stderr):
            self.assertEqual(main(["adapt"]), 1)
        self.assertNotIn("SECRET_API_KEY", stderr.getvalue())

    def test_process_environment_precedes_solution_dotenv(self):
        with TemporaryDirectory() as directory, patch.dict(os.environ):
            root = Path(directory) / "repo"
            (root / "solution").mkdir(parents=True)
            (root / "solution/.env").write_text("MODEL=solution-model\n")
            (root / ".env").write_text("MODEL=repo-model\n")
            os.environ.pop("MODEL", None)
            with patch("solution.config.ROOT", root):
                self.assertEqual(Config.from_env().inference.model, "solution-model")
                os.environ["MODEL"] = "process-model"
                self.assertEqual(Config.from_env().inference.model, "process-model")


if __name__ == "__main__":
    unittest.main()
