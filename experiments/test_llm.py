"""Offline checks for paid-call boundaries, native output and no-label leakage."""
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
from litellm import Usage
from experiments.llm import Budget, BudgetExceeded, Settings, judge_one, make_judge, prices_for


class PricingTests(unittest.TestCase):
    def test_known_free_gemma_has_zero_ceiling_but_unknown_zero_is_rejected(self):
        entry = {"supports_response_schema": True, "input_cost_per_token": 0.0, "output_cost_per_token": 0.0}
        with patch("experiments.llm.litellm.get_model_info", return_value=entry):
            self.assertEqual(prices_for("gemini/gemma-4-26b-a4b-it")["input_usd_per_token_upper"], 0)
            self.assertEqual(prices_for("gemini/gemma-4-31b-it")["output_usd_per_token_upper"], 0)
            with self.assertRaises(ValueError):
                prices_for("gemini/unknown-model")

    def test_missing_negative_and_nonfinite_prices_remain_blocked(self):
        for bad in [None, -1, float("nan"), float("inf")]:
            with self.subTest(price=bad), patch("experiments.llm.litellm.get_model_info", return_value={
                    "supports_response_schema": True, "input_cost_per_token": bad, "output_cost_per_token": 0.0}):
                with self.assertRaises(ValueError):
                    prices_for("gemini/gemma-4-26b-a4b-it")


class SettingsTests(unittest.TestCase):
    def test_experiment_model_precedes_shared_dotenv_but_respects_process_override(self):
        with TemporaryDirectory() as directory, patch.dict(os.environ):
            path = Path(directory) / ".env"
            path.write_text("MODEL=gemini/gemma-4-26b-a4b-it\n")
            os.environ.pop("MODEL", None)
            self.assertEqual(Settings.from_env(path).model, "gemini/gemma-4-26b-a4b-it")
            os.environ["MODEL"] = "gemini/explicit-override"
            self.assertEqual(Settings.from_env(path).model, "gemini/explicit-override")


class BudgetTests(unittest.TestCase):
    def test_reservations_survive_reload_and_settlement_replaces_reservation(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "budget.jsonl"
            budget = Budget(path, 1)
            attempt = budget.reserve(0.7, item_id="x", run_id="r")
            with self.assertRaises(BudgetExceeded):
                Budget(path, 1).require(0.4)
            budget.settle(attempt, 0.2)
            self.assertAlmostEqual(Budget(path, 1).used(), 0.2)
            Budget(path, 1).require(0.8)


class InferenceTests(unittest.IsolatedAsyncioTestCase):
    async def run_judge(self, text, limit=10, *, model_name="gemini/gemini-flash-latest", free=False):
        row = {"item_id": "test-id", "topic": "credit", "label": "bad", "input_en": "DO NOT SEND",
               "context": {"balance": 15}, "conversation": [{"role": "assistant", "content": "15"}]}
        response = SimpleNamespace(
            usage=Usage(prompt_tokens=12, completion_tokens=8), model="resolved-model", id="response-id",
            choices=[SimpleNamespace(message=SimpleNamespace(content=text), finish_reason="stop")])
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        directory = Path(temp.name)
        budget = Budget(directory / "budget.jsonl", limit)
        prices = {"input_usd_per_token_upper": 0.000001, "output_usd_per_token_upper": 0.000005}
        if free:
            prices = dict.fromkeys(prices, 0.0)
        agent, model = make_judge("Judge the conversation.", Settings(model_name),
                                  budget, directory / "calls.jsonl", "test", prices)
        api = AsyncMock(return_value=response)
        with patch("experiments.llm.litellm.acompletion", api), patch("experiments.llm.litellm.completion_cost", return_value=0.0 if free else 0.0001):
            try:
                result = await judge_one(agent, model, row)
                error = None
            except Exception as exc:
                result, error = None, exc
        return result, error, api, budget, directory

    async def test_native_schema_and_payload_allowlist(self):
        result, error, api, budget, directory = await self.run_judge('{"label":"good"}')
        self.assertIsNone(error)
        self.assertEqual(result.label, "good")
        request = api.call_args.kwargs
        self.assertEqual(request["response_format"]["type"], "json_schema")
        self.assertEqual(request["response_format"]["json_schema"]["schema"]["properties"]["label"]["enum"], ["good", "bad"])
        payload = json.loads(request["messages"][-1]["content"])
        self.assertEqual(set(payload), {"context", "conversation"})
        self.assertEqual(request["temperature"], 0)
        self.assertEqual(request["num_retries"], 0)
        self.assertAlmostEqual(budget.used(), 0.0001)
        self.assertTrue((directory / "calls.jsonl").exists())

    async def test_invalid_label_is_not_replaced_and_is_still_billed(self):
        result, error, api, budget, directory = await self.run_judge('{"label":"unknown"}')
        self.assertIsNone(result)
        self.assertIsNotNone(error)
        self.assertEqual(api.await_count, 1)
        self.assertAlmostEqual(budget.used(), 0.0001)

    async def test_extra_response_text_field_is_rejected(self):
        result, error, api, budget, directory = await self.run_judge('{"label":"good","reason":"fluff"}')
        self.assertIsNone(result)
        self.assertIsNotNone(error)
        self.assertEqual(api.await_count, 1)

    async def test_budget_blocks_before_the_api_call(self):
        result, error, api, budget, directory = await self.run_judge('{"label":"good"}', limit=0.000001)
        self.assertIsInstance(error, BudgetExceeded)
        self.assertEqual(api.await_count, 0)

    async def test_free_gemma_keeps_same_request_and_records_zero_cost(self):
        paid = await self.run_judge('{"label":"good"}')
        result, error, api, budget, directory = await self.run_judge(
            '{"label":"good"}', model_name="gemini/gemma-4-26b-a4b-it", free=True)
        self.assertIsNone(error)
        self.assertEqual(result.label, "good")
        paid_request, gemma_request = dict(paid[2].call_args.kwargs), dict(api.call_args.kwargs)
        self.assertEqual(gemma_request.pop("model"), "gemini/gemma-4-26b-a4b-it")
        paid_request.pop("model")
        self.assertEqual(paid_request, gemma_request)
        self.assertEqual(budget.used(), 0)
        call = json.loads((directory / "calls.jsonl").read_text())
        self.assertEqual(call["cost_usd_estimated"], 0)


if __name__ == "__main__":
    unittest.main()
