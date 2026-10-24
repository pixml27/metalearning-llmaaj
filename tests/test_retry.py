"""Transient-failure retries, backoff and budget release, offline with a fake provider."""
from contextlib import redirect_stdout
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
import io
import json
import unittest

import litellm
from litellm import Usage

from solution.common import Judgment, SolutionError, read_jsonl
from solution.config import Config, Settings
from solution.llm import Budget, Runner, backoff_delay, transient, unbilled

PRICES = {"input_usd_per_token_upper": 0.0000001, "output_usd_per_token_upper": 0.0000001}
COST = 0.000001


def rate_limit():
    return litellm.RateLimitError(message="slow down", llm_provider="gemini", model="m")


def server_error():
    return litellm.InternalServerError(message="unavailable", llm_provider="gemini", model="m")


def timed_out():
    return litellm.Timeout(message="timed out", llm_provider="gemini", model="m")


def rejected():
    return litellm.BadRequestError(message="rejected", llm_provider="gemini", model="m")


class FakeProvider:
    """Fails the first `failures` requests with `error`, then answers with `label`."""

    def __init__(self, error=None, failures=0, label="good"):
        self.requests, self.error, self.failures, self.label = [], error, failures, label

    async def complete(self, **kwargs):
        self.requests.append(kwargs)
        if self.error is not None and len(self.requests) <= self.failures:
            raise self.error
        return SimpleNamespace(usage=Usage(prompt_tokens=20, completion_tokens=30), model="resolved-test-model",
                               id=f"response-{len(self.requests)}",
                               choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps({"label": self.label})),
                                                        finish_reason="stop")])


class RetryTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        judge = Settings("gemini/gemini-flash-latest", max_output_tokens=2048)
        self.config = Config(judge, judge, judge, judge, concurrency=1,
                             request_interval_seconds=0, retry_backoff_seconds=0)
        self.silence = redirect_stdout(io.StringIO())
        self.silence.__enter__()
        self.addCleanup(self.silence.__exit__, None, None, None)

    async def stage(self, provider, *, jobs=1, **overrides):
        config = replace(self.config, **overrides) if overrides else self.config
        api = AsyncMock(side_effect=provider.complete)
        with patch("solution.llm.litellm.acompletion", api), \
             patch("solution.llm.litellm.completion_cost", return_value=COST):
            items = [(f"item-{index}", "Judge the whole conversation.",
                      {"context": {"balance": index}, "conversation": []}) for index in range(jobs)]
            return await Runner(self.directory, config).stage("inference", items, Judgment,
                                                              config.inference, PRICES)

    def used(self):
        return Budget(self.directory / "budget.jsonl", 10).used()

    def journal(self):
        return read_jsonl(self.directory / "inference_calls.jsonl")

    async def test_rate_limit_is_retried_and_releases_its_reservation(self):
        provider = FakeProvider(rate_limit(), failures=2)
        results = await self.stage(provider)
        self.assertEqual(results["item-0"].label, "good")
        self.assertEqual(len(provider.requests), 3)
        # Only the billed response remains accounted; rejected attempts generate no tokens.
        self.assertAlmostEqual(self.used(), COST)
        self.assertEqual([call.get("error_type") for call in self.journal()],
                         ["RateLimitError", "RateLimitError", None])
        self.assertEqual([call.get("reservation_released") for call in self.journal()], [True, True, None])

    async def test_server_error_and_timeout_are_retried_but_stay_accounted(self):
        for error in (server_error(), timed_out()):
            with self.subTest(error=type(error).__name__):
                self.setUp()
                provider = FakeProvider(error, failures=1)
                await self.stage(provider)
                self.assertEqual(len(provider.requests), 2)
                # Generation may have been billed before the failure, so the reservation is kept.
                self.assertGreater(self.used(), COST)
                self.assertEqual(self.journal()[0]["reservation_released"], False)

    async def test_exhausted_retries_stop_the_stage_with_the_provider_error(self):
        provider = FakeProvider(rate_limit(), failures=99)
        with self.assertRaises(litellm.RateLimitError):
            await self.stage(provider, jobs=2, max_attempts=2)
        # Two attempts for the first job; the remaining job is never started.
        self.assertEqual(len(provider.requests), 2)
        self.assertEqual(len(self.journal()), 2)

    async def test_rejected_request_is_not_retried(self):
        provider = FakeProvider(rejected(), failures=99)
        with self.assertRaises(litellm.BadRequestError):
            await self.stage(provider)
        self.assertEqual(len(provider.requests), 1)
        self.assertEqual(self.used(), 0.0)

    async def test_invalid_structured_reply_is_never_retried(self):
        provider = FakeProvider(label="invalid")
        with self.assertRaises(Exception) as caught:
            await self.stage(provider)
        self.assertNotIsInstance(caught.exception, litellm.APIError)
        self.assertEqual(len(provider.requests), 1)
        self.assertAlmostEqual(self.used(), COST)

    async def test_retried_stage_resumes_from_the_journal_without_repaying(self):
        provider = FakeProvider(rate_limit(), failures=99)
        with self.assertRaises(litellm.RateLimitError):
            await self.stage(provider, max_attempts=1)
        results = await self.stage(FakeProvider())
        self.assertEqual(results["item-0"].label, "good")
        self.assertAlmostEqual(self.used(), COST)


class ClassifierTests(unittest.TestCase):
    def test_transient_and_unbilled_classification(self):
        for error in (rate_limit(), server_error(), timed_out()):
            self.assertTrue(transient(error), type(error).__name__)
        for error in (rejected(), SolutionError("deliberate stop"), RuntimeError("unknown")):
            self.assertFalse(transient(error), type(error).__name__)
        self.assertTrue(unbilled(rate_limit()))
        self.assertTrue(unbilled(rejected()))
        self.assertFalse(unbilled(timed_out()))
        self.assertFalse(unbilled(SolutionError("deliberate stop")))

    def test_wrapped_provider_error_is_still_classified(self):
        wrapper = RuntimeError("adapter failure")
        wrapper.__cause__ = rate_limit()
        self.assertTrue(transient(wrapper))
        self.assertTrue(unbilled(wrapper))

    def test_backoff_grows_and_stays_within_the_cap(self):
        self.assertTrue(all(0 < backoff_delay(5, attempt) <= 60.0 for attempt in range(1, 12)))
        self.assertGreater(backoff_delay(5, 4), backoff_delay(5, 1))
        self.assertEqual(backoff_delay(0, 3), 0.0)


if __name__ == "__main__":
    unittest.main()
