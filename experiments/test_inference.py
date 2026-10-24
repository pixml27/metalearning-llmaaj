"""Concurrent execution through the real adapter, with a controlled offline API."""
import argparse
import asyncio
import csv
import importlib
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from experiments.inference import add_inference_arguments, cached_judgments, run_judgments
from experiments.llm import Budget, BudgetExceeded, Settings
from litellm import Usage

REPO = Path(__file__).resolve().parents[1]


class ConcurrentInferenceTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.budget = Budget(self.directory / "budget.jsonl", 10)
        self.calls = self.directory / "calls.jsonl"
        self.rows = [{"item_id": str(i), "topic": "credit", "label": "SECRET GOLD",
                      "input_en": "SECRET TRANSLATION", "context": {"marker": str(i)},
                      "conversation": [{"role": "assistant", "content": str(i)}]} for i in range(9)]
        self.active = self.peak = 0
        self.started, self.finished, self.saved, self.requests = [], [], [], []

    async def api(self, **kwargs):
        payload = json.loads(kwargs["messages"][-1]["content"])
        self.assertEqual(set(payload), {"context", "conversation"})
        marker = payload["context"]["marker"]
        self.started.append((marker, time.monotonic()))
        self.requests.append(kwargs)
        self.active += 1
        self.peak = max(self.peak, self.active)
        try:
            # Deliberately complete each group out of order.
            await asyncio.sleep((3 - int(marker) % 3) * 0.01)
            self.finished.append(marker)
            label = "good" if int(marker) % 2 == 0 else "bad"
            return SimpleNamespace(usage=Usage(prompt_tokens=12, completion_tokens=8),
                model="resolved-model", id="response-" + marker,
                choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps({"label": label})),
                                         finish_reason="stop")])
        finally:
            self.active -= 1

    async def execute(self, rows=None, *, concurrency=3, interval=0, api=None, on_result=None):
        with patch("experiments.llm.litellm.acompletion", side_effect=api or self.api), \
             patch("experiments.llm.litellm.completion_cost", return_value=0.0001):
            await run_judgments(self.rows if rows is None else rows,
                prompt_for=lambda row: "Judge the conversation.", settings=Settings("gemini/gemini-flash-latest"),
                budget=self.budget, calls_path=self.calls, run_id="test", prices={
                    "input_usd_per_token_upper": 0.000001, "output_usd_per_token_upper": 0.000005},
                on_result=on_result or (lambda row, result: self.saved.append((row["item_id"], result.label))),
                concurrency=concurrency, interval=interval)

    async def test_three_in_flight_preserve_ids_labels_and_accounting_out_of_order(self):
        await self.execute()
        self.assertEqual(self.peak, 3)
        self.assertNotEqual(self.finished, [r["item_id"] for r in self.rows])
        cache = cached_judgments(self.calls)
        self.assertEqual(set(cache), {r["item_id"] for r in self.rows})
        for iid, value in cache.items():
            self.assertEqual(value.label, "good" if int(iid) % 2 == 0 else "bad")
        calls = [json.loads(line) for line in self.calls.read_text().splitlines()]
        self.assertTrue(all(c["response_id"] == "response-" + c["item_id"] for c in calls))
        self.assertEqual(len(self.saved), 9)
        self.assertAlmostEqual(self.budget.used(), 9 * 0.0001)

    async def test_parallel_execution_keeps_identical_model_parameters(self):
        await self.execute(self.rows[:3], concurrency=1)
        sequential = self.requests.copy()
        self.requests.clear()
        self.assertEqual(self.peak, 1)
        await self.execute(self.rows[:3], concurrency=3)
        self.assertEqual(self.requests, sequential)

    async def test_api_error_stops_admission_drains_successes_and_resume_skips_them(self):
        async def fail_first(**kwargs):
            marker = json.loads(kwargs["messages"][-1]["content"])["context"]["marker"]
            if marker == "0":
                await asyncio.sleep(0.005)
                raise RuntimeError("API failed")
            return await self.api(**kwargs)

        with self.assertRaisesRegex(RuntimeError, "API failed"):
            await self.execute(api=fail_first)
        self.assertEqual({iid for iid, _ in self.saved}, {"1", "2"})
        self.assertEqual(self.active, 0)
        cache = cached_judgments(self.calls)
        pending = [row for row in self.rows if row["item_id"] not in cache]
        self.saved.clear()
        await self.execute(pending)
        self.assertEqual(len(cached_judgments(self.calls)), 9)
        self.assertNotIn("1", {iid for iid, _ in self.saved})
        self.assertNotIn("2", {iid for iid, _ in self.saved})
        events = [json.loads(line) for line in self.budget.path.read_text().splitlines()]
        # Failed calls retain their reservation; every successful call settles once.
        self.assertEqual(sum(e["event"] == "reserved" for e in events), 10)
        self.assertEqual(sum(e["event"] == "settled" for e in events), 9)

    async def test_writer_error_preserves_paid_responses_in_journal(self):
        def broken_writer(row, result):
            raise OSError("disk writer failed")
        with self.assertRaisesRegex(OSError, "disk writer failed"):
            await self.execute(on_result=broken_writer)
        self.assertEqual(set(cached_judgments(self.calls)), {"0", "1", "2"})
        self.assertEqual(len(self.started), 3)
        self.assertEqual(self.active, 0)

    async def test_global_start_interval_and_stop_waiting_work_after_failure(self):
        await self.execute(self.rows[:4], interval=0.02)
        starts = [started for _, started in self.started]
        self.assertTrue(all(b - a >= 0.018 for a, b in zip(starts, starts[1:])))
        self.started.clear()
        async def fail_immediately(**kwargs):
            self.started.append(("failed", time.monotonic()))
            raise RuntimeError("stop")
        with self.assertRaisesRegex(RuntimeError, "stop"):
            await self.execute(interval=0.01, api=fail_immediately)
        self.assertEqual(len(self.started), 1)

    async def test_budget_blocks_concurrent_calls_before_api(self):
        self.budget = Budget(self.budget.path, 0.000001)
        with self.assertRaises(BudgetExceeded):
            await self.execute()
        self.assertEqual(self.started, [])

    async def test_cancellation_leaves_no_background_requests(self):
        entered = asyncio.Event()
        async def slow_api(**kwargs):
            self.active += 1
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                self.active -= 1
        task = asyncio.create_task(self.execute(api=slow_api))
        await entered.wait()
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual(self.active, 0)
        self.assertFalse(self.calls.exists())


class EntrypointTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)

    async def test_all_inference_entrypoints_resume_without_repaying(self):
        with (REPO / "data/public_test.csv").open(newline="") as stream:
            reader = csv.DictReader(stream)
            fieldnames = reader.fieldnames
            selected = {}
            for row in reader:
                selected.setdefault(row["topic"], row)
        dataset = self.directory / "sample.csv"
        with dataset.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(selected.values())
        settings = Settings("gemini/gemini-flash-latest")
        async def simple_api(**kwargs):
            await asyncio.sleep(0.005)
            return SimpleNamespace(usage=Usage(prompt_tokens=12, completion_tokens=8),
                model="resolved-model", id="response",
                choices=[SimpleNamespace(message=SimpleNamespace(content='{"label":"good"}'), finish_reason="stop")])
        for experiment in ("baseline", "exp1", "exp2", "exp3"):
            with self.subTest(experiment=experiment):
                module = importlib.import_module(f"experiments.{experiment}.infer")
                here = self.directory / experiment
                here.mkdir()
                (here / "prompt.md").write_text("Judge the conversation.")
                output = here / "run"
                args = SimpleNamespace(dataset=dataset, test=dataset, split="test", language="original",
                    run_dir=output, limit=1, dry_run=False, concurrency=3, request_interval=0,
                    artifacts=REPO / f"experiments/{experiment}/artifacts/20261024-public-original",
                    experiment_dir=here, experiment_name=experiment, expected_model=None,
                    entrypoint=Path(module.__file__).resolve())
                settings_patch = patch.object(module.Settings, "from_env", return_value=settings) if experiment == "baseline" \
                    else patch.object(module, "stage_settings", return_value={"inference": settings})
                with patch.object(module, "HERE", here), settings_patch, \
                     patch.dict(os.environ, {"GEMINI_API_KEY": "offline-test-placeholder"}), \
                     patch.object(module, "update_results"), patch("builtins.print"), \
                     patch("experiments.llm.litellm.acompletion", side_effect=simple_api) as api, \
                     patch("experiments.llm.litellm.completion_cost", return_value=0.0001):
                    await module.run(args)
                    self.assertEqual(api.call_count, 1)
                    # Simulate losing CSV after the response was journaled.
                    (output / "predictions.csv").unlink()
                    args.limit = None
                    await module.run(args)
                    self.assertEqual(api.call_count, 4)
                    summary = json.loads((output / "summary.json").read_text())
                    self.assertEqual(summary["completed"], 4)
                    self.assertEqual(summary["status"], "complete")
                    self.assertEqual(json.loads((output / "score.json").read_text())["n"], 4)
                    await module.run(args)
                    self.assertEqual(api.call_count, 4)
                with (output / "predictions.csv").open() as stream:
                    self.assertEqual([r["item_id"] for r in csv.DictReader(stream)],
                                     [r["item_id"] for r in selected.values()])


class ArgumentTests(unittest.TestCase):
    def test_defaults_fallback_and_validation(self):
        parser = argparse.ArgumentParser()
        add_inference_arguments(parser)
        self.assertEqual(parser.parse_args([]).concurrency, 3)
        self.assertEqual(parser.parse_args(["--concurrency", "1"]).concurrency, 1)
        for args in (["--concurrency", "4"], ["--request-interval", "nan"], ["--request-interval", "-1"]):
            with self.subTest(args=args), patch("sys.stderr"), self.assertRaises(SystemExit):
                parser.parse_args(args)


if __name__ == "__main__":
    unittest.main()
