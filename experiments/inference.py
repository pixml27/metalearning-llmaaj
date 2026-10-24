"""Bounded judge execution; one model per request, one synchronous result writer."""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import time
from pathlib import Path

from experiments.llm import Judgment, judge_one, make_judge


def request_interval(value: str) -> float:
    seconds = float(value)
    if not math.isfinite(seconds) or seconds < 0:
        raise argparse.ArgumentTypeError("request interval must be finite and nonnegative")
    return seconds


def add_inference_arguments(parser):
    parser.add_argument("--concurrency", type=int, choices=(1, 2, 3), default=3,
                        help="Maximum in-flight judge requests (default: 3; 1 runs sequentially)")
    parser.add_argument("--request-interval", type=request_interval, default=0,
                        help="Minimum seconds between request starts across all workers")


def cached_judgments(path: Path) -> dict[str, Judgment]:
    valid = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            call = json.loads(line)
            if call.get("finish_reason") != "stop" or not call.get("raw_output"):
                continue
            try:
                valid[call["item_id"]] = Judgment.model_validate_json(call["raw_output"])
            except ValueError:
                pass
    return valid


async def run_judgments(rows, *, prompt_for, settings, budget, calls_path, run_id,
                        prices, on_result, concurrency=3, interval=0):
    """Stop admitting work on failure, drain paid requests, then raise the first error.

    Budget reservations/settlements and journal writes contain no await: under the
    caller's experiment file lock they run serially on this event loop. on_result
    is also synchronous, so CSV/summary writers never overlap. No threads/retries.
    """
    if concurrency not in (1, 2, 3) or not math.isfinite(interval) or interval < 0:
        raise ValueError("Invalid inference concurrency/request interval")
    remaining = iter(rows)
    gate = asyncio.Lock()
    last_started = None
    stopped = asyncio.Event()

    async def execute(row):
        nonlocal last_started
        try:
            async with gate:
                if last_started is not None:
                    await asyncio.sleep(max(0, interval - (time.monotonic() - last_started)))
                if stopped.is_set():
                    return row, None
                last_started = time.monotonic()
            # Shared JudgmentModel.item_id is mutable; never reuse it concurrently.
            agent, model = make_judge(prompt_for(row), settings, budget, calls_path, run_id, prices)
            return row, await judge_one(agent, model, row)
        except Exception:
            stopped.set()
            raise

    active = set()

    def fill():
        while not stopped.is_set() and len(active) < concurrency:
            row = next(remaining, None)
            if row is None:
                break
            active.add(asyncio.create_task(execute(row)))

    first_error = None
    try:
        fill()
        while active:
            done, active = await asyncio.wait(active, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                try:
                    row, result = task.result()
                    if result is not None:
                        on_result(row, result)
                except Exception as exc:
                    stopped.set()
                    if first_error is None:
                        first_error = exc
            fill()
        if first_error is not None:
            raise first_error
    finally:
        # External cancellation must leave no orphan tasks running after lock release.
        for task in active:
            task.cancel()
        if active:
            await asyncio.gather(*active, return_exceptions=True)
