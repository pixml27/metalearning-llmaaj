"""Native Pydantic-AI/LiteLLM requests with durable accounting and restart support."""
from dataclasses import asdict
from pathlib import Path
import asyncio
import json
import math
import os
import random
import time
import uuid

# Keep imports offline, disable telemetry and implicit dotenv discovery.
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
os.environ.setdefault("LITELLM_MODE", "PRODUCTION")
os.environ.setdefault("PYDANTIC_AI_NO_BANNER", "1")
import litellm
from pydantic_ai import Agent, NativeOutput
from pydantic_ai.messages import ModelResponse, TextPart
from pydantic_ai.models import check_allow_model_requests
from pydantic_ai.usage import RequestUsage, UsageLimits
from pydantic_ai_litellm import LiteLLMModel

from .common import SolutionError
from .common import append_jsonl, canonical, digest, read_jsonl, utc_now, write_json

litellm.suppress_debug_info = True
litellm.telemetry = False


class BudgetExceeded(SolutionError):
    pass


RETRY_DELAY_CAP_SECONDS = 60.0
# Provider-side rejections and server failures; the request can be sent again unchanged.
RETRYABLE_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504, 529})
# Rejected before generation, so the provider bills no tokens for the attempt.
UNBILLED_STATUS = frozenset({400, 401, 403, 404, 413, 429})


def _chain(exc):
    # Adapters can wrap the provider error; inspect a bounded cause chain.
    for _ in range(8):
        if exc is None:
            return
        yield exc
        exc = exc.__cause__


def transient(exc):
    """A rate limit, timeout or server-side failure is worth another identical attempt."""
    for error in _chain(exc):
        status = getattr(error, "status_code", None)
        if isinstance(status, int):
            if status in RETRYABLE_STATUS:
                return True
        elif isinstance(error, (litellm.Timeout, litellm.APIConnectionError, asyncio.TimeoutError)):
            return True
        if isinstance(error, SolutionError):
            # Our own validation, cost and budget checks are deliberate stops, never retried.
            return False
    return False


def unbilled(exc):
    """True only when the provider rejected the request without generating tokens."""
    for error in _chain(exc):
        if isinstance(error, SolutionError):
            return False
        status = getattr(error, "status_code", None)
        if isinstance(status, int):
            return status in UNBILLED_STATUS
    return False


def backoff_delay(base, attempt):
    # Exponential with half jitter so concurrent workers do not retry in lockstep.
    delay = min(base * 2 ** (attempt - 1), RETRY_DELAY_CAP_SECONDS)
    return delay / 2 + random.random() * delay / 2


class Budget:
    """Caller holds a directory lock. Unsettled reservations survive restarts."""

    def __init__(self, path: Path, limit: float):
        self.path, self.limit = path, limit

    def used(self):
        attempts = {}
        for event in read_jsonl(self.path):
            amount = event.get("accounted_usd")
            if (event.get("event") not in {"reserved", "settled"} or not event.get("attempt_id")
                    or not isinstance(amount, (int, float)) or not math.isfinite(amount) or amount < 0):
                raise SolutionError("Invalid budget journal; refusing requests")
            attempts[event["attempt_id"]] = amount
        return sum(attempts.values())

    def require(self, amount):
        if not math.isfinite(amount) or amount < 0:
            raise SolutionError("Cannot authorize unknown or invalid request cost")
        if self.used() + amount > self.limit:
            raise BudgetExceeded("RUN_BUDGET_USD would be exceeded; successful responses are saved")

    def reserve(self, amount, *, stage, item_id):
        self.require(amount)
        attempt_id = uuid.uuid4().hex
        append_jsonl(self.path, {"event": "reserved", "attempt_id": attempt_id,
                                "stage": stage, "item_id": item_id, "accounted_usd": amount, "time": utc_now()})
        return attempt_id

    def settle(self, attempt_id, amount):
        if not math.isfinite(amount) or amount < 0:
            raise SolutionError("Invalid response cost; reservation retained")
        append_jsonl(self.path, {"event": "settled", "attempt_id": attempt_id,
                                "accounted_usd": amount, "time": utc_now()})
        self.require(0)


def prices_for(model):
    info = litellm.get_model_info(model)
    if not info.get("supports_response_schema"):
        raise SolutionError("Model must declare native JSON schema support")
    inputs = [value for key, value in info.items()
              if key.startswith("input_cost_per_token") and isinstance(value, (int, float))]
    outputs = [value for key, value in info.items()
               if key.startswith(("output_cost_per_token", "output_cost_per_reasoning_token")) and isinstance(value, (int, float))]
    if not inputs or not outputs or any(not math.isfinite(value) or value < 0 for value in inputs + outputs):
        raise SolutionError("Unknown token prices; refusing requests")
    known_free = model in {"gemini/gemma-4-26b-a4b-it", "gemini/gemma-4-31b-it"}
    if (max(inputs) == 0 or max(outputs) == 0) and not (known_free and max(inputs) == max(outputs) == 0):
        raise SolutionError("Unverified zero token prices; refusing requests")
    return {"input_usd_per_token_upper": max(inputs), "output_usd_per_token_upper": max(outputs),
            "source": "installed LiteLLM snapshot; estimate, not provider invoice"}


def request_ceiling(messages, schema, settings, prices):
    # Text-only conservative estimate, including schema, framing and thinking output.
    size = len(canonical({"messages": messages, "schema": schema}).encode()) + 1024
    return size * prices["input_usd_per_token_upper"] + settings.max_output_tokens * prices["output_usd_per_token_upper"]


def estimate(prompt, payload, schema, settings, prices):
    return request_ceiling([{"role": "system", "content": prompt}, {"role": "user", "content": canonical(payload)}],
                           schema.model_json_schema(), settings, prices)


def request_key(prompt, payload, schema, settings):
    return digest(canonical({"prompt": prompt, "payload": payload,
                             "schema": schema.model_json_schema(), "settings": asdict(settings)}))


class StructuredModel(LiteLLMModel):
    """Bridge native schema and RequestUsage for the pinned adapter version."""

    def __init__(self, settings, budget, calls_path, stage, item_id, key, prices):
        super().__init__(settings.model)
        self._profile = {"supports_json_schema_output": True, "supports_tools": False}
        self.request_config, self.budget, self.calls_path = settings, budget, calls_path
        self.stage, self.item_id, self.key, self.prices = stage, item_id, key, prices

    async def request(self, messages, model_settings, model_request_parameters):
        check_allow_model_requests()
        mapped = await self._map_messages(messages, model_request_parameters)
        output = model_request_parameters.output_object
        if model_request_parameters.output_mode != "native" or output is None:
            raise SolutionError("Native structured output is required")
        ceiling = request_ceiling(mapped, output.json_schema, self.request_config, self.prices)
        attempt_id = self.budget.reserve(ceiling, stage=self.stage, item_id=self.item_id)
        started = time.monotonic()
        event = {"item_id": self.item_id, "request_sha256": self.key, "attempt_id": attempt_id,
                 "stage": self.stage, "time": utc_now(), "requested_model": self.model_name,
                 "reserved_usd": ceiling, "cacheable": False}
        try:
            response = await litellm.acompletion(
                model=self.model_name, messages=mapped,
                response_format={"type": "json_schema", "json_schema": {
                    "name": output.name or "StructuredOutput", "schema": output.json_schema, "strict": True}},
                temperature=self.request_config.temperature, max_tokens=self.request_config.max_output_tokens,
                timeout=self.request_config.timeout_seconds, num_retries=0,
            )
        except Exception as exc:
            # Error strings can contain provider URLs or credentials; persist types only.
            append_jsonl(self.calls_path, {**event, "error_type": type(exc).__name__,
                                         "latency_seconds": time.monotonic() - started,
                                         "reservation_released": unbilled(exc)})
            if unbilled(exc):
                # A rejected attempt produced no tokens, so retries keep their budget headroom.
                self.budget.settle(attempt_id, 0.0)
            raise
        choice = response.choices[0] if response.choices else None
        content = choice.message.content if choice and choice.message else None
        event.update(response_model=response.model, response_id=response.id,
                     usage=response.usage.model_dump(mode="json") if response.usage else None,
                     finish_reason=choice.finish_reason if choice else None, raw_output=content,
                     latency_seconds=time.monotonic() - started)
        cost = None
        if response.usage is not None:
            try:
                cost = litellm.completion_cost(completion_response=response, model=self.model_name)
            except Exception:
                pass
        valid_cost = isinstance(cost, (int, float)) and math.isfinite(cost) and cost >= 0
        event["cost_usd_estimated"] = cost if valid_cost else None
        event["cacheable"] = bool(valid_cost and cost <= ceiling and content and choice.finish_reason == "stop")
        # Persist raw output before parsing, including invalid or truncated replies.
        append_jsonl(self.calls_path, event)
        if not valid_cost:
            raise SolutionError("Response usage/cost unavailable; reservation retained")
        self.budget.settle(attempt_id, cost)
        if cost > ceiling:
            raise SolutionError("Response cost exceeded its estimate; inspect pricing before continuing")
        if not event["cacheable"]:
            raise SolutionError("Missing or truncated structured output; no label assigned")
        return ModelResponse([TextPart(content=content)], model_name=response.model,
                             provider_name=self.model_name.split("/", 1)[0], provider_response_id=response.id,
                             usage=RequestUsage(input_tokens=response.usage.prompt_tokens,
                                                output_tokens=response.usage.completion_tokens))


async def structured_call(prompt, payload, schema, settings, budget, calls_path, stage, item_id, prices):
    model = StructuredModel(settings, budget, calls_path, stage, item_id,
                            request_key(prompt, payload, schema, settings), prices)
    agent = Agent(model=model, system_prompt=prompt, output_type=NativeOutput(schema), retries=0,
                  name="solution_" + stage)
    result = await agent.run(canonical(payload), usage_limits=UsageLimits(request_limit=1))
    return result.output


class StartGate:
    def __init__(self, interval):
        self.interval, self.last = interval, 0.0
        self.lock = asyncio.Lock()

    async def wait(self):
        async with self.lock:
            await asyncio.sleep(max(0, self.interval - (time.monotonic() - self.last)))
            self.last = time.monotonic()


class Runner:
    def __init__(self, directory, config):
        self.directory, self.config = directory, config
        self.budget = Budget(directory / "budget.jsonl", config.budget_usd)
        self.gate = StartGate(config.request_interval_seconds)

    async def stage(self, name, jobs, schema, settings, prices, *, validate=None):
        """Jobs are (opaque ID, exact prompt, payload). Resume only matching valid outputs."""
        expected = {item_id: request_key(prompt, payload, schema, settings) for item_id, prompt, payload in jobs}
        if len(expected) != len(jobs):
            raise SolutionError("Duplicate stage job IDs")
        calls_path = self.directory / f"{name}_calls.jsonl"
        results = {}
        for call in read_jsonl(calls_path):
            item_id = call["item_id"]
            if expected.get(item_id) != call["request_sha256"]:
                raise SolutionError("Cached response does not match this request")
            if not call.get("cacheable"):
                continue
            try:
                value = schema.model_validate_json(call["raw_output"])
                if validate is not None:
                    validate(value)
            except ValueError:
                continue
            if item_id in results and results[item_id] != value:
                raise SolutionError("Conflicting successful responses for one job")
            results[item_id] = value
        pending = [job for job in jobs if job[0] not in results]
        upper = sum(estimate(prompt, payload, schema, settings, prices) for _, prompt, payload in pending)
        self.budget.require(upper)
        print(canonical({"stage": name, "cached": len(results), "pending": len(pending),
                         "estimated_upper_usd": upper, "accounted_usd": self.budget.used()}), flush=True)
        failures = []
        semaphore = asyncio.Semaphore(self.config.concurrency)

        def progress():
            state = {"stage": name, "completed": len(results), "total": len(jobs),
                     "accounted_usd": self.budget.used(), "updated_at": utc_now()}
            write_json(self.directory / "progress.json", state)
            print(canonical(state), flush=True)

        async def work(job):
            item_id, prompt, payload = job
            async with semaphore:
                for attempt in range(1, self.config.max_attempts + 1):
                    if failures:
                        return
                    await self.gate.wait()
                    if failures:
                        return
                    try:
                        value = await structured_call(prompt, payload, schema, settings, self.budget,
                                                      calls_path, name, item_id, prices)
                        if validate is not None:
                            validate(value)
                        results[item_id] = value
                        progress()
                        return
                    except Exception as exc:
                        if attempt >= self.config.max_attempts or not transient(exc):
                            failures.append(exc)
                            return
                        delay = backoff_delay(self.config.retry_backoff_seconds, attempt)
                        print(canonical({"stage": name, "item_id": item_id, "attempt": attempt,
                                         "error_type": type(exc).__name__,
                                         "retry_in_seconds": round(delay, 3)}), flush=True)
                        await asyncio.sleep(delay)
        progress()
        # Await in-flight requests before releasing the directory lock after failure.
        await asyncio.gather(*(work(job) for job in pending))
        if failures:
            raise failures[0]
        return results


def usage_summary(directory):
    stages = {}
    for path in sorted(directory.glob("*_calls.jsonl")):
        calls = read_jsonl(path)
        usage = [call.get("usage") or {} for call in calls]
        stages[path.stem.removesuffix("_calls")] = {
            "attempts": len(calls), "response_models": sorted({call["response_model"] for call in calls if call.get("response_model")}),
            "cost_usd_estimated": sum(call.get("cost_usd_estimated") or 0 for call in calls),
            "input_tokens": sum(value.get("prompt_tokens", 0) or 0 for value in usage),
            "output_tokens": sum(value.get("completion_tokens", 0) or 0 for value in usage),
        }
    return stages
