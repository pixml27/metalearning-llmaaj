"""Small, label-only Pydantic-AI/LiteLLM adapter with a persistent experiment budget."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal
import hashlib
import json
import math
import os
import time
import uuid

# Reproducible installed price snapshot; no background network requests on import.
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
os.environ.setdefault("PYDANTIC_AI_NO_BANNER", "1")
# Settings owns dotenv precedence; LiteLLM's DEV default otherwise loads an
# unrelated parent .env at import time, before the experiment can choose MODEL.
os.environ.setdefault("LITELLM_MODE", "PRODUCTION")
import litellm
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict
from pydantic_ai import Agent, NativeOutput
from pydantic_ai.messages import ModelResponse, TextPart
from pydantic_ai.models import check_allow_model_requests
from pydantic_ai.usage import RequestUsage, UsageLimits
from pydantic_ai_litellm import LiteLLMModel

litellm.suppress_debug_info = True
litellm.telemetry = False

REPO = Path(__file__).resolve().parents[1]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def digest(value: str | bytes) -> str:
    return hashlib.sha256(value.encode() if isinstance(value, str) else value).hexdigest()


def append_jsonl(path: Path, value: dict) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


@dataclass(frozen=True)
class Settings:
    model: str
    budget_usd: float = 10.0
    max_output_tokens: int = 2048
    timeout_seconds: float = 120.0
    temperature: float = 0.0

    @classmethod
    def from_env(cls, experiment_env: Path | None = None) -> Settings:
        # Process environment > experiment .env > repository .env > workspace ../.env.
        paths = [REPO / ".env", REPO.parent / ".env"]
        if experiment_env is not None:
            paths.insert(0, experiment_env)
        for path in paths:
            load_dotenv(path, override=False)
        model = os.getenv("MODEL", "gemini/gemini-flash-latest")
        settings = cls(model=model, budget_usd=float(os.getenv("EXPERIMENT_BUDGET_USD", "10")),
                       max_output_tokens=int(os.getenv("MAX_OUTPUT_TOKENS", "2048")),
                       timeout_seconds=float(os.getenv("REQUEST_TIMEOUT_SECONDS", "120")))
        if not math.isfinite(settings.budget_usd) or not 0 < settings.budget_usd <= 10:
            raise ValueError("EXPERIMENT_BUDGET_USD must be in (0, 10]; higher spending needs explicit authorization")
        if settings.max_output_tokens <= 0 or settings.timeout_seconds <= 0:
            raise ValueError("Token and timeout limits must be positive")
        return settings


class Judgment(BaseModel):
    model_config = ConfigDict(extra="forbid")
    label: Literal["good", "bad"]


class BudgetExceeded(RuntimeError):
    pass


class Budget:
    """One ledger for the entire experiment, including smoke tests and resumed runs.

    Reservations survive crashes/timeouts because a provider may have billed them.
    Caller holds the experiment lock. Amounts are price-map estimates, not invoices.
    """

    def __init__(self, path: Path, limit: float):
        self.path, self.limit = path, limit

    def used(self) -> float:
        attempts = {}
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                event = json.loads(line)
                attempts[event["attempt_id"]] = event["accounted_usd"]
        return sum(attempts.values())

    def require(self, amount: float) -> None:
        if not math.isfinite(amount) or amount < 0:
            raise ValueError("Cannot authorize unknown/invalid cost")
        if self.used() + amount > self.limit:
            raise BudgetExceeded("Experiment budget would be exceeded")

    def reserve(self, amount: float, *, item_id: str, run_id: str) -> str:
        self.require(amount)
        attempt_id = uuid.uuid4().hex
        append_jsonl(self.path, {"event": "reserved", "attempt_id": attempt_id, "item_id": item_id,
                                "run_id": run_id, "accounted_usd": amount, "time": utc_now()})
        return attempt_id

    def settle(self, attempt_id: str, amount: float) -> None:
        if not math.isfinite(amount) or amount < 0:
            raise ValueError("Unknown/invalid response cost: reservation retained")
        append_jsonl(self.path, {"event": "settled", "attempt_id": attempt_id,
                                "accounted_usd": amount, "time": utc_now()})
        if self.used() > self.limit:
            raise BudgetExceeded("Response cost exceeded experiment budget; no further calls permitted")


def prices_for(model: str) -> dict:
    info = litellm.get_model_info(model)
    if not info.get("supports_response_schema"):
        raise ValueError("Selected model has no declared native response-schema support")
    # Use the largest token price in the installed model entry, including tiers.
    inputs = [v for k, v in info.items() if k.startswith("input_cost_per_token") and isinstance(v, (int, float))]
    outputs = [v for k, v in info.items() if k.startswith(("output_cost_per_token", "output_cost_per_reasoning_token"))
               and isinstance(v, (int, float))]
    if (not inputs or not outputs
            or any(not math.isfinite(p) or p < 0 for p in inputs + outputs)):
        raise ValueError("Unknown token prices; refusing paid requests")
    # Google lists these Gemini API models as free. A zero for another model
    # may be an incomplete price entry, so do not silently authorize it.
    known_free = model in {"gemini/gemma-4-26b-a4b-it", "gemini/gemma-4-31b-it"}
    if (max(inputs) == 0 or max(outputs) == 0) and not (known_free and max(inputs) == max(outputs) == 0):
        raise ValueError("Unverified zero token prices; refusing requests")
    return {"input_usd_per_token_upper": max(inputs), "output_usd_per_token_upper": max(outputs),
            "source": "installed LiteLLM model cost map; estimate, not a provider invoice"}


def request_ceiling(messages: list[dict], schema: dict, settings: Settings, prices: dict) -> float:
    # Text-only input: one UTF-8 byte per token is a conservative estimate.
    # Include schema and generous framing overhead. Bound thinking + visible output.
    data = json.dumps({"messages": messages, "schema": schema}, ensure_ascii=False).encode()
    return ((len(data) + 1024) * prices["input_usd_per_token_upper"]
            + settings.max_output_tokens * prices["output_usd_per_token_upper"])


class JudgmentModel(LiteLLMModel):
    """Bridge native schema and request accounting missing from adapter 0.2.8.

    Reuses its Pydantic-AI message mapping and LiteLLM model interface. Non-streaming,
    no tools, no implicit retries, no reason text. Provider failures never become labels.
    """

    def __init__(self, settings: Settings, budget: Budget, calls_path: Path, run_id: str, prices: dict):
        super().__init__(settings.model)
        self._profile = {"supports_json_schema_output": True, "supports_tools": False}
        self.config, self.budget, self.calls_path, self.run_id = settings, budget, calls_path, run_id
        self.prices = prices
        self.item_id = ""

    async def request(self, messages, model_settings, model_request_parameters):
        check_allow_model_requests()
        mapped = await self._map_messages(messages, model_request_parameters)
        output = model_request_parameters.output_object
        if model_request_parameters.output_mode != "native" or output is None:
            raise ValueError("Native structured output is required")
        schema = output.json_schema
        ceiling = request_ceiling(mapped, schema, self.config, self.prices)
        attempt_id = self.budget.reserve(ceiling, item_id=self.item_id, run_id=self.run_id)
        started = time.monotonic()
        response = await litellm.acompletion(
            model=self.model_name, messages=mapped,
            response_format={"type": "json_schema", "json_schema": {
                "name": output.name or "Judgment", "schema": schema, "strict": True}},
            temperature=self.config.temperature, max_tokens=self.config.max_output_tokens,
            timeout=self.config.timeout_seconds, num_retries=0,
        )
        if response.usage is None:
            raise ValueError("Provider omitted usage: reservation retained, stopping")
        cost = litellm.completion_cost(completion_response=response, model=self.model_name)
        if cost is None or not math.isfinite(cost) or cost < 0:
            raise ValueError("Response cannot be priced: reservation retained, stopping")
        choice = response.choices[0] if response.choices else None
        content = choice.message.content if choice and choice.message else None
        # Save cost and raw schema answer BEFORE Pydantic parses it.
        append_jsonl(self.calls_path, {
            "item_id": self.item_id, "attempt_id": attempt_id, "time": utc_now(),
            "requested_model": self.model_name, "response_model": response.model,
            "response_id": response.id, "usage": response.usage.model_dump(mode="json"),
            "cost_usd_estimated": cost, "reserved_usd": ceiling,
            "latency_seconds": time.monotonic() - started,
            "finish_reason": choice.finish_reason if choice else None, "raw_output": content,
        })
        self.budget.settle(attempt_id, cost)
        if cost > ceiling:
            raise ValueError("Response cost exceeded preflight estimate; inspect pricing before continuing")
        if not content or choice.finish_reason != "stop":
            raise ValueError("Missing/truncated/non-final output; no label assigned")
        # Adapter 0.2.8 constructs RunUsage here; current Pydantic-AI needs RequestUsage.
        return ModelResponse([TextPart(content=content)], model_name=response.model,
                             provider_name=self.model_name.split("/", 1)[0],
                             provider_response_id=response.id,
                             usage=RequestUsage(input_tokens=response.usage.prompt_tokens,
                                                output_tokens=response.usage.completion_tokens))


def make_judge(prompt: str, settings: Settings, budget: Budget, calls_path: Path, run_id: str, prices: dict):
    model = JudgmentModel(settings, budget, calls_path, run_id, prices)
    agent = Agent(model=model, system_prompt=prompt, output_type=NativeOutput(Judgment), retries=0,
                  name="conversation_judge")
    return agent, model


async def judge_one(agent, model, row: dict) -> Judgment:
    model.item_id = row["item_id"]
    # Positive allowlist: labels, ID, topic and alternative translations cannot leak.
    payload = {"context": row["context"], "conversation": row["conversation"]}
    result = await agent.run(json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
                             usage_limits=UsageLimits(request_limit=1))
    return result.output
