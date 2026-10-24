"""Environment configuration shared by local and container entry points."""
from dataclasses import dataclass, replace
from pathlib import Path
import math
import os

from dotenv import load_dotenv
from .common import SolutionError

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Settings:
    model: str
    max_output_tokens: int = 2048
    timeout_seconds: float = 120.0
    temperature: float = 0.0

    def __post_init__(self):
        if not self.model.strip() or self.max_output_tokens <= 0:
            raise SolutionError("Model and positive output token limit are required")
        if not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0:
            raise SolutionError("Timeout must be finite and positive")


@dataclass(frozen=True)
class Config:
    inference: Settings
    analysis: Settings
    synthesis: Settings
    explanation: Settings
    budget_usd: float = 10.0
    concurrency: int = 3
    request_interval_seconds: float = 4.1
    max_attempts: int = 4
    retry_backoff_seconds: float = 5.0

    def __post_init__(self):
        if not math.isfinite(self.budget_usd) or self.budget_usd <= 0:
            raise SolutionError("RUN_BUDGET_USD must be finite and positive")
        if not 1 <= self.concurrency <= 8:
            raise SolutionError("MAX_CONCURRENCY must be between 1 and 8")
        if not math.isfinite(self.request_interval_seconds) or self.request_interval_seconds < 0:
            raise SolutionError("REQUEST_INTERVAL_SECONDS must be finite and nonnegative")
        if not 1 <= self.max_attempts <= 8:
            raise SolutionError("MAX_ATTEMPTS must be between 1 and 8")
        if not math.isfinite(self.retry_backoff_seconds) or self.retry_backoff_seconds < 0:
            raise SolutionError("RETRY_BACKOFF_SECONDS must be finite and nonnegative")

    @classmethod
    def from_env(cls):
        # Process > solution/.env > repository .env > workspace ../.env.
        for path in (ROOT / "solution/.env", ROOT / ".env", ROOT.parent / ".env"):
            load_dotenv(path, override=False)
        inference = Settings(
            model=os.getenv("MODEL", "gemini/gemini-flash-latest"),
            max_output_tokens=int(os.getenv("MAX_OUTPUT_TOKENS", "2048")),
            timeout_seconds=float(os.getenv("REQUEST_TIMEOUT_SECONDS", "120")),
        )
        analysis = replace(inference, model=os.getenv("ANALYSIS_MODEL", inference.model),
                           max_output_tokens=int(os.getenv("ANALYSIS_MAX_OUTPUT_TOKENS", "3072")))
        # One name for the stronger model that learns the prompts; per-stage overrides stay available.
        meta = os.getenv("META_MODEL", "gemini/gemini-4-argon")
        synthesis = replace(inference, model=os.getenv("SYNTHESIS_MODEL", meta),
                            max_output_tokens=int(os.getenv("SYNTHESIS_MAX_OUTPUT_TOKENS", "32768")),
                            timeout_seconds=float(os.getenv("SYNTHESIS_TIMEOUT_SECONDS", "900")))
        explanation = replace(synthesis, model=os.getenv("EXPLANATION_MODEL", synthesis.model),
                              max_output_tokens=int(os.getenv("EXPLANATION_MAX_OUTPUT_TOKENS", "16384")))
        return cls(inference, analysis, synthesis, explanation,
                   budget_usd=float(os.getenv("RUN_BUDGET_USD", "10")),
                   concurrency=int(os.getenv("MAX_CONCURRENCY", "3")),
                   request_interval_seconds=float(os.getenv("REQUEST_INTERVAL_SECONDS", "4.1")),
                   max_attempts=int(os.getenv("MAX_ATTEMPTS", "4")),
                   retry_backoff_seconds=float(os.getenv("RETRY_BACKOFF_SECONDS", "5")))
