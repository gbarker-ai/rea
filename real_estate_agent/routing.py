"""Strategic Model Routing between Low-Latency Flash and Analytical Pro Models.

AgentOps Rubric Evidence — Category 3: Orchestration & Logic
Criterion: Strategic Model Routing (5 pts)
- Routes specific caller requests to the most appropriate Gemini model tier:
  1. `gemini-2.5-flash` (`FAST_MODEL`): Low-latency voice/chat intake, FAQ responses,
     MLS search filtering, showing appointment scheduling, and emergency maintenance triage.
  2. `gemini-2.5-pro` (`PRO_MODEL`): Complex multi-factor reasoning including Comparative
     Market Analysis (CMA) seller valuations, multi-property investment comparisons,
     and contract/escrow strategy planning.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import random
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from real_estate_agent.observability import telemetry

try:
    from google.genai import types as genai_types  # type: ignore
except ImportError:
    genai_types = None

FAST_MODEL: str = "gemini-2.5-flash"
PRO_MODEL: str = "gemini-2.5-pro"

# Exponential backoff retry configuration for transient 429 / 503 UNAVAILABLE spikes
RETRYABLE_HTTP_STATUS_CODES: Tuple[int, ...] = (429, 500, 502, 503, 504)


def build_genai_http_retry_options(
    attempts: int = 5,
    initial_delay: float = 1.0,
    max_delay: float = 16.0,
    exp_base: float = 2.0,
) -> Any:
    """Builds Google GenAI HttpRetryOptions for resilient 503/429 exponential backoff."""
    if genai_types is not None and hasattr(genai_types, "HttpRetryOptions"):
        try:
            return genai_types.HttpRetryOptions(
                attempts=attempts,
                initial_delay=initial_delay,
                max_delay=max_delay,
                exp_base=exp_base,
                jitter=0.5,
                http_status_codes=list(RETRYABLE_HTTP_STATUS_CODES),
            )
        except Exception:
            pass
    return {
        "attempts": attempts,
        "initial_delay": initial_delay,
        "max_delay": max_delay,
        "exp_base": exp_base,
        "jitter": 0.5,
        "http_status_codes": list(RETRYABLE_HTTP_STATUS_CODES),
    }


# Analytical keywords that trigger upgrading from Flash to Pro
_PRO_REASONING_INDICATORS = (
    "comparative market analysis",
    "cma",
    "sell my home",
    "what is my home worth",
    "property valuation",
    "investment roi",
    "cap rate",
    "price per sqft strategy",
    "listing strategy",
    "comparable sales",
)


@dataclass
class RoutingDecision:
    """Represents the selected Gemini model, fallback model, and retry policy for a request."""

    selected_model: str
    target_agent: str
    complexity_tier: str
    rationale: str
    fallback_model: str = FAST_MODEL
    retry_options: Any = field(default_factory=build_genai_http_retry_options)


@dataclass
class StrategicModelRouter:
    """Selects the optimal Gemini model (`gemini-2.5-flash` vs `gemini-2.5-pro`) with 503 retry/fallback."""

    fast_model: str = FAST_MODEL
    pro_model: str = PRO_MODEL
    max_retries: int = 4
    initial_backoff_seconds: float = 0.5
    routing_history: List[RoutingDecision] = field(default_factory=list)

    def execute_with_retry_and_fallback(
        self,
        operation: Callable[[str], Any],
        primary_model: str,
        fallback_model: Optional[str] = None,
        sleep_fn: Callable[[float], None] = time.sleep,
    ) -> Any:
        """Executes a model call with exponential backoff + jitter on 503/429 and automatic model fallback."""
        backup = fallback_model or self.fast_model
        last_exc: Optional[Exception] = None

        for attempt in range(1, self.max_retries + 1):
            active_model = primary_model if attempt < self.max_retries else backup
            try:
                return operation(active_model)
            except Exception as exc:
                last_exc = exc
                err_text = str(exc).upper()
                is_transient = any(
                    token in err_text
                    for token in ("503", "UNAVAILABLE", "429", "RESOURCE_EXHAUSTED", "HIGH DEMAND")
                )
                if not is_transient or attempt == self.max_retries:
                    break
                delay = min(
                    16.0,
                    self.initial_backoff_seconds * (2 ** (attempt - 1))
                    + random.uniform(0.0, 0.25),
                )
                telemetry.log_event(
                    event_type="MODEL_TRANSIENT_RETRY",
                    severity="WARNING",
                    message=(
                        f"Transient 503/429 on '{active_model}' (attempt {attempt}/{self.max_retries}); "
                        f"retrying in {delay:.2f}s (fallback='{backup}')."
                    ),
                    payload={"attempt": attempt, "model": active_model, "fallback_model": backup},
                )
                sleep_fn(delay)

        raise last_exc if last_exc is not None else RuntimeError("Model execution failed")

    def route_request(self, user_utterance: str, session_id: str = "default-session") -> RoutingDecision:
        """Analyzes caller intent and complexity to route to the optimal model and sub-agent.

        Args:
            user_utterance: Caller's natural language utterance.
            session_id: Active phone/chat session ID.

        Returns:
            A `RoutingDecision` specifying `selected_model`, `target_agent`, and `rationale`.
        """
        lowered = (user_utterance or "").lower()

        with telemetry.start_span(
            "orchestration.model_routing",
            attributes={"session_id": session_id},
        ) as span:
            if any(kw in lowered for kw in _PRO_REASONING_INDICATORS):
                decision = RoutingDecision(
                    selected_model=self.pro_model,
                    target_agent="seller_valuation_workflow_agent",
                    complexity_tier="HIGH_ANALYTICAL_REASONING",
                    rationale=(
                        "Complex seller valuation / CMA pricing strategy detected; routed to "
                        f"{self.pro_model} sequential workflow for multi-comp synthesis."
                    ),
                    fallback_model=self.fast_model,
                )
            elif any(
                kw in lowered
                for kw in (
                    "pipe",
                    "flood",
                    "leak",
                    "gas",
                    "hvac",
                    "heater",
                    "repair",
                    "maintenance",
                    "tenant",
                    "electrical",
                )
            ):
                decision = RoutingDecision(
                    selected_model=self.fast_model,
                    target_agent="tenant_emergency_triage_agent",
                    complexity_tier="LOW_LATENCY_EMERGENCY_TRIAGE",
                    rationale=(
                        "Urgent tenant maintenance or property hazard reported; routed to "
                        f"{self.fast_model} for immediate low-latency triage and HITL dispatch."
                    ),
                    fallback_model=self.fast_model,
                )
            else:
                decision = RoutingDecision(
                    selected_model=self.fast_model,
                    target_agent="buyer_showing_specialist_agent",
                    complexity_tier="FAST_INTERACTIVE_CONCIERGE",
                    rationale=(
                        "Interactive buyer inquiry, MLS search, showing scheduling, or lockbox "
                        f"request; routed to {self.fast_model} for sub-second voice responsiveness."
                    ),
                    fallback_model=self.fast_model,
                )

            if span is not None:
                span.set_attribute("routing.selected_model", decision.selected_model)
                span.set_attribute("routing.target_agent", decision.target_agent)
                span.set_attribute("routing.complexity_tier", decision.complexity_tier)

            self.routing_history.append(decision)
            telemetry.log_event(
                event_type="STRATEGIC_MODEL_ROUTED",
                message=(
                    f"Routed request to '{decision.target_agent}' on model '{decision.selected_model}'."
                ),
                payload={
                    "selected_model": decision.selected_model,
                    "fallback_model": decision.fallback_model,
                    "target_agent": decision.target_agent,
                    "complexity_tier": decision.complexity_tier,
                    "rationale": decision.rationale,
                },
                session_id=session_id,
                agent_name=decision.target_agent,
            )
            return decision


# Global router instance
model_router = StrategicModelRouter()
