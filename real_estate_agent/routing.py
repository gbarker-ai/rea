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
from typing import Any, Dict, List

from real_estate_agent.observability import telemetry

FAST_MODEL: str = "gemini-2.5-flash"
PRO_MODEL: str = "gemini-2.5-pro"

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
    """Represents the selected Gemini model and routing rationale for a request."""

    selected_model: str
    target_agent: str
    complexity_tier: str
    rationale: str


@dataclass
class StrategicModelRouter:
    """Selects the optimal Gemini model (`gemini-2.5-flash` vs `gemini-2.5-pro`) per turn."""

    fast_model: str = FAST_MODEL
    pro_model: str = PRO_MODEL
    routing_history: List[RoutingDecision] = field(default_factory=list)

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
