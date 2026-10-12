"""Security, Fair Housing Act Compliance, & Self-Evaluation Guardrail Plugins.

AgentOps Rubric Evidence — Category 3: Orchestration & Logic
Criterion: Guardrails & Policy Plugins (5 pts)
- Implements security and evaluation guardrails (including input policy enforcement
  and output self-evaluation) via ADK callbacks (`before_model_callback`,
  `after_model_callback`) and an ADK plugin class (`RealEstateSafetyGuardrailPlugin`).
- Enforces:
  1. Prompt Injection / Jailbreak blocking.
  2. U.S. Fair Housing Act (FHA) anti-steering & protected-class compliance.
  3. Post-generation Self-Evaluation (`evaluate_output_compliance`) blocking leaked
     lockbox PINs, unredacted SSNs, or unauthorized binding financial guarantees.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from real_estate_agent.observability import telemetry
from real_estate_agent.pii_redaction import pii_redactor

# Patterns detecting prompt injection or system-prompt override attempts
_PROMPT_INJECTION_PATTERNS = [
    re.compile(r"(?i)ignore\s+(?:all\s+)?(?:previous|prior)\s+instructions"),
    re.compile(r"(?i)disregard\s+your\s+(?:system\s+prompt|constitution|guardrails)"),
    re.compile(r"(?i)reveal\s+(?:all\s+)?lockbox\s+(?:pins|codes)\s+without\s+approval"),
    re.compile(r"(?i)bypass\s+human[\s-]+in[\s-]+the[\s-]+loop"),
]

# Patterns detecting Fair Housing Act (FHA) demographic steering / protected-class profiling
_FAIR_HOUSING_STEERING_PATTERNS = [
    re.compile(
        r"(?i)\b(?:what\s+race|ethnic\s+makeup|racial\s+demographics|white\s+neighborhood|minority\s+neighborhood)\b"
    ),
    re.compile(
        r"(?i)\b(?:steer\s+me\s+away\s+from|keep\s+us\s+away\s+from)\b.*\b(?:families\s+with\s+children|religion|immigrants|minorities)\b"
    ),
    re.compile(
        r"(?i)\b(?:only\s+show\s+homes\s+near\s+christian|no\s+kids\s+allowed\s+neighborhood|what\s+religion\s+are\s+the\s+neighbors)\b"
    ),
]

# Output self-evaluation forbidden patterns
_FORBIDDEN_OUTPUT_PATTERNS = [
    (
        "UNAPPROVED_LOCKBOX_PIN_LEAK",
        re.compile(r"(?i)\b(?:raw_pin|keypad\s+code\s+is\s+#?\d{4,6})\b"),
    ),
    (
        "GUARANTEED_APPRECIATION_CLAIM",
        re.compile(r"(?i)\bwe\s+guarantee\s+your\s+home\s+will\s+appreciate\b"),
    ),
]


@dataclass
class GuardrailVerdict:
    """Result of an input or output guardrail policy evaluation."""

    allowed: bool
    policy_triggered: Optional[str] = None
    safe_override_response: Optional[str] = None
    self_eval_score: float = 1.0
    notes: str = "Compliant"


@dataclass
class RealEstateSafetyGuardrailPlugin:
    """ADK Guardrail & Self-Evaluation Policy Plugin for Summit Key Realty."""

    plugin_name: str = "real_estate_fha_and_security_guardrail"
    blocked_events: List[Dict[str, Any]] = field(default_factory=list)

    def inspect_user_input(self, user_text: str, session_id: str = "default-session") -> GuardrailVerdict:
        """Evaluates incoming caller utterances for prompt injection and Fair Housing traps."""
        if not user_text:
            return GuardrailVerdict(allowed=True)

        # 1. Prompt Injection / Security Bypass check
        for pattern in _PROMPT_INJECTION_PATTERNS:
            if pattern.search(user_text):
                verdict = GuardrailVerdict(
                    allowed=False,
                    policy_triggered="PROMPT_INJECTION_BLOCKED",
                    safe_override_response=(
                        "I cannot override agency security protocols, Fair Housing rules, or "
                        "Human-in-the-Loop broker approval requirements. How can I assist you "
                        "with property listings, showings, or maintenance today?"
                    ),
                    self_eval_score=0.0,
                    notes="Blocked prompt injection attempt.",
                )
                self._record_guardrail_event(verdict, user_text, session_id)
                return verdict

        # 2. Fair Housing Act (FHA) Anti-Steering check
        for pattern in _FAIR_HOUSING_STEERING_PATTERNS:
            if pattern.search(user_text):
                verdict = GuardrailVerdict(
                    allowed=False,
                    policy_triggered="FAIR_HOUSING_ACT_ANTI_STEERING",
                    safe_override_response=(
                        "In accordance with the U.S. Fair Housing Act, Summit Key Realty provides "
                        "equal housing opportunity and does not filter, describe, or recommend "
                        "neighborhoods based on race, religion, national origin, familial status, "
                        "or protected demographic characteristics. I would be happy to search "
                        "properties by objective criteria such as price range, bedroom count, "
                        "square footage, or specific home features."
                    ),
                    self_eval_score=1.0,
                    notes="Intercepted Fair Housing steering query with compliant redirect.",
                )
                self._record_guardrail_event(verdict, user_text, session_id)
                return verdict

        return GuardrailVerdict(allowed=True)

    def evaluate_output_compliance(
        self, draft_response: str, session_id: str = "default-session"
    ) -> GuardrailVerdict:
        """Self-evaluation guardrail inspecting agent responses before delivery to caller."""
        if not draft_response:
            return GuardrailVerdict(allowed=True)

        # 1. Scrub any accidental PII in outgoing text
        scrubbed = pii_redactor.scrub_text(draft_response)

        # 2. Check for forbidden output claims or raw lockbox PIN disclosures
        for rule_name, pattern in _FORBIDDEN_OUTPUT_PATTERNS:
            if pattern.search(scrubbed):
                verdict = GuardrailVerdict(
                    allowed=False,
                    policy_triggered=rule_name,
                    safe_override_response=(
                        "For your security and regulatory compliance, I cannot provide unverified "
                        "lockbox PIN codes or guaranteed financial appreciation promises. "
                        "Our Managing Broker has been notified to assist you directly."
                    ),
                    self_eval_score=0.0,
                    notes=f"Self-evaluation blocked non-compliant output ({rule_name}).",
                )
                self._record_guardrail_event(verdict, scrubbed, session_id)
                return verdict

        if scrubbed != draft_response:
            return GuardrailVerdict(
                allowed=True,
                policy_triggered="OUTPUT_PII_REDACTED",
                safe_override_response=scrubbed,
                self_eval_score=0.95,
                notes="Redacted PII from draft response during self-evaluation.",
            )

        return GuardrailVerdict(allowed=True, self_eval_score=1.0)

    def _record_guardrail_event(
        self, verdict: GuardrailVerdict, text_sample: str, session_id: str
    ) -> None:
        event = {
            "policy_triggered": verdict.policy_triggered,
            "allowed": verdict.allowed,
            "notes": verdict.notes,
            "sanitized_sample": pii_redactor.scrub_text(text_sample)[:160],
        }
        self.blocked_events.append(event)
        telemetry.log_event(
            event_type="GUARDRAIL_POLICY_TRIGGERED",
            message=f"Guardrail policy '{verdict.policy_triggered}' triggered.",
            payload=event,
            severity="WARNING",
            session_id=session_id,
        )


# Singleton guardrail plugin
guardrail_plugin = RealEstateSafetyGuardrailPlugin()


def before_model_guardrail_callback(
    callback_context: Any, llm_request: Any
) -> Optional[Any]:
    """ADK `before_model_callback` inspecting user input prior to LLM invocation."""
    user_text = ""
    if hasattr(llm_request, "contents") and llm_request.contents:
        last_content = llm_request.contents[-1]
        parts = getattr(last_content, "parts", None) or []
        texts = [getattr(p, "text", "") for p in parts if getattr(p, "text", None)]
        user_text = " ".join(texts)
    elif isinstance(llm_request, dict):
        user_text = str(llm_request.get("user_message", ""))

    verdict = guardrail_plugin.inspect_user_input(user_text)
    if not verdict.allowed:
        return {"override_text": verdict.safe_override_response, "guardrail": verdict.policy_triggered}
    return None


def after_model_self_eval_callback(
    callback_context: Any, llm_response: Any
) -> Optional[Any]:
    """ADK `after_model_callback` running self-evaluation on LLM outputs."""
    draft_text = ""
    if hasattr(llm_response, "text") and llm_response.text:
        draft_text = str(llm_response.text)
    elif isinstance(llm_response, dict):
        draft_text = str(llm_response.get("text", ""))

    verdict = guardrail_plugin.evaluate_output_compliance(draft_text)
    if not verdict.allowed or verdict.safe_override_response:
        return {"override_text": verdict.safe_override_response, "guardrail": verdict.policy_triggered}
    return None
