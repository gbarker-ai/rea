"""Interactive Walkthrough & Live Demo Runner for the Real Estate Front Desk Agent.

Run with:
    .venv/bin/python demo.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from real_estate_agent.agent import call_orchestrator
from real_estate_agent.hitl import hitl_gate
from real_estate_agent.observability import telemetry


def run_demo() -> None:
    """Executes all 8 golden phone scenarios and prints an AgentOps telemetry summary."""
    golden_path = Path(__file__).parent / "tests" / "golden_dataset.json"
    scenarios: List[Dict[str, Any]] = json.loads(golden_path.read_text(encoding="utf-8"))

    for item in scenarios:
        result = call_orchestrator.handle_incoming_call_turn(
            caller_phone=item["caller_phone"],
            caller_utterance=item["caller_utterance"],
            caller_name=item["caller_name"],
            tool_overrides=item.get("tool_overrides", {}),
        )
        telemetry.log_event(
            event_type="DEMO_SCENARIO_COMPLETED",
            message=f"Completed {item['scenario_id']} -> status={result['status']}",
            payload={
                "scenario_id": item["scenario_id"],
                "selected_model": result["selected_model"],
                "delegated_agent": result["delegated_agent"],
                "status": result["status"],
            },
        )

    # Demonstrate completing a Human-in-the-Loop approval cycle
    if hitl_gate.pending_requests:
        first_req = next(iter(hitl_gate.pending_requests.values()))
        token = hitl_gate.approve_request(
            approval_request_id=first_req.approval_request_id,
            broker_id="BROKER_ELENA_ROSTOVA",
            reference_id="742 Evergreen Terrace, Oak Creek",
        )
        call_orchestrator.handle_incoming_call_turn(
            caller_phone="555-876-5432",
            caller_utterance="Dispatch emergency plumber now that broker approved.",
            caller_name="David Miller",
            tool_overrides={
                "property_address": "742 Evergreen Terrace, Oak Creek",
                "issue_category": "burst_pipe_flood",
                "estimated_cost_usd": 850.0,
                "human_approval_token": token,
            },
        )


if __name__ == "__main__":
    run_demo()
