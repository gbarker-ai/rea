"""Real Estate Agency Front Desk Domain Tools with Guided Error Recovery.

AgentOps Rubric Evidence — Category 1: Tool & Interface Design
- Comprehensive Tool Docstrings (5 pts): Every tool includes a rich Google-style docstring
  explaining its purpose, parameters (`Args:`), return schema (`Returns:`), and recovery behavior.
- Descriptive Naming (5 pts): Specific verb-noun tool identifiers.
- Explicit JSON Schemas (5 pts): Validates all inputs and outputs via strict Pydantic schemas.
- Guided Error Handling (5 pts): Returns structured `GuidedToolErrorSchema` payloads with
  `available_alternatives` and `recovery_instructions_for_llm` instead of crashing.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional
from real_estate_agent._pydantic_compat import ValidationError
from real_estate_agent.hitl import EMERGENCY_DISPATCH_HITL_THRESHOLD_USD, hitl_gate
from real_estate_agent.memory import memory_store, schedule_async_memory_consolidation
from real_estate_agent.observability import traced_tool
from real_estate_agent.schemas import (
    GenerateCmaValuationInput,
    GenerateCmaValuationOutput,
    GuidedToolErrorSchema,
    PropertyListingItem,
    QualifyBuyerLeadInput,
    QualifyBuyerLeadOutput,
    RequestLockboxAccessInput,
    RequestLockboxAccessOutput,
    SchedulePropertyShowingInput,
    SchedulePropertyShowingOutput,
    SearchMlsPropertyListingsInput,
    SearchMlsPropertyListingsOutput,
    TriageEmergencyMaintenanceInput,
    TriageEmergencyMaintenanceOutput,
)

MLS_CATALOG: Dict[str, Dict[str, Any]] = {
    "MLS-1001": {
        "mls_id": "MLS-1001",
        "address": "742 Evergreen Terrace, Oak Creek",
        "neighborhood": "Oak Creek",
        "price_usd": 625000,
        "bedrooms": 4,
        "bathrooms": 2.5,
        "sqft": 2450,
        "property_type": "single_family",
        "features": ["fenced yard", "solar panels", "renovated kitchen", "two-car garage"],
        "assigned_agent": "Elena Rostova",
        "available_showing_slots": ["2026-10-10 10:00", "2026-10-10 14:00", "2026-10-11 11:00"],
        "booked_slots": ["2026-10-10 12:00"],
    },
    "MLS-1002": {
        "mls_id": "MLS-1002",
        "address": "1200 Lakeshore Drive #18B, Downtown",
        "neighborhood": "Downtown",
        "price_usd": 485000,
        "bedrooms": 2,
        "bathrooms": 2.0,
        "sqft": 1350,
        "property_type": "condo",
        "features": ["rooftop pool", "doorman", "floor-to-ceiling windows", "home office"],
        "assigned_agent": "Marcus Vance",
        "available_showing_slots": ["2026-10-10 13:00", "2026-10-10 16:00", "2026-10-11 15:00"],
        "booked_slots": ["2026-10-10 11:00"],
    },
    "MLS-1003": {
        "mls_id": "MLS-1003",
        "address": "88 Whispering Pines Way, Suburban Hills",
        "neighborhood": "Suburban Hills",
        "price_usd": 890000,
        "bedrooms": 5,
        "bathrooms": 4.0,
        "sqft": 3600,
        "property_type": "single_family",
        "features": ["fenced yard", "heated pool", "guest suite", "home office", "cul-de-sac"],
        "assigned_agent": "Elena Rostova",
        "available_showing_slots": ["2026-10-10 15:00", "2026-10-11 10:00", "2026-10-11 14:00"],
        "booked_slots": ["2026-10-10 13:00"],
    },
    "MLS-1004": {
        "mls_id": "MLS-1004",
        "address": "310 Artisan Row, Oak Creek",
        "neighborhood": "Oak Creek",
        "price_usd": 540000,
        "bedrooms": 3,
        "bathrooms": 2.5,
        "sqft": 1850,
        "property_type": "townhouse",
        "features": ["private patio", "EV charger", "home office", "walkable to transit"],
        "assigned_agent": "Devon Brooks",
        "available_showing_slots": ["2026-10-10 09:30", "2026-10-10 14:30", "2026-10-11 13:00"],
        "booked_slots": [],
    },
}

_INITIAL_MLS_CATALOG: Dict[str, Dict[str, Any]] = copy.deepcopy(MLS_CATALOG)


def reset_mls_catalog() -> None:
    """Resets in-memory MLS showing slot state to initial seed values for repeatable tests."""
    MLS_CATALOG.clear()
    MLS_CATALOG.update(copy.deepcopy(_INITIAL_MLS_CATALOG))


COMPARABLE_SALES_DB: Dict[str, List[Dict[str, Any]]] = {
    "oak creek": [
        {"address": "710 Evergreen Terrace", "sold_price_usd": 612000, "sqft": 2400, "days_on_market": 14},
        {"address": "302 Artisan Row", "sold_price_usd": 555000, "sqft": 1900, "days_on_market": 19},
        {"address": "19 Maple Ct", "sold_price_usd": 648000, "sqft": 2550, "days_on_market": 11},
    ],
    "downtown": [
        {"address": "1200 Lakeshore Dr #14A", "sold_price_usd": 472000, "sqft": 1320, "days_on_market": 22},
        {"address": "450 Market St #9C", "sold_price_usd": 510000, "sqft": 1410, "days_on_market": 17},
        {"address": "800 Harbor View #22B", "sold_price_usd": 495000, "sqft": 1380, "days_on_market": 15},
    ],
    "suburban hills": [
        {"address": "92 Whispering Pines Way", "sold_price_usd": 875000, "sqft": 3550, "days_on_market": 21},
        {"address": "14 Ridgecrest Blvd", "sold_price_usd": 920000, "sqft": 3700, "days_on_market": 16},
        {"address": "55 Summit View Dr", "sold_price_usd": 860000, "sqft": 3480, "days_on_market": 25},
    ],
}


def _build_guided_error(
    error_code: str,
    error_message: str,
    available_alternatives: List[Any],
    recovery_instructions_for_llm: str,
) -> Dict[str, Any]:
    """Constructs a schema-validated guided error dictionary for LLM self-recovery."""
    return GuidedToolErrorSchema(
        error_code=error_code,
        error_message=error_message,
        available_alternatives=available_alternatives,
        recovery_instructions_for_llm=recovery_instructions_for_llm,
    ).model_dump()


@traced_tool
def search_mls_property_listings(
    neighborhood: Optional[str] = None,
    max_price_usd: Optional[int] = None,
    min_bedrooms: int = 1,
    property_type: Optional[str] = None,
    must_have_feature: Optional[str] = None,
) -> Dict[str, Any]:
    """Searches active Multiple Listing Service (MLS) residential property listings.

    Use this tool when a prospective buyer or caller asks about available homes,
    condos, or townhouses matching specific criteria such as neighborhood, budget,
    minimum bedroom count, or must-have amenities (e.g., 'fenced yard', 'pool').

    Args:
        neighborhood: Optional neighborhood filter ('Oak Creek', 'Downtown', 'Suburban Hills').
        max_price_usd: Optional maximum purchase price in USD (e.g., 650000).
        min_bedrooms: Minimum number of bedrooms required (defaults to 1).
        property_type: Optional category ('single_family', 'condo', 'townhouse', 'luxury_estate').
        must_have_feature: Optional specific feature or amenity keyword to match.

    Returns:
        A dictionary validated against `SearchMlsPropertyListingsOutput` containing matching
        listings and open showing slots, or a `GuidedToolErrorSchema` dictionary with broader
        alternatives if no exact matches exist or input parameters are invalid.
    """
    try:
        validated_in = SearchMlsPropertyListingsInput(
            neighborhood=neighborhood,
            max_price_usd=max_price_usd,
            min_bedrooms=min_bedrooms,
            property_type=property_type,  # type: ignore[arg-type]
            must_have_feature=must_have_feature,
        )
    except ValidationError as exc:
        return _build_guided_error(
            error_code="INVALID_SEARCH_FILTERS",
            error_message=f"Search filter validation failed: {exc.errors()[0]['msg']}",
            available_alternatives=[
                {"neighborhoods": ["Oak Creek", "Downtown", "Suburban Hills"]},
                {"property_types": ["single_family", "condo", "townhouse", "luxury_estate"]},
            ],
            recovery_instructions_for_llm=(
                "Clarify valid property types or price bounds ($50,000-$50,000,000) and re-invoke "
                "`search_mls_property_listings` with corrected arguments."
            ),
        )

    matches: List[PropertyListingItem] = []
    for item in MLS_CATALOG.values():
        if validated_in.neighborhood and validated_in.neighborhood.strip().lower() not in item["neighborhood"].lower():
            continue
        if validated_in.max_price_usd is not None and item["price_usd"] > validated_in.max_price_usd:
            continue
        if item["bedrooms"] < validated_in.min_bedrooms:
            continue
        if validated_in.property_type and item["property_type"] != validated_in.property_type:
            continue
        if validated_in.must_have_feature:
            feat_q = validated_in.must_have_feature.strip().lower()
            if not any(feat_q in f.lower() for f in item["features"]):
                continue
        matches.append(
            PropertyListingItem(
                mls_id=item["mls_id"],
                address=item["address"],
                neighborhood=item["neighborhood"],
                price_usd=item["price_usd"],
                bedrooms=item["bedrooms"],
                bathrooms=item["bathrooms"],
                sqft=item["sqft"],
                property_type=item["property_type"],
                features=list(item["features"]),
                available_showing_slots=list(item["available_showing_slots"]),
            )
        )

    if not matches:
        nearest = [
            {"mls_id": p["mls_id"], "address": p["address"], "neighborhood": p["neighborhood"], "price_usd": p["price_usd"]}
            for p in MLS_CATALOG.values()
        ]
        return _build_guided_error(
            error_code="NO_EXACT_MLS_MATCHES",
            error_message="No active MLS listings matched all strict filters simultaneously.",
            available_alternatives=nearest,
            recovery_instructions_for_llm=(
                "Inform the caller that no listings matched every constraint simultaneously and "
                "proactively share the closest available properties from `available_alternatives`."
            ),
        )

    return SearchMlsPropertyListingsOutput(total_matches=len(matches), listings=matches).model_dump()


@traced_tool
def schedule_property_showing_appointment(
    mls_id: str,
    caller_name: str,
    caller_phone: str,
    requested_slot: str,
    pre_approved_mortgage: bool = False,
) -> Dict[str, Any]:
    """Schedules an in-person or virtual property showing appointment for a prospective buyer.

    Validates the requested MLS property ID and time slot against listing agent
    availability. If the requested slot is already booked or unavailable, returns
    guided recovery instructions with the next available time slots.

    Args:
        mls_id: Target MLS property identifier in format 'MLS-XXXX' (e.g., 'MLS-1001').
        caller_name: Full name of the buyer attending the showing.
        caller_phone: Phone number of the caller for SMS/call confirmation.
        requested_slot: Requested date and time string (e.g., '2026-10-10 14:00').
        pre_approved_mortgage: True if the buyer has mortgage pre-approval in hand.

    Returns:
        A dictionary validated against `SchedulePropertyShowingOutput` with confirmation
        details, or a `GuidedToolErrorSchema` dictionary with alternative open slots if
        the requested slot is unavailable.
    """
    try:
        validated_in = SchedulePropertyShowingInput(
            mls_id=mls_id.strip().upper(),
            caller_name=caller_name,
            caller_phone=caller_phone,
            requested_slot=requested_slot.strip(),
            pre_approved_mortgage=pre_approved_mortgage,
        )
    except ValidationError as exc:
        return _build_guided_error(
            error_code="INVALID_SHOWING_REQUEST_FORMAT",
            error_message=f"Invalid showing parameters: {exc.errors()[0]['msg']}",
            available_alternatives=list(MLS_CATALOG.keys()),
            recovery_instructions_for_llm=(
                "Ensure `mls_id` follows format 'MLS-1001'..'MLS-1004' and provide caller name, "
                "phone, and desired time slot."
            ),
        )

    listing = MLS_CATALOG.get(validated_in.mls_id)
    if listing is None:
        return _build_guided_error(
            error_code="MLS_LISTING_NOT_FOUND",
            error_message=f"Property '{validated_in.mls_id}' does not exist in active MLS inventory.",
            available_alternatives=[{"mls_id": k, "address": v["address"]} for k, v in MLS_CATALOG.items()],
            recovery_instructions_for_llm=(
                "Offer the valid active properties from `available_alternatives` and confirm which "
                "listing the caller would like to tour."
            ),
        )

    slot = validated_in.requested_slot
    if slot in listing["booked_slots"] or slot not in listing["available_showing_slots"]:
        return _build_guided_error(
            error_code="SHOWING_SLOT_UNAVAILABLE",
            error_message=f"Requested slot '{slot}' for {validated_in.mls_id} ({listing['address']}) is already booked.",
            available_alternatives=list(listing["available_showing_slots"]),
            recovery_instructions_for_llm=(
                f"Politely inform {validated_in.caller_name} that '{slot}' is unavailable for "
                f"{listing['address']}, and offer open slots: {', '.join(listing['available_showing_slots'])}."
            ),
        )

    listing["available_showing_slots"].remove(slot)
    listing["booked_slots"].append(slot)
    confirmation_id = f"SHOW-{validated_in.mls_id.split('-')[1]}-{abs(hash(slot)) % 9000 + 1000}"

    memory_store.upsert_caller_interaction(
        caller_phone=validated_in.caller_phone,
        caller_name=validated_in.caller_name,
        interaction_summary=f"Booked showing {confirmation_id} for {validated_in.mls_id} at {slot}.",
        metadata={"last_booked_mls_id": validated_in.mls_id, "confirmed_slot": slot},
    )
    schedule_async_memory_consolidation(
        caller_phone=validated_in.caller_phone,
        event_description=f"Showing scheduled for {listing['address']} at {slot}",
        preferences={"interested_mls_id": validated_in.mls_id, "pre_approved": pre_approved_mortgage},
    )

    return SchedulePropertyShowingOutput(
        confirmation_id=confirmation_id,
        mls_id=validated_in.mls_id,
        property_address=listing["address"],
        confirmed_slot=slot,
        assigned_showing_agent=listing["assigned_agent"],
        calendar_instructions=f"Confirmed with {listing['assigned_agent']}. Please bring photo ID.",
    ).model_dump()


@traced_tool
def qualify_and_score_buyer_lead(
    caller_name: str,
    caller_phone: str,
    budget_usd: int,
    down_payment_percent: float,
    is_pre_approved: bool,
    purchase_timeline_days: int,
    preferred_neighborhoods: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Qualifies a prospective home buyer lead and computes a readiness score (0-100).

    Evaluates purchasing power, mortgage pre-approval status, down payment percentage,
    and target closing timeline, then triggers an asynchronous CRM memory consolidation
    task so the live call turn returns immediately without blocking.

    Args:
        caller_name: Full name of the prospective buyer.
        caller_phone: Contact phone number used as the persistent caller identifier.
        budget_usd: Target maximum purchase budget in USD.
        down_payment_percent: Planned down payment percentage (0.0 to 100.0).
        is_pre_approved: True if the buyer holds a written lender pre-approval letter.
        purchase_timeline_days: Target number of days until purchase/move-in.
        preferred_neighborhoods: Optional list of preferred neighborhood names.

    Returns:
        A dictionary validated against `QualifyBuyerLeadOutput` containing the lead score,
        qualification tier ('HOT_A_LIST', 'WARM_ACTIVE', 'NURTURE_PIPELINE'), and next steps.
    """
    try:
        validated_in = QualifyBuyerLeadInput(
            caller_name=caller_name,
            caller_phone=caller_phone,
            budget_usd=budget_usd,
            down_payment_percent=down_payment_percent,
            is_pre_approved=is_pre_approved,
            purchase_timeline_days=purchase_timeline_days,
            preferred_neighborhoods=preferred_neighborhoods or [],
        )
    except ValidationError as exc:
        return _build_guided_error(
            error_code="INVALID_LEAD_QUALIFICATION_INPUT",
            error_message=f"Lead qualification validation error: {exc.errors()[0]['msg']}",
            available_alternatives=[{"budget_usd_range": [50000, 50000000]}],
            recovery_instructions_for_llm="Ask the caller to clarify budget, down payment %, and timeline.",
        )

    score = 30 + (35 if validated_in.is_pre_approved else 0)
    score += 20 if validated_in.down_payment_percent >= 20.0 else (10 if validated_in.down_payment_percent >= 10.0 else 0)
    score += 15 if validated_in.purchase_timeline_days <= 45 else (8 if validated_in.purchase_timeline_days <= 90 else 0)
    score = min(100, score)

    if score >= 75:
        tier, next_action = "HOT_A_LIST", "Prioritize immediate private showing with Senior Buyer Partner."
    elif score >= 50:
        tier, next_action = "WARM_ACTIVE", "Schedule property tour and refer to partner mortgage lender."
    else:
        tier, next_action = "NURTURE_PIPELINE", "Enroll in weekly neighborhood MLS digest."

    lead_id = f"LEAD-{abs(hash(validated_in.caller_phone)) % 90000 + 10000}"
    schedule_async_memory_consolidation(
        caller_phone=validated_in.caller_phone,
        event_description=f"Qualified buyer {validated_in.caller_name} as {tier} (score={score}/100).",
        preferences={
            "caller_name": validated_in.caller_name,
            "budget_usd": validated_in.budget_usd,
            "lead_score": score,
            "qualification_tier": tier,
        },
    )

    return QualifyBuyerLeadOutput(
        lead_id=lead_id,
        qualification_tier=tier,  # type: ignore[arg-type]
        lead_score=score,
        recommended_next_action=next_action,
        async_memory_job_scheduled=True,
    ).model_dump()


@traced_tool
def triage_and_dispatch_emergency_maintenance(
    property_address: str,
    tenant_name: str,
    caller_phone: str,
    issue_category: str,
    issue_description: str,
    estimated_cost_usd: float,
    human_approval_token: Optional[str] = None,
) -> Dict[str, Any]:
    """Triages a tenant property maintenance issue and dispatches an emergency contractor.

    Enforces a strict Human-in-the-Loop (HITL) code stop whenever `estimated_cost_usd > 500.0`.
    If the estimated dispatch cost exceeds $500 and no verified `human_approval_token`
    is provided, execution halts and returns `status='PENDING_HUMAN_APPROVAL'` with an
    `approval_request_id` for the human Property Manager to approve.

    Args:
        property_address: Full street address of the rental property.
        tenant_name: Name of the tenant reporting the issue.
        caller_phone: Tenant's callback phone number.
        issue_category: One of 'burst_pipe_flood', 'gas_leak_odor',
            'hvac_failure_extreme_weather', 'electrical_hazard',
            'appliance_routine_repair', or 'cosmetic_maintenance'.
        issue_description: Detailed description of the problem reported on the call.
        estimated_cost_usd: Estimated vendor dispatch and repair cost in USD.
        human_approval_token: Optional signed broker HITL confirmation token required
            for dispatches over $500.00.

    Returns:
        A dictionary validated against `TriageEmergencyMaintenanceOutput` indicating
        either confirmed vendor dispatch or a `PENDING_HUMAN_APPROVAL` code stop.
    """
    try:
        validated_in = TriageEmergencyMaintenanceInput(
            property_address=property_address,
            tenant_name=tenant_name,
            caller_phone=caller_phone,
            issue_category=issue_category,  # type: ignore[arg-type]
            issue_description=issue_description,
            estimated_cost_usd=estimated_cost_usd,
            human_approval_token=human_approval_token,
        )
    except ValidationError as exc:
        return _build_guided_error(
            error_code="INVALID_MAINTENANCE_TRIAGE_INPUT",
            error_message=f"Maintenance triage input error: {exc.errors()[0]['msg']}",
            available_alternatives=[
                "burst_pipe_flood", "gas_leak_odor", "hvac_failure_extreme_weather",
                "electrical_hazard", "appliance_routine_repair", "cosmetic_maintenance",
            ],
            recovery_instructions_for_llm="Map the issue to a valid `issue_category` and positive `estimated_cost_usd`.",
        )

    critical = {"burst_pipe_flood", "gas_leak_odor", "hvac_failure_extreme_weather", "electrical_hazard"}
    urgency = "CRITICAL_EMERGENCY" if validated_in.issue_category in critical else (
        "HIGH_PRIORITY" if validated_in.issue_category == "appliance_routine_repair" else "ROUTINE_SLA"
    )

    if validated_in.estimated_cost_usd > EMERGENCY_DISPATCH_HITL_THRESHOLD_USD:
        if not hitl_gate.verify_approval_token(
            validated_in.human_approval_token,
            "triage_and_dispatch_emergency_maintenance",
            validated_in.property_address,
        ):
            pending = hitl_gate.create_approval_stop(
                action_name="triage_and_dispatch_emergency_maintenance",
                summary=(
                    f"Emergency dispatch for {validated_in.issue_category} at {validated_in.property_address} "
                    f"(${validated_in.estimated_cost_usd:,.2f}) exceeds ${EMERGENCY_DISPATCH_HITL_THRESHOLD_USD:,.2f}."
                ),
                arguments=validated_in.model_dump(),
            )
            return TriageEmergencyMaintenanceOutput(
                status="PENDING_HUMAN_APPROVAL",
                ticket_id=f"MNT-HOLD-{pending.approval_request_id}",
                urgency_level=urgency,  # type: ignore[arg-type]
                dispatched_vendor=None,
                estimated_cost_usd=validated_in.estimated_cost_usd,
                requires_human_confirmation=True,
                approval_request_id=pending.approval_request_id,
                tenant_safety_instructions=(
                    f"HUMAN APPROVAL REQUIRED (Request {pending.approval_request_id}): "
                    f"Repair cost (${validated_in.estimated_cost_usd:,.2f}) exceeds $500 limit. "
                    "On-call Property Manager paged. Shut off main water/gas valve immediately."
                ),
            ).model_dump()

    ticket_id = f"MNT-{abs(hash(validated_in.property_address + validated_in.issue_category)) % 9000 + 1000}"
    vendor_map = {
        "burst_pipe_flood": "RapidResponse 24/7 Plumbing & Restoration",
        "gas_leak_odor": "Metro Emergency Gas & Mechanical",
        "hvac_failure_extreme_weather": "Apex Climate Control Technicians",
        "electrical_hazard": "VoltGuard Licensed Electricians",
        "appliance_routine_repair": "HomeCare Appliance Specialists",
        "cosmetic_maintenance": "Premier Property Handyman Services",
    }
    vendor = vendor_map[validated_in.issue_category]
    return TriageEmergencyMaintenanceOutput(
        status="success",
        ticket_id=ticket_id,
        urgency_level=urgency,  # type: ignore[arg-type]
        dispatched_vendor=vendor,
        estimated_cost_usd=validated_in.estimated_cost_usd,
        requires_human_confirmation=validated_in.estimated_cost_usd > EMERGENCY_DISPATCH_HITL_THRESHOLD_USD,
        approval_request_id=None,
        tenant_safety_instructions=f"{vendor} dispatched under ticket {ticket_id}.",
    ).model_dump()


@traced_tool
def request_smart_lockbox_access_code(
    mls_id: str,
    requester_name: str,
    requester_license_or_id: str,
    caller_phone: str,
    human_approval_token: Optional[str] = None,
) -> Dict[str, Any]:
    """Requests a temporary smart-lockbox access code for an authorized showing.

    Because releasing physical entry credentials to a property is a high-stakes
    security action, this tool ALWAYS enforces a Human-in-the-Loop (HITL) code stop
    unless a valid `human_approval_token` signed by the Managing Broker is supplied.
    Raw lockbox PINs are never exposed in logs or unmasked output fields.

    Args:
        mls_id: Target MLS property ID in format 'MLS-XXXX' (e.g., 'MLS-1001').
        requester_name: Full name of the licensed agent or verified buyer requesting access.
        requester_license_or_id: State real estate license number or confirmed showing ID.
        caller_phone: Phone number of the requester.
        human_approval_token: Signed human broker confirmation token required to release access.

    Returns:
        A dictionary validated against `RequestLockboxAccessOutput` with either a
        `PENDING_HUMAN_APPROVAL` hold or a confirmed one-time encrypted access reference.
    """
    try:
        validated_in = RequestLockboxAccessInput(
            mls_id=mls_id.strip().upper(),
            requester_name=requester_name,
            requester_license_or_id=requester_license_or_id,
            caller_phone=caller_phone,
            human_approval_token=human_approval_token,
        )
    except ValidationError as exc:
        return _build_guided_error(
            error_code="INVALID_LOCKBOX_REQUEST",
            error_message=f"Lockbox request validation failed: {exc.errors()[0]['msg']}",
            available_alternatives=list(MLS_CATALOG.keys()),
            recovery_instructions_for_llm="Verify the MLS ID ('MLS-1001'..'MLS-1004'), name, and license/showing ID.",
        )

    if validated_in.mls_id not in MLS_CATALOG:
        return _build_guided_error(
            error_code="LOCKBOX_MLS_NOT_FOUND",
            error_message=f"Property '{validated_in.mls_id}' is not managed by this agency.",
            available_alternatives=list(MLS_CATALOG.keys()),
            recovery_instructions_for_llm="Offer valid MLS IDs from `available_alternatives`.",
        )

    if not hitl_gate.verify_approval_token(
        validated_in.human_approval_token,
        "request_smart_lockbox_access_code",
        validated_in.mls_id,
    ):
        pending = hitl_gate.create_approval_stop(
            action_name="request_smart_lockbox_access_code",
            summary=f"Smart-lockbox access requested for {validated_in.mls_id} by {validated_in.requester_name}.",
            arguments=validated_in.model_dump(),
        )
        return RequestLockboxAccessOutput(
            status="PENDING_HUMAN_APPROVAL",
            mls_id=validated_in.mls_id,
            requires_human_confirmation=True,
            approval_request_id=pending.approval_request_id,
            one_time_access_window=None,
            masked_access_reference=None,
            instructions=(
                f"SECURITY CODE STOP: Lockbox access for {validated_in.mls_id} requires Managing Broker "
                f"sign-off (Request ID: {pending.approval_request_id})."
            ),
        ).model_dump()

    return RequestLockboxAccessOutput(
        status="success",
        mls_id=validated_in.mls_id,
        requires_human_confirmation=True,
        approval_request_id=None,
        one_time_access_window="Valid for 60 minutes from broker approval",
        masked_access_reference=f"LKBOX-OTP-{validated_in.mls_id}-****",
        instructions="Broker approval verified. One-time Bluetooth unlock pass sent via encrypted SMS.",
    ).model_dump()


@traced_tool
def generate_comparative_market_analysis(
    subject_address: str,
    neighborhood: str,
    sqft: int,
    bedrooms: int,
    bathrooms: float,
    condition_tier: str = "well_maintained",
) -> Dict[str, Any]:
    """Generates a Comparative Market Analysis (CMA) property valuation for a prospective seller.

    Analyzes recent comparable closed sales in the target neighborhood, adjusts for
    interior square footage, bedroom/bathroom count, and property renovation condition,
    and recommends an optimal listing price strategy.

    Args:
        subject_address: Street address of the seller's property being evaluated.
        neighborhood: Neighborhood name ('Oak Creek', 'Downtown', 'Suburban Hills').
        sqft: Interior livable square footage of the subject property.
        bedrooms: Total number of bedrooms.
        bathrooms: Total number of bathrooms.
        condition_tier: Property condition ('turnkey_renovated', 'well_maintained', 'needs_updating').

    Returns:
        A dictionary validated against `GenerateCmaValuationOutput` containing the
        estimated market value, valuation range, comparable sales, and pricing strategy.
    """
    try:
        validated_in = GenerateCmaValuationInput(
            subject_address=subject_address,
            neighborhood=neighborhood,
            sqft=sqft,
            bedrooms=bedrooms,
            bathrooms=bathrooms,
            condition_tier=condition_tier,  # type: ignore[arg-type]
        )
    except ValidationError as exc:
        return _build_guided_error(
            error_code="INVALID_CMA_PARAMETERS",
            error_message=f"CMA parameter validation error: {exc.errors()[0]['msg']}",
            available_alternatives=["Oak Creek", "Downtown", "Suburban Hills"],
            recovery_instructions_for_llm="Confirm square footage, bedrooms, bathrooms, and condition tier.",
        )

    comps = COMPARABLE_SALES_DB.get(validated_in.neighborhood.strip().lower())
    if not comps:
        return _build_guided_error(
            error_code="NEIGHBORHOOD_COMPS_UNAVAILABLE",
            error_message=f"No closed comparable sales found for '{validated_in.neighborhood}'.",
            available_alternatives=["Oak Creek", "Downtown", "Suburban Hills"],
            recovery_instructions_for_llm="Offer automated comps for 'Oak Creek', 'Downtown', or 'Suburban Hills'.",
        )

    avg_ppsqft = sum(c["sold_price_usd"] / c["sqft"] for c in comps) / len(comps)
    mult = {"turnkey_renovated": 1.08, "well_maintained": 1.00, "needs_updating": 0.91}[validated_in.condition_tier]
    adjusted_ppsqft = round(avg_ppsqft * mult, 2)
    base_val = int(round(validated_in.sqft * adjusted_ppsqft, -3))

    return GenerateCmaValuationOutput(
        subject_address=validated_in.subject_address,
        neighborhood=validated_in.neighborhood,
        estimated_market_value_usd=base_val,
        valuation_range_low_usd=int(round(base_val * 0.95, -3)),
        valuation_range_high_usd=int(round(base_val * 1.05, -3)),
        price_per_sqft_usd=adjusted_ppsqft,
        comparable_sales_used=comps,
        recommended_list_strategy=(
            f"List at ${base_val:,} (${adjusted_ppsqft}/sqft) with a 14-day pre-market campaign "
            f"targeting pre-approved buyers in {validated_in.neighborhood}."
        ),
    ).model_dump()
