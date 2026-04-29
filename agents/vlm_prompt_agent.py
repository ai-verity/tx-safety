"""
VLM Prompt Generation Agent
============================
Reads incidents from the last 24 hours and generates structured prompts
for Vision Language Models (VLMs) analysing city video surveillance feeds.

Three personas:
  - law_enforcement   : patrol officers, detectives, dispatch
  - city_official     : mayor's office, city manager, emergency management
  - physical_security : private security operators, facility managers, SOC analysts

Each prompt is a self-contained instruction a human or automated system
can paste directly into a VLM (GPT-4o Vision, LLaVA, Gemini Vision, etc.)
along with a camera frame to get actionable situational intelligence.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

from pydantic import BaseModel, Field
import uuid

logger = logging.getLogger(__name__)

# ── Persona definitions ────────────────────────────────────────────────────────

PERSONAS: dict[str, dict] = {
    "law_enforcement": {
        "label": "Law Enforcement",
        "role":  "sworn patrol officer, detective, or dispatch supervisor",
        "goal":  "detect criminal activity, locate suspects, protect officers and public",
        "tone":  "tactical, direct, evidence-focused",
        "priorities": [
            "suspect identification and direction of travel",
            "weapons or contraband visible in frame",
            "victim location and condition",
            "escape routes and vehicle descriptions",
            "crowd dynamics that threaten officer safety",
        ],
        "output_format": (
            "Respond with: (1) THREAT ASSESSMENT [CRITICAL/HIGH/MODERATE/LOW], "
            "(2) OBSERVED ACTIVITY, (3) SUSPECT/VEHICLE DESCRIPTION if present, "
            "(4) RECOMMENDED UNIT RESPONSE, (5) EVIDENCE PRESERVATION NOTES."
        ),
    },
    "city_official": {
        "label": "City Official",
        "role":  "city manager, mayor's emergency liaison, or public information officer",
        "goal":  "understand community impact, resource deployment, and public communications",
        "tone":  "measured, policy-oriented, community-aware",
        "priorities": [
            "scale of impact on residents and businesses",
            "infrastructure or public property damage",
            "crowd size and public sentiment",
            "traffic and transit disruption",
            "need for emergency declarations or mutual aid",
        ],
        "output_format": (
            "Respond with: (1) SITUATION SUMMARY (2 sentences for public briefing), "
            "(2) AFFECTED AREA & POPULATION ESTIMATE, (3) INFRASTRUCTURE IMPACT, "
            "(4) RECOMMENDED CITY RESOURCES, (5) PUBLIC MESSAGING GUIDANCE."
        ),
    },
    "physical_security": {
        "label": "Physical Security Operator",
        "role":  "private security SOC analyst, facility manager, or loss-prevention officer",
        "goal":  "protect people, assets, and facilities; coordinate with law enforcement",
        "tone":  "observational, procedure-driven, liability-aware",
        "priorities": [
            "unauthorised access or perimeter breach",
            "suspicious packages, vehicles, or behaviour",
            "fire, hazmat, or structural hazard indicators",
            "crowd flow and choke-point congestion",
            "camera blind spots being exploited",
        ],
        "output_format": (
            "Respond with: (1) SECURITY ALERT LEVEL [RED/AMBER/GREEN], "
            "(2) OBSERVED ANOMALIES, (3) AFFECTED ZONE/CAMERA ID, "
            "(4) IMMEDIATE SECURITY ACTION, (5) LAW ENFORCEMENT NOTIFICATION required? [YES/NO + reason]."
        ),
    },
}

# ── Camera zone templates (represent real city surveillance placement types) ──

CAMERA_ZONES = {
    "Shooting":           ["intersection", "alley", "parking_lot", "convenience_store", "residential_street"],
    "Vehicle Accident":   ["highway", "intersection", "bridge", "freeway_onramp", "school_zone"],
    "Fire":               ["commercial_district", "industrial_zone", "residential_block", "parking_structure"],
    "Medical Emergency":  ["public_park", "transit_station", "mall_entrance", "sports_venue", "sidewalk"],
    "Pursuit":            ["highway", "arterial_road", "intersection", "parking_garage", "alley"],
    "Hazmat":             ["industrial_zone", "freight_yard", "highway", "warehouse_district"],
    "Burglary":           ["commercial_district", "atm_vestibule", "parking_lot", "alley", "storefront"],
    "Assault":            ["nightlife_district", "transit_station", "parking_lot", "residential_street", "park"],
    "Disturbance":        ["nightlife_district", "public_plaza", "sports_venue", "transit_hub"],
    "Suspicious Activity":["transit_station", "government_building", "critical_infrastructure", "parking_lot"],
    "Natural Disaster":   ["flood_plain", "overpass", "open_field", "coastal_area", "low_water_crossing"],
    "Missing Person":     ["transit_station", "public_park", "school_zone", "mall", "waterfront"],
    "Major Traffic":      ["highway", "freeway_interchange", "bridge", "tunnel", "arterial_road"],
    "Other":              ["public_space", "downtown_core", "commercial_district"],
}

CAMERA_ZONE_DESCRIPTIONS = {
    "intersection":          "fixed overhead PTZ at a signalised four-way intersection",
    "alley":                 "fixed low-angle camera covering a rear-access alleyway",
    "parking_lot":           "elevated fisheye camera covering an open surface parking lot",
    "convenience_store":     "exterior camera at a 24-hour convenience store entrance",
    "residential_street":    "neighbourhood watch camera on a residential block",
    "highway":               "TXDOT overhead gantry camera on a freeway",
    "bridge":                "overhead fixed camera at a bridge approach",
    "freeway_onramp":        "ramp-metering camera at a freeway on-ramp",
    "school_zone":           "city-owned PTZ in an active school zone",
    "commercial_district":   "overhead PTZ covering a downtown commercial block",
    "industrial_zone":       "perimeter camera at an industrial facility fence line",
    "residential_block":     "pole-mounted camera on a residential block",
    "parking_structure":     "interior camera at a covered parking structure entry lane",
    "public_park":           "pan-tilt camera overlooking a public park and trail",
    "transit_station":       "platform-level camera at a bus or rail transit station",
    "mall_entrance":         "exterior PTZ at a major retail mall entrance",
    "sports_venue":          "perimeter camera at a stadium or arena gate",
    "sidewalk":              "storefront fixed camera covering a busy pedestrian sidewalk",
    "arterial_road":         "overhead camera on a major arterial road",
    "parking_garage":        "multi-level camera in a downtown parking garage",
    "freight_yard":          "perimeter camera at a rail freight yard",
    "warehouse_district":    "pole camera in a warehouse and logistics district",
    "atm_vestibule":         "close-angle camera covering a bank ATM vestibule",
    "storefront":            "angled exterior camera covering a retail storefront",
    "nightlife_district":    "overhead PTZ in a bar and restaurant entertainment district",
    "public_plaza":          "wide-angle camera on a city public plaza",
    "transit_hub":           "overhead camera at a major bus or intermodal transit hub",
    "government_building":   "perimeter camera at a city hall or courthouse entrance",
    "critical_infrastructure":"fixed camera covering critical infrastructure access point",
    "flood_plain":           "elevated camera overlooking a known flood-prone low area",
    "overpass":              "camera mounted on a highway overpass structure",
    "open_field":            "wide-angle camera covering open public space",
    "coastal_area":          "camera on elevated structure overlooking coastal or bay area",
    "low_water_crossing":    "fixed camera at a low-water crossing subject to flash flooding",
    "mall":                  "interior/exterior PTZ at a regional shopping mall",
    "waterfront":            "camera on waterfront promenade or river walk",
    "freeway_interchange":   "TxDOT overhead camera at a major freeway interchange",
    "tunnel":                "interior tunnel safety camera",
    "public_space":          "overhead PTZ covering a general public space",
    "downtown_core":         "elevated PTZ covering downtown streets and sidewalks",
}

# ── Prompt templates per persona × incident type ──────────────────────────────

def _build_system_context(persona_key: str, incident: dict, camera_zone: str) -> str:
    p = PERSONAS[persona_key]
    zone_desc = CAMERA_ZONE_DESCRIPTIONS.get(camera_zone, f"{camera_zone} camera")
    reported  = incident.get("reported_at", "")[:16].replace("T", " ")
    return (
        f"You are a VLM (Vision Language Model) assistant supporting a {p['role']} "
        f"in Texas. Your goal is to {p['goal']}. "
        f"Maintain a {p['tone']} tone. "
        f"You are analysing a live or recorded frame from a {zone_desc} "
        f"in {incident.get('city', 'an unknown city')}, TX. "
        f"An incident was reported nearby at {reported} CDT: "
        f"{incident.get('incident_type', 'Unknown')} — {incident.get('title', '')}. "
        f"Description: {incident.get('description', 'No further details.')} "
        f"Incident severity: {incident.get('severity', 'P4')}. "
        f"Focus on: {'; '.join(p['priorities'])}. "
        f"{p['output_format']}"
    )


def _build_user_prompt(persona_key: str, incident: dict, camera_zone: str, prompt_variant: int) -> str:
    """Generate the user-turn prompt (the actual question sent with the image)."""
    p = PERSONAS[persona_key]
    itype  = incident.get("incident_type", "incident")
    city   = incident.get("city", "the area")
    sev    = incident.get("severity", "P4")
    active = incident.get("active", True)
    status = "active and ongoing" if active else "recently resolved"

    variants = {
        "law_enforcement": [
            f"This camera is near the scene of a {sev} {itype} in {city}. "
            f"The incident is {status}. Analyse this frame: identify any individuals, "
            f"vehicles, or behaviours that match the reported incident. "
            f"Flag anything relevant to officer response or evidence collection.",

            f"A {itype} was reported at this location in {city}. "
            f"Review this surveillance frame and tell me: Is any suspect or associated vehicle visible? "
            f"What direction are people moving? Are there any weapons, injuries, or threats visible? "
            f"What should responding units know before arrival?",

            f"Dispatch needs an immediate read on this camera near the {itype} scene in {city}. "
            f"Describe all persons in frame — clothing, direction of travel, behaviour. "
            f"Identify any vehicles by type and colour. Note any items of evidentiary value. "
            f"Assess threat level to responding officers.",
        ],
        "city_official": [
            f"A {itype} has occurred in {city} and is {status}. "
            f"This camera covers the affected area. Assess the scene for community impact: "
            f"How many people appear affected? Is there visible property damage? "
            f"What city services — fire, EMS, public works, transit — appear needed based on what you see?",

            f"The city emergency operations centre needs a situation picture for a {itype} in {city}. "
            f"From this camera frame, estimate the scale of disruption, visible damage to public infrastructure, "
            f"and whether the scene warrants a public advisory or declaration of local emergency.",

            f"As part of a city situational awareness review of the {itype} in {city}, "
            f"analyse this camera feed. Provide a two-sentence public-facing summary of conditions, "
            f"note any visible hazards to residents, and recommend whether street closures or "
            f"public messaging are warranted.",
        ],
        "physical_security": [
            f"A {itype} has been reported in {city} near this camera's coverage area. "
            f"Review this frame for any security anomalies: unauthorised individuals, "
            f"suspicious objects, perimeter breaches, or behaviour that deviates from baseline. "
            f"Assign a security alert level and recommend an immediate action.",

            f"Security operations centre — a {sev} {itype} is active near your facility in {city}. "
            f"Analyse this camera frame: Is there any spillover threat into your perimeter? "
            f"Are access control points compromised? Are there individuals behaving suspiciously? "
            f"Should law enforcement be notified from this site?",

            f"Loss-prevention and security review: {itype} activity detected in the {city} area. "
            f"Using this camera frame, assess whether your facility or site is at risk. "
            f"Identify any persons loitering, any vehicles blocking emergency access, "
            f"or any physical indicators of escalation. State what security protocol to activate.",
        ],
    }

    prompts = variants.get(persona_key, variants["law_enforcement"])
    return prompts[prompt_variant % len(prompts)]


# ── Main output model ──────────────────────────────────────────────────────────

class VLMPrompt(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    incident_id: str
    incident_type: str
    severity: str
    city: str
    persona: str
    persona_label: str
    camera_zone: str
    camera_zone_description: str
    system_prompt: str
    user_prompt: str
    combined_prompt: str       # system + user merged for single-turn VLMs
    suggested_vlm_models: list[str]
    generated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


# ── Generator ─────────────────────────────────────────────────────────────────

SUGGESTED_MODELS = {
    "law_enforcement":   ["GPT-4o Vision", "Gemini 1.5 Pro Vision", "LLaVA-1.6-34B", "InternVL2"],
    "city_official":     ["GPT-4o Vision", "Gemini 1.5 Flash Vision", "Claude 3.5 Sonnet Vision"],
    "physical_security": ["GPT-4o Vision", "LLaVA-1.6-34B", "Gemini 1.5 Pro Vision", "Qwen-VL-Max"],
}


def generate_prompts_for_incident(incident: dict) -> list[VLMPrompt]:
    """Generate all persona × variant prompts for a single incident."""
    prompts: list[VLMPrompt] = []
    itype   = incident.get("incident_type", "Other")
    zones   = CAMERA_ZONES.get(itype, CAMERA_ZONES["Other"])

    for persona_key in PERSONAS:
        # One prompt per camera zone (up to 3 zones per incident/persona)
        for i, zone in enumerate(zones[:3]):
            sys_prompt  = _build_system_context(persona_key, incident, zone)
            user_prompt = _build_user_prompt(persona_key, incident, zone, i)
            combined    = (
                f"[SYSTEM]\n{sys_prompt}\n\n"
                f"[USER]\n{user_prompt}\n\n"
                f"[IMAGE: attach surveillance frame from {zone} camera in "
                f"{incident.get('city','TX')} here]"
            )
            zone_desc = CAMERA_ZONE_DESCRIPTIONS.get(zone, zone)
            prompts.append(VLMPrompt(
                incident_id=incident.get("id", ""),
                incident_type=itype,
                severity=incident.get("severity", "P4"),
                city=incident.get("city", ""),
                persona=persona_key,
                persona_label=PERSONAS[persona_key]["label"],
                camera_zone=zone,
                camera_zone_description=zone_desc,
                system_prompt=sys_prompt,
                user_prompt=user_prompt,
                combined_prompt=combined,
                suggested_vlm_models=SUGGESTED_MODELS[persona_key],
            ))
    return prompts


async def generate_all_prompts(hours: int = 24) -> list[VLMPrompt]:
    """Load recent incidents from DB and generate VLM prompts for all of them."""
    from core.database import get_recent_incidents

    incidents = await get_recent_incidents(hours=hours, limit=200)
    if not incidents:
        logger.warning(f"[vlm_prompts] No incidents found in last {hours}h — using demo data")
        incidents = _demo_incidents()

    logger.info(f"[vlm_prompts] Generating prompts for {len(incidents)} incidents")
    all_prompts: list[VLMPrompt] = []
    for inc in incidents:
        all_prompts.extend(generate_prompts_for_incident(inc))

    logger.info(f"[vlm_prompts] Generated {len(all_prompts)} VLM prompts total")
    return all_prompts


def _demo_incidents() -> list[dict]:
    """Representative demo incidents when DB is empty."""
    now = datetime.now(timezone.utc).isoformat()
    return [
        {"id":"demo-001","title":"Officer-involved shooting near downtown intersection","incident_type":"Shooting","severity":"P1","city":"Houston","county":"Harris","description":"Reports of gunfire at the intersection of Main St and Commerce St. Multiple units responding. Suspect fled on foot northbound.","active":True,"reported_at":now},
        {"id":"demo-002","title":"Multi-vehicle accident blocks I-35 northbound","incident_type":"Vehicle Accident","severity":"P2","city":"Austin","county":"Travis","description":"Three-car collision blocking two northbound lanes near exit 238. One vehicle on fire. EMS en route.","active":True,"reported_at":now},
        {"id":"demo-003","title":"Structure fire at commercial warehouse","incident_type":"Fire","severity":"P2","city":"Dallas","county":"Dallas","description":"Large smoke column visible from warehouse on Industrial Blvd. Multiple fire units on scene. Evacuation of surrounding businesses ordered.","active":True,"reported_at":now},
        {"id":"demo-004","title":"Suspicious package reported at city hall","incident_type":"Suspicious Activity","severity":"P2","city":"San Antonio","county":"Bexar","description":"Unattended bag reported near the main entrance of city hall. Building being evacuated as precaution. Bomb squad notified.","active":True,"reported_at":now},
        {"id":"demo-005","title":"Hazmat spill on State Highway 6","incident_type":"Hazmat","severity":"P2","city":"Houston","county":"Harris","description":"Chemical tanker rollover on SH-6 southbound. Unknown substance leaking. 500-ft exclusion zone established. HAZMAT unit responding.","active":True,"reported_at":now},
        {"id":"demo-006","title":"Aggravated assault outside nightclub","incident_type":"Assault","severity":"P3","city":"Fort Worth","county":"Tarrant","description":"Victim transported to hospital after altercation outside venue on West 7th. Suspect described as male, 6ft, dark jacket, last seen on foot.","active":False,"reported_at":now},
        {"id":"demo-007","title":"Missing juvenile last seen at Riverwalk","incident_type":"Missing Person","severity":"P2","city":"San Antonio","county":"Bexar","description":"14-year-old female missing from River Walk area since 6 PM. Last seen near Arneson River Theatre. AMBER Alert not yet issued.","active":True,"reported_at":now},
        {"id":"demo-008","title":"Flash flood warning issued for low-water crossings","incident_type":"Natural Disaster","severity":"P3","city":"Austin","county":"Travis","description":"NWS has issued flash flood warning for Travis County through midnight. Multiple low-water crossings closed. BartonCreek expected to exceed banks.","active":True,"reported_at":now},
    ]
