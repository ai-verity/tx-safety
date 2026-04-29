"""
VLM Prompt Generator for Texas Public Safety Situational Awareness.

Reads incidents from the last 24 hours and generates targeted prompts
for three operator personas submitting a video frame to a Vision Language
Model (VLM):

  • Law Enforcement      — patrol, detectives, SWAT, watch commanders
  • City Official        — emergency management, mayor's office, public works
  • Physical Security    — VSO, SOC analysts, facility/infrastructure security
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional


# ─── Persona definitions ──────────────────────────────────────────────────────

PERSONAS: dict[str, dict] = {
    "law_enforcement": {
        "label": "Law Enforcement",
        "badge": "LE",
        "color": "#378add",
        "roles": [
            "Patrol Officer", "Detective", "SWAT / Tactical",
            "Watch Commander", "Gang / Vice Unit", "Traffic Division",
        ],
        "mission": (
            "Identify threats, suspects, and evidence to protect life, "
            "apprehend offenders, and support prosecution."
        ),
        "vlm_focus": [
            "individuals matching active BOLO / suspect descriptions",
            "visible firearms, edged weapons, or dangerous objects",
            "suspect or wanted vehicles (make, model, color, plate fragments)",
            "direction of travel and escape routes",
            "number of subjects and their roles (shooter, lookout, driver)",
            "victim locations, conditions, and need for medical aid",
            "bystander risk zones and crowd dynamics",
            "officer approach vectors and cover positions",
        ],
        "tone": "tactical, precise, officer-safety-first, evidence-aware",
    },
    "city_official": {
        "label": "City Official / Emergency Management",
        "badge": "GOV",
        "color": "#a78bfa",
        "roles": [
            "Emergency Management Director", "Mayor's Situation Room",
            "Public Works / Infrastructure", "Fire Marshal",
            "Health & Human Services", "Transportation Director",
        ],
        "mission": (
            "Protect public welfare, allocate resources, protect infrastructure, "
            "and manage inter-agency coordination and public communications."
        ),
        "vlm_focus": [
            "civilian exposure and crowd size in the impact zone",
            "critical infrastructure at risk (utilities, bridges, hospitals, roads)",
            "hazmat plume spread or fire perimeter relative to populated areas",
            "road blockages and alternate evacuation / emergency vehicle routes",
            "vulnerable populations visible (elderly, children, mobility-impaired)",
            "resource staging areas and access routes for mutual aid",
            "secondary hazards (flooding, structural collapse, smoke drift)",
            "media / press presence that may require public statement coordination",
        ],
        "tone": "strategic, resource-focused, public-impact-aware, inter-agency",
    },
    "physical_security": {
        "label": "Physical Security Operator",
        "badge": "SOC",
        "color": "#40c97e",
        "roles": [
            "Video Surveillance Officer (VSO)",
            "Security Operations Center (SOC) Analyst",
            "Corporate / Campus Security Director",
            "Venue / Stadium / Event Security",
            "Critical Infrastructure Protection (CIP)",
        ],
        "mission": (
            "Detect and deter threats before perimeter breach, track persons "
            "of interest across the camera network, and preserve evidence."
        ),
        "vlm_focus": [
            "unauthorized access attempts or active perimeter breaches",
            "pre-attack surveillance behavior (loitering, photographing, repeated passes)",
            "abandoned packages, bags, or unattended vehicles",
            "persons matching active BOLOs entering camera field of view",
            "access control anomalies (tailgating, forced entry, credential bypass)",
            "camera tampering, obstruction, or lens vandalism",
            "license plates of vehicles of interest entering / exiting the facility",
            "employee or contractor behavior anomalies near sensitive areas",
        ],
        "tone": "operational, camera-network-aware, evidence-chain-of-custody-focused",
    },
}

# ─── Incident-type → visual intelligence cues ────────────────────────────────

INCIDENT_CUES: dict[str, dict] = {
    "Shooting": {
        "scene": "active shooter / shots fired scene",
        "visuals": [
            "person brandishing or firing a firearm",
            "individuals fleeing, ducking, or taking cover",
            "fallen or injured victim on ground",
            "muzzle flash, smoke, or bullet impact dust",
            "shell casings on pavement",
            "law enforcement tactical approach and perimeter formation",
            "suspect vehicle idling or fleeing",
        ],
        "cameras": [
            "intersection cameras within 3-block radius",
            "ATM / bank cameras for high-resolution facial capture",
            "business exterior cameras facing the street",
            "parking structure overview cameras",
            "transit stop cameras for suspect flight path",
        ],
        "time_sensitivity": "IMMEDIATE — scene is dynamic, suspect may be actively fleeing",
    },
    "Vehicle Accident": {
        "scene": "multi-vehicle collision / major traffic incident",
        "visuals": [
            "vehicle damage, deployment of airbags, structural deformation",
            "injured occupants inside or exiting vehicles",
            "debris field and fluid spills creating secondary hazards",
            "oncoming traffic not yet aware of blockage",
            "emergency responder positioning and triage zones",
            "fuel fire or smoke from engine compartment",
        ],
        "cameras": [
            "traffic signal cameras at the intersection",
            "highway overhead / gantry cameras",
            "adjacent business parking lot cameras",
            "school / hospital zone cameras if nearby",
        ],
        "time_sensitivity": "HIGH — secondary collisions likely, triage timeline critical",
    },
    "Fire": {
        "scene": "structure fire / wildfire / vehicle fire",
        "visuals": [
            "flame visibility and smoke plume direction and height",
            "building structural integrity and collapse risk indicators",
            "civilians evacuating or trapped at windows",
            "overhead utility lines threatened by fire",
            "fire apparatus access routes and water supply points",
            "exposure risk to adjacent structures",
        ],
        "cameras": [
            "building exterior cameras on all four elevations",
            "traffic cameras for perimeter and evacuation routing",
            "overhead PTZ camera for smoke column direction",
            "neighboring property cameras for exposure assessment",
        ],
        "time_sensitivity": "IMMEDIATE — fire growth rate and wind-driven spread can change rapidly",
    },
    "Medical Emergency": {
        "scene": "medical emergency / mass casualty / overdose cluster",
        "visuals": [
            "number of patients and visible distress indicators",
            "bystanders performing CPR or rescue breathing",
            "AED retrieval and use",
            "ambulance and fire apparatus access path obstructions",
            "crowd impeding EMS access to patient",
            "substance paraphernalia if overdose suspected",
        ],
        "cameras": [
            "venue interior cameras and lobby cameras",
            "parking lot cameras for ambulance routing",
            "exterior entrance cameras for crowd management",
            "transit platform cameras if event is transit-adjacent",
        ],
        "time_sensitivity": "HIGH — time-to-treatment determines patient outcome",
    },
    "Pursuit": {
        "scene": "active vehicle or foot pursuit",
        "visuals": [
            "suspect vehicle description, color, damage, and plate fragments",
            "speed estimation and reckless driving indicators",
            "pedestrian and bystander proximity in vehicle path",
            "intersection approach speed for collision risk prediction",
            "subject on foot if vehicle was abandoned",
            "discarded items (weapons, contraband) during flight",
        ],
        "cameras": [
            "arterial road cameras for real-time handoff",
            "intersection traffic cameras at predicted route",
            "freeway on-ramp and overhead cameras",
            "gas station and convenience store cameras (subject may bail here)",
            "school zone cameras if pursuit is near schools",
        ],
        "time_sensitivity": "IMMEDIATE — real-time camera-to-camera handoff required",
    },
    "Hazmat": {
        "scene": "hazardous material spill / chemical release / explosion",
        "visuals": [
            "visible vapor cloud, plume color, or liquid pooling",
            "contamination spread direction and rate",
            "civilian proximity to exposure zone",
            "responder PPE level indicating hazard severity",
            "source container, tanker, or facility of origin",
            "wind direction indicators (flags, smoke, trees)",
            "wildlife or animal distress as contamination indicator",
        ],
        "cameras": [
            "industrial facility cameras and loading dock cameras",
            "traffic cameras for evacuation route monitoring",
            "overhead PTZ for plume tracking and zone mapping",
            "hospital and shelter approach cameras for refugee tracking",
        ],
        "time_sensitivity": "IMMEDIATE — contamination zone is expanding, evacuation routing urgent",
    },
    "Burglary": {
        "scene": "burglary in progress or freshly occurred",
        "visuals": [
            "point of forced entry (broken window, pried door, glass on floor)",
            "suspect physical description, clothing, and distinguishing features",
            "stolen property being carried",
            "getaway vehicle make, model, color, and plate",
            "accomplices acting as lookout",
            "direction of flight from scene",
        ],
        "cameras": [
            "business exterior cameras covering all entry/exit points",
            "ATM cameras for facial capture",
            "adjacent parking structure cameras",
            "alley and rear-entry cameras",
            "neighboring business cameras covering suspect flight path",
        ],
        "time_sensitivity": "HIGH — suspect likely within 2-block radius, act within 10 minutes",
    },
    "Assault": {
        "scene": "physical assault / aggravated assault in progress or just occurred",
        "visuals": [
            "suspect and victim physical descriptions and clothing",
            "weapons used or displayed",
            "number of assailants and their roles",
            "direction of suspect flight post-assault",
            "victim condition and need for medical response",
            "bystander witnesses present",
        ],
        "cameras": [
            "street-level fixed cameras",
            "bar, restaurant, and entertainment venue exterior cameras",
            "transit station platform cameras",
            "ATM cameras for high-resolution facial capture",
            "rideshare pickup zone cameras",
        ],
        "time_sensitivity": "HIGH — suspect in immediate area, victim medical assessment needed",
    },
    "Disturbance": {
        "scene": "civil disturbance / large fight / crowd disorder",
        "visuals": [
            "crowd size, density, and spatial distribution",
            "weapons or improvised weapons in crowd",
            "instigators vs passive bystanders",
            "crowd movement direction and momentum",
            "property damage in progress",
            "police line positioning and potential flash points",
        ],
        "cameras": [
            "wide-angle overview PTZ for crowd mapping",
            "venue and event entrance cameras",
            "transit platform cameras",
            "street-level fisheye cameras for individual identification",
            "elevated cameras for overhead crowd flow analysis",
        ],
        "time_sensitivity": "MODERATE — monitor for escalation to violence or weapon use",
    },
    "Suspicious Activity": {
        "scene": "suspicious person, vehicle, or unattended package",
        "visuals": [
            "subject behavior patterns (loitering, conducting surveillance, photography)",
            "unattended bags, packages, or containers",
            "vehicle parked illegally near sensitive location",
            "clothing inconsistent with weather or environment",
            "repeated passes of the same location over time",
            "interest in security cameras, access points, or utility infrastructure",
        ],
        "cameras": [
            "fixed perimeter cameras at approaches",
            "PTZ camera for close facial detail capture",
            "license plate reader camera angles",
            "access point and entry cameras",
            "cameras with overlapping fields of view for triangulation",
        ],
        "time_sensitivity": "MODERATE — document and track, assess for escalation indicators",
    },
    "Natural Disaster": {
        "scene": "severe weather / flash flood / tornado / wildfire spread",
        "visuals": [
            "flood water depth relative to road markers and vehicles",
            "structural damage extent and collapse indicators",
            "stranded civilians requiring rescue",
            "road passability for emergency vehicles",
            "downed power lines and utility hazards",
            "rescue team positioning and access paths",
        ],
        "cameras": [
            "flood-prone underpass and low-water crossing cameras",
            "bridge and drainage channel cameras",
            "critical infrastructure cameras (hospitals, EOC, power substations)",
            "shelter approach and staging area cameras",
            "highway cameras for evacuation route monitoring",
        ],
        "time_sensitivity": "HIGH — rapidly changing conditions, rescue window may be closing",
    },
    "Missing Person": {
        "scene": "missing person / Amber Alert / Silver Alert",
        "visuals": [
            "individual matching last known clothing and physical description",
            "associated vehicle (Amber/Silver Alert vehicle)",
            "direction of travel from last known location",
            "interactions with other individuals",
            "distress indicators or signs of coercion",
            "mobile device use that may indicate communication",
        ],
        "cameras": [
            "transit station cameras (bus, rail, rideshare hubs)",
            "school zone and playground cameras",
            "retail corridor and mall entrance cameras",
            "highway on-ramp cameras for regional tracking",
            "ATM cameras for high-resolution facial capture",
            "gas station cameras near last known location",
        ],
        "time_sensitivity": "HIGH — first 48 hours critical, each camera minute is irreplaceable",
    },
    "Major Traffic": {
        "scene": "major road closure / traffic incident",
        "visuals": [
            "blockage extent and lane impact",
            "diverted traffic buildup and secondary congestion points",
            "secondary incident risk from rubbernecking",
            "emergency vehicle access path through stopped traffic",
            "pedestrian diversion onto roadway",
        ],
        "cameras": [
            "arterial traffic management cameras",
            "freeway management system overhead cameras",
            "alternate route intersection cameras",
            "incident command post positioning camera",
        ],
        "time_sensitivity": "MODERATE — cascade effect on city traffic grid within 15 minutes",
    },
    "Other": {
        "scene": "public safety incident",
        "visuals": [
            "unusual or threatening activity",
            "persons or vehicles of interest",
            "environmental hazards",
            "crowd behavior changes",
        ],
        "cameras": [
            "nearest available camera to incident coordinates",
            "perimeter cameras covering approach routes",
            "access point cameras",
        ],
        "time_sensitivity": "MODERATE — assess and classify",
    },
}

# ─── Severity modifiers ───────────────────────────────────────────────────────

SEVERITY_CONTEXT = {
    "P1": {
        "urgency_label": "CRITICAL — IMMEDIATE ACTION REQUIRED",
        "operator_instruction": (
            "This is a life-threatening, actively evolving incident. "
            "Dedicate all available camera resources to the incident zone immediately. "
            "Provide continuous updates. Do not wait for confirmation before escalating."
        ),
    },
    "P2": {
        "urgency_label": "HIGH PRIORITY",
        "operator_instruction": (
            "Serious incident with confirmed or likely injury or significant property risk. "
            "Monitor primary cameras and all adjacent camera zones continuously."
        ),
    },
    "P3": {
        "urgency_label": "ELEVATED — ACTIVE MONITORING",
        "operator_instruction": (
            "Notable incident in progress or recently reported. "
            "Maintain active camera coverage and document all observations."
        ),
    },
    "P4": {
        "urgency_label": "INFORMATIONAL",
        "operator_instruction": (
            "Low-level or resolved incident. "
            "Document observations for situational awareness and shift log."
        ),
    },
}


# ─── Core prompt builder ──────────────────────────────────────────────────────

def build_vlm_prompt(incident: dict, persona_key: str) -> dict:
    """
    Build a complete, ready-to-submit VLM prompt package for one
    incident × persona combination.

    Returns a dict with:
      system_prompt   — sets the VLM's role and context
      user_prompt     — the actual query to submit with the video frame
      metadata        — structured tags for logging / UI display
    """
    persona = PERSONAS[persona_key]
    inc_type = incident.get("incident_type", "Other")
    cues = INCIDENT_CUES.get(inc_type, INCIDENT_CUES["Other"])
    sev = incident.get("severity", "P4")
    sev_ctx = SEVERITY_CONTEXT.get(sev, SEVERITY_CONTEXT["P4"])

    city = incident.get("city") or "Unknown location"
    county = incident.get("county") or ""
    location_str = f"{city}{', ' + county + ' County' if county else ''}, Texas"

    reported_at = incident.get("reported_at", "")
    try:
        dt = datetime.fromisoformat(str(reported_at).replace("Z", "+00:00"))
        time_str = dt.strftime("%B %d, %Y at %H:%M CDT")
    except Exception:
        time_str = str(reported_at) if reported_at else "Unknown time"

    title = incident.get("title") or f"{inc_type} in {city}"
    description = incident.get("description") or ""
    source = incident.get("source") or "multi-source"
    inc_id = incident.get("id", "unknown")[:8]

    focus_list = "\n".join(f"  • {f}" for f in persona["vlm_focus"])
    visuals_list = "\n".join(f"  • {v}" for v in cues["visuals"])
    cameras_list = "\n".join(f"  • {c}" for c in cues["cameras"])

    # ── SYSTEM PROMPT ─────────────────────────────────────────
    system_prompt = f"""You are an advanced Vision Language Model (VLM) integrated into the Texas Public Safety Situational Awareness System.

OPERATOR ROLE: {persona['label']}
OPERATOR MISSION: {persona['mission']}

You are analyzing live or recorded video surveillance footage in direct support of an active public safety incident. Your output will be used in real-time operational decision-making.

INCIDENT CONTEXT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Incident ID   : {inc_id}
Type          : {inc_type}
Severity      : {sev} — {sev_ctx['urgency_label']}
Location      : {location_str}
Reported      : {time_str}
Source        : {source}
Scene Context : {cues['scene']}

Incident Summary:
{title}
{('Detail: ' + description) if description else ''}

OPERATOR DIRECTIVE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{sev_ctx['operator_instruction']}

YOUR INTELLIGENCE PRIORITIES FOR THIS OPERATOR
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{focus_list}

RESPONSE FORMAT REQUIREMENTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Structure every response as follows:

1. IMMEDIATE OBSERVATIONS — What do you see RIGHT NOW that is directly relevant?
2. PERSONS OF INTEREST — Describe any individuals requiring attention (appearance, behavior, location in frame, direction of movement).
3. VEHICLES OF INTEREST — Make, model, color, plate (full or partial), direction.
4. THREAT INDICATORS — Weapons, hazards, or behaviors that elevate risk.
5. RECOMMENDED CAMERA ACTIONS — Pan, zoom, switch to adjacent camera, or preserve recording.
6. OPERATOR ACTION — Specific recommendation for the {persona['label']} based on what you see.

Use precise, factual, non-speculative language. If something is not visible or not determinable from the frame, state that explicitly. Do not hallucinate details."""

    # ── USER PROMPT ───────────────────────────────────────────
    user_prompt = f"""[VIDEO FRAME SUBMITTED FOR ANALYSIS]

Incident: {sev} {inc_type} — {location_str}
Time: {time_str}

{title}{chr(10) + description if description else ''}

Analyze this video frame and provide actionable intelligence for the responding {persona['label']}.

WHAT TO LOOK FOR IN THIS FRAME:
{visuals_list}

PRIORITY CAMERA SOURCES FOR THIS INCIDENT TYPE:
{cameras_list}

Specific questions for this operator role:
{_persona_specific_questions(persona_key, inc_type, city, sev)}

Provide your structured analysis now. Flag any P1-level observations (imminent threat to life) at the very top of your response before the structured sections."""

    return {
        "incident_id": inc_id,
        "incident_type": inc_type,
        "severity": sev,
        "location": location_str,
        "persona": persona_key,
        "persona_label": persona["label"],
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "metadata": {
            "reported_at": time_str,
            "source": source,
            "scene_context": cues["scene"],
            "time_sensitivity": cues["time_sensitivity"],
            "urgency": sev_ctx["urgency_label"],
            "recommended_cameras": cues["cameras"],
        },
    }


def _persona_specific_questions(persona_key: str, inc_type: str, city: str, sev: str) -> str:
    """Return persona × incident-type specific follow-up questions."""

    q = {
        "law_enforcement": {
            "Shooting": [
                f"Is there an active shooter still visible in frame? Direction of travel?",
                "Can you describe the suspect's clothing, height/build, and any distinguishing features?",
                "Is the firearm still visible? Long gun or handgun?",
                "Are there victims requiring immediate medical attention visible in frame?",
                "What is the safest approach vector for responding officers?",
            ],
            "Pursuit": [
                "What is the current direction of travel and estimated speed?",
                "Has the subject bailed from the vehicle? If so, direction on foot?",
                "Are there any weapons visible on the subject?",
                "What is the pedestrian risk level along the current pursuit path?",
            ],
            "Burglary": [
                "How many suspects are visible and what are their descriptions?",
                "Is a getaway vehicle visible? Plate number or partial?",
                "Are the suspects still on scene or in flight?",
                "What property is being removed?",
            ],
            "default": [
                "Are there any individuals matching a suspect description visible?",
                "Is there an imminent threat to officer or public safety visible in frame?",
                "What is the current activity level in the scene?",
                "Are there any weapons or dangerous objects visible?",
            ],
        },
        "city_official": {
            "Fire": [
                f"How many city blocks is the fire perimeter currently affecting in {city}?",
                "Are any critical infrastructure assets (power lines, gas mains, water mains) threatened?",
                "Is the main arterial road for emergency access clear?",
                "How many civilians appear to still be in the evacuation zone?",
                "Is there evidence of the fire spreading to adjacent structures?",
            ],
            "Natural Disaster": [
                "What is the flood water depth at road level — is the road passable by emergency vehicles?",
                "Are there stranded civilians visible who require immediate rescue?",
                "Is any critical infrastructure (bridge, underpass, power station) compromised?",
                "How many civilians appear to be in the impact zone?",
            ],
            "Hazmat": [
                "What is the visible plume direction and estimated spread distance?",
                "Are there populated areas (schools, hospitals, residences) in the plume path?",
                f"Is the evacuation route on the main corridor out of {city} clear?",
                "What is the estimated number of civilians still in the exposure zone?",
            ],
            "default": [
                "What is the estimated civilian population exposed in this frame?",
                "Are any critical public infrastructure elements visible and at risk?",
                "Is there a clear route visible for emergency vehicle access?",
                "What is the overall scale of the incident footprint visible in this frame?",
            ],
        },
        "physical_security": {
            "Suspicious Activity": [
                "Is the subject still in camera field of view? Which direction are they moving?",
                "Has the subject been photographing or sketching the facility or camera positions?",
                "Are there any unattended items left by the subject?",
                "Can you capture a clear facial image for BOLO comparison?",
                "Is the subject alone or are there associates visible outside the frame?",
            ],
            "Burglary": [
                "Which access point was breached and is it still unsecured?",
                "Are suspects still inside the facility perimeter?",
                "Has the camera covering the breach point been tampered with?",
                "Can you get a plate read on the getaway vehicle from this angle?",
            ],
            "Shooting": [
                "Has the threat entered or is it approaching the facility perimeter?",
                "Are there employees or protected persons in the line of fire visible in frame?",
                "Which access points need immediate lockdown based on threat direction?",
                "Is the emergency evacuation route clear of the threat?",
            ],
            "default": [
                "Has the perimeter been breached? If so, at which access point?",
                "Are there any individuals behaving anomalously near secure areas?",
                "Are all visible cameras operational and unobstructed?",
                "Is a license plate capture possible from this frame angle?",
                "Are there any unattended items visible near entry/exit points?",
            ],
        },
    }

    persona_q = q.get(persona_key, q["law_enforcement"])
    questions = persona_q.get(inc_type, persona_q.get("default", []))
    return "\n".join(f"  {i+1}. {question}" for i, question in enumerate(questions))


# ─── Batch generator ──────────────────────────────────────────────────────────

def generate_all_prompts(incidents: list[dict]) -> list[dict]:
    """Generate prompts for all three personas for every incident."""
    results = []
    for incident in incidents:
        for persona_key in PERSONAS:
            try:
                prompt = build_vlm_prompt(incident, persona_key)
                results.append(prompt)
            except Exception as e:
                results.append({
                    "incident_id": incident.get("id", "?")[:8],
                    "persona": persona_key,
                    "error": str(e),
                })
    return results


def generate_persona_prompts(incidents: list[dict], persona_key: str) -> list[dict]:
    """Generate prompts for one persona across all incidents."""
    return [build_vlm_prompt(i, persona_key) for i in incidents]


def demo_incidents() -> list[dict]:
    """Representative demo incidents used when the DB has no recent data."""
    now = datetime.now(timezone.utc).isoformat()
    return [
        {"id":"demo-001","title":"Officer-involved shooting near downtown intersection","incident_type":"Shooting","severity":"P1","city":"Houston","county":"Harris","description":"Reports of gunfire at the intersection of Main St and Commerce St. Multiple units responding. Suspect fled on foot northbound.","active":True,"reported_at":now,"source":"demo"},
        {"id":"demo-002","title":"Multi-vehicle accident blocks I-35 northbound","incident_type":"Vehicle Accident","severity":"P2","city":"Austin","county":"Travis","description":"Three-car collision blocking two northbound lanes near exit 238. One vehicle on fire. EMS en route.","active":True,"reported_at":now,"source":"demo"},
        {"id":"demo-003","title":"Structure fire at commercial warehouse","incident_type":"Fire","severity":"P2","city":"Dallas","county":"Dallas","description":"Large smoke column visible from warehouse on Industrial Blvd. Multiple fire units on scene. Evacuation of surrounding businesses ordered.","active":True,"reported_at":now,"source":"demo"},
        {"id":"demo-004","title":"Suspicious package reported at city hall","incident_type":"Suspicious Activity","severity":"P2","city":"San Antonio","county":"Bexar","description":"Unattended bag reported near the main entrance of city hall. Building being evacuated as precaution. Bomb squad notified.","active":True,"reported_at":now,"source":"demo"},
        {"id":"demo-005","title":"Hazmat spill on State Highway 6","incident_type":"Hazmat","severity":"P2","city":"Houston","county":"Harris","description":"Chemical tanker rollover on SH-6 southbound. Unknown substance leaking. 500-ft exclusion zone established.","active":True,"reported_at":now,"source":"demo"},
        {"id":"demo-006","title":"Aggravated assault outside nightclub","incident_type":"Assault","severity":"P3","city":"Fort Worth","county":"Tarrant","description":"Victim transported to hospital after altercation outside venue on West 7th. Suspect described as male, 6ft, dark jacket.","active":False,"reported_at":now,"source":"demo"},
        {"id":"demo-007","title":"Missing juvenile last seen at Riverwalk","incident_type":"Missing Person","severity":"P2","city":"San Antonio","county":"Bexar","description":"14-year-old female missing from River Walk area since 6 PM. Last seen near Arneson River Theatre.","active":True,"reported_at":now,"source":"demo"},
        {"id":"demo-008","title":"Flash flood warning issued for low-water crossings","incident_type":"Natural Disaster","severity":"P3","city":"Austin","county":"Travis","description":"NWS issued flash flood warning for Travis County through midnight. Multiple low-water crossings closed.","active":True,"reported_at":now,"source":"demo"},
    ]
