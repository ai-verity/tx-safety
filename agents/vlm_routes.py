"""
FastAPI routes for VLM prompt generation.
Mounted into main.py under /api/vlm/...
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import Optional

from core.database import get_recent_incidents
from agents.vlm_prompts import generate_all_prompts, generate_persona_prompts, build_vlm_prompt, PERSONAS

router = APIRouter(prefix="/api/vlm", tags=["vlm"])


class SinglePromptRequest(BaseModel):
    incident: dict
    persona: str


@router.get("/prompts")
async def get_vlm_prompts(
    hours: int = Query(24, description="Lookback window in hours"),
    persona: Optional[str] = Query(None, description="law_enforcement | city_official | physical_security"),
    severity: Optional[str] = Query(None, description="P1 | P2 | P3 | P4"),
    limit: int = Query(50, description="Max incidents to process"),
):
    """Generate VLM prompts for incidents in the last N hours."""
    incidents = await get_recent_incidents(hours=hours, limit=limit)
    if severity:
        incidents = [i for i in incidents if i.get("severity") == severity.upper()]
    if not incidents:
        return {"count": 0, "prompts": [], "message": f"No incidents in last {hours}h"}
    if persona:
        if persona not in PERSONAS:
            raise HTTPException(400, detail=f"Unknown persona. Valid: {list(PERSONAS.keys())}")
        prompts = generate_persona_prompts(incidents, persona)
    else:
        prompts = generate_all_prompts(incidents)
    return {
        "count": len(prompts),
        "incident_count": len(incidents),
        "hours_window": hours,
        "personas": list(PERSONAS.keys()) if not persona else [persona],
        "prompts": prompts,
    }


@router.get("/prompts/summary")
async def get_vlm_prompt_summary(hours: int = 24):
    """Listing of available prompts without full text (for UI)."""
    incidents = await get_recent_incidents(hours=hours, limit=100)
    summary = []
    for inc in incidents:
        for persona_key, persona in PERSONAS.items():
            summary.append({
                "incident_id": inc.get("id", "")[:8],
                "incident_type": inc.get("incident_type", "Other"),
                "severity": inc.get("severity", "P4"),
                "location": f"{inc.get('city','')}, TX",
                "persona": persona_key,
                "persona_label": persona["label"],
                "reported_at": inc.get("reported_at", ""),
                "title": (inc.get("title") or "")[:80],
            })
    return {"count": len(summary), "items": summary}


@router.post("/prompts/single")
async def build_single_prompt(req: SinglePromptRequest):
    """Build a prompt for a manually supplied incident + persona."""
    if req.persona not in PERSONAS:
        raise HTTPException(400, detail=f"Unknown persona. Valid: {list(PERSONAS.keys())}")
    return build_vlm_prompt(req.incident, req.persona)


@router.get("/personas")
async def get_personas():
    """Return all persona definitions."""
    return {k: {
        "label": v["label"], "badge": v["badge"],
        "color": v["color"], "roles": v["roles"], "mission": v["mission"],
    } for k, v in PERSONAS.items()}
