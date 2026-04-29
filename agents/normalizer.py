"""
Normalization Agent — uses the local LLM (Ollama) to extract structured
incident data from raw text. Deduplicates, geo-tags, and persists to DB.
Broadcasts new incidents over WebSocket.
"""
from __future__ import annotations
import asyncio
import logging
from datetime import datetime, timezone
from core.models import RawItem, Incident, IncidentType, Severity
from core.database import upsert_incident, get_active_incidents
from core.llm import chat_json
from core.geocoder import geocode_city, map_coords

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a Texas public safety incident extraction system.
Given raw text from news articles, government feeds, or social media, extract
incident information and return ONLY a JSON object with these exact fields:

{
  "is_incident": true/false,
  "is_texas": true/false,
  "title": "brief 1-line title",
  "incident_type": one of: Shooting, Vehicle Accident, Fire, Medical Emergency,
                   Pursuit, Hazmat, Burglary, Assault, Disturbance,
                   Suspicious Activity, Natural Disaster, Missing Person,
                   Major Traffic, Other,
  "severity": "P1" (critical/life-threatening), "P2" (high/serious),
              "P3" (medium/notable), or "P4" (low/informational),
  "city": "city name in Texas or empty string",
  "county": "county name or empty string",
  "description": "1-3 sentence factual summary of what happened"
}

Return is_incident=false if the text is not about a public safety incident.
Return is_texas=false if the incident is not in Texas.
Always respond with valid JSON only."""


class NormalizationAgent:
    """Lightweight normalizer — not a looping agent, called by ingestion agents."""

    async def normalize_and_save(self, item: RawItem) -> Incident | None:
        """Extract, validate, geocode, and persist an incident."""
        try:
            result = await chat_json(SYSTEM_PROMPT, item.raw_text[:1500])
        except Exception as e:
            logger.warning(f"[normalizer] LLM call failed ({type(e).__name__}): {e}")
            return None

        if not result:
            return None
        if not result.get("is_incident") or not result.get("is_texas"):
            return None

        # Map LLM output to typed model
        try:
            inc_type = IncidentType(result.get("incident_type", "Other"))
        except ValueError:
            inc_type = IncidentType.OTHER

        try:
            severity = Severity(result.get("severity", "P4"))
        except ValueError:
            severity = Severity.P4

        city   = (result.get("city") or "").strip()
        county = (result.get("county") or "").strip()

        # Geocode
        lat, lon = await geocode_city(city, county)
        map_x, map_y = None, None
        if lat is not None and lon is not None:
            map_x, map_y = map_coords(lat, lon)

        incident = Incident(
            title=result.get("title", item.raw_text[:80]),
            incident_type=inc_type,
            severity=severity,
            city=city,
            county=county,
            lat=lat,
            lon=lon,
            description=result.get("description", ""),
            source=item.source,
            source_url=item.url,
            active=True,
            reported_at=item.retrieved_at,
            updated_at=datetime.now(timezone.utc),
        )

        # Quick dedup: skip if very similar title seen in last 6 hours
        if await self._is_duplicate(incident):
            logger.debug(f"[normalizer] duplicate skipped: {incident.title[:60]}")
            return None

        await upsert_incident(incident)
        logger.info(f"[normalizer] saved {incident.severity.value} {incident.incident_type.value} — {incident.city}")

        # Broadcast to dashboard over WebSocket
        await self._broadcast(incident, map_x, map_y)
        return incident

    async def _is_duplicate(self, inc: Incident) -> bool:
        """Rough dedup: check if an active incident with similar title exists."""
        try:
            active = await get_active_incidents(100)
            title_words = set(inc.title.lower().split())
            for existing in active:
                existing_words = set((existing.get("title") or "").lower().split())
                if len(title_words) >= 2:
                    if len(title_words) <= 3:
                        overlap = 1.0 if title_words == existing_words else 0.0
                    else:
                        overlap = len(title_words & existing_words) / len(title_words)
                    if overlap > 0.7 and existing.get("city", "") == inc.city:
                        return True
        except Exception:
            pass
        return False

    async def _broadcast(self, inc: Incident, map_x, map_y):
        """Broadcast new incident to connected WebSocket clients."""
        try:
            # Import here to avoid circular at module load
            from core.ws_manager import manager
            data = inc.to_dict()
            data["map_x"] = map_x
            data["map_y"] = map_y
            await manager.broadcast("new_incident", data)
        except Exception as e:
            logger.debug(f"[normalizer] broadcast error: {e}")
