"""
Analysis Agents:
  - ThreatClassifierAgent: re-evaluates severity of active incidents using LLM
  - TrendAgent: detects surges and geographic clusters, broadcasts alerts
"""
from __future__ import annotations
import logging
from datetime import datetime, timedelta
from collections import Counter, defaultdict
from agents.base import BaseAgent
from core.database import get_active_incidents, get_recent_incidents, upsert_incident, get_stats
from core.models import Incident, Severity, IncidentType
from core.llm import chat_json
from core.ws_manager import manager

logger = logging.getLogger(__name__)


class ThreatClassifierAgent(BaseAgent):
    """
    Periodically reviews active P3/P4 incidents and upgrades severity
    if contextual signals (e.g., multiple nearby shootings) warrant it.
    """
    name = "threat_classifier"
    interval_seconds = 300

    async def run_once(self) -> int:
        active = await get_active_incidents(50)
        # Focus on items that might need re-classification
        candidates = [i for i in active if i.get("severity") in ("P3", "P4")]
        count = 0

        for item in candidates[:10]:   # Don't over-call LLM
            context = (
                f"Incident: {item.get('title')}\n"
                f"Type: {item.get('incident_type')}\n"
                f"Current severity: {item.get('severity')}\n"
                f"Description: {item.get('description','')}\n"
                f"City: {item.get('city')}, TX\n\n"
                "Should the severity be upgraded? Respond with JSON: "
                '{"upgrade": true/false, "new_severity": "P1"/"P2"/"P3"/"P4", "reason": "..."}'
            )
            try:
                result = await chat_json(
                    "You are a public safety threat assessment AI. "
                    "Evaluate whether this incident's severity should be upgraded based on its description.",
                    context
                )
                if result.get("upgrade") and result.get("new_severity") in ("P1", "P2"):
                    new_sev = Severity(result["new_severity"])
                    # Update in DB
                    inc = Incident(
                        id=item["id"],
                        title=item["title"],
                        incident_type=IncidentType(item.get("incident_type", "Other")),
                        severity=new_sev,
                        city=item.get("city", ""),
                        county=item.get("county", ""),
                        description=item.get("description", ""),
                        source=item.get("source", ""),
                        source_url=item.get("source_url"),
                        active=True,
                        reported_at=datetime.fromisoformat(item["reported_at"]),
                        updated_at=datetime.utcnow(),
                        lat=item.get("lat"),
                        lon=item.get("lon"),
                    )
                    await upsert_incident(inc)
                    await manager.broadcast("severity_upgrade", {
                        "id": item["id"],
                        "new_severity": new_sev.value,
                        "reason": result.get("reason", ""),
                    })
                    logger.info(f"[classifier] upgraded {item['id'][:8]} to {new_sev.value}")
                    count += 1
            except Exception as e:
                logger.debug(f"[classifier] error: {e}")

        return count


class TrendAgent(BaseAgent):
    """
    Detects surges: if >3 incidents of same type in same city in 2 hours,
    broadcasts a surge alert. Also emits periodic stats snapshots.
    """
    name = "trend_agent"
    interval_seconds = 120

    async def run_once(self) -> int:
        recent = await get_recent_incidents(hours=2, limit=200)
        stats  = await get_stats()

        # Broadcast stats snapshot
        await manager.broadcast("stats_update", stats)

        # Detect surges
        by_city_type: defaultdict[tuple, list] = defaultdict(list)
        for inc in recent:
            key = (inc.get("city", ""), inc.get("incident_type", ""))
            by_city_type[key].append(inc)

        surges = []
        for (city, itype), items in by_city_type.items():
            if len(items) >= 3 and city:
                surges.append({
                    "city": city,
                    "type": itype,
                    "count": len(items),
                    "message": f"SURGE: {len(items)} {itype} incidents in {city} in the last 2 hours",
                })

        if surges:
            await manager.broadcast("surge_alert", surges)
            logger.warning(f"[trend_agent] {len(surges)} surges detected")

        # Broadcast full active incident list for map refresh
        active = await get_active_incidents(300)
        await manager.broadcast("full_refresh", {"incidents": active, "stats": stats})

        return len(recent)


class ReportAgent(BaseAgent):
    """
    Generates an AI-written situational awareness briefing every hour.
    """
    name = "report_agent"
    interval_seconds = 3600   # hourly

    async def run_once(self) -> int:
        recent = await get_recent_incidents(hours=1, limit=100)
        if not recent:
            return 0

        # Summarize for LLM
        lines = []
        for inc in recent[:30]:
            lines.append(
                f"- [{inc.get('severity')}] {inc.get('incident_type')} in "
                f"{inc.get('city','Unknown')}: {inc.get('title')}"
            )
        summary_text = "\n".join(lines)

        briefing = await chat_json(
            "You are a Texas DPS situational awareness officer. "
            "Write a concise operational briefing from these incidents.",
            f"Last 1 hour of incidents:\n{summary_text}\n\n"
            "Respond with JSON: "
            '{"title": "Briefing title", "summary": "2-3 paragraph briefing", '
            '"priority_areas": ["city1","city2"], "threat_level": "LOW/MODERATE/HIGH/CRITICAL"}'
        )

        if briefing:
            await manager.broadcast("briefing", briefing)
            logger.info(f"[report_agent] briefing published, threat={briefing.get('threat_level')}")

        return len(recent)
