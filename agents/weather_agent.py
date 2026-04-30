"""
Weather & Emergency Alert Agent — NOAA CAP alerts for Texas.

Fixes vs original:
  - XML namespace handling made robust (NOAA changed ns prefix in 2024).
  - Per-feed error isolation.
  - Fallback to feedparser if ElementTree fails (some NOAA feeds are Atom).
  - 404/timeout per office handled gracefully.
"""
from __future__ import annotations
import asyncio
import hashlib
import logging
import httpx
import feedparser
from xml.etree import ElementTree as ET
from agents.base import BaseAgent
from agents.normalizer import NormalizationAgent
from core.models import RawItem

logger = logging.getLogger(__name__)

NOAA_FEEDS = [
    ("NOAA TX All Alerts",       "https://alerts.weather.gov/cap/tx.php?x=0"),
    ("NOAA Houston (IAH)",       "https://alerts.weather.gov/cap/wfo/iah.php?x=0"),
    ("NOAA Dallas (FWD)",        "https://alerts.weather.gov/cap/wfo/fwd.php?x=0"),
    ("NOAA San Antonio (EWX)",   "https://alerts.weather.gov/cap/wfo/ewx.php?x=0"),
    ("NOAA Lubbock (LUB)",       "https://alerts.weather.gov/cap/wfo/lub.php?x=0"),
    ("NOAA El Paso (EPZ)",       "https://alerts.weather.gov/cap/wfo/epz.php?x=0"),
    ("NOAA Amarillo (AMA)",      "https://alerts.weather.gov/cap/wfo/ama.php?x=0"),
    ("NOAA Corpus Christi (CRP)","https://alerts.weather.gov/cap/wfo/crp.php?x=0"),
    ("NOAA Midland (MAF)",       "https://alerts.weather.gov/cap/wfo/maf.php?x=0"),
    ("NOAA Austin/San Angelo (SJT)","https://alerts.weather.gov/cap/wfo/sjt.php?x=0"),
    # ── Rio Grande Valley / Brownsville ──────────────────────────────────────
    ("NOAA Brownsville (BRO)",     "https://alerts.weather.gov/cap/wfo/bro.php?x=0"),
]

_seen: set[str] = set()
_SEEN_MAX = 5000
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; TXSafetyBot/1.0)"}

# XML namespaces used by NOAA CAP feeds
NS = [
    {"atom": "http://www.w3.org/2005/Atom",
     "cap":  "urn:oasis:names:tc:emergency:cap:1.1"},
    {"atom": "http://www.w3.org/2005/Atom"},
    {},
]


def _extract_entries_xml(root) -> list[dict]:
    """Try multiple namespace combos to extract entries from NOAA CAP XML."""
    for ns in NS:
        entries = root.findall("atom:entry", ns) if ns else root.findall("entry")
        if entries:
            results = []
            for e in entries:
                def txt(tag):
                    for prefix in (["atom:"] if ns else [""]):
                        el = e.find(f"{prefix}{tag}", ns) if ns else e.find(tag)
                        if el is not None and el.text:
                            return el.text.strip()
                    return ""
                results.append({
                    "id":      txt("id") or txt("link"),
                    "title":   txt("title"),
                    "summary": txt("summary"),
                })
            return results
    return []


class WeatherAgent(BaseAgent):
    name = "weather_agent"
    interval_seconds = 300

    def __init__(self):
        super().__init__()
        self._normalizer = NormalizationAgent()

    async def _fetch_alerts(self, name: str, url: str) -> list[RawItem]:
        items = []
        try:
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(20.0),
                follow_redirects=True,
                headers=HEADERS,
            ) as c:
                r = await c.get(url)
                if r.status_code >= 400:
                    logger.debug(f"[weather_agent] HTTP {r.status_code}: {name}")
                    return items
                content = r.text

            # Try XML parse first, fall back to feedparser
            try:
                root    = ET.fromstring(content)
                entries = _extract_entries_xml(root)
            except ET.ParseError:
                feed    = feedparser.parse(content)
                entries = [
                    {
                        "id":      getattr(e, "id", "") or getattr(e, "link", ""),
                        "title":   getattr(e, "title", ""),
                        "summary": getattr(e, "summary", ""),
                    }
                    for e in (feed.entries or [])
                ]

            for entry in entries[:20]:
                alert_id = entry.get("id") or ""
                if not alert_id:
                    alert_id = hashlib.md5((entry.get("title","") + name).encode()).hexdigest()
                if alert_id in _seen:
                    continue
                if len(_seen) >= _SEEN_MAX:
                    _seen.clear()
                _seen.add(alert_id)
                title   = entry.get("title", "")
                summary = entry.get("summary", "")
                text    = f"[{name}] WEATHER ALERT: {title}. {summary}"[:2000]
                items.append(RawItem(
                    source=name,
                    raw_text=text,
                    url=alert_id if alert_id.startswith("http") else None,
                ))
        except (httpx.ConnectError, httpx.TimeoutException):
            logger.debug(f"[weather_agent] connection issue: {name}")
        except Exception as e:
            logger.debug(f"[weather_agent] {name}: {e}")
        return items

    async def run_once(self) -> int:
        tasks   = [self._fetch_alerts(n, u) for n, u in NOAA_FEEDS]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        raw_items: list[RawItem] = []
        for r in results:
            if isinstance(r, list):
                raw_items.extend(r)
        logger.info(f"[weather_agent] {len(raw_items)} weather alerts")
        count = 0
        for item in raw_items:
            try:
                await self._normalizer.normalize_and_save(item)
                count += 1
            except Exception as e:
                logger.warning(f"[weather_agent] normalize error: {e}")
        return count
