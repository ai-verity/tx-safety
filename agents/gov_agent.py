"""
Government Data Agent — polls official Texas public safety data sources.

Fixes vs original:
  - Per-feed 404/error isolation — one dead URL cannot crash the cycle.
  - Verified-live feed URLs as of April 2025.
  - TxDPS RSS URL corrected (old URL returned 404).
  - HTTP 4xx logged at DEBUG not ERROR.
"""
from __future__ import annotations
import asyncio
import hashlib
import logging
import feedparser
import httpx
from agents.base import BaseAgent
from agents.normalizer import NormalizationAgent
from core.models import RawItem

logger = logging.getLogger(__name__)

GOV_FEEDS = [
    # Texas Department of Public Safety — news/press
    ("TxDPS News",              "https://www.dps.texas.gov/rss/vNewsRSS.cfm"),
    # Texas Division of Emergency Management
    ("Texas DEM",               "https://tdem.texas.gov/feed/"),
    # NOAA Weather Alerts (state-level CAP feed)
    ("NOAA TX Alerts",          "https://alerts.weather.gov/cap/tx.php?x=0"),
    # TxDOT Newsroom
    ("TxDOT Newsroom",          "https://www.txdot.gov/about/newsroom/news-releases.html.rss"),
    # Texas Attorney General
    ("TX Attorney General",     "https://www.texasattorneygeneral.gov/news/rss"),
    # FBI Field Offices in Texas
    ("FBI Dallas",              "https://www.fbi.gov/contact-us/field-offices/dallas/news/rss"),
    ("FBI Houston",             "https://www.fbi.gov/contact-us/field-offices/houston/news/rss"),
    ("FBI San Antonio",         "https://www.fbi.gov/contact-us/field-offices/sanantonio/news/rss"),
    ("FBI El Paso",             "https://www.fbi.gov/contact-us/field-offices/elpaso/news/rss"),
    # US Marshals Service
    ("US Marshals SWRO",        "https://www.usmarshals.gov/news/rss"),
    # DEA Dallas Division
    ("DEA Dallas",              "https://www.dea.gov/press-releases/rss"),
]

_seen: set[str] = set()
_SEEN_MAX = 5000
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; TXSafetyBot/1.0)"}


class GovDataAgent(BaseAgent):
    name = "gov_data_agent"
    interval_seconds = 180

    def __init__(self):
        super().__init__()
        self._normalizer = NormalizationAgent()

    def _entry_id(self, entry) -> str:
        raw = getattr(entry, "id", "") or getattr(entry, "link", "") or getattr(entry, "title", "")
        return hashlib.md5(raw.encode()).hexdigest()

    async def _fetch(self, name: str, url: str) -> list[RawItem]:
        items = []
        try:
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(20.0),
                follow_redirects=True,
                headers=HEADERS,
            ) as c:
                r = await c.get(url)
                if r.status_code == 404:
                    logger.debug(f"[gov_agent] 404 — feed missing: {name} {url}")
                    return items
                if r.status_code >= 400:
                    logger.debug(f"[gov_agent] HTTP {r.status_code}: {name}")
                    return items
                feed = feedparser.parse(r.text)
        except (httpx.ConnectError, httpx.TimeoutException) as e:
            logger.debug(f"[gov_agent] {name}: {type(e).__name__}")
            return items
        except Exception as e:
            logger.debug(f"[gov_agent] {name}: {e}")
            return items

        for entry in (feed.entries or [])[:15]:
            eid = self._entry_id(entry)
            if eid in _seen:
                continue
            if len(_seen) >= _SEEN_MAX:
                _seen.clear()
            _seen.add(eid)
            title   = getattr(entry, "title", "") or ""
            summary = getattr(entry, "summary", "") or ""
            text    = f"[{name}] {title}. {summary}"[:2000]
            items.append(RawItem(
                source=name,
                raw_text=text,
                url=getattr(entry, "link", None),
            ))
        return items

    async def run_once(self) -> int:
        tasks   = [self._fetch(n, u) for n, u in GOV_FEEDS]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        raw_items: list[RawItem] = []
        for r in results:
            if isinstance(r, list):
                raw_items.extend(r)
        logger.info(f"[gov_agent] {len(raw_items)} new gov items")
        count = 0
        for item in raw_items:
            try:
                await self._normalizer.normalize_and_save(item)
                count += 1
            except Exception as e:
                logger.warning(f"[gov_agent] normalize error: {e}")
        return count
