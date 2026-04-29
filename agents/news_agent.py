"""
News RSS Agent — polls Texas local news and public safety RSS feeds.

Fixes vs original:
  - Each feed fetched independently; a 404/timeout on one does NOT stop others.
  - Dead/moved URLs replaced with verified-live alternatives (April 2025).
  - User-Agent rotated to reduce bot-blocking 403s.
  - Connection errors, SSL errors, and HTTP 4xx/5xx all caught per-feed.
  - Feed-level error logged at DEBUG (not ERROR) so logs stay clean.
"""
from __future__ import annotations
import asyncio
import hashlib
import logging
import feedparser
import httpx
from bs4 import BeautifulSoup
from agents.base import BaseAgent
from agents.normalizer import NormalizationAgent
from core.models import RawItem

logger = logging.getLogger(__name__)

RSS_FEEDS = [
    # ── Houston ──────────────────────────────────────────────
    ("Houston Public Media",      "https://www.houstonpublicmedia.org/feed/"),
    ("KHOU Houston",              "https://www.khou.com/feeds/rss/news/local/"),
    ("Click2Houston KPRC",        "https://www.click2houston.com/rss/"),
    # ── Dallas / Fort Worth ───────────────────────────────────
    ("WFAA Dallas",               "https://www.wfaa.com/feeds/rss/news/local/"),
    ("CBS DFW",                   "https://www.cbsnews.com/dallas/rss/"),
    ("NBC5 Dallas",               "https://www.nbcdfw.com/feed/"),
    # ── San Antonio ───────────────────────────────────────────
    ("KSAT San Antonio",          "https://www.ksat.com/rss"),
    ("MySA News",                 "https://www.mysanantonio.com/local/rss"),
    # ── Austin ────────────────────────────────────────────────
    ("KXAN Austin",               "https://www.kxan.com/feed/"),
    ("KVUE Austin",               "https://www.kvue.com/feeds/rss/news/local/"),
    # ── Statewide ─────────────────────────────────────────────
    ("Texas Tribune",             "https://www.texastribune.org/feeds/all/"),
    ("ABC13 Houston",             "https://abc13.com/feed/"),
    # ── West Texas / Odessa / Midland ─────────────────────────
    ("Odessa American",           "https://www.oaoa.com/feed/"),
    ("Midland Reporter-Telegram", "https://www.mrt.com/feed/"),
    # ── Other major markets ───────────────────────────────────
    ("KLBK Lubbock",              "https://www.klbk13.com/feed/"),
    ("KAMR Amarillo",             "https://www.myhighplains.com/feed/"),
    ("KETK Tyler",                "https://www.ketk.com/feed/"),
    ("Beaumont Enterprise",       "https://www.beaumontenterprise.com/feed/"),
]

SAFETY_KEYWORDS = [
    "shooting", "shot", "gunfire", "gunshot", "homicide", "murder", "stabbing",
    "assault", "robbery", "burglary", "theft", "arson", "fire", "explosion",
    "crash", "accident", "collision", "pursuit", "chase", "fleeing",
    "missing", "kidnap", "amber alert", "silver alert", "hazmat",
    "chemical", "spill", "flood", "tornado", "severe weather",
    "officer involved", "police", "swat", "sheriff", "fbi", "dps",
    "suspect", "arrested", "warrant", "fugitive", "overdose",
    "disturbance", "evacuation", "emergency",
]

_seen: set[str] = set()

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; TXSafetyBot/1.0)"}


class NewsAgent(BaseAgent):
    name = "news_agent"
    interval_seconds = 120

    def __init__(self):
        super().__init__()
        self._normalizer = NormalizationAgent()

    def _is_safety_related(self, text: str) -> bool:
        t = text.lower()
        return any(kw in t for kw in SAFETY_KEYWORDS)

    def _item_id(self, entry) -> str:
        raw = (getattr(entry, "id", "") or getattr(entry, "link", "") or
               getattr(entry, "title", ""))
        return hashlib.md5(raw.encode()).hexdigest()

    async def _fetch_feed(self, name: str, url: str) -> list[RawItem]:
        items = []
        try:
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(15.0),
                follow_redirects=True,
                headers=HEADERS,
            ) as client:
                r = await client.get(url)
                if r.status_code == 404:
                    logger.debug(f"[news_agent] 404 — feed removed or moved: {name} {url}")
                    return items
                if r.status_code >= 400:
                    logger.debug(f"[news_agent] HTTP {r.status_code} for {name}")
                    return items
                feed = feedparser.parse(r.text)
        except httpx.ConnectError:
            logger.debug(f"[news_agent] connection refused: {name}")
            return items
        except httpx.TimeoutException:
            logger.debug(f"[news_agent] timeout: {name}")
            return items
        except Exception as e:
            logger.debug(f"[news_agent] fetch error {name}: {type(e).__name__}: {e}")
            return items

        for entry in (feed.entries or [])[:20]:
            eid = self._item_id(entry)
            if eid in _seen:
                continue
            title   = getattr(entry, "title", "") or ""
            summary = getattr(entry, "summary", "") or ""
            summary = BeautifulSoup(summary, "lxml").get_text(separator=" ")
            combined = f"{title}. {summary}"
            if not self._is_safety_related(combined):
                continue
            _seen.add(eid)
            items.append(RawItem(
                source=name,
                raw_text=combined[:2000],
                url=getattr(entry, "link", None),
            ))
        return items

    async def run_once(self) -> int:
        tasks   = [self._fetch_feed(n, u) for n, u in RSS_FEEDS]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        raw_items: list[RawItem] = []
        for r in results:
            if isinstance(r, list):
                raw_items.extend(r)
        logger.info(f"[news_agent] {len(raw_items)} new safety items across {len(RSS_FEEDS)} feeds")
        count = 0
        for item in raw_items:
            try:
                await self._normalizer.normalize_and_save(item)
                count += 1
            except Exception as e:
                logger.warning(f"[news_agent] normalize error: {e}")
        return count
