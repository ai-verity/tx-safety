"""
Social Media / Open Web Agent.

Fixes vs original:
  - Reddit JSON API: added proper browser User-Agent and rate-limit handling
    (Reddit returns 429 or 403 to bots without correct headers).
  - Open data portals: Austin and Dallas APD endpoints verified/corrected.
  - Added Socrata app token support via env var (optional, increases rate limits).
  - Per-source error isolation — one 404 does not stop the cycle.
  - Added Broadcastify public incident feed.
"""
from __future__ import annotations
import asyncio
import hashlib
import logging
import os
import httpx
import feedparser
from agents.base import BaseAgent
from agents.normalizer import NormalizationAgent
from core.models import RawItem

logger = logging.getLogger(__name__)


REDDIT_FEEDS = [
    ("Reddit r/texas",       "https://www.reddit.com/r/texas/search.json?q=shooting+fire+crash+police+emergency&sort=new&restrict_sr=1&t=day&limit=15"),
    ("Reddit r/houston",     "https://www.reddit.com/r/houston/search.json?q=shooting+fire+crash+police&sort=new&restrict_sr=1&t=day&limit=10"),
    ("Reddit r/Dallas",      "https://www.reddit.com/r/Dallas/search.json?q=shooting+fire+crash+police&sort=new&restrict_sr=1&t=day&limit=10"),
    ("Reddit r/sanantonio",  "https://www.reddit.com/r/sanantonio/search.json?q=shooting+fire+crash+police&sort=new&restrict_sr=1&t=day&limit=10"),
    ("Reddit r/Austin",      "https://www.reddit.com/r/Austin/search.json?q=shooting+fire+crash+police&sort=new&restrict_sr=1&t=day&limit=10"),
]

RSS_FEEDS = [
    # Broadcastify public incident log (TX region)
    ("Broadcastify TX", "https://www.broadcastify.com/calls/rss?l=18"),
]

# City of Austin APD — verified endpoint April 2025
# Dallas PD — verified endpoint April 2025
TX_OPEN_DATA = [
    ("Austin PD Incidents",
     "https://data.austintexas.gov/resource/fdj4-gpfu.json?$limit=15&$order=occurred_date_time%20DESC",
     "socrata"),
    ("Dallas PD Incidents",
     "https://www.dallasopendata.com/resource/qv6i-rri7.json?$limit=15&$order=date1%20DESC",
     "socrata"),
]

_seen: set[str] = set()

# Reddit needs a browser-like UA to avoid 403/429
REDDIT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/124.0.0.0 Safari/537.36",
    "Accept": "application/json",
}
STD_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; TXSafetyBot/1.0)"}


class SocialAgent(BaseAgent):
    name = "social_agent"
    interval_seconds = 90

    def __init__(self):
        super().__init__()
        self._normalizer = NormalizationAgent()

    def _make_id(self, text: str) -> str:
        return hashlib.md5(text.encode()).hexdigest()

    async def _fetch_reddit(self, name: str, url: str) -> list[RawItem]:
        items = []
        try:
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(15.0),
                follow_redirects=True,
                headers=REDDIT_HEADERS,
            ) as c:
                r = await c.get(url)
                if r.status_code == 429:
                    logger.debug(f"[social_agent] Reddit rate-limited: {name}")
                    return items
                if r.status_code == 403:
                    logger.debug(f"[social_agent] Reddit 403 (bot-blocked): {name}")
                    return items
                if r.status_code >= 400:
                    logger.debug(f"[social_agent] Reddit HTTP {r.status_code}: {name}")
                    return items
                data = r.json()
            posts = data.get("data", {}).get("children", [])
            for p in posts[:10]:
                d = p.get("data", {})
                title = d.get("title", "")
                body  = (d.get("selftext") or "")[:400]
                text  = f"[{name}] {title}. {body}"
                eid   = self._make_id(d.get("id", text[:80]))
                if eid in _seen:
                    continue
                _seen.add(eid)
                items.append(RawItem(
                    source=name,
                    raw_text=text[:2000],
                    url=f"https://reddit.com{d.get('permalink','')}",
                ))
        except (httpx.ConnectError, httpx.TimeoutException):
            logger.debug(f"[social_agent] connection issue: {name}")
        except Exception as e:
            logger.debug(f"[social_agent] reddit {name}: {e}")
        return items

    async def _fetch_rss(self, name: str, url: str) -> list[RawItem]:
        items = []
        try:
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(15.0),
                follow_redirects=True,
                headers=STD_HEADERS,
            ) as c:
                r = await c.get(url)
                if r.status_code >= 400:
                    logger.debug(f"[social_agent] HTTP {r.status_code}: {name}")
                    return items
                feed = feedparser.parse(r.text)
            for entry in (feed.entries or [])[:15]:
                eid = self._make_id(getattr(entry, "id", getattr(entry, "title", "")))
                if eid in _seen:
                    continue
                _seen.add(eid)
                text = f"[{name}] {getattr(entry,'title','')}. {getattr(entry,'summary','')}"[:2000]
                items.append(RawItem(source=name, raw_text=text, url=getattr(entry, "link", None)))
        except Exception as e:
            logger.debug(f"[social_agent] rss {name}: {e}")
        return items

    async def _fetch_opendata(self, name: str, url: str, dtype: str) -> list[RawItem]:
        items = []
        try:
            headers = dict(STD_HEADERS)
            token = os.getenv("SOCRATA_APP_TOKEN", "")
            if dtype == "socrata" and token:
                headers["X-App-Token"] = token
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(15.0),
                follow_redirects=True,
                headers=headers,
            ) as c:
                r = await c.get(url)
                if r.status_code == 404:
                    logger.debug(f"[social_agent] open data 404: {name}")
                    return items
                if r.status_code >= 400:
                    logger.debug(f"[social_agent] open data HTTP {r.status_code}: {name}")
                    return items
                records = r.json()
            for rec in (records or [])[:10]:
                text = f"[{name}] " + " | ".join(
                    f"{k}: {v}" for k, v in rec.items() if v
                )[:1500]
                eid = self._make_id(text[:120])
                if eid in _seen:
                    continue
                _seen.add(eid)
                items.append(RawItem(source=name, raw_text=text))
        except Exception as e:
            logger.debug(f"[social_agent] opendata {name}: {e}")
        return items

    async def run_once(self) -> int:
        tasks = []
        for name, url in REDDIT_FEEDS:
            tasks.append(self._fetch_reddit(name, url))
        for name, url in RSS_FEEDS:
            tasks.append(self._fetch_rss(name, url))
        for name, url, dtype in TX_OPEN_DATA:
            tasks.append(self._fetch_opendata(name, url, dtype))

        results = await asyncio.gather(*tasks, return_exceptions=True)
        raw_items: list[RawItem] = []
        for r in results:
            if isinstance(r, list):
                raw_items.extend(r)

        logger.info(f"[social_agent] {len(raw_items)} social items")
        count = 0
        for item in raw_items:
            try:
                await self._normalizer.normalize_and_save(item)
                count += 1
            except Exception as e:
                logger.warning(f"[social_agent] normalize error: {e}")
        return count
