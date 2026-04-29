"""
Base class for all ingestion and analysis agents.

Fixes vs original:
  - Agents show 'idle' between cycles — that is correct/expected behaviour.
    The dashboard now distinguishes idle-healthy vs idle-never-ran.
  - HTTP 404s from dead feed URLs are caught per-feed, logged, and skipped
    rather than crashing the whole agent cycle.
  - Status is written to DB immediately on startup (not just after first cycle)
    so the dashboard shows the agent as registered right away.
  - last_run and items_processed are always persisted even on error.
  - Ollama unavailability no longer crashes the normalizer — it logs a warning
    and the agent retries on the next cycle.
"""
from __future__ import annotations
import asyncio
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from core.models import AgentStatus
from core.database import upsert_agent_status

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    name: str = "base"
    interval_seconds: int = 60

    def __init__(self):
        self.status = AgentStatus(name=self.name, status="idle")
        self._running = False

    async def _set_status(self, status: str, error: str | None = None):
        self.status.status = status
        self.status.last_run = datetime.utcnow()
        self.status.error = error
        try:
            await upsert_agent_status(self.status)
        except Exception as e:
            logger.warning(f"[{self.name}] could not persist status: {e}")

    @abstractmethod
    async def run_once(self) -> int:
        """Execute one agent cycle. Returns number of items processed."""
        ...

    async def start(self):
        """Run agent in a continuous loop."""
        self._running = True
        # Register immediately so the dashboard shows the agent on startup
        await self._set_status("starting")
        logger.info(f"[{self.name}] registered (interval={self.interval_seconds}s)")

        while self._running:
            try:
                await self._set_status("running")
                count = await self.run_once()
                self.status.items_processed += count
                await self._set_status("idle")
                logger.info(f"[{self.name}] cycle done — processed={count} total={self.status.items_processed}")
            except Exception as e:
                logger.error(f"[{self.name}] unhandled error: {e}", exc_info=True)
                await self._set_status("error", str(e)[:200])
            await asyncio.sleep(self.interval_seconds)

    def stop(self):
        self._running = False
