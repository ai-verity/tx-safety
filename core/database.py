"""SQLite persistence layer for incidents and agent state."""
from __future__ import annotations
import json
import aiosqlite
from datetime import datetime
from pathlib import Path
from typing import Optional
from core.models import Incident, AgentStatus

DB_PATH = Path(__file__).parent.parent / "data" / "incidents.db"


async def init_db():
    DB_PATH.parent.mkdir(exist_ok=True)
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS incidents (
                id TEXT PRIMARY KEY,
                title TEXT,
                incident_type TEXT,
                severity TEXT,
                city TEXT,
                county TEXT,
                state TEXT,
                lat REAL,
                lon REAL,
                description TEXT,
                source TEXT,
                source_url TEXT,
                active INTEGER DEFAULT 1,
                reported_at TEXT,
                updated_at TEXT,
                resolved_at TEXT,
                embedding_id TEXT
            )
        """)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS agent_status (
                name TEXT PRIMARY KEY,
                status TEXT,
                last_run TEXT,
                items_processed INTEGER DEFAULT 0,
                error TEXT
            )
        """)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS raw_items (
                id TEXT PRIMARY KEY,
                source TEXT,
                raw_text TEXT,
                url TEXT,
                retrieved_at TEXT,
                processed INTEGER DEFAULT 0
            )
        """)
        await db.commit()


async def upsert_incident(inc: Incident):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("""
            INSERT OR REPLACE INTO incidents
            (id, title, incident_type, severity, city, county, state, lat, lon,
             description, source, source_url, active, reported_at, updated_at,
             resolved_at, embedding_id)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            inc.id, inc.title, inc.incident_type.value, inc.severity.value,
            inc.city, inc.county, inc.state, inc.lat, inc.lon,
            inc.description, inc.source, inc.source_url,
            1 if inc.active else 0,
            inc.reported_at.isoformat(), inc.updated_at.isoformat(),
            inc.resolved_at.isoformat() if inc.resolved_at else None,
            inc.embedding_id
        ))
        await db.commit()


async def get_active_incidents(limit: int = 200) -> list[dict]:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT * FROM incidents WHERE active=1 ORDER BY reported_at DESC LIMIT ?",
            (limit,)
        ) as cur:
            rows = await cur.fetchall()
            return [dict(r) for r in rows]


async def get_recent_incidents(hours: int = 24, limit: int = 500) -> list[dict]:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cutoff = datetime.utcnow().replace(microsecond=0).isoformat()
        async with db.execute(
            """SELECT * FROM incidents
               WHERE reported_at > datetime(?, '-' || ? || ' hours')
               ORDER BY reported_at DESC LIMIT ?""",
            (cutoff, hours, limit)
        ) as cur:
            rows = await cur.fetchall()
            return [dict(r) for r in rows]


async def resolve_incident(incident_id: str):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "UPDATE incidents SET active=0, resolved_at=? WHERE id=?",
            (datetime.utcnow().isoformat(), incident_id)
        )
        await db.commit()


async def upsert_agent_status(status: AgentStatus):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("""
            INSERT OR REPLACE INTO agent_status (name, status, last_run, items_processed, error)
            VALUES (?,?,?,?,?)
        """, (
            status.name, status.status,
            status.last_run.isoformat() if status.last_run else None,
            status.items_processed, status.error
        ))
        await db.commit()


async def get_all_agent_statuses() -> list[dict]:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute("SELECT * FROM agent_status ORDER BY name") as cur:
            rows = await cur.fetchall()
            return [dict(r) for r in rows]


async def get_stats() -> dict:
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute("SELECT COUNT(*) FROM incidents WHERE active=1") as c:
            active = (await c.fetchone())[0]
        async with db.execute("SELECT COUNT(*) FROM incidents WHERE active=1 AND severity='P1'") as c:
            p1 = (await c.fetchone())[0]
        async with db.execute(
            "SELECT COUNT(*) FROM incidents WHERE reported_at > datetime('now', '-24 hours')"
        ) as c:
            last24h = (await c.fetchone())[0]
        async with db.execute(
            "SELECT COUNT(*) FROM incidents WHERE active=0 AND resolved_at > datetime('now', '-24 hours')"
        ) as c:
            resolved = (await c.fetchone())[0]
        async with db.execute(
            """SELECT AVG((julianday(COALESCE(resolved_at, datetime('now'))) - julianday(reported_at))*1440)
               FROM incidents WHERE reported_at > datetime('now', '-24 hours')"""
        ) as c:
            avg_min_raw = (await c.fetchone())[0]
        avg_min = round(avg_min_raw) if avg_min_raw else 0
        return {
            "active": active, "p1": p1,
            "last24h": last24h, "resolved": resolved,
            "avg_response_min": avg_min
        }
