"""
TX Public Safety — FastAPI server.
Port: 8006
"""
from __future__ import annotations
import asyncio
import logging
import sys
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse

sys.path.insert(0, str(Path(__file__).parent))

from core.database import (
    init_db, get_active_incidents, get_recent_incidents,
    get_stats, get_all_agent_statuses,
)
from core.ws_manager import manager
from agents.orchestrator import run_orchestrator
from agents.vlm_routes import router as vlm_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("main")

app = FastAPI(title="TX Public Safety", version="1.0")
app.include_router(vlm_router)

STATIC_DIR = Path(__file__).parent / "dashboard" / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.on_event("startup")
async def startup():
    logger.info("TX Public Safety starting on port 8006...")
    await init_db()
    asyncio.create_task(run_orchestrator())
    logger.info("Orchestrator launched ✓")


# ─── WebSocket ────────────────────────────────────────────────
@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await manager.connect(ws)
    logger.info(f"WebSocket connected ({manager.count} total)")
    try:
        incidents = await get_active_incidents(200)
        stats     = await get_stats()
        agents    = await get_all_agent_statuses()
        await ws.send_json({
            "event": "full_refresh",
            "data":  {"incidents": incidents, "stats": stats, "agents": agents},
        })
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(ws)
    except Exception:
        manager.disconnect(ws)


# ─── Pages ────────────────────────────────────────────────────
@app.get("/")
async def root():
    return FileResponse(STATIC_DIR / "index.html")

@app.get("/vlm")
async def vlm_page():
    return FileResponse(STATIC_DIR / "vlm.html")


# ─── REST API ─────────────────────────────────────────────────
@app.get("/api/incidents")
async def api_incidents(limit: int = 300):
    return await get_active_incidents(limit)

@app.get("/api/incidents/recent")
async def api_recent(hours: int = 24, limit: int = 500):
    return await get_recent_incidents(hours, limit)

@app.get("/api/stats")
async def api_stats():
    return await get_stats()

@app.get("/api/agents")
async def api_agents():
    return await get_all_agent_statuses()

@app.post("/api/briefing")
async def api_briefing():
    from core.llm import chat_json, check_ollama, get_llm_base_url
    if not await check_ollama():
        return JSONResponse(status_code=503, content={
            "error": "LLM backend not reachable",
            "llm_base_url": get_llm_base_url(),
        })
    recent = await get_recent_incidents(hours=3, limit=50)
    if not recent:
        return {"title": "No recent incidents", "summary": "No incidents in last 3h.",
                "threat_level": "LOW", "priority_areas": []}
    lines = [
        f"- [{r.get('severity')}] {r.get('incident_type')} in {r.get('city','TX')}: {r.get('title')}"
        for r in recent[:25]
    ]
    result = await chat_json(
        "You are a Texas DPS situational awareness officer.",
        "Last 3 hours:\n" + "\n".join(lines) +
        '\n\nRespond with JSON: {"title":"...","summary":"2-3 paragraph briefing",'
        '"priority_areas":["city1"],"threat_level":"LOW/MODERATE/HIGH/CRITICAL"}',
    )
    if not result:
        logger.warning("[briefing] LLM returned empty or unparseable response")
        return {"title": "Unavailable", "summary": "LLM error.", "threat_level": "UNKNOWN"}
    return result

@app.get("/api/health")
async def health():
    from core.llm import check_ollama, get_llm_base_url, get_llm_model
    llm_ok = await check_ollama()
    stats  = await get_stats()
    agents = await get_all_agent_statuses()
    return {
        "status":      "ok",
        "port":        8006,
        "llm_ok":      llm_ok,
        "llm_base_url": get_llm_base_url(),
        "llm_model":   get_llm_model(),
        "ws_clients":  manager.count,
        "agent_count": len(agents),
        "agents":      agents,
        **stats,
    }
