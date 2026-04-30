"""
TX Public Safety — Incident Graph Server  |  Port: 8009
Serves the social-network force-directed graph visualization.
Pulls live data from the tx-safety SQLite database when available,
falls back to rich demo data otherwise.
"""
from __future__ import annotations
import asyncio, json, logging, os, sys
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()
from datetime import datetime
from collections import defaultdict
from itertools import combinations
from typing import Optional

import aiosqlite
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("graph")

app = FastAPI(title="TX Safety Graph", version="1.0")

STATIC_DIR = Path(__file__).parent / "static"

DB_CANDIDATES = [
    os.getenv("TX_SAFETY_DB", ""),
    str(Path.home() / "Downloads" / "tx-safety" / "data" / "incidents.db"),
    str(Path.home() / "Downloads" / "tx-safety 2" / "data" / "incidents.db"),
    str(Path(__file__).parent.parent / "tx-safety" / "data" / "incidents.db"),
    str(Path(__file__).parent / "data" / "incidents.db"),
]

SEV_WEIGHT = {"P1": 100, "P2": 60, "P3": 30, "P4": 10}
SEV_COLOR  = {"P1": "#ff3d3d", "P2": "#ff9933", "P3": "#4da6ff", "P4": "#3dd68c"}
SEV_RADIUS = {"P1": 34, "P2": 26, "P3": 18, "P4": 11}
INC_COLOR  = {
    "Shooting": "#ff3d3d", "Vehicle Accident": "#ff9933", "Fire": "#ff6b2b",
    "Medical Emergency": "#4da6ff", "Pursuit": "#b39dfa", "Hazmat": "#fbbf24",
    "Burglary": "#f472b6", "Assault": "#fc8181", "Disturbance": "#6ee7b7",
    "Suspicious Activity": "#93c5fd", "Natural Disaster": "#2dd4bf",
    "Missing Person": "#f9a8d4", "Major Traffic": "#fcd34d", "Other": "#9ca3af",
}

# Kept so startup can cancel on shutdown
_push_task: Optional[asyncio.Task] = None


def find_db() -> Optional[str]:
    for p in DB_CANDIDATES:
        if p and Path(p).exists():
            return p
    return None


async def build_graph(hours: int = 24, min_weight: int = 0,
                      city_filter: str = "", severity_filter: str = "") -> dict[str, object]:
    db_path = find_db()
    if db_path:
        try:
            return await _from_db(db_path, hours, min_weight, city_filter, severity_filter)
        except Exception as e:
            logger.warning(f"DB error ({e}), using demo data")
    return _demo(min_weight=min_weight)


async def _from_db(db_path: str, hours: int, min_weight: int = 0,
                   city_filter: str = "", severity_filter: str = "") -> dict[str, object]:
    counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    async with aiosqlite.connect(db_path) as db:
        db.row_factory = aiosqlite.Row
        sql = """SELECT city, incident_type, severity, COUNT(*) as n
                 FROM incidents
                 WHERE reported_at > datetime('now','-'||?||' hours') AND active=1"""
        params: list = [hours]
        if city_filter:
            sql += " AND city = ?"
            params.append(city_filter)
        if severity_filter:
            sql += " AND severity = ?"
            params.append(severity_filter)
        sql += " GROUP BY city, incident_type, severity"
        async with db.execute(sql, params) as cur:
            for row in await cur.fetchall():
                city = (row["city"] or "Unknown").strip() or "Unknown"
                counts[city][row["incident_type"] or "Other"][row["severity"] or "P4"] += row["n"]

    nodes, edges = {}, []
    city_id, inc_id = {}, {}
    ni = 0

    for i, city in enumerate(sorted(counts)):
        cid = f"city_{i}"
        city_id[city] = cid
        total = sum(c for t in counts[city].values() for c in t.values())
        nodes[cid] = {"id": cid, "type": "city", "label": city, "city": city,
                      "color": "#0f172a", "border": "#4da6ff",
                      "radius": min(22 + total * 2, 58),
                      "total_incidents": total, "group": city}

    for city, inc_types in counts.items():
        for inc_type, sevs in inc_types.items():
            if not sevs:
                continue
            nid = f"inc_{ni}"; ni += 1
            inc_id[(city, inc_type)] = nid
            dom = max(sevs, key=lambda s: SEV_WEIGHT.get(s, 0))
            wt  = sum(SEV_WEIGHT.get(s, 0) * c for s, c in sevs.items())
            tot = sum(sevs.values())
            nodes[nid] = {
                "id": nid, "type": "incident", "label": inc_type,
                "incident_type": inc_type, "city": city,
                "city_node": city_id[city],
                "color": INC_COLOR.get(inc_type, "#9ca3af"),
                "border": SEV_COLOR.get(dom, "#4da6ff"),
                "radius": SEV_RADIUS.get(dom, 11),
                "dominant_sev": dom, "total_weight": wt,
                "total_count": tot, "severity_counts": dict(sevs), "group": city,
            }
            edges.append({"id": f"ec_{nid}", "source": city_id[city], "target": nid,
                          "weight": tot, "type": "city_link",
                          "color": "rgba(77,166,255,0.12)", "width": 1})

    edge_threshold = max(min_weight, 10)
    by_city = defaultdict(list)
    for (city, _), nid in inc_id.items():
        by_city[city].append(nid)
    for city, nids in by_city.items():
        for n1, n2 in combinations(nids, 2):
            w = min(nodes[n1]["total_weight"], nodes[n2]["total_weight"])
            if w < edge_threshold: continue
            dom = nodes[n1]["dominant_sev"] if nodes[n1]["total_weight"] >= nodes[n2]["total_weight"] else nodes[n2]["dominant_sev"]
            edges.append({"id": f"e_{n1}_{n2}", "source": n1, "target": n2,
                          "weight": w, "type": "incident_link",
                          "color": SEV_COLOR.get(dom, "#4da6ff") + "55",
                          "width": max(1, min(w // 20, 8)), "label": f"W:{w}"})

    return {"nodes": list(nodes.values()), "edges": edges,
            "meta": {"source": "live_db", "db_path": db_path, "hours": hours,
                     "node_count": len(nodes), "edge_count": len(edges),
                     "city_count": len(city_id),
                     "generated_at": datetime.utcnow().isoformat()}}


def _demo(min_weight: int = 0) -> dict[str, object]:
    raw = {
        "Houston":      [("Shooting",3,5,1,0),("Fire",0,4,6,2),("Pursuit",2,4,3,0),("Medical Emergency",0,8,10,4),("Assault",2,4,2,0)],
        "Dallas":       [("Shooting",4,3,1,0),("Vehicle Accident",0,7,9,3),("Burglary",0,0,8,12),("Disturbance",0,3,5,2)],
        "San Antonio":  [("Hazmat",1,2,1,0),("Fire",0,5,7,2),("Missing Person",0,4,6,2),("Suspicious Activity",0,0,9,11)],
        "Austin":       [("Disturbance",0,6,8,3),("Vehicle Accident",0,5,7,2),("Medical Emergency",0,7,9,4),("Shooting",1,3,1,0)],
        "Odessa":       [("Pursuit",3,4,2,0),("Shooting",2,3,1,0),("Vehicle Accident",0,4,5,2)],
        "Midland":      [("Fire",0,3,4,2),("Hazmat",0,2,3,1),("Burglary",0,0,5,7)],
        "El Paso":      [("Assault",2,4,2,0),("Shooting",3,2,1,0),("Natural Disaster",0,1,2,1)],
        "Lubbock":      [("Vehicle Accident",0,3,5,2),("Medical Emergency",0,4,6,3),("Fire",0,0,3,4)],
        "Corpus Christi":[("Natural Disaster",1,3,2,0),("Major Traffic",0,4,6,3),("Suspicious Activity",0,0,5,8)],
        "Amarillo":     [("Shooting",1,2,1,0),("Pursuit",0,3,4,2),("Disturbance",0,0,4,6)],
        "Waco":         [("Assault",0,2,4,2),("Shooting",1,2,1,0),("Vehicle Accident",0,3,4,3)],
        "Fort Worth":   [("Shooting",2,4,2,0),("Burglary",0,2,6,8),("Pursuit",1,3,3,0)],
        "Beaumont":     [("Fire",0,2,3,2),("Hazmat",1,2,1,0),("Medical Emergency",0,3,5,3)],
    }
    nodes, edges, inc_id = {}, [], {}
    ni = 0
    for i, (city, incidents) in enumerate(raw.items()):
        cid = f"city_{i}"
        total = sum(p1+p2+p3+p4 for _,p1,p2,p3,p4 in incidents)
        nodes[cid] = {"id":cid,"type":"city","label":city,"city":city,
                      "color":"#0f172a","border":"#4da6ff",
                      "radius":min(22+total,58),"total_incidents":total,"group":city}
        for inc_type,p1,p2,p3,p4 in incidents:
            nid = f"inc_{ni}"; ni += 1
            inc_id[(city,inc_type)] = nid
            sevs = {k:v for k,v in {"P1":p1,"P2":p2,"P3":p3,"P4":p4}.items() if v}
            dom = max(sevs, key=lambda s: SEV_WEIGHT.get(s,0)) if sevs else "P4"
            wt  = sum(SEV_WEIGHT.get(s,0)*c for s,c in sevs.items())
            tot = sum(sevs.values())
            nodes[nid] = {"id":nid,"type":"incident","label":inc_type,
                          "incident_type":inc_type,"city":city,"city_node":cid,
                          "color":INC_COLOR.get(inc_type,"#9ca3af"),
                          "border":SEV_COLOR.get(dom,"#4da6ff"),
                          "radius":SEV_RADIUS.get(dom,11),
                          "dominant_sev":dom,"total_weight":wt,
                          "total_count":tot,"severity_counts":sevs,"group":city}
            edges.append({"id":f"ec_{nid}","source":cid,"target":nid,
                          "weight":tot,"type":"city_link",
                          "color":"rgba(77,166,255,0.12)","width":1})

    edge_threshold = max(min_weight, 10)
    by_city = defaultdict(list)
    for (city,_),nid in inc_id.items(): by_city[city].append(nid)
    for city, nids in by_city.items():
        for n1,n2 in combinations(nids,2):
            w = min(nodes[n1]["total_weight"],nodes[n2]["total_weight"])
            if w < edge_threshold: continue
            dom = nodes[n1]["dominant_sev"] if nodes[n1]["total_weight"]>=nodes[n2]["total_weight"] else nodes[n2]["dominant_sev"]
            edges.append({"id":f"e_{n1}_{n2}","source":n1,"target":n2,
                          "weight":w,"type":"incident_link",
                          "color":SEV_COLOR.get(dom,"#4da6ff")+"55",
                          "width":max(1,min(w//25,7)),"label":f"W:{w}"})

    return {"nodes":list(nodes.values()),"edges":edges,
            "meta":{"source":"demo","hours":24,"node_count":len(nodes),
                    "edge_count":len(edges),"city_count":len(raw),
                    "generated_at":datetime.utcnow().isoformat(),
                    "note":"Demo data — point TX_SAFETY_DB env var at tx-safety/data/incidents.db for live data"}}


class WSManager:
    def __init__(self): self.active: list[WebSocket] = []
    async def connect(self, ws): await ws.accept(); self.active.append(ws)
    def disconnect(self, ws):
        if ws in self.active: self.active.remove(ws)
    async def broadcast(self, data):
        dead = []
        for ws in self.active:
            try: await ws.send_json(data)
            except Exception as e:
                logger.warning(f"WS broadcast error: {e}")
                dead.append(ws)
        for ws in dead: self.disconnect(ws)

ws_mgr = WSManager()


@app.on_event("startup")
async def startup():
    global _push_task
    logger.info("TX Safety Graph starting on :8009")
    async def push():
        while True:
            await asyncio.sleep(30)
            if ws_mgr.active:
                d = await build_graph()
                await ws_mgr.broadcast({"event":"graph_update","data":d})
    _push_task = asyncio.create_task(push())


@app.get("/")
async def root(): return FileResponse(STATIC_DIR / "index.html")

@app.get("/api/graph")
async def api_graph(hours: int = 24, min_weight: int = 0,
                    city_filter: str = "", severity_filter: str = ""):
    hours = max(1, min(hours, 168))
    return await build_graph(hours, min_weight=min_weight,
                             city_filter=city_filter, severity_filter=severity_filter)

@app.get("/api/graph/cities")
async def api_cities(hours: int = 24):
    hours = max(1, min(hours, 168))
    db_path = find_db()
    if not db_path:
        return []
    try:
        async with aiosqlite.connect(db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                """SELECT city, COUNT(*) as n FROM incidents
                   WHERE reported_at > datetime('now','-'||?||' hours') AND active=1
                   AND city IS NOT NULL AND city != ''
                   GROUP BY city ORDER BY n DESC""",
                (hours,)
            ) as cur:
                rows = await cur.fetchall()
                return [{"city": row["city"], "n": row["n"]} for row in rows]
    except Exception as e:
        logger.warning(f"Cities query error: {e}")
        return []

@app.get("/api/health")
async def health():
    db = find_db()
    d  = await build_graph()
    return {"status":"ok","port":8009,"db_connected":db is not None,"db_path":db,**d["meta"]}

@app.websocket("/ws")
async def ws_ep(ws: WebSocket):
    await ws_mgr.connect(ws)
    try:
        d = await build_graph()
        await ws.send_json({"event":"graph_init","data":d})
        while True: await ws.receive_text()
    except WebSocketDisconnect: ws_mgr.disconnect(ws)
    except Exception as e:
        logger.warning(f"WS endpoint error: {e}")
        ws_mgr.disconnect(ws)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
