# TX Safety — Incident Network Graph

A standalone force-directed graph server (port **8009**) that visualizes public safety incident relationships across Texas cities. Built with FastAPI + D3 v7, it reads live data from the tx-safety SQLite database and falls back to rich demo data when the database is unavailable.

---

## What it shows

| Element | Represents |
|---------|-----------|
| **City node** | A Texas city — size scales with total incident count |
| **Incident node** | One incident type in one city (e.g. "Shooting in Houston") |
| **Node size** | Total severity weight (P1×100 + P2×60 + P3×30 + P4×10) |
| **Node color** | Incident type (consistent palette per type) |
| **Node border** | Dominant severity (red = P1, orange = P2, blue = P3, green = P4) |
| **Edge** | Co-occurrence of two incident types in the same city |
| **Edge weight** | Combined severity score of both incident types |
| **Edge color** | Severity-tinted with opacity (thin = low weight, thick = high) |
| **Cluster hull** | Translucent polygon grouping all nodes for a city |

---

## Quick start (local)

```bash
# 1. Start tx-safety first so the database is populated
cd tx-safety && bash scripts/start.sh          # runs on port 8007

# 2. In a new terminal — set up and start the graph server
cd tx-graph
bash scripts/setup.sh                          # creates .venv, installs deps
bash scripts/start.sh                          # runs on port 8009
```

Open **http://localhost:8009**

---

## Docker

The graph server is included in the root `docker-compose.yml` as the `tx-graph` service.

```bash
# From the repo root — starts tx-safety + tx-graph together
docker compose up tx-safety tx-graph

# With GPU/vLLM inference backend
docker compose --profile gpu up
```

The `tx-graph` container mounts `./data` from the `tx-safety` service so both share the same `incidents.db`.

Build the image standalone:

```bash
cd tx-graph
docker build -t tx-graph .
docker run -p 8009:8009 -e TX_SAFETY_DB=/app/data/incidents.db tx-graph
```

---

## Environment variables

Copy `.env.example` to `.env` and adjust as needed:

```bash
cp .env.example .env
```

| Variable | Default | Description |
|----------|---------|-------------|
| `TX_SAFETY_DB` | auto-detected | Absolute path to `incidents.db`. Tries several candidate paths if unset. |
| `HOST` | `0.0.0.0` | Bind address |
| `PORT` | `8009` | Listen port |

The server probes these candidate paths in order when `TX_SAFETY_DB` is not set:

1. `$TX_SAFETY_DB` env var
2. `~/Downloads/tx-safety/data/incidents.db`
3. `~/Downloads/tx-safety 2/data/incidents.db`
4. `../tx-safety/data/incidents.db` (sibling directory)
5. `./data/incidents.db` (local)

If none exist, the UI loads with built-in demo data covering 13 Texas cities.

---

## API reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/graph` | GET | Full graph payload |
| `/api/graph?hours=48` | GET | Incidents in last N hours (1–168) |
| `/api/graph?city_filter=Houston` | GET | Isolate one city |
| `/api/graph?severity_filter=P1` | GET | Filter to one severity level |
| `/api/graph?min_weight=20` | GET | Hide low-weight edges |
| `/api/graph/cities?hours=24` | GET | City list with incident counts (for dropdown) |
| `/api/health` | GET | Health check — db status, node/edge counts |
| `/docs` | GET | Auto-generated OpenAPI docs |
| `/ws` | WebSocket | Live push — sends `graph_init` on connect, `graph_update` every 30 s |

---

## Interactive controls

- **Scroll** — zoom in/out
- **Drag canvas** — pan
- **Drag any node** — reposition (simulation continues)
- **Click incident type** in legend — highlight all nodes of that type
- **Click city** in cluster panel — highlight that city's subgraph
- **Time window slider** — 1 h to 168 h (7 days)
- **Min edge weight** — filter out low-weight (low co-occurrence) edges
- **Link distance / Charge** — tune force simulation physics live
- **City filter dropdown** — isolate one city's subgraph

---

## Project structure

```
tx-graph/
├── main.py              # FastAPI server, graph builder, WebSocket push
├── requirements.txt     # Dependencies (fastapi, uvicorn, aiosqlite, python-dotenv)
├── Dockerfile           # Python 3.12-slim image, port 8009
├── .env.example         # Environment variable template
├── static/
│   ├── index.html       # D3 v7 force-directed graph UI
│   └── graph.html       # Alternate standalone graph page
└── scripts/
    ├── setup.sh         # Creates .venv and installs dependencies
    └── start.sh         # Activates venv, auto-detects DB, starts uvicorn
```

---

## Severity levels

| Level | Weight | Color | Meaning |
|-------|--------|-------|---------|
| P1 | 100 | Red `#ff3d3d` | Critical — active threat, life safety |
| P2 | 60 | Orange `#ff9933` | High — significant incident in progress |
| P3 | 30 | Blue `#4da6ff` | Medium — notable, non-critical |
| P4 | 10 | Green `#3dd68c` | Low — informational/minor |
