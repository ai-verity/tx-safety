# TX Public Safety — Situational Awareness System

A fully local, API-key-free agentic AI system for real-time public safety
monitoring across all 254 Texas counties. Runs on your Mac with Ollama,
scales to H200 GPUs with vLLM — same code, one env-var change.

---

## Architecture

```
Data Sources          Ingestion Agents        Analysis Agents
─────────────         ────────────────        ───────────────
Local News RSS   ──►  NewsAgent               ThreatClassifier
Gov Feeds        ──►  GovDataAgent    ──►     TrendAgent
Reddit/Social    ──►  SocialAgent    DB/WS    ReportAgent
NOAA Weather     ──►  WeatherAgent
                          │
                    NormalizationAgent (LLM)
                          │
                    SQLite Database
                          │
                    FastAPI + WebSocket
                          │
                    Live Dashboard (browser)
```

---

## Quick Start (Mac)

### 1. Prerequisites

```bash
# Install Homebrew (if needed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Ollama
brew install ollama

# Install Python 3.12 (if needed)
brew install python@3.12
```

### 2. Setup

```bash
cd tx-safety
bash scripts/setup.sh
```

This will:
- Create a Python virtual environment
- Install all dependencies
- Pull the `qwen2.5:14b` model via Ollama (~9GB download, one-time)

### 3. Start

```bash
bash scripts/start.sh
```

Open your browser: **http://localhost:8006**

---

## What Each Agent Does

| Agent | Source | Interval |
|-------|--------|----------|
| `NewsAgent` | 14 Texas local news RSS feeds | 2 min |
| `GovDataAgent` | TxDPS, NOAA, FBI, US Marshals | 3 min |
| `SocialAgent` | Reddit (r/Texas, r/Houston, etc), Open Data portals | 90 sec |
| `WeatherAgent` | NOAA CAP alerts for all TX forecast offices | 5 min |
| `NormalizationAgent` | Called by all above — uses LLM to extract structured data | per-item |
| `ThreatClassifierAgent` | Re-evaluates severity of active incidents | 5 min |
| `TrendAgent` | Detects surges, broadcasts stats | 2 min |
| `ReportAgent` | Generates hourly AI briefings | 1 hour |

---

## Dashboard Features

- **Real-time Texas map** — incidents plotted by lat/lon with severity color coding
- **WebSocket updates** — new incidents appear instantly without page refresh
- **Filter bar** — filter map and feed by incident type
- **Agent status panel** — live status of all 7 agents
- **Incident type breakdown** — animated bar chart
- **Live feed** — scrollable list of recent incidents with descriptions
- **AI Briefings** — click "Request Briefing" for an LLM-generated situation report
- **Surge alerts** — banner appears when 3+ incidents of same type in same city in 2h
- **P1/P2 toasts** — priority alerts pop up in bottom-right corner

---

## Scaling to H200 GPUs

1. Deploy vLLM on your H200 server:
   ```bash
   docker run --gpus all vllm/vllm-openai:latest \
     --model Qwen/Qwen2.5-14B-Instruct \
     --port 8001 \
     --tensor-parallel-size 8
   ```

2. Set one environment variable:
   ```bash
   export LLM_BASE_URL=http://your-h200-server:8001/v1
   export LLM_MODEL=Qwen/Qwen2.5-14B-Instruct
   ```

3. Run the same code — no changes needed.

For full GPU deployment with Docker:
```bash
docker compose --profile gpu up
```

---

## Project Structure

```
tx-safety/
├── main.py                    # FastAPI server + WebSocket
├── requirements.txt
├── .env.example
├── Dockerfile
├── docker-compose.yml
├── core/
│   ├── models.py              # Pydantic data models
│   ├── database.py            # SQLite via aiosqlite
│   ├── llm.py                 # Ollama/vLLM client
│   ├── geocoder.py            # TX city geocoding (no API key)
│   └── ws_manager.py          # WebSocket broadcast hub
├── agents/
│   ├── base.py                # BaseAgent abstract class
│   ├── orchestrator.py        # LangGraph startup graph
│   ├── normalizer.py          # LLM extraction agent
│   ├── news_agent.py          # RSS news ingestion
│   ├── gov_agent.py           # Government feeds
│   ├── social_agent.py        # Reddit + open data
│   ├── weather_agent.py       # NOAA weather alerts
│   └── analysis_agents.py     # Classifier, trend, report
├── dashboard/
│   └── static/
│       └── index.html         # Full dashboard UI
├── data/
│   └── incidents.db           # SQLite database (auto-created)
└── scripts/
    ├── setup.sh
    └── start.sh
```

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Dashboard UI |
| WS | `/ws` | WebSocket real-time feed |
| GET | `/api/incidents` | All active incidents |
| GET | `/api/incidents/recent?hours=24` | Recent incidents |
| GET | `/api/stats` | Aggregate statistics |
| GET | `/api/agents` | Agent status list |
| POST | `/api/briefing` | Generate AI briefing |
| GET | `/api/health` | Health check |

---

## Notes

- **No API keys required** — all data sources are public RSS/JSON feeds
- **Privacy-first** — no data ever leaves your machine
- **LLM optional** — system runs without Ollama; incidents are stored as raw text
- **Rate limiting** — agents space their requests appropriately to be good citizens
- The NOAA, Reddit, and open data sources are genuinely free and public
- For production law enforcement use, consider adding authenticated CAD system feeds
