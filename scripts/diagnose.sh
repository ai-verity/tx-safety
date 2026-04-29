#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────
#  TX Public Safety — Diagnostic script
#  Run this when agents show idle or HTTP 404 errors.
#  Usage: bash scripts/diagnose.sh
# ─────────────────────────────────────────────────────────

THIS_SCRIPT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$THIS_SCRIPT/.." && pwd)"

echo ""
echo "╔══════════════════════════════════════════════╗"
echo "║  TX PUBLIC SAFETY — DIAGNOSTICS             ║"
echo "╚══════════════════════════════════════════════╝"
echo ""

PASS="✓"; FAIL="✗"; WARN="⚠"

# ── 1. Python ────────────────────────────────────────────
echo "── Python ──────────────────────────────────────"
if command -v python3 &>/dev/null; then
  PYVER=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')")
  echo "  $PASS Python $PYVER"
else
  echo "  $FAIL Python not found — install: brew install python@3.12"
fi

# ── 2. Virtual environment ───────────────────────────────
echo ""
echo "── Virtual Environment ──────────────────────────"
if [ -d "$PROJECT_ROOT/.venv" ]; then
  echo "  $PASS .venv exists at $PROJECT_ROOT/.venv"
  source "$PROJECT_ROOT/.venv/bin/activate" 2>/dev/null
  # Check key packages
  for pkg in fastapi uvicorn openai langgraph pydantic aiosqlite httpx feedparser; do
    if python3 -c "import $pkg" 2>/dev/null; then
      VER=$(python3 -c "import $pkg; print(getattr($pkg,'__version__','?'))" 2>/dev/null)
      echo "  $PASS $pkg $VER"
    else
      echo "  $FAIL $pkg NOT installed — run: bash scripts/setup.sh"
    fi
  done
else
  echo "  $FAIL .venv not found — run: bash scripts/setup.sh"
fi

# ── 3. Ollama ────────────────────────────────────────────
echo ""
echo "── Ollama (LLM backend) ─────────────────────────"
if command -v ollama &>/dev/null; then
  echo "  $PASS ollama binary found"
  if curl -sf http://localhost:11434/api/tags >/dev/null 2>&1; then
    echo "  $PASS ollama is running on port 11434"
    if ollama list 2>/dev/null | grep -q "qwen2.5:14b"; then
      echo "  $PASS model qwen2.5:14b is pulled"
    else
      echo "  $FAIL model qwen2.5:14b NOT pulled"
      echo "       Fix: ollama pull qwen2.5:14b"
    fi
  else
    echo "  $FAIL ollama is NOT running"
    echo "       Fix: ollama serve &"
    echo "       (agents will show 'idle' and incidents won't be AI-classified)"
  fi
else
  echo "  $FAIL ollama not installed"
  echo "       Fix: brew install ollama"
fi

# ── 4. Server ────────────────────────────────────────────
echo ""
echo "── Server (port 8006) ───────────────────────────"
if curl -sf http://localhost:8006/api/health >/dev/null 2>&1; then
  echo "  $PASS server is responding on port 8006"
  HEALTH=$(curl -sf http://localhost:8006/api/health 2>/dev/null)
  echo "  Health response: $HEALTH" | head -c 300
  echo ""
else
  echo "  $FAIL server not responding on port 8006"
  echo "       Fix: bash scripts/start.sh"
  # Check if port is in use by something else
  if lsof -i :8006 >/dev/null 2>&1; then
    echo "  $WARN port 8006 is already in use by:"
    lsof -i :8006 | head -4
  fi
fi

# ── 5. Network / Feed reachability ───────────────────────
echo ""
echo "── Feed Reachability (sample) ───────────────────"
FEEDS=(
  "NOAA TX|https://alerts.weather.gov/cap/tx.php?x=0"
  "KXAN Austin|https://www.kxan.com/feed/"
  "KHOU Houston|https://www.khou.com/feeds/rss/news/local/"
  "Texas Tribune|https://www.texastribune.org/feeds/all/"
  "FBI Dallas|https://www.fbi.gov/contact-us/field-offices/dallas/news/rss"
)
for entry in "${FEEDS[@]}"; do
  NAME="${entry%%|*}"
  URL="${entry##*|}"
  STATUS=$(curl -o /dev/null -sf -w "%{http_code}" --max-time 8 "$URL" 2>/dev/null)
  if [ "$STATUS" = "200" ]; then
    echo "  $PASS $NAME ($STATUS)"
  elif [ "$STATUS" = "301" ] || [ "$STATUS" = "302" ]; then
    echo "  $WARN $NAME (redirect $STATUS — likely OK)"
  elif [ -z "$STATUS" ]; then
    echo "  $FAIL $NAME (timeout / no response)"
  else
    echo "  $FAIL $NAME (HTTP $STATUS)"
  fi
done

# ── 6. Data directory ────────────────────────────────────
echo ""
echo "── Data / Database ──────────────────────────────"
if [ -f "$PROJECT_ROOT/data/incidents.db" ]; then
  SIZE=$(du -sh "$PROJECT_ROOT/data/incidents.db" | cut -f1)
  echo "  $PASS incidents.db exists ($SIZE)"
else
  echo "  $WARN incidents.db not yet created (created on first run)"
fi

# ── 7. Agent status from DB ──────────────────────────────
echo ""
echo "── Agent Status (from DB) ───────────────────────"
if [ -f "$PROJECT_ROOT/data/incidents.db" ]; then
  if command -v sqlite3 &>/dev/null; then
    sqlite3 "$PROJECT_ROOT/data/incidents.db" \
      "SELECT name, status, last_run, items_processed FROM agent_status ORDER BY name;" \
      2>/dev/null | while IFS='|' read -r name status last_run items; do
      if [ "$status" = "running" ] || [ "$status" = "polling" ]; then
        echo "  $PASS $name — $status (processed: $items)"
      elif [ "$status" = "idle" ]; then
        echo "  $WARN $name — idle (last run: $last_run, processed: $items)"
      elif [ "$status" = "error" ]; then
        echo "  $FAIL $name — ERROR (last run: $last_run)"
      else
        echo "  $WARN $name — $status"
      fi
    done
  else
    echo "  $WARN sqlite3 not available — install: brew install sqlite"
  fi
else
  echo "  $WARN database not yet created"
fi

echo ""
echo "────────────────────────────────────────────────"
echo "  Quick fix summary:"
echo "  1. Ollama not running?  →  ollama serve &"
echo "  2. Model not pulled?    →  ollama pull qwen2.5:14b"
echo "  3. Packages missing?    →  bash scripts/setup.sh"
echo "  4. Server not running?  →  bash scripts/start.sh"
echo "  5. Port in use?         →  change PORT in .env.example and start.sh"
echo "────────────────────────────────────────────────"
echo ""
