#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────
#  TX Safety Graph Server — port 8009
#  Run: bash scripts/start.sh
# ─────────────────────────────────────────────────────────
set -e

THIS_SCRIPT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$THIS_SCRIPT/.." && pwd)"

echo ""
echo "╔══════════════════════════════════════════════╗"
echo "║  TX SAFETY GRAPH SERVER — STARTING          ║"
echo "╚══════════════════════════════════════════════╝"
echo ""
echo "  Project root : $PROJECT_ROOT"

if [ ! -d "$PROJECT_ROOT/.venv" ]; then
  echo "ERROR: Run setup first:  bash $THIS_SCRIPT/setup.sh"
  exit 1
fi

source "$PROJECT_ROOT/.venv/bin/activate"
cd "$PROJECT_ROOT"

# Auto-detect tx-safety DB location
if [ -z "$TX_SAFETY_DB" ]; then
  CANDIDATE="$(cd "$PROJECT_ROOT/.." && pwd)/tx-safety/data/incidents.db"
  if [ -f "$CANDIDATE" ]; then
    export TX_SAFETY_DB="$CANDIDATE"
    echo "  DB found     : $TX_SAFETY_DB"
  else
    echo "  ⚠  DB not found at $CANDIDATE"
    echo "     Set TX_SAFETY_DB env var, or start tx-safety first."
    echo "     Graph will show 503 until the DB exists."
  fi
else
  echo "  DB path      : $TX_SAFETY_DB"
fi

echo ""
echo "  Graph UI     : http://localhost:8009"
echo "  API          : http://localhost:8009/api/graph"
echo "  Health       : http://localhost:8009/api/health"
echo "  Docs         : http://localhost:8009/docs"
echo ""
echo "  Press Ctrl+C to stop"
echo ""

PYTHONPATH="$PROJECT_ROOT" uvicorn main:app \
  --host 0.0.0.0 \
  --port 8009 \
  --log-level info
