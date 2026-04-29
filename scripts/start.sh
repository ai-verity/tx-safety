#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────
#  TX Public Safety — Start script  (port 8006)
# ─────────────────────────────────────────────────────────
set -e

THIS_SCRIPT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$THIS_SCRIPT/.." && pwd)"

echo ""
echo "╔══════════════════════════════════════════════╗"
echo "║  TX PUBLIC SAFETY — STARTING                ║"
echo "╚══════════════════════════════════════════════╝"
echo ""
echo "  Project root: $PROJECT_ROOT"

if [ ! -d "$PROJECT_ROOT/.venv" ]; then
  echo "ERROR: Run setup first:  bash $THIS_SCRIPT/setup.sh"
  exit 1
fi

source "$PROJECT_ROOT/.venv/bin/activate"
cd "$PROJECT_ROOT"

# Start Ollama if installed but not running
if command -v ollama &>/dev/null; then
  if ! pgrep -x "ollama" > /dev/null 2>&1; then
    echo "▸ Starting Ollama..."
    ollama serve > /tmp/ollama.log 2>&1 &
    sleep 2
    echo "  Ollama started ✓"
  else
    echo "▸ Ollama already running ✓"
  fi
fi

echo "▸ Starting TX Public Safety server on port 8006..."
echo ""
echo "  Dashboard  : http://localhost:8006"
echo "  Health     : http://localhost:8006/api/health"
echo "  VLM Prompts: http://localhost:8006/api/vlm/prompts"
echo "  API Docs   : http://localhost:8006/docs"
echo ""
echo "  Press Ctrl+C to stop"
echo ""

PYTHONPATH="$PROJECT_ROOT" uvicorn main:app \
  --host 0.0.0.0 \
  --port 8006 \
  --reload \
  --log-level info
