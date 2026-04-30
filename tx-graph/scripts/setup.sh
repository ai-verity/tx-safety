#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────
#  TX Safety Graph Server — Setup
#  Run: bash scripts/setup.sh
# ─────────────────────────────────────────────────────────
set -e

THIS_SCRIPT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$THIS_SCRIPT/.." && pwd)"

echo ""
echo "╔══════════════════════════════════════════════╗"
echo "║  TX SAFETY GRAPH — SETUP                    ║"
echo "╚══════════════════════════════════════════════╝"
echo ""
echo "  Project root: $PROJECT_ROOT"

if [ ! -f "$PROJECT_ROOT/requirements.txt" ]; then
  echo "ERROR: requirements.txt not found at $PROJECT_ROOT"
  exit 1
fi

echo "▸ Creating virtual environment..."
python3 -m venv "$PROJECT_ROOT/.venv"
source "$PROJECT_ROOT/.venv/bin/activate"

echo "▸ Upgrading pip..."
pip install --upgrade pip setuptools wheel --quiet

echo "▸ Installing pydantic-core binary wheel..."
pip install "pydantic-core==2.46.3" "pydantic==2.13.3" --only-binary=:all: --quiet

echo "▸ Installing dependencies..."
pip install -r "$PROJECT_ROOT/requirements.txt" --quiet

echo ""
echo "╔══════════════════════════════════════════════╗"
echo "║  SETUP COMPLETE                              ║"
echo "║                                              ║"
echo "║  Start tx-safety first (port 8007), then:   ║"
echo "║    bash scripts/start.sh                     ║"
echo "║                                              ║"
echo "║  Graph UI: http://localhost:8009             ║"
echo "╚══════════════════════════════════════════════╝"
echo ""
