#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────
#  TX Public Safety — One-click Mac setup script
#  Usage (run from ANY directory):
#    bash /path/to/tx-safety/scripts/setup.sh
#  Or from inside tx-safety/:
#    bash scripts/setup.sh
# ─────────────────────────────────────────────────────────
set -e

# Resolve project root as the directory containing THIS script's parent
THIS_SCRIPT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$THIS_SCRIPT/.." && pwd)"

echo ""
echo "╔══════════════════════════════════════════════╗"
echo "║  TX PUBLIC SAFETY — SETUP                   ║"
echo "╚══════════════════════════════════════════════╝"
echo ""
echo "  Project root: $PROJECT_ROOT"
echo ""

# Sanity check — make sure requirements.txt exists where we expect it
if [ ! -f "$PROJECT_ROOT/requirements.txt" ]; then
  echo "ERROR: requirements.txt not found at $PROJECT_ROOT/requirements.txt"
  echo ""
  echo "Make sure you extracted the archive correctly:"
  echo "  tar -xzf tx-safety.tar.gz"
  echo "  bash tx-safety/scripts/setup.sh"
  exit 1
fi

cd "$PROJECT_ROOT"
echo "  requirements.txt found ✓"

# ── Python ──
echo "▸ Checking Python..."
PYTHON_BIN=""
for candidate in python python3.13 python3.12 python3.11 python3.10 python3; do
  if command -v "$candidate" &>/dev/null; then
    VER=$("$candidate" -c "import sys; print(sys.version_info.major * 10 + sys.version_info.minor)" 2>/dev/null)
    if [ "${VER:-0}" -ge 40 ]; then
      PYTHON_BIN="$candidate"
      break
    fi
  fi
done

if [ -z "$PYTHON_BIN" ]; then
  echo "  ERROR: Python 3.10+ required. Install via: brew install python@3.12"
  exit 1
fi
PYVER=$("$PYTHON_BIN" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "  Python $PYVER ($PYTHON_BIN) ✓"

# ── Virtual environment ──
echo "▸ Creating virtual environment..."
"$PYTHON_BIN" -m venv "$PROJECT_ROOT/.venv"
source "$PROJECT_ROOT/.venv/bin/activate"

# ── Upgrade build tools ──
echo "▸ Upgrading pip, setuptools, wheel..."
pip install --upgrade pip setuptools wheel --quiet
echo "  Build tools upgraded ✓"

# ── Pre-install pydantic-core binary wheel ──
# Python 3.14 pre-built wheels ship in pydantic-core 2.46+.
# --only-binary=:all: forces the wheel, never compiles from Rust source.
echo "▸ Installing pydantic-core binary wheel (Python 3.14 compatible)..."
pip install "pydantic-core==2.46.3" "pydantic==2.13.3" \
  --only-binary=:all: --quiet
echo "  pydantic-core installed ✓"

# ── Remaining dependencies ──
echo "▸ Installing remaining dependencies..."
pip install -r "$PROJECT_ROOT/requirements.txt" --quiet
echo "  Dependencies installed ✓"

# ── Data directory ──
mkdir -p "$PROJECT_ROOT/data"
echo "  Data directory ready ✓"

# ── Ollama check ──
echo ""
echo "▸ Checking Ollama..."
if command -v ollama &>/dev/null; then
  echo "  Ollama installed ✓"
  if ollama list 2>/dev/null | grep -q "qwen2.5:14b"; then
    echo "  Model qwen2.5:14b already pulled ✓"
  else
    echo "  Pulling qwen2.5:14b (this may take a few minutes)..."
    ollama pull qwen2.5:14b
    echo "  Model pulled ✓"
  fi
else
  echo ""
  echo "  ⚠  Ollama not found. Install it first:"
  echo "     brew install ollama"
  echo "     ollama serve &"
  echo "     ollama pull qwen2.5:14b"
  echo ""
  echo "  The system will work without Ollama but incidents won't be"
  echo "  AI-classified until it's running."
fi

echo ""
echo "╔══════════════════════════════════════════════╗"
echo "║  SETUP COMPLETE                              ║"
echo "║                                              ║"
echo "║  To start:                                   ║"
echo "║    bash $THIS_SCRIPT/start.sh"
echo "║                                              ║"
echo "║  Then open: http://localhost:8006            ║"
echo "╚══════════════════════════════════════════════╝"
echo ""
