"""
LLM client — routes to Ollama locally, drop-in for vLLM on H200s.

Backend is selected dynamically from env vars on every client creation,
so changing LLM_BASE_URL or LLM_MODEL takes effect on the next request
after the environment changes (e.g. container restart or env reload).
"""
from __future__ import annotations
import os
import json
import logging
import httpx
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

_DEFAULT_BASE_URL = "http://localhost:11434/v1"
_DEFAULT_MODEL    = "qwen2.5:14b"

_client: AsyncOpenAI | None = None
_client_base_url: str = ""


def get_llm_base_url() -> str:
    return os.getenv("LLM_BASE_URL", _DEFAULT_BASE_URL)


def get_llm_model() -> str:
    return os.getenv("LLM_MODEL", _DEFAULT_MODEL)


def get_client() -> AsyncOpenAI:
    global _client, _client_base_url
    current_url = get_llm_base_url()
    if _client is None or _client_base_url != current_url:
        _client = AsyncOpenAI(
            base_url=current_url,
            api_key="ollama",
            timeout=httpx.Timeout(60.0),
        )
        _client_base_url = current_url
    return _client


async def chat(system: str, user: str, temperature: float = 0.1) -> str:
    """Send a chat completion. Returns '' if LLM backend is not reachable."""
    try:
        client = get_client()
        resp = await client.chat.completions.create(
            model=get_llm_model(),
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            temperature=temperature,
            max_completion_tokens=1024,
        )
        return resp.choices[0].message.content or ""
    except Exception as e:
        logger.warning(
            f"[llm] chat failed ({type(e).__name__}: {e}). "
            f"LLM_BASE_URL={get_llm_base_url()}"
        )
        return ""


async def chat_json(system: str, user: str) -> dict:
    """Chat and parse JSON response. Returns {} on any failure."""
    raw = await chat(system + "\n\nRespond ONLY with valid JSON, no markdown.", user)
    if not raw:
        return {}
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    try:
        return json.loads(raw)
    except Exception:
        return {}


async def check_ollama() -> bool:
    """Return True if the configured LLM backend is reachable."""
    try:
        base_url = get_llm_base_url()
        base = base_url[:-3] if base_url.endswith("/v1") else base_url
        base = base.rstrip("/")
        async with httpx.AsyncClient(timeout=3) as c:
            for path in ("/v1/models", "/api/tags"):
                try:
                    r = await c.get(base + path)
                    if r.status_code == 200:
                        return True
                except Exception:
                    continue
        return False
    except Exception:
        return False
