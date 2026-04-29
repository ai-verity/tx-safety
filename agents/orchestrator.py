"""
LangGraph Orchestrator — coordinates all agents as a state machine.
Handles startup sequencing, health checks, and agent lifecycle.
"""
from __future__ import annotations
import asyncio
import logging
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END
from core.database import init_db, get_stats
from core.llm import check_ollama

logger = logging.getLogger(__name__)


class OrchestratorState(TypedDict):
    llm_ok: bool
    db_ok: bool
    agents_started: list[str]
    errors: list[str]


async def check_ollama_node(state: OrchestratorState) -> OrchestratorState:
    from core.llm import get_llm_base_url
    ok = await check_ollama()
    if ok:
        logger.info(f"[orchestrator] LLM backend reachable ({get_llm_base_url()}) ✓")
    else:
        logger.warning(f"[orchestrator] LLM backend not reachable ({get_llm_base_url()}) — agents will retry")
    return {**state, "llm_ok": ok}


async def init_db_node(state: OrchestratorState) -> OrchestratorState:
    try:
        await init_db()
        logger.info("[orchestrator] DB initialized ✓")
        return {**state, "db_ok": True}
    except Exception as e:
        logger.error(f"[orchestrator] DB init failed: {e}")
        return {**state, "db_ok": False, "errors": state["errors"] + [str(e)]}


async def start_agents_node(state: OrchestratorState) -> OrchestratorState:
    """Launch all agents as background asyncio tasks."""
    from agents.news_agent import NewsAgent
    from agents.gov_agent import GovDataAgent
    from agents.social_agent import SocialAgent
    from agents.weather_agent import WeatherAgent
    from agents.analysis_agents import ThreatClassifierAgent, TrendAgent, ReportAgent

    agents = [
        NewsAgent(),
        GovDataAgent(),
        SocialAgent(),
        WeatherAgent(),
        ThreatClassifierAgent(),
        TrendAgent(),
        ReportAgent(),
    ]

    started = []
    for agent in agents:
        asyncio.create_task(agent.start(), name=f"agent-{agent.name}")
        started.append(agent.name)
        logger.info(f"[orchestrator] started agent: {agent.name}")

    return {**state, "agents_started": started}


def build_graph() -> StateGraph:
    g = StateGraph(OrchestratorState)
    g.add_node("check_ollama", check_ollama_node)
    g.add_node("init_db",      init_db_node)
    g.add_node("start_agents", start_agents_node)

    g.set_entry_point("check_ollama")
    g.add_edge("check_ollama", "init_db")
    g.add_edge("init_db", "start_agents")
    g.add_edge("start_agents", END)
    return g.compile()


async def run_orchestrator():
    """Execute the startup graph and return final state."""
    graph = build_graph()
    initial: OrchestratorState = {
        "llm_ok": False,
        "db_ok": False,
        "agents_started": [],
        "errors": [],
    }
    result = await graph.ainvoke(initial)
    logger.info(f"[orchestrator] startup complete: {result}")
    return result
