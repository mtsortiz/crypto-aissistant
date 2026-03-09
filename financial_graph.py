"""LangGraph state graph for a crypto financial advisor agent.

Flow:
    Start -> Agent -> Tools -> Agent -> End
"""

from __future__ import annotations

import argparse
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any, Literal

from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from market_tools import get_crypto_prices_usd, search_whitepapers


class AdvisorState(TypedDict):
    messages: Annotated[list, add_messages]
    needs_market_data: bool


class QueryInput(BaseModel):
    question: str = Field(..., min_length=1)


class AgentResponse(BaseModel):
    answer: str
    sources: list[str] = Field(default_factory=list)
    risk_level: Literal["low", "medium", "high"]


def _build_llm() -> ChatGoogleGenerativeAI:
    if not os.getenv("GOOGLE_API_KEY"):
        raise EnvironmentError("GOOGLE_API_KEY is required for Gemini.")

    model_name = os.getenv("GOOGLE_MODEL", "gemini-2.5-flash")

    return ChatGoogleGenerativeAI(
        model=model_name,
        temperature=0.2,
        max_retries=1,
    )


def agent_node(state: AdvisorState) -> dict[str, Any]:
    tools = [get_crypto_prices_usd, search_whitepapers]
    llm = _build_llm().bind_tools(tools)

    system_prompt = (
        "You are a crypto financial advisor. "
        "1. For real-time prices, use get_crypto_prices_usd. "
        "2. For technical questions, whitepapers, or fundamentals, use search_whitepapers. "
        "If you use whitepapers, cite the sources. "
        "Use concise, practical answers and mention that this is not financial advice."
    )

    response = llm.invoke([
        {"role": "system", "content": system_prompt},
        *state["messages"],
    ])

    needs_market_data = bool(getattr(response, "tool_calls", None))
    return {
        "messages": [response],
        "needs_market_data": needs_market_data,
    }


def tools_router(state: AdvisorState) -> str:
    return "tools" if state["needs_market_data"] else END


@lru_cache(maxsize=1)
def build_graph():
    tool_node = ToolNode([get_crypto_prices_usd, search_whitepapers])

    graph = StateGraph(AdvisorState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)

    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", tools_router, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")

    return graph.compile()


def _extract_sources_from_messages(messages: list[Any]) -> list[str]:
    sources: set[str] = set()
    source_pattern = re.compile(r"^Source:\s*(.+)$", re.MULTILINE)

    for message in messages:
        content = getattr(message, "content", "")
        if not isinstance(content, str):
            continue

        for match in source_pattern.findall(content):
            clean = match.strip()
            if clean:
                sources.add(clean)

    return sorted(sources)


def _normalize_sources(sources: list[str]) -> list[str]:
    normalized: list[str] = []
    for source in sources:
        clean = source.strip().replace("\\", "/")
        if clean.startswith("http://") or clean.startswith("https://"):
            normalized.append(clean)
            continue

        # Convert local source paths into stable repo-relative links.
        source_name = Path(clean).name if clean else ""
        if source_name:
            normalized.append(f"docs/{source_name}")

    return sorted(set(normalized))


def _fallback_risk_level(question: str, answer: str) -> Literal["low", "medium", "high"]:
    text = f"{question} {answer}".lower()
    high_terms = ["leverage", "margin", "all in", "borrow", "loan", "futures", "short"]
    medium_terms = ["buy", "sell", "invest", "entry", "target", "volatility", "risk"]

    if any(term in text for term in high_terms):
        return "high"
    if any(term in text for term in medium_terms):
        return "medium"
    return "low"


def _extract_sources_from_tool_text(text: str) -> list[str]:
    source_pattern = re.compile(r"^Source:\s*(.+)$", re.MULTILINE)
    raw_sources = [match.strip() for match in source_pattern.findall(text) if match.strip()]
    return _normalize_sources(raw_sources)


def _quota_fallback_response(question: str) -> AgentResponse:
    question_lc = question.lower()

    # Keep service useful under LLM quota pressure by calling tools directly.
    if any(term in question_lc for term in ["precio", "price", "btc", "eth", "sol"]):
        try:
            market = get_crypto_prices_usd.invoke({})
            prices = market.get("prices", {}) if isinstance(market, dict) else {}
        except Exception:
            prices = {}

        answer = (
            "No pude usar el modelo de lenguaje por limite de cuota, "
            "pero estos son los precios actuales en USD: "
            f"BTC={prices.get('BTC', 'n/a')}, ETH={prices.get('ETH', 'n/a')}, SOL={prices.get('SOL', 'n/a')}. "
            "Esto no es asesoramiento financiero."
        )
        return AgentResponse(answer=answer, sources=[], risk_level="medium")

    try:
        whitepaper = search_whitepapers.invoke({"query": question})
    except Exception as exc:
        whitepaper = (
            "No se pudo consultar whitepapers en este momento. "
            f"Detalle tecnico: {exc}"
        )

    sources = _extract_sources_from_tool_text(whitepaper)
    answer = (
        "No pude usar el modelo de lenguaje por limite de cuota. "
        "Te comparto resultados directos de los whitepapers para tu consulta:\n\n"
        f"{whitepaper}\n\n"
        "Esto no es asesoramiento financiero."
    )

    return AgentResponse(
        answer=answer,
        sources=sources,
        risk_level=_fallback_risk_level(question, answer),
    )


def run_query(query: QueryInput) -> AgentResponse:
    app = build_graph()
    try:
        final_state = app.invoke(
            {
                "messages": [HumanMessage(content=query.question)],
                "needs_market_data": False,
            }
        )
    except Exception as exc:
        error_text = str(exc)
        if "RESOURCE_EXHAUSTED" in error_text or "429" in error_text:
            return _quota_fallback_response(query.question)
        raise

    final_message = final_state["messages"][-1]
    raw_answer = str(getattr(final_message, "content", "")).strip()
    raw_sources = _extract_sources_from_messages(final_state["messages"])
    normalized_sources = _normalize_sources(raw_sources)

    # Keep JSON shaping local to avoid an extra LLM call per request.
    return AgentResponse(
        answer=raw_answer,
        sources=normalized_sources,
        risk_level=_fallback_risk_level(query.question, raw_answer),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Crypto financial advisor with LangGraph")
    parser.add_argument("--question", required=True, help="User question for the advisor")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    query = QueryInput(question=args.question)
    result = run_query(query)
    print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
