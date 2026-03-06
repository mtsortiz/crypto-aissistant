"""LangGraph state graph for a crypto financial advisor agent.

Flow:
    Start -> Agent -> Tools -> Agent -> End
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Annotated, Any

from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict

from market_tools import get_crypto_prices_usd, search_whitepapers


class AdvisorState(TypedDict):
    messages: Annotated[list, add_messages]
    needs_market_data: bool


def _build_llm() -> ChatGoogleGenerativeAI:
    if not os.getenv("GOOGLE_API_KEY"):
        raise EnvironmentError("GOOGLE_API_KEY is required for Gemini 1.5 Pro.")

    return ChatGoogleGenerativeAI(
        model="gemini-1.5-pro",
        temperature=0.2,
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


def build_graph():
    tool_node = ToolNode([get_crypto_prices_usd, search_whitepapers])

    graph = StateGraph(AdvisorState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)

    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", tools_router, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")

    return graph.compile()


def run_query(question: str) -> dict[str, Any]:
    app = build_graph()
    final_state = app.invoke(
        {
            "messages": [HumanMessage(content=question)],
            "needs_market_data": False,
        }
    )

    final_message = final_state["messages"][-1]
    return {
        "question": question,
        "answer": final_message.content,
        "needs_market_data": final_state["needs_market_data"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Crypto financial advisor with LangGraph")
    parser.add_argument("--question", required=True, help="User question for the advisor")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_query(args.question)
    print(json.dumps(result, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
