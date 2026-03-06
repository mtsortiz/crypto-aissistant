"""Tools for fetching crypto market data."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yfinance as yf
from langchain_core.tools import tool

from rag_pipeline import build_or_load_vectorstore, reranked_top_k


@tool
def get_crypto_prices_usd() -> dict[str, Any]:
    """Return the current USD prices for BTC, ETH and SOL."""
    tickers = {
        "BTC": "BTC-USD",
        "ETH": "ETH-USD",
        "SOL": "SOL-USD",
    }

    prices: dict[str, float] = {}
    for asset, ticker in tickers.items():
        ticker_obj = yf.Ticker(ticker)
        history = ticker_obj.history(period="1d", interval="1m")
        if history.empty:
            daily = ticker_obj.history(period="5d")
            if daily.empty:
                raise ValueError(f"No market data available for {asset}")
            prices[asset] = round(float(daily["Close"].iloc[-1]), 4)
            continue
        prices[asset] = round(float(history["Close"].iloc[-1]), 4)

    return {
        "currency": "USD",
        "prices": prices,
    }


@lru_cache(maxsize=1)
def _get_vectorstore():
    # Cache avoids reopening Chroma for every tool call.
    base_dir = Path(__file__).resolve().parent
    return build_or_load_vectorstore(base_dir / "docs", base_dir / "chroma_db")


@tool
def search_whitepapers(query: str) -> str:
    """Search crypto whitepapers and technical docs for fundamentals and theory."""
    vectorstore = _get_vectorstore()
    docs = reranked_top_k(query=query, vectorstore=vectorstore, retrieve_k=12, top_n=3)

    if not docs:
        return "No se encontro informacion relevante en los documentos tecnicos."

    parts: list[str] = []
    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        content = doc.page_content.strip()
        parts.append(f"Source: {source}\nContent: {content}")

    return "\n\n".join(parts)
