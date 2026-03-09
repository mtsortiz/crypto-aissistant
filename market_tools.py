"""Tools for fetching crypto market data."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yfinance as yf
from langchain_core.tools import tool
from pypdf import PdfReader


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
    from rag_pipeline import build_or_load_vectorstore

    return build_or_load_vectorstore(
        base_dir / "docs",
        base_dir / "chroma_db",
        allow_force_rebuild=False,
    )


def _fallback_pdf_search(query: str, max_results: int = 3) -> str:
    """Fallback PDF search used when vector DB stack is unavailable.

    It extracts text from docs/*.pdf and scores pages by term overlap.
    """
    docs_dir = Path(__file__).resolve().parent / "docs"
    pdf_files = sorted(docs_dir.glob("*.pdf"))
    if not pdf_files:
        return "No se encontraron PDFs en la carpeta docs para buscar whitepapers."

    query_terms = {t.lower() for t in query.split() if len(t.strip()) >= 3}
    if not query_terms:
        query_terms = {query.lower().strip()}

    scored: list[tuple[int, str, int, str]] = []
    for pdf_file in pdf_files:
        try:
            reader = PdfReader(str(pdf_file))
        except Exception:
            continue

        for page_idx, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if not text:
                continue

            lowered = text.lower()
            score = sum(1 for term in query_terms if term in lowered)
            if score <= 0:
                continue

            snippet = " ".join(text.replace("\n", " ").split())[:700]
            scored.append((score, str(pdf_file), page_idx, snippet))

    if not scored:
        return "No se encontro informacion relevante en los documentos tecnicos."

    scored.sort(key=lambda item: item[0], reverse=True)
    top = scored[:max_results]

    parts: list[str] = []
    for _, source, page, snippet in top:
        parts.append(f"Source: {source}\nContent: [page {page}] {snippet}")

    return "\n\n".join(parts)


@tool
def search_whitepapers(query: str) -> str:
    """Search crypto whitepapers and technical docs for fundamentals and theory."""
    try:
        from rag_pipeline import reranked_top_k

        vectorstore = _get_vectorstore()
        docs = reranked_top_k(query=query, vectorstore=vectorstore, retrieve_k=12, top_n=3)
    except Exception as exc:
        fallback_result = _fallback_pdf_search(query)
        if "Source:" in fallback_result:
            return fallback_result

        return (
            "Whitepaper search is temporarily unavailable. "
            f"Underlying error: {exc}. "
            f"Fallback result: {fallback_result}"
        )

    if not docs:
        return "No se encontro informacion relevante en los documentos tecnicos."

    parts: list[str] = []
    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        content = doc.page_content.strip()
        parts.append(f"Source: {source}\nContent: {content}")

    return "\n\n".join(parts)
