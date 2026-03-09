"""RAG pipeline with LangChain + Chroma + FlashRank reranking.

Usage:
    python rag_pipeline.py --docs-dir ./docs --persist-dir ./chroma_db
    python rag_pipeline.py --query "What does the Bitcoin whitepaper say about PoW?"

Environment:
    GOOGLE_API_KEY must be set.
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import time
from pathlib import Path
from typing import Iterable

from langchain_chroma import Chroma
from langchain_community.document_compressors import FlashrankRerank
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def _build_embeddings() -> GoogleGenerativeAIEmbeddings:
    """Create embeddings client with resilient model selection.

    Different Google API setups can expose the same embedding model with
    slightly different names, so we try common variants.
    """
    preferred_model = os.getenv("GOOGLE_EMBEDDING_MODEL", "models/gemini-embedding-001")
    raw_candidates = [
        preferred_model,
        "models/gemini-embedding-001",
        "models/text-embedding-004",
        "text-embedding-004",
        "models/embedding-001",
    ]

    candidates: list[str] = []
    for model_name in raw_candidates:
        if model_name and model_name not in candidates:
            candidates.append(model_name)

    last_error: Exception | None = None
    for model_name in candidates:
        try:
            embeddings = GoogleGenerativeAIEmbeddings(model=model_name)
            # Lightweight probe to fail fast during startup instead of deep in ingestion.
            embeddings.embed_query("healthcheck")
            logger.info("Using embedding model: %s", model_name)
            return embeddings
        except Exception as exc:  # pragma: no cover - external API variability
            last_error = exc
            logger.warning("Embedding model not available: %s", model_name)

    raise RuntimeError(
        "No compatible Google embedding model found. "
        "Set GOOGLE_EMBEDDING_MODEL to a valid model for your API key."
    ) from last_error


def _discover_pdf_files(docs_dir: Path) -> list[Path]:
    if not docs_dir.exists():
        logger.warning("Docs directory does not exist: %s", docs_dir)
        return []
    return sorted(docs_dir.glob("*.pdf"))


def _has_persisted_store(persist_dir: Path) -> bool:
    db_file = persist_dir / "chroma.sqlite3"
    return db_file.exists() and db_file.stat().st_size > 0


def load_pdf_documents(docs_dir: Path) -> list[Document]:
    """Load all PDFs from docs_dir into LangChain documents."""
    pdf_files = _discover_pdf_files(docs_dir)
    if not pdf_files:
        logger.warning("No PDF files found in %s", docs_dir)
        return []

    documents: list[Document] = []
    for pdf_file in pdf_files:
        loader = PyPDFLoader(str(pdf_file))
        loaded_docs = loader.load()
        for doc in loaded_docs:
            doc.metadata.setdefault("source", str(pdf_file))
        documents.extend(loaded_docs)

    logger.info("Loaded %s pages from %s PDFs", len(documents), len(pdf_files))
    return documents


def split_documents(documents: Iterable[Document]) -> list[Document]:
    """Split docs with required chunking strategy (1000/100)."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
    )
    chunks = splitter.split_documents(list(documents))
    logger.info("Created %s chunks", len(chunks))
    return chunks


def build_or_load_vectorstore(
    docs_dir: Path,
    persist_dir: Path,
    allow_force_rebuild: bool = True,
) -> Chroma:
    """Build a persistent Chroma vector store (or load if already present)."""
    if not os.getenv("GOOGLE_API_KEY"):
        raise EnvironmentError("GOOGLE_API_KEY is required to use GoogleGenerativeAIEmbeddings.")

    persist_dir.mkdir(parents=True, exist_ok=True)

    embeddings = _build_embeddings()

    # Default behavior for runtime queries: load persisted DB to avoid
    # expensive and quota-heavy re-embedding on every process start.
    force_rebuild = (
        allow_force_rebuild
        and os.getenv("RAG_FORCE_REBUILD", "false").lower() == "true"
    )
    if _has_persisted_store(persist_dir) and not force_rebuild:
        vectorstore = Chroma(
            persist_directory=str(persist_dir),
            embedding_function=embeddings,
        )
        logger.info("Loaded existing vector store from %s", persist_dir)
        return vectorstore

    documents = load_pdf_documents(docs_dir)
    if documents:
        chunks = split_documents(documents)
        if force_rebuild and persist_dir.exists():
            shutil.rmtree(persist_dir, ignore_errors=True)
            persist_dir.mkdir(parents=True, exist_ok=True)

        vectorstore = Chroma(
            persist_directory=str(persist_dir),
            embedding_function=embeddings,
        )

        # Free-tier embed quotas are strict; index in batches with pauses.
        batch_size = int(os.getenv("RAG_INDEX_BATCH_SIZE", "80"))
        pause_seconds = int(os.getenv("RAG_INDEX_BATCH_PAUSE_SECONDS", "65"))

        for start in range(0, len(chunks), batch_size):
            end = min(start + batch_size, len(chunks))
            batch = chunks[start:end]
            vectorstore.add_documents(batch)
            logger.info("Indexed chunks %s-%s/%s", start + 1, end, len(chunks))

            if end < len(chunks):
                logger.info("Waiting %ss to respect embedding quota...", pause_seconds)
                time.sleep(pause_seconds)

        if hasattr(vectorstore, "persist"):
            vectorstore.persist()
        logger.info("Vector store built and persisted at %s", persist_dir)
        return vectorstore

    raise FileNotFoundError(
        f"No documents found in {docs_dir} and no persisted Chroma DB in {persist_dir}."
    )


def reranked_top_k(query: str, vectorstore: Chroma, retrieve_k: int = 12, top_n: int = 3) -> list[Document]:
    """Retrieve candidates from Chroma and rerank top results with FlashRank."""
    candidates = vectorstore.similarity_search(query=query, k=retrieve_k)
    if not candidates:
        return []

    reranker = FlashrankRerank(top_n=top_n)
    reranked = reranker.compress_documents(candidates, query)
    logger.info("Retrieved %s candidates, reranked to top %s", len(candidates), len(reranked))
    return reranked


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RAG pipeline for crypto PDFs with Chroma persistence")
    parser.add_argument("--docs-dir", default="docs", help="Directory with PDF documents")
    parser.add_argument("--persist-dir", default="chroma_db", help="Directory for persistent Chroma DB")
    parser.add_argument("--query", default=None, help="Optional query to test retrieval + reranking")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    docs_dir = Path(args.docs_dir)
    persist_dir = Path(args.persist_dir)

    vectorstore = build_or_load_vectorstore(docs_dir=docs_dir, persist_dir=persist_dir)

    if args.query:
        top_docs = reranked_top_k(query=args.query, vectorstore=vectorstore, retrieve_k=12, top_n=3)
        if not top_docs:
            logger.warning("No results found for query: %s", args.query)
            return

        print("\nTop 3 reranked results:\n")
        for idx, doc in enumerate(top_docs, start=1):
            source = doc.metadata.get("source", "unknown")
            page = doc.metadata.get("page", "n/a")
            snippet = doc.page_content.strip().replace("\n", " ")[:300]
            print(f"{idx}. source={source} | page={page}")
            print(f"   {snippet}...")
            print()


if __name__ == "__main__":
    main()
