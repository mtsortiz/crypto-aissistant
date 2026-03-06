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


def _discover_pdf_files(docs_dir: Path) -> list[Path]:
    if not docs_dir.exists():
        logger.warning("Docs directory does not exist: %s", docs_dir)
        return []
    return sorted(docs_dir.glob("*.pdf"))


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


def build_or_load_vectorstore(docs_dir: Path, persist_dir: Path) -> Chroma:
    """Build a persistent Chroma vector store (or load if already present)."""
    if not os.getenv("GOOGLE_API_KEY"):
        raise EnvironmentError("GOOGLE_API_KEY is required to use GoogleGenerativeAIEmbeddings.")

    persist_dir.mkdir(parents=True, exist_ok=True)

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-004")

    documents = load_pdf_documents(docs_dir)
    if documents:
        chunks = split_documents(documents)
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=str(persist_dir)
        )
        if hasattr(vectorstore, "persist"):
            vectorstore.persist()
        logger.info("Vector store built and persisted at %s", persist_dir)
        return vectorstore

    # Fallback to loading an existing persisted DB.
    vectorstore = Chroma(
        persist_directory=str(persist_dir),
        embedding_function=embeddings,
    )
    logger.info("Loaded existing vector store from %s", persist_dir)
    return vectorstore


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
