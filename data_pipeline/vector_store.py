"""ChromaDB vector store for unstructured financial text and research."""

from __future__ import annotations

import logging
import uuid
from datetime import date
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings
from sentence_transformers import SentenceTransformer

from config.settings import Settings

logger = logging.getLogger(__name__)

_DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"


class VectorStoreManager:
    """Manages a ChromaDB collection for unstructured financial text.

    Each document stores: ticker, timestamp, source, and text content.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or Settings()
        self._client = chromadb.PersistentClient(
            path=self._settings.chroma_persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._embedder = SentenceTransformer(_DEFAULT_EMBEDDING_MODEL)
        self._collection = self._client.get_or_create_collection(
            name=self._settings.chroma_collection_name,
            metadata={"description": "ETF research and financial news embeddings"},
        )

    # ── Write ────────────────────────────────────────────────

    def ingest_documents(
        self,
        texts: list[str],
        metadatas: list[dict[str, Any]] | None = None,
        ids: list[str] | None = None,
    ) -> list[str]:
        """Embed and store a batch of text documents.

        Args:
            texts: Raw text content to embed.
            metadatas: List of metadata dicts — each should contain
                       ticker, timestamp (YYYY-MM-DD), and source.
            ids: Optional document IDs.  Auto-generated UUIDs if omitted.

        Returns:
            List of document IDs stored.
        """
        if not texts:
            return []

        doc_ids = ids or [str(uuid.uuid4()) for _ in texts]
        embeddings = self._embedder.encode(texts, show_progress_bar=False).tolist()

        self._collection.add(
            ids=doc_ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas or [{}] * len(texts),
        )
        logger.info("Ingested %d documents into ChromaDB.", len(texts))
        return doc_ids

    def ingest_financial_news(
        self,
        ticker: str,
        entries: list[dict[str, Any]],
        source: str = "news_api",
    ) -> list[str]:
        """Convenience method: ingest a batch of news items for a single ticker.

        Each entry should have keys: 'text' and 'timestamp' (date or str).
        """
        texts = []
        metadatas = []
        for entry in entries:
            texts.append(entry["text"])
            metadatas.append(
                {
                    "ticker": ticker,
                    "timestamp": str(entry["timestamp"]),
                    "source": source,
                }
            )
        return self.ingest_documents(texts, metadatas)

    # ── Query ────────────────────────────────────────────────

    def query(
        self,
        query_text: str,
        ticker: str | None = None,
        n_results: int = 10,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> dict[str, Any]:
        """Semantic search over stored documents with optional filters.

        Args:
            query_text: Natural language query string.
            ticker: Optional ticker filter.
            n_results: Number of results to return.
            start_date: Filter documents with timestamp >= start_date.
            end_date: Filter documents with timestamp <= end_date.

        Returns:
            ChromaDB query result dict with ids, documents, metadatas, distances.
        """
        where_filter: dict[str, Any] | None = None
        if ticker:
            where_filter = where_filter or {}
            where_filter["ticker"] = ticker
        if start_date:
            where_filter = where_filter or {}
            where_filter["timestamp"] = where_filter.get("timestamp", {})
            where_filter["timestamp"]["$gte"] = str(start_date)
        if end_date:
            where_filter = where_filter or {}
            where_filter["timestamp"] = where_filter.get("timestamp", {})
            where_filter["timestamp"]["$lte"] = str(end_date)

        query_embedding = self._embedder.encode([query_text], show_progress_bar=False).tolist()

        return self._collection.query(
            query_embeddings=query_embedding,
            n_results=n_results,
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )

    # ── Maintenance ──────────────────────────────────────────

    def count(self) -> int:
        return self._collection.count()

    def delete_by_ticker(self, ticker: str) -> int:
        """Remove all documents for a given ticker."""
        results = self._collection.get(where={"ticker": ticker})
        if results["ids"]:
            self._collection.delete(ids=results["ids"])
            logger.info("Deleted %d documents for ticker=%s.", len(results["ids"]), ticker)
            return len(results["ids"])
        return 0

    def delete_collection(self) -> None:
        """Drop the entire collection."""
        self._client.delete_collection(self._settings.chroma_collection_name)
        logger.warning("Collection '%s' deleted.", self._settings.chroma_collection_name)
