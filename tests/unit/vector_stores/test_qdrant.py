"""Tests for the Qdrant vector store implementation."""

from __future__ import annotations

from graphrag.config.models.vector_store_schema_config import (
    VectorStoreSchemaConfig,
)
from graphrag.vector_stores.qdrant import QdrantVectorStore


def test_connect_strips_non_qdrant_kwargs(monkeypatch):
    """Ensure config keys meant for other stores are not forwarded to QdrantClient."""
    captured: dict[str, object] = {}

    class DummyClient:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(
        "graphrag.vector_stores.qdrant.QdrantClient",
        DummyClient,
    )

    schema = VectorStoreSchemaConfig(index_name="collection")
    store = QdrantVectorStore(vector_store_schema_config=schema)

    store.connect(
        host="localhost",
        port=6333,
        api_key="secret",
        type="qdrant",
        container_name="default",
        audience="should-remove",
        database_name="should-remove",
        db_uri="should-remove",
        embeddings_schema={"foo": {}},
        overwrite=False,
    )

    assert captured == {"host": "localhost", "port": 6333, "api_key": "secret"}
