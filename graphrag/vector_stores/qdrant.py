# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""The Qdrant vector storage implementation package."""

import re
import uuid
from typing import Any

from qdrant_client import QdrantClient, models

from graphrag.config.models.vector_store_schema_config import DEFAULT_VECTOR_SIZE
from graphrag.data_model.types import TextEmbedder
from graphrag.vector_stores.base import (
    BaseVectorStore,
    VectorStoreDocument,
    VectorStoreSearchResult,
)

TEXT_PAYLOAD_KEY = "_text"
METDATA_PAYLOAD_KEY = "_metadata"
ORIGINAL_ID_PAYLOAD_KEY = "_original_id"

UUID_REGEX = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)


class QdrantError(Exception):
    """An exception class for Qdrant errors."""


class QdrantVectorStore(BaseVectorStore):
    """The Qdrant vector storage implementation."""

    collection_name: str | None = None

    def connect(self, **kwargs: Any) -> Any:
        """Connect to the vector storage."""
        collection_name = kwargs.pop("collection_name", None) or self.kwargs.get(
            "collection_name"
        )
        if collection_name is None:
            collection_name = self.index_name
        if collection_name is None:
            msg = "collection_name not set"
            raise ValueError(msg)
        self.collection_name = collection_name

        self._default_vector_size = kwargs.pop(
            "default_vector_size", DEFAULT_VECTOR_SIZE
        )
        self._vector_params = kwargs.pop("vector_params", {})
        self._collection_config = kwargs.pop("collection_config", {})
        self._text_payload_key = kwargs.pop("text_payload_key", TEXT_PAYLOAD_KEY)
        self._metadata_payload_key = kwargs.pop(
            "metadata_payload_key", METDATA_PAYLOAD_KEY
        )
        kwargs.pop("type", None)
        kwargs.pop("container_name", None)
        kwargs.pop("audience", None)
        kwargs.pop("database_name", None)
        kwargs.pop("db_uri", None)
        kwargs.pop("embeddings_schema", None)
        kwargs.pop("overwrite", None)

        self.db_connection = QdrantClient(**kwargs)

    def load_documents(
        self, documents: list[VectorStoreDocument], overwrite: bool = True
    ) -> None:
        """Load documents into vector storage."""
        if self.db_connection is None:
            msg = "db_connection not set. Call connect() first."
            raise QdrantError(msg)

        collection_exists = self.db_connection.collection_exists(self.collection_name)

        if collection_exists and overwrite:
            self.db_connection.delete_collection(self.collection_name)
            collection_exists = False

        if not collection_exists:
            first_doc_embedding = documents[0].vector
            dimension = (
                len(first_doc_embedding)
                if first_doc_embedding
                else self._default_vector_size
            )

            self.db_connection.create_collection(
                self.collection_name,
                vectors_config=models.VectorParams(
                    size=dimension,
                    distance=models.Distance.COSINE,
                    **self._vector_params,
                ),
                **self._collection_config,
            )

        points: list[models.PointStruct] = []
        for document in documents:
            if document.vector is None:
                continue
            point_id = self._convert_document_id(document.id)
            metadata_payload = self._build_metadata_payload(document)
            points.append(
                models.PointStruct(
                    id=point_id,
                    vector=document.vector,
                    payload={
                        self._text_payload_key: document.text,
                        self._metadata_payload_key: metadata_payload,
                    },
                )
            )

        self.db_connection.upsert(self.collection_name, points)  # type: ignore

    def filter_by_id(self, include_ids: list[str] | list[int]) -> Any:
        """Build a query filter to filter documents by id."""
        if len(include_ids) == 0:
            self.query_filter = None
        else:
            converted_ids = [self._convert_document_id(id_) for id_ in include_ids]
            self.query_filter = models.Filter(
                must=[
                    models.HasIdCondition(has_id=converted_ids),  # type: ignore
                ]
            )
        return self.query_filter

    def similarity_search_by_vector(
        self, query_embedding: list[float], k: int = 10, **kwargs: Any
    ) -> list[VectorStoreSearchResult]:
        """Perform a vector-based similarity search."""
        if self.db_connection is None:
            msg = "db_connection not set. Call connect() first."
            raise QdrantError(msg)

        results = self.db_connection.query_points(
            self.collection_name,
            query=query_embedding,
            limit=k,
            query_filter=self.query_filter,
            with_payload=True,
            with_vectors=True,
            **kwargs,
        ).points
        search_results: list[VectorStoreSearchResult] = []
        for result in results:
            payload = result.payload if result.payload else {}
            original_id = self._extract_original_id(payload, result.id)
            text = self._extract_text(payload)
            attributes = self._extract_attributes(payload)
            search_results.append(
                VectorStoreSearchResult(
                    document=VectorStoreDocument(
                        id=original_id,
                        text=text,
                        vector=result.vector,  # type: ignore
                        attributes=attributes,
                    ),
                    score=result.score,
                )
            )
        return search_results

    def similarity_search_by_text(
        self, text: str, text_embedder: TextEmbedder, k: int = 10, **kwargs: Any
    ) -> list[VectorStoreSearchResult]:
        """Perform a similarity search using a given input text."""
        query_embedding = text_embedder(text)
        if query_embedding:
            return self.similarity_search_by_vector(query_embedding, k)
        return []

    def search_by_id(self, id: str) -> VectorStoreDocument:
        """Search for a document by id."""
        points = self.db_connection.retrieve(
            self.collection_name,
            ids=[self._convert_document_id(id)],
            with_payload=True,
            with_vectors=True,
        )

        if not len(points) > 0:
            msg = f"Document with id {id} not found."
            raise ValueError(msg)

        point = points[0]
        payload = point.payload if point.payload else {}
        return VectorStoreDocument(
            id=self._extract_original_id(payload, point.id),
            text=self._extract_text(payload),
            vector=point.vector,  # type: ignore
            attributes=self._extract_attributes(payload),
        )

    def _convert_document_id(self, doc_id: str | int) -> str | int:
        """Convert an arbitrary document id into a Qdrant-compatible id."""
        if isinstance(doc_id, int):
            return doc_id
        if isinstance(doc_id, str):
            if UUID_REGEX.match(doc_id):
                return doc_id
            try:
                return int(doc_id)
            except ValueError:
                return str(uuid.uuid5(uuid.NAMESPACE_URL, doc_id))
        msg = f"Unsupported document id type: {type(doc_id)}"
        raise ValueError(msg)

    @staticmethod
    def _build_metadata_payload(document: VectorStoreDocument) -> dict[str, Any]:
        """Create metadata payload containing both original attributes and id."""
        metadata: dict[str, Any] = {}
        if document.attributes:
            metadata.update(document.attributes)
        metadata[ORIGINAL_ID_PAYLOAD_KEY] = document.id
        return metadata

    def _extract_original_id(self, payload: dict[str, Any], fallback: Any) -> Any:
        """Recover the original document id from a payload."""
        metadata = payload.get(self._metadata_payload_key, {})
        if isinstance(metadata, dict):
            return metadata.get(ORIGINAL_ID_PAYLOAD_KEY, fallback)
        return fallback

    def _extract_attributes(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Return user metadata stripped of internal bookkeeping fields."""
        metadata = payload.get(self._metadata_payload_key, {})
        if not isinstance(metadata, dict):
            return {}
        return {
            key: value
            for key, value in metadata.items()
            if key != ORIGINAL_ID_PAYLOAD_KEY
        }

    def _extract_text(self, payload: dict[str, Any]) -> str:
        """Return stored text content from payload."""
        text = payload.get(self._text_payload_key, "")
        return text if isinstance(text, str) else ""
