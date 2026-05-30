import logging
from typing import Optional
import numpy as np

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from app.config import get_settings

logger = logging.getLogger(__name__)

# Vector dimension produced by ArcFace buffalo_l
VECTOR_DIM = 512


class VectorStore:
   

    def __init__(self) -> None:
        settings = get_settings()

        # Build the client — works for both local Docker and Qdrant Cloud
        client_kwargs: dict = {"url": settings.qdrant_url}
        if settings.qdrant_api_key:
            client_kwargs["api_key"] = settings.qdrant_api_key

        self._client = QdrantClient(**client_kwargs)
        self._collection = settings.collection_name

        logger.info("Connecting to Qdrant at %s", settings.qdrant_url)
        self._ensure_collection()


    def _ensure_collection(self) -> None:
        
        existing = {c.name for c in self._client.get_collections().collections}
        if self._collection in existing:
            logger.info("Qdrant collection '%s' already exists.", self._collection)
            return

        logger.info("Creating Qdrant collection '%s' …", self._collection)
        self._client.create_collection(
            collection_name=self._collection,
            vectors_config=qmodels.VectorParams(
                size=VECTOR_DIM,
                distance=qmodels.Distance.COSINE,
                
                hnsw_config=qmodels.HnswConfigDiff(
                    m=16,
                    ef_construct=200,  # higher = better recall, slower writes
                    on_disk=False,
                ),
            ),
            # Payload (metadata) stored on disk saves heap for large collections
            on_disk_payload=True,
        )
        logger.info("Collection '%s' created.", self._collection)

    @staticmethod
    def _id_to_int(identity_id: str) -> int:
       
        import hashlib
        digest = hashlib.sha256(identity_id.encode()).digest()
        # Take the first 8 bytes as a little-endian uint64
        return int.from_bytes(digest[:8], "little")


    def upsert(self, identity_id: str, embedding: np.ndarray) -> None:
      
        point_id = self._id_to_int(identity_id)
        self._client.upsert(
            collection_name=self._collection,
            points=[
                qmodels.PointStruct(
                    id=point_id,
                    vector=embedding.tolist(),     # Qdrant expects a plain list
                    payload={"identity_id": identity_id},  # stored as metadata
                )
            ],
        )
        logger.debug("Upserted identity '%s' (point_id=%d)", identity_id, point_id)

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 1,
        ef_search: Optional[int] = None,
    ) -> list[dict]:
      
        settings = get_settings()
        ef = ef_search or settings.hnsw_ef_search


        # search_params controls HNSW traversal for THIS query only.
        # It does NOT affect how the index was built (that's ef_construct above).
        results = self._client.search(
            collection_name=self._collection,
            query_vector=query_embedding.tolist(),
            limit=top_k,
            with_payload=True,        # we need the identity_id payload
            search_params=qmodels.SearchParams(
                hnsw_ef=ef,
                exact=False,          # exact=True would be brute-force — don't do this
            ),
        )

        return [
            {
                "identity_id": hit.payload["identity_id"],
                "score": round(hit.score, 6),
            }
            for hit in results
        ]

    def count(self) -> int:
        """Return the number of enrolled identities."""
        return self._client.count(
            collection_name=self._collection,
            exact=True,
        ).count

    def delete(self, identity_id: str) -> None:
        """Remove a single identity from the collection."""
        point_id = self._id_to_int(identity_id)
        self._client.delete(
            collection_name=self._collection,
            points_selector=qmodels.PointIdsList(points=[point_id]),
        )
        logger.info("Deleted identity '%s'", identity_id)