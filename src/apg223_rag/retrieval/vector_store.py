from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
from apg223_rag.config import (
    EMBED_MODEL_NAME,
    EMBED_DIMENSION,
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
    PINECONE_NAMESPACE,
    PINECONE_CLOUD,
    PINECONE_REGION,
)


class PineconeVectorStore:
    def __init__(self, embed_model: SentenceTransformer | None = None):
        if not PINECONE_API_KEY:
            raise ValueError("PINECONE_API_KEY is not set in .env")
        self.model = embed_model or SentenceTransformer(EMBED_MODEL_NAME)
        self.pc = Pinecone(api_key=PINECONE_API_KEY)

        existing = [idx.name for idx in self.pc.list_indexes()]
        if PINECONE_INDEX_NAME not in existing:
            self.pc.create_index(
                name=PINECONE_INDEX_NAME,
                dimension=EMBED_DIMENSION,
                metric="cosine",
                spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION),
            )
        self.index = self.pc.Index(PINECONE_INDEX_NAME)

    def upsert_children(self, children: List[Dict[str, Any]], batch_size: int = 50):
        try:
            self.index.delete(delete_all=True, namespace=PINECONE_NAMESPACE)
        except Exception:
            pass

        texts = [c["content"] for c in children]
        embeddings = self.model.encode(texts, normalize_embeddings=True)

        vectors = [
            {
                "id": child["chunk_id"],
                "values": emb.tolist(),
                "metadata": {
                    "parent_id": child["parent_id"],
                    "chunk_type": child["chunk_type"],
                    "section": child["section"],
                    "subsection": child["subsection"],
                    "page": child["page"],
                },
            }
            for child, emb in zip(children, embeddings)
        ]

        for i in range(0, len(vectors), batch_size):
            self.index.upsert(vectors=vectors[i : i + batch_size], namespace=PINECONE_NAMESPACE)

    def search(self, query: str, top_k: int = 10) -> List[str]:
        bge_query = f"Represent this sentence for searching relevant passages: {query}"
        q_vec = self.model.encode([bge_query], normalize_embeddings=True)[0].tolist()
        res = self.index.query(
            vector=q_vec, top_k=top_k, namespace=PINECONE_NAMESPACE, include_metadata=False
        )
        return [m["id"] for m in res["matches"]]