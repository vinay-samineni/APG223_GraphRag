# src/apg223_rag/retrieval/keyword_search.py
import re
import numpy as np
from typing import List, Dict, Any
from rank_bm25 import BM25Okapi


class BM25KeywordStore:
    def __init__(self, children: List[Dict[str, Any]]):
        self.chunk_ids = [c["chunk_id"] for c in children]
        tokenized = [self._tokenize(c["content"]) for c in children]
        self.bm25 = BM25Okapi(tokenized) if tokenized else None

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return re.findall(r"\w+", text.lower())

    def search(self, query: str, top_k: int = 10) -> List[str]:
        if not self.bm25:
            return []
        scores = self.bm25.get_scores(self._tokenize(query))
        top_idx = np.argsort(scores)[::-1][:top_k]
        return [self.chunk_ids[i] for i in top_idx if scores[i] > 0]