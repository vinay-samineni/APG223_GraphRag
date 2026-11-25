import re
import pickle
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
from rank_bm25 import BM25Okapi


class BM25KeywordStore:
    def __init__(self, children: List[Dict[str, Any]] | None = None):
        self.chunk_ids: List[str] = []
        self.bm25: BM25Okapi | None = None
        if children:
            self.build(children)

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return re.findall(r"\w+", text.lower())

    def build(self, children: List[Dict[str, Any]]):
        self.chunk_ids = [c["chunk_id"] for c in children]
        tokenized = [self._tokenize(c["content"]) for c in children]
        self.bm25 = BM25Okapi(tokenized)

    def save(self, path: Path):
        with open(path, "wb") as f:
            pickle.dump({"chunk_ids": self.chunk_ids, "bm25": self.bm25}, f)

    @classmethod
    def load(cls, path: Path) -> "BM25KeywordStore":
        instance = cls()
        with open(path, "rb") as f:
            data = pickle.load(f)
        instance.chunk_ids = data["chunk_ids"]
        instance.bm25 = data["bm25"]
        return instance

    def search(self, query: str, top_k: int = 10) -> List[str]:
        if not self.bm25:
            return []
        scores = self.bm25.get_scores(self._tokenize(query))
        top_idx = np.argsort(scores)[::-1][:top_k]
        return [self.chunk_ids[i] for i in top_idx if scores[i] > 0]