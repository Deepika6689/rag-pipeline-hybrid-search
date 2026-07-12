"""
Phase 2.2: Sparse (BM25) retrieval. Catches exact keyword matches —
function names, config keys, error codes — that dense/semantic search
often misses. Kept in sync with the vector store by always rebuilding
from the same chunk corpus (see build_from_vectorstore).
"""
import os
import pickle
import re
from typing import List

from rank_bm25 import BM25Okapi

from app import config


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z0-9_./-]+", text.lower())


class BM25Index:
    def __init__(self):
        self.bm25: BM25Okapi = None
        self.chunk_ids: List[str] = []
        self.chunk_texts: List[str] = []
        self.chunk_metas: List[dict] = []

    def build(self, chunks: List[dict]):
        """chunks: list of {"chunk_id", "text", "metadata"} — same shape as vectorstore.all_chunks()."""
        self.chunk_ids = [c["chunk_id"] for c in chunks]
        self.chunk_texts = [c["text"] for c in chunks]
        self.chunk_metas = [c["metadata"] for c in chunks]
        tokenized = [_tokenize(t) for t in self.chunk_texts]
        self.bm25 = BM25Okapi(tokenized) if tokenized else None

    def query(self, query_text: str, top_k: int = config.SPARSE_TOP_K) -> List[dict]:
        if self.bm25 is None or not self.chunk_ids:
            return []
        scores = self.bm25.get_scores(_tokenize(query_text))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        out = []
        for i in ranked:
            if scores[i] <= 0:
                continue
            out.append({
                "chunk_id": self.chunk_ids[i],
                "text": self.chunk_texts[i],
                "metadata": self.chunk_metas[i],
                "score": float(scores[i]),
            })
        return out

    def save(self, path: str = config.BM25_INDEX_PATH):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "chunk_ids": self.chunk_ids,
                "chunk_texts": self.chunk_texts,
                "chunk_metas": self.chunk_metas,
            }, f)

    def load(self, path: str = config.BM25_INDEX_PATH):
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.chunk_ids = data["chunk_ids"]
        self.chunk_texts = data["chunk_texts"]
        self.chunk_metas = data["chunk_metas"]
        tokenized = [_tokenize(t) for t in self.chunk_texts]
        self.bm25 = BM25Okapi(tokenized) if tokenized else None


def build_from_vectorstore(vector_store) -> BM25Index:
    """Keeps BM25 in sync with Chroma by rebuilding from the same source of truth."""
    idx = BM25Index()
    idx.build(vector_store.all_chunks())
    idx.save()
    return idx
