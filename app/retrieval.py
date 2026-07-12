"""
Phase 2: Hybrid retrieval engine — dense + sparse, fused with
Reciprocal Rank Fusion, then compressed with a cross-encoder reranker.
"""
from functools import lru_cache
from typing import List

from app import config
from app.embeddings import embed_query
from app.vectorstore import VectorStore
from app.bm25_index import BM25Index


@lru_cache(maxsize=1)
def _get_reranker():
    from sentence_transformers import CrossEncoder
    return CrossEncoder(config.RERANKER_MODEL)


def dense_retrieve(query: str, store: VectorStore, top_k: int = config.DENSE_TOP_K) -> List[dict]:
    q_emb = embed_query(query)
    return store.query(q_emb, top_k=top_k)


def sparse_retrieve(query: str, bm25: BM25Index, top_k: int = config.SPARSE_TOP_K) -> List[dict]:
    return bm25.query(query, top_k=top_k)


def reciprocal_rank_fusion(
    dense_results: List[dict],
    sparse_results: List[dict],
    dense_weight: float = config.DENSE_WEIGHT,
    sparse_weight: float = config.SPARSE_WEIGHT,
    k: int = config.RRF_K,
) -> List[dict]:
    """
    RRF score for a doc = weight * 1 / (k + rank).
    Weighting dense vs sparse is configurable per use case — technical
    docs full of exact config keys might want sparse weighted higher.
    """
    scores = {}
    chunk_lookup = {}

    for rank, r in enumerate(dense_results):
        cid = r["chunk_id"]
        chunk_lookup[cid] = r
        scores[cid] = scores.get(cid, 0) + dense_weight * (1 / (k + rank + 1))

    for rank, r in enumerate(sparse_results):
        cid = r["chunk_id"]
        chunk_lookup[cid] = r
        scores[cid] = scores.get(cid, 0) + sparse_weight * (1 / (k + rank + 1))

    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    out = []
    for cid, score in fused:
        item = dict(chunk_lookup[cid])
        item["rrf_score"] = score
        out.append(item)
    return out


def rerank(query: str, candidates: List[dict], top_k: int = config.FINAL_TOP_K) -> List[dict]:
    """Cross-encoder reranking — the second pass that dramatically
    improves precision over fusion alone."""
    if not candidates:
        return []
    model = _get_reranker()
    pairs = [(query, c["text"]) for c in candidates]
    scores = model.predict(pairs)
    for c, s in zip(candidates, scores):
        c["rerank_score"] = float(s)
    ranked = sorted(candidates, key=lambda c: c["rerank_score"], reverse=True)
    return ranked[:top_k]


def hybrid_retrieve(query: str, store: VectorStore, bm25: BM25Index, use_hybrid: bool = True) -> dict:
    """
    Returns both the final reranked chunks AND the intermediate dense-only
    result, so the dashboard can toggle "hybrid vs dense-only" like the
    guide asks for.
    """
    dense_results = dense_retrieve(query, store, top_k=config.DENSE_TOP_K)

    if use_hybrid:
        sparse_results = sparse_retrieve(query, bm25, top_k=config.SPARSE_TOP_K)
        fused = reciprocal_rank_fusion(dense_results, sparse_results)
    else:
        fused = dense_results

    candidates = fused[:config.RERANK_CANDIDATES]
    final = rerank(query, candidates, top_k=config.FINAL_TOP_K)

    dense_only_final = rerank(query, dense_results[:config.RERANK_CANDIDATES], top_k=config.FINAL_TOP_K)

    return {
        "final_chunks": final,
        "dense_only_chunks": dense_only_final,
        "dense_count": len(dense_results),
        "sparse_count": len(sparse_results) if use_hybrid else 0,
    }
