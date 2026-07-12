"""
Phase 1.3 + 1.4: ChromaDB storage for chunk embeddings, with
near-duplicate detection before insert (cosine similarity > threshold
against existing chunks is skipped, not overwritten, so the index never
wastes context-window slots on redundant content).
"""
import chromadb
from typing import List, Optional

from app import config
from app.chunking import Chunk
from app.embeddings import embed_texts, cosine_similarity


class VectorStore:
    def __init__(self, persist_dir: str = config.CHROMA_PERSIST_DIR):
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(
            name=config.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    def _is_near_duplicate(self, embedding: List[float]) -> bool:
        if self.collection.count() == 0:
            return False
        result = self.collection.query(
            query_embeddings=[embedding],
            n_results=1,
            include=["distances"],
        )
        if not result["distances"] or not result["distances"][0]:
            return False
        # chroma cosine distance = 1 - cosine_similarity
        distance = result["distances"][0][0]
        similarity = 1 - distance
        return similarity > config.DEDUP_SIMILARITY_THRESHOLD

    def add_chunks(self, chunks: List[Chunk]) -> dict:
        if not chunks:
            return {"added": 0, "skipped_duplicates": 0}

        texts = [c.text for c in chunks]
        embeddings = embed_texts(texts)

        added, skipped = 0, 0
        ids, metas, docs, embs = [], [], [], []

        for chunk, emb in zip(chunks, embeddings):
            if self._is_near_duplicate(emb):
                skipped += 1
                continue
            ids.append(chunk.chunk_id)
            metas.append({
                "doc_id": chunk.doc_id,
                "source_file": chunk.source_file,
                "section_heading": chunk.section_heading,
                "page": chunk.page if chunk.page is not None else -1,
                "char_count": chunk.char_count,
                "strategy": chunk.strategy,
            })
            docs.append(chunk.text)
            embs.append(emb)
            added += 1

        if ids:
            self.collection.upsert(ids=ids, embeddings=embs, metadatas=metas, documents=docs)

        return {"added": added, "skipped_duplicates": skipped}

    def query(self, query_embedding: List[float], top_k: int = config.DENSE_TOP_K) -> List[dict]:
        if self.collection.count() == 0:
            return []
        result = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self.collection.count()),
            include=["documents", "metadatas", "distances"],
        )
        out = []
        for doc, meta, dist in zip(result["documents"][0], result["metadatas"][0], result["distances"][0]):
            out.append({
                "chunk_id": meta.get("_id", None),
                "text": doc,
                "metadata": meta,
                "score": 1 - dist,  # convert distance back to similarity
            })
        # chroma doesn't return ids in the same include list; fetch separately
        ids_result = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self.collection.count()),
        )
        for o, cid in zip(out, ids_result["ids"][0]):
            o["chunk_id"] = cid
        return out

    def all_chunks(self) -> List[dict]:
        """Used to build/rebuild the BM25 index over the same corpus."""
        if self.collection.count() == 0:
            return []
        result = self.collection.get(include=["documents", "metadatas"])
        out = []
        for cid, doc, meta in zip(result["ids"], result["documents"], result["metadatas"]):
            out.append({"chunk_id": cid, "text": doc, "metadata": meta})
        return out

    def list_documents(self) -> List[dict]:
        chunks = self.all_chunks()
        seen = {}
        for c in chunks:
            doc_id = c["metadata"]["doc_id"]
            if doc_id not in seen:
                seen[doc_id] = {
                    "doc_id": doc_id,
                    "source_file": c["metadata"]["source_file"],
                    "chunk_count": 0,
                }
            seen[doc_id]["chunk_count"] += 1
        return list(seen.values())
