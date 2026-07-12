"""
Phase 1.2: Configurable chunking strategies.

Three switchable strategies, as the guide specifies:
  - fixed:      fixed-size window with overlap (baseline)
  - recursive:  structure-aware split on section headings, then by size
  - semantic:   splits on topic boundaries using embedding similarity

Each chunk records which strategy produced it, so the eval framework
(Phase 4) can compare strategies head-to-head with real numbers.
"""
from dataclasses import dataclass
from typing import List, Optional

from app import config
from app.ingestion import LoadedDocument


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    source_file: str
    section_heading: str
    page: Optional[int]
    text: str
    char_count: int
    strategy: str


def _fixed_size_chunks(text: str, size: int, overlap: int) -> List[str]:
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start += max(size - overlap, 1)
    return [c for c in chunks if c.strip()]


def chunk_fixed(doc: LoadedDocument) -> List[Chunk]:
    """Baseline: ignore structure, just slide a window over the full text."""
    pieces = _fixed_size_chunks(doc.text, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
    out = []
    for i, piece in enumerate(pieces):
        out.append(Chunk(
            chunk_id=f"{doc.doc_id}-fixed-{i}",
            doc_id=doc.doc_id,
            source_file=doc.source_file,
            section_heading="N/A",
            page=None,
            text=piece.strip(),
            char_count=len(piece.strip()),
            strategy="fixed",
        ))
    return out


def chunk_recursive(doc: LoadedDocument) -> List[Chunk]:
    """
    Structure-aware: chunk within each section (heading) first, only
    falling back to fixed-size splitting if a section is too long.
    Keeps headings/page numbers attached as metadata on every chunk.
    """
    out = []
    for sec in doc.sections:
        section_text = sec["text"]
        if len(section_text) <= config.CHUNK_SIZE:
            pieces = [section_text]
        else:
            pieces = _fixed_size_chunks(section_text, config.CHUNK_SIZE, config.CHUNK_OVERLAP)

        for i, piece in enumerate(pieces):
            if not piece.strip():
                continue
            out.append(Chunk(
                chunk_id=f"{doc.doc_id}-recursive-{sec['heading']}-{i}"[:120],
                doc_id=doc.doc_id,
                source_file=doc.source_file,
                section_heading=sec["heading"],
                page=sec.get("page"),
                text=piece.strip(),
                char_count=len(piece.strip()),
                strategy="recursive",
            ))
    return out


def chunk_semantic(doc: LoadedDocument, embed_fn) -> List[Chunk]:
    """
    Splits sentences into chunks, starting a new chunk whenever cosine
    similarity to the running chunk embedding drops below threshold —
    i.e. a topic shift. `embed_fn` is injected (from embeddings.py) so
    this module has no hard dependency on the model.
    """
    import re
    import numpy as np

    sentences = re.split(r"(?<=[.!?])\s+", doc.text)
    sentences = [s.strip() for s in sentences if s.strip()]
    if not sentences:
        return []

    embeddings = embed_fn(sentences)
    out = []
    current_sentences = [sentences[0]]
    current_vecs = [embeddings[0]]
    chunk_idx = 0

    def _cos_sim(a, b):
        a, b = np.array(a), np.array(b)
        denom = (np.linalg.norm(a) * np.linalg.norm(b))
        return float(np.dot(a, b) / denom) if denom else 0.0

    for sent, vec in zip(sentences[1:], embeddings[1:]):
        running_centroid = np.mean(current_vecs, axis=0)
        sim = _cos_sim(running_centroid, vec)

        joined_len = len(" ".join(current_sentences))
        if sim < config.SEMANTIC_SIMILARITY_THRESHOLD or joined_len >= config.CHUNK_SIZE * 1.5:
            text = " ".join(current_sentences).strip()
            if text:
                out.append(Chunk(
                    chunk_id=f"{doc.doc_id}-semantic-{chunk_idx}",
                    doc_id=doc.doc_id,
                    source_file=doc.source_file,
                    section_heading="semantic-segment",
                    page=None,
                    text=text,
                    char_count=len(text),
                    strategy="semantic",
                ))
                chunk_idx += 1
            current_sentences = [sent]
            current_vecs = [vec]
        else:
            current_sentences.append(sent)
            current_vecs.append(vec)

    if current_sentences:
        text = " ".join(current_sentences).strip()
        if text:
            out.append(Chunk(
                chunk_id=f"{doc.doc_id}-semantic-{chunk_idx}",
                doc_id=doc.doc_id,
                source_file=doc.source_file,
                section_heading="semantic-segment",
                page=None,
                text=text,
                char_count=len(text),
                strategy="semantic",
            ))
    return out


def chunk_document(doc: LoadedDocument, strategy: str = None, embed_fn=None) -> List[Chunk]:
    strategy = strategy or config.DEFAULT_CHUNK_STRATEGY
    if strategy == "fixed":
        return chunk_fixed(doc)
    if strategy == "recursive":
        return chunk_recursive(doc)
    if strategy == "semantic":
        if embed_fn is None:
            raise ValueError("semantic chunking requires embed_fn")
        return chunk_semantic(doc, embed_fn)
    raise ValueError(f"Unknown chunking strategy: {strategy}")
