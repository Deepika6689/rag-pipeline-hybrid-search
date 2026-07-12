"""
Phase 5.1: FastAPI service.

  POST /v1/ask        -> ask a question, get a grounded + cited answer
  GET  /v1/documents   -> list indexed documents
  POST /v1/ingest      -> ingest new documents from a directory
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

from app import config
from app.vectorstore import VectorStore
from app.bm25_index import BM25Index, build_from_vectorstore
from app.ingestion import load_directory, persist_processed
from app.chunking import chunk_document
from app.pipeline import ask as pipeline_ask

app = FastAPI(title="RAG Pipeline with Hybrid Search", version="1.0.0")

store = VectorStore()
bm25 = BM25Index()
try:
    bm25.load()
except FileNotFoundError:
    bm25.build(store.all_chunks())


class AskRequest(BaseModel):
    question: str
    use_hybrid: bool = True
    verify_citations: bool = True


class IngestRequest(BaseModel):
    directory: str = config.RAW_DOCS_DIR
    strategy: str = config.DEFAULT_CHUNK_STRATEGY


@app.post("/v1/ask")
def ask_endpoint(req: AskRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="question cannot be empty")
    result = pipeline_ask(req.question, store, bm25, use_hybrid=req.use_hybrid, verify=req.verify_citations)
    return result


@app.get("/v1/documents")
def list_documents():
    return {"documents": store.list_documents()}


@app.post("/v1/ingest")
def ingest_endpoint(req: IngestRequest):
    from app.embeddings import embed_texts

    docs = load_directory(req.directory)
    if not docs:
        raise HTTPException(status_code=404, detail=f"No supported documents found in {req.directory}")

    total_added, total_skipped = 0, 0
    for doc in docs:
        persist_processed(doc)
        embed_fn = (lambda texts: embed_texts(texts)) if req.strategy == "semantic" else None
        chunks = chunk_document(doc, strategy=req.strategy, embed_fn=embed_fn)
        result = store.add_chunks(chunks)
        total_added += result["added"]
        total_skipped += result["skipped_duplicates"]

    global bm25
    bm25 = build_from_vectorstore(store)

    return {
        "documents_ingested": len(docs),
        "chunks_added": total_added,
        "chunks_skipped_as_duplicates": total_skipped,
        "strategy_used": req.strategy,
    }


@app.get("/")
def root():
    return {
        "service": "RAG Pipeline with Hybrid Search",
        "docs": "/docs",
        "indexed_documents": len(store.list_documents()),
    }
