"""
Seed script: indexes data/sample_docs so reviewers can spin the project
up and start asking questions immediately, no setup required beyond
GROQ_API_KEY.

Usage: python scripts/ingest_sample_docs.py [--strategy recursive|fixed|semantic]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.vectorstore import VectorStore
from app.bm25_index import build_from_vectorstore
from app.ingestion import load_directory, persist_processed
from app.chunking import chunk_document
from app.embeddings import embed_texts
from app import config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", default="recursive", choices=["fixed", "recursive", "semantic"])
    parser.add_argument("--dir", default="data/sample_docs")
    args = parser.parse_args()

    print(f"Loading documents from {args.dir} ...")
    docs = load_directory(args.dir)
    print(f"Found {len(docs)} document(s).")

    store = VectorStore()
    total_added, total_skipped = 0, 0

    embed_fn = embed_texts if args.strategy == "semantic" else None

    for doc in docs:
        persist_processed(doc)
        chunks = chunk_document(doc, strategy=args.strategy, embed_fn=embed_fn)
        result = store.add_chunks(chunks)
        print(f"  {os.path.basename(doc.source_file)}: {len(chunks)} chunks -> {result['added']} added, {result['skipped_duplicates']} skipped as dupes")
        total_added += result["added"]
        total_skipped += result["skipped_duplicates"]

    print("Building BM25 index...")
    build_from_vectorstore(store)

    print(f"\nDone. {total_added} chunks indexed total, {total_skipped} duplicates skipped.")
    print("Start the API with: uvicorn app.main:app --reload")
    print("Or the dashboard with: streamlit run frontend/dashboard.py")


if __name__ == "__main__":
    main()
