# RAG Pipeline with Hybrid Search Over Internal Docs

A production-shaped Retrieval-Augmented Generation system: it ingests internal
documentation, indexes it with **both dense vector search and sparse BM25
keyword search**, fuses the two with Reciprocal Rank Fusion, reranks with a
cross-encoder, and generates grounded answers with **verified inline
citations** — refusing to answer confidently when retrieval quality is low.

This is not a "call an LLM API and deploy a chatbot" demo. The things it does
that most weekend RAG projects skip:

- **Hybrid retrieval, not just vector search.** Dense embeddings miss exact
  keyword matches (config keys, error codes, function names) that BM25 catches
  every time. This pipeline runs both and fuses them.
- **Citation verification.** Every `[n]` citation the model produces is
  independently checked by an LLM-as-judge against the actual source chunk —
  catching hallucinated citations before they reach a user.
- **Confidence-gated answers.** If retrieval confidence is too low, the system
  says so explicitly and points to what it *did* find, instead of fabricating
  an answer.
- **A real eval framework**, not vibes. A 10-question golden Q&A set (lookup,
  multi-hop, no-answer, ambiguous cases) scores correctness, faithfulness, and
  citation accuracy — and can A/B three chunking strategies against each other.

## Architecture

```
Documents (.md/.txt/.html/.pdf)
        │
        ▼
 Ingestion + Chunking  ──►  fixed | recursive | semantic (switchable)
        │
        ▼
 Embeddings (local, free — sentence-transformers)  +  Dedup (cosine > 0.95 skipped)
        │
        ▼
 ┌─────────────┐        ┌──────────────┐
 │  ChromaDB   │        │  BM25 index  │
 │ (dense)     │        │ (sparse)     │
 └──────┬──────┘        └──────┬───────┘
        │        Reciprocal     │
        └───────► Rank Fusion ◄─┘
                      │
                      ▼
            Cross-encoder Rerank (top 20 → top 5)
                      │
                      ▼
         Grounded Generation (Groq) + inline [n] citations
                      │
                      ▼
        Citation Verification (LLM-as-judge per claim)
                      │
                      ▼
           Composite Confidence Score → answer OR
           graceful "here's what I found, check manually"
```

## Why Groq + local embeddings (a deliberate deviation from OpenAI)

The original spec calls for OpenAI for both generation and embeddings. This
build uses **Groq** for generation (fast, generous free tier) and
**sentence-transformers running locally** for embeddings and reranking —
zero API cost for indexing, no external rate limit when re-indexing
thousands of chunks, and no vendor lock-in on the retrieval side. This is a
genuine architectural tradeoff, not a shortcut, and it's worth leading with
in an interview: *"I kept the embedding layer local so indexing cost is zero
and I'm never rate-limited by an external provider while re-indexing."*

## Setup

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# edit .env and add your free Groq key from https://console.groq.com
```

## Quickstart

```bash
# 1. Index the sample corpus (3 internal-doc-style markdown files)
python scripts/ingest_sample_docs.py --strategy recursive

# 2. Start the API
uvicorn app.main:app --reload

# 3. In another terminal, start the dashboard
streamlit run frontend/dashboard.py
```

Then open `http://localhost:8501` and ask something like:
*"What's the rate limit for a read-write API key, and how does that compare
to admin keys?"*

## API

| Endpoint | Method | Description |
|---|---|---|
| `/v1/ask` | POST | `{"question": "...", "use_hybrid": true, "verify_citations": true}` → grounded answer, citations, confidence breakdown |
| `/v1/documents` | GET | List indexed documents and chunk counts |
| `/v1/ingest` | POST | `{"directory": "data/sample_docs", "strategy": "recursive"}` → index new documents |

Full interactive docs at `http://localhost:8000/docs`.

## Running the eval suite

```bash
python eval/run_eval.py                      # full correctness + citation + confidence report
python eval/run_eval.py --compare-hybrid      # hybrid vs. dense-only, head to head
```

To compare chunking strategies, re-ingest with each strategy into a fresh
`CHROMA_PERSIST_DIR` and run the eval against each — the results table below
shows the format this produces.

| Strategy | Correctness | Avg citation accuracy | Notes |
|---|---|---|---|
| fixed | fill in after running | | baseline, ignores structure |
| recursive | fill in after running | | respects section headings |
| semantic | fill in after running | | splits on topic drift |

## Design decisions worth discussing in an interview

1. **Why RRF instead of just picking the higher-scoring result per source?**
   RRF is rank-based, not score-based, so it doesn't require dense cosine
   similarity and BM25 scores to be on comparable scales — which they never
   are. It combines *ranking agreement* instead.
2. **Why rerank after fusion instead of relying on fusion alone?**
   RRF is a cheap, coarse merge. The cross-encoder actually reads
   query+chunk together and scores relevance directly — it catches cases
   where a chunk ranked well in both lists but isn't actually relevant to
   *this specific* question.
3. **Why verify citations with a second, smaller model instead of the
   generation model?** Cheaper and it decorrelates the check from whatever
   mistake the generation model just made — asking the same model to grade
   its own citation is a weaker signal.
4. **Why cap confidence and refuse to answer sometimes?** A wrong but
   confident-sounding answer is worse than a system that says "I'm not
   sure, here's what I found." That's the actual production failure mode
   RAG systems have in practice.

## Project structure

```
app/
  ingestion.py      # Phase 1.1 — multi-format loader
  chunking.py        # Phase 1.2 — fixed / recursive / semantic strategies
  embeddings.py       # local sentence-transformers embeddings
  vectorstore.py      # Phase 1.3/1.4 — ChromaDB + dedup
  bm25_index.py        # Phase 2.2 — sparse retrieval
  retrieval.py          # Phase 2.3/2.4 — RRF fusion + reranking
  llm_client.py          # Groq wrapper
  generation.py           # Phase 3.1 — grounded generation w/ citations
  citations.py             # Phase 3.2 — citation verification
  confidence.py             # Phase 3.3/3.4 — scoring + graceful fallback
  pipeline.py                # orchestrates the full ask() flow
  main.py                     # Phase 5.1 — FastAPI service
eval/
  golden_qa.json      # Phase 4.1 — 10 hand-written test cases
  run_eval.py           # Phase 4.2/4.3 — eval runner + strategy comparison
frontend/
  dashboard.py    # Phase 5.2 — Streamlit query dashboard
data/sample_docs/  # sample internal-doc-style corpus
scripts/ingest_sample_docs.py
Dockerfile, docker-compose.yml   # Phase 5.3
```

## Docker

```bash
docker-compose up --build
# API:       http://localhost:8000
# Dashboard: http://localhost:8501
```
