
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

### `requirements.txt`

```
fastapi==0.115.0
uvicorn[standard]==0.30.6
pydantic==2.9.2
groq>=1.5.0
sentence-transformers==3.1.1
chromadb>=1.5.9
rank-bm25==0.2.2
pypdf==5.0.1
python-dotenv==1.0.1
markdown-it-py==3.0.0
beautifulsoup4==4.12.3
numpy==1.26.4
streamlit==1.38.0
requests==2.32.3
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

## Sample output
<img width="1907" height="921" alt="image" src="https://github.com/user-attachments/assets/09514100-a24d-4269-9ad4-d9d7c403b149" />
<img width="1908" height="917" alt="image" src="https://github.com/user-attachments/assets/52f67b57-0006-4330-945c-e58ee0b1cb6d" />
<img width="1907" height="916" alt="image" src="https://github.com/user-attachments/assets/1186beb0-09d6-48b4-aebe-cfa91e8c4c7e" />

**Query:** *"If a production deployment causes a SEV-1 incident, what's the
process end-to-end — from rollback to postmortem?"*

This question can't be answered from a single document — it requires pulling
the rollback command from `deployment_guide.md` and the incident-response
process from `incident_response.md`, then merging both into one coherent
answer. Hybrid retrieval + fusion + reranking pulled 5 chunks across both
files; the model generated an answer with an inline citation on every claim;
the citation verifier checked each one against its source chunk.

> If a production deployment causes a SEV-1 incident, the process would
> involve running `./scripts/rollback.sh <previous_version_tag>` to revert
> the Kubernetes deployment to the previous image and roll back any
> accompanying database migrations that were marked reversible [1]. The
> incident would be classified as SEV-1, requiring immediate paging of the
> on-call engineer and notification of the incident commander within 5
> minutes [3]. An Incident Commander (IC) would be appointed to coordinate
> the response... After the incident, a written postmortem would be
> required within 3 business days [2].
>
> Note that the context does not explicitly state who performs the rollback
> or how the IC is selected, only that the IC is not necessarily the person
> fixing the issue [5].

| Retrieval confidence | Citation accuracy | Completeness | Composite |
|---|---|---|---|
| 40% | 100% | 100% | 70% |

Every citation was independently verified as supported by its source chunk.
Retrieval confidence sits at 40% rather than near-100% by design — only 2 of
the 5 retrieved chunks scored positively with the cross-encoder (the rest
were pulled in by fusion but judged weakly relevant), and the confidence
metric reports that honestly instead of letting one strong match inflate the
whole score.

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
