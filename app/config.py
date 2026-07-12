import os
from dotenv import load_dotenv

load_dotenv()

# --- LLM (Groq) ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_JUDGE_MODEL = os.getenv("GROQ_JUDGE_MODEL", "llama-3.3-70b-versatile")  # was 8b-instant — too weak/inconsistent as a correctness judge

# --- Embeddings (local, free) ---
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# --- Reranker (local, free) ---
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

# --- Vector store ---
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./storage/chroma")
COLLECTION_NAME = "rag_docs"

# --- BM25 index ---
BM25_INDEX_PATH = os.getenv("BM25_INDEX_PATH", "./storage/bm25_index.pkl")

# --- Chunking ---
DEFAULT_CHUNK_STRATEGY = os.getenv("CHUNK_STRATEGY", "recursive")  # fixed | recursive | semantic
CHUNK_SIZE = 500          # characters
CHUNK_OVERLAP = 75        # characters
SEMANTIC_SIMILARITY_THRESHOLD = 0.55  # below this, start a new semantic chunk

# --- Dedup ---
DEDUP_SIMILARITY_THRESHOLD = 0.95

# --- Retrieval ---
DENSE_TOP_K = 10
SPARSE_TOP_K = 10
RRF_K = 60                # standard RRF constant
DENSE_WEIGHT = 0.7
SPARSE_WEIGHT = 0.3
RERANK_CANDIDATES = 20
FINAL_TOP_K = 5

# --- Confidence thresholds ---
LOW_CONFIDENCE_THRESHOLD = 0.45

# Empirical range for cross-encoder/ms-marco-MiniLM-L-6-v2 logits, used to
# calibrate retrieval_confidence() via min-max normalization instead of an
# uncalibrated sigmoid (which assumes scores are centered at 0 — they aren't).
RERANK_SCORE_FLOOR = -10.0
RERANK_SCORE_CEILING = 10.0

# Cheap pre-generation short-circuit: only skip generation entirely if the
# top retrieved chunk is so weak that generation is almost certainly a waste
# (e.g. genuinely no relevant document). Deliberately much lower than
# LOW_CONFIDENCE_THRESHOLD, which is now checked *after* generation using
# citation-verified confidence, not raw retrieval score alone.
MIN_RETRIEVAL_THRESHOLD = 0.15

# --- Storage for raw + processed docs ---
RAW_DOCS_DIR = os.getenv("RAW_DOCS_DIR", "./data/raw")
PROCESSED_DOCS_DIR = os.getenv("PROCESSED_DOCS_DIR", "./data/processed")
