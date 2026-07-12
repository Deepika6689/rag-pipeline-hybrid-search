"""
Orchestrates the full ask() flow: hybrid retrieve -> generate ->
verify citations -> score confidence -> (maybe) fall back to
"I don't know" gracefully. This is what main.py's /v1/ask calls.
"""
from app import config
from app.vectorstore import VectorStore
from app.bm25_index import BM25Index
from app.retrieval import hybrid_retrieve
from app.generation import generate_answer
from app.citations import verify_citations
from app.confidence import retrieval_confidence, composite_confidence, build_low_confidence_response


def ask(question: str, store: VectorStore, bm25: BM25Index, use_hybrid: bool = True, verify: bool = True) -> dict:
    retrieval = hybrid_retrieve(question, store, bm25, use_hybrid=use_hybrid)
    final_chunks = retrieval["final_chunks"]

    r_conf = retrieval_confidence(final_chunks)

    # DEBUG: temporary — confirms what retrieval_confidence() actually saw.
    # Compare this printed value/scores against what the dashboard displays
    # for the SAME question. If they don't match, the mismatch is happening
    # somewhere between here and the dashboard render (e.g. chunk objects
    # being mutated again after this point, or a stale cached result).
    # Remove this line once the numbers are confirmed consistent.
    print(f"[DEBUG] r_conf={r_conf}, scores={[c.get('rerank_score') for c in final_chunks]}")

    # Only short-circuit before generation for the extreme case: nothing
    # retrieved, or the top chunk is essentially noise (no plausible match
    # in the corpus at all). Anything past this bar gets a real generation
    # attempt rather than being pre-judged on retrieval score alone.
    if not final_chunks or r_conf < config.MIN_RETRIEVAL_THRESHOLD:
        fallback = build_low_confidence_response(question, final_chunks)
        conf = composite_confidence(r_conf, None, 0)
        return {
            "question": question,
            "answer": fallback["answer"],
            "low_confidence_fallback": fallback,
            "confidence": conf,
            "retrieved_chunks": final_chunks,
            "dense_only_chunks": retrieval["dense_only_chunks"],
        }

    gen = generate_answer(question, final_chunks)

    citation_report = None
    if verify:
        citation_report = verify_citations(gen["answer"], gen["context_used"], question=question)
        citation_accuracy = citation_report["citation_accuracy"]
        claim_count = citation_report["total_citations"]
    else:
        citation_accuracy = None
        claim_count = gen["answer"].count("[")

    conf = composite_confidence(r_conf, citation_accuracy, claim_count)

    # Deliberately NOT swapping the generated answer out for the canned
    # "I don't know" message just because citation_accuracy or composite
    # confidence came back under threshold here. The citation verifier is
    # itself an LLM judge with its own noise (same flakiness class as the
    # eval judge) -- gating the final answer on it caused far more good
    # answers to get discarded than bad ones caught (qa-002/003/006/009/010
    # all regressed to the fallback message once this gate was added, even
    # though several of those answers were correct).
    #
    # `is_low_confidence` is still computed and returned below so the caller
    # (dashboard, API response) can show it as a caveat/warning next to the
    # real answer -- the person gets both the answer and the "double check
    # this" signal, instead of losing the answer entirely to a shaky judge.
    return {
        "question": question,
        "answer": gen["answer"],
        "context_used": gen["context_used"],
        "citation_report": citation_report,
        "confidence": conf,
        "retrieved_chunks": final_chunks,
        "dense_only_chunks": retrieval["dense_only_chunks"],
    }