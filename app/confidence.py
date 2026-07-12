from typing import List
from app import config


def retrieval_confidence(chunks: List[dict]) -> float:
    """Fraction of retrieved chunks the cross-encoder judged relevant.

    Previous version used a top-heavy weighted average (0.7 * best score +
    0.3 * average of the rest) squashed into an arbitrary +/-10 range. That
    meant a single strong match could mask four weak ones -- a chunk set of
    [0.88, -0.01, -2.35, -4.14, -4.54] (only 1/5 chunks genuinely relevant)
    still scored ~49% confidence, which is misleading.

    This version instead counts what fraction of the retrieved chunks the
    reranker actually considered relevant, using its natural decision
    boundary (0.0 is the standard threshold for ms-marco cross-encoders --
    positive scores mean "relevant", negative mean "not"). The same chunk
    set above now correctly scores 1/5 = 20%, which is a much more honest
    signal when a query mostly returned filler from RRF fusion rather than
    strong matches.
    """
    if not chunks:
        return 0.0
    scores = [c.get("rerank_score", 0.0) for c in chunks]
    relevant = sum(1 for s in scores if s > 0)
    return relevant / len(scores)


def composite_confidence(retrieval_conf: float, citation_accuracy: float | None, claim_count: int) -> dict:
    """
    Blend retrieval confidence with citation coverage. If there were no
    citations to check (e.g. model said "I don't know"), fall back to
    retrieval confidence alone.
    """
    if citation_accuracy is None:
        composite = retrieval_conf * 0.7  # penalize slightly for no verifiable claims
    else:
        composite = 0.5 * retrieval_conf + 0.5 * citation_accuracy

    completeness = min(1.0, claim_count / 3) if claim_count else 0.0

    return {
        "retrieval_confidence": round(retrieval_conf, 3),
        "citation_accuracy": round(citation_accuracy, 3) if citation_accuracy is not None else None,
        "answer_completeness": round(completeness, 3),
        "composite_confidence": round(composite, 3),
        "is_low_confidence": composite < config.LOW_CONFIDENCE_THRESHOLD,
    }


def build_low_confidence_response(question: str, chunks: List[dict]) -> dict:
    found_sections = [
        f"{c.get('metadata', {}).get('source_file', 'unknown')} — {c.get('metadata', {}).get('section_heading', 'N/A')}"
        for c in chunks
    ]
    return {
        "answer": (
            "I don't have enough reliably relevant context to answer this confidently. "
            "Here's what I found that might be related — worth checking manually."
        ),
        "possibly_relevant_sources": found_sections,
        "recommendation": "Consider rephrasing the question or checking the listed sources directly.",
    }
