"""
Phase 3.2: Citation verification — the quality layer most RAG systems
skip. For every [n] citation in the answer, check whether the sentence
it's attached to is actually supported by that context chunk.
"""
import re
import json
from typing import List
from app.llm_client import chat_json

JUDGE_SYSTEM_PROMPT = """You are a fact-checker verifying a single CLAIM against a SOURCE passage.

The CLAIM is one isolated clause taken out of a longer answer to a QUESTION.
The clause-splitting step strips away surrounding context, so the CLAIM may
read as more general than it actually was in the original answer. Use the
QUESTION to recover the specific scope (e.g. a particular severity level,
key type, or environment) the CLAIM was really talking about before judging.

Follow these rules:
1. The SOURCE may contain more information than the CLAIM needs -- that's fine. Only check whether the specific fact in the CLAIM is present or directly derivable in the SOURCE. Do not penalize the claim for omitting other facts that are also in the SOURCE.
2. Interpret the CLAIM in light of the QUESTION's scope. If the QUESTION concerns one specific case (e.g. "a SEV-1 incident") and the SOURCE states a rule that covers that case -- even if the SOURCE's rule also applies to other cases (e.g. "SEV-1 and SEV-2") -- the CLAIM is supported. Do not flag a claim as an over-generalization merely because the clause itself omits a scope word the QUESTION already established.
3. Ignore differences in formatting: code fences, backticks, quotation marks, capitalization, and punctuation never make a claim unsupported. Only the underlying fact matters.
4. If the CLAIM states a number, first quote the exact matching number from the SOURCE in your reason, then decide.
5. If the CLAIM makes a comparison (e.g. "X is higher than Y" or "X is more restrictive than Y"), first restate both numeric values from the SOURCE in your reason, double-check the comparison is arithmetically correct, and only then decide. A comparison claim is unsupported only if the SOURCE values actually contradict it -- not if you're unsure without checking.
6. Minor rephrasing, rounding, or omitted units do not make a claim unsupported if the core fact matches.

Respond ONLY with JSON: {"supported": true|false, "reason": "one short sentence, quoting the source value(s) you checked"}"""


def _split_claims(answer: str) -> List[dict]:
    """Split the answer into (clause, [citation_indices]) pairs.

    Splits at each run of citation markers rather than at sentence
    boundaries. Sentence-splitting merges compound sentences like
    "X is 1000 [1], Y is 300 [1], and Z is 100 [1]." into one giant claim
    that then gets checked three times against the same merged text --
    the judge sees the whole blob each time and can't tell which specific
    number a given citation is meant to support. Splitting at each [n]
    marker instead means everything since the previous marker (or line
    start) becomes its own isolated claim, so each fact is checked once,
    on its own, regardless of whether the answer used bullets, commas, or
    periods to join facts together.
    """
    claims = []
    for line in answer.splitlines():
        line = line.strip()
        if not line:
            continue
        last_end = 0
        for m in re.finditer(r"(?:\[\d+\]\s*)+", line):
            segment = line[last_end:m.end()]
            indices = [int(n) for n in re.findall(r"\[(\d+)\]", segment)]
            clean = re.sub(r"\[\d+\]", "", segment)
            clean = re.sub(r"^[\*\-\u2022]\s*", "", clean)          # strip leading bullet marker
            clean = re.sub(r"^[,;:\s]+", "", clean)                  # strip leftover leading connective punctuation
            clean = re.sub(r"\s+([.,!?;:])", r"\1", clean).strip()
            if indices and clean:
                claims.append({"sentence": clean, "citation_indices": indices})
            last_end = m.end()
    return claims


def verify_citations(answer: str, context_used: List[dict], question: str = "") -> dict:
    claims = _split_claims(answer)
    context_by_index = {c["index"]: c for c in context_used}

    results = []
    supported_count = 0
    total_count = 0

    for claim in claims:
        for idx in claim["citation_indices"]:
            source = context_by_index.get(idx)
            total_count += 1
            if not source:
                results.append({
                    "claim": claim["sentence"],
                    "citation_index": idx,
                    "supported": False,
                    "reason": "Citation index does not match any retrieved context block.",
                })
                continue

            try:
                raw = chat_json([
                    {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                    {"role": "user", "content": f"QUESTION: {question}\n\nCLAIM: {claim['sentence']}\n\nSOURCE: {source['text']}"},
                ])
                parsed = json.loads(raw)
                supported = bool(parsed.get("supported", False))
                reason = parsed.get("reason", "")
            except Exception as e:
                supported = False
                reason = f"Judge call failed: {e}"

            if supported:
                supported_count += 1

            results.append({
                "claim": claim["sentence"],
                "citation_index": idx,
                "supported": supported,
                "reason": reason,
            })

    coverage = (supported_count / total_count) if total_count else None

    return {
        "citation_checks": results,
        "citation_accuracy": coverage,  # % of citations that actually check out
        "total_citations": total_count,
        "supported_citations": supported_count,
    }