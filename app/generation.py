from typing import List
from app.llm_client import chat

SYSTEM_PROMPT = """You are a documentation assistant. Answer the user's question using ONLY the numbered context blocks provided below. Follow these rules strictly:

1. Every factual claim you make must be followed by a bracketed citation like [1] or [2] referencing the context block that supports it.
2. You may state the direct logical consequence of something the context explicitly says — e.g. if it says "the IC is not the person fixing the issue" and "the IC does not write code," you may conclude the person fixing the issue writes the code — but never introduce facts, numbers, or specifics that aren't stated or directly implied by the context.
3. If the context genuinely does not contain enough information (even indirectly) to answer, say so explicitly. Do not guess or use outside knowledge.
4. If only part of the question can be answered from the context, answer that part and clearly state what's missing.
5. Keep the answer concise and directly responsive to the question.
"""


def _format_context(chunks: List[dict]) -> str:
    blocks = []
    for i, c in enumerate(chunks, start=1):
        meta = c.get("metadata", {})
        heading = meta.get("section_heading", "N/A")
        source = meta.get("source_file", "unknown")
        blocks.append(f"[{i}] (source: {source} | section: {heading})\n{c['text']}")
    return "\n\n".join(blocks)


def generate_answer(question: str, chunks: List[dict]) -> dict:
    if not chunks:
        return {
            "answer": "I don't have any indexed documents relevant to this question.",
            "context_used": [],
        }

    context_block = _format_context(chunks)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Context:\n\n{context_block}\n\nQuestion: {question}"},
    ]
    answer = chat(messages, temperature=0.2)

    return {
        "answer": answer,
        "context_used": [
            {
                "index": i + 1,
                "chunk_id": c["chunk_id"],
                "source_file": c.get("metadata", {}).get("source_file"),
                "section_heading": c.get("metadata", {}).get("section_heading"),
                "text": c["text"],
            }
            for i, c in enumerate(chunks)
        ],
    }
