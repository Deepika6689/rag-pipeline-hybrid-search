"""
Thin wrapper around Groq so the rest of the app never touches the SDK
directly — makes it a one-line swap to OpenAI/Anthropic later if needed.
"""
from functools import lru_cache
from app import config


@lru_cache(maxsize=1)
def _get_client():
    from groq import Groq
    if not config.GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY not set. Add it to your .env file.")
    return Groq(api_key=config.GROQ_API_KEY)


def chat(messages: list, model: str = None, temperature: float = 0.2, max_tokens: int = 1024) -> str:
    client = _get_client()
    response = client.chat.completions.create(
        model=model or config.GROQ_MODEL,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content


def chat_json(messages: list, model: str = None, temperature: float = 0.0, max_tokens: int = 512) -> str:
    """Ask for strict JSON back (used by the judge / citation verifier)."""
    client = _get_client()
    response = client.chat.completions.create(
        model=model or config.GROQ_JUDGE_MODEL,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        response_format={"type": "json_object"},
    )
    raw = response.choices[0].message.content
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:]
    return cleaned.strip()