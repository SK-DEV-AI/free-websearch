"""Query expansion via Groq API — round-robin multi-key, non-blocking.

Generates diverse search query variations by combining:
- Synonym/paraphrase variations
- Broader/narrower scope variations
- Different phrasings for the same intent
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from typing import Any

_GROQ_KEYS: list[str] = []
_key_idx = 0
_KEY_LOCK = asyncio.Lock()


def _init():
    global _GROQ_KEYS
    raw = os.environ.get("GROQ_API_KEYS", "")
    if raw:
        _GROQ_KEYS[:] = [k.strip() for k in raw.split(",") if k.strip()]


async def _next_key() -> str | None:
    global _key_idx
    async with _KEY_LOCK:
        if not _GROQ_KEYS:
            _init()
        if not _GROQ_KEYS:
            return None
        key = _GROQ_KEYS[_key_idx % len(_GROQ_KEYS)]
        _key_idx = (_key_idx + 1) % len(_GROQ_KEYS)
        return key


async def expand_query(query: str) -> list[str]:
    """Expand a short/vague query into diverse search variations.

    Returns [original, variation1, variation2, ...] up to 4 total.
    Only expands queries that are short (<5 words or <=60 chars).
    """
    if len(query.split()) >= 8 or len(query) > 80:
        return [query]
    key = await _next_key()
    if not key:
        return [query]

    prompt = (
        "You are a search query optimizer. Generate 2-3 diverse search query "
        "variations for the given query. Each variation should approach the "
        "topic from a different angle:\n"
        "- One broader/wider scope\n"
        "- One more specific/technical\n"
        "- One alternative phrasing\n\n"
        "Rules:\n"
        "- Each query must be concise (under 60 chars)\n"
        "- Do NOT include the original query\n"
        "- Do NOT number or explain\n"
        "- One query per line, plain text only\n\n"
        f"Original query: {query}"
    )
    try:
        import httpx
        async with httpx.AsyncClient(timeout=15) as c:
            r = await c.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                json={
                    "model": "openai/gpt-oss-20b",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.7,
                    "max_tokens": 150,
                },
            )
        if r.status_code == 200:
            text = r.json()["choices"][0]["message"]["content"].strip()
            lines = []
            for q in text.split("\n"):
                raw = q.strip()
                if not raw or len(raw) <= 5:
                    continue
                cleaned = re.sub(r'^[\s*\-•·>]+|^[\d]+[\.\)]\s*', '', raw).strip().strip('"\'[]')
                if not cleaned or len(cleaned) <= 5:
                    cleaned = raw
                if any(kw in cleaned.lower() for kw in [
                    "here", "variation", "query:", "---", "original",
                    "broad", "specific", "alternative",
                ]):
                    continue
                lines.append(cleaned)
            # Deduplicate and keep only unique queries
            seen = {query.lower()}
            unique = []
            for q in lines:
                ql = q.lower()
                if ql not in seen and len(q) > 5:
                    seen.add(ql)
                    unique.append(q)
            if unique:
                return [query] + unique[:3]
    except Exception:
        pass
    return [query]
