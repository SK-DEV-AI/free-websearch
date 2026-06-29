from __future__ import annotations

import os
import re
from typing import Any

import httpx

from config import get_http_client

ANYSEARCH_URL = "https://api.anysearch.com/v1/search"
ANYSEARCH_KEY = os.environ.get("ANYSEARCH_KEY", "")

# Simple domain heuristics — general web search is the fallback
_DOMAIN_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"(?i)\b(cve-\d{4}-\d{4,})\b"), "security"),
    (re.compile(r"(?i)\b(doi:\s*10\.\d{4,})\b"), "academic"),
    (re.compile(r"(?i)\barxiv\b"), "academic"),
    (re.compile(r"(?i)\$\w{1,5}\b"), "finance"),
    (re.compile(r"(?i)\b(stock|share price|market cap)\b"), "finance"),
    (re.compile(r"\b[A-Z]{2,4}\b.*\b(stock|NYSE|NASDAQ|ticker)\b"), "finance"),
    (re.compile(r"(?i)\b(code|grep|import |def |function|class |api )\b"), "code"),
    (re.compile(r"(?i)\b(python|rust|typescript|go(lang)?|react|node|fastapi|django|flask)\b"), "code"),
    (re.compile(r"(?i)\b(hospital|symptom|symptoms|diagnosis|treatment|cure|patient|disease)\b"), "health"),
    (re.compile(r"(?i)\b(travel|hotels?|flight|booking|destination)\b"), "travel"),
    (re.compile(r"(?i)\b(patent|us\d{7,11}b\d|ep\d{7})\b"), "legal"),
]


def _detect_domain(query: str) -> str:
    for pattern, domain in _DOMAIN_PATTERNS:
        if pattern.search(query):
            return domain
    return ""


async def search_anysearch(query: str, count: int = 10, domain: str = "") -> list[dict]:
    if not ANYSEARCH_KEY:
        return []
    if not domain:
        domain = _detect_domain(query)
    body: dict[str, Any] = {"query": query, "count": min(max(count, 1), 10)}
    if domain:
        body["domain"] = domain
    try:
        c = get_http_client()
        r = await c.post(
            ANYSEARCH_URL,
            json=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {ANYSEARCH_KEY}",
            },
            timeout=10,
        )
        if r.status_code != 200:
            return []
        data = r.json()
        if data.get("code") != 0:
            return []
        results: list[dict] = []
        for item in (data.get("data", {}).get("results", []) or []):
            url = item.get("url", "")
            if not url:
                continue
            entry: dict[str, Any] = {
                "title": item.get("title", ""),
                "url": url,
                "snippet": (item.get("snippet", "") or "")[:500],
            }
            content = item.get("content", "")
            if content:
                entry["content"] = content[:3000]
            results.append(entry)
        return results
    except (httpx.HTTPError, ValueError, KeyError):
        return []
