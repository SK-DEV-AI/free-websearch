from __future__ import annotations

import os
from typing import Any

import httpx

from config import get_http_client

ANYSEARCH_URL = "https://api.anysearch.com/v1/search"
ANYSEARCH_KEY = os.environ.get("ANYSEARCH_KEY", "")


async def search_anysearch(query: str, count: int = 10) -> list[dict]:
    if not ANYSEARCH_KEY:
        return []
    body: dict[str, Any] = {"query": query, "count": min(max(count, 1), 10)}
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
