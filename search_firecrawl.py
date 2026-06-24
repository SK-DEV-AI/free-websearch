from __future__ import annotations

from typing import Any

import httpx

from config import _next_firecrawl_key

FIRECRAWL_SEARCH = "https://api.firecrawl.dev/v2/search"
FIRECRAWL_MAP = "https://api.firecrawl.dev/v2/map"


async def search_firecrawl(query: str, n: int = 10, sources: str = "",
                           tbs: str = "", country: str = "") -> list[dict]:
    """Firecrawl search — 1K free credits/month per key."""
    key = _next_firecrawl_key()
    if not key:
        return []
    body: dict[str, Any] = {"query": query, "limit": min(n, 20),
                            "scrapeOptions": {"formats": ["markdown"], "onlyMainContent": True}}
    if sources:
        body["sources"] = sources
    if tbs:
        body["tbs"] = tbs
    if country:
        body["country"] = country
    try:
        async with httpx.AsyncClient(timeout=15) as c:
            r = await c.post(FIRECRAWL_SEARCH, json=body,
                             headers={"Authorization": f"Bearer {key}",
                                      "Content-Type": "application/json"})
        if r.status_code != 200:
            return []
        data = r.json()
        results: list[dict] = []
        for item in (data.get("data", {}).get("web", []) or []):
            url = item.get("url", "")
            if not url:
                continue
            results.append({"title": item.get("title", ""), "url": url,
                           "snippet": (item.get("markdown", "") or item.get("description", "") or "")[:500],
                           "engine": "firecrawl"})
        return results
    except httpx.HTTPError:
        return []


async def map_firecrawl(url: str, search: str = "", limit: int = 100,
                         include_subdomains: bool = True) -> list[dict]:
    """Firecrawl Map — discover URL structure of a site. Returns {url, title, description}[]."""
    key = _next_firecrawl_key()
    if not key:
        return []
    body: dict[str, Any] = {"url": url, "limit": min(limit, 5000),
                            "includeSubdomains": include_subdomains}
    if search:
        body["search"] = search
    try:
        async with httpx.AsyncClient(timeout=30) as c:
            r = await c.post(FIRECRAWL_MAP, json=body,
                             headers={"Authorization": f"Bearer {key}",
                                      "Content-Type": "application/json"})
        if r.status_code != 200:
            return []
        data = r.json()
        return data.get("links", [])
    except (httpx.HTTPError, ValueError):
        return []
