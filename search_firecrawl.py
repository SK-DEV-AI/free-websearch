from __future__ import annotations

from typing import Any

import httpx

from config import _next_firecrawl_key

FIRECRAWL_SEARCH = "https://api.firecrawl.dev/v2/search"
FIRECRAWL_MAP = "https://api.firecrawl.dev/v2/map"
FIRECRAWL_SCRAPE = "https://api.firecrawl.dev/v2/scrape"


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
        for item in (data.get("data", {}).get("news", []) or []):
            url = item.get("url", "")
            if not url:
                continue
            results.append({"title": item.get("title", ""), "url": url,
                           "snippet": (item.get("markdown", "") or item.get("description", "") or "")[:500],
                           "engine": "firecrawl-news"})
        for item in (data.get("data", {}).get("images", []) or []):
            url = item.get("url", "")
            if not url:
                continue
            results.append({"title": item.get("title", ""), "url": url,
                           "snippet": (item.get("description", "") or "")[:500],
                           "engine": "firecrawl-images"})
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


async def firecrawl_scrape(url: str, formats: list[str] | None = None,
                           only_main_content: bool = True,
                           include_tags: list[str] | None = None,
                           exclude_tags: list[str] | None = None,
                           wait_for: str = "",
                           actions: list[dict] | None = None,
                           timeout: int = 30000) -> dict:
    """Firecrawl Scrape — extract content from a single URL with fine-grained control."""
    key = _next_firecrawl_key()
    if not key:
        return {"success": False, "error": "no API key"}
    body: dict[str, Any] = {"url": url, "formats": formats or ["markdown"],
                            "onlyMainContent": only_main_content}
    if include_tags:
        body["includeTags"] = include_tags
    if exclude_tags:
        body["excludeTags"] = exclude_tags
    if wait_for:
        body["waitFor"] = wait_for
    if actions:
        body["actions"] = actions
    if timeout:
        body["timeout"] = timeout
    try:
        async with httpx.AsyncClient(timeout=60) as c:
            r = await c.post(FIRECRAWL_SCRAPE, json=body,
                             headers={"Authorization": f"Bearer {key}",
                                      "Content-Type": "application/json"})
        if r.status_code != 200:
            return {"success": False, "error": f"HTTP {r.status_code}"}
        data = r.json()
        result_data = data.get("data", {})
        return {"success": True, "url": url,
                "markdown": (result_data.get("markdown", "") or "")[:50000],
                "html": (result_data.get("html", "") or "")[:50000],
                "title": result_data.get("metadata", {}).get("title", ""),
                "description": result_data.get("metadata", {}).get("description", ""),
                "source": "firecrawl-scrape"}
    except (httpx.HTTPError, ValueError) as e:
        return {"success": False, "error": str(e)}
