from __future__ import annotations

import asyncio
import urllib.parse
from typing import Any

import httpx

from config import EXA_KEY, EXA_SIMILAR, get_http_client

EXA_SEARCH = "https://api.exa.ai/search"


async def _exa_request(method: str, url: str, body: dict | None, headers: dict,
                       timeout: int = 30, retries: int = 2) -> httpx.Response:
    """Make Exa API request with automatic retry on transient failures."""
    last_err = None
    for attempt in range(retries + 1):
        try:
            c = get_http_client()
            if method == "POST":
                r = await c.post(url, json=body, headers=headers, timeout=timeout)
            else:
                r = await c.get(url, headers=headers, timeout=timeout)
            if r.status_code == 429:
                if attempt < retries:
                    await asyncio.sleep(2 * (attempt + 1))
                    continue
            return r
        except (httpx.ConnectError, httpx.ReadTimeout, httpx.WriteTimeout) as e:
            last_err = e
            if attempt < retries:
                await asyncio.sleep(1 * (attempt + 1))
            continue
    raise last_err or RuntimeError("Exa request failed after retries")


async def exa_search(query: str, num_results: int = 10, type: str = "auto",
                     include_domains: list | None = None,
                     exclude_domains: list | None = None,
                     start_published_date: str = "",
                     end_published_date: str = "",
                     highlights: bool = False, summary: bool = False) -> list[dict]:
    """Exa search — find content across the web by semantic query."""
    if not EXA_KEY:
        return []
    body: dict[str, Any] = {"query": query, "numResults": min(num_results, 30), "type": type}
    contents: dict[str, Any] = {"text": True}
    if highlights:
        contents["highlights"] = True
    if summary:
        contents["summary"] = True
    body["contents"] = contents
    if include_domains:
        body["includeDomains"] = include_domains
    if exclude_domains:
        body["excludeDomains"] = exclude_domains
    if start_published_date:
        body["startPublishedDate"] = start_published_date
    if end_published_date:
        body["endPublishedDate"] = end_published_date
    r = await _exa_request("POST", EXA_SEARCH, body,
                           {"x-api-key": EXA_KEY, "Content-Type": "application/json"}, timeout=15)
    data = r.json()
    return [{"title": h.get("title", ""), "url": h.get("url", ""),
             "snippet": (h.get("text", "") or "")[:300],
             "source": "exa-search", "score": h.get("score", 0)}
            for h in (data.get("results", []) or []) if h.get("url")]


async def exa_similar(url: str, count: int = 5, highlights: bool = False,
                      summary: bool = False, subpages: int = 0,
                      include_domains: list | None = None,
                      exclude_domains: list | None = None,
                      category: str = "", system_prompt: str = "",
                      output_schema: dict | None = None,
                      stream: bool = False,
                      user_location: str = "",
                      start_published_date: str = "",
                      end_published_date: str = "") -> list[dict]:
    if not EXA_KEY:
        return []
    body: dict[str, Any] = {"url": url, "numResults": count}
    contents: dict[str, Any] = {"text": True}
    if highlights:
        contents["highlights"] = True
    if summary:
        contents["summary"] = True
    if subpages > 0:
        contents["subpages"] = min(subpages, 100)
    body["contents"] = contents
    if include_domains:
        body["includeDomains"] = include_domains
    if exclude_domains:
        body["excludeDomains"] = exclude_domains
    if category:
        body["category"] = category
    if system_prompt:
        body["systemPrompt"] = system_prompt
    if output_schema:
        body["outputSchema"] = output_schema
    if stream:
        body["stream"] = True
    if user_location:
        body["userLocation"] = user_location
    if start_published_date:
        body["startPublishedDate"] = start_published_date
    if end_published_date:
        body["endPublishedDate"] = end_published_date
    r = await _exa_request("POST", EXA_SIMILAR, body,
                           {"x-api-key": EXA_KEY, "Content-Type": "application/json"}, timeout=15)
    data = r.json()
    results: list[dict] = []
    for h in (data.get("results", []) or []):
        if not h.get("url"):
            continue
        entry: dict[str, Any] = {"title": h.get("title", ""), "url": h.get("url", ""),
                                  "snippet": (h.get("text", "") or "")[:300],
                                  "source": "exa-similar", "score": h.get("score", 0)}
        if highlights and h.get("highlights"):
            entry["highlights"] = h["highlights"]
        if summary and h.get("summary"):
            entry["summary"] = h["summary"]
        if h.get("publishedDate"):
            entry["published_date"] = h["publishedDate"]
        results.append(entry)
    return results
