from __future__ import annotations

import asyncio
import urllib.parse
from typing import Any

import httpx

from config import EXA_KEY, EXA_SIMILAR


async def _exa_request(method: str, url: str, body: dict | None, headers: dict,
                       timeout: int = 30, retries: int = 2) -> httpx.Response:
    """Make Exa API request with automatic retry on transient failures."""
    last_err = None
    for attempt in range(retries + 1):
        try:
            async with httpx.AsyncClient(timeout=timeout) as c:
                if method == "POST":
                    r = await c.post(url, json=body, headers=headers)
                else:
                    r = await c.get(url, headers=headers)
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


async def exa_similar(url: str, count: int = 5, highlights: bool = False,
                      summary: bool = False, subpages: int = 0,
                      include_domains: list | None = None,
                      exclude_domains: list | None = None,
                      category: str = "", system_prompt: str = "",
                      output_schema: dict | None = None,
                      user_location: str = "") -> list[dict]:
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
    if user_location:
        body["userLocation"] = user_location
    r = await _exa_request("POST", EXA_SIMILAR, body,
                           {"x-api-key": EXA_KEY, "Content-Type": "application/json"}, timeout=15)
    data = r.json()
    return [{"title": h.get("title", ""), "url": h.get("url", ""),
             "snippet": (h.get("text", "") or "")[:300],
             "source": "exa-similar", "score": h.get("score", 0)}
            for h in (data.get("results", []) or []) if h.get("url")]
