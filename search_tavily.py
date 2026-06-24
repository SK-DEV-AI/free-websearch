from __future__ import annotations

import asyncio
from typing import Any

import httpx

from config import _next_tavily_key

TAVILY_SEARCH = "https://api.tavily.com/search"
TAVILY_EXTRACT = "https://api.tavily.com/extract"
TAVILY_RESEARCH = "https://api.tavily.com/research"


async def search_tavily(query: str, n: int = 10, topic: str = "general",
                        time_range: str = "", include_raw_content: bool = False,
                        search_depth: str = "basic", include_answer: bool = True,
                        include_domains: list | None = None,
                        exclude_domains: list | None = None,
                        country: str = "",
                        chunks_per_source: int = 0,
                        start_date: str = "", end_date: str = "",
                        exact_phrase: bool = False) -> list[dict]:
    """Tavily search — AI-optimized with 1K free reqs/month per key. Returns answer + results."""
    key = _next_tavily_key()
    if not key:
        return []
    body: dict[str, Any] = {"query": query, "search_depth": search_depth,
                            "max_results": min(n, 10), "include_answer": include_answer,
                            "include_raw_content": include_raw_content,
                            "topic": topic}
    if start_date:
        body["start_date"] = start_date
    if end_date:
        body["end_date"] = end_date
    if time_range and not start_date and not end_date:
        body["time_range"] = time_range
    if exact_phrase:
        body["include_answer"] = True
        body["query"] = f'"{query}"'
    if include_domains:
        body["include_domains"] = include_domains
    if exclude_domains:
        body["exclude_domains"] = exclude_domains
    if country:
        body["country"] = country
    if chunks_per_source > 0:
        body["chunks_per_source"] = chunks_per_source
    try:
        async with httpx.AsyncClient(timeout=15) as c:
            r = await c.post(TAVILY_SEARCH, json=body,
                             headers={"Authorization": f"Bearer {key}",
                                      "Content-Type": "application/json"})
        if r.status_code != 200:
            return []
        data = r.json()
        results: list[dict] = []
        answer = data.get("answer", "")
        if answer:
            results.append({"title": "Tavily Answer", "url": "",
                           "snippet": answer[:1000], "engine": "tavily-answer"})
        for item in (data.get("results", []) or []):
            url = item.get("url", "")
            if not url:
                continue
            entry: dict[str, Any] = {"title": item.get("title", ""), "url": url,
                           "snippet": (item.get("content", "") or "")[:500],
                           "engine": "tavily"}
            if item.get("score"):
                entry["score"] = item["score"]
            results.append(entry)
        return results
    except httpx.HTTPError:
        return []


async def tavily_extract(urls: list[str], include_images: bool = False) -> list[dict]:
    """Tavily Extract — pull content from specific URLs via Tavily API."""
    key = _next_tavily_key()
    if not key or not urls:
        return []
    body: dict[str, Any] = {"urls": urls[:10]}
    if include_images:
        body["include_images"] = True
    try:
        async with httpx.AsyncClient(timeout=30) as c:
            r = await c.post(TAVILY_EXTRACT, json=body,
                             headers={"Authorization": f"Bearer {key}",
                                      "Content-Type": "application/json"})
        if r.status_code != 200:
            return []
        data = r.json()
        results: list[dict] = []
        for item in (data.get("results", []) or []):
            url = item.get("url", "")
            if not url:
                continue
            entry: dict[str, Any] = {"url": url,
                          "title": item.get("title", ""),
                          "content": (item.get("raw_content", "") or item.get("content", ""))[:10000],
                          "engine": "tavily-extract"}
            if include_images and item.get("images"):
                entry["images"] = item["images"]
            results.append(entry)
        return results
    except httpx.HTTPError:
        return []


async def research_tavily(query: str, model: str = "auto", output_schema: dict | None = None,
                           wait_seconds: int = 30, output_length: str = "standard",
                           citation_format: str = "numbered") -> dict:
    """Tavily Research — async multi-angle AI research agent. Returns synthesis + sources."""
    key = _next_tavily_key()
    if not key:
        return {"success": False, "error": "no API key"}
    body: dict[str, Any] = {"input": query, "model": model, "output_length": output_length,
                            "citation_format": citation_format}
    if output_schema:
        body["output_schema"] = output_schema
    try:
        async with httpx.AsyncClient(timeout=15) as c:
            r = await c.post(TAVILY_RESEARCH, json=body,
                             headers={"Authorization": f"Bearer {key}",
                                      "Content-Type": "application/json"})
            if r.status_code not in (200, 201):
                return {"success": False, "error": f"HTTP {r.status_code}"}
            data = r.json()
            request_id = data.get("request_id", "")
            if not request_id:
                return {"success": False, "error": "no request_id"}
            deadline = asyncio.get_event_loop().time() + wait_seconds
            while asyncio.get_event_loop().time() < deadline:
                await asyncio.sleep(3)
                sr = await c.get(f"{TAVILY_RESEARCH}/{request_id}",
                                 headers={"Authorization": f"Bearer {key}"})
                if sr.status_code != 200:
                    continue
                sd = sr.json()
                status = sd.get("status", "")
                if status == "completed":
                    return {"success": True, "request_id": request_id,
                            "answer": sd.get("answer", ""),
                            "sources": sd.get("sources", []) or sd.get("results", []),
                            "model_used": sd.get("model", model)}
                if status in ("failed", "error"):
                    return {"success": False, "error": sd.get("error", status),
                            "request_id": request_id}
            return {"success": False, "error": "timeout", "request_id": request_id,
                    "hint": "use a larger wait_seconds or keep checking manually"}
    except (httpx.HTTPError, asyncio.TimeoutError, ValueError) as e:
        return {"success": False, "error": f"{type(e).__name__}: {e}"}
