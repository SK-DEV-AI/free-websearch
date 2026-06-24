from __future__ import annotations

import asyncio
from typing import Any

import httpx

from config import _next_tavily_key

TAVILY_SEARCH = "https://api.tavily.com/search"
TAVILY_RESEARCH = "https://api.tavily.com/research"


async def search_tavily(query: str, n: int = 10, topic: str = "general",
                        time_range: str = "", include_raw_content: bool = False,
                        search_depth: str = "basic") -> list[dict]:
    """Tavily search — AI-optimized with 1K free reqs/month per key. Returns answer + results."""
    key = _next_tavily_key()
    if not key:
        return []
    body: dict[str, Any] = {"query": query, "search_depth": search_depth,
                            "max_results": min(n, 10), "include_answer": True,
                            "include_raw_content": include_raw_content,
                            "topic": topic}
    if topic != "general":
        body["topic"] = topic
    if time_range:
        body["time_range"] = time_range
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
            results.append({"title": item.get("title", ""), "url": url,
                           "snippet": (item.get("content", "") or "")[:500],
                           "engine": "tavily"})
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
