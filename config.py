from __future__ import annotations

import functools
import hashlib
import json
import os
import time
from typing import Any

CACHE_TTL = 120
MAX_RESULTS = 20
GOOGLE_AI_URL = "https://www.google.com/search"
GNEWS_RSS = "https://news.google.com/rss/search"
HELIUM_CDP = "http://127.0.0.1:9222"

NV_KEY = os.environ.get("NV_KEY", "")
NV_BASE = "https://integrate.api.nvidia.com/v1"
NV_EMBED_MODEL = "nvidia/llama-nemotron-embed-1b-v2"

EXA_KEY = os.environ.get("EXA_KEY", "")
EXA_SIMILAR = "https://api.exa.ai/similar"

TAVILY_KEYS = [k.strip() for k in os.environ.get("TAVILY_KEYS", "").split(",") if k.strip()]
FIRECRAWL_KEYS = [k.strip() for k in os.environ.get("FIRECRAWL_KEYS", "").split(",") if k.strip()]

_tavily_idx = 0
_firecrawl_idx = 0

def _next_tavily_key() -> str:
    global _tavily_idx
    if not TAVILY_KEYS:
        return ""
    key = TAVILY_KEYS[_tavily_idx % len(TAVILY_KEYS)]
    _tavily_idx += 1
    return key

def _next_firecrawl_key() -> str:
    global _firecrawl_idx
    if not FIRECRAWL_KEYS:
        return ""
    key = FIRECRAWL_KEYS[_firecrawl_idx % len(FIRECRAWL_KEYS)]
    _firecrawl_idx += 1
    return key

GROQ_API_KEYS = [k.strip() for k in os.environ.get("GROQ_API_KEYS", "").split(",") if k.strip()]
RERANKER_MODEL = os.environ.get("RERANKER_MODEL", "Alibaba-NLP/gte-reranker-modernbert-base")


_cache: dict[str, tuple[float, Any]] = {}
_MAX_CACHE = 500
_CACHE_TTL: dict[str, int] = {"emb": 600, "search": 90, "fetch": 120}


def _cached(key: str) -> Any | None:
    entry = _cache.get(key)
    if entry and time.monotonic() - entry[0] < _CACHE_TTL.get(key.split(":")[0], 300):
        return entry[1]
    return None


def _set_cache(key: str, val: Any):
    _cache[key] = (time.monotonic(), val)
    if len(_cache) > _MAX_CACHE:
        now = time.monotonic()
        stale = [k for k, (t, _) in list(_cache.items()) if now - t > 300]
        for k in stale:
            _cache.pop(k, None)
        if len(_cache) > _MAX_CACHE:
            oldest = sorted(_cache.keys(), key=lambda k: _cache[k][0])[:len(_cache) - _MAX_CACHE]
            for k in oldest:
                _cache.pop(k, None)


def cached(ttl: int = CACHE_TTL):
    def deco(fn):
        @functools.wraps(fn)
        async def wrapper(*args, **kw):
            raw = json.dumps([args, kw], sort_keys=True, default=str)
            key = f"{fn.__name__}:{hashlib.md5(raw.encode()).hexdigest()}"
            now = time.monotonic()
            entry = _cache.get(key)
            if entry and now - entry[0] < ttl:
                return entry[1]
            result = await fn(*args, **kw)
            _cache[key] = (now, result)
            if len(_cache) > _MAX_CACHE:
                cutoff = now - 300
                stale = [k for k, (t, _) in _cache.items() if t < cutoff]
                for k in stale:
                    del _cache[k]
            return result
        return wrapper
    return deco
