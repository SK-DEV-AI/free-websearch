"""Persistent cross-encoder reranker — keeps a Python 3.12 worker subprocess alive.

Features:
- Auto-restart on subprocess crash (broken pipe, segfault, etc.)
- Timeout protection on stdin writes and stdout reads
- Graceful fallback: returns passages unranked on any failure
"""

import asyncio
import json
import os
import logging

logger = logging.getLogger(__name__)

_PYTHON = os.path.expanduser("~/.local/share/venvs/reranker/bin/python3")
_WORKER = os.path.expanduser("~/.local/share/venvs/reranker/bin/reranker_worker.py")
_PROC = None
_LOCK = asyncio.Lock()
_START_ATTEMPTS = 0
_MAX_START_ATTEMPTS = 3


async def _start_worker():
    """Start a new reranker worker subprocess."""
    global _PROC, _START_ATTEMPTS
    if _START_ATTEMPTS >= _MAX_START_ATTEMPTS:
        logger.warning("reranker: max start attempts reached, skipping")
        return False
    try:
        _PROC = await asyncio.create_subprocess_exec(
            _PYTHON, _WORKER,
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        _START_ATTEMPTS = 0
        return True
    except Exception as e:
        logger.warning(f"reranker: failed to start worker: {e}")
        _START_ATTEMPTS += 1
        _PROC = None
        return False


async def _restart_worker():
    """Kill and restart the worker subprocess."""
    global _PROC
    if _PROC:
        try:
            _PROC.kill()
            await _PROC.wait()
        except Exception:
            pass
        _PROC = None
    return await _start_worker()


async def rerank(query: str, passages: list[dict], top_k: int = 20) -> list[dict]:
    """Rerank passages by relevance to query. Returns passages sorted by score.

    Falls back to returning passages unranked on any failure.
    """
    if not passages:
        return []

    async with _LOCK:
        global _PROC

        # Check if worker is alive, restart if needed
        if _PROC is not None and _PROC.returncode is not None:
            logger.info("reranker: worker died, restarting")
            _PROC = None
        if _PROC is None:
            if not await _start_worker():
                return passages[:top_k]

        normalized = []
        for p in passages:
            item = dict(p)
            if "snippet" not in item and "text" in item:
                item["snippet"] = item["text"][:2000]
            elif "snippet" not in item and "content" in item:
                item["snippet"] = item["content"][:2000]
            normalized.append(item)

        req = json.dumps({"query": query, "passages": normalized, "top_k": top_k})
        try:
            _PROC.stdin.write((req + "\n").encode())
            await asyncio.wait_for(_PROC.stdin.drain(), timeout=5)
        except (BrokenPipeError, OSError, asyncio.TimeoutError) as e:
            logger.warning(f"reranker: stdin write failed: {e}")
            _PROC = None
            # Try one restart
            if await _restart_worker():
                try:
                    _PROC.stdin.write((req + "\n").encode())
                    await asyncio.wait_for(_PROC.stdin.drain(), timeout=5)
                except Exception:
                    return passages[:top_k]
            else:
                return passages[:top_k]

        try:
            line = await asyncio.wait_for(_PROC.stdout.readline(), timeout=120)
        except asyncio.TimeoutError:
            logger.warning("reranker: stdout read timed out")
            await _restart_worker()
            return passages[:top_k]
        except (BrokenPipeError, OSError) as e:
            logger.warning(f"reranker: stdout read failed: {e}")
            _PROC = None
            return passages[:top_k]

        if not line:
            logger.warning("reranker: worker closed stdout")
            _PROC = None
            return passages[:top_k]

        try:
            result = json.loads(line)
        except json.JSONDecodeError as e:
            logger.warning(f"reranker: invalid JSON response: {e}")
            return passages[:top_k]

    if result.get("error"):
        logger.warning(f"reranker error: {result['error']}")
        return passages[:top_k]
    scored = result.get("scores", [])
    for s in scored:
        if "score" in s:
            s["_rerank"] = s.pop("score")
    return scored
