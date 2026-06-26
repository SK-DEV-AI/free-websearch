"""Shared reranker bridge — one subprocess via Unix socket, both MCP servers connect to it."""

import asyncio
import json
import logging
import os

logger = logging.getLogger(__name__)

_SOCKET_PATH = "/tmp/reranker_worker.sock"
_RERANKER_PYTHON = "/usr/bin/python3"
_RERANKER_WORKER = os.path.expanduser("~/.local/share/reranker-rust/worker.py")

_PROC = None
_LOCK = asyncio.Lock()
_READER = None
_WRITER = None


async def _ensure_worker():
    global _PROC, _READER, _WRITER
    if _PROC is not None and _PROC.returncode is None:
        return True
    try:
        _READER, _WRITER = await asyncio.open_unix_connection(_SOCKET_PATH)
        return True
    except (FileNotFoundError, ConnectionRefusedError, OSError):
        pass
    try:
        _PROC = await asyncio.create_subprocess_exec(
            _RERANKER_PYTHON, _RERANKER_WORKER,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        for _ in range(50):
            try:
                _READER, _WRITER = await asyncio.open_unix_connection(_SOCKET_PATH)
                return True
            except (FileNotFoundError, ConnectionRefusedError):
                await asyncio.sleep(0.1)
            except OSError:
                await asyncio.sleep(0.1)
        return False
    except Exception as e:
        logger.warning(f"reranker: start failed: {e}")
        return False


async def warmup() -> bool:
    """Pre-load the reranker model. Returns True if warmup succeeded."""
    async with _LOCK:
        global _READER, _WRITER, _PROC
        if not await _ensure_worker():
            return False
        try:
            req = json.dumps({"query": "warmup", "passages": [{"snippet": "warmup"}], "top_k": 1})
            _WRITER.write((req + "\n").encode())
            await asyncio.wait_for(_WRITER.drain(), timeout=5)
            r = await asyncio.wait_for(_READER.readuntil(b"\n"), timeout=120)
            result = json.loads(r)
            return not result.get("error")
        except Exception:
            return False


async def rerank(query: str, passages: list[dict], top_k: int = 20) -> list[dict]:
    if not passages:
        return []
    async with _LOCK:
        global _READER, _WRITER, _PROC
        if not await _ensure_worker():
            return passages[:top_k]
        normalized = []
        for p in passages:
            item = dict(p)
            text = item.get("snippet") or item.get("text") or item.get("content") or ""
            item["snippet"] = text[:8000]
            normalized.append(item)
        req = json.dumps({"query": query, "passages": normalized, "top_k": top_k})
        try:
            _WRITER.write((req + "\n").encode())
            await asyncio.wait_for(_WRITER.drain(), timeout=5)
        except (BrokenPipeError, OSError, asyncio.TimeoutError) as e:
            logger.warning(f"reranker: write failed: {e}")
            _WRITER = None
            _PROC = None
            return passages[:top_k]
        try:
            r = await asyncio.wait_for(_READER.readuntil(b"\n"), timeout=120)
        except (asyncio.IncompleteReadError, ConnectionResetError, asyncio.TimeoutError) as e:
            logger.warning(f"reranker: read failed: {e}")
            _WRITER = None
            _PROC = None
            return passages[:top_k]
        try:
            result = json.loads(r)
        except json.JSONDecodeError:
            return passages[:top_k]
    if result.get("error"):
        logger.warning(f"reranker error: {result['error']}")
        return passages[:top_k]
    scored = result.get("scores", [])
    for s in scored:
        if "score" in s:
            s["_rerank"] = s.pop("score")
    return scored
