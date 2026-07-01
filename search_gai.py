from __future__ import annotations

import asyncio
import base64
import json
import os
import re
import time
import urllib.parse
from typing import Any

import httpx

from config import GOOGLE_AI_URL, HELIUM_CDP, get_http_client

# ── Multi-language constants ──────────────────────────────────────

CITATION_SELECTORS = [
    '[aria-label="View related links"]',
    '[aria-label*="Related links"]',
    '[aria-label*="Sources"]',
    '[aria-label="Zugehörige Links anzeigen"]',
    '[aria-label*="Zugehörige Links"]',
    '[aria-label*="Quellen"]',
    '[aria-label*="Liens associés"]',
    '[aria-label*="Sources"]',
    '[aria-label*="Enlaces relacionados"]',
    '[aria-label*="Fuentes"]',
    '[aria-label*="Gerelateerde links"]',
    '[aria-label*="Bronnen"]',
    '[aria-label*="Link correlati"]',
    '[aria-label*="Fonti"]',
    'button[aria-label*="links" i]',
]

AI_COMPLETION_TEXT_INDICATORS = [
    "AI-generated", "AI Overview", "Generative AI is experimental",
    "KI-Antworten", "KI-generiert", "Generative KI",
    "AI-gegenereerd", "AI-overzicht",
    "Las respuestas de la IA", "Resumen de IA",
    "Réponses IA", "Aperçu de l'IA",
    "Risposte IA", "Panoramica IA",
]

CUTOFF_MARKERS = [
    "AI-generated answers may contain mistakes", "AI can make mistakes",
    "Generative AI is experimental", "AI overviews are experimental",
    "KI-Antworten können Fehler enthalten", "KI-Antworten können Fehler",
    "AI-reacties kunnen fouten bevatten",
    "Las respuestas de la IA pueden contener errores",
    "Les réponses de l'IA peuvent contenir des erreurs",
    "Le risposte dell'IA possono contenere errori",
]

MIME_MAP = {
    '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.png': 'image/png',
    '.gif': 'image/gif', '.webp': 'image/webp', '.pdf': 'application/pdf',
    '.svg': 'image/svg+xml', '.bmp': 'image/bmp', '.avif': 'image/avif',
    '.mp4': 'video/mp4', '.mov': 'video/mp4', '.mp3': 'audio/mpeg',
    '.wav': 'audio/wav', '.ogg': 'audio/ogg',
    '.txt': 'text/plain', '.md': 'text/markdown', '.csv': 'text/csv',
    '.json': 'application/json', '.xml': 'text/xml', '.html': 'text/html',
}

# ── CDP browser singleton ─────────────────────────────────────────

_cdp_pw = None
_cdp_browser = None
_cdp_lock = asyncio.Lock()

ANTI_DETECT_JS = """
Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
Object.defineProperty(navigator, 'languages', {get: () => ['en-US', 'en']});
Object.defineProperty(navigator, 'plugins', {get: () => [1,2,3,4,5]});
window.chrome = {runtime: {}, csi: function(){}, loadTimes: function(){}};
Object.defineProperty(navigator, 'permissions', {
    get: () => ({
        query: (params) => Promise.resolve({state: 'granted', onchange: null})
    })
});
Object.defineProperty(navigator, 'maxTouchPoints', {get: () => 0});
delete navigator.__proto__.webdriver;
"""

CDP_BLOCK_PATTERNS = [
    "*.png", "*.jpg", "*.jpeg", "*.gif", "*.svg", "*.webp", "*.ico",
    "*.woff", "*.woff2", "*.ttf", "*.eot", "*.otf",
    "*.mp4", "*.webm", "*.ogg", "*.mp3", "*.wav",
    "*/ads/*", "*/analytics/*", "*/tracking/*",
]

async def _get_cdp_browser():
    global _cdp_pw, _cdp_browser
    from playwright.async_api import async_playwright
    async with _cdp_lock:
        if _cdp_pw and _cdp_browser:
            try:
                ctx = _cdp_browser.contexts
                if ctx:
                    pgs = ctx[0].pages
                    if pgs:
                        await asyncio.wait_for(pgs[0].title(), timeout=5)
                return _cdp_browser
            except Exception:
                pass
            try:
                await _cdp_pw.stop()
            except Exception:
                pass
        _cdp_pw = await async_playwright().start()
        _cdp_browser = await _cdp_pw.chromium.connect_over_cdp(HELIUM_CDP)
        return _cdp_browser

async def _create_hidden_page(ctx, url="about:blank"):
    """Create a page not visible in the tab UI strip, without stealing focus."""
    try:
        cdp_session = await ctx.new_cdp_session(ctx.pages[0] if ctx.pages else await ctx.new_page())
        result = await cdp_session.send("Target.createTarget",
                                        {"url": url, "hidden": True, "focus": False})
        target_id = result.get("targetId")
        for _ in range(10):
            await asyncio.sleep(0.05)
            for p in ctx.pages:
                try:
                    if p.url == url:
                        return p
                except Exception:
                    pass
        if target_id:
            try:
                await cdp_session.send("Target.closeTarget", {"targetId": target_id})
            except Exception:
                pass
    except Exception:
        pass
    return await ctx.new_page()


_page_semaphore = asyncio.Semaphore(5)

async def _get_optimized_page(block_resources: bool = True):
    async with _page_semaphore:
        b = await _get_cdp_browser()
        ctx = b.contexts[0]
        page = await _create_hidden_page(ctx)
        await page.add_init_script(ANTI_DETECT_JS)
        if block_resources:
            try:
                cdp = await ctx.new_cdp_session(page)
                await cdp.send("Network.enable")
                await cdp.send("Network.setBlockedURLs", {"urls": CDP_BLOCK_PATTERNS})
                await cdp.detach()
            except Exception:
                pass
        return page


async def _cleanup_orphan_tabs():
    """Close hidden about:blank pages that are not the active GoogleAIClient page."""
    global _cdp_browser
    if _cdp_browser is None:
        return
    try:
        ctx = _cdp_browser.contexts[0]
    except Exception:
        return
    gaiclient = _gaiclient
    active_page = gaiclient._page if gaiclient else None
    for p in list(ctx.pages):
        try:
            if p.url == "about:blank" and p is not active_page:
                await p.close()
        except Exception:
            pass


async def shutdown():
    """Close CDP browser, playwright, and shared httpx client on server shutdown."""
    global _cdp_pw, _cdp_browser, _gaiclient
    if _gaiclient:
        try:
            await _gaiclient.close()
        except Exception:
            pass
        _gaiclient = None
    if _cdp_browser:
        try:
            await _cdp_browser.close()
        except Exception:
            pass
        _cdp_browser = None
    if _cdp_pw:
        try:
            await _cdp_pw.stop()
        except Exception:
            pass
        _cdp_pw = None
    try:
        from config import close_http_client
        await close_http_client()
    except Exception:
        pass

# ── GoogleAIClient ─────────────────────────────────────────────────

CAPTCHA_INDICATORS = [
    "/sorry/index", "unusual traffic", "captcha", "recaptcha",
    "are you a robot", "verify you are human", "automated queries",
]

class GoogleAIClient:
    def __init__(self, cdp_url: str | None = None):
        self._cdp_url = cdp_url
        self._available: bool | None = None
        self._available_at: float = 0
        self._page = None
        self._page_target_id = None
        self._mutex = asyncio.Lock()

    async def _get_shared_browser(self):
        if not self._cdp_url:
            raise RuntimeError("GoogleAIClient requires a CDP URL")
        return await _get_cdp_browser()

    async def _get_cdp_session(self):
        b = await self._get_shared_browser()
        ctx = b.contexts[0]
        existing = [p for p in ctx.pages if p.url != "about:blank"]
        target = existing[0] if existing else ctx.pages[0] if ctx.pages else await ctx.new_page()
        return await ctx.new_cdp_session(target)

    async def _get_page(self):
        b = await self._get_shared_browser()
        ctx = b.contexts[0]
        if self._page:
            try:
                _ = self._page.url
                return self._page
            except Exception:
                self._page = None
                self._page_target_id = None
        try:
            cdp = await self._get_cdp_session()
            existing = set(ctx.pages) if ctx else set()
            result = await cdp.send("Target.createTarget",
                                    {"url": "about:blank", "hidden": True, "focus": False})
            tid = result["targetId"]
            for _ in range(10):
                await asyncio.sleep(0.05)
                for p in (ctx.pages if ctx else []):
                    if p not in existing and p.url == "about:blank":
                        self._page = p
                        self._page_target_id = tid
                        return p
            if tid:
                try:
                    await cdp.send("Target.closeTarget", {"targetId": tid})
                except Exception:
                    pass
        except Exception:
            pass
        self._page = await _create_hidden_page(ctx)
        self._page_target_id = None
        return self._page

    async def _close_page(self):
        if self._page:
            try:
                await self._page.close()
            except Exception:
                pass
            self._page = None
            self._page_target_id = None

    async def _detect_captcha(self, p) -> bool:
        try:
            if any(i in p.url.lower() for i in CAPTCHA_INDICATORS):
                return True
            body = (await p.evaluate("document.body.innerText")).lower()
            if any(i in body for i in CAPTCHA_INDICATORS):
                return True
            if len(body) < 600 and "search" not in body:
                return True
        except Exception:
            pass
        return False

    async def check_available(self) -> bool:
        async with self._mutex:
            if self._available is not None and time.monotonic() - self._available_at < 600:
                return self._available
            try:
                b = await self._get_shared_browser()
                p = await self._get_page()
                try:
                    await p.goto(f"{GOOGLE_AI_URL}?q=test&udm=50&hl=en",
                                 wait_until="commit", timeout=15000, referer="https://www.google.com/")
                    await p.wait_for_load_state("domcontentloaded", timeout=10000)
                    aimc = await p.query_selector("[data-subtree=aimc]")
                    if not aimc:
                        body = (await p.evaluate("document.body.innerText")).lower()
                        blocked = [
                            "not available in your country", "not available in your region",
                            "not available in your language", "ai mode is not available",
                            "ai mode isn't available", "der ki-modus ist in ihrem land",
                            "le mode ia n'est pas disponible",
                        ]
                        self._available = not any(b in body for b in blocked)
                        self._available_at = time.monotonic()
                        return self._available
                    deadline = time.monotonic() + 25
                    while time.monotonic() < deadline:
                        ok = await p.evaluate("""() => {
                            const t = document.querySelectorAll('[data-subtree=aimc]');
                            const l = t[t.length-1]; return l && l.innerText.trim().length > 50
                                && !l.innerText.includes('Searching'); }""")
                        if ok:
                            break
                        await asyncio.sleep(0.5)
                    self._available = True
                    self._available_at = time.monotonic()
                    return True
                finally:
                    await self._close_page()
                    asyncio.ensure_future(_cleanup_orphan_tabs())
            except Exception:
                self._available = False
                self._available_at = time.monotonic()
                return False

    # ── Wait for AI completion (4-stage, ported from google-ai-mode-mcp) ──

    async def _wait_for_completion(self, p, deadline: float) -> bool:
        """Poll all indicators simultaneously. SVG + aimc existence = done. Text backup."""
        while time.monotonic() < deadline:
            try:
                svg = await p.query_selector('button svg[viewBox="3 3 18 18"]')
                if svg:
                    has_aimc = await p.evaluate("!!document.querySelector('[data-subtree=aimc]')")
                    if has_aimc:
                        return True
            except Exception:
                pass
            try:
                body_text = await p.evaluate("document.body.innerText")
                if any(indicator in body_text for indicator in AI_COMPLETION_TEXT_INDICATORS):
                    has_aimc = await p.evaluate("!!document.querySelector('[data-subtree=aimc]')")
                    if has_aimc:
                        return True
            except Exception:
                pass
            await asyncio.sleep(0.5)
        return True

    def _html_to_markdown(self, html: str) -> str:
        try:
            import markdownify
            md = markdownify.markdownify(
                html,
                heading_style="ATX",
                bullets="-",
                strip=["script", "style", "noscript"],
            )
            md = re.sub(r'==+([^=]+)==+', r'\1', md)
            md = re.sub(r'!\[[^\]]*\]\(data:image\/[^)]+\)', '', md)
            md = re.sub(r'\[\]\([^)]*\)', '', md)
            for marker in CUTOFF_MARKERS:
                idx = md.find(marker)
                if idx >= 0:
                    md = md[:idx].strip()
                    break
            md = re.sub(r'(?<![.!?\n])\n(?![*\-#\d\n])', ' ', md)
            md = re.sub(r'\n{3,}', '\n\n', md)
            return md.strip()
        except Exception:
            return html

    # ── Search ──────────────────────────────────────────────────────────

    async def _upload_files(self, p, upload_urls: list[str] | None) -> bool:
        """Upload files via browser fetch + DragEvent drop.
        Supports: remote URLs (browser fetch), local files (disk read + data URL).
        Returns True if at least one file was uploaded successfully."""
        if not upload_urls:
            return False
        for _ in range(20):
            if await p.evaluate("!!document.querySelector('textarea')"):
                break
            await asyncio.sleep(0.5)
        uploaded = False
        for url in upload_urls:
            try:
                path = url.replace('file://', '')
                is_local = os.path.isfile(path)
                if is_local:
                    # Local file: server-side read → base64 data URL → browser
                    with open(path, 'rb') as f:
                        content = f.read()
                    if len(content) > 10_000_000:
                        continue
                    b64 = base64.b64encode(content).decode('ascii')
                    ext = os.path.splitext(path.lower())[1]
                    name = os.path.basename(path)
                    mime = MIME_MAP.get(ext, 'application/octet-stream')
                    ok = await p.evaluate("""async ({b64, mime, name}) => {
                        try {
                            const r = await fetch(`data:${mime};base64,${b64}`);
                            const blob = await r.blob();
                            const f = new File([blob], name, {type: mime});
                            const dt = new DataTransfer(); dt.items.add(f);
                            const ta = document.querySelector('textarea');
                            if (!ta) return false;
                            ta.dispatchEvent(new DragEvent('dragenter',
                                {bubbles: true, cancelable: true, dataTransfer: dt}));
                            ta.dispatchEvent(new DragEvent('dragover',
                                {bubbles: true, cancelable: true, dataTransfer: dt}));
                            ta.dispatchEvent(new DragEvent('drop',
                                {bubbles: true, cancelable: true, dataTransfer: dt}));
                            await new Promise(r => setTimeout(r, 2000));
                            return {ok: true, size: blob.size, type: mime};
                        } catch(e) { return {ok: false, error: e.message}; }
                    }""", {"b64": b64, "mime": mime, "name": name})
                    if isinstance(ok, dict) and ok.get("ok"):
                        uploaded = True
                        await asyncio.sleep(1)
                    continue
                # Remote URL: try browser fetch() first
                ok = await p.evaluate("""async (url) => {
                    try {
                        const resp = await fetch(url);
                        if (!resp.ok) return {error: `HTTP ${resp.status}`};
                        const blob = await resp.blob();
                        const name = url.split('/').pop()?.split('?')[0] || 'upload';
                        const file = new File([blob], name, {type: blob.type});
                        const dt = new DataTransfer(); dt.items.add(file);
                        const ta = document.querySelector('textarea');
                        if (!ta) return {error: 'no textarea'};
                        ta.dispatchEvent(new DragEvent('dragenter',
                            {bubbles: true, cancelable: true, dataTransfer: dt}));
                        ta.dispatchEvent(new DragEvent('dragover',
                            {bubbles: true, cancelable: true, dataTransfer: dt}));
                        ta.dispatchEvent(new DragEvent('drop',
                            {bubbles: true, cancelable: true, dataTransfer: dt}));
                        await new Promise(r => setTimeout(r, 2000));
                        return {ok: true, size: blob.size, type: blob.type};
                    } catch(e) { return {error: e.message}; }
                }""", url)
                if isinstance(ok, dict) and ok.get("ok"):
                    uploaded = True
                    await asyncio.sleep(1)
                    continue
                # Fallback: server-side download → base64 data URL
                c = get_http_client()
                resp = await c.get(url, headers={
                    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"},
                    follow_redirects=True)
                if resp.status_code == 200 and len(resp.content) <= 10_000_000:
                    b64 = base64.b64encode(resp.content).decode('ascii')
                    ext = os.path.splitext(urllib.parse.urlparse(url).path)[1].lower() or '.bin'
                    mime = MIME_MAP.get(ext, resp.headers.get('content-type', 'application/octet-stream'))
                    ok2 = await p.evaluate("""async ({b64, mime, name}) => {
                        try {
                            const resp = await fetch(`data:${mime};base64,${b64}`);
                            const blob = await resp.blob();
                            const file = new File([blob], name, {type: mime});
                            const dt = new DataTransfer(); dt.items.add(file);
                            const ta = document.querySelector('textarea');
                            if (!ta) return false;
                            ta.dispatchEvent(new DragEvent('dragenter',
                                {bubbles: true, cancelable: true, dataTransfer: dt}));
                            ta.dispatchEvent(new DragEvent('dragover',
                                {bubbles: true, cancelable: true, dataTransfer: dt}));
                            ta.dispatchEvent(new DragEvent('drop',
                                {bubbles: true, cancelable: true, dataTransfer: dt}));
                            await new Promise(r => setTimeout(r, 2000));
                            return true;
                        } catch(e) { return false; }
                    }""", {"b64": b64, "mime": mime, "name": f"upload{ext}"})
                    if ok2:
                        uploaded = True
                        await asyncio.sleep(1)
            except Exception:
                pass
        return uploaded

    async def search(self, query: str, search_prompt: str = "", pro_mode: bool = False,
                     gl: str = "", hl: str = "en", tbs: str = "", pws: str = "",
                     upload_urls: list[str] | None = None) -> dict:
        async with self._mutex:
            b = await self._get_shared_browser()
            ctx = b.contexts[0]
            p = await _create_hidden_page(ctx)
            await p.add_init_script(ANTI_DETECT_JS)
            try:
                final_query = " ".join(filter(None, [query, search_prompt]))

                if upload_urls:
                    bare_url = f"{GOOGLE_AI_URL}?udm=50"
                    for k, v in [("hl", hl), ("gl", gl), ("tbs", tbs), ("pws", pws)]:
                        if v:
                            bare_url += f"&{k}={v}"
                    await p.goto(bare_url, wait_until="commit", timeout=20000,
                                 referer="https://www.google.com/")
                    await p.wait_for_load_state("domcontentloaded", timeout=10000)
                    if await self._detect_captcha(p):
                        return {"success": False, "error": "CAPTCHA detected"}
                    uploaded = await self._upload_files(p, upload_urls)
                    if uploaded:
                        await p.evaluate(f"""() => {{
                            const ta = document.querySelector('textarea');
                            if (ta) {{
                                ta.focus();
                                ta.value = {json.dumps(final_query)};
                                ta.dispatchEvent(new Event('input', {{bubbles: true}}));
                            }}
                        }}""")
                        await asyncio.sleep(0.5)
                        await p.evaluate("""() => {
                            const btn = document.querySelector('[aria-label="Send"]');
                            if (btn) { btn.disabled = false; btn.click(); }
                        }""")
                        await asyncio.sleep(0.3)
                else:
                    params = [f"q={urllib.parse.quote_plus(final_query)}", "udm=50"]
                    for k, v in [("hl", hl), ("gl", gl), ("tbs", tbs), ("pws", pws)]:
                        if v:
                            params.append(f"{k}={v}")
                    url = f"{GOOGLE_AI_URL}?{'&'.join(params)}"
                    await p.goto(url, wait_until="commit", timeout=20000,
                                 referer="https://www.google.com/")
                    await p.wait_for_load_state("domcontentloaded", timeout=10000)
                    if await self._detect_captcha(p):
                        return {"success": False, "error": "CAPTCHA detected"}

                # ── Completion detection ──
                deadline = time.monotonic() + (300 if upload_urls else 240)
                await self._wait_for_completion(p, deadline)

                remaining = deadline - time.monotonic()
                if remaining > 0:
                    try:
                        await asyncio.wait_for(
                            p.evaluate("""() => {
                                for (const btn of document.querySelectorAll('[aria-expanded="false"]')) {
                                    const t = btn.innerText.toLowerCase();
                                    if (t.includes('show more') || t.includes('mehr anzeigen')
                                        || t.includes('meer weergeven')) btn.click();
                                }
                            }"""),
                            timeout=3,
                        )
                        await asyncio.sleep(0.5)
                    except Exception:
                        pass

                # ── SERPO: batch click all citation buttons ──
                try:
                    await p.evaluate("""(selectors) => {
                        function isVisible(el) {
                            if (!el) return false;
                            try {
                                const style = window.getComputedStyle(el);
                                const rect = el.getBoundingClientRect();
                                return style.display!=='none' && style.visibility!=='hidden'
                                    && style.opacity!=='0' && el.offsetParent!==null
                                    && rect.width > 0 && rect.height > 0;
                            } catch(e) { return false; }
                        }
                        const turns = document.querySelectorAll('[data-subtree=aimc]');
                        const lastTurn = turns[turns.length - 1];
                        if (!lastTurn) return;
                        const mainCol = lastTurn.closest('[data-container-id="main-col"]');
                        const container = mainCol || lastTurn;
                        let buttons = [];
                        for (const sel of selectors) {
                            buttons = Array.from(container.querySelectorAll(sel));
                            if (buttons.filter(b => isVisible(b)).length > 0) break;
                        }
                        buttons = buttons.filter(b => isVisible(b));
                        for (let i = 0; i < buttons.length; i++) {
                            const marker = document.createElement('span');
                            marker.className = 'citation-marker';
                            marker.innerHTML = '<code>[CITE-' + i + ']</code>';
                            const ref = buttons[i].nextSibling ? buttons[i] : buttons[i];
                            ref.parentNode.insertBefore(marker, ref.nextSibling);
                            try { buttons[i].click(); } catch(e) {}
                        }
                    }""", CITATION_SELECTORS)
                    await asyncio.sleep(0.5)
                except Exception:
                    pass

                serpo_result = await p.evaluate("""() => {
                    const turns = document.querySelectorAll('[data-subtree=aimc]');
                    const lastTurn = turns[turns.length - 1];
                    const mainCol = lastTurn ? lastTurn.closest('[data-container-id="main-col"]') : null;
                    const container = mainCol || lastTurn;
                    const ansHtml = (() => {
                        if (!lastTurn) return '';
                        const c = lastTurn.cloneNode(true);
                        for (const e of c.querySelectorAll('[role=button], button, style, script')) e.remove();
                        for (const m of c.querySelectorAll('.citation-marker')) m.remove();
                        return c.innerHTML;
                    })();
                    const seen = new Set();
                    const srcs = [];
                    for (const a of document.querySelectorAll('a[href]')) {
                        let h = a.href;
                        try {
                            const u = new URL(h);
                            if (u.hostname.includes('google.com') && u.pathname === '/url') {
                                const t = u.searchParams.get('q') || u.searchParams.get('url');
                                if (t) h = t;
                            }
                        } catch(e) {}
                        if (!h.startsWith('http') || seen.has(h)) continue;
                        if (['google.com','gstatic.com'].some(d => h.includes(d))) continue;
                        seen.add(h);
                        let t = a.innerText.trim().split('\\n')[0];
                        if (!t || t.length < 4) { try { t = new URL(h).hostname.replace('www.',''); } catch(e) {} }
                        let s = '';
                        const card = a.closest('li, [class]');
                        if (card) {
                            const ls = card.innerText.trim().split('\\n').filter(l => l.length > 30);
                            for (const lx of ls) { if (lx !== t && lx.length > 30) { s = lx.substring(0, 200); break; } }
                        }
                        srcs.push({ title: t.substring(0, 200), url: h.split('#')[0], snippet: s });
                    }
                    let fu = '';
                    if (lastTurn) {
                        const blocks = lastTurn.innerText.trim().split('\\n').map(l => l.trim()).filter(l => l.length > 15);
                        if (blocks.length > 0) { const lb = blocks[blocks.length - 1];
                            if (lb.includes('?') && lb.length < 150) fu = lb; }
                    }
                    return { html: (container ? container.innerHTML : ''), answerHtml: ansHtml, sources: srcs.slice(0, 20), followUp: fu };
                }""")

                await p.close()
                serpo_result = serpo_result or {}

                # ── Offline: HTML → markdown → sequential CITE footnotes ──
                answer_html = serpo_result.get("answerHtml", "")
                if not answer_html:
                    return {"success": False, "error": "Could not extract AI response"}

                md = self._html_to_markdown(answer_html) if answer_html else ""

                # Replace [CITE-N] with sequential footnote references
                cite_count = 0
                cite_nums = {}
                def replace_cite(m):
                    nonlocal cite_count
                    cite_count += 1
                    cite_nums[cite_count] = m.group(0)
                    return f"[{cite_count}]"
                if md:
                    md = re.sub(r'\[CITE-\d+\]', replace_cite, md)

                citation_sources = serpo_result.get("sources", [])
                if citation_sources:
                    md += "\n\n---\n\n## Sources\n\n"
                    for i, src in enumerate(citation_sources):
                        t = src.get("title", src.get("source", "") or src.get("url", ""))
                        u = src.get("url", "")
                        md += f"[{i+1}] {t}  \n{u}\n\n"

                return {
                    "success": True,
                    "result": {
                        "answer": md or "No content extracted",
                        "sources": citation_sources,
                        "followUp": serpo_result.get("followUp", ""),
                    },
                }
            finally:
                try:
                    await p.close()
                except Exception:
                    pass
                asyncio.ensure_future(_cleanup_orphan_tabs())

    async def close(self):
        try:
            if self._page:
                await self._page.close()
        except Exception:
            pass
        self._page = None
        self._page_target_id = None

# ── Singleton accessor ─────────────────────────────────────────────

_gaiclient: GoogleAIClient | None = None
_gai_lock = asyncio.Lock()

async def get_gai_client(cdp_url: str | None = None) -> GoogleAIClient | None:
    global _gaiclient
    if _gaiclient is not None:
        return _gaiclient
    async with _gai_lock:
        if _gaiclient is not None:
            return _gaiclient
        url = cdp_url or HELIUM_CDP
        if not url:
            return None
        c = GoogleAIClient(cdp_url=url)
        if await c.check_available():
            _gaiclient = c
            return _gaiclient
        await c.close()
    return None
