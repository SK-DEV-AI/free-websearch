from __future__ import annotations

import asyncio
import re
import urllib.parse
from typing import Any

from bs4 import BeautifulSoup
import defusedxml.ElementTree as DET
import httpx

from config import get_http_client
from fetch import fetch_url

SITEMAP_NS = "http://www.sitemaps.org/schemas/sitemap/0.9"
COMMON_SITEMAP_PATHS = ["/sitemap.xml", "/sitemap_index.xml", "/sitemap/",
                        "/sitemap-index.xml", "/sitemaps/sitemap.xml",
                        "/sitemap/sitemap.xml", "/sitemap-index.xml.gz",
                        "/sitemap.xml.gz", "/sitemaps/sitemap.xml.gz"]
XML_SITEMAP_PATHS = ["/sitemap.xml", "/sitemap_index.xml", "/sitemap/",
                     "/sitemap-index.xml", "/sitemaps/sitemap.xml",
                     "/sitemap/sitemap.xml"]


async def map_site(
    url: str,
    max_depth: int = 0,
    max_urls: int = 1000,
    timeout: int = 30,
    include_sitemap: bool = True,
    include_links: bool = True,
    same_domain: bool = True,
    exclude_patterns: list[str] | None = None,
) -> dict[str, Any]:
    parsed = urllib.parse.urlparse(url)
    base = f"{parsed.scheme}://{parsed.netloc}"
    exclude = re.compile("|".join(exclude_patterns)) if exclude_patterns else None
    seen: set[str] = set()
    results: list[dict[str, Any]] = []
    source = "none"
    c = get_http_client()

    # ── Phase 1: Sitemap discovery ────────────────────────────────────
    if include_sitemap:
        sitemap_urls: list[str] = []
        try:
            r = await c.get(f"{base}/robots.txt", timeout=10, follow_redirects=True)
            if r.status_code == 200:
                for line in r.text.splitlines():
                    line = line.strip()
                    if line.lower().startswith("sitemap:"):
                        sitemap_urls.append(line.split(":", 1)[1].strip())
        except Exception:
            pass
        if not sitemap_urls:
            for path in XML_SITEMAP_PATHS:
                try:
                    r = await c.get(f"{base}{path}", timeout=8, follow_redirects=True)
                    if r.status_code == 200 and "xml" in (r.headers.get("content-type", "") or "xml"):
                        sitemap_urls.append(str(r.url))
                        if len(sitemap_urls) >= 3:
                            break
                except Exception:
                    pass
        for su in sitemap_urls:
            await _parse_sitemap_recursive(su, c, seen, results, exclude, base if same_domain else "",
                                           max_urls)
        if results:
            source = "sitemap"

    # ── Phase 2: HTML link extraction ─────────────────────────────────
    if include_links and len(results) < max_urls:
        html = await _fetch_page_html(url, c)
        if html:
            soup = BeautifulSoup(html, "html.parser")
            links: set[str] = set()
            for a in soup.find_all("a", href=True):
                href = a["href"].strip()
                if not href or href.startswith("#") or href.startswith("javascript:"):
                    continue
                absolute = urllib.parse.urljoin(url, href)
                parsed_link = urllib.parse.urlparse(absolute)
                if not parsed_link.scheme.startswith("http"):
                    continue
                normalised = urllib.parse.urlunparse((
                    parsed_link.scheme, parsed_link.netloc,
                    parsed_link.path.rstrip("/") or "/",
                    "", "", "",
                ))
                if exclude and exclude.search(normalised):
                    continue
                if normalised not in seen:
                    seen.add(normalised)
                    links.add(normalised)

            for link in sorted(links):
                if link.startswith(base):
                    results.append({"url": link, "source": "link"})
                    if len(results) >= max_urls:
                        break
            if results and source == "none":
                source = "links"

    return {
        "success": True,
        "url": url,
        "base_domain": base,
        "source": source,
        "total": len(results),
        "urls": results[:max_urls],
    }


async def _parse_sitemap_recursive(
    sitemap_url: str,
    c: httpx.AsyncClient,
    seen: set[str],
    results: list[dict[str, Any]],
    exclude: re.Pattern | None,
    domain_filter: str,
    max_urls: int,
) -> None:
    try:
        r = await c.get(sitemap_url, timeout=15, follow_redirects=True)
        if r.status_code != 200:
            return
        root = DET.fromstring(r.content)
    except Exception:
        return

    tag = root.tag
    if tag.endswith("sitemapindex"):
        for sm in root:
            ns = sm.tag
            loc_tag = sm.find(f"{{{SITEMAP_NS}}}loc")
            if loc_tag is not None and loc_tag.text:
                await _parse_sitemap_recursive(loc_tag.text.strip(), c, seen,
                                              results, exclude, domain_filter, max_urls)
    elif tag.endswith("urlset"):
        for u in root:
            loc = u.find(f"{{{SITEMAP_NS}}}loc")
            if loc is None or not loc.text:
                continue
            url = loc.text.strip()
            if url in seen:
                continue
            seen.add(url)
            if exclude and exclude.search(url):
                continue
            if domain_filter and not url.startswith(domain_filter):
                continue
            entry: dict[str, Any] = {"url": url, "source": "sitemap"}
            lastmod = u.find(f"{{{SITEMAP_NS}}}lastmod")
            if lastmod is not None and lastmod.text:
                entry["last_modified"] = lastmod.text.strip()
            priority = u.find(f"{{{SITEMAP_NS}}}priority")
            if priority is not None and priority.text:
                entry["priority"] = float(priority.text.strip())
            changefreq = u.find(f"{{{SITEMAP_NS}}}changefreq")
            if changefreq is not None and changefreq.text:
                entry["changefreq"] = changefreq.text.strip()
            results.append(entry)
            if len(results) >= max_urls:
                break


async def _fetch_page_html(url: str, c: httpx.AsyncClient) -> str | None:
    try:
        r = await c.get(url, timeout=10, follow_redirects=True)
        if r.status_code == 200 and "text/html" in (r.headers.get("content-type", "") or ""):
            return r.text
    except Exception:
        pass
    return None
