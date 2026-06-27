"""Unified web MCP server — multi-engine search + stealth scraping + content extraction."""

from __future__ import annotations

import asyncio
import json
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import CallToolResult, TextContent, Tool

from config import MAX_RESULTS, HELIUM_CDP
from search_ddg import search_ddg, ddgs_extract
from search_exa import exa_similar, exa_search
from search_gai import GoogleAIClient, get_gai_client
from fetch import fetch_url, scrapling_stealthy_fetch
from crawl import crawl_url
from pdf_extract import extract_pdf
from screenshot import cdpa11y_snapshot, screenshot_cdp
from wikipedia import (search_wikipedia, fetch_wikipedia_summary, fetch_wikipedia_summary_rest,
                       fetch_wikipedia_categories, fetch_wikipedia_links, fetch_wikipedia_extlinks,
                       fetch_wikipedia_pageviews, fetch_wikipedia_revisions,
                       search_wikipedia_category, search_wikipedia_backlinks,
                       search_wikipedia_geosearch, search_wikipedia_random,
                       search_wikipedia_recentchanges, fetch_wikipedia_langlinks,
                       search_wikipedia_allpages)
from arxiv import search_arxiv
from search_tavily import tavily_extract
from search_firecrawl import map_firecrawl, firecrawl_scrape, firecrawl_research
from research import search_multi, enrich

server = Server("free-websearch")

@server.list_tools()
async def handle_list_tools() -> list[Tool]:
    return [
        Tool(name="search",
            description="Multi-engine web search with dedup and reranking. depth=1 returns snippets; depth>=2 also fetches full page content and re-ranks. Use upload_urls for image/PDF upload to GAI.",
            inputSchema={"type": "object", "properties": {
                "query": {"type": "string"}, "count": {"type": "integer", "default": 10},
                "depth": {"type": "integer", "default": 1, "description": "1=snippets, 2+=fetch full pages + rerank"},
                "search_type": {"type": "string", "enum": ["auto","text","news","images","videos","books"]},
                "timelimit": {"type": "string", "description": "Time filter: d/w/m/y"},
                "safesearch": {"type": "string", "enum": ["off","moderate","strict"], "default": "moderate"},
                "language": {"type": "string", "default": "en", "description": "Content language for Wikipedia/summaries"},
                "country": {"type": "string", "description": "Geo-targeting country code (Firecrawl, search results)"},
                "upload_urls": {"type": "array", "items": {"type": "string"}, "description": "Image/PDF URLs or local file paths for GAI"},
                "extract_links": {"type": "boolean", "description": "Discover related links via exa_similar before fetching (depth>=2 only)"},
                "start_date": {"type": "string", "description": "Tavily date filter start (YYYY-MM-DD)"},
                "end_date": {"type": "string", "description": "Tavily date filter end (YYYY-MM-DD)"}},
                "required": ["query"]}),
        Tool(name="fetch",
            description="URL to markdown/text. Use stealth=True for Cloudflare sites. Supports PDF, EPUB, DOCX.",
            inputSchema={"type": "object", "properties": {
                "url": {"type": "string"}, "max_chars": {"type": "integer", "default": 5000},
                "css_selector": {"type": "string"},
                "extraction_type": {"type": "string", "enum": ["markdown","text","html"]},
                "stealth": {"type": "boolean"},
                "target_language": {"type": "string"},
                "output_format": {"type": "string", "enum": ["markdown","txt","json","xml","csv"], "default": "markdown"},
                "fast": {"type": "boolean"}, "prune_xpath": {"type": "string", "description": "XPath prune selectors"},
                "proxy": {"type": "string"},
                "wait_selector": {"type": "string"},
                "init_script": {"type": "string", "description": "JS inject on load"},
                "extra_headers": {"type": "object", "description": "Extra HTTP headers"},
                },
                "required": ["url"]}),
        Tool(name="crawl",
            description="BFS/DFS deep crawl. Returns per-page markdown + combined text.",
            inputSchema={"type": "object", "properties": {
                "url": {"type": "string"}, "max_depth": {"type": "integer", "default": 1},
                "max_pages": {"type": "integer", "default": 10},
                "extract_links": {"type": "boolean", "default": True},
                "strategy": {"type": "string", "enum": ["bfs","dfs"], "default": "bfs"},
                "css_selector": {"type": "string"}, "exclude_domains": {"type": "array", "items": {"type": "string"}},
                "js_code": {"type": "string"}, "wait_for": {"type": "string"},
                "proxy": {"type": "string"},
                "headers": {"type": "object"},
                "content_filter": {"type": "string", "enum": ["","pruning","bm25","bm25_hq","cosine"]},
                "filter_query": {"type": "string"},
                "css_extract": {"type": "object", "description": "JSON CSS extraction schema with 'name' and 'selector' keys.", "properties": {"name": {"type": "string"}, "selector": {"type": "string"}}}},
                "required": ["url"]}),
        Tool(name="screenshot",
            description="CDP screenshot or ARIA accessibility snapshot.",
            inputSchema={"type": "object", "properties": {
                "url": {"type": "string"}, "full_page": {"type": "boolean", "default": True},
                "type": {"type": "string", "enum": ["screenshot","snapshot","both"]},
                "max_chars": {"type": "integer", "default": 10000},
                "depth": {"type": "integer"},
                "scale": {"type": "string", "enum": ["css","device"]},
                "quality": {"type": "integer", "description": "JPEG quality 1-100"}},
                "required": ["url"]}),
        Tool(name="wikipedia",
            description="Search Wikipedia: articles, summaries, geosearch, random. Actions: search, summary (REST API v1 fast), summary_action (Action API with images/sections), categories, links, extlinks, categorymembers, pageviews, revisions, backlinks, recentchanges.",
            inputSchema={"type": "object", "properties": {
                "action": {"type": "string", "enum": ["search","summary","summary_action","geosearch","random","categories","links","extlinks","categorymembers","pageviews","revisions","backlinks","recentchanges","langlinks","allpages"], "default": "search"},
                "query": {"type": "string"},
                "count": {"type": "integer", "default": 3},
                "language": {"type": "string", "default": "en"},
                "lat": {"type": "number"}, "lon": {"type": "number"},
                "category": {"type": "string", "description": "Category name for categorymembers action (without Category: prefix)"},
                "type_filter": {"type": "string", "description": "Change type filter for recentchanges: edit|new|log"},
                "namespace": {"type": "integer", "default": 0, "description": "Namespace filter (links, random, search, allpages)"}},
                "required": []}),
        Tool(name="arxiv",
            description="Search arXiv academic papers. Use raw_query for boolean operators (AND, OR, ANDNOT), phrase search (ti:\"exact phrase\"), wildcards (au:smith*).",
            inputSchema={"type": "object", "properties": {
                "query": {"type": "string"}, "count": {"type": "integer", "default": 3},
                "search_field": {"type": "string", "enum": ["all","ti","au","abs","cat","co","jr","id"], "default": "all"},
                "sort_by": {"type": "string", "enum": ["relevance","lastUpdatedDate","submittedDate"], "default": "relevance"},
                "sort_order": {"type": "string", "enum": ["ascending","descending"], "default": "descending"},
                "start": {"type": "integer", "default": 0},
                "id_list": {"type": "string", "description": "Comma-delimited arXiv IDs"},
                "category": {"type": "string", "description": "arXiv category filter (e.g. cs.AI, math.CO)"},
                "raw_query": {"type": "string", "description": "Raw arXiv search_query syntax with boolean operators (AND/OR/ANDNOT), phrase, wildcards. Overrides query+search_field."}},
                "required": ["query"]}),
        Tool(name="firecrawl_map",
            description="Firecrawl Map — discover URL structure of a site. Returns {url, title, description}[].",
            inputSchema={"type": "object", "properties": {
                "url": {"type": "string"},
                "search": {"type": "string", "description": "Filter links by relevance to this query"},
                "limit": {"type": "integer", "default": 100},
                "include_subdomains": {"type": "boolean", "default": True}},
                "required": ["url"]}),
        Tool(name="exa_similar",
            description="Find pages similar to a given URL via Exa.",
            inputSchema={"type": "object", "properties": {
                "url": {"type": "string"},
                "count": {"type": "integer", "default": 5},
                "highlights": {"type": "boolean"},
                "summary": {"type": "boolean"},
                "include_domains": {"type": "array", "items": {"type": "string"}},
                "exclude_domains": {"type": "array", "items": {"type": "string"}},
                "category": {"type": "string", "description": "Category filter: company, research paper, news, tweet, movie, song, personal site, pdf"},
                "start_published_date": {"type": "string", "description": "Filter results published after this date (YYYY-MM-DD)"},
                "end_published_date": {"type": "string", "description": "Filter results published before this date (YYYY-MM-DD)"}},
                "required": ["url"]}),
         Tool(name="ddgs_extract",
            description="Lightweight URL content extraction via DuckDuckGo's extract endpoint. Faster than fetch for simple pages — markdown or plain text. Best for search snippets and quick page reads where trafilatura is overkill.",
            inputSchema={"type": "object", "properties": {
                "url": {"type": "string"}, "extract_type": {"type": "string", "enum": ["markdown","text_plain","raw"], "default": "markdown"}},
                "required": ["url"]}),
        Tool(name="pdf_extract",
            description="PDF to structured data via opendataloader-pdf (#1 benchmark, 0.907 accuracy). Extracts text, tables, formulas, images with bounding boxes. Supports scanned PDFs (OCR), complex tables, and accessibility tagging.",
            inputSchema={"type": "object", "properties": {
                "input_path": {"type": "array", "items": {"type": "string"}, "description": "PDF file paths or URLs (local files, http/https, file://)"},
                "format": {"type": "string", "enum": ["markdown","json","html","tagged-pdf","markdown,json","markdown,json,html"], "default": "markdown"},
                "password": {"type": "string", "description": "PDF password for protected files"},
                "pages": {"type": "string", "description": "Page range e.g. 1-5,8,10-12"},
                "hybrid": {"type": "string", "enum": ["", "docling-fast", "docling-enterprise", "marker"], "description": "AI hybrid mode for complex layouts, scanned PDFs, tables"},
                "hybrid_mode": {"type": "string", "enum": ["", "full"], "description": "full enables formula/picture enrichment (requires hybrid)"},
                "force_ocr": {"type": "boolean", "description": "Enable OCR for scanned/image-based PDFs (requires --hybrid docling-fast)"},
                "enrich_formula": {"type": "boolean", "description": "Extract mathematical formulas as LaTeX (requires hybrid_mode=full)"},
                "enrich_picture": {"type": "boolean", "description": "Generate AI descriptions for charts/images (requires hybrid_mode=full)"},
                "table_method": {"type": "string", "enum": ["", "fast", "accurate"], "description": "Table extraction method"},
                "reading_order": {"type": "string", "enum": ["", "natural", "xy-cut"], "description": "Reading order algorithm"},
                "image_output": {"type": "string", "enum": ["", "placeholders", "embedded"], "description": "Image handling in output"}},
                "required": ["input_path"]}),
        Tool(name="exa_search",
            description="Exa search — find content across the web by semantic query. Supports domain filtering and date ranges.",
            inputSchema={"type": "object", "properties": {
                "query": {"type": "string"},
                "num_results": {"type": "integer", "default": 10},
                "search_type": {"type": "string", "enum": ["auto","keyword","neural","magic"], "default": "auto"},
                "include_domains": {"type": "array", "items": {"type": "string"}},
                "exclude_domains": {"type": "array", "items": {"type": "string"}},
                "start_published_date": {"type": "string", "description": "Filter results published after this date (YYYY-MM-DD)"},
                "end_published_date": {"type": "string", "description": "Filter results published before this date (YYYY-MM-DD)"},
                "highlights": {"type": "boolean"},
                "summary": {"type": "boolean"}},
                "required": ["query"]}),
        Tool(name="tavily_extract",
            description="Tavily Extract — pull content from specific URLs via Tavily API. Returns raw content and metadata.",
            inputSchema={"type": "object", "properties": {
                "urls": {"type": "array", "items": {"type": "string"}, "description": "URLs to extract (max 10)"},
                "include_images": {"type": "boolean", "default": False}},
                "required": ["urls"]}),
        Tool(name="firecrawl_scrape",
            description="Firecrawl Scrape — extract content from a single URL with fine-grained control over extraction.",
            inputSchema={"type": "object", "properties": {
                "url": {"type": "string"},
                "formats": {"type": "array", "items": {"type": "string"}, "default": ["markdown"], "description": "Output formats: markdown, html, etc."},
                "includeTags": {"type": "array", "items": {"type": "string"}},
                "excludeTags": {"type": "array", "items": {"type": "string"}},
                "waitFor": {"type": "string", "description": "CSS selector to wait for before extraction"},
                "actions": {"type": "array", "items": {"type": "object"}, "description": "Browser actions to perform before extraction"},
                "timeout": {"type": "integer", "default": 30000}},
                "required": ["url"]}),
        Tool(name="firecrawl_research",
            description="Firecrawl Research — search academic papers, read passages, find related papers. Uses Firecrawl's research index for AI/ML papers.",
            inputSchema={"type": "object", "properties": {
                "query": {"type": "string", "description": "Natural-language query to search papers"},
                "action": {"type": "string", "enum": ["search","detail","similar","github"], "default": "search",
                    "description": "search=find papers, detail=read paper metadata/passages, similar=find related papers, github=search GitHub"},
                "paper_id": {"type": "string", "description": "Paper ID for detail/similar actions (paperId or primaryId)"},
                "authors": {"type": "string", "description": "Author substring filter (search only)"},
                "categories": {"type": "string", "description": "Paper category filter e.g. cs.LG (search only)"},
                "from_date": {"type": "string", "description": "Inclusive lower bound YYYY-MM-DD (search only)"},
                "to_date": {"type": "string", "description": "Inclusive upper bound YYYY-MM-DD (search only)"},
                "limit": {"type": "integer", "default": 10}},
                "required": ["query"]}),
    ]


@server.call_tool()
async def handle_call_tool(name: str, arguments: dict) -> CallToolResult:
    if not isinstance(arguments, dict):
        return CallToolResult(content=[TextContent(type="text", text=json.dumps({"error": "arguments must be a dict"}))], isError=True)

    def safe_int(v, default=0):
        try:
            return int(v)
        except (TypeError, ValueError):
            return default

    def safe_float(v, default=0.0):
        try:
            return float(v)
        except (TypeError, ValueError):
            return default

    def _res(data) -> CallToolResult:
        ok = isinstance(data, dict) and data.get("success", False)
        return CallToolResult(content=[TextContent(type="text", text=json.dumps(data, default=str))], isError=not ok)

    try:
        if name == "search":
            query = arguments["query"]
            count = min(safe_int(arguments.get("count",10)), MAX_RESULTS)
            depth = safe_int(arguments.get("depth",1))
            lang = str(arguments.get("language","en"))
            r = await search_multi(query, count=max(count, depth * 3),
                google_ai_only=bool(arguments.get("google_ai_only",False)),
                search_type=str(arguments.get("search_type","auto")),
                search_prompt=str(arguments.get("search_prompt","")),
                pro_mode=bool(arguments.get("pro_mode",False)),
                gl=str(arguments.get("gl","")), hl=str(arguments.get("hl","en")),
                tbs=str(arguments.get("tbs","")), pws=str(arguments.get("pws","")),
                backend=str(arguments.get("backend","auto")),
                timelimit=str(arguments.get("timelimit","")),
                page=safe_int(arguments.get("page",1)),
                region=str(arguments.get("region","wt-wt")),
                safesearch=str(arguments.get("safesearch","moderate")),
                language=lang,
                country=str(arguments.get("country","")),
                upload_urls=arguments.get("upload_urls"),
                query_expand=bool(arguments.get("query_expand",True)),
                tavily_topic=str(arguments.get("tavily_topic","general")),
                tavily_depth=str(arguments.get("tavily_depth","basic")),
                size=str(arguments.get("size","")),
                color=str(arguments.get("color","")),
                type_image=str(arguments.get("type_image","")),
                layout=str(arguments.get("layout","")),
                license_image=str(arguments.get("license_image","")),
                resolution=str(arguments.get("resolution","")),
                duration=str(arguments.get("duration","")),
                license_videos=str(arguments.get("license_videos","")),
                start_date=str(arguments.get("start_date","")),
                end_date=str(arguments.get("end_date","")),
                exact_phrase=bool(arguments.get("exact_phrase",False)),
                cdp_url=HELIUM_CDP)
            if r.get("success") and depth >= 2 and r.get("results"):
                fetched = await enrich(r["results"], query, depth=depth,
                    extract_links=bool(arguments.get("extract_links",False)),
                    cdp_url=HELIUM_CDP, count=count, language=lang)
                if fetched.get("fetched_content"):
                    r["fetched_content"] = fetched["fetched_content"]
            return _res(r)
        elif name == "fetch":
            if bool(arguments.get("stealth",False)):
                r = await scrapling_stealthy_fetch(arguments["url"],
                    css_selector=arguments.get("css_selector"),
                    extraction_type=str(arguments.get("extraction_type","markdown")),
                    cdp_url=HELIUM_CDP, block_webrtc=bool(arguments.get("block_webrtc",False)),
                    hide_canvas=True, disable_resources=bool(arguments.get("disable_resources",True)),
                    google_search=True, proxy=str(arguments.get("proxy","")),
                    locale=str(arguments.get("locale","")), timezone_id=str(arguments.get("timezone_id","")),
                    network_idle=bool(arguments.get("network_idle",True)),
                    allow_webgl=bool(arguments.get("allow_webgl",True)),
                    block_ads=bool(arguments.get("block_ads",True)),
                    dns_over_https=bool(arguments.get("dns_over_https",True)),
                    solve_cloudflare=bool(arguments.get("solve_cloudflare",True)),
                    retries=safe_int(arguments.get("retries",5)),
                    capture_xhr=str(arguments.get("capture_xhr","")),
                    wait_selector=str(arguments.get("wait_selector","")),
                    wait_selector_state=str(arguments.get("wait_selector_state","attached")),
                    blocked_domains=arguments.get("blocked_domains"),
                    init_script=str(arguments.get("init_script","")),
                    extra_headers=arguments.get("extra_headers"),
                    useragent=str(arguments.get("useragent","")),
                    load_dom=bool(arguments.get("load_dom",False)))
            else:
                r = await fetch_url(arguments["url"],
                    max_chars=safe_int(arguments.get("max_chars",5000)),
                    main_content_only=bool(arguments.get("main_content_only",True)),
                    target_language=str(arguments.get("target_language","")),
                    favor_precision=bool(arguments.get("favor_precision",False)),
                    favor_recall=bool(arguments.get("favor_recall",False)),
                    fast=bool(arguments.get("fast",False)),
                    deduplicate=bool(arguments.get("deduplicate",True)),
                    output_format=str(arguments.get("output_format","markdown")),
                    include_images=bool(arguments.get("include_images",True)),
                    include_comments=bool(arguments.get("include_comments",True)),
                    include_formatting=bool(arguments.get("include_formatting",True)),
                    include_links=bool(arguments.get("include_links",True)),
                    prune_xpath=str(arguments.get("prune_xpath","")),
                    url_blacklist=str(arguments.get("url_blacklist","")),
                    author_blacklist=str(arguments.get("author_blacklist","")),
                    cdp_url=HELIUM_CDP or "",
                    min_output_size=safe_int(arguments.get("min_output_size", 0)))
            return _res(r)
        elif name == "crawl":
            r = await crawl_url(arguments["url"], max_depth=safe_int(arguments.get("max_depth",1)),
                max_pages=safe_int(arguments.get("max_pages",10)),
                extract_links=bool(arguments.get("extract_links",True)),
                js_code=str(arguments.get("js_code","")), wait_for=str(arguments.get("wait_for","")),
                session_id=str(arguments.get("session_id","")), locale=str(arguments.get("locale","")),
                timezone_id=str(arguments.get("timezone_id","")),
                check_robots_txt=bool(arguments.get("check_robots_txt",False)),
                strategy=str(arguments.get("strategy","bfs")),
                screenshot=bool(arguments.get("screenshot",False)),
                pdf=bool(arguments.get("pdf",False)), proxy=str(arguments.get("proxy","")),
                magic=bool(arguments.get("magic",True)),
                flatten_shadow_dom=bool(arguments.get("flatten_shadow_dom",True)),
                process_iframes=bool(arguments.get("process_iframes",True)),
                scan_full_page=bool(arguments.get("scan_full_page",True)),
                wait_for_images=bool(arguments.get("wait_for_images",True)),
                word_count_threshold=safe_int(arguments.get("word_count_threshold",30)),
                page_timeout=safe_int(arguments.get("page_timeout",60000)),
                max_retries=safe_int(arguments.get("max_retries",2)),
                browser_type=str(arguments.get("browser_type","chromium")),
                user_agent=str(arguments.get("user_agent","")),
                headers=arguments.get("headers"),
                extra_args=arguments.get("extra_args"),
                geolocation_lat=safe_float(arguments.get("geolocation_lat",0)),
                geolocation_lng=safe_float(arguments.get("geolocation_lng",0)),
                delay_before_return_html=safe_float(arguments.get("delay_before_return_html",0.1)),
                stream=bool(arguments.get("stream",False)),
                capture_mhtml=bool(arguments.get("capture_mhtml",False)),
                capture_network_requests=bool(arguments.get("capture_network_requests",False)),
                capture_console_messages=bool(arguments.get("capture_console_messages",False)),
                exclude_external_links=bool(arguments.get("exclude_external_links",False)),
                exclude_social_media_links=bool(arguments.get("exclude_social_media_links",False)),
                exclude_domains=arguments.get("exclude_domains"),
                css_selector=str(arguments.get("css_selector","")),
                target_elements=arguments.get("target_elements"),
                memory_saving_mode=bool(arguments.get("memory_saving_mode",False)),
                remove_forms=bool(arguments.get("remove_forms",False)),
                keep_data_attributes=bool(arguments.get("keep_data_attributes",False)),
                method=str(arguments.get("method","GET")),
                score_links=bool(arguments.get("score_links",False)),
                simulate_user=bool(arguments.get("simulate_user",True)),
                override_navigator=bool(arguments.get("override_navigator",True)),
                cdp_url=str(arguments.get("cdp_url", HELIUM_CDP)),
                adjust_viewport_to_content=bool(arguments.get("adjust_viewport_to_content",False)),
                log_console=bool(arguments.get("log_console",False)),
                content_filter=str(arguments.get("content_filter","")),
                filter_query=str(arguments.get("filter_query","")),
                css_extract=arguments.get("css_extract"))
            return _res(r)
        elif name == "screenshot":
            url = arguments["url"]
            cap_type = str(arguments.get("type","screenshot"))
            full = bool(arguments.get("full_page",True))
            snap = None
            ss = None
            if cap_type in ("snapshot","both"):
                snap = await cdpa11y_snapshot(url, verbose=bool(arguments.get("verbose",False)),
                    max_chars=safe_int(arguments.get("max_chars",10000)),
                    depth=arguments.get("depth"),
                    boxes=bool(arguments.get("boxes",False)))
            if cap_type in ("screenshot","both"):
                ss = await screenshot_cdp(url, full_page=full,
                    clip_x=safe_float(arguments.get("clip_x",0)),
                    clip_y=safe_float(arguments.get("clip_y",0)),
                    clip_width=safe_float(arguments.get("clip_width",0)),
                    clip_height=safe_float(arguments.get("clip_height",0)),
                    scale=str(arguments.get("scale","css")),
                    animations=str(arguments.get("animations","allow")),
                    quality=arguments.get("quality"),
                    image_type=str(arguments.get("image_type","png")))
            if cap_type == "snapshot":
                return _res(snap)
            if cap_type == "both":
                return _res({"screenshot": ss, "snapshot": snap})
            return _res(ss)
        elif name == "wikipedia":
            action = str(arguments.get("action", "search"))
            lang = str(arguments.get("language", "en"))
            q = str(arguments.get("query", ""))
            cnt = safe_int(arguments.get("count", 3))
            if action == "summary":
                r = await fetch_wikipedia_summary_rest(title=q, language=lang)
                return _res({"success": True, "result": r} if r else {"success": False, "error": "Not found"})
            if action == "summary_action":
                r = await fetch_wikipedia_summary(query=q, language=lang,
                    include_images=bool(arguments.get("include_images",False)))
                return _res({"success": True, "result": r} if r else {"success": False, "error": "Not found"})
            if action == "categories":
                r = await fetch_wikipedia_categories(title=q, language=lang)
                return _res({"success": True, "results": r})
            if action == "links":
                r = await fetch_wikipedia_links(title=q, language=lang,
                    namespace=safe_int(arguments.get("namespace", 0)), count=cnt)
                return _res({"success": True, "results": r})
            if action == "extlinks":
                r = await fetch_wikipedia_extlinks(title=q, language=lang, count=cnt)
                return _res({"success": True, "results": r})
            if action == "categorymembers":
                r = await search_wikipedia_category(category=str(arguments.get("category", q)),
                    language=lang, count=cnt)
                return _res({"success": True, "results": r})
            if action == "pageviews":
                r = await fetch_wikipedia_pageviews(title=q, language=lang,
                    days=safe_int(arguments.get("days", 30)))
                return _res({"success": True, "results": r})
            if action == "revisions":
                r = await fetch_wikipedia_revisions(title=q, language=lang, count=cnt)
                return _res({"success": True, "results": r})
            if action == "backlinks":
                r = await search_wikipedia_backlinks(title=q, language=lang, count=cnt)
                return _res({"success": True, "results": r})
            if action == "recentchanges":
                r = await search_wikipedia_recentchanges(language=lang, count=cnt,
                    type_filter=str(arguments.get("type_filter","")))
                return _res({"success": True, "results": r})
            if action == "geosearch":
                r = await search_wikipedia_geosearch(
                    lat=float(arguments.get("lat",0)), lon=float(arguments.get("lon",0)),
                    distance=safe_int(arguments.get("distance",1000)),
                    count=safe_int(arguments.get("count",10)), language=str(arguments.get("language","en")))
                return _res({"success": True, "results": r})
            if action == "random":
                r = await search_wikipedia_random(count=safe_int(arguments.get("count",5)),
                    language=str(arguments.get("language","en")))
                return _res({"success": True, "results": r})
            if action == "langlinks":
                r = await fetch_wikipedia_langlinks(title=q, language=lang, count=cnt)
                return _res({"success": True, "results": r})
            if action == "allpages":
                r = await search_wikipedia_allpages(
                    namespace=safe_int(arguments.get("namespace", 0)),
                    limit=safe_int(arguments.get("count", 50)),
                    language=lang)
                return _res({"success": True, "results": r})
            r = await search_wikipedia(query=q,
                count=cnt, language=lang,
                namespace=safe_int(arguments.get("namespace", 0)))
            return _res({"success": True, "results": r})
        elif name == "arxiv":
            r = await search_arxiv(query=str(arguments.get("query","")),
                count=safe_int(arguments.get("count",3)),
                search_field=str(arguments.get("search_field","all")),
                sort_by=str(arguments.get("sort_by","relevance")),
                sort_order=str(arguments.get("sort_order","descending")),
                start=safe_int(arguments.get("start",0)),
                id_list=str(arguments.get("id_list","")),
                category=str(arguments.get("category","")),
                raw_query=str(arguments.get("raw_query","")))
            return _res({"success": True, "results": r})
        elif name == "firecrawl_map":
            r = await map_firecrawl(url=str(arguments.get("url","")),
                search=str(arguments.get("search","")),
                limit=safe_int(arguments.get("limit",100)),
                include_subdomains=bool(arguments.get("include_subdomains",True)))
            return _res({"success": True, "links": r})
        elif name == "exa_similar":
            r = await exa_similar(url=str(arguments.get("url","")),
                count=safe_int(arguments.get("count",5)),
                highlights=bool(arguments.get("highlights",False)),
                summary=bool(arguments.get("summary",False)),
                subpages=safe_int(arguments.get("subpages",0)),
                include_domains=arguments.get("include_domains"),
                exclude_domains=arguments.get("exclude_domains"),
                category=str(arguments.get("category","")),
                system_prompt=str(arguments.get("system_prompt","")),
                output_schema=arguments.get("output_schema"),
                stream=bool(arguments.get("stream",False)),
                user_location=str(arguments.get("user_location","")),
                start_published_date=str(arguments.get("start_published_date","")),
                end_published_date=str(arguments.get("end_published_date","")))
            return _res({"success": True, "results": r})
        elif name == "ddgs_extract":
            r = await ddgs_extract(url=str(arguments.get("url", "")),
                extract_type=str(arguments.get("extract_type", "markdown")))
            return _res({"success": True, "result": r})
        elif name == "pdf_extract":
            paths = arguments.get("input_path", [])
            if isinstance(paths, str):
                paths = [paths]
            hy = str(arguments.get("hybrid", ""))
            hm = str(arguments.get("hybrid_mode", ""))
            if bool(arguments.get("force_ocr", False)):
                hy = hy or "docling-fast"
                hm = hm or "full"
            if bool(arguments.get("enrich_formula", False)):
                hy = hy or "docling-fast"
                hm = "full"
            if bool(arguments.get("enrich_picture", False)):
                hy = hy or "docling-fast"
                hm = "full"
            r = await extract_pdf(paths,
                format=str(arguments.get("format", "markdown")),
                password=str(arguments.get("password", "")),
                pages=str(arguments.get("pages", "")),
                hybrid=hy, hybrid_mode=hm,
                hybrid_url=str(arguments.get("hybrid_url", "")),
                hybrid_timeout=str(arguments.get("hybrid_timeout", "")),
                table_method=str(arguments.get("table_method", "")),
                reading_order=str(arguments.get("reading_order", "")),
                image_output=str(arguments.get("image_output", "")),
                image_format=str(arguments.get("image_format", "")),
                sanitize=bool(arguments.get("sanitize", False)),
                keep_line_breaks=bool(arguments.get("keep_line_breaks", False)),
                markdown_with_html=bool(arguments.get("markdown_with_html", False)))
            return _res(r)
        elif name == "exa_search":
            r = await exa_search(query=str(arguments.get("query","")),
                num_results=safe_int(arguments.get("num_results",10)),
                search_type=str(arguments.get("search_type","auto")),
                include_domains=arguments.get("include_domains"),
                exclude_domains=arguments.get("exclude_domains"),
                start_published_date=str(arguments.get("start_published_date","")),
                end_published_date=str(arguments.get("end_published_date","")),
                highlights=bool(arguments.get("highlights",False)),
                summary=bool(arguments.get("summary",False)))
            return _res({"success": True, "results": r})
        elif name == "tavily_extract":
            urls = arguments.get("urls", [])
            if isinstance(urls, str):
                urls = [urls]
            r = await tavily_extract(urls=urls,
                include_images=bool(arguments.get("include_images",False)))
            return _res({"success": True, "results": r})
        elif name == "firecrawl_scrape":
            r = await firecrawl_scrape(url=str(arguments.get("url","")),
                formats=arguments.get("formats"),
                only_main_content=bool(arguments.get("onlyMainContent",True)),
                include_tags=arguments.get("includeTags"),
                exclude_tags=arguments.get("excludeTags"),
                wait_for=str(arguments.get("waitFor","")),
                actions=arguments.get("actions"),
                timeout=safe_int(arguments.get("timeout",30000)))
            return _res(r)
        elif name == "firecrawl_research":
            r = await firecrawl_research(
                query=str(arguments.get("query","")),
                action=str(arguments.get("action","search")),
                paper_id=str(arguments.get("paper_id","")),
                authors=str(arguments.get("authors","")),
                categories=str(arguments.get("categories","")),
                from_date=str(arguments.get("from_date","")),
                to_date=str(arguments.get("to_date","")),
                limit=safe_int(arguments.get("limit", 10)))
            return _res({"success": True, "results": r} if r else {"success": False, "error": "No results"})
        else:
            return CallToolResult(content=[TextContent(type="text", text=f"Unknown tool: {name}")], isError=True)
    except ValueError as e:
        return CallToolResult(content=[TextContent(type="text", text=json.dumps({"error": str(e)}))], isError=True)
    except KeyError as e:
        return CallToolResult(content=[TextContent(type="text", text=json.dumps({"error": f"Missing required argument: {e}"}))], isError=True)
    except TypeError as e:
        return CallToolResult(content=[TextContent(type="text", text=json.dumps({"error": str(e)}))], isError=True)
    except RuntimeError as e:
        return CallToolResult(content=[TextContent(type="text", text=json.dumps({"error": str(e)}))], isError=True)
    except Exception as e:
        return CallToolResult(content=[TextContent(type="text", text=json.dumps({"error": f"{type(e).__name__}: {e}"}))], isError=True)


async def _warmup_reranker():
    """Pre-load reranker model in background to avoid cold-start delay."""
    try:
        from reranker import warmup
        await warmup()
    except Exception:
        pass

async def _warmup_gai():
    """Pre-warm GAI CDP connection at server start."""
    try:
        from search_gai import get_gai_client
        await get_gai_client()
    except Exception:
        pass

async def main():
    asyncio.create_task(_warmup_reranker())
    asyncio.create_task(_warmup_gai())
    async with stdio_server() as (rs, ws):
        await server.run(rs, ws, server.create_initialization_options())

if __name__ == "__main__":
    asyncio.run(main())
