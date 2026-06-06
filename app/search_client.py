import httpx
import asyncio
from ddgs import DDGS

from app.config import (
    WEB_SEARCH_API_KEY,
    WEB_SEARCH_MAX_RESULTS,
    WEB_SEARCH_PROVIDER,
    WEB_SEARCH_TIMEOUT,
)


class WebSearchError(RuntimeError):
    pass


def _ddg_search_sync(query, limit):
    try:
        with DDGS() as ddgs:
            raw_results = list(ddgs.text(query, max_results=limit))
        results = []
        for item in raw_results:
            results.append({
                "title": item.get("title", ""),
                "url": item.get("href", ""),
                "snippet": item.get("body", "")
            })
        return results
    except Exception as exc:
        raise WebSearchError(f"DuckDuckGo search failed: {exc}") from exc


async def _search_duckduckgo(query, limit):
    return await asyncio.to_thread(_ddg_search_sync, query, limit)


async def _search_serper(query, limit):
    if not WEB_SEARCH_API_KEY:
        raise WebSearchError("WEB_SEARCH_API_KEY is required for serper.")
    url = "https://google.serper.dev/search"
    headers = {"X-API-KEY": WEB_SEARCH_API_KEY}
    payload = {"q": query, "num": limit}
    timeout = httpx.Timeout(WEB_SEARCH_TIMEOUT)
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(url, json=payload, headers=headers)
    if response.is_error:
        detail = response.text
        if detail and len(detail) > 1000:
            detail = f"{detail[:1000]}..."
        raise WebSearchError(f"Serper error {response.status_code}: {detail}")
    data = response.json()
    results = []
    for item in data.get("organic", []):
        title = (item.get("title") or "").strip()
        url = (item.get("link") or "").strip()
        snippet = (item.get("snippet") or "").strip()
        if not title and not url and not snippet:
            continue
        results.append({"title": title, "url": url, "snippet": snippet})
        if len(results) >= limit:
            break
    return results


async def search_web(query, limit=None):
    if not query:
        return []
    provider = (WEB_SEARCH_PROVIDER or "duckduckgo").strip().casefold()
    limit = limit or WEB_SEARCH_MAX_RESULTS
    if limit < 1:
        return []
    if provider in {"duckduckgo", "ddg"}:
        return await _search_duckduckgo(query, limit)
    if provider in {"serper", "google"}:
        return await _search_serper(query, limit)
    raise WebSearchError(f"Unknown WEB_SEARCH_PROVIDER: {WEB_SEARCH_PROVIDER}")
