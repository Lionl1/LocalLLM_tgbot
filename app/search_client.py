import httpx

from app.config import (
    WEB_SEARCH_API_KEY,
    WEB_SEARCH_MAX_RESULTS,
    WEB_SEARCH_PROVIDER,
    WEB_SEARCH_TIMEOUT,
)


class WebSearchError(RuntimeError):
    pass


def _split_title_snippet(text):
    if not text:
        return "", ""
    if " - " in text:
        title, snippet = text.split(" - ", 1)
        return title.strip(), snippet.strip()
    return text.strip(), ""


def _collect_ddg_results(data, limit):
    results = []
    abstract_text = (data.get("AbstractText") or "").strip()
    if abstract_text:
        title = (data.get("Heading") or "").strip() or "Кратко"
        url = (data.get("AbstractURL") or "").strip()
        results.append({"title": title, "url": url, "snippet": abstract_text})
    answer = (data.get("Answer") or "").strip()
    if answer:
        results.append({"title": "Ответ", "url": "", "snippet": answer})
    definition = (data.get("Definition") or "").strip()
    if definition:
        url = (data.get("DefinitionURL") or "").strip()
        results.append({"title": "Определение", "url": url, "snippet": definition})
    for item in data.get("Results", []):
        text = item.get("Text", "").strip()
        url = item.get("FirstURL", "").strip()
        if not text and not url:
            continue
        title, snippet = _split_title_snippet(text)
        results.append({"title": title, "url": url, "snippet": snippet})
        if len(results) >= limit:
            return results
    for item in data.get("RelatedTopics", []):
        if "Topics" in item:
            for sub in item.get("Topics", []):
                text = sub.get("Text", "").strip()
                url = sub.get("FirstURL", "").strip()
                if not text and not url:
                    continue
                title, snippet = _split_title_snippet(text)
                results.append({"title": title, "url": url, "snippet": snippet})
                if len(results) >= limit:
                    return results
            continue
        text = item.get("Text", "").strip()
        url = item.get("FirstURL", "").strip()
        if not text and not url:
            continue
        title, snippet = _split_title_snippet(text)
        results.append({"title": title, "url": url, "snippet": snippet})
        if len(results) >= limit:
            return results
    return results


async def _search_duckduckgo(query, limit):
    url = "https://api.duckduckgo.com/"
    params = {
        "q": query,
        "format": "json",
        "no_redirect": "1",
        "no_html": "1",
        "skip_disambig": "1",
    }
    timeout = httpx.Timeout(WEB_SEARCH_TIMEOUT)
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.get(url, params=params)
    if response.status_code not in {200, 202}:
        detail = response.text
        if detail and len(detail) > 1000:
            detail = f"{detail[:1000]}..."
        raise WebSearchError(
            f"DuckDuckGo error {response.status_code}: {detail}"
        )
    data = response.json()
    return _collect_ddg_results(data, limit)


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
