import logging

import httpx

from config import (
    MAX_TOKENS,
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
    OPENAI_MODEL,
    REQUEST_TIMEOUT,
    TEMPERATURE,
)


logger = logging.getLogger(__name__)


async def chat_completion(messages, model=None, max_tokens=None, temperature=None):
    url = f"{OPENAI_BASE_URL.rstrip('/')}/chat/completions"
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
    payload = {
        "model": model or OPENAI_MODEL,
        "messages": messages,
        "max_tokens": max_tokens if max_tokens is not None else MAX_TOKENS,
        "temperature": temperature if temperature is not None else TEMPERATURE,
        "stream": False,
    }
    timeout = httpx.Timeout(REQUEST_TIMEOUT)
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(url, json=payload, headers=headers)
        if response.is_error:
            detail = response.text
            if detail and len(detail) > 1000:
                detail = f"{detail[:1000]}..."
            logger.error("LLM error %s: %s", response.status_code, detail)
            raise RuntimeError(f"LLM error {response.status_code}: {detail}")
        data = response.json()
    return data["choices"][0]["message"]["content"]
