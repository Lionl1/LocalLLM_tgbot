import asyncio
import os
import random
from urllib.parse import quote, urlparse

import httpx

from app.config import (
    IMAGE_GENERATION_ENDPOINT,
    IMAGE_GENERATION_HEIGHT,
    IMAGE_GENERATION_TIMEOUT,
    IMAGE_GENERATION_WIDTH,
)


class ImageGenerationError(Exception):
    """Ошибка при обращении к сервису генерации картинок."""


async def generate_image(prompt: str) -> bytes:
    if not prompt:
        raise ImageGenerationError("Пустой запрос.")

    headers = {
        "User-Agent": "Mozilla/5.0",
    }

    url = IMAGE_GENERATION_ENDPOINT
    method = "GET"
    json_payload = None
    params = None

    parsed = urlparse(IMAGE_GENERATION_ENDPOINT)

    # ===== Hugging Face (router / inference) =====
    if "huggingface.co" in parsed.netloc:
        token = os.getenv("HUGGING_FACE_TOKEN")
        if token:
            headers["Authorization"] = f"Bearer {token}"

        headers["Content-Type"] = "application/json"

        json_payload = {"inputs": prompt}
        method = "POST"

    # ===== Pollinations =====
    else:
        endpoint = IMAGE_GENERATION_ENDPOINT.rstrip("/")
        encoded_prompt = quote(prompt)
        url = f"{endpoint}/{encoded_prompt}"

        params = {"model": "flux", "nologo": "true"}

        if IMAGE_GENERATION_WIDTH and IMAGE_GENERATION_WIDTH > 0:
            params["width"] = IMAGE_GENERATION_WIDTH
        if IMAGE_GENERATION_HEIGHT and IMAGE_GENERATION_HEIGHT > 0:
            params["height"] = IMAGE_GENERATION_HEIGHT

    last_exc: Exception | None = None
    response: httpx.Response | None = None

    timeout = httpx.Timeout(
                    connect=10.0,
                    read=IMAGE_GENERATION_TIMEOUT,  # у тебя уже конфиг
                    write=20.0,
                    pool=30.0,
                )

    async with httpx.AsyncClient(
        timeout=timeout,
        follow_redirects=True,
        headers=headers,
    ) as client:

        # ⬅ увеличили число попыток для FLUX/router
        for attempt in range(5):
            try:
                if method == "POST":
                    response = await client.post(url, json=json_payload)
                else:
                    response = await client.get(url, params=params)

                # ===== SUCCESS =====
                if response.status_code == 200:
                    content_type = response.headers.get("content-type", "").lower()

                    # HF router иногда возвращает JSON (loading / error)
                    if "application/json" in content_type:
                        text_lower = response.text.lower()

                        # 🔥 модель грузится → ретрай
                        if "loading" in text_lower:
                            await asyncio.sleep(2 + attempt + random.random())
                            continue

                        raise ImageGenerationError(
                            f"Сервис вернул JSON вместо изображения: {response.text}"
                        )

                    return response.content

                # ===== retry-friendly статусы HF router =====
                if response.status_code in (500, 502, 503, 504, 529):
                    await asyncio.sleep(2 + attempt + random.random())
                    continue

                # ===== НЕ ретраим фатальные клиентские =====
                if response.status_code in (400, 401, 403, 404):
                    break

            except httpx.HTTPError as exc:
                last_exc = exc
                if attempt == 4:
                    raise ImageGenerationError(
                        f"Сбой подключения: {exc}"
                    ) from exc

                await asyncio.sleep(2 + attempt + random.random())

    # ===== ERROR =====
    if response is not None:
        error_text = response.text

        if response.status_code == 404 and "huggingface.co" in url:
            error_text += " (Модель не найдена. Убедись, что URL верный)"

        raise ImageGenerationError(
            f"Сервис вернул {response.status_code}: {error_text}"
        )

    raise ImageGenerationError(
        f"Неизвестная ошибка: {last_exc}"
    )