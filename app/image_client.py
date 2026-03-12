import asyncio
import logging
import random
from urllib.parse import quote

import httpx

from app.config import (
    IMAGE_GENERATION_ENDPOINT,
    IMAGE_GENERATION_HEIGHT,
    IMAGE_GENERATION_TIMEOUT,
    IMAGE_GENERATION_WIDTH,
    POLLINATIONS_API_KEY,
)

logger = logging.getLogger(__name__)


class ImageGenerationError(Exception):
    """Ошибка при обращении к сервису генерации картинок."""


async def generate_image(prompt: str) -> bytes:
    if not prompt:
        raise ImageGenerationError("Пустой запрос.")

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Referer": "https://pollinations.ai/",
    }

    endpoint = IMAGE_GENERATION_ENDPOINT.rstrip("/")
    
    if POLLINATIONS_API_KEY:
        headers["Authorization"] = f"Bearer {POLLINATIONS_API_KEY}"
        # Если есть ключ, но адрес публичный -> меняем на enter
        if "image.pollinations.ai" in endpoint:
            endpoint = endpoint.replace("image.pollinations.ai", "enter.pollinations.ai")
            logger.info("Using authenticated endpoint: %s", endpoint)

    # Строим URL запроса
    url = f"{endpoint}/{quote(prompt)}"

    # Параметры генерации
    params = {
        "model": "flux",
        "nologo": "true",
        "seed": random.randint(0, 1000000)
    }
    if IMAGE_GENERATION_WIDTH and IMAGE_GENERATION_WIDTH > 0:
        params["width"] = IMAGE_GENERATION_WIDTH
    if IMAGE_GENERATION_HEIGHT and IMAGE_GENERATION_HEIGHT > 0:
        params["height"] = IMAGE_GENERATION_HEIGHT

    timeout = httpx.Timeout(
        connect=10.0,
        read=IMAGE_GENERATION_TIMEOUT,
        write=20.0,
        pool=30.0,
    )

    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True, headers=headers) as client:
        last_exc: Exception | None = None

        for attempt in range(3):
            try:
                # Копируем заголовки, чтобы не менять глобальные (важно для fallback)
                req_headers = headers.copy()
                
                # Если переключились на публичный URL (fallback), убираем авторизацию
                if "image.pollinations.ai" in url:
                    req_headers.pop("Authorization", None)
                
                logger.info(
                    "Sending image request to %s (attempt %d/3, model=%s)...",
                    url,
                    attempt + 1,
                    params.get("model")
                )

                response = await client.get(url, params=params, headers=req_headers)

                # ==== SUCCESS ====
                if response.status_code == 200:
                    content_type = response.headers.get("content-type", "").lower()

                    # Проверяем, что это изображение
                    if "image" in content_type or "application/octet-stream" in content_type:
                        if not response.content:
                            raise ImageGenerationError("Сервис вернул пустой файл.")
                        return response.content

                    # Если вернулся JSON (например, loading)
                    if "application/json" in content_type and "loading" in response.text.lower():
                        await asyncio.sleep(2 + attempt + random.random())
                        continue

                    # Если вернулся HTML при авторизованном запросе — пробуем публичный эндпоинт
                    if POLLINATIONS_API_KEY and "enter.pollinations.ai" in url:
                        logger.warning("Auth endpoint failed (returned HTML). Fallback to public.")
                        url = url.replace("enter.pollinations.ai", "image.pollinations.ai")
                        await asyncio.sleep(1)
                        continue

                    # Любой другой текст/HTML
                    raise ImageGenerationError(
                        f"Сервис вернул не изображение, content-type={content_type}. Ответ: {response.text[:300]}"
                    )

                # ==== retry-friendly статусы ====
                if response.status_code in (500, 502, 503, 504, 529):
                    # Переключаем модель на turbo, если flux падает
                    if attempt == 1 and params.get("model") == "flux":
                        logger.info("Switching model to 'turbo' due to server errors")
                        params["model"] = "turbo"

                    await asyncio.sleep(2 + attempt + random.random())
                    continue

                # ==== фатальные клиентские ошибки ====
                if response.status_code in (400, 401, 403, 404):
                    raise ImageGenerationError(
                        f"Сервис вернул {response.status_code}: {response.text}"
                    )

            except httpx.HTTPError as exc:
                logger.warning("Request failed: %s", exc)
                last_exc = exc

                # Переключаем модель на turbo при проблемах с соединением
                if attempt == 1 and params.get("model") == "flux":
                    logger.info("Switching model to 'turbo' due to connection issues")
                    params["model"] = "turbo"

                if attempt == 2:
                    raise ImageGenerationError(f"Сбой подключения: {exc}") from exc

                await asyncio.sleep(2 + attempt + random.random())

    # ==== UNKNOWN ERROR ====
    raise ImageGenerationError(f"Неизвестная ошибка: {last_exc}")