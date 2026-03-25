import asyncio
import logging
import gc
from io import BytesIO

from app.llm_client import chat_completion

logger = logging.getLogger(__name__)

class ImageGenerationError(RuntimeError):
    """Исключение, возникающее при ошибках генерации изображения."""
    pass


_pipeline = None
_generation_lock = None

def _get_lock():
    global _generation_lock
    if _generation_lock is None:
        _generation_lock = asyncio.Lock()
    return _generation_lock

def _get_pipeline():
    global _pipeline
    if _pipeline is None:
        try:
            import torch
            from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler

            logger.info("Загрузка локальной модели DreamShaper (SD 1.5)...")
            
            device = "cpu"
            dtype = torch.float32
            
            if torch.cuda.is_available():
                device = "cuda"
                dtype = torch.float16
            elif torch.backends.mps.is_available():
                device = "mps"
                dtype = torch.float32

            _pipeline = DiffusionPipeline.from_pretrained(
                "segmind/tiny-sd",
                torch_dtype=dtype,
                safety_checker=None
            )

            _pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(_pipeline.scheduler.config)

            if device == "cuda":
                try:
                    _pipeline.enable_xformers_memory_efficient_attention()
                    logger.info("Включено ускорение генерации через xformers")
                except Exception:
                    logger.info("Ускорение xformers недоступно (для CUDA рекомендуется: pip install xformers)")

            _pipeline = _pipeline.to(device)

            logger.info(f"Модель DreamShaper успешно загружена на {device}")
        except ImportError as exc:
            raise ImageGenerationError(
                "Не установлены библиотеки для генерации. "
                "Выполните: pip install diffusers transformers torch accelerate"
            ) from exc
        except Exception as exc:
            logger.error("Ошибка при инициализации пайплайна: %s", exc)
            raise ImageGenerationError(f"Ошибка загрузки модели: {exc}")

    return _pipeline


def _generate_sync(prompt: str) -> bytes:
    import torch
    try:
        pipe = _get_pipeline()
        logger.info("Начинаем локальную генерацию для запроса: %s", prompt)
        
        enhanced_prompt = (
                        f"{prompt}, accurate depiction, matching the prompt, clear subject, "
                        "well-defined details, coherent composition, realistic proportions"
                        )

        negative_prompt = (
                            "blurry, low quality, distorted, deformed, bad anatomy, extra limbs, "
                            "missing fingers, malformed, noisy, artifacts, text, watermark, oversaturated"
                        )

        result = pipe(
            prompt=enhanced_prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=15,
            guidance_scale=7,
            height=256,
            width=256
        )
        image = result.images[0]
        
        stream = BytesIO()
        image.save(stream, format="PNG")
        return stream.getvalue()
    except Exception as exc:
        logger.error("Ошибка при локальной генерации изображения: %s", exc)
        raise ImageGenerationError(f"Ошибка генерации: {exc}")
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()
        gc.collect()


async def generate_image(prompt: str) -> bytes:
    """
    Генерирует изображение асинхронно через локальную модель DreamShaper.
    Работает в отдельном потоке (to_thread), чтобы не блокировать бота.
    """
    translation_prompt = (
        "Translate the following text into an English prompt for Stable Diffusion. "
        "Keep it concise, descriptive, and return ONLY the English translation without quotes or extra text.\n\n"
        f"Text: {prompt}"
    )
    
    try:
        english_prompt = await chat_completion(
            [{"role": "user", "content": translation_prompt}],
            max_tokens=200,
            temperature=0.3
        )
        english_prompt = (english_prompt or prompt).strip()
        logger.info("Оригинальный промпт: %s | Промпт для SD: %s", prompt, english_prompt)
    except Exception as exc:
        logger.warning("Не удалось перевести промпт, используем оригинал: %s", exc)
        english_prompt = prompt

    async with _get_lock():
        return await asyncio.to_thread(_generate_sync, english_prompt)