import asyncio
import logging
import gc
from io import BytesIO

from app.llm_client import chat_completion

logger = logging.getLogger(__name__)

class ImageGenerationError(RuntimeError):
    """Raised when image generation fails."""
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

            logger.info("Loading local DreamShaper pipeline...")
            
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
                    logger.info("Enabled xformers acceleration for image generation")
                except Exception:
                    logger.info("xformers acceleration is unavailable (recommended for CUDA: pip install xformers)")

            _pipeline = _pipeline.to(device)

            logger.info("DreamShaper pipeline loaded on %s", device)
        except ImportError as exc:
            raise ImageGenerationError(
                "Image generation dependencies are missing. "
                "Install them with: pip install diffusers transformers torch accelerate"
            ) from exc
        except Exception as exc:
            logger.error("Failed to initialize the image pipeline: %s", exc)
            raise ImageGenerationError(f"Model loading failed: {exc}")

    return _pipeline


def _generate_sync(prompt: str) -> bytes:
    import torch
    try:
        pipe = _get_pipeline()
        logger.info("Starting local image generation for prompt: %s", prompt)
        
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
        logger.error("Local image generation failed: %s", exc)
        raise ImageGenerationError(f"Generation failed: {exc}")
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()
        gc.collect()


async def generate_image(prompt: str) -> bytes:
    """
    Generate an image asynchronously with the local DreamShaper pipeline.
    Uses a worker thread so the bot event loop is not blocked.
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
        logger.info("Original prompt: %s | Stable Diffusion prompt: %s", prompt, english_prompt)
    except Exception as exc:
        logger.warning("Prompt translation failed, using the original text: %s", exc)
        english_prompt = prompt

    async with _get_lock():
        return await asyncio.to_thread(_generate_sync, english_prompt)
