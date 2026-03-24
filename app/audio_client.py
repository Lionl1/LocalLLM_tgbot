import asyncio
import logging
import whisper

logger = logging.getLogger(__name__)

_whisper_model = None

def _get_whisper_model():
    global _whisper_model
    if _whisper_model is None:
        logger.info("Загрузка локальной модели Whisper (base)...")
        # Размеры моделей: "tiny", "base", "small", "medium", "large"
        _whisper_model = whisper.load_model("base")
    return _whisper_model

def _transcribe_sync(file_path: str) -> str:
    try:
        model = _get_whisper_model()
        result = model.transcribe(file_path)
        return result.get("text", "").strip()
    except Exception as exc:
        logger.error("Ошибка при распознавании аудио через Whisper: %s", exc)
        return ""

async def transcribe_audio(file_path: str) -> str:
    """Асинхронная обертка для синхронной транскрибации локальным Whisper."""
    return await asyncio.to_thread(_transcribe_sync, file_path)