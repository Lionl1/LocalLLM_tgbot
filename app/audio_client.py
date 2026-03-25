import asyncio
import logging
import whisper
import subprocess
import tempfile
import os
import glob

logger = logging.getLogger(__name__)

_whisper_model = None

def _get_whisper_model():
    global _whisper_model
    if _whisper_model is None:
        logger.info("Загрузка локальной модели Whisper (base)...")
        _whisper_model = whisper.load_model("base")
    return _whisper_model

def _transcribe_sync(file_path: str) -> str:
    try:
        model = _get_whisper_model()
        
        file_size = os.path.getsize(file_path)
        if file_size > 15 * 1024 * 1024:
            logger.info("Аудио превышает лимит размера (15MB), нарезаем на части...")
            with tempfile.TemporaryDirectory() as tmpdir:
                base_name = os.path.join(tmpdir, "chunk_%03d.mp3")
                cmd = [
                    "ffmpeg", "-i", file_path,
                    "-f", "segment", "-segment_time", "300",
                    "-c:a", "libmp3lame",
                    base_name
                ]
                subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                
                chunks = sorted(glob.glob(os.path.join(tmpdir, "chunk_*.mp3")))
                full_text = []
                for chunk in chunks:
                    result = model.transcribe(chunk)
                    full_text.append(result.get("text", "").strip())
                return " ".join(full_text).strip()
                
        result = model.transcribe(file_path)
        return result.get("text", "").strip()
    except Exception as exc:
        logger.error("Ошибка при распознавании аудио через Whisper: %s", exc)
        return ""

async def transcribe_audio(file_path: str) -> str:
    """Асинхронная обертка для синхронной транскрибации локальным Whisper."""
    return await asyncio.to_thread(_transcribe_sync, file_path)