import asyncio
import logging
from io import BytesIO

logger = logging.getLogger(__name__)

_silero_model = None
_silero_sample_rate = 48000

def _get_silero_model():
    global _silero_model
    if _silero_model is None:
        import torch
        logger.info("Загрузка локальной модели Silero TTS (v4_ru)...")
        # Для TTS мощности CPU более чем достаточно, генерация происходит почти мгновенно
        device = torch.device("cpu")
        model, _ = torch.hub.load(
            repo_or_dir="snakers4/silero-models",
            model="silero_tts",
            language="ru", 
            speaker="v4_ru",
            trust_repo=True
        )
        model.to(device)
        _silero_model = model
    return _silero_model

def _generate_silero_sync(text: str, voice: str) -> bytes:
    import soundfile as sf
    
    # У Silero есть техническое ограничение на длину текста за один проход (~1000 символов)
    safe_text = text[:1000] if len(text) > 1000 else text
    
    model = _get_silero_model()
    
    valid_speakers = {"aidar", "baya", "kseniya", "xenia", "eugene", "random"}
    if voice in {"male", "aidar"}:
        speaker = "aidar"
    elif voice in valid_speakers:
        speaker = voice
    else:
        speaker = "kseniya"
    
    audio_tensor = model.apply_tts(text=safe_text, speaker=speaker, sample_rate=_silero_sample_rate)
    audio_np = audio_tensor.cpu().numpy()
    
    fp = BytesIO()
    # Сохраняем сгенерированный массив в байтовый поток формата WAV напрямую через soundfile
    sf.write(fp, audio_np, _silero_sample_rate, format="WAV")
    return fp.getvalue()

async def generate_speech(text: str, voice: str = "kseniya") -> bytes:
    if not text.strip():
        return b""
        
    try:
        return await asyncio.to_thread(_generate_silero_sync, text, voice)
    except ImportError as e:
        logger.error("Ошибка импорта (не хватает библиотеки): %s", e)
        return b""
    except Exception as exc:
        logger.error("Сбой в локальной Silero TTS: %s", exc)
        return b""