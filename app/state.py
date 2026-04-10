import json
from pathlib import Path
import random

import aiofiles
from app.config import (
    ALLOWED_USER_IDS as _RAW_ALLOWED_USER_IDS,
    CONTEXT_LIMIT_TOKENS,
    CHECK_SYNTAX,
    CONTEXT_POLICY,
    ENFORCE_LAST_MESSAGE_PRIORITY,
    FORMAT_WITH_LLM,
    HISTORY_LIMIT,
    MAX_RESPONSE_CHARS,
    MAX_TOKENS,
    PLAIN_TEXT_OUTPUT,
    RENDER_MARKDOWN,
    RESPONSE_FORMAT,
    RANDOM_QUESTIONS,
    RANDOM_QUESTION_PROBABILITY,
    RANDOM_PARTICIPATION_PROBABILITY,
    STRIP_MARKDOWN,
    SYSTEM_PROMPT,
    TEMPERATURE,
    TRIGGER_WORD,
)

def _normalize_allowed_user_ids(value):
    if isinstance(value, str):
        items = [item.strip() for item in value.split(",") if item.strip()]
        return {int(item) for item in items}
    return value


ALLOWED_USER_IDS = _normalize_allowed_user_ids(_RAW_ALLOWED_USER_IDS)

CHAT_MEMORY = {}
CHAT_SETTINGS = {}
CHAT_SEEN_USERS = {}
LAST_RAW_TRANSCRIPTION = {}

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CHAT_SETTINGS_FILE = DATA_DIR / "chat_settings.json"

DEFAULT_SETTINGS = {
    "chat_title": "",
    "added_by": None,
    "system_prompt": SYSTEM_PROMPT,
    "context_policy": CONTEXT_POLICY,
    "extra_prompt": "",
    "mood": "",
    "response_format": RESPONSE_FORMAT,
    "trigger_word": TRIGGER_WORD,
    "max_tokens": MAX_TOKENS,
    "max_response_chars": MAX_RESPONSE_CHARS,
    "temperature": TEMPERATURE,
    "format_with_llm": FORMAT_WITH_LLM,
    "enforce_last_message_priority": ENFORCE_LAST_MESSAGE_PRIORITY,
    "plain_text_output": PLAIN_TEXT_OUTPUT,
    "render_markdown": RENDER_MARKDOWN,
    "check_syntax": CHECK_SYNTAX,
    "strip_markdown": STRIP_MARKDOWN,
    "pending_action": "",
    "pending_user_id": None,
    "voice_response": False,
    "random_questions": RANDOM_QUESTIONS,
    "random_question_prob": RANDOM_QUESTION_PROBABILITY,
    "random_participation_prob": RANDOM_PARTICIPATION_PROBABILITY,
}


async def load_persisted_chat_settings():
    if not CHAT_SETTINGS_FILE.exists():
        return
    try:
        async with aiofiles.open(CHAT_SETTINGS_FILE, "r", encoding="utf-8") as stream:
            content = await stream.read()
            raw = json.loads(content)
    except (OSError, json.JSONDecodeError):
        return
    for raw_chat_id, settings in raw.items():
        try:
            chat_id = int(raw_chat_id)
        except (TypeError, ValueError):
            continue
        CHAT_SETTINGS[chat_id] = settings


async def _write_chat_settings():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    payload = {str(chat_id): settings for chat_id, settings in CHAT_SETTINGS.items()}
    try:
        async with aiofiles.open(CHAT_SETTINGS_FILE, "w", encoding="utf-8") as stream:
            await stream.write(json.dumps(payload, ensure_ascii=False, indent=2))
    except OSError:
        return


async def persist_settings():
    await _write_chat_settings()

def get_settings(chat_id):
    settings = CHAT_SETTINGS.get(chat_id)
    if settings is None:
        settings = dict(DEFAULT_SETTINGS)
        CHAT_SETTINGS[chat_id] = settings
    else:
        for key, value in DEFAULT_SETTINGS.items():
            settings.setdefault(key, value)
    return settings


def get_all_known_groups():
    return [cid for cid in CHAT_SETTINGS.keys() if cid < 0]


async def reset_settings(chat_id):
    CHAT_SETTINGS.pop(chat_id, None)
    await persist_settings()


def mark_user_seen(chat_id, user_id, username, first_name):
    if chat_id not in CHAT_SEEN_USERS:
        CHAT_SEEN_USERS[chat_id] = {}
    CHAT_SEEN_USERS[chat_id][user_id] = {"username": username, "first_name": first_name}


def get_random_seen_user(chat_id, exclude_user_id=None):
    users = CHAT_SEEN_USERS.get(chat_id, {})
    choices = [u for uid, u in users.items() if uid != exclude_user_id]
    if choices:
        return random.choice(choices)
    return None


def set_raw_transcription(chat_id, text):
    LAST_RAW_TRANSCRIPTION[chat_id] = text


def get_raw_transcription(chat_id):
    return LAST_RAW_TRANSCRIPTION.get(chat_id, "Нет сохраненной транскрибации.")

def clear_history(chat_id):
    CHAT_MEMORY.pop(chat_id, None)


def set_history(chat_id, history):
    CHAT_MEMORY[chat_id] = list(history)


def get_history(chat_id):
    return CHAT_MEMORY.setdefault(chat_id, [])


def append_history(chat_id, role, content):
    history = get_history(chat_id)
    history.append({"role": role, "content": content})
    max_items = max(HISTORY_LIMIT * 2, 2)
    if len(history) > max_items:
        del history[:-max_items]


def trim_oldest_history(history):
    if len(history) >= 2:
        return history[2:]
    return []


def set_pending(settings, action, user_id):
    settings["pending_action"] = action
    settings["pending_user_id"] = user_id


def clear_pending(settings):
    settings["pending_action"] = ""
    settings["pending_user_id"] = None


async def apply_pending_action(action, text, settings):
    value = text.strip()
    if action == "set_mood":
        if not value:
            return False, "Нужно указать настроение."
        settings["mood"] = value
        await persist_settings()
        return True, "Настроение обновлено."
    if action == "set_prompt":
        if not value:
            return False, "Нужно указать текст промпта."
        settings["extra_prompt"] = value
        await persist_settings()
        return True, "Дополнительный промпт сохранен."
    if action == "set_trigger":
        if not value:
            return False, "Нужно указать слово-триггер."
        trigger = value.split()[0]
        settings["trigger_word"] = trigger
        await persist_settings()
        return True, f"Триггер обновлен: {trigger}"
    if action == "set_max":
        try:
            max_tokens = int(value)
        except ValueError:
            return False, "Нужно число токенов, например: 512."
        if max_tokens < 16:
            return False, "Минимум 16 токенов."
        if max_tokens >= CONTEXT_LIMIT_TOKENS:
            return False, "Слишком большое значение для контекста."
        settings["max_tokens"] = max_tokens
        await persist_settings()
        return True, f"Лимит ответа обновлен: {max_tokens} токенов."
    return False, "Неизвестная команда."


def is_allowed_user(user_id):
    if not ALLOWED_USER_IDS:
        return True
    if user_id is None:
        return False
    return user_id in ALLOWED_USER_IDS
