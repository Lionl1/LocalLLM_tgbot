import math
import re

from telegram.constants import ChatType

from app.config import MAX_TELEGRAM_MESSAGE, TOKEN_CHAR_RATIO


def _normalize(text):
    return text.casefold()


def _strip_leading(text):
    return text.lstrip(" \t\r\n")


def _strip_after_prefix(text, prefix):
    stripped = _strip_leading(text)
    if _normalize(stripped).startswith(_normalize(prefix)):
        remainder = stripped[len(prefix) :]
        return remainder.lstrip(" \t\r\n,.:;—-")
    return text


def _strip_bot_mention(text, bot_username):
    if not bot_username:
        return text
    mention = f"@{bot_username}"
    return _strip_after_prefix(text, mention)


def _strip_trigger(text, trigger_word):
    return _strip_after_prefix(text, trigger_word)


def _starts_with_prefix(text, prefix):
    stripped = _strip_leading(text)
    return _normalize(stripped).startswith(_normalize(prefix))


def _extract_web_query(prompt):
    stripped = _strip_leading(prompt)
    normalized = _normalize(stripped)
    for prefix in ("web:", "search:", "поиск:"):
        if normalized.startswith(prefix):
            query = stripped[len(prefix) :].strip(" \t\r\n,.:;—-")
            return query, query
    for prefix in (
        "найди в интернете",
        "найди в интернет",
        "найди в интрнете",
        "найди в сети",
    ):
        if normalized.startswith(prefix):
            query = stripped[len(prefix) :].strip(" \t\r\n,.:;—-")
            return query, query
    return "", prompt


def _format_search_results(results, query):
    lines = [f"Результаты поиска для запроса: {query}"]
    for idx, item in enumerate(results, 1):
        title = (item.get("title") or "").strip() or "Без названия"
        url = (item.get("url") or "").strip()
        snippet = (item.get("snippet") or "").strip()
        lines.append(f"{idx}. {title}")
        if url:
            lines.append(url)
        if snippet:
            lines.append(snippet)
        lines.append("")
    return "\n".join(lines).strip()


def _is_reply_to_bot(update, bot_id):
    reply = update.message.reply_to_message
    if not reply or not reply.from_user:
        return False
    return reply.from_user.id == bot_id


def _is_triggered(update, text, bot_id, bot_username, trigger_word):
    if update.message.chat.type == ChatType.PRIVATE:
        return True
    if _is_reply_to_bot(update, bot_id):
        return True
    if not text:
        return False
    if _starts_with_prefix(text, trigger_word):
        return True
    if bot_username and _starts_with_prefix(text, f"@{bot_username}"):
        return True
    return False


def _extract_prompt(text, bot_username, trigger_word):
    prompt = _strip_trigger(text, trigger_word)
    prompt = _strip_bot_mention(prompt, bot_username)
    return prompt.strip()


def _get_reply_text(message):
    if not message or not message.reply_to_message:
        return ""
    reply = message.reply_to_message
    if reply.from_user and reply.from_user.is_bot:
        return ""
    text = reply.text or reply.caption or ""
    return text.strip()


RESET_TOKENS = {
    "reset",
    "/reset",
    "clear",
    "сброс",
    "очисти",
    "очистить",
    "очистка",
    "сбрось",
}


def _split_reset_request(text):
    stripped = text.strip()
    if not stripped:
        return False, ""
    parts = stripped.split(maxsplit=1)
    head = parts[0].strip(" \t\r\n,.:;—-").casefold()
    if head not in RESET_TOKENS:
        return False, ""
    remainder = ""
    if len(parts) > 1:
        remainder = parts[1].strip()
    return True, remainder


def _get_command_text(message_text):
    if not message_text:
        return ""
    parts = message_text.split(" ", 1)
    if len(parts) == 1:
        return ""
    return parts[1].strip()


def _split_message(text, limit=MAX_TELEGRAM_MESSAGE):
    chunks = []
    remaining = text or ""
    while remaining:
        if len(remaining) <= limit:
            chunks.append(remaining)
            break
        split_at = remaining.rfind("\n", 0, limit)
        if split_at == -1:
            split_at = remaining.rfind(" ", 0, limit)
        if split_at == -1 or split_at < limit // 2:
            split_at = limit
        chunk = remaining[:split_at].rstrip()
        remaining = remaining[split_at:].lstrip()
        if chunk:
            chunks.append(chunk)
    return chunks


def _estimate_tokens(text):
    if not text:
        return 0
    ratio = TOKEN_CHAR_RATIO if TOKEN_CHAR_RATIO > 0 else 4
    return max(1, math.ceil(len(text) / ratio))


def _estimate_messages_tokens(messages):
    total = 0
    for message in messages:
        content = message.get("content", "")
        total += 4 + _estimate_tokens(content)
    return total


_IMAGE_ACTION_WORDS = [
    "сгенерируй",
    "нарисуй",
    "создай",
    "сделай",
    "придумай",
    "отрисуй",
    "построй",
    "покажи",
]
_IMAGE_NOUN_WORDS = [
    "картин",
    "рисунк",
    "фото",
    "изображен",
    "иллюстрац",
    "арт",
    "скетч",
    "постер",
]
_IMAGE_POLITE_WORDS = ["пожалуйста", "пж", "плиз", "пжл"]
_IMAGE_REMOVAL_WORDS = _IMAGE_ACTION_WORDS + _IMAGE_NOUN_WORDS + _IMAGE_POLITE_WORDS
_IMAGE_REMOVAL_REGEX = re.compile("|".join(re.escape(word) for word in _IMAGE_REMOVAL_WORDS), re.IGNORECASE)


def detect_image_request(text):
    if not text:
        return ""
    normalized = _normalize(text)
    if not any(action in normalized for action in _IMAGE_ACTION_WORDS):
        return ""
    if not any(noun in normalized for noun in _IMAGE_NOUN_WORDS):
        return ""
    cleaned = _IMAGE_REMOVAL_REGEX.sub(" ", text)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" \t\r\n,.:;—-")
    return cleaned or text.strip()


_SEARCH_ACTION_WORDS = [
    "найди",
    "поищи",
    "проведи поиск",
    "выполни поиск",
    "покажи",
    "узнай",
    "поиск",
]
_SEARCH_CONTEXT_WORDS = [
    "в интернете",
    "в сети",
    "в вебе",
    "онлайн",
    "в веб",
]
_SEARCH_REMOVAL_WORDS = _SEARCH_ACTION_WORDS + _SEARCH_CONTEXT_WORDS + _IMAGE_POLITE_WORDS
_SEARCH_REMOVAL_REGEX = re.compile("|".join(re.escape(word) for word in _SEARCH_REMOVAL_WORDS), re.IGNORECASE)


def detect_search_request(text):
    if not text:
        return ""
    normalized = _normalize(text)
    if not any(action in normalized for action in _SEARCH_ACTION_WORDS):
        return ""
    cleaned = _SEARCH_REMOVAL_REGEX.sub(" ", text)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" \t\r\n,.:;—-")
    return cleaned
