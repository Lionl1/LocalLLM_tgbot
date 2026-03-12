import logging
import re

from app.config import CONTEXT_LIMIT_TOKENS, SYSTEM_PROMPT
from app.llm_client import chat_completion
from app.state import trim_oldest_history
from app.text_utils import _estimate_messages_tokens

logger = logging.getLogger(__name__)

_SUMMARY_PREFIX = "Краткое содержание предыдущего разговора:"


def _priority_instruction(settings):
    if settings.get("enforce_last_message_priority", True):
        return (
            "Всегда отвечай на последний запрос пользователя. "
            "Историю используй только если она напрямую связана с текущим запросом."
        )
    return ""


def _compose_system_prompt(settings):
    parts = [settings["system_prompt"]]
    if settings["context_policy"]:
        parts.append(f"Правила контекста: {settings['context_policy']}")
    priority = _priority_instruction(settings)
    if priority:
        parts.append(priority)
    if settings.get("plain_text_output"):
        parts.append(
            "Ответ по умолчанию без Markdown. "
            "Если используешь Markdown (код, таблицы), сделай разметку валидной."
        )
    if settings["extra_prompt"]:
        parts.append(f"Дополнительные инструкции: {settings['extra_prompt']}")
    if settings["mood"]:
        parts.append(f"Настроение ответа: {settings['mood']}")
    if settings["response_format"]:
        parts.append(f"Формат ответа: {settings['response_format']}")
    if settings["max_response_chars"] > 0:
        parts.append(
            f"Ограничение: не более {settings['max_response_chars']} символов."
        )
    return "\n\n".join(parts)


def _build_messages(history, prompt, reply_text, settings, web_context=""):
    system_prompt = _compose_system_prompt(settings)
    if web_context:
        system_prompt = f"{system_prompt}\n\n{web_context}"
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(history)
    if reply_text:
        messages.append(
            {
                "role": "user",
                "content": f"Сообщение, на которое идет ответ:\n{reply_text}",
            }
        )
    messages.append({"role": "user", "content": prompt})
    return messages


def _build_flat_fallback_messages(
    history, prompt, reply_text, settings, web_context=""
):
    system_prompt = settings.get("system_prompt") or SYSTEM_PROMPT
    priority = _priority_instruction(settings)
    context_parts = []
    if priority:
        context_parts.append(priority)
    if history:
        context_parts.append("История диалога:")
        for message in history[-8:]:
            role = message.get("role", "user")
            label = "Пользователь" if role == "user" else "Ассистент"
            content = (message.get("content") or "").strip()
            if content:
                context_parts.append(f"{label}: {content}")
    if reply_text:
        context_parts.append("Сообщение, на которое идет ответ:")
        context_parts.append(reply_text)
    if web_context:
        context_parts.append("Контекст из интернета:")
        context_parts.append(web_context)
    context_block = "\n".join(context_parts).strip()
    if context_block:
        user_content = f"{context_block}\n\nТекущий запрос:\n{prompt}"
    else:
        user_content = prompt
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]


def _trim_to_char_limit(text, max_chars):
    if max_chars <= 0 or not text or len(text) <= max_chars:
        return text
    if max_chars <= 1:
        return text[:max_chars]
    snippet = text[:max_chars]
    cut_at = max(snippet.rfind("\n"), snippet.rfind(" "))
    if cut_at >= max_chars * 0.6:
        snippet = snippet[:cut_at]
    return snippet.rstrip()


def _looks_like_markdown(text):
    pattern = r"(```|`[^`]+`|\*\*.+?\*\*|__.+?__|\[(.+?)\]\((.+?)\)|^#{1,6}\s)"
    return bool(re.search(pattern, text or "", flags=re.M))


def _strip_markdown_syntax(text):
    if not text:
        return text
    text = re.sub(r"```(?:[^\n]*\n)?(.*?)```", lambda m: m.group(1).strip(), text, flags=re.S)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"__(.+?)__", r"\1", text)
    text = re.sub(r"\*(.+?)\*", r"\1", text)
    text = re.sub(r"_(.+?)_", r"\1", text)
    text = re.sub(r"(?m)^#{1,6}\s+", "", text)
    text = re.sub(r"(?m)^>\s?", "", text)
    return text


async def _format_response_with_llm(prompt, response_text, settings):
    if not settings.get("response_format") or not settings.get("format_with_llm"):
        return response_text
    instructions = [settings.get("system_prompt") or SYSTEM_PROMPT]
    if settings.get("context_policy"):
        instructions.append(f"Правила контекста: {settings['context_policy']}")
    if settings.get("extra_prompt"):
        instructions.append(f"Дополнительные инструкции: {settings['extra_prompt']}")
    if settings.get("mood"):
        instructions.append(f"Настроение ответа: {settings['mood']}")
    instructions.append(f"Формат ответа: {settings['response_format']}")
    if settings.get("max_response_chars", 0) > 0:
        instructions.append(
            f"Ограничение: не более {settings['max_response_chars']} символов."
        )
    requirements = "\n\n".join(instructions)
    user_content = (
        "Соблюдай требования и формат. Не добавляй новых фактов и не меняй смысл. "
        "Если используешь Markdown, сделай разметку валидной.\n\n"
        f"Требования:\n{requirements}\n\n"
        f"Запрос:\n{prompt}\n\n"
        f"Черновик ответа:\n{response_text}\n\n"
        "Перепиши ответ строго по требованиям."
    )
    messages = [
        {
            "role": "system",
            "content": "Ты редактор ответов. Перепиши ответ под заданную роль и формат.",
        },
        {"role": "user", "content": user_content},
    ]
    try:
        formatted = await chat_completion(
            messages,
            max_tokens=settings["max_tokens"],
            temperature=min(settings["temperature"], 0.2),
        )
    except Exception as exc:
        logger.warning("Response formatting failed: %s", exc)
        return response_text
    formatted = (formatted or "").strip()
    return formatted or response_text


async def _fix_syntax_with_llm(text, settings):
    if not settings.get("check_syntax"):
        return text
    
    prompt = (
        "Проверь следующий текст на наличие грамматических, орфографических и пунктуационных ошибок. "
        "Исправь их, сохранив исходный стиль и структуру (включая Markdown). "
        "Не меняй сленг, неологизмы и творческие обороты.\n"
        "Верни ТОЛЬКО исправленный текст, без вступлений и пояснений.\n\n"
        f"Текст:\n{text}"
    )
    
    messages = [
        {"role": "system", "content": "Ты корректор. Твоя задача — исправить грубые ошибки, сохраняя авторский стиль."},
        {"role": "user", "content": prompt}
    ]
    
    try:
        logger.info("Performing syntax check request...")
        corrected = await chat_completion(
            messages,
            max_tokens=settings["max_tokens"],
            temperature=0.1,
        )
        return (corrected or "").strip() or text
    except Exception as exc:
        logger.warning("Syntax check failed: %s", exc)
        return text


async def _postprocess_response(prompt, response_text, settings):
    text = (response_text or "").strip()
    if not text:
        return text, None
    text = await _format_response_with_llm(prompt, text, settings)
    text = await _fix_syntax_with_llm(text, settings)
    if settings.get("strip_markdown"):
        text = _strip_markdown_syntax(text)
        return text, None
    parse_mode = None
    if settings.get("render_markdown") and _looks_like_markdown(text):
        parse_mode = "Markdown"
    if settings.get("max_response_chars", 0) > 0:
        text = _trim_to_char_limit(text, settings["max_response_chars"])
    return text, parse_mode


def _max_prompt_tokens(max_tokens):
    return max(CONTEXT_LIMIT_TOKENS - max_tokens, 1)


def _context_limit_exceeded(messages, max_tokens):
    return _estimate_messages_tokens(messages) > _max_prompt_tokens(max_tokens)


async def _generate_summary(text_to_summarize):
    prompt = (
        "Твоя задача — обновить краткое содержание (summary) диалога.\n"
        "1. Сохрани важные факты, имена и контекст из текущего саммари (если есть).\n"
        "2. Добавь ключевую информацию из новых сообщений.\n"
        "3. Будь предельно лаконичен, убирай воду и приветствия.\n\n"
        "4. Учити шутки сарказм и юмор если он был"
        f"ТЕКСТ ДЛЯ АНАЛИЗА:\n{text_to_summarize}"
    )
    
    try:
        summary = await chat_completion(
            [{"role": "user", "content": prompt}],
            max_tokens=400,
            temperature=0.3
        )
        return summary
    except Exception as exc:
        logger.warning("Failed to generate summary: %s", exc)
        return None


async def _trim_history_to_fit(history, prompt, reply_text, settings, web_context=""):
    trimmed = False
    messages = _build_messages(history, prompt, reply_text, settings, web_context)
    
    # Если контекст уже влезает — возвращаем как есть
    if not _context_limit_exceeded(messages, settings["max_tokens"]):
        return history, trimmed, messages

    # Проверяем, есть ли уже саммари в начале истории (System message с префиксом)
    existing_summary_text = ""
    start_idx = 0
    if history and history[0].get("role") == "system" and history[0]["content"].startswith(_SUMMARY_PREFIX):
        existing_summary_text = history[0]["content"].replace(_SUMMARY_PREFIX, "").strip()
        start_idx = 1  # Пропускаем сообщение с саммари при нарезке, учтем его отдельно

    # Работаем только с "живыми" сообщениями (без старого саммари)
    active_history = history[start_idx:]

    # Попытка саммаризации, если есть что сжимать
    if len(active_history) > 1:
        # Сжимаем старые 60% сообщений, оставляя свежие 40%
        split_idx = max(1, int(len(active_history) * 0.6))
        msgs_to_compress = active_history[:split_idx]
        recent_history = active_history[split_idx:]

        # Формируем текст для LLM: Старое саммари + Старые сообщения
        text_block = ""
        if existing_summary_text:
            text_block += f"=== ТЕКУЩЕЕ САММАРИ ===\n{existing_summary_text}\n\n"
        
        text_block += "=== НОВЫЕ СООБЩЕНИЯ ===\n"
        for msg in msgs_to_compress:
            role = "User" if msg.get("role") == "user" else "Assistant"
            text_block += f"{role}: {msg.get('content', '')}\n"

        logger.info("Context exceeded. Updating summary with %d messages...", len(msgs_to_compress))
        new_summary = await _generate_summary(text_block)

        if new_summary:
            summary_message = {
                "role": "system",
                "content": f"{_SUMMARY_PREFIX} {new_summary}"
            }
            history = [summary_message] + recent_history
            trimmed = True
            # Пересобираем сообщения с учетом саммари
            messages = _build_messages(history, prompt, reply_text, settings, web_context)

    # Если даже с саммари всё равно не влезает (или саммари не удалось) — режем по старинке
    while history and _context_limit_exceeded(messages, settings["max_tokens"]):
        history = trim_oldest_history(history)
        trimmed = True
        messages = _build_messages(history, prompt, reply_text, settings, web_context)
        
    return history, trimmed, messages


def _is_context_overflow_error(exc):
    text = str(exc).casefold()
    tokens = [
        "context",
        "context_length_exceeded",
        "maximum context",
        "max context",
        "token limit",
        "too many tokens",
        "max_tokens",
    ]
    return any(token in text for token in tokens)


def _is_message_header_error(exc):
    text = str(exc).casefold()
    tokens = [
        "message header",
        "unexpected tokens remaining",
        "chat template",
    ]
    return any(token in text for token in tokens)
