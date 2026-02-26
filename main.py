import logging
import math
from telegram import BotCommand, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.constants import ChatType
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from config import (
    CONTEXT_LIMIT_TOKENS,
    HISTORY_LIMIT,
    MAX_TOKENS,
    SYSTEM_PROMPT,
    TELEGRAM_BOT_TOKEN,
    TEMPERATURE,
    TOKEN_CHAR_RATIO,
    TRIGGER_WORD,
    ALLOWED_USER_IDS,
    WEB_SEARCH_ENABLED,
    WEB_SEARCH_MAX_RESULTS,
)
from llm_client import chat_completion
from search_client import WebSearchError, search_web


logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


CHAT_MEMORY = {}
CHAT_SETTINGS = {}
MAX_TELEGRAM_MESSAGE = 4096

DEFAULT_SETTINGS = {
    "system_prompt": SYSTEM_PROMPT,
    "extra_prompt": "",
    "mood": "",
    "trigger_word": TRIGGER_WORD,
    "max_tokens": MAX_TOKENS,
    "temperature": TEMPERATURE,
    "pending_action": "",
    "pending_user_id": None,
}


def _get_settings(chat_id):
    settings = CHAT_SETTINGS.get(chat_id)
    if settings is None:
        settings = dict(DEFAULT_SETTINGS)
        CHAT_SETTINGS[chat_id] = settings
    else:
        for key, value in DEFAULT_SETTINGS.items():
            settings.setdefault(key, value)
    return settings


def _reset_settings(chat_id):
    CHAT_SETTINGS.pop(chat_id, None)


def _compose_system_prompt(settings):
    parts = [settings["system_prompt"]]
    if settings["extra_prompt"]:
        parts.append(f"Дополнительные инструкции: {settings['extra_prompt']}")
    if settings["mood"]:
        parts.append(f"Настроение ответа: {settings['mood']}")
    return "\n\n".join(parts)


def _is_allowed_user(user_id):
    if not ALLOWED_USER_IDS:
        return True
    if user_id is None:
        return False
    return user_id in ALLOWED_USER_IDS


def _format_settings(settings):
    lines = [
        "Текущие настройки:",
        f"Триггер: {settings['trigger_word']}",
        f"Настроение: {settings['mood'] or 'не задано'}",
        f"Доп. промпт: {settings['extra_prompt'] or 'не задан'}",
        f"Макс. ответ (tokens): {settings['max_tokens']}",
    ]
    return "\n".join(lines)


def _settings_keyboard():
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton("Настроение", callback_data="set_mood"),
                InlineKeyboardButton("Очистить настроение", callback_data="clear_mood"),
            ],
            [
                InlineKeyboardButton("Доп. промпт", callback_data="set_prompt"),
                InlineKeyboardButton("Очистить промпт", callback_data="clear_prompt"),
            ],
            [
                InlineKeyboardButton("Лимит ответа", callback_data="set_max"),
                InlineKeyboardButton("Триггер", callback_data="set_trigger"),
            ],
            [
                InlineKeyboardButton(
                    "Показать настройки", callback_data="show_settings"
                ),
                InlineKeyboardButton(
                    "Сбросить настройки", callback_data="reset_settings"
                ),
            ],
            [InlineKeyboardButton("Отмена", callback_data="cancel")],
        ]
    )


def _cancel_keyboard():
    return InlineKeyboardMarkup(
        [[InlineKeyboardButton("Отмена", callback_data="cancel")]]
    )


def _set_pending(settings, action, user_id):
    settings["pending_action"] = action
    settings["pending_user_id"] = user_id


def _clear_pending(settings):
    settings["pending_action"] = ""
    settings["pending_user_id"] = None


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


def _apply_pending_action(action, text, settings):
    value = text.strip()
    if action == "set_mood":
        if not value:
            return False, "Нужно указать настроение."
        settings["mood"] = value
        return True, "Настроение обновлено."
    if action == "set_prompt":
        if not value:
            return False, "Нужно указать текст промпта."
        settings["extra_prompt"] = value
        return True, "Дополнительный промпт сохранен."
    if action == "set_trigger":
        if not value:
            return False, "Нужно указать слово-триггер."
        trigger = value.split()[0]
        settings["trigger_word"] = trigger
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
        return True, f"Лимит ответа обновлен: {max_tokens} токенов."
    return False, "Неизвестная команда."


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


def _max_prompt_tokens(max_tokens):
    return max(CONTEXT_LIMIT_TOKENS - max_tokens, 1)


def _context_limit_exceeded(messages, max_tokens):
    return _estimate_messages_tokens(messages) > _max_prompt_tokens(max_tokens)


def _trim_history_to_fit(history, prompt, reply_text, settings, web_context=""):
    trimmed = False
    messages = _build_messages(history, prompt, reply_text, settings, web_context)
    while history and _context_limit_exceeded(messages, settings["max_tokens"]):
        history = _trim_oldest_history(history)
        trimmed = True
        messages = _build_messages(history, prompt, reply_text, settings, web_context)
    return history, trimmed, messages


def _get_history(chat_id):
    return CHAT_MEMORY.setdefault(chat_id, [])


def _append_history(chat_id, role, content):
    history = _get_history(chat_id)
    history.append({"role": role, "content": content})
    max_items = max(HISTORY_LIMIT * 2, 2)
    if len(history) > max_items:
        del history[:-max_items]


async def start_command(update, context):
    if not _is_allowed_user(update.effective_user.id if update.effective_user else None):
        if update.effective_chat.type == ChatType.PRIVATE:
            await update.message.reply_text("Доступ ограничен.")
        return
    chat_id = update.effective_chat.id
    trigger_word = _get_settings(chat_id)["trigger_word"]
    await update.message.reply_text(
        f"Привет! Напиши сообщение (в группе начни с '{trigger_word}').",
        reply_markup=_settings_keyboard(),
    )


async def help_command(update, context):
    if not _is_allowed_user(update.effective_user.id if update.effective_user else None):
        if update.effective_chat.type == ChatType.PRIVATE:
            await update.message.reply_text("Доступ ограничен.")
        return
    text = (
        "Команды:\n"
        "/settings — открыть настройки\n"
        "/reset — очистить контекст\n"
        "/search <запрос> — поиск в интернете\n"
        "/setmood <текст> — задать настроение\n"
        "/setprompt <текст> — доп. системный промпт\n"
        "/setmax <число> — лимит ответа в токенах\n"
        "/settrigger <имя> — слово-триггер\n"
        "/resetsettings — сбросить настройки\n"
        "/cancel — отменить ввод значения"
    )
    await update.message.reply_text(text, reply_markup=_settings_keyboard())


async def search_command(update, context):
    if not _is_allowed_user(update.effective_user.id if update.effective_user else None):
        if update.effective_chat.type == ChatType.PRIVATE:
            await update.message.reply_text("Доступ ограничен.")
        return
    query = _get_command_text(update.message.text)
    if not query:
        await update.message.reply_text("Укажи запрос: /search <текст>")
        return
    if not WEB_SEARCH_ENABLED:
        await update.message.reply_text(
            "Поиск отключен. Включи WEB_SEARCH_ENABLED=1 в .env."
        )
        return
    try:
        results = await search_web(query, limit=WEB_SEARCH_MAX_RESULTS)
    except WebSearchError as exc:
        logger.exception("Web search failed: %s", exc)
        await update.message.reply_text("Не удалось выполнить поиск.")
        return
    if not results:
        await update.message.reply_text("Ничего не нашел по этому запросу.")
        return
    text = _format_search_results(results, query)
    chunks = _split_message(text)
    await update.message.reply_text(chunks[0])
    for chunk in chunks[1:]:
        await context.bot.send_message(chat_id=update.effective_chat.id, text=chunk)


async def reset_command(update, context):
    if not _is_allowed_user(update.effective_user.id if update.effective_user else None):
        if update.effective_chat.type == ChatType.PRIVATE:
            await update.message.reply_text("Доступ ограничен.")
        return
    chat_id = update.effective_chat.id
    CHAT_MEMORY.pop(chat_id, None)
    await update.message.reply_text("Контекст очищен.")


async def settings_command(update, context):
    if not _is_allowed_user(update.effective_user.id if update.effective_user else None):
        if update.effective_chat.type == ChatType.PRIVATE:
            await update.message.reply_text("Доступ ограничен.")
        return
    chat_id = update.effective_chat.id
    settings = _get_settings(chat_id)
    await update.message.reply_text(
        _format_settings(settings),
        reply_markup=_settings_keyboard(),
    )


async def set_mood_command(update, context):
    if not _is_allowed_user(update.effective_user.id if update.effective_user else None):
        if update.effective_chat.type == ChatType.PRIVATE:
            await update.message.reply_text("Доступ ограничен.")
        return
    chat_id = update.effective_chat.id
    text = _get_command_text(update.message.text)
    if not text:
        settings = _get_settings(chat_id)
        _set_pending(settings, "set_mood", update.effective_user.id)
        await update.message.reply_text(
            "Введи настроение для ответов.",
            reply_markup=_cancel_keyboard(),
        )
        return
    settings = _get_settings(chat_id)
    settings["mood"] = text
    await update.message.reply_text("Настроение обновлено.")


async def clear_mood_command(update, context):
    if not _is_allowed_user(update.effective_user.id if update.effective_user else None):
        if update.effective_chat.type == ChatType.PRIVATE:
            await update.message.reply_text("Доступ ограничен.")
        return
    chat_id = update.effective_chat.id
    settings = _get_settings(chat_id)
    settings["mood"] = ""
    await update.message.reply_text("Настроение очищено.")


async def set_prompt_command(update, context):
    if not _is_allowed_user(update.effective_user.id if update.effective_user else None):
        if update.effective_chat.type == ChatType.PRIVATE:
            await update.message.reply_text("Доступ ограничен.")
        return
    chat_id = update.effective_chat.id
    text = _get_command_text(update.message.text)
    if not text:
        settings = _get_settings(chat_id)
        _set_pending(settings, "set_prompt", update.effective_user.id)
        await update.message.reply_text(
            "Введи дополнительный системный промпт.",
            reply_markup=_cancel_keyboard(),
        )
        return
    settings = _get_settings(chat_id)
    settings["extra_prompt"] = text
    await update.message.reply_text("Дополнительный промпт сохранен.")


async def clear_prompt_command(update, context):
    if not _is_allowed_user(update.effective_user.id if update.effective_user else None):
        if update.effective_chat.type == ChatType.PRIVATE:
            await update.message.reply_text("Доступ ограничен.")
        return
    chat_id = update.effective_chat.id
    settings = _get_settings(chat_id)
    settings["extra_prompt"] = ""
    await update.message.reply_text("Дополнительный промпт очищен.")


async def set_trigger_command(update, context):
    if not _is_allowed_user(update.effective_user.id if update.effective_user else None):
        if update.effective_chat.type == ChatType.PRIVATE:
            await update.message.reply_text("Доступ ограничен.")
        return
    chat_id = update.effective_chat.id
    text = _get_command_text(update.message.text)
    if not text:
        settings = _get_settings(chat_id)
        _set_pending(settings, "set_trigger", update.effective_user.id)
        await update.message.reply_text(
            "Введи новое слово-триггер.",
            reply_markup=_cancel_keyboard(),
        )
        return
    trigger = text.split()[0].strip()
    settings = _get_settings(chat_id)
    settings["trigger_word"] = trigger
    await update.message.reply_text(f"Триггер обновлен: {trigger}")


async def set_max_command(update, context):
    if not _is_allowed_user(update.effective_user.id if update.effective_user else None):
        if update.effective_chat.type == ChatType.PRIVATE:
            await update.message.reply_text("Доступ ограничен.")
        return
    chat_id = update.effective_chat.id
    text = _get_command_text(update.message.text)
    if not text:
        settings = _get_settings(chat_id)
        _set_pending(settings, "set_max", update.effective_user.id)
        await update.message.reply_text(
            "Введи лимит ответа в токенах (например 512).",
            reply_markup=_cancel_keyboard(),
        )
        return
    try:
        value = int(text)
    except ValueError:
        await update.message.reply_text("Нужно число токенов, например: /setmax 512")
        return
    if value < 16:
        await update.message.reply_text("Минимум 16 токенов.")
        return
    if value >= CONTEXT_LIMIT_TOKENS:
        await update.message.reply_text("Слишком большое значение для контекста.")
        return
    settings = _get_settings(chat_id)
    settings["max_tokens"] = value
    await update.message.reply_text(f"Лимит ответа обновлен: {value} токенов.")


async def reset_settings_command(update, context):
    if not _is_allowed_user(update.effective_user.id if update.effective_user else None):
        if update.effective_chat.type == ChatType.PRIVATE:
            await update.message.reply_text("Доступ ограничен.")
        return
    chat_id = update.effective_chat.id
    _reset_settings(chat_id)
    await update.message.reply_text("Настройки сброшены к значениям по умолчанию.")


async def cancel_command(update, context):
    if not _is_allowed_user(update.effective_user.id if update.effective_user else None):
        if update.effective_chat.type == ChatType.PRIVATE:
            await update.message.reply_text("Доступ ограничен.")
        return
    chat_id = update.effective_chat.id
    settings = _get_settings(chat_id)
    _clear_pending(settings)
    await update.message.reply_text("Отменено.")


async def settings_button(update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    chat_id = query.message.chat_id
    settings = _get_settings(chat_id)
    user_id = query.from_user.id if query.from_user else None
    if not _is_allowed_user(user_id):
        await query.answer("Доступ ограничен.", show_alert=True)
        return
    data = query.data or ""

    if data == "show_settings":
        await query.message.reply_text(
            _format_settings(settings),
            reply_markup=_settings_keyboard(),
        )
        return
    if data == "reset_settings":
        _reset_settings(chat_id)
        await query.message.reply_text(
            "Настройки сброшены к значениям по умолчанию.",
            reply_markup=_settings_keyboard(),
        )
        return
    if data == "clear_mood":
        settings["mood"] = ""
        await query.message.reply_text("Настроение очищено.")
        return
    if data == "clear_prompt":
        settings["extra_prompt"] = ""
        await query.message.reply_text("Дополнительный промпт очищен.")
        return
    if data == "cancel":
        _clear_pending(settings)
        await query.message.reply_text("Отменено.")
        return
    if data in {"set_mood", "set_prompt", "set_max", "set_trigger"}:
        _set_pending(settings, data, user_id)
        prompts = {
            "set_mood": "Введи настроение для ответов.",
            "set_prompt": "Введи дополнительный системный промпт.",
            "set_max": "Введи лимит ответа в токенах (например 512).",
            "set_trigger": "Введи новое слово-триггер.",
        }
        await query.message.reply_text(
            prompts.get(data, "Введи значение."),
            reply_markup=_cancel_keyboard(),
        )
        return


async def _post_init(application):
    commands = [
        BotCommand("settings", "Настройки бота"),
        BotCommand("reset", "Сбросить контекст диалога"),
        BotCommand("help", "Краткая справка"),
        BotCommand("search", "Поиск в интернете"),
    ]
    try:
        await application.bot.set_my_commands(commands)
    except Exception as exc:
        logger.warning("Failed to set bot commands: %s", exc)


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


def _trim_oldest_history(history):
    if len(history) >= 2:
        return history[2:]
    return []


async def handle_message(update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message or not update.message.text:
        return
    # if update.message.from_user and update.message.from_user.is_bot:
    #     return

    text = update.message.text
    bot_username = context.bot.username or ""
    chat_id = update.effective_chat.id
    if not _is_allowed_user(update.effective_user.id if update.effective_user else None):
        if update.effective_chat.type == ChatType.PRIVATE:
            await update.message.reply_text("Доступ ограничен.")
        return
    settings = _get_settings(chat_id)
    pending_action = settings.get("pending_action")
    pending_user_id = settings.get("pending_user_id")
    if pending_action and update.message.from_user:
        if update.message.from_user.id == pending_user_id:
            normalized = text.strip().casefold()
            if normalized in {"/cancel", "отмена", "cancel"}:
                _clear_pending(settings)
                await update.message.reply_text("Отменено.")
                return
            success, message = _apply_pending_action(pending_action, text, settings)
            if success:
                _clear_pending(settings)
            await update.message.reply_text(message, reply_markup=_settings_keyboard())
            return
    trigger_word = settings["trigger_word"]
    if not _is_triggered(update, text, context.bot.id, bot_username, trigger_word):
        return

    prompt = _extract_prompt(text, bot_username, trigger_word)
    reply_text = _get_reply_text(update.message)
    if not prompt and reply_text:
        prompt = reply_text
        reply_text = ""
    if not prompt:
        await update.message.reply_text(
            f"Сформулируй запрос после '{trigger_word}'."
        )
        return
    web_context = ""
    web_results_text = ""
    web_query, prompt = _extract_web_query(prompt)
    if web_query:
        if not WEB_SEARCH_ENABLED:
            await update.message.reply_text(
                "Поиск отключен. Включи WEB_SEARCH_ENABLED=1 в .env."
            )
            return
        try:
            results = await search_web(web_query, limit=WEB_SEARCH_MAX_RESULTS)
        except WebSearchError as exc:
            logger.exception("Web search failed: %s", exc)
            await update.message.reply_text("Не удалось выполнить поиск.")
            return
        if not results:
            await update.message.reply_text("Ничего не нашел по этому запросу.")
            return
        web_results_text = _format_search_results(results, web_query)
        web_context = (
            "Данные из интернета (результаты поиска; проверь факты):\n"
            f"{web_results_text}"
        )

    reset_used, reset_remainder = _split_reset_request(prompt)
    if reset_used:
        CHAT_MEMORY.pop(chat_id, None)
        if not reset_remainder:
            await update.message.reply_text("Контекст очищен.")
            return
        prompt = reset_remainder
        reply_text = ""

    history = list(_get_history(chat_id))
    history, trimmed, messages = _trim_history_to_fit(
        history, prompt, reply_text, settings, web_context
    )
    if trimmed:
        CHAT_MEMORY[chat_id] = list(history)
    if _context_limit_exceeded(messages, settings["max_tokens"]):
        await update.message.reply_text(
            "Запрос слишком длинный для контекста модели. Сократи текст."
        )
        return

    attempts = 0
    max_attempts = max(1, len(history) // 2 + 1)
    while True:
        try:
            response_text = await chat_completion(
                messages,
                max_tokens=settings["max_tokens"],
                temperature=settings["temperature"],
            )
            break
        except Exception as exc:
            if history and _is_context_overflow_error(exc) and attempts < max_attempts:
                logger.info("Context overflow, trimming history for chat %s", chat_id)
                history = _trim_oldest_history(history)
                CHAT_MEMORY[chat_id] = list(history)
                messages = _build_messages(
                    history, prompt, reply_text, settings, web_context
                )
                attempts += 1
                if history:
                    continue
                attempts = max_attempts
            logger.exception("LLM request failed: %s", exc)
            await update.message.reply_text("Не удалось получить ответ от локальной LLM.")
            return

    response_text = (response_text or "").strip()
    if not response_text:
        logger.info("Empty LLM response, retrying without history for chat %s", chat_id)
        try:
            retry_system_prompt = _compose_system_prompt(settings)
            if web_context:
                retry_system_prompt = f"{retry_system_prompt}\n\n{web_context}"
            retry_messages = [{"role": "system", "content": retry_system_prompt}]
            retry_messages.append(
                {
                    "role": "user",
                    "content": f"{prompt}\n\nОтветь одним-двумя предложениями.",
                }
            )
            response_text = await chat_completion(
                retry_messages,
                max_tokens=settings["max_tokens"],
                temperature=settings["temperature"],
            )
        except Exception as exc:
            logger.exception("LLM request failed: %s", exc)
            await update.message.reply_text("Не удалось получить ответ от локальной LLM.")
            return
        response_text = (response_text or "").strip()
        if not response_text:
            if web_results_text:
                fallback = (
                    "Модель вернула пустой ответ. Вот результаты поиска:\n"
                    f"{web_results_text}"
                )
                chunks = _split_message(fallback)
                await update.message.reply_text(chunks[0])
                for chunk in chunks[1:]:
                    await context.bot.send_message(chat_id=chat_id, text=chunk)
                return
            await update.message.reply_text(
                "Модель вернула пустой ответ. Попробуй переформулировать."
            )
            return

    _append_history(chat_id, "user", prompt)
    _append_history(chat_id, "assistant", response_text)

    chunks = _split_message(response_text)
    if not chunks:
        await update.message.reply_text(
            "Модель вернула пустой ответ. Попробуй переформулировать."
        )
        return
    await update.message.reply_text(chunks[0])
    for chunk in chunks[1:]:
        await context.bot.send_message(chat_id=chat_id, text=chunk)


def main():
    application = (
        Application.builder().token(TELEGRAM_BOT_TOKEN).post_init(_post_init).build()
    )

    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("search", search_command))
    application.add_handler(CommandHandler("reset", reset_command))
    application.add_handler(CommandHandler("settings", settings_command))
    application.add_handler(CommandHandler("setmood", set_mood_command))
    application.add_handler(CommandHandler("clearmood", clear_mood_command))
    application.add_handler(CommandHandler("setprompt", set_prompt_command))
    application.add_handler(CommandHandler("clearprompt", clear_prompt_command))
    application.add_handler(CommandHandler("setmax", set_max_command))
    application.add_handler(CommandHandler("settrigger", set_trigger_command))
    application.add_handler(CommandHandler("setname", set_trigger_command))
    application.add_handler(CommandHandler("resetsettings", reset_settings_command))
    application.add_handler(CommandHandler("cancel", cancel_command))
    application.add_handler(CallbackQueryHandler(settings_button))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    application.run_polling(allowed_updates=["message", "callback_query"])


if __name__ == "__main__":
    main()
