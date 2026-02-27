import logging

from telegram import BotCommand
from telegram.constants import ChatType
from telegram.error import BadRequest
from telegram.ext import ContextTypes

from app.config import CONTEXT_LIMIT_TOKENS, WEB_SEARCH_ENABLED, WEB_SEARCH_MAX_RESULTS
from app.llm_client import LLMRequestError, chat_completion
from app.pipeline import (
    _build_flat_fallback_messages,
    _build_messages,
    _compose_system_prompt,
    _context_limit_exceeded,
    _is_context_overflow_error,
    _is_message_header_error,
    _postprocess_response,
    _trim_history_to_fit,
)
from app.search_client import WebSearchError, search_web
from app.state import (
    apply_pending_action,
    append_history,
    clear_history,
    clear_pending,
    get_history,
    get_settings,
    is_allowed_user,
    reset_settings,
    set_history,
    set_pending,
    trim_oldest_history,
)
from app.text_utils import (
    _extract_prompt,
    _extract_web_query,
    _format_search_results,
    _get_command_text,
    _get_reply_text,
    _is_triggered,
    _split_message,
    _split_reset_request,
)
from app.ui import _cancel_keyboard, _format_settings, _settings_keyboard

logger = logging.getLogger(__name__)


async def _safe_reply_text(message, text, parse_mode=None, reply_markup=None):
    try:
        return await message.reply_text(
            text, parse_mode=parse_mode, reply_markup=reply_markup
        )
    except BadRequest as exc:
        if parse_mode:
            logger.warning("Markdown parse failed, retrying without parse_mode: %s", exc)
            return await message.reply_text(text, reply_markup=reply_markup)
        raise


async def _safe_send_message(bot, chat_id, text, parse_mode=None):
    try:
        return await bot.send_message(chat_id=chat_id, text=text, parse_mode=parse_mode)
    except BadRequest as exc:
        if parse_mode:
            logger.warning("Markdown parse failed, retrying without parse_mode: %s", exc)
            return await bot.send_message(chat_id=chat_id, text=text)
        raise


async def start_command(update, context):
    if not is_allowed_user(update.effective_user.id if update.effective_user else None):
        if update.effective_chat.type == ChatType.PRIVATE:
            await update.message.reply_text("Доступ ограничен.")
        return
    chat_id = update.effective_chat.id
    trigger_word = get_settings(chat_id)["trigger_word"]
    await _safe_reply_text(
        update.message,
        f"Привет! Напиши сообщение (в группе начни с '{trigger_word}').",
        reply_markup=_settings_keyboard(),
    )


async def help_command(update, context):
    if not is_allowed_user(update.effective_user.id if update.effective_user else None):
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
    await _safe_reply_text(update.message, text, reply_markup=_settings_keyboard())


async def search_command(update, context):
    if not is_allowed_user(update.effective_user.id if update.effective_user else None):
        if update.effective_chat.type == ChatType.PRIVATE:
            await update.message.reply_text("Доступ ограничен.")
        return
    query = _get_command_text(update.message.text)
    if not query:
        await update.message.reply_text("Укажи запрос: /search <текст>")
        return
    if not WEB_SEARCH_ENABLED:
        await _safe_reply_text(
            update.message,
            "Поиск отключен. Включи WEB_SEARCH_ENABLED=1 в .env."
        )
        return
    try:
        results = await search_web(query, limit=WEB_SEARCH_MAX_RESULTS)
    except WebSearchError as exc:
        logger.exception("Web search failed: %s", exc)
        await _safe_reply_text(update.message, "Не удалось выполнить поиск.")
        return
    if not results:
        await _safe_reply_text(update.message, "Ничего не нашел по этому запросу.")
        return
    text = _format_search_results(results, query)
    chunks = _split_message(text)
    await _safe_reply_text(update.message, chunks[0])
    for chunk in chunks[1:]:
        await _safe_send_message(context.bot, update.effective_chat.id, chunk)


async def reset_command(update, context):
    if not is_allowed_user(update.effective_user.id if update.effective_user else None):
        if update.effective_chat.type == ChatType.PRIVATE:
            await update.message.reply_text("Доступ ограничен.")
        return
    chat_id = update.effective_chat.id
    clear_history(chat_id)
    await _safe_reply_text(update.message, "Контекст очищен.")


async def settings_command(update, context):
    if not is_allowed_user(update.effective_user.id if update.effective_user else None):
        if update.effective_chat.type == ChatType.PRIVATE:
            await update.message.reply_text("Доступ ограничен.")
        return
    chat_id = update.effective_chat.id
    settings = get_settings(chat_id)
    await _safe_reply_text(
        update.message,
        _format_settings(settings),
        reply_markup=_settings_keyboard(),
    )


async def set_mood_command(update, context):
    if not is_allowed_user(update.effective_user.id if update.effective_user else None):
        if update.effective_chat.type == ChatType.PRIVATE:
            await update.message.reply_text("Доступ ограничен.")
        return
    chat_id = update.effective_chat.id
    text = _get_command_text(update.message.text)
    if not text:
        settings = get_settings(chat_id)
        set_pending(settings, "set_mood", update.effective_user.id)
        await _safe_reply_text(
            update.message,
            "Введи настроение для ответов.",
            reply_markup=_cancel_keyboard(),
        )
        return
    settings = get_settings(chat_id)
    settings["mood"] = text
    await _safe_reply_text(update.message, "Настроение обновлено.")


async def clear_mood_command(update, context):
    if not is_allowed_user(update.effective_user.id if update.effective_user else None):
        if update.effective_chat.type == ChatType.PRIVATE:
            await update.message.reply_text("Доступ ограничен.")
        return
    chat_id = update.effective_chat.id
    settings = get_settings(chat_id)
    settings["mood"] = ""
    await _safe_reply_text(update.message, "Настроение очищено.")


async def set_prompt_command(update, context):
    if not is_allowed_user(update.effective_user.id if update.effective_user else None):
        if update.effective_chat.type == ChatType.PRIVATE:
            await update.message.reply_text("Доступ ограничен.")
        return
    chat_id = update.effective_chat.id
    text = _get_command_text(update.message.text)
    if not text:
        settings = get_settings(chat_id)
        set_pending(settings, "set_prompt", update.effective_user.id)
        await _safe_reply_text(
            update.message,
            "Введи дополнительный системный промпт.",
            reply_markup=_cancel_keyboard(),
        )
        return
    settings = get_settings(chat_id)
    settings["extra_prompt"] = text
    await _safe_reply_text(update.message, "Дополнительный промпт сохранен.")


async def clear_prompt_command(update, context):
    if not is_allowed_user(update.effective_user.id if update.effective_user else None):
        if update.effective_chat.type == ChatType.PRIVATE:
            await update.message.reply_text("Доступ ограничен.")
        return
    chat_id = update.effective_chat.id
    settings = get_settings(chat_id)
    settings["extra_prompt"] = ""
    await _safe_reply_text(update.message, "Дополнительный промпт очищен.")


async def set_trigger_command(update, context):
    if not is_allowed_user(update.effective_user.id if update.effective_user else None):
        if update.effective_chat.type == ChatType.PRIVATE:
            await update.message.reply_text("Доступ ограничен.")
        return
    chat_id = update.effective_chat.id
    text = _get_command_text(update.message.text)
    if not text:
        settings = get_settings(chat_id)
        set_pending(settings, "set_trigger", update.effective_user.id)
        await _safe_reply_text(
            update.message,
            "Введи новое слово-триггер.",
            reply_markup=_cancel_keyboard(),
        )
        return
    trigger = text.split()[0].strip()
    settings = get_settings(chat_id)
    settings["trigger_word"] = trigger
    await _safe_reply_text(update.message, f"Триггер обновлен: {trigger}")


async def set_max_command(update, context):
    if not is_allowed_user(update.effective_user.id if update.effective_user else None):
        if update.effective_chat.type == ChatType.PRIVATE:
            await update.message.reply_text("Доступ ограничен.")
        return
    chat_id = update.effective_chat.id
    text = _get_command_text(update.message.text)
    if not text:
        settings = get_settings(chat_id)
        set_pending(settings, "set_max", update.effective_user.id)
        await _safe_reply_text(
            update.message,
            "Введи лимит ответа в токенах (например 512).",
            reply_markup=_cancel_keyboard(),
        )
        return
    try:
        value = int(text)
    except ValueError:
        await _safe_reply_text(update.message, "Нужно число токенов, например: /setmax 512")
        return
    if value < 16:
        await _safe_reply_text(update.message, "Минимум 16 токенов.")
        return
    if value >= CONTEXT_LIMIT_TOKENS:
        await _safe_reply_text(update.message, "Слишком большое значение для контекста.")
        return
    settings = get_settings(chat_id)
    settings["max_tokens"] = value
    await _safe_reply_text(update.message, f"Лимит ответа обновлен: {value} токенов.")


async def reset_settings_command(update, context):
    if not is_allowed_user(update.effective_user.id if update.effective_user else None):
        if update.effective_chat.type == ChatType.PRIVATE:
            await update.message.reply_text("Доступ ограничен.")
        return
    chat_id = update.effective_chat.id
    reset_settings(chat_id)
    await _safe_reply_text(update.message, "Настройки сброшены к значениям по умолчанию.")


async def cancel_command(update, context):
    if not is_allowed_user(update.effective_user.id if update.effective_user else None):
        if update.effective_chat.type == ChatType.PRIVATE:
            await update.message.reply_text("Доступ ограничен.")
        return
    chat_id = update.effective_chat.id
    settings = get_settings(chat_id)
    clear_pending(settings)
    await _safe_reply_text(update.message, "Отменено.")


async def settings_button(update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    chat_id = query.message.chat_id
    settings = get_settings(chat_id)
    user_id = query.from_user.id if query.from_user else None
    if not is_allowed_user(user_id):
        await query.answer("Доступ ограничен.", show_alert=True)
        return
    data = query.data or ""

    if data == "show_settings":
        await _safe_reply_text(
            query.message,
            _format_settings(settings),
            reply_markup=_settings_keyboard(),
        )
        return
    if data == "reset_settings":
        reset_settings(chat_id)
        await _safe_reply_text(
            query.message,
            "Настройки сброшены к значениям по умолчанию.",
            reply_markup=_settings_keyboard(),
        )
        return
    if data == "clear_mood":
        settings["mood"] = ""
        await _safe_reply_text(query.message, "Настроение очищено.")
        return
    if data == "clear_prompt":
        settings["extra_prompt"] = ""
        await _safe_reply_text(query.message, "Дополнительный промпт очищен.")
        return
    if data == "cancel":
        clear_pending(settings)
        await _safe_reply_text(query.message, "Отменено.")
        return
    if data in {"set_mood", "set_prompt", "set_max", "set_trigger"}:
        set_pending(settings, data, user_id)
        prompts = {
            "set_mood": "Введи настроение для ответов.",
            "set_prompt": "Введи дополнительный системный промпт.",
            "set_max": "Введи лимит ответа в токенах (например 512).",
            "set_trigger": "Введи новое слово-триггер.",
        }
        await _safe_reply_text(
            query.message,
            prompts.get(data, "Введи значение."),
            reply_markup=_cancel_keyboard(),
        )
        return


async def post_init(application):
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


async def handle_message(update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message or not update.message.text:
        return
    # if update.message.from_user and update.message.from_user.is_bot:
    #     return

    text = update.message.text
    bot_username = context.bot.username or ""
    chat_id = update.effective_chat.id
    if not is_allowed_user(update.effective_user.id if update.effective_user else None):
        if update.effective_chat.type == ChatType.PRIVATE:
            await _safe_reply_text(update.message, "Доступ ограничен.")
        return
    settings = get_settings(chat_id)
    pending_action = settings.get("pending_action")
    pending_user_id = settings.get("pending_user_id")
    if pending_action and update.message.from_user:
        if update.message.from_user.id == pending_user_id:
            normalized = text.strip().casefold()
            if normalized in {"/cancel", "отмена", "cancel"}:
                clear_pending(settings)
                await _safe_reply_text(update.message, "Отменено.")
                return
            success, message = apply_pending_action(pending_action, text, settings)
            if success:
                clear_pending(settings)
            await _safe_reply_text(
                update.message, message, reply_markup=_settings_keyboard()
            )
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
        await _safe_reply_text(
            update.message,
            f"Сформулируй запрос после '{trigger_word}'."
        )
        return
    web_context = ""
    web_results_text = ""
    web_query, prompt = _extract_web_query(prompt)
    if web_query:
        if not WEB_SEARCH_ENABLED:
            await _safe_reply_text(
                update.message,
                "Поиск отключен. Включи WEB_SEARCH_ENABLED=1 в .env."
            )
            return
        try:
            results = await search_web(web_query, limit=WEB_SEARCH_MAX_RESULTS)
        except WebSearchError as exc:
            logger.exception("Web search failed: %s", exc)
            await _safe_reply_text(update.message, "Не удалось выполнить поиск.")
            return
        if not results:
            await _safe_reply_text(update.message, "Ничего не нашел по этому запросу.")
            return
        web_results_text = _format_search_results(results, web_query)
        web_context = (
            "Данные из интернета (результаты поиска; проверь факты):\n"
            f"{web_results_text}"
        )

    reset_used, reset_remainder = _split_reset_request(prompt)
    if reset_used:
        clear_history(chat_id)
        if not reset_remainder:
            await _safe_reply_text(update.message, "Контекст очищен.")
            return
        prompt = reset_remainder
        reply_text = ""

    history = list(get_history(chat_id))
    history, trimmed, messages = _trim_history_to_fit(
        history, prompt, reply_text, settings, web_context
    )
    if trimmed:
        set_history(chat_id, history)
    if _context_limit_exceeded(messages, settings["max_tokens"]):
        await _safe_reply_text(
            update.message,
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
        except LLMRequestError as exc:
            if history and _is_context_overflow_error(exc) and attempts < max_attempts:
                logger.info("Context overflow, trimming history for chat %s", chat_id)
                history = trim_oldest_history(history)
                set_history(chat_id, history)
                messages = _build_messages(
                    history, prompt, reply_text, settings, web_context
                )
                attempts += 1
                if history:
                    continue
                attempts = max_attempts
            if _is_message_header_error(exc):
                logger.warning(
                    "Chat template error, retrying with minimal prompt for chat %s",
                    chat_id,
                )
                try:
                    response_text = await chat_completion(
                        _build_flat_fallback_messages(
                            history, prompt, reply_text, settings, web_context
                        ),
                        max_tokens=settings["max_tokens"],
                        temperature=settings["temperature"],
                    )
                    break
                except Exception as fallback_exc:
                    logger.exception("Fallback LLM request failed: %s", fallback_exc)
                    await _safe_reply_text(
                        update.message,
                        "Не удалось получить ответ от локальной LLM."
                    )
                    return
            logger.exception("LLM request failed: %s", exc)
            await _safe_reply_text(update.message, "Не удалось получить ответ от локальной LLM.")
            return
        except Exception as exc:
            logger.exception("LLM request failed: %s", exc)
            await _safe_reply_text(update.message, "Не удалось получить ответ от локальной LLM.")
            return

    response_text = (response_text or "").strip()
    parse_mode = None
    if response_text:
        response_text, parse_mode = await _postprocess_response(
            prompt, response_text, settings
        )
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
            await _safe_reply_text(update.message, "Не удалось получить ответ от локальной LLM.")
            return
        response_text = (response_text or "").strip()
        if response_text:
            response_text, parse_mode = await _postprocess_response(
                prompt, response_text, settings
            )
        if not response_text:
            if web_results_text:
                fallback = (
                    "Модель вернула пустой ответ. Вот результаты поиска:\n"
                    f"{web_results_text}"
                )
                chunks = _split_message(fallback)
                await _safe_reply_text(update.message, chunks[0])
                for chunk in chunks[1:]:
                    await _safe_send_message(context.bot, chat_id, chunk)
                return
            await _safe_reply_text(
                update.message,
                "Модель вернула пустой ответ. Попробуй переформулировать."
            )
            return

    append_history(chat_id, "user", prompt)
    append_history(chat_id, "assistant", response_text)

    chunks = _split_message(response_text)
    if parse_mode and len(chunks) > 1:
        parse_mode = None
    if not chunks:
        await _safe_reply_text(
            update.message,
            "Модель вернула пустой ответ. Попробуй переформулировать."
        )
        return
    await _safe_reply_text(update.message, chunks[0], parse_mode=parse_mode)
    for chunk in chunks[1:]:
        await _safe_send_message(context.bot, chat_id, chunk, parse_mode=parse_mode)
