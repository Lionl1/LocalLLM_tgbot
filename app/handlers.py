import logging
import random
import os
import tempfile
import json
import asyncio
from io import BytesIO

from telegram import (
    BotCommand,
    BotCommandScopeDefault,
    BotCommandScopeAllPrivateChats,
    BotCommandScopeAllChatAdministrators,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    ForceReply,
    ReplyKeyboardRemove,
)
from telegram.constants import ChatAction, ChatType
from telegram.error import BadRequest
from telegram.ext import ContextTypes

from app.config import (
    CONTEXT_LIMIT_TOKENS,
    IMAGE_GENERATION_ENABLED,
    RANDOM_PARTICIPATION_PROBABILITY,
    RANDOM_QUESTION_PROBABILITY,
    TOKEN_CHAR_RATIO,
    WEB_SEARCH_ENABLED,
    WEB_SEARCH_MAX_RESULTS,
)
from app.llm_client import chat_completion
from app.llm_service import (
    generate_random_question,
    process_chat_request,
    summarize_search_results,
    format_transcribed_text,
    summarize_transcription,
)
from app.search_client import WebSearchError, search_web
from app.image_client import ImageGenerationError, generate_image
from app.audio_client import transcribe_audio
from app.tts_client import generate_speech
from app.pipeline import _strip_markdown_syntax
from app.state import (
    apply_pending_action,
    append_history,
    clear_history,
    clear_pending,
    get_settings,
    get_random_seen_user,
    is_allowed_user,
    load_persisted_chat_settings,
    mark_user_seen,
    reset_settings,
    persist_settings,
    set_pending,
    set_raw_transcription,
    get_raw_transcription,
    get_all_known_groups,
)
from app.text_utils import (
    _estimate_tokens,
    _extract_prompt,
    _format_search_results,
    _get_command_text,
    _get_reply_text,
    _is_triggered,
    _split_message,
    _split_reset_request,
    detect_transcription_request,
)
from app.ui import _cancel_keyboard, _format_settings, _settings_keyboard

logger = logging.getLogger(__name__)

# --- Function Calling Tools Definition ---
# Эти определения описывают LLM, какие функции доступны и как их вызывать.
_FUNCTION_CALLING_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "generate_image",
            "description": "Генерирует изображение по текстовому описанию. Используй, когда пользователь просит нарисовать, создать картинку, сгенерировать изображение и т.п.",
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "Подробное описание изображения, которое нужно сгенерировать, на русском языке. Например: 'Кот в скафандре летит в космосе, стиль киберпанк, неоновые цвета'",
                    }
                },
                "required": ["prompt"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Ищет информацию в интернете по заданному запросу. Используй, когда пользователь просит найти, узнать, поискать что-либо в интернете или если для ответа требуется актуальная информация.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Поисковый запрос для интернета, на русском языке. Например: 'новости технологий за сегодня' или 'рецепт борща'",
                    }
                },
                "required": ["query"],
            },
        },
    },
]
# --- End Function Calling Tools Definition ---


_SETTINGS_ADMIN_ONLY_TEXT = "Только администратор может менять настройки."


async def _is_group_admin(bot, chat_id, user_id):
    if user_id is None:
        return False
        
    settings = get_settings(chat_id)
    if settings.get("added_by") == user_id:
        return True
        
    try:
        member = await bot.get_chat_member(chat_id, user_id)
    except Exception:
        return False
    return member.status in {"administrator", "creator"}


async def _require_settings_admin(update, context):
    chat = update.effective_chat
    if not chat:
        return False
    if chat.type == ChatType.PRIVATE:
        return True
    user_id = update.effective_user.id if update.effective_user else None
    if await _is_group_admin(context.bot, chat.id, user_id):
        return True
    if update.message:
        await _safe_reply_text(update.message, _SETTINGS_ADMIN_ONLY_TEXT)
    return False


async def _require_settings_admin_query(query, context):
    chat = query.message.chat if query.message else None
    if not chat:
        return False
    if chat.type == ChatType.PRIVATE:
        return True
    user_id = query.from_user.id if query.from_user else None
    if await _is_group_admin(context.bot, chat.id, user_id):
        return True
    await query.answer(_SETTINGS_ADMIN_ONLY_TEXT, show_alert=True)
    return False


async def _ensure_update_allowed(update, context):
    user_id = update.effective_user.id if update.effective_user else None
    if is_allowed_user(user_id):
        return True
    if update.message:
        await update.message.reply_text("Доступ ограничен.")
    return False


async def _ensure_query_allowed(query, context):
    user_id = query.from_user.id if query.from_user else None
    if is_allowed_user(user_id):
        return True
    await query.answer("Доступ ограничен.", show_alert=True)
    return False


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


async def _get_user_manageable_chats(bot, user_id, private_chat_id):
    chats = []
    pm_settings = get_settings(private_chat_id)
    chats.append({
        "id": private_chat_id,
        "title": "👤 Личные сообщения",
        "settings": pm_settings
    })
    for gid in get_all_known_groups():
        if await _is_group_admin(bot, gid, user_id):
            g_settings = get_settings(gid)
            title = g_settings.get("chat_title") or f"Группа {gid}"
            chats.append({
                "id": gid,
                "title": f"👥 {title}",
                "settings": g_settings
            })
    return chats

async def _get_admin_settings_keyboard(bot, user_id, private_chat_id, current_chat_id):
    chats = await _get_user_manageable_chats(bot, user_id, private_chat_id)
    return _settings_keyboard(chats, current_chat_id)

async def _generate_and_send_image(message, prompt):
    status_msg = None
    try:
        status_msg = await _safe_reply_text(message, "⏳ Запрос добавлен в очередь на генерацию. Пожалуйста, подождите...")
        await message.chat.send_action(action=ChatAction.UPLOAD_PHOTO)
        image_bytes = await generate_image(prompt)
        if not image_bytes:
            raise ImageGenerationError("Получен пустой файл от сервиса.")
            
    except Exception as exc:
        # Ловим любые ошибки, так как функция теперь работает в фоне и мы не должны ронять таску
        logger.exception("Image generation failed: %s", exc)
        error_msg = "Не удалось сгенерировать изображение."
        if "500" in str(exc) or "503" in str(exc):
            error_msg += " Сервис генерации временно перегружен."
        if status_msg:
            try:
                await status_msg.edit_text(error_msg)
            except Exception:
                pass
        else:
            try:
                await _safe_reply_text(message, error_msg)
            except Exception:
                pass
        return False
    stream = BytesIO(image_bytes)
    stream.name = "generated.png"
    stream.seek(0)
    caption = f"Запрос: {prompt}"
    if len(caption) > 180:
        caption = f"Запрос: {prompt[:177]}..."
    
    try:
        await message.reply_photo(
            photo=stream,
            caption=caption,
            reply_to_message_id=message.message_id,
            read_timeout=60,
            write_timeout=60,
            connect_timeout=60,
        )
    except BadRequest as exc:
        # Если Telegram не смог обработать картинку, пробуем отправить как файл (документ)
        # Это поможет диагностировать проблему (например, если пришел текстовый файл с ошибкой)
        logger.warning("reply_photo failed (%s), trying reply_document...", exc)
        stream.seek(0)
        await message.reply_document(
            document=stream,
            caption=caption + " (отправлено файлом из-за ошибки обработки)",
            reply_to_message_id=message.message_id,
            read_timeout=60,
            write_timeout=60,
            connect_timeout=60,
        )

    if status_msg:
        try:
            await status_msg.delete()
        except Exception:
            pass

    return True


async def _execute_search_from_tool(update, context, query):
    """Executes web search and summarizes results, called by tool or /search command."""
    if not WEB_SEARCH_ENABLED:
        await _safe_reply_text(
            update.message,
            "Поиск отключен. Включи WEB_SEARCH_ENABLED=1 в .env."
        )
        return
    try:
        await update.message.chat.send_action(action=ChatAction.TYPING)
        results = await search_web(query, limit=WEB_SEARCH_MAX_RESULTS)
    except WebSearchError as exc:
        logger.exception("Web search failed: %s", exc)
        await _safe_reply_text(update.message, "Не удалось выполнить поиск.")
        return
    if not results:
        await _safe_reply_text(update.message, "Ничего не нашел по этому запросу.")
        return
    text = _format_search_results(results, query)
    
    chat_id = update.effective_chat.id
    settings = get_settings(chat_id)
    
    safe_tokens = max(500, CONTEXT_LIMIT_TOKENS - settings.get("max_tokens", 512) - 500)
    text = _truncate_web_text(text, safe_tokens)
    
    await update.message.chat.send_action(action=ChatAction.TYPING)
    response_text = await summarize_search_results(query, text, settings)
        
    if not response_text:
        response_text = text
        
    # Логируем это как действие пользователя для истории
    append_history(chat_id, "user", f"Поиск по запросу: {query}")
    append_history(chat_id, "assistant", response_text)
        
    chunks = _split_message(response_text)
    await _safe_reply_text(update.message, chunks[0])
    for chunk in chunks[1:]:
        await _safe_send_message(context.bot, update.effective_chat.id, chunk)



def _truncate_web_text(text, max_tokens):
    if not text or _estimate_tokens(text) <= max_tokens:
        return text
    char_limit = max_tokens * TOKEN_CHAR_RATIO
    truncated = text[:char_limit]
    cut_at = max(truncated.rfind("\n"), truncated.rfind(" "))
    if cut_at > char_limit // 2:
        truncated = truncated[:cut_at]
    return truncated.strip() + "\n\n... [результаты обрезаны из-за лимита контекста]"


async def start_command(update, context):
    if not await _ensure_update_allowed(update, context):
        return
    chat_id = update.effective_chat.id

    # Обработка Deep Linking: переход из группы в ЛС для настройки
    if context.args and context.args[0].startswith("set_"):
        try:
            target_chat_id = int(context.args[0][4:])
        except ValueError:
            target_chat_id = chat_id
            
        if target_chat_id != chat_id:
            if not await _is_group_admin(context.bot, target_chat_id, update.effective_user.id):
                await _safe_reply_text(update.message, "У вас нет прав администратора в этой группе.")
                return
                
        settings = get_settings(target_chat_id)
        title = settings.get("chat_title") or str(target_chat_id)
        keyboard = await _get_admin_settings_keyboard(context.bot, update.effective_user.id, chat_id, target_chat_id)
        await _safe_reply_text(
            update.message,
            f"✅ Вы настраиваете чат: {title}\n\n{_format_settings(settings)}",
            reply_markup=keyboard,
        )
        return

    settings = get_settings(chat_id)
    trigger_word = settings["trigger_word"]
    keyboard = await _get_admin_settings_keyboard(context.bot, update.effective_user.id, chat_id, chat_id) if update.effective_chat.type == ChatType.PRIVATE else None
    await _safe_reply_text(update.message, f"Привет! Напиши сообщение (в группе начни с '{trigger_word}').", reply_markup=keyboard)


async def help_command(update, context):
    if not await _ensure_update_allowed(update, context):
        return
    chat_id = update.effective_chat.id
    settings = get_settings(chat_id)
    text = (
        "Команды:\n"
        "/settings — открыть настройки\n"
        "/reset — очистить контекст\n"
        "/search <запрос> — поиск в интернете\n"
        "/image <описание> — генерация картинки\n"
        "/setmood <текст> — задать настроение\n"
        "/setprompt <текст> — доп. системный промпт\n"
        "/setmax <число> — лимит ответа в токенах\n"
        "/settrigger <имя> — слово-триггер\n"
        "/resetsettings — сбросить настройки\n"
        "/cancel — отменить ввод значения"
    )
    keyboard = None
    if update.effective_chat.type == ChatType.PRIVATE:
        keyboard = await _get_admin_settings_keyboard(context.bot, update.effective_user.id, chat_id, chat_id)
    await _safe_reply_text(update.message, text, reply_markup=keyboard)


async def search_command(update, context):
    if not await _ensure_update_allowed(update, context):
        return
    query = _get_command_text(update.message.text) # Используется для парсинга команды /search
    if not query:
        await update.message.reply_text("Укажи запрос: /search <текст>")
        return
    if not WEB_SEARCH_ENABLED:
        await _safe_reply_text(
            update.message,
            "Поиск отключен. Включи WEB_SEARCH_ENABLED=1 в .env."
        )
        return
    # Делегируем выполнение общей логике поиска
    await _execute_search_from_tool(update, context, query)


async def image_command(update, context):
    if not await _ensure_update_allowed(update, context):
        return
    if not IMAGE_GENERATION_ENABLED:
        await _safe_reply_text(update.message, "Генерация изображений отключена.")
        return
    prompt = _get_command_text(update.message.text) # Используется для парсинга команды /image
    if not prompt:
        await _safe_reply_text(
            update.message, "Опиши, что нарисовать: /image <описание>."
        )
        return
    # Запускаем в фоне, чтобы бот мог сразу отвечать на другие сообщения
    asyncio.create_task(_generate_and_send_image(update.message, prompt))


async def reset_command(update, context):
    if not await _ensure_update_allowed(update, context):
        return
    chat_id = update.effective_chat.id
    clear_history(chat_id)
    await _safe_reply_text(update.message, "Контекст очищен.")


async def settings_command(update, context):
    if not await _ensure_update_allowed(update, context):
        return
    if not await _require_settings_admin(update, context):
        return
    chat_id = update.effective_chat.id
    
    # Telegram запрещает WebApp (sendData) из обычных кнопок в группах.
    if update.effective_chat.type != ChatType.PRIVATE:
        bot_username = context.bot.username
        url = f"https://t.me/{bot_username}?start=set_{chat_id}"
        keyboard = InlineKeyboardMarkup([[InlineKeyboardButton("⚙️ Настроить в ЛС", url=url)]])
        await _safe_reply_text(update.message, "Из-за ограничений Telegram сохранять настройки через Mini App можно только в личных сообщениях с ботом.\n\nНажмите кнопку ниже:", reply_markup=keyboard)
        
        settings = get_settings(chat_id)
        if update.effective_chat.title and settings.get("chat_title") != update.effective_chat.title:
            settings["chat_title"] = update.effective_chat.title
        return
        
    settings = get_settings(chat_id)
    keyboard = await _get_admin_settings_keyboard(context.bot, update.effective_user.id, chat_id, chat_id)
    
    await _safe_reply_text(
        update.message,
        f"Настройки для: Личные сообщения\n\n{_format_settings(settings)}",
        reply_markup=keyboard,
    )


async def set_mood_command(update, context):
    if not await _ensure_update_allowed(update, context):
        return
    if not await _require_settings_admin(update, context):
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
    await persist_settings()
    await _safe_reply_text(update.message, "Настроение обновлено.")


async def clear_mood_command(update, context):
    if not await _ensure_update_allowed(update, context):
        return
    if not await _require_settings_admin(update, context):
        return
    chat_id = update.effective_chat.id
    settings = get_settings(chat_id)
    settings["mood"] = ""
    await persist_settings()
    await _safe_reply_text(update.message, "Настроение очищено.")


async def set_prompt_command(update, context):
    if not await _ensure_update_allowed(update, context):
        return
    if not await _require_settings_admin(update, context):
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
    await persist_settings()
    await _safe_reply_text(update.message, "Дополнительный промпт сохранен.")


async def clear_prompt_command(update, context):
    if not await _ensure_update_allowed(update, context):
        return
    if not await _require_settings_admin(update, context):
        return
    chat_id = update.effective_chat.id
    settings = get_settings(chat_id)
    settings["extra_prompt"] = ""
    await persist_settings()
    await _safe_reply_text(update.message, "Дополнительный промпт очищен.")


async def set_trigger_command(update, context):
    if not await _ensure_update_allowed(update, context):
        return
    if not await _require_settings_admin(update, context):
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
    await persist_settings()
    await _safe_reply_text(update.message, f"Триггер обновлен: {trigger}")


async def set_max_command(update, context):
    if not await _ensure_update_allowed(update, context):
        return
    if not await _require_settings_admin(update, context):
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
    await persist_settings()
    await _safe_reply_text(update.message, f"Лимит ответа обновлен: {value} токенов.")


async def toggle_syntax_command(update, context):
    if not await _ensure_update_allowed(update, context):
        return
    if not await _require_settings_admin(update, context):
        return
    chat_id = update.effective_chat.id
    settings = get_settings(chat_id)
    settings["check_syntax"] = not settings.get("check_syntax", False)
    await persist_settings()
    state = "включена" if settings["check_syntax"] else "выключена"
    await _safe_reply_text(update.message, f"Проверка синтаксиса {state}.")


async def reset_settings_command(update, context):
    if not await _ensure_update_allowed(update, context):
        return
    if not await _require_settings_admin(update, context):
        return
    chat_id = update.effective_chat.id
    await reset_settings(chat_id)
    await _safe_reply_text(update.message, "Настройки сброшены к значениям по умолчанию.")


async def cancel_command(update, context):
    if not await _ensure_update_allowed(update, context):
        return
    if not await _require_settings_admin(update, context):
        return
    chat_id = update.effective_chat.id
    settings = get_settings(chat_id)
    clear_pending(settings)
    await _safe_reply_text(update.message, "Отменено.")


async def settings_button(update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    if not await _ensure_query_allowed(query, context):
        return
    await query.answer()
    chat_id = query.message.chat_id
    settings = get_settings(chat_id)
    user_id = query.from_user.id if query.from_user else None
    data = query.data or ""

    keyboard = None
    if query.message.chat.type == ChatType.PRIVATE:
        keyboard = await _get_admin_settings_keyboard(context.bot, user_id, chat_id, chat_id)

    if data == "show_settings":
        await _safe_reply_text(
            query.message,
            _format_settings(settings),
            reply_markup=keyboard,
        )
        return
    if not await _require_settings_admin_query(query, context):
        return
    if data == "reset_settings":
        await reset_settings(chat_id)
        settings = get_settings(chat_id)
        await _safe_reply_text(
            query.message,
            "Настройки сброшены к значениям по умолчанию.",
            reply_markup=keyboard,
        )
        return
    if data == "list_groups":
        await query.edit_message_text("⏳ Ищу группы, в которых вы администратор...")
        admin_groups = []
        for gid in get_all_known_groups():
            if await _is_group_admin(context.bot, gid, user_id):
                g_settings = get_settings(gid)
                title = g_settings.get("chat_title") or str(gid)
                admin_groups.append((gid, title))
        
        if not admin_groups:
            await query.edit_message_text("Вы не являетесь администратором ни в одной известной мне группе.")
            return
            
        buttons = [[InlineKeyboardButton(f"Настроить: {title}", callback_data=f"select_group_{gid}")] for gid, title in admin_groups]
        buttons.append([InlineKeyboardButton("👤 Настроить ЛС (этот чат)", callback_data=f"select_group_{chat_id}")])
        
        await query.edit_message_text("Выберите чат для настройки:", reply_markup=InlineKeyboardMarkup(buttons))
        return
    if data.startswith("select_group_"):
        try:
            target_id = int(data.split("select_group_")[1])
        except ValueError:
            target_id = chat_id
            
        set_config_target(user_id, target_id)
        settings = get_settings(target_id)
        title = settings.get("chat_title") or str(target_id)
        display_name = title if target_id != chat_id else "Личные сообщения"
        
        await query.message.delete()
        await context.bot.send_message(
            chat_id=chat_id,
            text=f"✅ Выбран чат: {display_name}\n\n{_format_settings(settings)}",
            reply_markup=_settings_keyboard(settings)
        )
        return
    if data == "toggle_voice":
        current = settings.get("voice_response", False)
        if not current:
            settings["voice_response"] = "kseniya"
            state = "включен (Ксения)"
        elif current in ("female", "kseniya", "xenia"):
            settings["voice_response"] = "aidar"
            state = "включен (Айдар)"
        elif current in ("male", "aidar"):
            settings["voice_response"] = "eugene"
            state = "включен (Евгений)"
        elif current == "eugene":
            settings["voice_response"] = "baya"
            state = "включен (Байя)"
        else:
            settings["voice_response"] = False
            state = "выключен"
        await persist_settings()
        await _safe_reply_text(
            query.message,
            f"Голосовой ответ {state}.",
            reply_markup=keyboard,
        )
        return
    if data == "clear_mood":
        settings["mood"] = ""
        await persist_settings()
        await _safe_reply_text(query.message, "Настроение очищено.")
        return
    if data == "clear_prompt":
        settings["extra_prompt"] = ""
        await persist_settings()
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
        text_prompt = prompts.get(data, "Введи значение.") + "\n\n(Для отмены напиши «отмена»)"
        await context.bot.send_message(
            chat_id=chat_id,
            text=text_prompt,
            reply_markup=ForceReply(),
        )
        return
    if data == "show_raw_transcription":
        raw_text = get_raw_transcription(chat_id)
        raw_out = f"📝 Оригинал транскрибации:\n{raw_text}"
        chunks = _split_message(raw_out)
        for chunk in chunks:
            await _safe_send_message(context.bot, chat_id, chunk)
        return


async def web_app_data_handler(update, context: ContextTypes.DEFAULT_TYPE):
    """Обрабатывает данные, присланные из Mini App (Web App)"""
    if not await _ensure_update_allowed(update, context):
        return
        
    data = update.message.web_app_data.data
    user_id = update.effective_user.id
    
    try:
        payload = json.loads(data)
        target_chat_id = int(payload.get("chat_id", update.effective_chat.id))
        
        if target_chat_id != update.effective_chat.id:
            if not await _is_group_admin(context.bot, target_chat_id, user_id):
                await _safe_reply_text(update.message, "❌ У вас нет прав администратора в настраиваемой группе.")
                return
        elif not await _require_settings_admin(update, context):
            return
            
        if payload.get("action") == "reset":
            await reset_settings(target_chat_id)
            settings = get_settings(target_chat_id)
            title = settings.get("chat_title") or str(target_chat_id)
            display_name = title if target_chat_id != update.effective_chat.id else "Личные сообщения"
            await _safe_reply_text(
                update.message, 
                f"🔄 Настройки для: {display_name} сброшены к значениям по умолчанию!\n\n{_format_settings(settings)}",
                reply_markup=ReplyKeyboardRemove()
            )
            return
            
        settings = get_settings(target_chat_id)
        
        if "mood" in payload:
            settings["mood"] = payload["mood"].strip()
        if "extra_prompt" in payload:
            settings["extra_prompt"] = payload["extra_prompt"].strip()
        if "trigger_word" in payload:
            settings["trigger_word"] = payload["trigger_word"].strip()
        if "max_tokens" in payload:
            settings["max_tokens"] = int(payload["max_tokens"])
        if "voice_response" in payload:
            settings["voice_response"] = payload.get("voice_response", False)
        if "check_syntax" in payload:
            settings["check_syntax"] = bool(payload.get("check_syntax", False))
            
        await persist_settings()
        
        title = settings.get("chat_title") or str(target_chat_id)
        display_name = title if target_chat_id != update.effective_chat.id else "Личные сообщения"
        
        await _safe_reply_text(
            update.message, 
            f"✅ Настройки для: {display_name} успешно обновлены!\n\n{_format_settings(settings)}",
            reply_markup=ReplyKeyboardRemove()
        )
        
    except Exception as exc:
        logger.error("Failed to parse WebApp data: %s", exc)
        await _safe_reply_text(update.message, "❌ Ошибка при сохранении настроек.")


async def post_init(application):
    await load_persisted_chat_settings()
    
    base_commands = [
        BotCommand("reset", "Сбросить контекст диалога"),
        BotCommand("image", "Сгенерировать картинку"),
        BotCommand("help", "Краткая справка"),
        BotCommand("search", "Поиск в интернете"),
    ]
    
    admin_commands = [
        BotCommand("settings", "Настройки бота"),
    ] + base_commands

    try:
        # Обычные пользователи в группах увидят только базу
        await application.bot.set_my_commands(base_commands, scope=BotCommandScopeDefault())
        # В личных сообщениях будут все команды
        await application.bot.set_my_commands(admin_commands, scope=BotCommandScopeAllPrivateChats())
        # Администраторы групп увидят все команды
        await application.bot.set_my_commands(admin_commands, scope=BotCommandScopeAllChatAdministrators())
    except Exception as exc:
        logger.warning("Failed to set bot commands: %s", exc)


async def chat_member_handler(update, context: ContextTypes.DEFAULT_TYPE):
    """Отслеживает добавление бота в группы, чтобы назначить права настройщика."""
    result = update.my_chat_member
    if not result:
        return
        
    new_status = result.new_chat_member.status
    old_status = result.old_chat_member.status
    
    # Если бота только что добавили в чат
    if new_status in {"member", "administrator"} and old_status not in {"member", "administrator"}:
        chat_id = result.chat.id
        user_id = result.from_user.id
        
        settings = get_settings(chat_id)
        settings["added_by"] = user_id
        if result.chat.title:
            settings["chat_title"] = result.chat.title
        
        await persist_settings()
        logger.info("Bot added to group %s by user %s", chat_id, user_id)


async def handle_message(update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message:
        return
        
    audio_obj = update.message.voice or update.message.audio or update.message.video_note
    text = update.message.text or update.message.caption or ""

    if not text and not audio_obj:
        return

    if audio_obj and update.message.audio:
        if not detect_transcription_request(text):
            audio_obj = None

    if audio_obj:
        duration = getattr(audio_obj, "duration", 0)
        if not await _ensure_update_allowed(update, context):
            return
        await update.message.chat.send_action(action=ChatAction.TYPING)
        try:
            voice_file = await context.bot.get_file(audio_obj.file_id)
            with tempfile.NamedTemporaryFile(suffix=".tmp", delete=False) as temp_audio:
                temp_path = temp_audio.name
            
            try:
                await voice_file.download_to_drive(temp_path)
                status_msg = await _safe_reply_text(update.message, "⏳ Распознаю аудио...")
                
                transcribed_text = await transcribe_audio(temp_path)
                if not transcribed_text:
                    await status_msg.edit_text("❌ Не удалось распознать аудио.")
                    return
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)

            set_raw_transcription(update.effective_chat.id, transcribed_text)

            await status_msg.edit_text("⏳ Улучшаю читаемость текста...")
            settings = get_settings(update.effective_chat.id)
            formatted_text = await format_transcribed_text(transcribed_text, settings)

            summary_text = ""
            if duration > 120:
                await status_msg.edit_text("⏳ Формирую выжимку (summary)...")
                summary_text = await summarize_transcription(formatted_text, settings)

            if text:
                text = f"{text}\n\n[Распознанный текст]: {formatted_text}"
            else:
                text = formatted_text
                
            out_text = f"🎤 Распознано:\n{formatted_text}"
            if summary_text:
                out_text += f"\n\n📋 Главные тезисы:\n{summary_text}"
            chunks = _split_message(out_text)
            keyboard = InlineKeyboardMarkup([[InlineKeyboardButton("Показать оригинал", callback_data="show_raw_transcription")]])
            
            await status_msg.edit_text(chunks[0], reply_markup=keyboard if len(chunks) == 1 else None)
            for i, chunk in enumerate(chunks[1:], start=1):
                is_last = (i == len(chunks) - 1)
                await _safe_reply_text(update.message, chunk, reply_markup=keyboard if is_last else None)
        except Exception as exc:
            logger.exception("Voice processing failed: %s", exc)
            await _safe_reply_text(update.message, "Ошибка обработки голосового сообщения.")
            return

    bot_username = context.bot.username or ""
    chat_id = update.effective_chat.id
    if not await _ensure_update_allowed(update, context):
        return
        
    user = update.effective_user
    if user and not user.is_bot:
        mark_user_seen(chat_id, user.id, user.username, user.first_name)
        
    settings = get_settings(chat_id)
    if update.effective_chat.type in {ChatType.GROUP, ChatType.SUPERGROUP} and update.effective_chat.title:
        if settings.get("chat_title") != update.effective_chat.title:
            settings["chat_title"] = update.effective_chat.title
            
    pending_action = settings.get("pending_action")
    pending_user_id = settings.get("pending_user_id")
    if pending_action and update.message.from_user:
        if update.message.from_user.id == pending_user_id:
            normalized = text.strip().casefold()
            if normalized in {"/cancel", "отмена", "cancel"}:
                clear_pending(settings)
                await _safe_reply_text(update.message, "Отменено.")
                return
            success, message = await apply_pending_action(pending_action, text, settings)
            if success:
                clear_pending(settings)
                keyboard = None
                if update.effective_chat.type == ChatType.PRIVATE:
                    keyboard = await _get_admin_settings_keyboard(context.bot, update.effective_user.id, chat_id, chat_id)
            await _safe_reply_text(
                update.message, message, reply_markup=keyboard
            )
            return

    trigger_word = settings["trigger_word"]
    if not _is_triggered(update, text, context.bot.id, bot_username, trigger_word):
        if update.effective_chat.type in {ChatType.GROUP, ChatType.SUPERGROUP}:
            if random.random() < RANDOM_QUESTION_PROBABILITY:
                target_user = get_random_seen_user(chat_id, exclude_user_id=user.id if user else None)
                if target_user:
                    await update.message.chat.send_action(action=ChatAction.TYPING)
                    response_text = await generate_random_question(target_user, settings)
                    if response_text:
                        chunks = _split_message(response_text)
                        for chunk in chunks:
                            await _safe_send_message(context.bot, chat_id, chunk)
                        append_history(chat_id, "assistant", response_text)
                return
            
            if random.random() >= RANDOM_PARTICIPATION_PROBABILITY:
                return
        else:
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
    
    available_tools = []
    if IMAGE_GENERATION_ENABLED:
        available_tools.append(_FUNCTION_CALLING_TOOLS[0])
    if WEB_SEARCH_ENABLED:
        available_tools.append(_FUNCTION_CALLING_TOOLS[1])
        
    if available_tools and prompt:
        try:
            tool_messages = [{"role": "user", "content": prompt}]
            _, tool_calls = await chat_completion(
                tool_messages,
                tools=available_tools,
                tool_choice="auto",
                temperature=0.1,
                max_tokens=50
            )
            
            if tool_calls:
                tool_call = tool_calls[0]
                function_name = tool_call.get("name")
                function_args = tool_call.get("args", {})
                
                if function_name == "generate_image" and IMAGE_GENERATION_ENABLED:
                    asyncio.create_task(_generate_and_send_image(update.message, function_args.get("prompt", prompt)))
                    return
                elif function_name == "search_web" and WEB_SEARCH_ENABLED:
                    search_query = function_args.get("query", prompt)
                    await update.message.chat.send_action(action=ChatAction.TYPING)
                    try:
                        results = await search_web(search_query, limit=WEB_SEARCH_MAX_RESULTS)
                        if results:
                            web_results_text = _format_search_results(results, search_query)
                            safe_tokens = max(500, CONTEXT_LIMIT_TOKENS - settings.get("max_tokens", 512) - 1000)
                            web_results_text = _truncate_web_text(web_results_text, safe_tokens)
                            web_context = (
                                "Данные из интернета (это лишь краткие выдержки). "
                                "Если пользователь ищет стих, песню или известный текст, а здесь только обрывки, "
                                "ОБЯЗАТЕЛЬНО воспроизведи текст целиком по памяти. Не давай ссылки, если не просили:\n"
                                f"{web_results_text}"
                            )
                    except WebSearchError as exc:
                        logger.warning("Web search failed during tool execution: %s", exc)
        except Exception as exc:
            logger.warning("Ошибка Tool Calling: %s", exc)

    reset_used, reset_remainder = _split_reset_request(prompt)
    if reset_used:
        clear_history(chat_id)
        if not reset_remainder:
            await _safe_reply_text(update.message, "Контекст очищен.")
            return
        prompt = reset_remainder
        reply_text = ""

    await update.message.chat.send_action(action=ChatAction.TYPING)
    
    # Динамически подсказываем LLM её имя (из триггера), чтобы она правильно склоняла глаголы (он/она)
    req_settings = settings.copy()
    bot_identity = trigger_word if trigger_word else context.bot.first_name
    gender_hint = f"\n[Системное правило: Твое имя — «{bot_identity}». Пиши о себе строго в соответствующем этому имени роде.]"
    req_settings["extra_prompt"] = f"{req_settings.get('extra_prompt', '')}{gender_hint}".strip()

    response_text, parse_mode, error_msg = await process_chat_request(
        chat_id, prompt, reply_text, req_settings, web_context, web_results_text
    )
    
    if error_msg:
        await _safe_reply_text(update.message, error_msg)
        return

    chunks = _split_message(response_text)
    if not chunks:
        await _safe_reply_text(
            update.message,
            "Модель вернула пустой ответ. Попробуй переформулировать."
        )
        return

    voice_sent = False
    if settings.get("voice_response") and response_text:
        await update.message.chat.send_action(action=ChatAction.RECORD_VOICE)
        clean_text = _strip_markdown_syntax(response_text)
        
        if len(clean_text) > 1500:
            snippet = clean_text[:1500]
            cut_at = max(snippet.rfind("."), snippet.rfind("!"), snippet.rfind("?"))
            if cut_at > 1000:
                clean_text = snippet[:cut_at + 1]
            else:
                cut_at = snippet.rfind(" ")
                clean_text = snippet[:cut_at] + "..." if cut_at > 0 else snippet + "..."
                
        voice_setting = settings.get("voice_response")
        audio_bytes = await generate_speech(clean_text, voice=voice_setting)
        if audio_bytes:
            try:
                await context.bot.send_voice(
                    chat_id=chat_id,
                    voice=audio_bytes,
                    reply_to_message_id=update.message.message_id
                )
                voice_sent = True
            except Exception as exc:
                logger.warning("Failed to send voice response: %s", exc)

    if not voice_sent:
        await _safe_reply_text(update.message, chunks[0], parse_mode=parse_mode)
        for chunk in chunks[1:]:
            await _safe_send_message(context.bot, chat_id, chunk, parse_mode=parse_mode)
