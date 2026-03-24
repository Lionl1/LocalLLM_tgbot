import logging

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
from app.state import (
    append_history,
    get_history,
    set_history,
    trim_oldest_history,
)

logger = logging.getLogger(__name__)


async def summarize_search_results(query, text, settings):
    system_prompt = (
        "Твоя задача — обработать результаты веб-поиска и дать пользователю "
        "понятный, связный и краткий ответ на его запрос. "
        "Важно: не выводи список ссылок и источников, если пользователь строго не попросил об этом "
        "(например, словами 'дай ссылки' или 'выведи списком')."
    )
    user_prompt = f"Запрос: {query}\n\nРезультаты поиска:\n{text}"
    
    try:
        response_text = await chat_completion(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=settings.get("max_tokens", 512),
            temperature=0.3,
        )
        return (response_text or "").strip()
    except Exception as exc:
        logger.exception("LLM generation failed for search command: %s", exc)
        return ""


async def generate_random_question(target_user, settings):
    name_mention = f"@{target_user['username']}" if target_user.get("username") else target_user.get("first_name", "пользователь")
    system_prompt = _compose_system_prompt(settings)
    q_prompt = (
        f"Сгенерируй случайный, забавный, странный или провокационный вопрос "
        f"для пользователя {name_mention}. Не пиши приветствий, просто задай вопрос."
    )
    try:
        response_text = await chat_completion(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": q_prompt}
            ],
            max_tokens=settings.get("max_tokens", 512),
            temperature=0.8,
        )
        return (response_text or "").strip()
    except Exception as exc:
        logger.warning("Failed to generate random question: %s", exc)
        return ""


async def process_chat_request(chat_id, prompt, reply_text, settings, web_context="", web_results_text=""):
    history = list(get_history(chat_id))
    history, trimmed, messages = await _trim_history_to_fit(
        history, prompt, reply_text, settings, web_context
    )
    if trimmed:
        set_history(chat_id, history)
    if _context_limit_exceeded(messages, settings["max_tokens"]):
        return None, None, "Запрос слишком длинный для контекста модели. Сократи текст."

    attempts = 0
    max_attempts = max(1, len(history) // 2 + 1)
    response_text = ""
    
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
                logger.warning("Chat template error, retrying with minimal prompt for chat %s", chat_id)
                try:
                    response_text = await chat_completion(
                        _build_flat_fallback_messages(history, prompt, reply_text, settings, web_context),
                        max_tokens=settings["max_tokens"],
                        temperature=settings["temperature"],
                    )
                    break
                except Exception as fallback_exc:
                    logger.exception("Fallback LLM request failed: %s", fallback_exc)
                    return None, None, "Не удалось получить ответ от локальной LLM."
            logger.exception("LLM request failed: %s", exc)
            return None, None, "Не удалось получить ответ от локальной LLM."
        except Exception as exc:
            logger.exception("LLM request failed: %s", exc)
            return None, None, "Не удалось получить ответ от локальной LLM."

    response_text = (response_text or "").strip()
    parse_mode = None
    if response_text:
        response_text, parse_mode = await _postprocess_response(prompt, response_text, settings)
        
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
            return None, None, "Не удалось получить ответ от локальной LLM."
        
        response_text = (response_text or "").strip()
        if response_text:
            response_text, parse_mode = await _postprocess_response(prompt, response_text, settings)
        if not response_text:
            if web_results_text:
                fallback = "Модель вернула пустой ответ. Вот результаты поиска:\n" + web_results_text
                return fallback, None, None
            return None, None, "Модель вернула пустой ответ. Попробуй переформулировать."

    append_history(chat_id, "user", prompt)
    append_history(chat_id, "assistant", response_text)

    return response_text, parse_mode, None