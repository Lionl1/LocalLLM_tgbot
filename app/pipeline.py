import logging
import re

from app.config import CONTEXT_LIMIT_TOKENS, SYSTEM_PROMPT
from app.llm_client import chat_completion
from app.state import trim_oldest_history
from app.text_utils import _estimate_messages_tokens

logger = logging.getLogger(__name__)

_SUMMARY_PREFIX = "Summary of the previous conversation:"

def _priority_instruction(settings):
    if settings.get("enforce_last_message_priority", True):
        return (
            "Always answer the user's latest request. "
            "Use prior history only when it is directly relevant to the current request."
        )
    return ""

def _compose_system_prompt(settings):
    parts = [settings["system_prompt"]]
    if settings["context_policy"]:
        parts.append(f"Context rules: {settings['context_policy']}")
    priority = _priority_instruction(settings)
    if priority:
        parts.append(priority)
    if settings.get("plain_text_output"):
        parts.append(
            "Default to plain text without Markdown. "
            "If you do use Markdown for code or tables, keep it valid."
        )
    if settings["extra_prompt"]:
        parts.append(f"Additional instructions: {settings['extra_prompt']}")
    if settings["mood"]:
        parts.append(f"Reply mood: {settings['mood']}")
    if settings["response_format"]:
        parts.append(f"Response format: {settings['response_format']}")
    if settings["max_response_chars"] > 0:
        parts.append(
            f"Limit: no more than {settings['max_response_chars']} characters."
        )
    if settings.get("voice_response"):
        parts.append(
            "Your reply will be spoken aloud. Keep it concise, conversational, and easy to read aloud. "
            "Avoid long lists, code blocks, and heavy formatting."
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
                "content": f"Message being replied to:\n{reply_text}",
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
        context_parts.append("Conversation history:")
        for message in history[-8:]:
            role = message.get("role", "user")
            label = "User" if role == "user" else "Assistant"
            content = (message.get("content") or "").strip()
            if content:
                context_parts.append(f"{label}: {content}")
    if reply_text:
        context_parts.append("Message being replied to:")
        context_parts.append(reply_text)
    if web_context:
        context_parts.append("Web context:")
        context_parts.append(web_context)
    context_block = "\n".join(context_parts).strip()
    if context_block:
        user_content = f"{context_block}\n\nCurrent request:\n{prompt}"
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


def _fix_markdown_formatting(text):
    if not text:
        return text
    
    if text.count("```") % 2 != 0:
        text += "\n```"

    lines = text.split('\n')
    in_code_block = False
    in_table = False
    new_lines = []
    
    for line in lines:
        stripped = line.strip()
        if stripped.startswith('```'):
            if in_table:
                new_lines.append('```')
                in_table = False
            in_code_block = not in_code_block
            new_lines.append(line)
            continue
        
        is_table_row = bool(re.match(r'^\s*\|.+\|\s*$', line)) and line.count('|') >= 2
        
        if not in_code_block and is_table_row:
            if not in_table:
                new_lines.append('```text')
                in_table = True
            new_lines.append(line)
        else:
            if in_table:
                new_lines.append('```')
                in_table = False
            new_lines.append(line)
    
    if in_table:
        new_lines.append('```')
    return '\n'.join(new_lines)

async def _format_response_with_llm(prompt, response_text, settings):
    if not settings.get("response_format") or not settings.get("format_with_llm"):
        return response_text
    instructions = [settings.get("system_prompt") or SYSTEM_PROMPT]
    if settings.get("context_policy"):
        instructions.append(f"Context rules: {settings['context_policy']}")
    if settings.get("extra_prompt"):
        instructions.append(f"Additional instructions: {settings['extra_prompt']}")
    if settings.get("mood"):
        instructions.append(f"Reply mood: {settings['mood']}")
    instructions.append(f"Response format: {settings['response_format']}")
    if settings.get("max_response_chars", 0) > 0:
        instructions.append(
            f"Limit: no more than {settings['max_response_chars']} characters."
        )
    if settings.get("voice_response"):
        instructions.append(
            "Your reply will be spoken aloud. Keep it concise, conversational, and easy to read aloud. "
            "Avoid long lists, code blocks, and heavy formatting."
        )
    requirements = "\n\n".join(instructions)
    user_content = (
        "Follow the requirements and target format. Do not add new facts and do not change the meaning. "
        "If you use Markdown, keep it valid.\n\n"
        f"Requirements:\n{requirements}\n\n"
        f"Request:\n{prompt}\n\n"
        f"Draft reply:\n{response_text}\n\n"
        "Rewrite the reply so it strictly follows the requirements."
    )
    messages = [
        {
            "role": "system",
            "content": "You are a response editor. Rewrite replies to match the requested role and format.",
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
        "Check the following text for grammar, spelling, and punctuation errors. "
        "Fix them while preserving the original style and structure, including Markdown. "
        "Do not normalize slang, neologisms, or intentionally creative phrasing.\n"
        "Return ONLY the corrected text, with no introduction or explanation.\n\n"
        f"Text:\n{text}"
    )
    
    messages = [
        {"role": "system", "content": "You are a proofreader. Fix obvious mistakes while preserving the author's style."},
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
        
    if settings.get("max_response_chars", 0) > 0:
        text = _trim_to_char_limit(text, settings["max_response_chars"])
        
    text = _fix_markdown_formatting(text)
    
    parse_mode = None
    if settings.get("render_markdown") and _looks_like_markdown(text):
        parse_mode = "Markdown"
    return text, parse_mode

def _max_prompt_tokens(max_tokens):
    return max(CONTEXT_LIMIT_TOKENS - max_tokens, 1)

def _context_limit_exceeded(messages, max_tokens):
    return _estimate_messages_tokens(messages) > _max_prompt_tokens(max_tokens)

async def _generate_summary(text_to_summarize):
    prompt = (
        "Update the conversation summary.\n"
        "1. Preserve important facts, names, and context from the existing summary, if any.\n"
        "2. Add the key information from the new messages.\n"
        "3. Be extremely concise and remove filler or greetings.\n"
        "4. Preserve notable jokes, sarcasm, or humor when relevant.\n\n"
        f"TEXT TO ANALYZE:\n{text_to_summarize}"
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
    
    if not _context_limit_exceeded(messages, settings["max_tokens"]):
        return history, trimmed, messages

    existing_summary_text = ""
    start_idx = 0
    if history and history[0].get("role") == "system" and history[0]["content"].startswith(_SUMMARY_PREFIX):
        existing_summary_text = history[0]["content"].replace(_SUMMARY_PREFIX, "").strip()
        start_idx = 1

    active_history = history[start_idx:]

    if len(active_history) > 1:
        split_idx = max(1, int(len(active_history) * 0.6))
        msgs_to_compress = active_history[:split_idx]
        recent_history = active_history[split_idx:]

        text_block = ""
        if existing_summary_text:
            text_block += f"=== CURRENT SUMMARY ===\n{existing_summary_text}\n\n"
        
        text_block += "=== NEW MESSAGES ===\n"
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
            messages = _build_messages(history, prompt, reply_text, settings, web_context)

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
