import logging

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from app.config import (
    MAX_TOKENS,
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
    OPENAI_MODEL,
    REQUEST_TIMEOUT,
    TEMPERATURE,
)


logger = logging.getLogger(__name__)


class LLMRequestError(RuntimeError):
    def __init__(self, status_code, detail):
        super().__init__(f"LLM error {status_code}: {detail}")
        self.status_code = status_code
        self.detail = detail


def _to_lc_messages(messages):
    converted = []
    for message in messages or []:
        role = message.get("role", "user")
        content = message.get("content", "")
        if role == "system":
            converted.append(SystemMessage(content=content))
        elif role == "assistant":
            converted.append(AIMessage(content=content))
        else:
            converted.append(HumanMessage(content=content))
    return converted


def _build_llm(model, max_tokens, temperature):
    kwargs = {
        "base_url": OPENAI_BASE_URL,
        "api_key": OPENAI_API_KEY,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    try:
        return ChatOpenAI(model=model, timeout=REQUEST_TIMEOUT, **kwargs)
    except TypeError:
        try:
            return ChatOpenAI(model=model, request_timeout=REQUEST_TIMEOUT, **kwargs)
        except TypeError:
            try:
                return ChatOpenAI(model_name=model, request_timeout=REQUEST_TIMEOUT, **kwargs)
            except TypeError:
                return ChatOpenAI(model=model, **kwargs)


def _extract_status_code(exc):
    status_code = getattr(exc, "status_code", None)
    if status_code is not None:
        return status_code
    response = getattr(exc, "response", None)
    if response is not None:
        return getattr(response, "status_code", None)
    return None


async def chat_completion(messages, model=None, max_tokens=None, temperature=None):
    llm = _build_llm(
        model or OPENAI_MODEL,
        max_tokens if max_tokens is not None else MAX_TOKENS,
        temperature if temperature is not None else TEMPERATURE,
    )
    try:
        response = await llm.ainvoke(_to_lc_messages(messages))
    except Exception as exc:
        detail = str(exc)
        if detail and len(detail) > 1000:
            detail = f"{detail[:1000]}..."
        status_code = _extract_status_code(exc) or 0
        logger.error("LLM error %s: %s", status_code, detail)
        raise LLMRequestError(status_code, detail) from exc
    return getattr(response, "content", "")
