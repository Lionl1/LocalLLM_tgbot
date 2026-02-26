import os

from dotenv import load_dotenv


load_dotenv()


def _get_env(name, default=None, required=False):
    value = os.getenv(name, default)
    if required and not value:
        raise RuntimeError(f"Missing required env var: {name}")
    return value


def _parse_int_set(value):
    if not value:
        return set()
    result = set()
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            result.add(int(item))
        except ValueError as exc:
            raise RuntimeError(f"Invalid integer in ALLOWED_USER_IDS: {item}") from exc
    return result


def _parse_bool(value, default=False):
    if value is None:
        return default
    normalized = str(value).strip().casefold()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off", ""}:
        return False
    return default


TELEGRAM_BOT_TOKEN = _get_env("TELEGRAM_BOT_TOKEN", required=True)

OPENAI_BASE_URL = _get_env("OPENAI_BASE_URL", "http://localhost:1234/v1")
OPENAI_API_KEY = _get_env("OPENAI_API_KEY", "not-needed")
OPENAI_MODEL = _get_env("OPENAI_MODEL", "local-model")

SYSTEM_PROMPT = _get_env("SYSTEM_PROMPT", "You are a helpful assistant.")
TRIGGER_WORD = _get_env("TRIGGER_WORD", "Нука")

HISTORY_LIMIT = int(_get_env("HISTORY_LIMIT", "8"))
MAX_TOKENS = int(_get_env("MAX_TOKENS", "512"))
TEMPERATURE = float(_get_env("TEMPERATURE", "0.7"))
REQUEST_TIMEOUT = float(_get_env("REQUEST_TIMEOUT", "60"))
CONTEXT_LIMIT_TOKENS = int(_get_env("CONTEXT_LIMIT_TOKENS", "32000"))
TOKEN_CHAR_RATIO = float(_get_env("TOKEN_CHAR_RATIO", "4"))

ALLOWED_USER_IDS = _parse_int_set(_get_env("ALLOWED_USER_IDS", ""))

WEB_SEARCH_ENABLED = _parse_bool(_get_env("WEB_SEARCH_ENABLED", "0"))
WEB_SEARCH_PROVIDER = _get_env("WEB_SEARCH_PROVIDER", "duckduckgo")
WEB_SEARCH_API_KEY = _get_env("WEB_SEARCH_API_KEY", "")
WEB_SEARCH_MAX_RESULTS = int(_get_env("WEB_SEARCH_MAX_RESULTS", "5"))
WEB_SEARCH_TIMEOUT = float(_get_env("WEB_SEARCH_TIMEOUT", "15"))
