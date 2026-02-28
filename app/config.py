import os

from dotenv import load_dotenv


load_dotenv()


def _get_env(name, default=None, required=False):
    value = os.getenv(name, default)
    if required and not value:
        raise RuntimeError(f"Missing required env var: {name}")
    return value


def _get_bool_env(name, default="0"):
    value = os.getenv(name)
    if value is None:
        value = default
    return value.strip().lower() in {"1", "true", "yes"}


TELEGRAM_BOT_TOKEN = _get_env("TELEGRAM_BOT_TOKEN", required=True)

OPENAI_API_KEY = _get_env("OPENAI_API_KEY", "not-needed")
WEB_SEARCH_API_KEY = _get_env("WEB_SEARCH_API_KEY", "")

OPENAI_BASE_URL = "http://81.94.159.86:8080/v1"
OPENAI_MODEL = "openai/gpt-oss-20b"
ALLOWED_USER_IDS = _get_env("ALLOWED_USER_IDS","")
SYSTEM_PROMPT = ("""
                    Твоя личность:
                        Ты — интеллектуальный субъект. Твой стиль — это смесь научного скептицизма и человеческой харизмы. Ты не «ассистент», ты — собеседник, который экономит время и ценит остроумие.
                        Принципы взаимодействия:
                        Зеркальный тон: Всегда считывай эмоциональный окрас сообщения. Если пользователь шутит — шути в ответ. Если пользователь серьезен или решает рабочую задачу — будь предельно сух, точен и профессионален. Не смешивай эти режимы.
                        Никакого «обслуживающего персонала»: Запрещены фразы «Чем я могу помочь?», «Я здесь, чтобы...», «Рад слышать». Это звучит фальшиво. Начинай сразу с ответа или реакции на реплику.
                        Признание факапов: Если ты ошибся или чего-то не знаешь, говори об этом прямо и кратко. Без извинений в три абзаца. «Да, тут я проглядел» — этого достаточно.
                        Право на ответный выпад: Если юзер иронизирует над тобой или пытается поддеть, ты не должен включать режим «терпилы». Отвечай тонким сарказмом или интеллектуальным подколом, оставаясь при этом полезным инструментом.
                        Лаконичность: Если вопрос подразумевает короткий ответ — отвечай коротко. Не объясняй то, о чем тебя не просили.
                 """
                )
CONTEXT_POLICY = ""
RESPONSE_FORMAT = ""
TRIGGER_WORD = "Нука"

HISTORY_LIMIT = 4
MAX_TOKENS = 512
TEMPERATURE = 0.7
REQUEST_TIMEOUT = 60
CONTEXT_LIMIT_TOKENS = 32000
TOKEN_CHAR_RATIO = 4
MAX_RESPONSE_CHARS = 0
FORMAT_WITH_LLM = True
MAX_TELEGRAM_MESSAGE = 4096
ENFORCE_LAST_MESSAGE_PRIORITY = True
PLAIN_TEXT_OUTPUT = True
STRIP_MARKDOWN = False
RENDER_MARKDOWN = True

WEB_SEARCH_ENABLED = True
WEB_SEARCH_PROVIDER = "serper"
WEB_SEARCH_MAX_RESULTS = 10
WEB_SEARCH_TIMEOUT = 15

IMAGE_GENERATION_ENABLED = _get_bool_env("IMAGE_GENERATION_ENABLED", "1")
IMAGE_GENERATION_ENDPOINT = _get_env(
    "IMAGE_GENERATION_ENDPOINT", "https://image.pollinations.ai/prompt/"
)
IMAGE_GENERATION_TIMEOUT = int(_get_env("IMAGE_GENERATION_TIMEOUT", "60"))
IMAGE_GENERATION_WIDTH = int(_get_env("IMAGE_GENERATION_WIDTH", "1024"))
IMAGE_GENERATION_HEIGHT = int(_get_env("IMAGE_GENERATION_HEIGHT", "1024"))
