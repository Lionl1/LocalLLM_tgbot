# Telegram LLM Bot

Бот на Python, который отвечает через локальную LLM с OpenAI-совместимым API
(vLLM или LM Studio) и использует LangChain для пайплайна ответов.
В группе отвечает, если сообщение начинается с "Нука" или если это ответ
на сообщение бота.

## Быстрый старт

1) Создай бота в BotFather и возьми токен.
2) Скопируй `.env.example` в `.env` и заполни ключи (`TELEGRAM_BOT_TOKEN`, при необходимости `OPENAI_API_KEY`, `WEB_SEARCH_API_KEY`).
3) Проверь настройки в `app/config.py` (модель, URL, лимиты, промпты).
4) Запусти локальную LLM:
   - LM Studio: включи Server и используй `http://localhost:1234/v1`.
   - vLLM: `python -m vllm.entrypoints.openai.api_server --model <model> --port 8000`
5) Установи зависимости и запусти:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

## Структура проекта

- `main.py` — точка входа.
- `app/bot.py` — сборка и запуск приложения Telegram.
- `app/handlers.py` — команды и обработчики сообщений.
- `app/config.py` — конфигурация приложения (ключи берутся из `.env`).
- `app/llm_client.py` — LLM клиент и пайплайн запроса.
- `app/pipeline.py` — сбор сообщений, контекст и постобработка ответа.
- `app/search_client.py` — клиент веб-поиска.
- `app/state.py` — память диалога и настройки.
- `app/text_utils.py` — утилиты разбора текста и лимитов.
- `app/ui.py` — клавиатуры и форматирование настроек.

## Docker

Сборка и запуск:

```bash
docker build -t telegram-llm-bot .
docker run --env-file .env --name telegram-llm-bot --restart unless-stopped telegram-llm-bot
```

Если локальная LLM запущена на хосте, внутри контейнера `localhost` не виден.
Для macOS/Windows укажи `OPENAI_BASE_URL` в `.env` как
`http://host.docker.internal:1234/v1`.
Для Linux можно использовать `--network=host` при запуске контейнера.

## Использование

- Личные сообщения: бот отвечает на любой текст.
- Группы: напиши `Нука, ...` или ответь на сообщение бота.
  Если включен privacy mode в BotFather, бот видит только команды/упоминания,
  поэтому для триггера без упоминания нужно отключить privacy mode.
- `/reset` — очистить контекст диалога в этом чате.
- В группе можно очистить контекст сообщением `Нука, сброс` или `Нука, очистить`.
- При переполнении контекста бот удаляет самые старые сообщения и повторяет запрос.
- Если отвечаешь на сообщение человека, бот добавит этот текст в контекст запроса.
- `/help` — список команд и кнопки настроек.
- `/search <текст>` — поиск в интернете (если включен).
- `/image <описание>` — генерация картинки через бесплатный сервис (по умолчанию Pollinations), бот отправит изображение.
- Бот автоматически запускает генерацию, если видит текст вроде «нарисуй картинку с...», даже без команды.
- Бот автоматически выполняет поиск по тексту вроде «найди в интернете», чтобы показать результаты без команды.
- Префикс `web:`/`search:`/`поиск:`/`найди в интернете` — выполнить поиск и передать результаты модели.

### Настройки в чате

Команды работают в личке и в группах:

- `/settings` — показать текущие настройки.
- `/setmood <текст>` — задать настроение ответов.
- `/clearmood` — очистить настроение.
- `/setprompt <текст>` — добавить доп. системные инструкции.
- `/clearprompt` — удалить доп. инструкции.
- `/setmax <число>` — ограничить размер ответа в токенах.
- `/settrigger <имя>` — изменить слово-триггер (например `Нука`).
- `/resetsettings` — сбросить настройки к значениям по умолчанию.
- `/cancel` — отменить ожидание ввода значения.

Настройки хранятся в памяти и сбрасываются при перезапуске бота.

Также при `/start` и `/settings` бот показывает кнопки для настройки.
Список команд в меню Telegram обновляется автоматически при запуске бота.

## Настройки

Секреты лежат в `.env`, все остальные параметры — в `app/config.py`.

В `.env` должны быть ключи и параметры подключения:

- `TELEGRAM_BOT_TOKEN` — токен Telegram-бота.
- `OPENAI_API_KEY` — ключ LLM API (для локальных серверов можно оставить `not-needed`).
- `OPENAI_BASE_URL` — URL локального API (`http://localhost:1234/v1` для LM Studio).
- `OPENAI_MODEL` — имя модели.
- `WEB_SEARCH_API_KEY` — ключ API (нужен для `serper`).
- `ALLOWED_USER_IDS` — список `user_id` через запятую. Если пусто, доступ открыт.
- `IMAGE_GENERATION_ENABLED` — `1`/`0`, включает генерацию картинок внутри бота (по умолчанию `1`).
- `IMAGE_GENERATION_ENDPOINT` — URL генератора. Для Pollinations: `https://image.pollinations.ai/prompt/`. Для Hugging Face: `https://api-inference.huggingface.co/models/<model_id>`.
- `HUGGING_FACE_TOKEN` — (опционально) токен Hugging Face (Write), если используется их API.
- `IMAGE_GENERATION_TIMEOUT` — таймаут запроса в секундах (по умолчанию `60`).
- `IMAGE_GENERATION_WIDTH`, `IMAGE_GENERATION_HEIGHT` — желаемый размер картинки (по умолчанию `1024`).

Остальные параметры (лимиты, промпты, поиск, правила контекста)
настраиваются прямо в `app/config.py`.


docker build -t telegram-llm-bot .
docker rm -f telegram-llm-bot
docker run --env-file .env --name telegram-llm-bot --restart unless-stopped telegram-llm-bot
