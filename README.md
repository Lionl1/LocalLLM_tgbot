# Telegram LLM Bot

Python Telegram bot that talks through a local LLM exposed via an OpenAI-compatible API
(such as vLLM, LM Studio, or Ollama) and uses LangChain for the response pipeline.
In group chats it replies when a message starts with the trigger word (default: `Nuka`, configurable)
or when someone replies to one of the bot's messages.

The bot is intended for international use:
- code comments and project documentation are written in English;
- the bot itself should reply in the language used by the user, or in the language explicitly requested by the user.

## Features

- 🤖 **Local LLMs:** Works with any OpenAI-compatible API (vLLM, LM Studio, Ollama).
- 🎙 **Voice transcription:** Automatically transcribes audio messages into text, cleans formatting, fixes punctuation, and can generate a short summary for long voice messages.
- 🗣 **Voice replies (TTS):** Can read its replies aloud with realistic Silero TTS voices.
- 🧠 **Memory and context management:** Keeps conversation history, understands reply chains, trims old context when needed, and falls back to safer prompt layouts on model errors.
- 🔎 **Web search:** Can search the web and use the LLM to produce a concise, readable answer from the results instead of dumping raw links.
- 🎨 **Image generation:** Local image generation through DreamShaper-compatible flow (`/image` or natural language triggers).
- 🎲 **Random engagement prompts:** Can generate weird, funny, or provocative questions to keep group chats active.
- ⚙️ **Mini App management:** Telegram Web App interface for per-chat settings such as system prompt, voice, response length, and trigger word.
- 👥 **Private chats and groups:** Supports both private and group chat workflows, including Telegram privacy mode considerations.
- 🐳 **Docker-ready:** Simple container build and deployment flow.
- ⚡ **Fast setup:** Uses `uv` for quick dependency installation and virtual environment management.

## System Requirements

Because the bot runs local models (for example Whisper for audio and Stable Diffusion-compatible image generation), it needs enough host resources for stable operation:

- **Minimum (slow CPU-only usage):**
  - RAM: 8 GB
  - Disk: about 5-10 GB free for model caches (Stable Diffusion, Whisper, and the local LLM)
- **Recommended (faster GPU-accelerated usage):**
  - RAM: 16+ GB
  - GPU: NVIDIA with 8+ GB VRAM (CUDA) **or** Apple Silicon (M1/M2/M3/M4) with 16+ GB unified memory (MPS)

> **Important:** If you run the bot through Docker Desktop on macOS or Windows, allocate at least **6 GB of RAM** to containers in `Settings -> Resources`. Otherwise the OS may kill the container with OOM during image generation.

## Mini App Setup

The project includes a Telegram Web App interface for bot settings. To use it, host the HTML file on GitHub Pages:

1. Create a public GitHub repository, for example `bot-settings`.
2. Copy the contents of `app/index.html` from this project into an `index.html` file in that repository.
3. Open `Settings -> Pages`, choose the `main` (or `master`) branch, and save.
4. GitHub will publish a URL like `https://your-name.github.io/bot-settings/`.
5. Add that URL to `.env`:

```env
WEB_APP_URL="https://your-name.github.io/bot-settings/"
```

After that, calling `/settings` will open the settings Mini App inside Telegram.

## Quick Start

1. Create a bot in BotFather and get the token.
2. Copy `.env.example` to `.env` and fill in the required values (`TELEGRAM_BOT_TOKEN`, and optionally `OPENAI_API_KEY`, `WEB_SEARCH_API_KEY`).
3. Review `app/config.py` for model, API URL, limits, and prompt settings.
4. Start your local LLM server:
   - LM Studio: enable the server and use `http://localhost:1234/v1`
   - vLLM: `python -m vllm.entrypoints.openai.api_server --model <model> --port 8000`
5. Install `ffmpeg` (required for voice transcription):
   - Ubuntu/Debian: `sudo apt update && sudo apt install ffmpeg`
   - macOS: `brew install ffmpeg`
   - Windows: install `ffmpeg` and make sure it is available in `PATH`
6. Install dependencies and run the bot:

```bash
# 1. Install uv if it is not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh
# Windows:
# powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# 2. uv will create the virtual environment, download Python if needed,
#    and install all dependencies automatically
uv run python main.py
```

## Project Structure

- `main.py` - application entry point
- `app/bot.py` - Telegram application assembly and startup
- `app/handlers.py` - commands and message handlers
- `app/config.py` - application configuration (`.env` values are loaded here)
- `app/llm_service.py` - high-level LLM workflows
- `app/llm_client.py` - LLM client and request pipeline
- `app/pipeline.py` - message assembly, context handling, and response post-processing
- `app/image_client.py` - image generation client
- `app/tts_client.py` - text-to-speech client
- `app/audio_client.py` - speech-to-text client
- `app/search_client.py` - web search client
- `app/state.py` - chat memory and settings storage
- `app/text_utils.py` - text parsing and message splitting helpers
- `app/ui.py` - keyboard builders and settings formatting
- `app/index.html` - Telegram Mini App template for settings

## Docker

The easiest deployment option is **Docker Compose**:

```bash
docker compose up -d --build
```

This command builds the image and starts the bot in the background. Neural model caches
(such as Whisper and Stable Diffusion) and chat settings are stored in Docker volumes so they survive restarts and upgrades.

<details>
<summary>Run with plain Docker CLI</summary>

```bash
docker build -t telegram-llm-bot .

docker run -d --env-file .env \
  -v hf_cache:/root/.cache/huggingface \
  -v bot_data:/app/data \
  --name telegram-llm-bot --restart unless-stopped telegram-llm-bot
```

If the local LLM runs on the host machine, `localhost` inside the container will not reach it.

- On macOS or Windows, set `OPENAI_BASE_URL` in `.env` to `http://host.docker.internal:1234/v1`
- On Linux, you can use `--network=host`

</details>

## Usage

- **Private chats:** the bot replies to any text message.
- **Groups:** mention the trigger word (for example `Nuka, hello`) or reply to one of the bot's messages.
- If privacy mode is enabled in BotFather, the bot only sees commands and mentions, so a plain trigger word without a mention may not work until privacy mode is disabled.
- `/reset` clears the chat context.
- In groups you can also clear context with messages such as `<BotName>, reset`.
- 🎙 **Voice messages:** send a voice message and the bot will transcribe it, clean it up, and answer.
- If the context grows too large, the bot automatically drops the oldest messages and retries.
- If you reply to another person's message, that text is added to the request context.
- `/help` shows the command list and settings controls.
- `/search <text>` runs web search if it is enabled.
- `/image <description>` generates an image locally.
- The bot can also auto-trigger image generation from natural-language requests like "draw a picture of ...".
- The bot can auto-trigger web search from natural-language requests like "find this on the internet".
- Prefixes such as `web:` or `search:` can be used to force web search before the LLM response.

### Chat Settings

Send `/settings` to the bot in a private chat. It will open a Mini App where you can configure the bot separately for each group where you are an admin.

Settings are stored in `data/chat_settings.json` and persist across restarts.

The bot also shows settings buttons on `/start` and `/settings`, and refreshes the Telegram command menu automatically at startup.

## Configuration

Secrets live in `.env`. Other application defaults are defined in `app/config.py`.

Expected `.env` values:

- `TELEGRAM_BOT_TOKEN` - Telegram bot token
- `OPENAI_API_KEY` - LLM API key (`not-needed` is fine for many local servers)
- `OPENAI_BASE_URL` - local LLM API URL (`http://localhost:1234/v1` for LM Studio)
- `OPENAI_MODEL` - model name
- `WEB_SEARCH_API_KEY` - search API key (used for `serper`)
- `ALLOWED_USER_IDS` - comma-separated list of allowed Telegram `user_id` values; empty means open access
- `IMAGE_GENERATION_ENABLED` - `1` or `0`, enables image generation inside the bot
- `WEB_APP_URL` - URL of your hosted copy of `app/index.html`
- `IMAGE_GENERATION_TIMEOUT` - request timeout in seconds (default `60`)
- `IMAGE_GENERATION_WIDTH`, `IMAGE_GENERATION_HEIGHT` - desired image size (default `1024`)

Other parameters such as limits, prompts, search behavior, and context rules are configured directly in `app/config.py`.
