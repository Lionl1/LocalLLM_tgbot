import os

# app.config refuses to import without a bot token.
os.environ.setdefault("TELEGRAM_BOT_TOKEN", "test-token")
