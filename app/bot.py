import logging

from telegram.ext import Application, CallbackQueryHandler, CommandHandler, MessageHandler, filters

from app.config import TELEGRAM_BOT_TOKEN
from app.handlers import (
    cancel_command,
    clear_mood_command,
    clear_prompt_command,
    handle_message,
    help_command,
    image_command,
    post_init,
    reset_command,
    reset_settings_command,
    search_command,
    set_max_command,
    set_mood_command,
    set_prompt_command,
    set_trigger_command,
    settings_button,
    settings_command,
    start_command,
)


logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)


def main():
    application = (
        Application.builder().token(TELEGRAM_BOT_TOKEN).post_init(post_init).build()
    )

    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("search", search_command))
    application.add_handler(CommandHandler("reset", reset_command))
    application.add_handler(CommandHandler("settings", settings_command))
    application.add_handler(CommandHandler("image", image_command))
    application.add_handler(CommandHandler("setmood", set_mood_command))
    application.add_handler(CommandHandler("clearmood", clear_mood_command))
    application.add_handler(CommandHandler("setprompt", set_prompt_command))
    application.add_handler(CommandHandler("clearprompt", clear_prompt_command))
    application.add_handler(CommandHandler("setmax", set_max_command))
    application.add_handler(CommandHandler("settrigger", set_trigger_command))
    application.add_handler(CommandHandler("setname", set_trigger_command))
    application.add_handler(CommandHandler("resetsettings", reset_settings_command))
    application.add_handler(CommandHandler("cancel", cancel_command))
    application.add_handler(CallbackQueryHandler(settings_button))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    application.run_polling(allowed_updates=["message", "callback_query"])


if __name__ == "__main__":
    main()
