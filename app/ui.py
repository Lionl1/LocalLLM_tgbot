from telegram import InlineKeyboardButton, InlineKeyboardMarkup


def _format_settings(settings):
    lines = [
        "Текущие настройки:",
        f"Триггер: {settings['trigger_word']}",
        f"Настроение: {settings['mood'] or 'не задано'}",
        f"Доп. промпт: {settings['extra_prompt'] or 'не задан'}",
        f"Формат ответа: {settings['response_format'] or 'не задан'}",
        (
            "Лимит символов: "
            f"{settings['max_response_chars']}"
            if settings["max_response_chars"] > 0
            else "Лимит символов: не задан"
        ),
        f"Макс. ответ (tokens): {settings['max_tokens']}",
    ]
    return "\n".join(lines)


def _settings_keyboard():
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton("Настроение", callback_data="set_mood"),
                InlineKeyboardButton("Очистить настроение", callback_data="clear_mood"),
            ],
            [
                InlineKeyboardButton("Доп. промпт", callback_data="set_prompt"),
                InlineKeyboardButton("Очистить промпт", callback_data="clear_prompt"),
            ],
            [
                InlineKeyboardButton("Лимит ответа", callback_data="set_max"),
                InlineKeyboardButton("Триггер", callback_data="set_trigger"),
            ],
            [
                InlineKeyboardButton(
                    "Показать настройки", callback_data="show_settings"
                ),
                InlineKeyboardButton(
                    "Сбросить настройки", callback_data="reset_settings"
                ),
            ],
            [InlineKeyboardButton("Отмена", callback_data="cancel")],
        ]
    )


def _cancel_keyboard():
    return InlineKeyboardMarkup(
        [[InlineKeyboardButton("Отмена", callback_data="cancel")]]
    )
