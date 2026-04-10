import urllib.parse
import json
from telegram import ReplyKeyboardMarkup, KeyboardButton, WebAppInfo
from app.config import WEB_APP_URL


def _format_settings(settings):
    voice_state = settings.get('voice_response')
    voice_labels = {
        "female": "ВКЛ (Ксения)",
        "male": "ВКЛ (Айдар)",
        "kseniya": "ВКЛ (Ксения)",
        "xenia": "ВКЛ (Ксения 2)",
        "baya": "ВКЛ (Байя)",
        "aidar": "ВКЛ (Айдар)",
        "eugene": "ВКЛ (Евгений)",
        "random": "ВКЛ (Случайный)"
    }
    if voice_state in voice_labels:
        voice_label = voice_labels[voice_state]
    elif voice_state:
        voice_label = "ВКЛ"
    else:
        voice_label = "ВЫКЛ"

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
        f"Синтаксис: {'ВКЛ' if settings.get('check_syntax') else 'ВЫКЛ'}",
        f"Голосовой ответ: {voice_label}",
        f"Случайные вопросы: {'ВКЛ' if settings.get('random_questions', True) else 'ВЫКЛ'}",
        f"Вероятность вопроса: {settings.get('random_question_prob', 0)}",
        f"Вероятность участия: {settings.get('random_participation_prob', 0)}",
    ]
    return "\n".join(lines)


def _settings_keyboard(manageable_chats=None, current_chat_id=None):
    web_app_url = WEB_APP_URL 
    if manageable_chats and current_chat_id:
        payload = {
            "current": current_chat_id,
            "chats": []
        }
        for c in manageable_chats:
            s = c["settings"]
            voice_val = s.get("voice_response")
            payload["chats"].append({
                "id": c["id"],
                "title": c["title"],
                "tw": s.get("trigger_word", ""),
                "md": s.get("mood", ""),
                "ep": s.get("extra_prompt", ""),
                "mt": s.get("max_tokens", 4096),
                "vr": voice_val if voice_val else "false",
                "cs": True if s.get("check_syntax") else False,
                "rq": True if s.get("random_questions", True) else False,
                "rqp": s.get("random_question_prob", 0.000000000000001),
                "rpp": s.get("random_participation_prob", 0.000000000000001)
            })
            
        json_str = json.dumps(payload)
        query_string = urllib.parse.urlencode({"config": json_str})
        web_app_url = f"{web_app_url}?{query_string}"
        
    return ReplyKeyboardMarkup(
        [[KeyboardButton("⚙️ Открыть настройки", web_app=WebAppInfo(url=web_app_url))]],
        resize_keyboard=True
    )


def _cancel_keyboard():
    return InlineKeyboardMarkup(
        [[InlineKeyboardButton("Отмена", callback_data="cancel")]]
    )
