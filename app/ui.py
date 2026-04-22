import urllib.parse
import json
from telegram import (
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    KeyboardButton,
    ReplyKeyboardMarkup,
    WebAppInfo,
)
from app.config import WEB_APP_URL


def _format_settings(settings):
    voice_state = settings.get('voice_response')
    voice_labels = {
        "female": "ON (Kseniya)",
        "male": "ON (Aidar)",
        "kseniya": "ON (Kseniya)",
        "xenia": "ON (Xenia)",
        "baya": "ON (Baya)",
        "aidar": "ON (Aidar)",
        "eugene": "ON (Eugene)",
        "random": "ON (Random)"
    }
    if voice_state in voice_labels:
        voice_label = voice_labels[voice_state]
    elif voice_state:
        voice_label = "ON"
    else:
        voice_label = "OFF"

    lines = [
        "Current settings:",
        f"Trigger: {settings['trigger_word']}",
        f"Mood: {settings['mood'] or 'not set'}",
        f"Extra prompt: {settings['extra_prompt'] or 'not set'}",
        f"Response format: {settings['response_format'] or 'not set'}",
        (
            "Character limit: "
            f"{settings['max_response_chars']}"
            if settings["max_response_chars"] > 0
            else "Character limit: not set"
        ),
        f"Max reply (tokens): {settings['max_tokens']}",
        f"Syntax correction: {'ON' if settings.get('check_syntax') else 'OFF'}",
        f"Voice reply: {voice_label}",
        f"Random questions: {'ON' if settings.get('random_questions', True) else 'OFF'}",
        f"Question probability: {settings.get('random_question_prob', 0)}",
        f"Participation probability: {settings.get('random_participation_prob', 0)}",
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
        [[KeyboardButton("⚙️ Open Settings", web_app=WebAppInfo(url=web_app_url))]],
        resize_keyboard=True
    )


def _cancel_keyboard():
    return InlineKeyboardMarkup(
        [[InlineKeyboardButton("Cancel", callback_data="cancel")]]
    )
