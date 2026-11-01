from types import SimpleNamespace

from telegram.constants import ChatType

from app.text_utils import (
    _estimate_messages_tokens,
    _extract_prompt,
    _get_command_text,
    _is_triggered,
    _split_message,
    _split_reset_request,
    detect_transcription_request,
)


def make_update(chat_type=ChatType.GROUP, reply_from_id=None):
    reply = None
    if reply_from_id is not None:
        reply = SimpleNamespace(from_user=SimpleNamespace(id=reply_from_id), forward_from=None)
    message = SimpleNamespace(chat=SimpleNamespace(type=chat_type), reply_to_message=reply)
    return SimpleNamespace(message=message)


def test_private_chat_always_triggers():
    assert _is_triggered(make_update(ChatType.PRIVATE), "", 1, "bot", "Нука")


def test_group_triggers_on_word_mention_or_reply():
    assert _is_triggered(make_update(), "нука, расскажи", 1, "bot", "Нука")
    assert _is_triggered(make_update(), "@bot hi", 1, "bot", "Нука")
    assert _is_triggered(make_update(reply_from_id=1), "hi", 1, "bot", "Нука")
    assert not _is_triggered(make_update(), "просто текст", 1, "bot", "Нука")


def test_extract_prompt_strips_trigger_and_mention():
    assert _extract_prompt("Нука, @bot расскажи анекдот", "bot", "Нука") == "расскажи анекдот"


def test_split_reset_request():
    assert _split_reset_request("Сброс. Давай заново") == (True, "Давай заново")
    assert _split_reset_request("/reset") == (True, "")
    assert _split_reset_request("resetting things") == (False, "")


def test_get_command_text():
    assert _get_command_text("/search погода в Тюмени") == "погода в Тюмени"
    assert _get_command_text("/search") == ""


def test_split_message_respects_limit_and_keeps_text():
    text = "\n".join(f"line {i}" for i in range(200))
    chunks = _split_message(text, limit=100)
    assert all(len(chunk) <= 100 for chunk in chunks)
    assert " ".join(chunks).split() == text.split()


def test_split_message_closes_and_reopens_code_fences():
    text = "```\n" + "\n".join("x = 1" for _ in range(60)) + "\n```"
    chunks = _split_message(text, limit=100)
    assert len(chunks) > 1
    assert all(chunk.count("```") % 2 == 0 for chunk in chunks)


def test_estimate_messages_tokens():
    assert _estimate_messages_tokens([{"content": "abcd" * 10}, {"content": ""}]) == 4 + 10 + 4


def test_detect_transcription_request():
    assert detect_transcription_request("Расшифруй, пожалуйста")
    assert not detect_transcription_request("привет")
