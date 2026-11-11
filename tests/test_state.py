import asyncio
import json

import pytest

from app import state


@pytest.fixture()
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(state, "DATA_DIR", tmp_path)
    for name, filename in [
        ("CHAT_SETTINGS_FILE", "chat_settings.json"),
        ("CHAT_HISTORY_FILE", "chat_history.json"),
        ("CHAT_KNOWLEDGE_FILE", "chat_knowledge.json"),
        ("CHAT_LOGS_FILE", "chat_logs.json"),
    ]:
        monkeypatch.setattr(state, name, tmp_path / filename)
    for name in ("CHAT_SETTINGS", "CHAT_MEMORY", "CHAT_KNOWLEDGE", "CHAT_LOGS"):
        monkeypatch.setattr(state, name, {})
    return tmp_path


def test_state_survives_a_restart(data_dir):
    state.CHAT_SETTINGS[-100] = {"trigger_word": "Эй"}
    state.CHAT_MEMORY[7] = [{"role": "user", "content": "привет"}]
    asyncio.run(state.persist_settings())
    asyncio.run(state.persist_history())
    state.CHAT_SETTINGS.clear()
    state.CHAT_MEMORY.clear()

    asyncio.run(state.load_persisted_chat_settings())

    assert state.CHAT_SETTINGS == {-100: {"trigger_word": "Эй"}}
    assert state.CHAT_MEMORY == {7: [{"role": "user", "content": "привет"}]}
    assert sorted(p.name for p in data_dir.iterdir()) == ["chat_history.json", "chat_settings.json"]


def test_failed_write_keeps_the_previous_file(data_dir, monkeypatch):
    state.CHAT_SETTINGS[1] = {"trigger_word": "old"}
    asyncio.run(state.persist_settings())

    def broken_replace(src, dst):
        raise OSError("disk full")

    monkeypatch.setattr(state.os, "replace", broken_replace)
    state.CHAT_SETTINGS[1] = {"trigger_word": "new"}
    asyncio.run(state.persist_settings())

    saved = json.loads((data_dir / "chat_settings.json").read_text(encoding="utf-8"))
    assert saved == {"1": {"trigger_word": "old"}}


def test_damaged_file_is_set_aside_not_overwritten(data_dir):
    damaged = '{"1": {"trigger_word": "half a fi'
    (data_dir / "chat_settings.json").write_text(damaged, encoding="utf-8")

    asyncio.run(state.load_persisted_chat_settings())
    assert state.CHAT_SETTINGS == {}
    asyncio.run(state.persist_settings())

    assert (data_dir / "chat_settings.json.corrupt").read_text(encoding="utf-8") == damaged
    assert json.loads((data_dir / "chat_settings.json").read_text(encoding="utf-8")) == {}
