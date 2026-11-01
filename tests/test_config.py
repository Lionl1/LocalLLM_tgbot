import importlib

import pytest

import app.config


@pytest.fixture
def load_config(monkeypatch):
    def load(**env):
        for name in ("IMAGE_GENERATION_ENABLED", "IMAGE_GENERATION_WIDTH", "IMAGE_GENERATION_HEIGHT"):
            monkeypatch.delenv(name, raising=False)
        for name, value in env.items():
            monkeypatch.setenv(name, value)
        return importlib.reload(app.config)

    yield load
    monkeypatch.undo()
    importlib.reload(app.config)


def test_image_generation_defaults(load_config):
    config = load_config()
    assert config.IMAGE_GENERATION_ENABLED is True
    assert (config.IMAGE_GENERATION_WIDTH, config.IMAGE_GENERATION_HEIGHT) == (256, 256)


@pytest.mark.parametrize("value", ["0", "false", "no", "off"])
def test_image_generation_can_be_disabled(load_config, value):
    assert load_config(IMAGE_GENERATION_ENABLED=value).IMAGE_GENERATION_ENABLED is False


def test_image_size_from_env(load_config):
    config = load_config(IMAGE_GENERATION_WIDTH="512", IMAGE_GENERATION_HEIGHT="768")
    assert (config.IMAGE_GENERATION_WIDTH, config.IMAGE_GENERATION_HEIGHT) == (512, 768)
