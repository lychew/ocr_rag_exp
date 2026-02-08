"""Integration test — verifies config loading and factory wiring."""

from src.config import load_config
from src import factory


def test_load_default_config():
    cfg = load_config()
    assert "ocr" in cfg
    assert "chunking" in cfg
    assert "embedding" in cfg
    assert cfg["ocr"]["name"] == "tesseract"


def test_load_config_with_overrides():
    cfg = load_config(ocr_name="tesseract", chunking_name="page")
    assert cfg["ocr"]["name"] == "tesseract"
    assert cfg["chunking"]["name"] == "page"


def test_factory_create_chunker():
    chunker = factory.create("chunking", "page")
    assert chunker.name == "page"


def test_factory_unknown_raises():
    import pytest
    with pytest.raises(ValueError, match="Unknown component"):
        factory.create("ocr", "nonexistent")
