"""Tests for OCR preprocessing."""

from src.ocr.preprocessing import (
    clean_ocr_text,
    collapse_whitespace,
    rejoin_broken_lines,
    rejoin_hyphenated_words,
    strip_page_numbers,
)


def test_rejoin_hyphenated_words():
    assert rejoin_hyphenated_words("preven-\ntion") == "prevention"
    assert rejoin_hyphenated_words("no hyphens here") == "no hyphens here"


def test_collapse_whitespace():
    assert collapse_whitespace("the    grounds   of") == "the grounds of"
    assert collapse_whitespace("a\t\tb") == "a b"


def test_strip_page_numbers():
    text = "Some text\n  42  \nMore text"
    result = strip_page_numbers(text)
    assert "42" not in result
    assert "Some text" in result


def test_rejoin_broken_lines():
    text = "The importance of\nclean water is\nvital."
    result = rejoin_broken_lines(text)
    assert "The importance of clean water is vital." in result


def test_clean_ocr_text_full():
    raw = "The impor-\ntance   of\n  47  \nclean water."
    result = clean_ocr_text(raw)
    assert "importance" in result
    assert "47" not in result
    assert "   " not in result
