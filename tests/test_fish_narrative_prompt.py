"""Scoped Fish preparation preserves controls without changing legacy prompts."""
from types import SimpleNamespace

import pytest

from tts_audiobook_tool.app_types import Book, BookSegmentationSettings, SegmentationStrategy
from tts_audiobook_tool.app_types.book_serialization import (
    segmentation_settings_from_json_dict,
    segmentation_settings_to_json_dict,
)
from tts_audiobook_tool.tts_models.mlx_audio_base_model import (
    FishDrama3MlxAudioBaseModel as Fish,
    QwenMlxAudioBaseModel as Qwen,
)


def project(*, applied="normal", configured="narrative", substitutions=None):
    strategy = SegmentationStrategy.from_id(applied) or SimpleNamespace(id=applied)
    return SimpleNamespace(
        language_code="en", word_substitutions=substitutions or {},
        segmentation_strategy=SimpleNamespace(id=configured),
        book=Book(sections=[], segmentation_settings=BookSegmentationSettings(strategy=strategy)),
    )


def test_explicit_helper_preserves_newline_delimiters_and_normalizes_prose():
    text = "  First 2 stars.\r\n\r\nSecond 3 stars.\n\nLast 4.\r"
    assert Fish.prepare_narrative_text_for_inference(project(), text) == (
        "First two stars.\r\n\r\nSecond three stars.\n\nLast four.\r")


def test_standard_fish_and_qwen_preparation_remain_unchanged():
    text = "First 2.\n\nSecond 3."
    p = project(applied="normal", configured="narrative")
    assert Fish.prepare_text_for_inference(p, text) == "First two. Second three."
    assert Qwen.prepare_text_for_inference(p, text) == "First two. Second three."


def test_applied_narrative_preserves_paragraphs_without_mutating_book_or_qwen():
    p = project(applied="narrative", configured="normal")
    book, settings = p.book, p.book.segmentation_settings
    text = "First 2.\r\n\r\nSecond 3.\n"
    assert Fish.prepare_text_for_inference(p, text) == "First two.\r\n\r\nSecond three.\n"
    assert Qwen.prepare_text_for_inference(p, text) == "First two. Second three."
    assert p.book is book and p.book.segmentation_settings is settings


def test_persisted_applied_narrative_strategy_selects_preparation():
    strategy = SegmentationStrategy.from_id("narrative")
    assert strategy is not None
    settings = BookSegmentationSettings(strategy=strategy)
    saved = segmentation_settings_to_json_dict(settings)
    assert saved["strategy"] == "narrative"
    restored = segmentation_settings_from_json_dict(saved)
    assert restored == settings
    p = project(configured="normal")
    p.book.segmentation_settings = restored
    assert Fish.prepare_text_for_inference(p, "First 2.\n\nSecond 3.") == "First two.\n\nSecond three."


@pytest.mark.parametrize("applied", ["normal", "unknown", "", None])
def test_unknown_or_legacy_applied_strategy_defaults_to_standard_preparation(applied):
    p = project(applied=applied)
    assert Fish.prepare_text_for_inference(p, "First 2.\n\nSecond 3.") == "First two. Second three."


@pytest.mark.parametrize("missing", ["book", "settings", "strategy"])
def test_missing_applied_metadata_defaults_to_standard_preparation(missing):
    p = project()
    if missing == "book":
        del p.book
    elif missing == "settings":
        p.book = SimpleNamespace()
    else:
        p.book.segmentation_settings = SimpleNamespace()
    assert Fish.prepare_text_for_inference(p, "First 2.\n\nSecond 3.") == "First two. Second three."


def test_applied_narrative_dispatch_retains_tags_and_substitution_option():
    p = project(applied="narrative", configured="normal", substitutions={"Darian": "Dare Ian"})
    text = "<|speaker:0|> Darian saw 2 moons.\n\nDarian left."
    assert Fish.prepare_text_for_inference(p, text) == "<|speaker:0|> Dare Ian saw two moons.\n\nDare Ian left."
    assert Fish.prepare_text_for_inference(p, text, apply_word_substitutions=False) == (
        "<|speaker:0|> Darian saw two moons.\n\nDarian left.")
    with pytest.raises(ValueError, match="Fish narrative"):
        Fish.prepare_text_for_inference(p, "<|speaker:16|> Darian left.")


def test_substitutions_and_literal_speaker_controls_are_separate():
    p = project(substitutions={"Darian": "Dare Ian"})
    text = "<|speaker:0|> Darian saw 2 moons.\n<|speaker:15|> Darian saw 3."
    assert Fish.prepare_narrative_text_for_inference(p, text) == (
        "<|speaker:0|> Dare Ian saw two moons.\n<|speaker:15|> Dare Ian saw three.")
    assert Fish.prepare_narrative_text_for_inference(
        p, text, apply_word_substitutions=False) == (
            "<|speaker:0|> Darian saw two moons.\n<|speaker:15|> Darian saw three.")


@pytest.mark.parametrize("text", [
    "<|speaker:16|> Hello", "<|speaker:00|> Hello", "<|speaker:-1|> Hello",
    "<|speaker:zero|> Hello", "<|speaker|> Hello", "<|speaker:0> Hello",
    "Hello <|im_start|>", "<|im_end|>", "<|voice|>", "<|text|>",
    "<|interleave|>", "<|semantic:123|>",
])
def test_invalid_or_structural_controls_rejected_without_loading(text):
    with pytest.raises(ValueError, match="Fish narrative"):
        Fish.prepare_narrative_text_for_inference(project(), text)


@pytest.mark.parametrize("replacement", ["<|im_start|>", "<|speaker:0|>", "＜|im_start|＞"])
def test_normalization_cannot_inject_control_structure(replacement):
    with pytest.raises(ValueError, match="Fish narrative"):
        Fish.prepare_narrative_text_for_inference(project(substitutions={"alias": replacement}), "alias")


def test_legacy_fish_tag_normalization_contract_is_not_silently_changed():
    assert Fish.prepare_text_for_inference(project(), "<|speaker:0|> Hi.\nThere.") == (
        "<|speaker:zero|> Hi. There.")
