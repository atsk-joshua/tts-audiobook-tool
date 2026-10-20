"""Settings and editor regressions use disposable projects, never production assets."""
from types import SimpleNamespace
import asyncio

import pytest
from textual.widgets import Static

from tts_audiobook_tool.app_types import Book, BookSection, BookSegmentationSettings, SegmentationStrategy
from tts_audiobook_tool.app_types.book_serialization import book_to_json_dict
from tts_audiobook_tool.app_types.phrase import Phrase, PhraseGroup, Reason
from tts_audiobook_tool.menus import text_menu
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.project_support.project_load_util import ProjectLoadUtil
from tts_audiobook_tool.project_support.project_text_io_util import ProjectTextIOUtil
from tts_audiobook_tool.project_support.sound_segment_util import SoundSegmentUtil
from tts_audiobook_tool.textual.text_editor import TextEditor
from tts_audiobook_tool.tts_models.segmentation_capabilities import SegmentationCapabilities


@pytest.fixture(autouse=True)
def no_native_directory_watcher(monkeypatch):
    # These tests explicitly rescan disposable files; filesystem event delivery
    # is outside their scope and can fail inside a macOS sandbox.
    import tts_audiobook_tool.project_support.project_sound_segments as segments
    monkeypatch.setattr(segments, "Observer", lambda: SimpleNamespace(
        schedule=lambda *args, **kwargs: None,
        start=lambda: None,
        is_alive=lambda: False,
    ))


def test_max_words_prompt_uses_model_policy_for_future_imports(monkeypatch):
    project = Project()
    messages = []
    captured = {}
    monkeypatch.setattr(text_menu.MenuUtil, "print_screen_heading", lambda *args: None)
    monkeypatch.setattr(text_menu, "printt", lambda message="": messages.append(message))
    monkeypatch.setattr(
        text_menu, "resolve_segmentation_capabilities",
        lambda _: SegmentationCapabilities(max_words=320),
    )

    def choose(saveable, **kwargs):
        captured.update(kwargs)
        saveable.max_words = 160

    monkeypatch.setattr(text_menu.ask, "ask_number_and_save", choose)
    text_menu.on_ask_max_size(SimpleNamespace(project=project), None)

    assert captured["max_value"] == 320
    assert captured["min_value"] == 20
    assert captured["default_value"] == 40
    assert any("future text imports" in message for message in messages)


def test_legacy_model_advisory_preserves_high_saved_setting(monkeypatch):
    project = Project(max_words=240)
    monkeypatch.setattr(
        text_menu, "resolve_segmentation_capabilities", lambda _: SegmentationCapabilities()
    )
    advisory = text_menu.make_segmentation_advisory(project, project.max_words)
    assert "240-word ceiling" in advisory
    assert "80-word configurable maximum" in advisory
    assert project.max_words == 240


def test_text_menu_shows_saved_model_mismatch_before_import(monkeypatch):
    project = Project(max_words=240)
    captured = {}
    monkeypatch.setattr(text_menu.MenuUtil, "menu", lambda *args, **kwargs: captured.update(kwargs))
    state = SimpleNamespace(project=project)
    text_menu.TextMenu.menu(state)
    assert "240-word ceiling" in captured["subheading"](state)
    assert project.max_words == 240


def test_import_info_warns_for_existing_long_book_with_narrower_backend():
    project = Project(book=Book(
        segmentation_settings=BookSegmentationSettings(max_words_per_segment=160),
        sections=[BookSection(phrase_groups=[PhraseGroup([Phrase("One.", Reason.SENTENCE)])])],
    ))
    editor = TextEditor(project)
    editor.load_content()

    async def exercise():
        async with editor.run_test() as pilot:
            await pilot.press("i")
            content = editor.screen.query_one("#segmentation-info-copy", Static).content.plain
            assert "Max words per segment: 160" in content
            assert "editor retains this book's applied limit" in content
            assert "configurable maximum is 80 words" in content
            assert "use a compatible model for generation" not in content

    asyncio.run(exercise())
    assert project.book.segmentation_settings.max_words_per_segment == 160


@pytest.mark.parametrize("model_id", ["none", "qwen3tts_mlxaudio", "fish_drama3_mlxaudio"])
def test_settings_and_model_switch_roundtrip_preserves_book_audio_and_markers(tmp_path, model_id):
    groups = [
        PhraseGroup([Phrase(" ".join(["narration"] * 100) + ". ", Reason.PARAGRAPH)], voice_index=0),
        PhraseGroup([Phrase("A second section.", Reason.SENTENCE)], voice_index=0),
    ]
    book = Book(
        title="Scratch migration fixture",
        segmentation_settings=BookSegmentationSettings(
            language_code="en", max_words_per_segment=160,
            strategy=SegmentationStrategy.MULTI_SENTENCE,
        ),
        sections=[BookSection(title="Chapter", phrase_groups=groups)],
    )
    project = Project.model_validate({"dir_path": str(tmp_path), "book": book, "markers": [1]})
    assert ProjectTextIOUtil.save_book(project) == ""
    assert project.save() == ""
    segment_hash = SoundSegmentUtil.calc_segment_hash(0, groups[0].text)
    audio = tmp_path / "segments" / f"[00001] [{segment_hash}] [qwen3tts] [narrator] fixture.flac"
    audio.write_bytes(b"preserved generated audio fixture")
    sidecar = audio.with_suffix(".json")
    sidecar.write_text('{"preserved": true}')
    before_book = book_to_json_dict(project.book)
    preserved = {path: path.read_bytes() for path in [tmp_path / "project_text.json", audio, sidecar]}

    project.max_words = 240
    project.tts_model_type = model_id
    project.segmentation_strategy = SegmentationStrategy.MAX_LEN
    assert project.save() == ""
    loaded = ProjectLoadUtil.load_using_dir_path(str(tmp_path), prompt_on_warnings=False)

    assert isinstance(loaded, Project)
    assert loaded.max_words == 240
    assert loaded.tts_model_type == model_id
    assert book_to_json_dict(loaded.book) == before_book
    assert loaded.markers == {1}
    assert loaded.sound_segments.get_best_file_for(0) == audio.name
    assert all(path.read_bytes() == content for path, content in preserved.items())
    # A narrower backend or a future-import value cannot reject a valid old edit.
    editor = TextEditor(loaded)
    assert editor.is_edit_word_count_valid(160)
    assert not editor.is_edit_word_count_valid(161)
    assert editor.make_edit_word_count_text(100).plain == "Words: 100/160"
    description = text_menu.make_segmentation_settings_description(loaded)
    assert "future imports" in description
    assert "Current book: 160 words per segment; multiple sentences." in description


@pytest.mark.parametrize('route', ['text', 'manual', 'epub'])
def test_future_import_rejects_preserved_incompatible_ceiling_without_touching_book(tmp_path, monkeypatch, route):
    project = Project(dir_path=str(tmp_path), tts_model_type='glm_local', max_words=240)
    assert project.save() == ''
    sentinel = tmp_path / 'segments' / 'existing.flac'
    sentinel.write_bytes(b'protected existing audio')
    messages = []
    monkeypatch.setattr(text_menu.MenuUtil, 'print_screen_heading', lambda *args: None)
    monkeypatch.setattr(text_menu.ask, 'ask_error', messages.append)
    monkeypatch.setattr(text_menu.ask, 'ask_confirm', lambda *args: pytest.fail('Confirmed destructive replacement'))
    assert text_menu.on_select_import(SimpleNamespace(project=project),
        text_menu.MenuItem('', lambda *_: None, data=route)) is False
    assert '240-word ceiling' in messages[0]
    assert 'before importing' in messages[0]
    assert sentinel.read_bytes() == b'protected existing audio'
    assert project.max_words == 240


@pytest.mark.parametrize('model_id', ['glm_local', 'chatterbox_local'])
def test_legacy_recommendation_does_not_label_existing_range_experimental(model_id):
    project = Project(tts_model_type=model_id)
    assert 'experimental' not in text_menu.make_segmentation_advisory(project, 40)
