"""Policy and persistence invariants for model-aware long-form prompts."""
from dataclasses import replace

import pytest

from tts_audiobook_tool.app_types import Book, BookSection, BookSegmentationSettings
from tts_audiobook_tool.app_types.book_serialization import (
    book_from_project_text_json_dict, book_to_project_text_json_dict,
)
from tts_audiobook_tool.app_types.phrase import Phrase, PhraseGroup, Reason
from tts_audiobook_tool.constants_config import ROLLING_CONTINUATION_MAX_WORDS
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.project_support.project_serialization_util import ProjectSerializationUtil
from tts_audiobook_tool.tts import Tts
from tts_audiobook_tool.tts_models import mlx_audio_definition
from tts_audiobook_tool.tts_models.segmentation_capabilities import (
    prompt_word_limit_issue,
    resolve_segmentation_capabilities,
)


@pytest.mark.parametrize("model_id", ["qwen3tts_mlxaudio", "fish_drama3_mlxaudio"])
def test_known_mlx_checkpoint_allows_extended_prompts_without_loading(model_id, monkeypatch):
    monkeypatch.setattr(Tts, "get_instance", lambda: pytest.fail("loaded inference model"))
    project = Project(tts_model_type=model_id)
    caps = resolve_segmentation_capabilities(project)
    assert (caps.default_words, caps.max_words) == (40, 320)
    assert (caps.recommended_min_words, caps.recommended_max_words) == (40, 80)
    assert not prompt_word_limit_issue(caps, 320)
    assert "320" in prompt_word_limit_issue(caps, 321)


def test_new_checkpoint_revision_keeps_default_ceiling(monkeypatch):
    model_id = "qwen3tts_mlxaudio"
    definition = mlx_audio_definition.get_mlx_audio_definition(model_id)
    checkpoint = replace(definition.checkpoints[0], revision="new-unqualified-revision")
    replacement = replace(definition, checkpoints=(checkpoint,))
    monkeypatch.setitem(mlx_audio_definition._DEFINITIONS, model_id, replacement)
    caps = resolve_segmentation_capabilities(Project(tts_model_type=model_id))
    assert caps.max_words == 80


@pytest.mark.parametrize("model_id,recommended", [
    ("glm_local", (20, 40)),
    ("indextts2_local", (40, 60)),
    ("vibevoice_local", (40, 80)),
])
def test_unknown_longform_models_keep_legacy_policy(model_id, recommended):
    caps = resolve_segmentation_capabilities(Project(tts_model_type=model_id))
    assert caps.max_words == 80
    assert (caps.recommended_min_words, caps.recommended_max_words) == recommended
    assert prompt_word_limit_issue(caps, 81)
    # Twenty words bounds the configurable ceiling, not individual short prompts.
    assert not prompt_word_limit_issue(caps, 1)


@pytest.mark.parametrize("value", [20, 40, 80, 120, 160, 240, 320])
def test_storage_setting_round_trips_without_active_backend_clamp(value):
    project = Project.model_validate({"tts_model_type": "glm_local", "max_words": value})
    assert project.max_words == value
    reloaded = Project.model_validate(ProjectSerializationUtil.to_project_json_dict(project))
    assert reloaded.max_words == value


@pytest.mark.parametrize("value", [True, False, "120", None, -1, 0, 19, 321, float("inf")])
def test_invalid_stored_limits_keep_legacy_default(value):
    assert Project.model_validate({"max_words": value}).max_words == 40


def test_long_book_roundtrip_preserves_identity_and_applied_settings():
    group = PhraseGroup([Phrase("A sentence. ", Reason.SENTENCE), Phrase("Another.\n", Reason.SECTION_BREAK)], voice_index=2)
    book = Book(
        sections=[BookSection(phrase_groups=[group], title="Chapter")],
        segmentation_settings=BookSegmentationSettings(max_words_per_segment=240),
    )
    saved = book_to_project_text_json_dict(book)
    loaded = book_from_project_text_json_dict(saved)
    assert isinstance(loaded, Book)
    assert book_to_project_text_json_dict(loaded) == saved
    assert loaded.phrase_groups[0].text == group.text
    assert loaded.phrase_groups[0].voice_index == 2
    # Existing Torch continuation context is independent of expanded MLX prompts.
    assert ROLLING_CONTINUATION_MAX_WORDS == 160


def test_legacy_recommendation_keeps_variant_label():
    from tts_audiobook_tool.tts_models.chatterbox_base_model import ChatterboxType
    project = Project(tts_model_type='chatterbox_local')
    project.set_model_setting('chatterbox_local', 'type', ChatterboxType.MULTILINGUAL_V2)
    caps = resolve_segmentation_capabilities(project)
    assert (caps.recommended_min_words, caps.recommended_max_words) == (40, 40)
    assert caps.recommendation_label == 'Chatterbox Multilingual V2'
