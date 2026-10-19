from types import SimpleNamespace

import pytest

from tts_audiobook_tool.project import Project
from tts_audiobook_tool.tts import Tts
from tts_audiobook_tool.tts_models.tts_model_type import TtsModelType


@pytest.mark.parametrize('model_id', ['qwen3tts_mlxaudio', 'fish_drama3_mlxaudio'])
def test_actual_prepared_prompt_cap_checks_entire_batch_before_inference(monkeypatch, model_id):
    project = Project(tts_model_type=model_id)
    captured = []
    instance = SimpleNamespace(
        prepare_text_for_inference=lambda _project, text, **_kwargs: text,
        generate_using_project=lambda **kwargs: captured.append(kwargs) or [],
    )
    monkeypatch.setattr(Tts, '_type', TtsModelType.require_by_id(model_id))
    monkeypatch.setattr(Tts, 'get_instance', staticmethod(lambda: instance))
    error = Tts.generate_using_project(project, ['short prompt', 'word ' * 321], voice_selection_index=0)
    assert 'Prompt 2' in error
    assert '320' in error
    assert captured == []
    # The configured import value is deliberately irrelevant to existing audio.
    project.max_words = 40
    assert Tts.generate_using_project(project, ['word ' * 160], voice_selection_index=0) == []
    assert len(captured) == 1


def test_legacy_dispatch_preserves_unsplit_prompts_and_generated_tags(monkeypatch):
    project = Project(tts_model_type='vibevoice_local')
    captured = []
    instance = SimpleNamespace(
        prepare_text_for_inference=lambda _project, text, **_kwargs: 'Speaker 1: ' + text,
        generate_using_project=lambda **kwargs: captured.append(kwargs) or [],
    )
    monkeypatch.setattr(Tts, '_type', TtsModelType.require_by_id(project.tts_model_type))
    monkeypatch.setattr(Tts, 'get_instance', staticmethod(lambda: instance))
    assert Tts.generate_using_project(project, ['word ' * 80], voice_selection_index=0) == []
    assert captured[0]['prompts'][0].startswith('Speaker 1: ')
    assert Tts.generate_using_project(project, ['word ' * 81], voice_selection_index=0) == []
    assert len(captured) == 2


def test_cap_uses_substituted_prepared_text_and_rejects_empty(monkeypatch):
    project = Project(tts_model_type='qwen3tts_mlxaudio')
    captured = []
    instance = SimpleNamespace(
        prepare_text_for_inference=lambda _project, text, **_kwargs: 'word ' * 321 if text == 'expand' else '',
        generate_using_project=lambda **kwargs: captured.append(kwargs) or [],
    )
    monkeypatch.setattr(Tts, '_type', TtsModelType.require_by_id(project.tts_model_type))
    monkeypatch.setattr(Tts, 'get_instance', staticmethod(lambda: instance))
    assert '321' in Tts.generate_using_project(project, ['expand'], voice_selection_index=0)
    assert 'vocalizable' in Tts.generate_using_project(project, ['◇'], voice_selection_index=0)
    assert captured == []


@pytest.mark.parametrize('model_id', ['chatterbox_local', 'qwen3tts_sglomni', 'fish_drama3_audiocpp'])
def test_legacy_dispatch_retains_unsplit_prompt_behavior_after_model_switch(monkeypatch, model_id):
    project = Project(tts_model_type=model_id)
    project.max_words = 240  # Retained after switching away from an MLX model.
    captured = []
    instance = SimpleNamespace(
        prepare_text_for_inference=lambda _project, text, **_kwargs: text,
        generate_using_project=lambda **kwargs: captured.append(kwargs) or [],
    )
    monkeypatch.setattr(Tts, '_type', TtsModelType.require_by_id(model_id))
    monkeypatch.setattr(Tts, 'get_instance', staticmethod(lambda: instance))
    assert Tts.generate_using_project(
        project, ['short prompt', 'word ' * 81], voice_selection_index=0
    ) == []
    assert captured[0]['prompts'] == ['short prompt', 'word ' * 81]
    assert project.max_words == 240
    assert Tts.generate_using_project(project, ['word ' * 80], voice_selection_index=0) == []
    assert len(captured) == 2
