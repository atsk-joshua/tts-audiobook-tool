"""Application dispatch, persistence and runtime ownership contracts."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tts_audiobook_tool.app_types import Sound
from tts_audiobook_tool.app_types.voice import Voice
from tts_audiobook_tool.constants import PROJECT_SPEC_VERSION
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.project_support.project_serialization_util import ProjectSerializationUtil
from tts_audiobook_tool.tts import Tts, TtsRuntimeMode
from tts_audiobook_tool.tts_models.mlx_audio_definition import get_mlx_audio_definition
from tts_audiobook_tool.tts_models.tts_model_type import TtsBackendKind, TtsModelType


MODELS = ("qwen3tts_mlxaudio", "fish_drama3_mlxaudio")


@pytest.mark.parametrize("model_id", MODELS)
def test_app_dispatch_keeps_normal_prompt_pipeline_and_stream_callbacks(monkeypatch, model_id):
    model = TtsModelType.require_by_id(model_id)
    project = Project(tts_model_type=model_id)
    project.word_substitutions = {"old": "new"}
    captured = []
    chunks = []
    ends = []
    sound = Sound(np.ones(8, dtype=np.float32), model.value.default_output_sample_rate)

    def prepare(project, text, **kwargs):
        captured.append(("prepare", text, kwargs))
        return text.replace("old", "new")

    def generate(**kwargs):
        captured.append(("generate", kwargs))
        kwargs["on_stream_chunk"](sound.data)
        kwargs["on_stream_end"]()
        return [sound]

    monkeypatch.setattr(Tts, "_type", model)
    monkeypatch.setattr(Tts, "get_instance", staticmethod(lambda: SimpleNamespace(
        prepare_text_for_inference=prepare, generate_using_project=generate)))
    result = Tts.generate_using_project(project, ["old words"], on_stream_chunk=chunks.append,
                                       on_stream_end=lambda: ends.append(True), voice_selection_index=0)
    assert result == [sound]
    assert captured[0] == ("prepare", "old words", {"apply_word_substitutions": True})
    values = captured[1][1]
    assert values["prompts"] == ["new words"]
    assert values["voice_selection_index"] == 0
    assert "print_generation_request" not in values
    assert len(chunks) == 1 and chunks[0] is sound.data
    assert ends == [True]


@pytest.mark.parametrize("model_id", MODELS)
def test_real_project_json_snapshot_preserves_checkpoint_and_shared_voice_pairs(model_id):
    definition = get_mlx_audio_definition(model_id)
    project = Project(tts_model_type=model_id)
    project.voice_references = [Voice("first.wav", "First words.", pan=0.3,
                                      instructions={"qwen3tts_local": "Speak softly."},
                                      presets={"pocket_local": "alba"}),
                                Voice("second.wav", "Second words.")]
    checkpoint = definition.checkpoints[-1]
    project.set_model_setting(model_id, "checkpoint", checkpoint.repo_id)
    payload = ProjectSerializationUtil.to_project_json_dict(project)
    restored = Project.model_validate(json.loads(json.dumps(payload)))
    assert restored.tts_model_type == model_id
    assert restored.voice_references == project.voice_references
    assert definition.get_checkpoint(restored) == checkpoint
    assert payload["version"] == PROJECT_SPEC_VERSION


@pytest.mark.parametrize("model_id", MODELS)
def test_local_project_binding_never_contacts_remote_discovery(monkeypatch, model_id):
    from tts_audiobook_tool.app_support.remote_tts_discovery import RemoteTtsDiscovery
    monkeypatch.setattr(Tts, "_backend_mode", TtsRuntimeMode.LOCAL)
    monkeypatch.setattr(Tts, "_available_local_models", tuple(TtsModelType.require_by_id(item) for item in MODELS))
    def forbidden(*args, **kwargs):
        pytest.fail("MLX local binding called remote discovery")
    monkeypatch.setattr(RemoteTtsDiscovery, "refresh", forbidden)
    project = Project(tts_model_type=model_id)
    assert Tts.bind_project(project) is None
    assert Tts.get_active_type().id == model_id
    assert Tts.get_active_type().value.backend_kind is TtsBackendKind.MLX_AUDIO
    assert Tts.get_backend_mode() is TtsRuntimeMode.LOCAL


def test_ui_cannot_construct_inference_runtime(monkeypatch):
    from tts_audiobook_tool import model_runtime
    monkeypatch.setattr(model_runtime, "_role", model_runtime.ModelRuntimeRole.INTERACTIVE_MAIN)
    with pytest.raises(RuntimeError, match="ModelWorker"):
        Tts.get_instance()


@pytest.mark.parametrize("model_id", MODELS)
def test_real_app_entrypoint_through_shared_runtime_uses_copied_reference_and_audio(monkeypatch, tmp_path, model_id):
    import soundfile as sf
    from tts_audiobook_tool import model_runtime
    from tts_audiobook_tool.tts_models.mlx_audio_runtime import get_mlx_audio_runtime_class
    monkeypatch.setattr(model_runtime, "_role", model_runtime.ModelRuntimeRole.STANDALONE)
    model_type = TtsModelType.require_by_id(model_id)
    rate = model_type.value.default_output_sample_rate
    voice_dir = tmp_path / "voice"
    voice_dir.mkdir()
    reference = voice_dir / "reference.wav"
    sf.write(reference, np.full(2400, 0.1, dtype=np.float32), 24000)
    original = reference.read_bytes()
    project = Project(tts_model_type=model_id)
    project.dir_path = str(tmp_path)
    project.voice_references = [Voice(reference.name, "Exact reference words.")]
    project.word_substitutions = {"old": "new"}
    generated = []
    loaded = []
    prepared_refs = []

    class Policy:
        supports_multiple_references = False
        reference_sample_rate = 24000
        def build_kwargs(self, values, language_code, references, stream):
            prepared_refs.extend(references)
            return {"stream": stream, "max_tokens": 100}
        def validate_model(self, model):
            pass
        def clear_reference_cache(self, model):
            pass
        def reset_stream_state(self, model):
            pass

    class Model:
        sample_rate = rate
        def generate(self, text, **kwargs):
            generated.append((text, kwargs))
            yield SimpleNamespace(audio=np.full(4, 0.2), sample_rate=rate, token_count=2, segment_idx=0)
            yield SimpleNamespace(audio=np.full(3, 0.3), sample_rate=rate, token_count=1, segment_idx=0)
            yield SimpleNamespace(audio=np.array([]), sample_rate=rate, is_final_chunk=True, token_count=0, segment_idx=0)

    def loader(repo_id, **kwargs):
        loaded.append((repo_id, kwargs))
        return Model()

    mx = SimpleNamespace(random=SimpleNamespace(seed=lambda seed: None), synchronize=lambda: None,
                         clear_cache=lambda: None, set_cache_limit=lambda limit: 0)
    runtime = get_mlx_audio_runtime_class(model_id)(model_type, loader=loader, mx=mx, policy=Policy())
    monkeypatch.setattr(Tts, "_type", model_type)
    monkeypatch.setattr(Tts, "get_instance", staticmethod(lambda: runtime))
    callbacks = []
    def callback(chunk):
        callbacks.append(chunk.copy())
        chunk[:] = 0  # Consumer mutation must not alter assembled Sound.
    try:
        assert loaded == []
        result = Tts.generate_using_project(project, ["old words."], on_stream_chunk=callback, voice_selection_index=0)
        assert not isinstance(result, str), result
        assert len(result) == 1 and result[0].sr == rate
        np.testing.assert_allclose(result[0].data, [0.2] * 4 + [0.3] * 3)
        assert generated[0][0] == "new words."
        assert len(loaded) == 1 and loaded[0][1]["strict"] is True
        assert generated[0][1]["stream"] is model_type.value.can_stream
        assert len(callbacks) == (2 if model_type.value.can_stream else 0)
        assert prepared_refs[0].transcript == "Exact reference words."
        assert Path(prepared_refs[0].path).resolve() != reference.resolve()
        assert reference.read_bytes() == original
    finally:
        runtime.kill()
    assert not Path(prepared_refs[0].path).exists()
