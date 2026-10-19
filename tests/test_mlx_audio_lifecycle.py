"""Application ownership and residency regressions discovered in independent review."""
from types import SimpleNamespace

import pytest

from tts_audiobook_tool import model_worker
from tts_audiobook_tool.app_support.interrupts import Interrupts
from tts_audiobook_tool.model_manager import ModelManager
from tts_audiobook_tool.stt import Stt
from tts_audiobook_tool.tts import Tts
from tts_audiobook_tool.tts_models.mlx_audio_definition import get_mlx_audio_definition
from tts_audiobook_tool.tts_models.mlx_audio_policy import get_policy
from tts_audiobook_tool.tts_models.tts_model_type import TtsModelType


@pytest.mark.parametrize("loaded", [False, True])
def test_preview_watchdog_tracks_weights_in_lazy_wrapper(monkeypatch, loaded):
    monkeypatch.setattr(model_worker, "_preview_inferences_this_process", 3)
    monkeypatch.setattr(Tts, "get_instance_if_exists", lambda: SimpleNamespace(is_loaded=loaded))
    assert model_worker._should_watch_preview_inference() is loaded


def test_cache_cleanup_error_still_disposes_runtime_and_clears_ownership(monkeypatch):
    calls = []
    def bad_clear():
        raise RuntimeError("upstream cache failure")
    instance = SimpleNamespace(clear_voice_clone_cache=bad_clear, kill=lambda: calls.append("kill"))
    monkeypatch.setattr(Tts, "get_instance_if_exists", lambda: instance)
    monkeypatch.setattr(Tts, "_mlx_audio", instance)
    monkeypatch.setattr("tts_audiobook_tool.tts.app_memory.gc_ram_vram", lambda: None)
    with pytest.raises(RuntimeError, match="cache failure"):
        Tts.clear_tts_model()
    assert calls == ["kill"]
    assert Tts._mlx_audio is None


def test_model_warmup_explicitly_loads_lazy_mlx_weights(monkeypatch):
    calls = []
    project = SimpleNamespace()
    instance = SimpleNamespace(prepare_for_project=lambda p: calls.append(p))
    monkeypatch.setattr(Tts, "_type", TtsModelType.require_by_id("qwen3tts_mlxaudio"))
    monkeypatch.setattr(Tts, "instance_exists", lambda: False)
    monkeypatch.setattr(Tts, "get_instance", lambda: instance)
    monkeypatch.setattr(Stt, "should_skip", lambda _: True)
    monkeypatch.setattr(Stt, "has_instance", lambda: False)
    monkeypatch.setattr(Stt, "clear_stt_model", lambda: None)
    monkeypatch.setattr(ModelManager, "clear_yamnet_detector", lambda: None)
    result = ModelManager.warm_up_models(SimpleNamespace(project=project), skip_yamnet=True, want_stt=False)
    assert not result.should_stop
    assert calls == [project]


@pytest.mark.parametrize("cancelled", [False, True])
def test_mlx_warmup_distinguishes_cancellation_from_load_failure(monkeypatch, cancelled):
    interrupts = Interrupts()
    def prepare(project):
        monkeypatch.setattr(interrupts, "_flag", cancelled)
        raise RuntimeError("load stopped")
    monkeypatch.setattr(Tts, "_type", TtsModelType.require_by_id("fish_drama3_mlxaudio"))
    monkeypatch.setattr(Tts, "instance_exists", lambda: False)
    monkeypatch.setattr(Tts, "get_instance", lambda: SimpleNamespace(prepare_for_project=prepare))
    monkeypatch.setattr(Stt, "should_skip", lambda _: True)
    monkeypatch.setattr(Stt, "has_instance", lambda: False)
    monkeypatch.setattr(Stt, "clear_stt_model", lambda: None)
    monkeypatch.setattr(ModelManager, "clear_yamnet_detector", lambda: None)
    monkeypatch.setattr("tts_audiobook_tool.model_manager.app_memory.gc_ram_vram", lambda: None)
    result = ModelManager.warm_up_models(SimpleNamespace(project=object()), skip_yamnet=True, want_stt=False)
    assert result.did_interrupt is cancelled
    assert result.error == ("" if cancelled else "load stopped")


@pytest.mark.parametrize("model_id", ["qwen3tts_mlxaudio", "fish_drama3_mlxaudio"])
def test_catalog_parameter_endpoints_reach_real_family_policy(tmp_path, model_id):
    import numpy as np
    import soundfile as sf
    from tts_audiobook_tool.tts_models.mlx_audio_policy import PreparedReference
    path = tmp_path / "reference.wav"
    sf.write(path, np.ones(24000, dtype=np.float32) * .1, 24000)
    references = [PreparedReference(str(path), "An exact transcript.", "hash")]
    definition = get_mlx_audio_definition(model_id)
    policy = get_policy(definition.family)
    defaults = {name: param["default"] for name, param in definition.parameters.items()}
    for name, param in definition.parameters.items():
        for endpoint in ("min", "max"):
            policy.build_kwargs({**defaults, name: param[endpoint]}, "en", references, False)
