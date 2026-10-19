"""Fish family contracts without importing MLX or loading model weights."""
from collections import OrderedDict
from types import SimpleNamespace

import pytest

from tts_audiobook_tool.tts_models.mlx_audio_fish import FishDrama3Policy


def reference(path="narrator.wav", transcript="Exact narrator transcript."):
    return SimpleNamespace(path=path, transcript=transcript, fingerprint="content-hash")


def model():
    resets = []
    return SimpleNamespace(
        config=SimpleNamespace(model_type="fish_drama3", speech_template_version="reference-bracket-v1"),
        sample_rate=44_100,
        codec=SimpleNamespace(reset_streaming_state=lambda: resets.append(True)),
        _voices=SimpleNamespace(_voices=OrderedDict([("voice", object())])),
        generate=lambda **kwargs: iter(()), resets=resets,
    )


def test_explicit_native_defaults_and_single_reference():
    policy = FishDrama3Policy()
    assert policy.reference_sample_rate == 44_100
    assert policy.supports_multiple_references
    kwargs = policy.build_kwargs({}, "en", [reference()], False)
    assert kwargs == dict(temperature=0.7, top_p=0.7, top_k=30, max_tokens=1024,
                         ref_audio="narrator.wav", ref_text="Exact narrator transcript.", stream=True)
    assert "lang_code" not in kwargs and "text" not in kwargs


def test_multiple_references_preserve_order_transcripts_and_cues():
    refs = [reference("one.wav", " [whispers] First. "), reference("two.wav", "<|speaker:0|> Second.")]
    kwargs = FishDrama3Policy().build_kwargs(
        dict(temperature=0.8, top_p=1.0, top_k=50, max_tokens=1600), "fr", refs, True)
    assert kwargs["ref_audio"] == ["one.wav", "two.wav"]
    assert kwargs["ref_text"] == [r.transcript for r in refs]
    assert kwargs["stream"] is True
    assert kwargs["max_tokens"] == 1600
    assert kwargs["temperature"] == 0.8


@pytest.mark.parametrize("setting", ["speed", "instruct", "repetition_penalty", "seed", "chunk_length", "rolling_cont"])
def test_never_silently_accept_unsupported_controls(setting):
    with pytest.raises(ValueError, match="Unsupported Fish Drama 3 settings"):
        FishDrama3Policy().build_kwargs({setting: 1}, "en", [reference()], False)


@pytest.mark.parametrize("name,value", [
    ("temperature", 0), ("temperature", float("nan")), ("temperature", True),
    ("top_p", 0), ("top_p", 1.1), ("top_p", float("inf")),
    ("top_k", 0), ("top_k", 1.5), ("top_k", True),
    ("max_tokens", 0), ("max_tokens", None), ("max_tokens", 2.5),
])
def test_invalid_sampling_fails_before_inference(name, value):
    with pytest.raises(ValueError, match=name):
        FishDrama3Policy().build_kwargs({name: value}, "en", [reference()], False)


@pytest.mark.parametrize("refs", [[], [reference(transcript=" ")],
                                   [reference(), reference(transcript="")], [reference(path="")]])
def test_every_reference_needs_path_and_exact_transcript(refs):
    with pytest.raises(ValueError, match="reference|transcript"):
        FishDrama3Policy().build_kwargs({}, "en", refs, False)


def test_validate_model_and_release_only_reference_conditioning():
    fake = model()
    policy = FishDrama3Policy()
    policy.validate_model(fake)
    policy.clear_reference_cache(fake)
    assert fake._voices._voices == {}
    assert fake.resets == []
    policy.reset_stream_state(fake)
    policy.reset_stream_state(fake)
    assert fake.resets == [True, True]
    assert callable(fake.generate)


@pytest.mark.parametrize("change,error", [
    (lambda m: setattr(m.config, "model_type", "fish_s2_pro"), "checkpoint"),
    (lambda m: setattr(m.config, "speech_template_version", "unknown"), "template"),
    (lambda m: setattr(m, "sample_rate", 24000), "44100"),
    (lambda m: setattr(m.codec, "reset_streaming_state", None), "reset"),
    (lambda m: setattr(m._voices, "_voices", None), "cache API"),
    (lambda m: setattr(m, "generate", None), "generate"),
])
def test_reject_wrong_family_and_incompatible_api(change, error):
    fake = model()
    change(fake)
    with pytest.raises(ValueError, match=error):
        FishDrama3Policy().validate_model(fake)
