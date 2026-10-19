import logging
from types import SimpleNamespace

import pytest

from tts_audiobook_tool.tts_models.mlx_audio_qwen import QwenPolicy


def reference(transcript="  The reference's exact words.\n", path="/scratch/narrator.wav"):
    return SimpleNamespace(path=path, transcript=transcript, fingerprint="reference-content")


def test_production_recipe_uses_public_language_keyword_and_exact_reference(caplog):
    ref = reference()
    with caplog.at_level(logging.INFO):
        kwargs = QwenPolicy().build_kwargs({}, "en", [ref], False)
    assert kwargs == {
        "temperature": 0.9,
        "top_p": 1.0,
        "top_k": 50,
        "repetition_penalty": 1.05,
        "max_tokens": 1600,
        "lang_code": "english",
        "ref_audio": ref.path,
        "ref_text": ref.transcript,
        "stream": False,
    }
    assert "requested repetition_penalty=1.05" in caplog.text
    assert "effective repetition_penalty=1.5" in caplog.text


def test_explicit_sampling_is_preserved_and_stream_is_explicit(caplog):
    values = dict(temperature=0, top_p=0.85, top_k=0, repetition_penalty=1.7, max_tokens=12)
    with caplog.at_level(logging.INFO):
        kwargs = QwenPolicy().build_kwargs(values, "english", [reference()], True)
    assert {key: kwargs[key] for key in values} == values
    assert kwargs["stream"] is True
    assert "effective repetition_penalty=1.7" in caplog.text


@pytest.mark.parametrize("code,expected", [
    ("en-US", "english"), ("ZH_hant", "chinese"), ("ja", "japanese"),
    ("French", "french"), ("auto", "auto"), ("", "auto"),
])
def test_language_mapping(code, expected):
    assert QwenPolicy().build_kwargs({}, code, [reference()], False)["lang_code"] == expected


@pytest.mark.parametrize("code", ["xx", "ar", None])
def test_unsupported_language_fails_instead_of_silent_auto(code):
    with pytest.raises(ValueError, match="language"):
        QwenPolicy().build_kwargs({}, code, [reference()], False)


@pytest.mark.parametrize("values", [
    {"speed": 1.1}, {"instruct": "happy"}, {"language": "English"},
    {"seed": 5}, {"voice": "Ryan"}, {"streaming_interval": 0.3},
])
def test_unknown_controls_fail_instead_of_upstream_silent_ignore(values):
    with pytest.raises(ValueError, match="Unsupported.*controls"):
        QwenPolicy().build_kwargs(values, "en", [reference()], False)


@pytest.mark.parametrize("values", [
    {"temperature": float("nan")}, {"temperature": float("inf")},
    {"temperature": -1}, {"temperature": True}, {"temperature": "0.9"},
    {"top_p": 0}, {"top_p": 1.1}, {"top_k": -1}, {"top_k": 2.5},
    {"max_tokens": 0}, {"max_tokens": 3.0}, {"max_tokens": True},
    {"repetition_penalty": 0}, {"repetition_penalty": None},
])
def test_invalid_sampling_rejected(values):
    with pytest.raises(ValueError):
        QwenPolicy().build_kwargs(values, "en", [reference()], False)


@pytest.mark.parametrize("refs", [
    [], [reference(), reference()], [reference(transcript=" \n")],
    [reference(transcript=None)], [reference(path="")],
])
def test_reference_must_be_single_path_with_exact_nonempty_transcript(refs):
    with pytest.raises(ValueError, match="reference"):
        QwenPolicy().build_kwargs({}, "en", refs, False)


def test_stream_requires_boolean():
    with pytest.raises(ValueError, match="boolean"):
        QwenPolicy().build_kwargs({}, "en", [reference()], "false")


@pytest.mark.parametrize("model", [
    SimpleNamespace(model_type="qwen3_tts", config=SimpleNamespace(tts_model_type="base")),
    SimpleNamespace(config={"tts_model_type": "base"}), SimpleNamespace(),
])
def test_base_model_validation(model):
    QwenPolicy().validate_model(model)


@pytest.mark.parametrize("model", [
    SimpleNamespace(model_type="fish_drama3"),
    SimpleNamespace(config=SimpleNamespace(tts_model_type="custom_voice")),
    SimpleNamespace(config={"tts_model_type": "voice_design"}),
])
def test_incompatible_model_validation(model):
    with pytest.raises(ValueError):
        QwenPolicy().validate_model(model)


def test_reference_invalidation_releases_native_cache_and_resets_decoder():
    reset_calls = []
    cached = object()
    cache = {("transcript", (100, 0.0)): cached}
    model = SimpleNamespace(
        _icl_cache=cache,
        speech_tokenizer=SimpleNamespace(
            decoder=SimpleNamespace(reset_streaming_state=lambda: reset_calls.append(True))
        ),
    )
    policy = QwenPolicy()
    policy.clear_reference_cache(model)
    policy.reset_stream_state(model)
    assert cache == {}
    assert reset_calls == [True]


def test_cleanup_before_fully_initialized_model_is_safe():
    policy = QwenPolicy()
    policy.clear_reference_cache(SimpleNamespace())
    policy.reset_stream_state(SimpleNamespace(speech_tokenizer=None))
