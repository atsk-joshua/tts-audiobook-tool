"""CPU-only tests of ownership, reference identity and upstream stream contracts."""
from collections import OrderedDict
from dataclasses import replace
import os
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

from tts_audiobook_tool.app_support.interrupts import Interrupts
from tts_audiobook_tool import model_runtime
from tts_audiobook_tool.app_types.voice import Voice, VoiceCrop
from tts_audiobook_tool.tts_models.mlx_audio_base_model import get_mlx_audio_base_class
from tts_audiobook_tool.tts_models.mlx_audio_definition import get_mlx_audio_definition
from tts_audiobook_tool.tts_models.mlx_audio_fish import FishDrama3Policy
from tts_audiobook_tool.tts_models.mlx_audio_runtime import get_mlx_audio_runtime_class
from tts_audiobook_tool.tts_models.tts_model_type import TtsModelType


MODEL_ID = "qwen3tts_mlxaudio"


class FakePolicy:
    reference_sample_rate = 24000
    supports_multiple_references = False

    def __init__(self):
        self.clears = 0
        self.resets = 0
        self.calls = []

    def validate_model(self, model):
        pass

    def clear_reference_cache(self, model):
        self.clears += 1

    def reset_stream_state(self, model):
        self.resets += 1

    def build_kwargs(self, values, language_code, references, stream):
        self.calls.append((values, language_code, references, stream))
        return {**values, "stream": stream}


class FakeModel:
    sample_rate = 24000

    def __init__(self, chunks):
        self.chunks = chunks
        self.calls = []
        self.closed = 0

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        try:
            for chunk in self.chunks:
                if isinstance(chunk, Exception):
                    raise chunk
                if callable(chunk):
                    chunk = chunk()
                yield chunk
        finally:
            self.closed += 1


def chunk(audio=(0.1, 0.2), *, rate=24000, final=False, **extra):
    return SimpleNamespace(audio=np.array(audio), sample_rate=rate,
                           is_final_chunk=final, **extra)


@pytest.fixture(autouse=True)
def reset_interrupts():
    Interrupts().clear()
    Interrupts().set_external_event(None)
    original_role = model_runtime.current_role()
    model_runtime.mark_standalone()
    yield
    Interrupts().clear()
    Interrupts().set_external_event(None)
    if original_role is model_runtime.ModelRuntimeRole.INTERACTIVE_MAIN:
        model_runtime.mark_interactive_main()
    elif original_role is model_runtime.ModelRuntimeRole.MODEL_WORKER:
        model_runtime.mark_model_worker()


def make_runtime(tmp_path, *, chunks=None, model_id=MODEL_ID, with_voice=True):
    definition = get_mlx_audio_definition(model_id)
    settings = {name: parameter["default"] for name, parameter in definition.parameters.items()}
    settings.update(checkpoint=definition.checkpoints[0].repo_id, seed=42, reference_mode="selected")
    voice_dir = tmp_path / "voice"
    voice_dir.mkdir(exist_ok=True)
    sf.write(voice_dir / "speaker.wav", np.array([0.2, -0.2, 0.1]), 48000, subtype="PCM_16")
    project = SimpleNamespace(
        dir_path=str(tmp_path), language_code="en",
        voice_references=[Voice("speaker.wav", "reference words")] if with_voice else [Voice()],
        get_model_setting=lambda model_id, name: settings[name],
    )
    policy = FakePolicy()
    fake = FakeModel(chunks if chunks is not None else [chunk()])
    fake.sample_rate = definition.spec.default_output_sample_rate
    loads = []
    events = []
    seeds = []
    cache_limits = []
    allocator = SimpleNamespace(cache_limit=9 * 1024**3)

    def set_cache_limit(limit):
        previous = allocator.cache_limit
        allocator.cache_limit = limit
        cache_limits.append(limit)
        return previous

    def loader(*args, **kwargs):
        loads.append((args, kwargs))
        events.append("load")
        return fake

    mx = SimpleNamespace(random=SimpleNamespace(seed=seeds.append),
                         synchronize=lambda: events.append("sync"),
                         clear_cache=lambda: events.append("clear"),
                         set_cache_limit=set_cache_limit)
    runtime = get_mlx_audio_runtime_class(model_id)(TtsModelType.require_by_id(model_id),
                                                   loader=loader, mx=mx, policy=policy)
    return SimpleNamespace(runtime=runtime, project=project, settings=settings,
                           fake=fake, policy=policy, loads=loads, seeds=seeds, events=events,
                           allocator=allocator, cache_limits=cache_limits)


def test_family_metadata_is_bound_to_class_not_last_constructed_runtime(tmp_path):
    qwen = get_mlx_audio_base_class(MODEL_ID)
    fish = get_mlx_audio_base_class("fish_drama3_mlxaudio")
    make_runtime(tmp_path, model_id="fish_drama3_mlxaudio")
    assert qwen.INFO.id == MODEL_ID
    assert fish.INFO.id == "fish_drama3_mlxaudio"
    assert qwen.get_output_sample_rate(None) == 24000
    assert fish.get_output_sample_rate(None) == 44100


def test_lazy_official_load_with_checkpoint_revision_reused_between_calls(tmp_path):
    env = make_runtime(tmp_path)
    assert env.loads == []
    for _ in range(2):
        result = env.runtime.generate_using_project(env.project, ["unchanged prompt"])
        assert len(result) == 1
        assert result[0].data.dtype == np.float32
        assert result[0].sr == 24000
    checkpoint = env.runtime.definition.checkpoints[0]
    assert env.loads == [((checkpoint.repo_id,), {"revision": checkpoint.revision, "strict": True})]
    assert env.seeds == [42, 42]
    assert env.fake.calls[0]["text"] == "unchanged prompt"
    assert env.policy.clears == 1
    assert env.fake.closed == 2
    env.runtime.kill()


def test_interactive_owner_guard_precedes_loader(tmp_path):
    env = make_runtime(tmp_path)
    model_runtime.mark_interactive_main()
    result = env.runtime.generate_using_project(env.project, ["hello"])
    assert "ModelWorker" in result
    assert env.loads == []


def test_stream_exhaustion_consumes_each_chunk_once_and_ends_once(tmp_path):
    env = make_runtime(tmp_path, chunks=[chunk([0.1]), chunk([0.2]), chunk([], final=True)])
    delivered, ended = [], []
    result = env.runtime.generate_using_project(env.project, ["hello"],
        on_stream_chunk=delivered.append, on_stream_end=lambda: ended.append(True))
    np.testing.assert_allclose(result[0].data, [0.1, 0.2])
    assert len(delivered) == 2
    assert ended == [True]
    assert env.fake.closed == 1
    assert env.policy.calls[0][3] is True
    env.runtime.kill()


def test_final_flag_is_optional_and_does_not_stop_iterator_early(tmp_path):
    env = make_runtime(tmp_path, chunks=[chunk([0.1], final=True), chunk([0.2])])
    result = env.runtime.generate_using_project(env.project, ["hello"])
    np.testing.assert_allclose(result[0].data, [0.1, 0.2])
    env.runtime.kill()


@pytest.mark.parametrize("bad_chunk, expected", [
    (chunk([], final=False), "empty non-final"),
    (chunk([np.nan]), "non-finite"),
    (chunk([[1, 2], [3, 4]]), "must be mono"),
    (chunk([1], rate=44100), "catalog rate"),
    (SimpleNamespace(sample_rate=24000), "without audio"),
])
def test_bad_audio_fails_before_callback_and_unloads(tmp_path, bad_chunk, expected):
    env = make_runtime(tmp_path, chunks=[bad_chunk])
    delivered, ended = [], []
    result = env.runtime.generate_using_project(env.project, ["hello"],
        on_stream_chunk=delivered.append, on_stream_end=lambda: ended.append(True))
    assert expected in result
    assert delivered == []
    assert ended == [True]
    assert env.runtime.model is None
    assert env.fake.closed == 1


def test_empty_stream_rejected_even_with_final_marker(tmp_path):
    env = make_runtime(tmp_path, chunks=[chunk([], final=True)])
    assert "no audio" in env.runtime.generate_using_project(env.project, ["hello"])


def test_cancellation_after_next_closes_iterator_and_never_delivers_chunk(tmp_path):
    def interrupt():
        Interrupts()._flag = True
        return chunk()
    env = make_runtime(tmp_path, chunks=[interrupt])
    delivered, ended = [], []
    result = env.runtime.generate_using_project(env.project, ["hello"],
        on_stream_chunk=delivered.append, on_stream_end=lambda: ended.append(True))
    assert "interrupted" in result
    assert delivered == []
    assert ended == [True]
    assert env.fake.closed == 1
    assert env.runtime.model is None


def test_backend_exception_recovers_on_next_request(tmp_path):
    env = make_runtime(tmp_path, chunks=[RuntimeError("backend failure")])
    assert "backend failure" in env.runtime.generate_using_project(env.project, ["hello"])
    assert env.runtime.model is None
    env.fake.chunks = [chunk()]
    assert len(env.runtime.generate_using_project(env.project, ["hello"])) == 1
    assert len(env.loads) == 2
    env.runtime.kill()


def test_scratch_reference_preserves_source_rate_and_is_removed_on_kill(tmp_path):
    env = make_runtime(tmp_path)
    assert isinstance(env.runtime.generate_using_project(env.project, ["hello"]), list)
    reference = env.policy.calls[-1][2][0]
    info = sf.info(reference.path)
    assert info.samplerate == 48000
    assert info.channels == 1
    assert info.subtype == "PCM_16"
    assert reference.transcript == "reference words"
    env.runtime.kill()
    assert not os.path.exists(reference.path)
    assert env.events[-2:] == ["sync", "clear"]


def test_reference_content_and_transcript_invalidate_native_cache(tmp_path):
    env = make_runtime(tmp_path)
    generate = lambda: env.runtime.generate_using_project(env.project, ["hello"])
    generate()
    first = env.policy.calls[-1][2][0]
    path = tmp_path / "voice" / "speaker.wav"
    original_stat = path.stat()
    sf.write(path, [-0.2, 0.2, -0.1], 48000, subtype="PCM_16")
    os.utime(path, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
    generate()
    second = env.policy.calls[-1][2][0]
    assert first.fingerprint != second.fingerprint
    env.project.voice_references[0] = replace(env.project.voice_references[0], transcript="different words")
    generate()
    third = env.policy.calls[-1][2][0]
    assert second.fingerprint != third.fingerprint
    assert env.policy.clears == 3
    env.runtime.kill()


def test_crop_and_matching_crop_transcript_used_together(tmp_path):
    env = make_runtime(tmp_path)
    crops = tmp_path / "voice" / "crops"
    crops.mkdir()
    sf.write(crops / "crop.flac", [0.1, -0.1], 32000)
    env.project.voice_references[0] = replace(
        env.project.voice_references[0], crop=VoiceCrop("crop.flac", 0, 1, "cropped words"))
    result = env.runtime.generate_using_project(env.project, ["hello"])
    assert isinstance(result, list)
    reference = env.policy.calls[-1][2][0]
    assert reference.transcript == "cropped words"
    assert sf.info(reference.path).samplerate == 32000
    env.runtime.kill()


def test_checkpoint_change_disposes_before_load(tmp_path):
    env = make_runtime(tmp_path)
    env.runtime.generate_using_project(env.project, ["hello"])
    old_path = env.policy.calls[-1][2][0].path
    original = env.runtime.definition
    new_checkpoint = replace(original.checkpoints[0], repo_id="test/other", revision="new-revision")
    env.runtime.definition = replace(original, checkpoints=(new_checkpoint,))
    env.settings["checkpoint"] = new_checkpoint.repo_id
    env.events.clear()
    assert isinstance(env.runtime.generate_using_project(env.project, ["hello"]), list)
    assert env.events == ["sync", "clear", "load"]
    assert not os.path.exists(old_path)
    assert env.loads[-1][1]["revision"] == "new-revision"
    env.runtime.kill()


def test_random_seed_cap_only_changes_random_draws(tmp_path):
    env = make_runtime(tmp_path)
    env.runtime.generate_using_project(env.project, ["hello"], max_random_seed=0)
    assert env.seeds == [42]
    env.runtime.generate_using_project(env.project, ["hello"], force_random_seed=True, max_random_seed=0)
    assert env.seeds[-1] == 0
    env.settings["seed"] = -1
    env.runtime.generate_using_project(env.project, ["hello"], max_random_seed=0)
    assert env.seeds[-1] == 0
    env.runtime.kill()


@pytest.mark.parametrize("saved_seed,force_random,expected,draw_count", [
    (42, False, 42, 0), (-1, False, 17, 1), (42, True, 17, 1),
])
def test_one_seed_initializes_entire_prompt_batch(
        tmp_path, monkeypatch, saved_seed, force_random, expected, draw_count):
    env = make_runtime(tmp_path)
    env.settings["seed"] = saved_seed
    draws = []

    def draw(low, high):
        draws.append((low, high))
        return 17

    monkeypatch.setattr("tts_audiobook_tool.tts_models.mlx_audio_runtime.random.randint", draw)
    result = env.runtime.generate_using_project(
        env.project, ["First prompt.", "Second prompt."], force_random_seed=force_random)
    assert isinstance(result, list) and len(result) == 2
    assert len(draws) == draw_count
    assert env.seeds == [expected]
    env.runtime.kill()


def test_callbacks_cannot_mutate_assembled_audio(tmp_path):
    env = make_runtime(tmp_path)
    def mutate(audio):
        audio[:] = 0
    result = env.runtime.generate_using_project(env.project, ["hello"], on_stream_chunk=mutate)
    np.testing.assert_allclose(result[0].data, [0.1, 0.2])
    env.runtime.kill()


def test_token_budget_cap_is_reported_even_for_streamed_delta_counts(tmp_path):
    env = make_runtime(tmp_path, chunks=[chunk(token_count=2, segment_idx=0),
                                        chunk(token_count=1, segment_idx=0)])
    env.settings["max_tokens"] = 3
    result = env.runtime.generate_using_project(env.project, ["hello"])
    assert "token budget" in result
    assert env.runtime.model is None


@pytest.mark.parametrize("deliver_callbacks", [False, True])
def test_fish_internal_streaming_preserves_one_request_and_callback_contract(tmp_path, deliver_callbacks):
    env = make_runtime(tmp_path, model_id="fish_drama3_mlxaudio", chunks=[
        chunk([0.1], rate=44100, token_count=2, segment_idx=0),
        chunk([0.2], rate=44100, token_count=2, segment_idx=1),
        chunk([], rate=44100, final=True, token_count=0, segment_idx=1),
    ])
    env.settings["max_tokens"] = 3
    env.fake.config = SimpleNamespace(model_type="fish_drama3",
                                     speech_template_version="reference-bracket-v1")
    resets = []
    env.fake.codec = SimpleNamespace(reset_streaming_state=lambda: resets.append(True))
    env.fake._voices = SimpleNamespace(_voices=OrderedDict())
    env.runtime.policy = FishDrama3Policy()
    delivered, ended = [], []
    result = env.runtime.generate_using_project(
        env.project, ["one logical segment"],
        on_stream_chunk=delivered.append if deliver_callbacks else None,
        on_stream_end=lambda: ended.append(True))
    np.testing.assert_allclose(result[0].data, [0.1, 0.2])
    assert len(env.fake.calls) == 1
    assert env.fake.calls[0]["stream"] is True
    assert len(delivered) == (2 if deliver_callbacks else 0)
    assert ended == ([True] if deliver_callbacks else [])
    assert env.fake.closed == 1 and resets == [True]
    env.runtime.kill()


def test_all_zero_output_rejected_and_runtime_state_recoverable(tmp_path):
    env = make_runtime(tmp_path, chunks=[chunk([0, 0]), chunk([0])])
    result = env.runtime.generate_using_project(env.project, ["hello"])
    assert "silent (all-zero)" in result
    assert env.runtime.model is None
    env.fake.chunks = [chunk([0, 1e-20, 0])]
    result = env.runtime.generate_using_project(env.project, ["quiet speech"])
    np.testing.assert_array_equal(result[0].data, np.array([0, 1e-20, 0], dtype=np.float32))
    env.runtime.kill()


def test_fish_all_references_is_explicit_and_selected_respects_index(tmp_path):
    env = make_runtime(tmp_path, model_id="fish_drama3_mlxaudio", chunks=[chunk(rate=44100)])
    sf.write(tmp_path / "voice" / "second.wav", [0.2, -0.2], 22050)
    env.project.voice_references.append(Voice("second.wav", "second voice"))
    env.policy.supports_multiple_references = True
    env.runtime.generate_using_project(env.project, ["hello"], voice_selection_index=1)
    assert [ref.transcript for ref in env.policy.calls[-1][2]] == ["second voice"]
    env.settings["reference_mode"] = "all"
    env.runtime.generate_using_project(env.project, ["hello"])
    assert [ref.transcript for ref in env.policy.calls[-1][2]] == ["reference words", "second voice"]
    env.runtime.kill()


@pytest.mark.parametrize("all_references", [False, True])
def test_missing_reference_crop_never_falls_back_to_original(tmp_path, all_references):
    env = make_runtime(tmp_path, model_id="fish_drama3_mlxaudio", chunks=[chunk(rate=44100)])
    env.policy.supports_multiple_references = True
    env.project.voice_references.append(Voice(
        "speaker.wav", "original", VoiceCrop("gone.flac", 0.0, 3.0, "trim")))
    env.settings["reference_mode"] = "all" if all_references else "selected"
    result = env.runtime.generate_using_project(env.project, ["hello"], voice_selection_index=1)
    assert "trim does not exist" in result
    assert env.loads == []
    assert env.fake.calls == []


def test_prepared_reference_cache_bounded_across_reference_changes(tmp_path):
    env = make_runtime(tmp_path)
    env.runtime.MAX_PREPARED_REFERENCES = 2
    old_paths = []
    for index in range(4):
        env.project.voice_references[0] = replace(env.project.voice_references[0], transcript=f"words {index}")
        env.runtime.generate_using_project(env.project, ["hello"])
        old_paths.append(env.policy.calls[-1][2][0].path)
    assert len(env.runtime._references) == 2
    assert [os.path.exists(path) for path in old_paths] == [False, False, True, True]
    env.runtime.kill()


def test_cleanup_hook_failure_still_drops_model_and_scratch(tmp_path):
    env = make_runtime(tmp_path)
    env.runtime.generate_using_project(env.project, ["hello"])
    path = env.policy.calls[-1][2][0].path
    def bad_reset(model):
        raise RuntimeError("cleanup failure")
    env.policy.reset_stream_state = bad_reset
    with pytest.raises(RuntimeError, match="cleanup failure"):
        env.runtime.kill()
    assert env.runtime.model is None
    assert not os.path.exists(path)


def test_model_rate_mismatch_fails_before_generation(tmp_path):
    env = make_runtime(tmp_path)
    env.fake.sample_rate = 44100
    assert "catalog rate" in env.runtime.generate_using_project(env.project, ["hello"])
    assert env.fake.calls == []
    assert env.runtime.model is None


def test_stream_completion_error_unloads_and_is_reported(tmp_path):
    env = make_runtime(tmp_path)
    def fail_end():
        raise RuntimeError("bad callback")
    result = env.runtime.generate_using_project(env.project, ["hello"],
        on_stream_chunk=lambda audio: None, on_stream_end=fail_end)
    assert "completion callback failed" in result
    assert env.runtime.model is None


@pytest.mark.parametrize("model_id,expected", [
    (MODEL_ID, 4 * 1024**3), ("fish_drama3_mlxaudio", 4 * 1024**3),
])
def test_family_allocator_cache_defaults_are_lazy(tmp_path, model_id, expected):
    env = make_runtime(tmp_path, model_id=model_id)
    assert env.runtime.cache_limit_bytes == expected
    assert env.cache_limits == []


@pytest.mark.parametrize("model_id", [MODEL_ID, "fish_drama3_mlxaudio"])
def test_cache_limit_applies_before_load_is_held_until_unload_and_restored(tmp_path, model_id):
    rate = TtsModelType.require_by_id(model_id).value.default_output_sample_rate
    env = make_runtime(tmp_path, model_id=model_id, chunks=[chunk(rate=rate)])
    original_loader = env.runtime._loader

    def load_under_budget(*args, **kwargs):
        assert env.allocator.cache_limit == 4 * 1024**3
        return original_loader(*args, **kwargs)

    env.runtime._loader = load_under_budget
    env.runtime.prepare_for_project(env.project)
    assert env.cache_limits == [4 * 1024**3]
    for _ in range(2):
        assert isinstance(env.runtime.generate_using_project(env.project, ["hello"]), list)
    env.runtime.release_inference_memory()
    assert env.cache_limits == [4 * 1024**3]
    with pytest.raises(ValueError, match="Unload"):
        env.runtime.cache_limit_bytes = None
    env.runtime.kill()
    assert env.allocator.cache_limit == 9 * 1024**3
    assert env.cache_limits == [4 * 1024**3, 9 * 1024**3]
    env.runtime.kill()
    assert len(env.cache_limits) == 2


@pytest.mark.parametrize("override", [None, 0, 1024])
def test_explicit_cache_override_and_constructor_none_preserve_baseline(tmp_path, override):
    env = make_runtime(tmp_path)
    runtime = type(env.runtime)(env.runtime.model_type, loader=env.runtime._loader,
                               mx=env.runtime._mx, policy=env.policy,
                               cache_limit_bytes=override)
    runtime.prepare_for_project(env.project)
    assert runtime.cache_limit_bytes == override
    assert env.cache_limits == ([] if override is None else [override])
    runtime.kill()
    assert env.allocator.cache_limit == 9 * 1024**3


@pytest.mark.parametrize("failure", ["load", "validation", "generation", "sync", "clear", "stream_state"])
@pytest.mark.parametrize("model_id", [MODEL_ID, "fish_drama3_mlxaudio"])
def test_allocator_policy_restored_on_failure_even_when_cleanup_fails(tmp_path, failure, model_id):
    env = make_runtime(tmp_path, model_id=model_id)

    def fail(*args, **kwargs):
        raise RuntimeError(f"{failure} failed")

    if failure == "load":
        env.runtime._loader = fail
    elif failure == "validation":
        env.policy.validate_model = fail
    elif failure == "generation":
        env.fake.chunks = [RuntimeError("generation failed")]
    else:
        env.runtime.prepare_for_project(env.project)
        if failure == "stream_state":
            env.runtime.clear_stream_state = fail
        else:
            setattr(env.runtime._mx, "synchronize" if failure == "sync" else "clear_cache", fail)
    if failure in ("sync", "clear", "stream_state"):
        with pytest.raises(RuntimeError, match=failure):
            env.runtime.kill()
    else:
        assert f"{failure} failed" in env.runtime.generate_using_project(env.project, ["hello"])
    assert env.runtime.model is None
    assert env.allocator.cache_limit == 9 * 1024**3
    assert env.cache_limits == [4 * 1024**3, 9 * 1024**3]


def test_checkpoint_reload_restores_then_reacquires_prior_cache_policy(tmp_path):
    env = make_runtime(tmp_path)
    env.runtime.prepare_for_project(env.project)
    original = env.runtime.definition
    checkpoint = replace(original.checkpoints[0], repo_id="test/other", revision="another")
    env.runtime.definition = replace(original, checkpoints=(checkpoint,))
    env.settings["checkpoint"] = checkpoint.repo_id
    env.runtime.prepare_for_project(env.project)
    assert env.cache_limits == [4 * 1024**3, 9 * 1024**3, 4 * 1024**3]
    env.runtime.kill()
    fish = make_runtime(tmp_path, model_id="fish_drama3_mlxaudio")
    fish.runtime._mx = env.runtime._mx
    fish.runtime.prepare_for_project(fish.project)
    assert env.allocator.cache_limit == 4 * 1024**3
    fish.runtime.kill()
    assert env.allocator.cache_limit == 9 * 1024**3
    assert env.cache_limits == [4 * 1024**3, 9 * 1024**3, 4 * 1024**3, 9 * 1024**3,
                                4 * 1024**3, 9 * 1024**3]


def test_cache_budget_requires_supported_mlx_api_before_loading(tmp_path):
    env = make_runtime(tmp_path)
    del env.runtime._mx.set_cache_limit
    result = env.runtime.generate_using_project(env.project, ["hello"])
    assert "does not support" in result
    assert env.loads == []


@pytest.mark.parametrize("value", [True, -1, 1.5, "1024"])
def test_invalid_allocator_budget_rejected_without_loading(tmp_path, value):
    env = make_runtime(tmp_path)
    with pytest.raises(ValueError, match="cache_limit_bytes"):
        env.runtime.cache_limit_bytes = value
    assert env.loads == [] and env.cache_limits == []


def test_invalid_saved_checkpoint_stays_renderable_and_blocks_lightweight(tmp_path):
    env = make_runtime(tmp_path)
    env.settings["checkpoint"] = "missing/checkpoint"
    base = get_mlx_audio_base_class(MODEL_ID)
    assert "invalid saved checkpoint" in base.get_menu_text(env.project)
    assert base.get_blocking_issues(env.project)
    assert env.loads == []


@pytest.mark.parametrize("name,value", [("temperature", float("nan")), ("seed", -2),
                                         ("seed", True), ("max_tokens", 0)])
def test_lightweight_settings_validation_blocks_invalid_values(tmp_path, name, value):
    env = make_runtime(tmp_path)
    env.settings[name] = value
    assert get_mlx_audio_base_class(MODEL_ID).get_blocking_issues(env.project)
    assert env.loads == []


def test_fish_lightweight_readiness_blocks_unknown_reference_mode(tmp_path):
    env = make_runtime(tmp_path, model_id="fish_drama3_mlxaudio")
    env.settings["reference_mode"] = "rotate"
    assert get_mlx_audio_base_class("fish_drama3_mlxaudio").get_blocking_issues(env.project)
    assert env.loads == []


def test_prepare_for_project_loads_under_init_boundary_without_synthesis(tmp_path):
    env = make_runtime(tmp_path)
    assert env.runtime.is_loaded is False
    env.runtime.prepare_for_project(env.project)
    assert env.runtime.is_loaded is True
    assert len(env.loads) == 1
    assert env.fake.calls == []
    assert env.runtime._scratch is None
    env.runtime.prepare_for_project(env.project)
    assert len(env.loads) == 1
    env.runtime.kill()
    assert env.runtime.is_loaded is False


def test_cancel_after_load_avoids_reference_io_and_releases_model(tmp_path):
    env = make_runtime(tmp_path)
    loader = env.runtime._loader
    def cancelled_loader(*args, **kwargs):
        model = loader(*args, **kwargs)
        Interrupts()._flag = True
        return model
    env.runtime._loader = cancelled_loader
    assert "interrupted" in env.runtime.generate_using_project(env.project, ["hello"])
    assert env.runtime._scratch is None
    assert env.fake.calls == []
    assert env.runtime.model is None


@pytest.mark.parametrize("failure", ["missing", "transcript"])
def test_obvious_reference_failure_precedes_weight_load(tmp_path, failure):
    env = make_runtime(tmp_path)
    if failure == "missing":
        (tmp_path / "voice" / "speaker.wav").unlink()
    else:
        env.project.voice_references[0] = replace(env.project.voice_references[0], transcript=" ")
    assert isinstance(env.runtime.generate_using_project(env.project, ["hello"]), str)
    assert env.loads == []


@pytest.mark.parametrize("extra", [{"token_count": 2}, {"finish_reason": "length"}])
def test_fish_budget_failure_rejects_audio_and_requests_fresh_seed(tmp_path, extra):
    env = make_runtime(tmp_path, model_id="fish_drama3_mlxaudio",
                       chunks=[chunk([0.1], rate=44100, **extra)])
    env.settings["max_tokens"] = 2
    delivered = []
    result = env.runtime.generate_using_project(env.project, ["Short text."],
                                                on_stream_chunk=delivered.append)
    assert "possible native termination failure" in result
    assert "even with short text" in result
    assert "fresh seed" in result
    assert "increase" not in result
    assert delivered == []
    assert env.runtime.model is None and env.fake.closed == 1
