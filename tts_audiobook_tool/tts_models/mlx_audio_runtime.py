"""One lazy, worker-owned inference runtime for the native MLX-Audio stack."""
from __future__ import annotations

from collections import OrderedDict
import gc
import hashlib
import os
import random
import tempfile
from typing import Any

import numpy as np

from tts_audiobook_tool.app_support.interrupts import Interrupts
from tts_audiobook_tool.app_types import Sound
from tts_audiobook_tool.model_runtime import require_model_owner
from tts_audiobook_tool.project_support.project_voice_util import ProjectVoiceUtil
from tts_audiobook_tool.seed_util import get_random_seed_max
from tts_audiobook_tool.tts_models.mlx_audio_base_model import (
    FishDrama3MlxAudioBaseModel, QwenMlxAudioBaseModel,
    validate_mlx_audio_settings,
)
from tts_audiobook_tool.tts_models.mlx_audio_definition import get_mlx_audio_definition
from tts_audiobook_tool.tts_models.mlx_audio_policy import PreparedReference, get_policy
from tts_audiobook_tool.tts_models.tts_model_type import TtsModelType


class _GenerationInterrupted(Exception):
    pass


_DEFAULT_CACHE_LIMIT = object()
MLX_AUDIO_CACHE_LIMIT_BYTES = 4 * 1024**3


def default_cache_limit_bytes(family: str) -> int | None:
    """Allocator cache policy, separate from live-memory admission budgets."""
    return (MLX_AUDIO_CACHE_LIMIT_BYTES
            if family in ("qwen3_tts", "fish_drama3") else None)


class MlxAudioRuntime:
    """Mixin shared by the concrete family-bound ``TtsBaseModel`` classes.

    Injecting ``loader``, ``mx``, and ``policy`` permits CPU-only contract tests.
    Construction owns no tensors. The first generation loads official upstream
    code and weights, after enforcing the application's model-worker boundary.
    """

    MAX_PREPARED_REFERENCES = 32

    def __init__(self, model_type, *, loader=None, mx=None, policy=None,
                 cache_limit_bytes=_DEFAULT_CACHE_LIMIT):
        self.model_type = (TtsModelType.require_by_id(model_type)
                           if isinstance(model_type, str) else model_type)
        self.definition = get_mlx_audio_definition(self.model_type.id)
        if self.definition is None or self.INFO.id != self.model_type.id:
            raise ValueError(f"Invalid MLX-Audio runtime family: {self.model_type.id}")
        self.policy = policy or get_policy(self.definition.family)
        self._loader = loader
        self._mx = mx
        self._model = None
        self._checkpoint = None
        self._previous_cache_limit: int | None = None
        self._cache_limit_owned = False
        self.cache_limit_bytes = (default_cache_limit_bytes(self.definition.family)
                                  if cache_limit_bytes is _DEFAULT_CACHE_LIMIT else cache_limit_bytes)
        self.output_sample_rate = None
        self._scratch = None
        self._references: OrderedDict[str, PreparedReference] = OrderedDict()
        self._native_reference_identity: tuple[str, ...] | None = None

    @property
    def device_value(self) -> str:
        return "MLX / Metal"

    @property
    def model(self):
        return self._model

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def cache_limit_bytes(self) -> int | None:
        """Process-wide MLX allocator cache budget while this TTS owns weights.

        None preserves the prior process policy. Set an experimental override
        before loading. A resident MLX STT model shares this allocator limit.
        This bounds retained cache, not live tensors or total process memory.
        """
        return self._cache_limit_bytes

    @cache_limit_bytes.setter
    def cache_limit_bytes(self, value: int | None) -> None:
        if value is not None and (type(value) is not int or value < 0):
            raise ValueError("MLX cache_limit_bytes must be a nonnegative integer or None")
        if ((self._model is not None or self._cache_limit_owned)
                and value != self._cache_limit_bytes):
            raise ValueError("Unload MLX-Audio before changing cache_limit_bytes")
        self._cache_limit_bytes = value

    def _apply_cache_limit(self) -> None:
        if self.cache_limit_bytes is None or self._cache_limit_owned:
            return
        setter = getattr(self._mx, "set_cache_limit", None)
        if not callable(setter):
            raise RuntimeError("MLX does not support the configured allocator cache limit")
        self._previous_cache_limit = setter(self.cache_limit_bytes)
        self._cache_limit_owned = True

    def _restore_cache_limit(self) -> None:
        if self._cache_limit_owned:
            self._mx.set_cache_limit(self._previous_cache_limit)
            self._cache_limit_owned = False
            self._previous_cache_limit = None

    def prepare_for_project(self, project) -> None:
        """Warm weights under the worker's initialization timeout, without synthesis."""
        _, checkpoint, _ = validate_mlx_audio_settings(self.definition, project)
        try:
            self._check_interrupt()
            self._load(checkpoint)
            self._check_interrupt()
        except Exception:
            try:
                self.kill()
            except Exception:
                pass
            raise

    def _import_backend(self) -> None:
        require_model_owner("MLX-Audio")
        if self._mx is None:
            import mlx.core as mx
            self._mx = mx
        if self._loader is None:
            from mlx_audio.tts.utils import load
            self._loader = load

    def _allocator_cleanup(self) -> None:
        gc.collect()
        if self._mx is not None:
            clear = getattr(self._mx, "clear_cache", None)
            if clear is not None:
                clear()

    def _synchronize(self) -> None:
        if self._mx is not None:
            sync = getattr(self._mx, "synchronize", None)
            if sync is not None:
                sync()


    def clear_voice_clone_cache(self) -> None:
        try:
            if self._model is not None:
                self.policy.clear_reference_cache(self._model)
        finally:
            self._native_reference_identity = None
            self._references.clear()
            if self._scratch is not None:
                self._scratch.cleanup()
                self._scratch = None

    def kill(self) -> None:
        """Release model and scratch ownership; a later request can reload."""
        try:
            try:
                self._synchronize()
                if self._model is not None:
                    self.policy.reset_stream_state(self._model)
            finally:
                self.clear_voice_clone_cache()
        finally:
            # Release even when upstream cleanup fails. In particular a failed
            # load or inference must not poison the next worker request.
            self._model = None
            self._checkpoint = None
            self.output_sample_rate = None
            try:
                try:
                    self.clear_stream_state()
                finally:
                    self._allocator_cleanup()
            finally:
                self._restore_cache_limit()

    def release_inference_memory(self) -> None:
        self._synchronize()
        self._allocator_cleanup()

    def _load(self, checkpoint) -> None:
        require_model_owner("MLX-Audio")
        if self._model is not None and checkpoint == self._checkpoint:
            return
        if self._model is not None:
            self.kill()
        self._import_backend()
        self._apply_cache_limit()
        # Assign first so failed validation still releases the newly loaded
        # model through the same error-recovery ownership path.
        self._model = self._loader(checkpoint.repo_id, revision=checkpoint.revision, strict=True)
        self.policy.validate_model(self._model)
        rate = getattr(self._model, "sample_rate", None)
        self._validate_rate(rate)
        self.output_sample_rate = int(rate)
        self._checkpoint = checkpoint

    def _validate_rate(self, rate) -> None:
        if (isinstance(rate, bool) or not isinstance(rate, (int, np.integer))
                or int(rate) != self.INFO.default_output_sample_rate):
            raise ValueError(f"MLX-Audio sample rate {rate!r} does not match "
                             f"{self.INFO.id} catalog rate {self.INFO.default_output_sample_rate}")

    @staticmethod
    def _check_interrupt() -> None:
        if Interrupts().did_interrupt:
            raise _GenerationInterrupted("Generation interrupted")

    def _prepare_reference(self, source_path: str, transcript: str) -> PreparedReference:
        # Hash bytes rather than mtime/size: replacing a source in-place with
        # identical filesystem metadata must invalidate native conditioning.
        digest = hashlib.sha256()
        with open(source_path, "rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
        digest.update(b"\0transcript\0")
        digest.update(transcript.encode("utf-8"))
        fingerprint = digest.hexdigest()
        cached = self._references.get(fingerprint)
        if cached is not None:
            self._references.move_to_end(fingerprint)
            return cached

        import soundfile as sf
        audio, rate = sf.read(source_path, dtype="float32", always_2d=True)
        if audio.size == 0 or not np.isfinite(audio).all():
            raise ValueError("Voice reference must contain finite, nonempty audio")
        mono = audio.mean(axis=1, dtype=np.float32)
        if self._scratch is None:
            self._scratch = tempfile.TemporaryDirectory(prefix="tts-mlx-audio-")
        path = os.path.join(self._scratch.name, fingerprint + ".wav")
        # Preserve source bandwidth. Official MLX-Audio reference loaders own
        # resampling to the family's reference_sample_rate, matching the bridge.
        sf.write(path, mono, rate, subtype="PCM_16", format="WAV")
        prepared = PreparedReference(path, transcript, fingerprint)
        self._references[fingerprint] = prepared
        return prepared

    def _reference_pairs(self, project, voice_selection_index: int):
        mode = (project.get_model_setting(self.INFO.id, "reference_mode")
                if self.definition.family == "fish_drama3" else "selected")
        if mode not in ("selected", "all"):
            raise ValueError(f"Invalid MLX-Audio reference mode: {mode!r}")
        if mode == "all":
            if not self.policy.supports_multiple_references:
                raise ValueError("This MLX-Audio family supports one selected reference")
            pairs = ProjectVoiceUtil.voice_reference_pairs(project, self.model_type)
        else:
            pairs = [ProjectVoiceUtil.effective_voice_reference(project, self.model_type,
                                                               voice_selection_index)]
        pairs = [(name, transcript) for name, transcript in pairs if name]
        if self.INFO.requires_voice and not pairs:
            raise ValueError(f"{self.INFO.ui.get('proper_name', self.INFO.id)} requires a voice reference")
        for name, transcript in pairs:
            if not isinstance(transcript, str) or not transcript.strip():
                raise ValueError("MLX-Audio voice references require their exact transcripts")
            path = ProjectVoiceUtil.resolve_voice_file_path(project, name)
            if not os.path.isfile(path):
                raise ValueError(f"MLX-Audio voice reference does not exist: {path}")
        return pairs

    def _prepare_references(self, project, voice_selection_index: int, pairs=None):
        if pairs is None:
            pairs = self._reference_pairs(project, voice_selection_index)
        refs = [self._prepare_reference(ProjectVoiceUtil.resolve_voice_file_path(project, name), transcript)
                for name, transcript in pairs if name]
        identity = tuple(ref.fingerprint for ref in refs)
        if identity != self._native_reference_identity:
            self.policy.clear_reference_cache(self._model)
            self._native_reference_identity = identity
        # Keep all currently selected files alive, while bounding the reusable
        # cache across changes in selected voices. An explicit multi-reference
        # request naturally retains its own references until it has finished.
        active = set(identity)
        for key in list(self._references):
            if len(self._references) <= max(self.MAX_PREPARED_REFERENCES, len(active)):
                break
            if key not in active:
                old = self._references.pop(key)
                os.remove(old.path)
        return refs

    def _generate_one(self, text: str, kwargs: dict, on_stream_chunk) -> Sound:
        self._check_interrupt()
        iterator = iter(self._model.generate(text=text, **kwargs))
        chunks = []
        token_counts: dict[int, int] = {}
        has_nonzero_audio = False
        try:
            while True:
                self._check_interrupt()
                try:
                    chunk = next(iterator)
                except StopIteration:
                    break
                self._check_interrupt()
                self._validate_rate(getattr(chunk, "sample_rate", None))
                token_count = getattr(chunk, "token_count", None)
                segment = int(getattr(chunk, "segment_idx", 0))
                budget = kwargs.get("max_tokens")
                cap_reached = False
                if isinstance(token_count, (int, np.integer)) and not isinstance(token_count, bool):
                    token_counts[segment] = token_counts.get(segment, 0) + int(token_count)
                    cap_reached = isinstance(budget, int) and token_counts[segment] >= budget
                if cap_reached or getattr(chunk, "finish_reason", None) in ("length", "max_tokens"):
                    if self.definition.family == "fish_drama3":
                        raise ValueError(
                            f"Fish Drama 3 native chunk {segment} exhausted its acoustic token budget "
                            f"({budget} frames). Output was rejected: possible native termination "
                            "failure, which can occur even with short text. Retry with a fresh seed.")
                    raise ValueError("MLX-Audio generation exhausted its token budget; shorten the segment or review max_tokens")
                raw = getattr(chunk, "audio", None)
                if raw is None:
                    raise ValueError("MLX-Audio returned a chunk without audio")
                audio = np.asarray(raw, dtype=np.float32)
                if audio.ndim == 2 and 1 in audio.shape:
                    audio = audio.reshape(-1)
                if audio.ndim != 1:
                    raise ValueError(f"MLX-Audio output must be mono; got shape {audio.shape}")
                if audio.size == 0:
                    if getattr(chunk, "is_final_chunk", False) is True:
                        continue
                    raise ValueError("MLX-Audio returned an empty non-final audio chunk")
                if not np.isfinite(audio).all():
                    raise ValueError("MLX-Audio returned non-finite audio")
                # Upstream and playback callbacks may reuse or mutate buffers.
                audio = np.array(audio, dtype=np.float32, copy=True)
                chunks.append(audio)
                has_nonzero_audio |= bool(np.any(audio != 0))
                self._check_interrupt()
                if on_stream_chunk is not None:
                    on_stream_chunk(audio.copy())
                self._check_interrupt()
            self._check_interrupt()
            if not chunks:
                raise ValueError("MLX-Audio returned no audio")
            # Reject digital silence without trimming quiet speech or pauses.
            if not has_nonzero_audio:
                raise ValueError("MLX-Audio returned silent (all-zero) audio")
            return Sound(np.concatenate(chunks), self.output_sample_rate)
        finally:
            try:
                close = getattr(iterator, "close", None)
                if close is not None:
                    close()
            finally:
                self.policy.reset_stream_state(self._model)

    def generate_using_project(self, project, prompts: list[str], force_random_seed: bool = False,
                               on_stream_chunk=None, on_stream_end=None,
                               voice_selection_index: int = 0, print_params: bool = False,
                               max_random_seed: int = -1) -> list[Sound] | str:
        stream = on_stream_chunk is not None and self.INFO.can_stream
        result: list[Sound] | str
        try:
            self._check_interrupt()
            values, checkpoint, saved_seed = validate_mlx_audio_settings(self.definition, project)
            seed = -1 if force_random_seed else saved_seed
            if seed == -1:
                seed = random.randint(0, get_random_seed_max(2**32 - 1, max_random_seed))
            pairs = self._reference_pairs(project, voice_selection_index)
            self._load(checkpoint)
            self._check_interrupt()
            references = self._prepare_references(project, voice_selection_index, pairs)
            kwargs = self.policy.build_kwargs(values, project.language_code, references, stream)
            self._mx.random.seed(seed)
            if print_params:
                self.print_params({**values, "seed": seed, "checkpoint": checkpoint.label})
            result = [self._generate_one(text, kwargs, on_stream_chunk if stream else None)
                      for text in prompts]
        except Exception as exc:
            result = str(exc) or type(exc).__name__
            try:
                self.kill()
            except Exception:
                # Preserve the inference/cancellation error; kill's finally
                # already relinquished model ownership even if hooks failed.
                pass
        finally:
            self.clear_stream_state()
            if stream and on_stream_end is not None:
                try:
                    on_stream_end()
                except Exception as exc:
                    result = f"MLX-Audio stream completion callback failed: {exc}"
                    try:
                        self.kill()
                    except Exception:
                        pass
        return result


class QwenMlxAudioModel(MlxAudioRuntime, QwenMlxAudioBaseModel):
    pass


class FishDrama3MlxAudioModel(MlxAudioRuntime, FishDrama3MlxAudioBaseModel):
    pass


def get_mlx_audio_runtime_class(model_id: str):
    classes = {
        "qwen3tts_mlxaudio": QwenMlxAudioModel,
        "fish_drama3_mlxaudio": FishDrama3MlxAudioModel,
    }
    try:
        return classes[model_id]
    except KeyError:
        raise ValueError(f"Unsupported MLX-Audio model: {model_id!r}") from None
