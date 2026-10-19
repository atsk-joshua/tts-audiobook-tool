"""Fish Drama 3 policy for the shared, local MLX-Audio runtime.

The verified API is MLX-Audio 4ef6c3a577ade12d946e4b1d1870e4ba67fb5b9a.
Drama 3 owns text validation, speaker tags, inline cues and internal chunking.
Its first/latest chunk history is local to one generate() call; the cached
reference prefix is voice conditioning, not continuation between book segments.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from tts_audiobook_tool.tts_models.mlx_audio_policy import PreparedReference


class FishDrama3Policy:
    reference_sample_rate = 44_100
    supports_multiple_references = True

    def build_kwargs(
        self,
        values: Mapping[str, Any],
        language_code: str,
        references: Sequence[PreparedReference],
        stream: bool,
    ) -> dict[str, Any]:
        """Map only implemented controls; reference selection belongs to runtime.

        The common language hint is deliberately not sent: Drama 3 infers the
        language from text, and its **kwargs would silently ignore lang_code.
        max_tokens is an acoustic frame budget for each native text chunk.
        """
        defaults = {"temperature": 0.7, "top_p": 0.7, "top_k": 30, "max_tokens": 1024}
        unknown = set(values) - defaults.keys()
        if unknown:
            raise ValueError(f"Unsupported Fish Drama 3 settings: {', '.join(sorted(unknown))}")
        result = defaults | dict(values)
        for name in ("temperature", "top_p"):
            value = result[name]
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or value <= 0):
                raise ValueError(f"Fish Drama 3 {name} must be a finite positive number")
        if result["top_p"] > 1:
            raise ValueError("Fish Drama 3 top_p must be at most 1")
        for name in ("top_k", "max_tokens"):
            value = result[name]
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"Fish Drama 3 {name} must be a positive integer")
        if not isinstance(stream, bool):
            raise ValueError("Fish Drama 3 stream must be a boolean")
        if not references:
            raise ValueError("Fish Drama 3 voice cloning requires a reference and exact transcript")
        for reference in references:
            if not isinstance(reference.path, str) or not reference.path.strip():
                raise ValueError("Fish Drama 3 requires a prepared reference audio path")
            if not isinstance(reference.transcript, str) or not reference.transcript.strip():
                raise ValueError("Every Fish Drama 3 reference requires an exact transcript")
        # Keep strings, including cue punctuation and transcript whitespace,
        # untouched. Native _voice pairs the two lists in their supplied order.
        result["ref_audio"] = (references[0].path if len(references) == 1
                               else [reference.path for reference in references])
        result["ref_text"] = (references[0].transcript if len(references) == 1
                              else [reference.transcript for reference in references])
        # Decode in small causal pieces even when the application only wants
        # assembled audio. The pinned non-streaming path decodes an entire
        # internal chunk at once, including a possible 1024-frame silent tail.
        # This keeps one generate() call and its first/latest acoustic history;
        # the runtime separately controls whether UI callbacks receive audio.
        result["stream"] = True
        return result

    def validate_model(self, model: Any) -> None:
        """Fail clearly if a checkpoint does not implement the verified API."""
        config = getattr(model, "config", None)
        if getattr(config, "model_type", None) != "fish_drama3":
            raise ValueError("Expected a Fish Drama 3 MLX-Audio checkpoint")
        if getattr(config, "speech_template_version", None) != "reference-bracket-v1":
            raise ValueError("Fish Drama 3 requires the reference-bracket-v1 speech template")
        if getattr(model, "sample_rate", None) != self.reference_sample_rate:
            raise ValueError("Fish Drama 3 checkpoint must produce native 44100 Hz audio")
        codec = getattr(model, "codec", None)
        if not callable(getattr(codec, "reset_streaming_state", None)):
            raise ValueError("Fish Drama 3 codec does not support streaming state reset")
        self._reference_cache(model)
        if not callable(getattr(model, "generate", None)):
            raise ValueError("Fish Drama 3 checkpoint has no generate method")

    @staticmethod
    def _reference_cache(model: Any) -> Any:
        # Pinned upstream has no public clear method. Its bounded cache retains
        # encoded references and backbone/tower prefix arrays in this mapping.
        cache = getattr(getattr(model, "_voices", None), "_voices", None)
        if not isinstance(cache, dict):
            raise ValueError("Fish Drama 3 reference cache API is incompatible with the pinned MLX-Audio version")
        return cache

    def clear_reference_cache(self, model: Any) -> None:
        self._reference_cache(model).clear()

    @staticmethod
    def reset_stream_state(model: Any) -> None:
        # Closing an interrupted generator does not execute an upstream finally
        # block; reset its causal codec explicitly before reuse or disposal.
        model.codec.reset_streaming_state()
