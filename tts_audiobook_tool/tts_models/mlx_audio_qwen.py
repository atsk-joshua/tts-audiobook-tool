"""Qwen3-TTS Base policy for the shared in-process MLX-Audio runtime.

The public MLX-Audio Base API applies a repetition-penalty floor of 1.5 in
ICL mode. We preserve the requested value in the call for bridge parity and
report that effective floor, rather than bypassing the public API.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping, Sequence
from numbers import Integral, Real
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .mlx_audio_policy import PreparedReference


logger = logging.getLogger(__name__)

_LANGUAGES = {
    "zh": "chinese",
    "en": "english",
    "fr": "french",
    "de": "german",
    "it": "italian",
    "ja": "japanese",
    "ko": "korean",
    "pt": "portuguese",
    "ru": "russian",
    "es": "spanish",
}


class QwenPolicy:
    reference_sample_rate = 24000
    supports_multiple_references = False

    # Explicit defaults preserve the production bridge recipe across package
    # revisions whose default token budgets and checkpoint metadata differ.
    DEFAULTS = {
        "temperature": 0.9,
        "top_p": 1.0,
        "top_k": 50,
        "repetition_penalty": 1.05,
        "max_tokens": 1600,
    }

    def build_kwargs(
        self,
        values: Mapping[str, Any],
        language_code: str,
        references: Sequence[PreparedReference],
        stream: bool,
    ) -> dict[str, Any]:
        unknown = set(values) - self.DEFAULTS.keys()
        if unknown:
            raise ValueError(
                "Unsupported Qwen3-TTS Base controls: "
                + ", ".join(sorted(str(key) for key in unknown))
            )
        if len(references) != 1:
            raise ValueError("Qwen3-TTS Base requires exactly one voice reference")
        reference = references[0]
        if not isinstance(reference.path, str) or not reference.path.strip():
            raise ValueError("Qwen3-TTS Base requires a reference audio path")
        if not isinstance(reference.transcript, str) or not reference.transcript.strip():
            raise ValueError("Qwen3-TTS Base requires the exact reference transcript")
        if not isinstance(stream, bool):
            raise ValueError("Qwen3-TTS stream must be a boolean")

        options = {**self.DEFAULTS, **values}
        for name, value in options.items():
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
                raise ValueError(f"Qwen3-TTS {name} must be a finite number")
        for name in ("top_k", "max_tokens"):
            if not isinstance(options[name], Integral):
                raise ValueError(f"Qwen3-TTS {name} must be an integer")
            options[name] = int(options[name])
        if options["temperature"] < 0:
            raise ValueError("Qwen3-TTS temperature must be nonnegative")
        if not 0 < options["top_p"] <= 1:
            raise ValueError("Qwen3-TTS top_p must be greater than 0 and at most 1")
        if options["top_k"] < 0:
            raise ValueError("Qwen3-TTS top_k must be nonnegative (0 disables top-k)")
        if options["max_tokens"] < 1:
            raise ValueError("Qwen3-TTS max_tokens must be positive")
        if options["repetition_penalty"] <= 0:
            raise ValueError("Qwen3-TTS repetition_penalty must be positive")

        if not isinstance(language_code, str):
            raise ValueError("Qwen3-TTS language must be a language code or name")
        language = language_code.strip().lower().replace("_", "-")
        if language in ("", "auto"):
            language = "auto"
        elif language in _LANGUAGES.values():
            pass
        else:
            # Project language codes can include a region, e.g. en-US or zh-Hant.
            language = _LANGUAGES.get(language.split("-", 1)[0], "")
            if not language:
                raise ValueError(f"Unsupported Qwen3-TTS language: {language_code!r}")

        requested_penalty = options["repetition_penalty"]
        logger.info(
            "Qwen3-TTS Base reference cloning: requested repetition_penalty=%s; "
            "effective repetition_penalty=%s (MLX-Audio public ICL floor)",
            requested_penalty,
            max(requested_penalty, 1.5),
        )
        return {
            **options,
            "lang_code": language,
            "ref_audio": reference.path,
            "ref_text": reference.transcript,
            "stream": stream,
        }

    def validate_model(self, model: Any) -> None:
        family = getattr(model, "model_type", None)
        if family is not None and family != "qwen3_tts":
            raise ValueError(f"Expected a Qwen3-TTS model, got {family!r}")
        config = getattr(model, "config", None)
        variant = (
            config.get("tts_model_type")
            if isinstance(config, Mapping)
            else getattr(config, "tts_model_type", None)
        )
        if variant is not None and variant != "base":
            raise ValueError(
                f"Local (MLX-Audio) Qwen voice cloning requires a Base checkpoint; got {variant!r}"
            )

    def clear_reference_cache(self, model: Any) -> None:
        # MLX-Audio 0.5.9 exposes no public conditioning-cache release method.
        # Its Base cache is unbounded and uses a weak (length, sum) audio key.
        # Runtime calls this on a strong reference-fingerprint change/unload.
        cache = getattr(model, "_icl_cache", None)
        if isinstance(cache, dict):
            cache.clear()

    def reset_stream_state(self, model: Any) -> None:
        tokenizer = getattr(model, "speech_tokenizer", None)
        decoder = getattr(tokenizer, "decoder", None)
        reset = getattr(decoder, "reset_streaming_state", None)
        if callable(reset):
            reset()
