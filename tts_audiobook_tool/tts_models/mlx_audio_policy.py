"""Small, dependency-free contract for the native MLX-Audio families."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, Sequence


@dataclass(frozen=True)
class PreparedReference:
    path: str
    transcript: str
    fingerprint: str


class MlxAudioPolicy(Protocol):
    reference_sample_rate: int
    supports_multiple_references: bool

    def build_kwargs(self, values: dict, language_code: str,
                     references: Sequence[PreparedReference], stream: bool) -> dict: ...
    def validate_model(self, model: Any) -> None: ...
    def clear_reference_cache(self, model: Any) -> None: ...
    def reset_stream_state(self, model: Any) -> None: ...


def get_policy(family: str) -> MlxAudioPolicy:
    """Explicit dispatch; importing metadata never imports an inference library."""
    if family == "qwen3_tts":
        from tts_audiobook_tool.tts_models.mlx_audio_qwen import QwenPolicy
        return QwenPolicy()
    if family == "fish_drama3":
        from tts_audiobook_tool.tts_models.mlx_audio_fish import FishDrama3Policy
        return FishDrama3Policy()
    raise ValueError(f"Unsupported MLX-Audio family: {family!r}")
