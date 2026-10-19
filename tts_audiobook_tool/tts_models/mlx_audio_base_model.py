"""Family-bound support classes usable without importing MLX or MLX-Audio."""
from __future__ import annotations

from tts_audiobook_tool.app_types import ReadinessIssue
from tts_audiobook_tool.tts_models.tts_base_model import TtsBaseModel
from tts_audiobook_tool.tts_models.tts_model_type import TtsModelType


def validate_mlx_audio_settings(definition, project):
    """Validate persisted controls without importing inference dependencies."""
    values = definition.get_values(project)
    checkpoint = definition.get_checkpoint(project)
    seed = project.get_model_setting(definition.spec.id, "seed")
    if type(seed) is not int or not -1 <= seed <= 2**32 - 1:
        raise ValueError("MLX-Audio seed must be -1 or an integer between 0 and 4294967295")
    if definition.family == "fish_drama3":
        mode = project.get_model_setting(definition.spec.id, "reference_mode")
        if mode not in ("selected", "all"):
            raise ValueError("Fish Drama 3 reference_mode must be selected or all")
    return values, checkpoint, seed


class _MlxAudioSupport:
    @classmethod
    def get_output_sample_rate(cls, project, instance=None) -> int:
        return getattr(instance, "output_sample_rate", None) or cls.INFO.default_output_sample_rate

    @classmethod
    def get_menu_text(cls, project, instance=None) -> str:
        from tts_audiobook_tool.tts_models.mlx_audio_definition import get_mlx_audio_definition
        definition = get_mlx_audio_definition(cls.INFO.id)
        try:
            label = definition.get_checkpoint(project).label
        except ValueError:
            label = "invalid saved checkpoint"
        return f"{cls.INFO.ui.get('proper_name', cls.INFO.id)} ({label})"

    @classmethod
    def get_blocking_issues(cls, project, instance=None):
        from tts_audiobook_tool.tts_models.mlx_audio_definition import get_mlx_audio_definition
        try:
            validate_mlx_audio_settings(get_mlx_audio_definition(cls.INFO.id), project)
        except ValueError as exc:
            return [ReadinessIssue("MLX-Audio settings", str(exc))]
        return []


class QwenMlxAudioBaseModel(_MlxAudioSupport, TtsBaseModel):
    INFO = TtsModelType.require_by_id("qwen3tts_mlxaudio").value


class FishDrama3MlxAudioBaseModel(_MlxAudioSupport, TtsBaseModel):
    INFO = TtsModelType.require_by_id("fish_drama3_mlxaudio").value


def get_mlx_audio_base_class(model_id: str) -> type[TtsBaseModel]:
    classes = {
        "qwen3tts_mlxaudio": QwenMlxAudioBaseModel,
        "fish_drama3_mlxaudio": FishDrama3MlxAudioBaseModel,
    }
    try:
        return classes[model_id]
    except KeyError:
        raise ValueError(f"Unsupported MLX-Audio model: {model_id!r}") from None
