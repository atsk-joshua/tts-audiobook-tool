"""Prompt-size policy without loading model weights or changing existing books."""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from tts_audiobook_tool.constants import (
    MAX_WORDS_PER_SEGMENT_DEFAULT, MAX_WORDS_PER_SEGMENT_MAX,
    MAX_WORDS_PER_SEGMENT_MIN, MAX_WORDS_PER_SEGMENT_STORAGE_MAX,
)

if TYPE_CHECKING:
    from tts_audiobook_tool.project import Project


@dataclass(frozen=True)
class SegmentationCapabilities:
    min_words: int = MAX_WORDS_PER_SEGMENT_MIN
    default_words: int = MAX_WORDS_PER_SEGMENT_DEFAULT
    max_words: int = MAX_WORDS_PER_SEGMENT_MAX
    recommended_min_words: int = 40
    recommended_max_words: int = 80
    recommendation_label: str = ""


# These checkpoints expose an experimental configurable ceiling, independent of
# acoustic budgets and without a synthesis guarantee. Only Fish chunks text
# internally; Qwen Base ICL streams audio from one generation sequence.
_EXTENDED_PROMPT_CHECKPOINTS = {
    ("qwen3tts_mlxaudio", "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16",
     "a6eb4f68e4b056f1215157bb696209bc82a6db48"),
    ("fish_drama3_mlxaudio", "mlx-community/fish-audio-drama-3-4bit",
     "da72954212fc2a0dfbb4f0779308a82ee705c851"),
}


def resolve_segmentation_capabilities(project: Project) -> SegmentationCapabilities:
    from tts_audiobook_tool.tts_models.mlx_audio_definition import get_mlx_audio_definition

    model_id = project.get_tts_model_type().id
    definition = get_mlx_audio_definition(model_id)
    if definition is None:
        from tts_audiobook_tool.tts import Tts
        support = Tts.get_model_support(project)
        recommended_min, recommended_max, label = support.get_max_words_range_reco(project)
        return SegmentationCapabilities(
            recommended_min_words=recommended_min, recommended_max_words=recommended_max,
            recommendation_label=label)
    try:
        checkpoint = definition.get_checkpoint(project)
    except ValueError:
        return SegmentationCapabilities()
    if (model_id, checkpoint.repo_id, checkpoint.revision) in _EXTENDED_PROMPT_CHECKPOINTS:
        return SegmentationCapabilities(max_words=MAX_WORDS_PER_SEGMENT_STORAGE_MAX)
    return SegmentationCapabilities()


def prompt_word_limit_issue(capabilities: SegmentationCapabilities, word_count: int) -> str:
    if word_count <= 0:
        return "The synthesis prompt must contain vocalizable text."
    if word_count > capabilities.max_words:
        return (
            f"The synthesis prompt contains {word_count} words, exceeding the selected "
            f"model's configurable maximum of {capabilities.max_words}. "
            "Use a compatible model or explicitly import a separately segmented copy."
        )
    return ""
