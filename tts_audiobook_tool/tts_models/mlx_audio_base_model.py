"""Family-bound support classes usable without importing MLX or MLX-Audio."""
from __future__ import annotations

import re

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

    @classmethod
    def prepare_narrative_text_for_inference(
            cls, project, text: str, *, apply_word_substitutions: bool = True,
    ) -> str:
        """Normalize narration while retaining newline delimiters and Fish tags.

        This Fish policy helper does not change the general normalizer.
        Tags remain control syntax rather than numbers to verbalize. Validate
        both sides of normalization so substitutions/Unicode folding cannot
        introduce conversation structure or malformed speaker controls.
        """
        speaker = re.compile(r"<\|speaker:([0-9]+)\|>")

        def validate_controls(value):
            for token in ("<|im_start|>", "<|im_end|>", "<|voice|>", "<|text|>",
                          "<|interleave|>", "<|semantic:"):
                if token in value:
                    raise ValueError(f"Fish narrative text must not contain {token}")
            for match in re.finditer(r"<\|speaker", value):
                tag = speaker.match(value, match.start())
                number = tag.group(1) if tag else ""
                if not number or number != str(int(number)) or int(number) > 15:
                    raise ValueError("Fish narrative speaker tags must run from <|speaker:0|> to <|speaker:15|>")

        validate_controls(text)
        def normalize_prose(value):
            prose = super(FishDrama3MlxAudioBaseModel, cls).prepare_text_for_inference(
                project, value, apply_word_substitutions=apply_word_substitutions)
            if "<|speaker" in prose:
                raise ValueError("Fish narrative normalization must not introduce speaker tags")
            return prose

        prepared = []
        # Retain every physical newline delimiter, including blank paragraphs;
        # normalize only the prose between them using the existing contract.
        for paragraph in re.split(r"(\r\n|[\r\n])", text):
            if paragraph in ("\r\n", "\r", "\n"):
                prepared.append(paragraph)
                continue
            parts, start = [], 0
            for tag in speaker.finditer(paragraph):
                prose = normalize_prose(paragraph[start:tag.start()])
                if prose:
                    parts.append(prose)
                parts.append(tag.group(0))
                start = tag.end()
            prose = normalize_prose(paragraph[start:])
            if prose:
                parts.append(prose)
            prepared.append(" ".join(parts))
        result = "".join(prepared)
        validate_controls(result)
        return result

    @classmethod
    def prepare_text_for_inference(cls, project, text: str, *, apply_word_substitutions: bool = True) -> str:
        # Only the strategy that produced this book's current groups selects
        # paragraph-preserving preparation. Future import settings cannot
        # change generation of an existing book.
        settings = getattr(getattr(project, "book", None), "segmentation_settings", None)
        if getattr(getattr(settings, "strategy", None), "id", None) == "narrative":
            return cls.prepare_narrative_text_for_inference(
                project, text, apply_word_substitutions=apply_word_substitutions)
        return super().prepare_text_for_inference(
            project, text, apply_word_substitutions=apply_word_substitutions)


def get_mlx_audio_base_class(model_id: str) -> type[TtsBaseModel]:
    classes = {
        "qwen3tts_mlxaudio": QwenMlxAudioBaseModel,
        "fish_drama3_mlxaudio": FishDrama3MlxAudioBaseModel,
    }
    try:
        return classes[model_id]
    except KeyError:
        raise ValueError(f"Unsupported MLX-Audio model: {model_id!r}") from None
