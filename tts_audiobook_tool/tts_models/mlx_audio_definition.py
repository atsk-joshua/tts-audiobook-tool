"""Lightweight catalog definitions for the in-process MLX-Audio backend."""
from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, TYPE_CHECKING

from tts_audiobook_tool.tts_models.model_catalog import (
    CATALOG_PATH, _load_catalog, parse_mlx_audio_declaration, parse_spec,
)
from tts_audiobook_tool.tts_models.model_spec import TtsBackendKind, TtsModelSpec

if TYPE_CHECKING:
    from tts_audiobook_tool.project import Project


@dataclass(frozen=True)
class MlxAudioCheckpoint:
    repo_id: str
    revision: str
    label: str


@dataclass(frozen=True)
class MlxAudioDefinition:
    spec: TtsModelSpec
    family: str
    checkpoints: tuple[MlxAudioCheckpoint, ...]
    parameters: dict[str, dict[str, Any]]

    def get_values(self, project: Project) -> dict[str, int | float]:
        """Resolve only the family controls declared by the catalog."""
        values = {}
        for name, parameter in self.parameters.items():
            value = project.get_model_setting(self.spec.id, name)
            typ = int if parameter['type'] == 'int' else (int, float)
            if (isinstance(value, bool) or not isinstance(value, typ) or not math.isfinite(value)
                    or not parameter['min'] <= value <= parameter['max']):
                raise ValueError(f"{self.spec.id}.{name} must be a {parameter['type']} between "
                                 f"{parameter['min']} and {parameter['max']}")
            values[name] = value
        return values

    def get_checkpoint(self, project: Project) -> MlxAudioCheckpoint:
        selected = project.get_model_setting(self.spec.id, 'checkpoint')
        for checkpoint in self.checkpoints:
            if selected == checkpoint.repo_id:
                return checkpoint
        allowed = ', '.join(checkpoint.repo_id for checkpoint in self.checkpoints)
        raise ValueError(f"{self.spec.id}.checkpoint must be one of: {allowed}; got {selected!r}")


def load_mlx_audio_definitions(path: Path = CATALOG_PATH) -> dict[str, MlxAudioDefinition]:
    """Read validated native definitions; never discover or import MLX."""
    document, _, _, _ = _load_catalog(path)
    definitions = {}
    for entry in document['models']:
        if entry.get('backend_kind') != TtsBackendKind.MLX_AUDIO.value:
            continue
        spec = parse_spec(entry)
        declaration = parse_mlx_audio_declaration(entry, f'model {spec.id}')
        definitions[spec.id] = MlxAudioDefinition(
            spec, declaration['family'],
            tuple(MlxAudioCheckpoint(**checkpoint) for checkpoint in declaration['checkpoints']),
            declaration['parameters'],
        )
    return definitions


def get_mlx_audio_definition(model_id: str) -> MlxAudioDefinition | None:
    return _DEFINITIONS.get(model_id)


_DEFINITIONS = load_mlx_audio_definitions()
