"""Catalog and local-selection tests do not import or load MLX."""
import sys

import pytest

from tts_audiobook_tool.project import Project
from tts_audiobook_tool.project_support.model_settings import REGISTRY
from tts_audiobook_tool.tts import Tts, TtsRuntimeMode
from tts_audiobook_tool.tts_models.mlx_audio_definition import get_mlx_audio_definition, load_mlx_audio_definitions
from tts_audiobook_tool.tts_models.model_catalog import parse_mlx_audio_declaration
from tts_audiobook_tool.tts_models.model_spec import TtsBackendKind
from tts_audiobook_tool.tts_models.tts_model_type import TtsModelType

IDS = ('fish_drama3_mlxaudio', 'qwen3tts_mlxaudio')


def test_mlx_catalog_definitions_are_pinned_native_and_metadata_only():
    before = set(sys.modules)
    definitions = load_mlx_audio_definitions()
    assert tuple(definitions) == IDS
    for definition in definitions.values():
        assert definition.spec.backend_kind is TtsBackendKind.MLX_AUDIO
        assert definition.spec.local_torch_devices == []
        assert not definition.spec.requires_ffmpeg_libs
        assert definition.spec.can_voice_clone
        assert not definition.spec.can_instructions
        assert definition.spec.requirements_file_name == 'requirements-mlx-audio.txt'
        assert REGISTRY.voice_binding(definition.spec.id)
        assert REGISTRY.transcript_binding(definition.spec.id)
        project = Project(tts_model_type=definition.spec.id)
        support = Tts.get_model_support(project)
        assert support.can_voice_clone(project)
        assert not support.can_instructions(project)
        assert not support.can_voice_presets(project)
        checkpoint = definition.get_checkpoint(project)
        assert len(checkpoint.revision) == 40
        assert checkpoint == definition.checkpoints[0]
        assert definition.get_values(project) == {name: value['default'] for name, value in definition.parameters.items()}
    assert not any(name == 'mlx' or name.startswith('mlx.') or name == 'mlx_audio' or name.startswith('mlx_audio.') for name in set(sys.modules) - before)
    assert get_mlx_audio_definition('fish_drama3_audiocpp') is None


@pytest.mark.parametrize('model_id', IDS)
def test_selection_and_settings_roundtrip_without_remapping(model_id):
    project = Project(tts_model_type=model_id)
    definition = get_mlx_audio_definition(model_id)
    name = next(iter(definition.parameters))
    value = definition.parameters[name]['max']
    project.set_model_setting(model_id, name, value)
    restored = Project.model_validate(project.model_dump())
    assert restored.tts_model_type == model_id
    assert definition.get_values(restored)[name] == value
    restored.set_model_setting(model_id, 'checkpoint', 'unconfigured/model')
    with pytest.raises(ValueError, match='checkpoint must be one of'):
        definition.get_checkpoint(restored)


@pytest.mark.parametrize('bad', [True, float('nan'), -1, '0.9'])
def test_runtime_parameter_validation_rejects_invalid_values(bad):
    definition = get_mlx_audio_definition('qwen3tts_mlxaudio')
    class Values:
        def get_model_setting(self, model_id, name):
            return bad if name == 'temperature' else definition.parameters[name]['default']
    with pytest.raises(ValueError, match='temperature'):
        definition.get_values(Values())


def test_mlx_family_capabilities_are_one_compatible_stack(monkeypatch):
    models = tuple(TtsModelType.require_by_id(id) for id in IDS)
    monkeypatch.setattr(Tts, '_available_local_models', models)
    monkeypatch.setattr(Tts, '_backend_mode', TtsRuntimeMode.LOCAL)
    assert Tts.has_compatible_local_models()
    assert Tts.get_available_tts_models() == list(models)
    assert Tts.get_requirements_file_name() == 'requirements-mlx-audio.txt'
    for saved in (*IDS, 'qwen3tts_local', 'future_model_mlxaudio'):
        project = Project(tts_model_type=saved)
        assert Tts.reconcile_project_model(project) is None
        assert project.tts_model_type == saved
    assert not Tts.can_batch(models[0])
    monkeypatch.setattr(Tts, '_available_local_models', (*models, TtsModelType.require_by_id('qwen3tts_local')))
    assert not Tts.has_compatible_local_models()
    with pytest.raises(RuntimeError, match='More than one'):
        Tts.get_available_tts_models()


def test_mlx_discovery_probes_shared_package_once(monkeypatch):
    from tts_audiobook_tool import tts as tts_module
    probes = []
    def find(module):
        probes.append(module)
        return object() if module == 'mlx_audio' else None
    monkeypatch.setattr(tts_module.util, 'find_spec', find)
    monkeypatch.setattr(tts_module.metadata, 'version', lambda name: (_ for _ in ()).throw(ModuleNotFoundError(name)))
    _, count = Tts.init_local_model_type()
    assert count == len(IDS)
    assert probes.count('mlx_audio') == 1
    assert tuple(model.id for model in Tts.get_available_tts_models()) == IDS


def test_invalid_checkpoint_revision_rejected_by_catalog():
    entry = {'mlx_audio': {'family': 'qwen3_tts', 'checkpoints': [{'repo_id': 'org/model', 'revision': 'main', 'label': 'test'}], 'parameters': {}}}
    with pytest.raises(ValueError, match='immutable'):
        parse_mlx_audio_declaration(entry, 'test')


def test_unimplemented_native_family_is_rejected():
    entry = {'mlx_audio': {'family': 'omnivoice', 'checkpoints': [], 'parameters': {}}}
    with pytest.raises(ValueError, match='unsupported MLX-Audio family'):
        parse_mlx_audio_declaration(entry, 'test')
