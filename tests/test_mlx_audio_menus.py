"""CPU-only UI contracts; no model loading, synthesis or production project files."""
from dataclasses import replace
from types import SimpleNamespace

import pytest
import launch
from tts_audiobook_tool.menus.menu_util import MenuUtil, get_string_from
from tts_audiobook_tool.menus.model.model_mlx_audio_menu import ModelMlxAudioMenu
from tts_audiobook_tool.menus.model.model_menu_shared import ModelMenuShared
from tts_audiobook_tool.menus.model.model_select_menu import ModelSelectMenu
from tts_audiobook_tool.menus.voice.voice_mlx_audio_menu import VoiceMlxAudioMenu
from tts_audiobook_tool.menus.main_menu import on_model
from tts_audiobook_tool.menus.menu_status import _make_local_tts_text
from tts_audiobook_tool.prefs import Prefs
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.tts import Tts, TtsRuntimeMode
from tts_audiobook_tool.tts_models.mlx_audio_definition import get_mlx_audio_definition
from tts_audiobook_tool.tts_models.tts_model_type import TtsModelType


@pytest.fixture
def mlx_environment(monkeypatch):
    models = tuple(TtsModelType.require_by_id(i) for i in (
        'qwen3tts_mlxaudio', 'fish_drama3_mlxaudio'))
    monkeypatch.setattr(Tts, '_available_local_models', models)
    monkeypatch.setattr(Tts, '_backend_mode', TtsRuntimeMode.LOCAL)
    return models


def state(model_id='qwen3tts_mlxaudio'):
    return SimpleNamespace(project=Project(dir_path='/scratch/project', tts_model_type=model_id), prefs=Prefs())


def labels(current, items):
    return [get_string_from(current, item.label) for item in items]


@pytest.mark.parametrize('model_id, controls', [
    ('qwen3tts_mlxaudio', ['Temperature', 'Top-P', 'Top-K', 'Repetition penalty', 'Maximum tokens']),
    ('fish_drama3_mlxaudio', ['Temperature', 'Top-P', 'Top-K', 'Maximum tokens']),
])
def test_model_controls_follow_catalog(model_id, controls):
    current = state(model_id)
    text = labels(current, ModelMlxAudioMenu.make_items(current, get_mlx_audio_definition(model_id)))
    assert text[0].startswith('Checkpoint ')
    assert [line.split(' \x1b')[0] for line in text[1:-1]] == controls
    assert text[-1].startswith('Seed ')
    assert not any('URL' in line for line in text)


def test_checkpoint_edit_persists_and_unloads_without_loading(monkeypatch):
    current = state()
    definition = get_mlx_audio_definition(current.project.tts_model_type)
    alternate = replace(definition.checkpoints[0], repo_id='mlx-community/alternate', label='alternate')
    definition = replace(definition, checkpoints=definition.checkpoints+(alternate,))
    captured, actions = {}, []
    monkeypatch.setattr(MenuUtil, 'options_menu', lambda **kwargs: captured.update(kwargs))
    monkeypatch.setattr(Project, 'save', lambda self: actions.append('save'))
    monkeypatch.setattr('tts_audiobook_tool.menus.model.model_mlx_audio_menu.ModelWorker.clear_models_if_running_blocking', lambda: actions.append('unload'))
    item = ModelMlxAudioMenu.make_items(current, definition)[0]
    item.handler(current, item)
    captured['on_select'](alternate.repo_id)
    assert current.project.get_model_setting(definition.spec.id, 'checkpoint') == alternate.repo_id
    assert actions == ['save', 'unload']


def test_fish_reference_mode_is_explicit_and_selected_by_default(monkeypatch):
    current = state('fish_drama3_mlxaudio')
    captured = {}
    monkeypatch.setattr(MenuUtil, 'options_menu', lambda **kwargs: captured.update(kwargs))
    monkeypatch.setattr(Project, 'save', lambda self: '')
    items = VoiceMlxAudioMenu.make_items(current, get_mlx_audio_definition(current.project.tts_model_type))
    item = items[-1]
    assert 'selected sample' in get_string_from(current, item.label)
    item.handler(current, item)
    assert captured['default_value'] == 'selected'
    captured['on_select']('all')
    assert current.project.get_model_setting('fish_drama3_mlxaudio', 'reference_mode') == 'all'


def test_unselected_local_mlx_opens_model_picker(monkeypatch, mlx_environment):
    current, opened = state('none'), []
    monkeypatch.setattr(Tts, 'bind_project', lambda project: None)
    monkeypatch.setattr(ModelMenuShared, 'menu', lambda current: opened.append(True))
    on_model(current, None)
    assert opened == [True]


@pytest.mark.parametrize('model_id, expected', [('none', '1'), ('qwen3tts_mlxaudio', '')])
def test_mlx_model_menu_opens_picker_immediately_only_when_required(
        monkeypatch, mlx_environment, model_id, expected):
    current, captured = state(model_id), {}
    monkeypatch.setattr(MenuUtil, 'menu', lambda **kwargs: captured.update(kwargs))
    ModelMenuShared.menu(current)
    assert captured['initial_hotkey'] == expected
    picker = captured['items'](current)[0]
    assert get_string_from(current, picker.label).startswith('Switch TTS model')


def test_local_picker_avoids_remote_discovery(monkeypatch, mlx_environment):
    current, captured = state('none'), {}
    monkeypatch.setattr(Tts, 'get_available_tts_models', lambda **kwargs: list(mlx_environment))
    monkeypatch.setattr('tts_audiobook_tool.menus.model.model_select_menu.RemoteTtsDiscovery.get_snapshot', lambda: pytest.fail('Local picker read server discovery'))
    monkeypatch.setattr(MenuUtil, 'options_menu', lambda **kwargs: captured.update(kwargs))
    ModelSelectMenu.menu(current)
    assert {model.id for model in captured['values']} == {'none', *(model.id for model in mlx_environment)}


def test_shared_menu_redraw_tracks_family(monkeypatch, mlx_environment):
    current = state()
    monkeypatch.setattr(Tts, 'get_available_tts_models', lambda: list(mlx_environment))
    assert any(line.startswith('Repetition penalty') for line in labels(current, ModelMenuShared.make_remote_items(current)))
    current.project.tts_model_type = 'fish_drama3_mlxaudio'
    assert not any(line.startswith('Repetition penalty') for line in labels(current, ModelMenuShared.make_remote_items(current)))


def test_status_identifies_native_mlx_without_cpu_override(mlx_environment):
    current = state()
    current.prefs.tts_force_cpu = True
    text = _make_local_tts_text(current, None)
    assert 'Local (MLX-Audio)' in text
    assert 'force cpu' not in text


def test_launcher_groups_mlx_families_into_one_probe():
    assert [(probe, label) for probe, label in launch.QUALIFIED_MODELS if probe == 'mlx_audio'] == [('mlx_audio', 'Local (MLX-Audio)')]

@pytest.mark.parametrize('compatible', [True, False])
def test_startup_allows_only_compatible_multi_model_stacks(monkeypatch, mlx_environment, compatible):
    from tts_audiobook_tool.start import Start
    monkeypatch.setattr(Tts, 'init_local_model_type', lambda: (TtsModelType.require_by_id('none'), len(mlx_environment)))
    monkeypatch.setattr(Tts, 'has_compatible_local_models', lambda: compatible)
    current = object.__new__(Start)
    if compatible:
        current.init_tts_or_exit(False)
    else:
        with pytest.raises(SystemExit):
            current.init_tts_or_exit(False)
