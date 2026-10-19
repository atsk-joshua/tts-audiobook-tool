"""Reuse project-managed reference audio, transcripts, crops and selection."""
from tts_audiobook_tool.menus.menu_util import MenuItem, MenuUtil
from tts_audiobook_tool.menus.voice.voice_menu_shared import VoiceMenuShared
from tts_audiobook_tool.state import State
from tts_audiobook_tool.tts_models.mlx_audio_definition import MlxAudioDefinition
from tts_audiobook_tool.tts_models.tts_model_type import TtsModelType
from tts_audiobook_tool.util import make_menu_label


class VoiceMlxAudioMenu:
    @staticmethod
    def make_items(state: State, definition: MlxAudioDefinition) -> list[MenuItem]:
        items = VoiceMenuShared.make_voice_sample_items(state, TtsModelType.require_by_id(definition.spec.id))
        if definition.family == "fish_drama3":
            def label(current: State) -> str:
                mode = current.project.get_model_setting(definition.spec.id, "reference_mode")
                return make_menu_label("Reference conditioning", "all project samples" if mode == "all" else "selected sample")

            def edit(current: State, _: MenuItem) -> None:
                def apply(mode: str) -> None:
                    current.project.set_model_setting(definition.spec.id, "reference_mode", mode)
                    current.project.save()
                MenuUtil.options_menu(
                    state=current, heading_text="Reference conditioning",
                    labels=["Selected sample (single narrator)", "All project samples"],
                    values=["selected", "all"],
                    current_value=current.project.get_model_setting(definition.spec.id, "reference_mode"),
                    default_value="selected", on_select=apply, breadcrumb="References",
                )
            items.append(MenuItem(label, edit, blank_line_before=True))
        return items

    @staticmethod
    def menu(state: State) -> None:
        VoiceMenuShared.menu_wrapper(state, VoiceMenuShared.make_remote_items)
