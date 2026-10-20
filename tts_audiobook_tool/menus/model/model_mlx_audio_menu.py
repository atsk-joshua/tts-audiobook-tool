"""Catalog-driven controls for native MLX-Audio; opening menus loads no weights."""
from tts_audiobook_tool import ask
from tts_audiobook_tool.menus.menu_util import MenuItem, MenuUtil
from tts_audiobook_tool.menus.model.model_menu_shared import ModelMenuShared
from tts_audiobook_tool.model_worker import ModelWorker
from tts_audiobook_tool.project_support.model_settings import SettingRef
from tts_audiobook_tool.state import State
from tts_audiobook_tool.tts_models.mlx_audio_definition import MlxAudioDefinition
from tts_audiobook_tool.util import make_menu_label


class ModelMlxAudioMenu:
    @staticmethod
    def make_items(state: State, definition: MlxAudioDefinition) -> list[MenuItem]:
        model_id = definition.spec.id

        def checkpoint_label(current: State) -> str:
            try:
                value = definition.get_checkpoint(current.project).label
            except ValueError:
                value = f"Invalid: {current.project.get_model_setting(model_id, 'checkpoint')}"
            return make_menu_label("Checkpoint", value)

        def choose_checkpoint(current: State, _: MenuItem) -> None:
            def apply(repo_id: str) -> None:
                if current.project.get_model_setting(model_id, "checkpoint") == repo_id:
                    return
                previous = current.project.get_model_setting(model_id, "checkpoint")
                current.project.set_model_setting(model_id, "checkpoint", repo_id)
                error = current.project.save()
                if error:
                    current.project.set_model_setting(model_id, "checkpoint", previous)
                    ask.ask_error(error)
                    return
                error = ModelWorker.clear_models_if_running_blocking()
                if error:
                    ask.ask_error(error)

            MenuUtil.options_menu(
                state=current, heading_text="MLX-Audio checkpoint",
                labels=[checkpoint.label for checkpoint in definition.checkpoints],
                values=[checkpoint.repo_id for checkpoint in definition.checkpoints],
                current_value=current.project.get_model_setting(model_id, "checkpoint"),
                default_value=definition.checkpoints[0].repo_id, on_select=apply,
                breadcrumb="Checkpoint",
            )

        # Keep a picker only when there is a real choice or a saved value to repair.
        checkpoint = state.project.get_model_setting(model_id, "checkpoint")
        show_picker = (len(definition.checkpoints) > 1
                       or checkpoint != definition.checkpoints[0].repo_id)
        items = [MenuItem(checkpoint_label, choose_checkpoint)] if show_picker else []
        for name, parameter in definition.parameters.items():
            label = {"top_p": "Top-P", "top_k": "Top-K", "max_tokens": "Maximum tokens"}.get(
                name, name.replace("_", " ").capitalize())
            if name == "max_tokens" and definition.family == "fish_drama3":
                label = "Acoustic frames per chunk"
            elif name == "repetition_penalty":
                label = "Requested repetition penalty"
            number_type = int if parameter["type"] == "int" else float
            items.append(MenuUtil.make_number_item(
                state=state, target=SettingRef(model_id, name), base_label=label,
                prompt=f"Enter {label.lower()}", default_value=number_type(parameter["default"]),
                min_value=number_type(parameter["min"]), max_value=number_type(parameter["max"]),
                is_minus_one_default=False,
                num_decimals=0 if parameter["type"] == "int" else 2,
            ))
            if name == "repetition_penalty":
                items[-1].sublabel = lambda current: (
                    "Effective penalty: "
                    f"{max(1.5, current.project.get_model_setting(model_id, 'repetition_penalty')):.2f} "
                    "(MLX-Audio Base reference-cloning minimum: 1.50)")
            elif name == "max_tokens" and definition.family == "fish_drama3":
                items[-1].sublabel = "Budget for each native acoustic chunk; not a text word limit."
        if not show_picker:
            items[0].superlabel = checkpoint_label
        items.append(ModelMenuShared.make_seed_item(state, SettingRef(model_id, "seed")))
        return items
