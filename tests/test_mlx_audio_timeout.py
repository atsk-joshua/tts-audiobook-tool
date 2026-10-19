import time

from tts_audiobook_tool import gen_timeout_util
from tts_audiobook_tool.generation_events import GenerationEvents, GenerationTimedOut
from tts_audiobook_tool.tts import Tts
from tts_audiobook_tool.tts_models.tts_model_type import TtsModelType


def test_mlx_first_prompt_remains_watched_after_explicit_model_warmup(monkeypatch):
    monkeypatch.setattr(Tts, 'is_remote_mode', staticmethod(lambda: False))
    monkeypatch.setattr(Tts, 'get_active_type', staticmethod(lambda: TtsModelType.require_by_id('qwen3tts_mlxaudio')))
    monkeypatch.setattr(gen_timeout_util, 'GEN_TIMEOUT', .01)
    events = []
    with GenerationEvents.using_sink(events.append):
        with gen_timeout_util.make_backend_gen_timeout_tracker().scope() as guard:
            time.sleep(.03)
    assert guard.did_time_out
    assert events == [GenerationTimedOut(timeout_seconds=.01)]
