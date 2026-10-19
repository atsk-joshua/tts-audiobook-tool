import pytest

from tts_audiobook_tool.app_types import SegmentationStrategy
from tts_audiobook_tool.app_types.phrase import Phrase, PhraseGroup, Reason
from tts_audiobook_tool.text_ops.phrase_grouper import PhraseGrouper


def sentence(word: str, count: int) -> str:
    return " ".join([word] * count) + ". "


def balanced(text: str, cap: int, **kwargs) -> list[PhraseGroup]:
    return PhraseGrouper.text_to_groups(
        text, cap, SegmentationStrategy.BALANCED, **kwargs
    )


def test_balanced_combines_sentences_with_multiple_phrases():
    text = "One two, three four. Five six, seven eight. Nine ten, eleven twelve."
    groups = balanced(text, 20)

    assert [group.num_words for group in groups] == [12]
    assert groups[0].text == text
    assert len(groups[0].phrases) == 6


def test_balanced_prefers_extra_prompt_over_avoidable_mid_sentence_cut():
    text = sentence("one", 12) + " ".join(["two"] * 8) + ", " + sentence("three", 8) + sentence("four", 12)
    groups = balanced(text, 20)
    greedy = PhraseGrouper.text_to_groups(text, 20, SegmentationStrategy.MAX_LEN)

    assert [group.num_words for group in groups] == [12, 16, 12]
    assert all(group.last_reason == Reason.SENTENCE for group in groups)
    assert [group.num_words for group in greedy] == [20, 20]
    assert greedy[0].last_reason == Reason.PHRASE


def test_balanced_avoids_short_tail_without_adding_prompt():
    text = sentence("one", 20) + sentence("two", 20) + sentence("three", 5)
    groups = balanced(text, 40)

    assert [group.num_words for group in groups] == [20, 25]
    assert "".join(group.text for group in groups) == text


def test_balanced_prefers_quote_end_over_quote_interior_when_lengths_tie():
    text = '"' + sentence("One", 10) + sentence("Two", 10).rstrip() + '" ' + sentence("Three", 10)
    groups = balanced(text, 20)

    assert [group.num_words for group in groups] == [20, 10]
    assert groups[0].text.rstrip().endswith('"')


def test_balanced_target_is_optional_and_does_not_change_legacy_strategies():
    text = sentence("one", 20) * 4
    assert len(balanced(text, 80)) == 1
    assert len(balanced(text, 80, target_words=10)) == 4
    with_target = PhraseGrouper.text_to_groups(
        text, 80, SegmentationStrategy.SENTENCE_PLUS, target_words=10
    )
    without_target = PhraseGrouper.text_to_groups(text, 80, SegmentationStrategy.SENTENCE_PLUS)
    assert [group.to_json_dict() for group in with_target] == [group.to_json_dict() for group in without_target]


@pytest.mark.parametrize("reason", [Reason.PARAGRAPH, Reason.HEADING_END, Reason.SPACE_BREAK, Reason.SECTION_BREAK])
def test_balanced_never_crosses_structural_boundary_or_changes_phrase_metadata(reason):
    phrases = [Phrase("First.\n", reason), Phrase("Second.", Reason.SENTENCE)]
    groups = PhraseGrouper.balanced_groups(phrases, 160)

    assert [group.phrases for group in groups] == [[phrases[0]], [phrases[1]]]
    assert groups[0].phrases[0] is phrases[0]
    assert phrases[0].reason is reason


def test_balanced_retains_oversized_indivisible_phrase_alone():
    phrase = Phrase(sentence("word", 21), Reason.SENTENCE)
    groups = PhraseGrouper.balanced_groups(
        [phrase, Phrase("Done.", Reason.SENTENCE)], 20
    )
    assert [group.num_words for group in groups] == [21, 1]
    assert groups[0].phrases == [phrase]


def test_balanced_preserves_dialog_pipeline_and_attribution_reason():
    text = 'He said, "One. Two!" she answered.\nNext paragraph.'
    groups = balanced(text, 160, dialog_segmentation=True)

    assert "".join(group.text for group in groups) == text
    assert [group.voice_index for group in groups] == [-1, 1, -1, -1]
    assert groups[1].last_reason == Reason.PHRASE_QUOTE_END
    assert groups[2].last_reason == Reason.PARAGRAPH


def test_balanced_empty_input_and_strategy_lookup():
    assert balanced("", 20) == []
    assert SegmentationStrategy.from_id("balanced") is SegmentationStrategy.BALANCED


@pytest.mark.parametrize("cap,target", [(0, None), (-1, None), (20, 0), (20, 21)])
def test_balanced_rejects_invalid_limits(cap, target):
    with pytest.raises(ValueError):
        balanced("Some text.", cap, target_words=target)
