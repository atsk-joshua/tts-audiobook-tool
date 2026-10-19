"""Narrative grouping preserves source, boundaries and phrase metadata."""
from copy import deepcopy

import pytest

from tts_audiobook_tool.app_types.phrase import Phrase, PhraseGroup, Reason
from tts_audiobook_tool.app_support import app_text
from tts_audiobook_tool.text_ops.narrative_grouper import (
    normalize_narrative_source,
    pack_narrative_paragraphs, plan_narrative_groups,
)
from tts_audiobook_tool.text_ops.phrase_segmenter import PhraseSegmenter


def paragraph(text, reason=Reason.PARAGRAPH, voice=-1):
    return PhraseGroup([Phrase(text, reason)], voice_index=voice)


def flatten(groups):
    return [phrase for group in groups for phrase in group.phrases]


def test_complete_narrative_paragraphs_pack_without_losing_breaks_or_reasons():
    source = "The wind crossed the valley.\n\nThe river carried the rain.\n\nThe distant trees stood still."
    groups = plan_narrative_groups(source, 20, protect_section_opening=False)
    assert len(groups) == 1
    assert groups[0].text == source
    assert [phrase.reason for phrase in groups[0].phrases] == [Reason.PARAGRAPH, Reason.PARAGRAPH, Reason.SENTENCE]
    assert groups[0].num_words == 15


def test_default_protects_opening_paragraph_before_packing():
    source = "First came the heading.\nSecond came the narrative.\nThird came the narrative."
    groups = plan_narrative_groups(source, 40)
    assert len(groups) == 2
    assert groups[0].text == "First came the heading.\n"
    assert groups[1].text == "Second came the narrative.\nThird came the narrative."


@pytest.mark.parametrize("text,reason", [
    ("Chapter Two\n", Reason.PARAGRAPH),
    ("II\n", Reason.PARAGRAPH),
    ("The marked heading.\n", Reason.HEADING_END),
    ("The scene ended.\n\n\n", Reason.SPACE_BREAK),
    ("The section ended.\n", Reason.SECTION_BREAK),
    ('"He waited," she said.\n', Reason.PARAGRAPH),
    ('“An unfinished quotation.\n', Reason.PARAGRAPH),
    ("— He waited quietly.\n", Reason.PARAGRAPH),
    ("The scene ended.\n◇\n", Reason.PARAGRAPH),
])
def test_barriers_are_isolated_on_both_sides(text, reason):
    inputs = [paragraph("The preceding narration ended.\n"), paragraph(text, reason),
              paragraph("The following narration began.\n")]
    groups = pack_narrative_paragraphs(inputs, 80, protect_section_opening=False)
    assert [group.text for group in groups] == [group.text for group in inputs]


def test_explicit_heading_span_protects_heading_before_any_merging():
    source = "Ordinary narrative begins.\nThis unusually long heading ends with punctuation.\nOrdinary narrative continues."
    start = source.index("This")
    end = source.index("Ordinary", start)
    groups = plan_narrative_groups(source, 40, protected_spans=[(start, end)], protect_section_opening=False)
    assert len(groups) == 3
    assert groups[1].text == source[start:end]


def test_normalization_and_phrase_metadata_match_common_segmenter():
    source = "Rain arrived, slowly. \t\r\n\r\nThe meadow shone.\rThe wind settled."
    normalized = normalize_narrative_source(source)
    expected = PhraseSegmenter.text_to_phrases(normalized, max_words=30, pysbd_lang="en")
    first = plan_narrative_groups(source, 30, protect_section_opening=False)
    second = plan_narrative_groups(source, 30, protect_section_opening=False)
    assert flatten(first) == expected
    assert "".join(group.text for group in first) == "".join(phrase.text for phrase in expected)
    assert [group.to_json_dict() for group in first] == [group.to_json_dict() for group in second]


def test_leading_linefeeds_do_not_shift_protected_spans():
    raw = "\r\n\n\nThe meadow shone.\nThe explicit heading ends here.\nThe wind settled."
    source = normalize_narrative_source(raw)
    assert source.startswith("The meadow")
    start = source.index("The explicit")
    end = source.index("The wind")
    groups = plan_narrative_groups(raw, 40, protected_spans=[(start, end)],
                                   protect_section_opening=False)
    assert [group.text for group in groups] == source.splitlines(keepends=True)
    expected = PhraseSegmenter.text_to_phrases(source, max_words=40, pysbd_lang="en")
    assert flatten(groups) == expected


def test_presegmented_inputs_are_not_mutated_and_keep_phrase_identity():
    inputs = [paragraph("The meadow shone.\n"), paragraph("The wind settled.\n")]
    before = deepcopy([group.to_json_dict() for group in inputs])
    groups = pack_narrative_paragraphs(inputs, 20, protect_section_opening=False)
    assert len(groups) == 1
    assert groups[0].phrases[0] is inputs[0].phrases[0]
    assert groups[0].phrases[1] is inputs[1].phrases[0]
    assert [group.to_json_dict() for group in inputs] == before


def test_different_voices_never_merge_and_same_voice_is_retained():
    inputs = [paragraph("The meadow shone.\n", voice=0), paragraph("The wind settled.\n", voice=1),
              paragraph("The rain stopped.\n", voice=1)]
    groups = pack_narrative_paragraphs(inputs, 20, protect_section_opening=False)
    assert [group.voice_index for group in groups] == [0, 1]
    assert groups[1].text == inputs[1].text + inputs[2].text


def test_long_paragraph_falls_back_to_balanced_without_joining_neighbors():
    long_text = " ".join("The distant valley was quiet." for _ in range(10)) + "\n"
    source = "The meadow shone.\n" + long_text + "The rain stopped.\n"
    groups = plan_narrative_groups(source, 20, protect_section_opening=False)
    assert groups[0].text == "The meadow shone.\n"
    assert groups[-1].text == "The rain stopped.\n"
    assert all(group.num_words <= 20 for group in groups)
    assert "".join(group.text for group in groups[1:-1]) == long_text


def test_dialog_refinement_retains_dialog_voice_without_joining_neighbors():
    source = 'The wind settled.\n"Stay here," she said.\nThe river rose.\n'
    groups = plan_narrative_groups(source, 40, dialog_segmentation=True, protect_section_opening=False)
    assert "".join(group.text for group in groups) == source
    assert any(group.voice_index == 1 for group in groups)
    assert groups[0].text == "The wind settled.\n"
    assert groups[-1].text == "The river rose.\n"
    assert all(not ('"' in group.text and "river" in group.text) for group in groups)


@pytest.mark.parametrize("dialog", [False, True])
@pytest.mark.parametrize("source", [
    "The end.\n\n◇",
    '"One."\n\n◇',
    "◇\n\nThe river rose.\nThe wind settled.",
    '◇\n\n"One."\nThe wind settled.',
])
def test_final_ornamental_cleanup_retains_text_without_unspeakable_groups(source, dialog):
    groups = plan_narrative_groups(source, 40, dialog_segmentation=dialog,
                                   protect_section_opening=False)
    assert "".join(group.text for group in groups) == source
    assert all(app_text.is_vocalizable(group.text) for group in groups)
    if source.endswith("◇"):
        assert groups[-1].last_reason == Reason.SPACE_BREAK


@pytest.mark.parametrize("dialog", [False, True])
def test_ornament_does_not_join_narration_across_scene(dialog):
    source = 'The wind settled.\n\n◇\n\nThe river rose.\n'
    groups = plan_narrative_groups(source, 40, dialog_segmentation=dialog,
                                   protect_section_opening=False)
    assert len(groups) == 2
    assert groups[0].last_reason == Reason.SPACE_BREAK
    assert "river" not in groups[0].text
    assert "wind" not in groups[1].text


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_invalid_word_cap_is_rejected(value):
    with pytest.raises(ValueError, match="positive integer"):
        plan_narrative_groups("A sentence.", value)


@pytest.mark.parametrize("span", [(-1, 2), (0, 0), (2, 1), (0, 100), (True, 2)])
def test_invalid_protected_span_is_rejected(span):
    with pytest.raises(ValueError, match="half-open"):
        plan_narrative_groups("A sentence.", 20, protected_spans=[span])


def test_presegmented_oversized_atomic_phrase_is_explicitly_rejected():
    with pytest.raises(ValueError, match="oversized phrases"):
        pack_narrative_paragraphs([paragraph("one two three four five.\n")], 4)
