"""Independent manuscript/metadata invariants for extended grouping limits.

These checks judge source coverage and downstream metadata, not a particular
choice of dynamic-programming costs or expected number of prompts.
"""

import re

import pytest

from tts_audiobook_tool.app_types import SegmentationStrategy
from tts_audiobook_tool.app_types.phrase import Phrase, Reason
from tts_audiobook_tool.text_ops.phrase_grouper import PhraseGrouper


STRATEGIES = (
    SegmentationStrategy.SENTENCE_PLUS,
    SegmentationStrategy.MULTI_SENTENCE,
    SegmentationStrategy.MAX_LEN,
    SegmentationStrategy.BALANCED,
)

PASSAGES = (
    "Dr. Vale arrived at 5 p.m. Nobody had moved. Go. Wait. They listened.",
    'He said, "One. Two. Three!" Then she answered, "Stay here," and waited.',
    "“Could’ve stayed…” he whispered. Her hands trembled—then grew still.",
    "A narrow corridor, lit by a distant lamp, curved through the old house; "
    "beyond it, beneath the stairs, somebody waited quietly for the rain to stop.",
    "First paragraph ends here.  \r\nSecond paragraph continues.\rThird one ends.",
    "Scene one ends.\n\nScene two begins.\n\nFinal scene follows.",
    "The repeated phrase stayed. The repeated phrase stayed. "
    "The repeated phrase stayed. Last word.",
    "\n\nChapter I\nThe beginning followed the heading.\n\nThe next scene opened.",
)


@pytest.mark.parametrize("strategy", STRATEGIES)
@pytest.mark.parametrize("ceiling", (40, 80, 120, 160, 240, 320))
@pytest.mark.parametrize("text", PASSAGES)
@pytest.mark.parametrize("dialog", (False, True))
def test_extended_strategies_cover_normalized_source_once(strategy, ceiling, text, dialog):
    normalized = re.sub(r"[ \t]+\n", "\n", text.replace("\r\n", "\n").replace("\r", "\n"))
    # Legacy text_to_phrases drops leading zero-word break phrases. The pure
    # sentence-slicing contract preserves them, but high-level import does not.
    normalized = normalized.lstrip("\n")
    groups = PhraseGrouper.text_to_groups(
        text, ceiling, strategy, "en", dialog_segmentation=dialog
    )
    assert "".join(group.text for group in groups) == normalized
    assert groups
    assert all(group.text.strip() and group.num_words > 0 for group in groups)
    assert all(group.num_words <= ceiling for group in groups)
    assert all(isinstance(phrase.reason, Reason) for group in groups for phrase in group.phrases)
    if not dialog:
        assert all(group.voice_index == -1 for group in groups)


@pytest.mark.parametrize("ceiling", (40, 80, 120, 160, 240, 320))
def test_long_clause_is_split_without_lost_or_repeated_tokens(ceiling):
    # Distinct tokens expose skipped/duplicated ranges even when prose repeats.
    text = " ".join(f"word{index:04d}" for index in range(ceiling * 3 + 17)) + "."
    groups = PhraseGrouper.text_to_groups(text, ceiling, SegmentationStrategy.BALANCED)
    assert "".join(group.text for group in groups) == text
    assert len(groups) >= 4
    assert all(0 < group.num_words <= ceiling for group in groups)


@pytest.mark.parametrize("boundary", (Reason.PARAGRAPH, Reason.HEADING_END,
                                      Reason.SPACE_BREAK, Reason.SECTION_BREAK))
def test_balanced_packing_preserves_phrase_identity_and_structural_barrier(boundary):
    phrases = [
        Phrase("Before the barrier. ", Reason.SENTENCE),
        Phrase("Last words before it.\n", boundary),
        Phrase("First words after it. ", Reason.SENTENCE),
        Phrase("End of next passage.", Reason.SECTION_BREAK),
    ]
    original = [(phrase.text, phrase.reason) for phrase in phrases]
    groups = PhraseGrouper.balanced_groups(phrases, 320, target_words=240)
    flattened = [phrase for group in groups for phrase in group.phrases]
    assert len(flattened) == len(phrases)
    assert all(actual is expected for actual, expected in zip(flattened, phrases))
    assert [(phrase.text, phrase.reason) for phrase in phrases] == original
    assert groups[0].phrases[-1] is phrases[1]
    assert groups[1].phrases[0] is phrases[2]


@pytest.mark.parametrize("ceiling", (40, 80, 160, 320))
def test_balanced_dialog_voice_assignment_survives_larger_groups(ceiling):
    text = 'He said, "Stay here." She answered, "I will wait." They listened.'
    groups = PhraseGrouper.text_to_groups(
        text, ceiling, SegmentationStrategy.BALANCED, dialog_segmentation=True
    )
    assert "".join(group.text for group in groups) == text
    spoken = "".join(group.text for group in groups if group.voice_index == 1)
    narration = "".join(group.text for group in groups if group.voice_index == -1)
    assert spoken == '"Stay here." "I will wait." '
    assert narration == "He said, She answered, They listened."


def test_balanced_scene_ornament_never_becomes_a_generation_prompt():
    groups = PhraseGrouper.text_to_groups(
        "Before the scene.\n\n✦\n\nAfter the scene.\n\n✦",
        160, SegmentationStrategy.BALANCED,
    )
    assert len(groups) == 2
    assert all(any(char.isalnum() for char in group.text) for group in groups)
    assert "Before" in groups[0].text and "After" not in groups[0].text
    assert "After" in groups[1].text
    assert groups[0].last_reason == Reason.SPACE_BREAK


def test_balanced_whole_section_is_repeatable_without_mutating_input():
    text = "\n\n".join(
        f"Paragraph {index} opens quietly. Wait. "
        "A much longer sentence follows, winding through the room, before it ends."
        for index in range(250)
    )
    snapshots = []
    for _ in range(2):
        groups = PhraseGrouper.text_to_groups(text, 160, SegmentationStrategy.BALANCED)
        assert "".join(group.text for group in groups) == text
        snapshots.append([
            (group.text, group.voice_index, tuple(p.reason for p in group.phrases))
            for group in groups
        ])
    assert snapshots[0] == snapshots[1]


def test_balanced_large_single_paragraph_preserves_order_and_limits():
    # Exercise one DP window across thousands of phrases, rather than many
    # short paragraphs that reset the search before its bound matters.
    phrases = [Phrase(f"Sentence {index} ends here. ", Reason.SENTENCE)
               for index in range(6000)]
    groups = PhraseGrouper.balanced_groups(phrases, 80)
    assert all(0 < group.num_words <= 80 for group in groups)
    flattened = [phrase for group in groups for phrase in group.phrases]
    assert len(flattened) == len(phrases)
    assert all(actual is expected for actual, expected in zip(flattened, phrases))
