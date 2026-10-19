"""Opt-in dispatcher/provenance and persisted-book lifecycle, CPU only."""
from copy import deepcopy
import json
from unittest.mock import patch

import pytest

from tts_audiobook_tool.app_types import Book, BookSection, BookSegmentationSettings, SegmentationStrategy
from tts_audiobook_tool.app_types.book_serialization import book_from_json_dict, book_to_json_dict
from tts_audiobook_tool.app_types.phrase import Reason
from tts_audiobook_tool.text_ops.narrative_grouper import normalize_narrative_source, plan_narrative_groups
from tts_audiobook_tool.text_ops.phrase_grouper import PhraseGrouper
from tts_audiobook_tool.text_ops.epub_extractor import EpubExtractor, EpubSourceChapter


BODY = "The meadow shone.\n\nThe wind settled.\n\nThe river rose."


def rows(groups):
    return [group.to_json_dict() for group in groups]


def test_enum_is_explicit_experimental_and_default_is_unchanged():
    strategy = SegmentationStrategy.from_id('narrative')
    assert strategy is SegmentationStrategy.NARRATIVE
    assert 'experimental' in strategy.label.lower()
    assert all(word in strategy.description for word in ['EPUB', 'native model', 'Plain text', 'manual'])
    assert BookSegmentationSettings().strategy is SegmentationStrategy.SENTENCE_PLUS
    assert rows(PhraseGrouper.text_to_groups(BODY, 40)) == rows(
        PhraseGrouper.text_to_groups(BODY, 40, SegmentationStrategy.SENTENCE_PLUS))


def test_verified_no_headings_packs_but_unknown_provenance_is_balanced():
    opted = PhraseGrouper.text_to_groups(BODY, 40, SegmentationStrategy.NARRATIVE,
                                       narrative_heading_spans=[])
    assert len(opted) == 2 and ''.join(group.text for group in opted) == BODY
    assert opted[0].text == 'The meadow shone.\n\n'
    assert opted[1].text == 'The wind settled.\n\nThe river rose.'
    unknown = PhraseGrouper.text_to_groups(BODY, 40, SegmentationStrategy.NARRATIVE)
    assert len(unknown) == 3
    assert rows(unknown) == rows(PhraseGrouper.text_to_groups(BODY, 40, SegmentationStrategy.BALANCED))


def test_real_epub_punctuated_paragraph_title_keeps_existing_heading_pause():
    chapter = EpubSourceChapter('A quiet morning.', 'chapter.xhtml', 'application/xhtml+xml',
        '<body><p>A quiet morning.</p><p>The meadow shone.</p><p>The river rose.</p></body>')
    with patch.object(EpubExtractor, 'load_source_chapters', return_value=([chapter], '', [], [])):
        result = EpubExtractor.import_epub('unused.epub', 80, SegmentationStrategy.NARRATIVE, 'en')
    assert result.chapters[0].heading_spans == []
    assert len(result.phrase_groups) == 2
    title, body = result.phrase_groups
    assert title.text == 'A quiet morning.\n\n'
    assert title.last_reason == Reason.HEADING_END
    assert body.text == 'The meadow shone.\n\nThe river rose.'
    assert body.last_reason == Reason.SECTION_BREAK
    assert ''.join(group.text for group in result.phrase_groups) == result.raw_text


@pytest.mark.parametrize('separator', ['\n','\r','\r\n'])
@pytest.mark.parametrize('leading', ['', '\r\n \t\r\n\n'])
def test_raw_heading_offsets_translate_through_common_normalization(separator,leading):
    parts = ['Ordinary narrative begins.', 'A surprisingly lengthy heading ends in punctuation.',
             'The meadow shone.', 'The river rose.']
    raw = leading + (' \t'+separator+separator).join(parts)
    start = len(leading) + len(parts[0]) + len(' \t'+separator+separator)
    end = start + len(parts[1])
    normalized = normalize_narrative_source(raw)
    heading = normalized.index(parts[1])
    mapped = PhraseGrouper._normalize_narrative_heading_spans(raw, [(start,end)])
    assert mapped == (normalized, [(heading,heading+len(parts[1]))])
    actual = PhraseGrouper.text_to_groups(raw, 80, SegmentationStrategy.NARRATIVE,
                                        narrative_heading_spans=[(start,end)])
    expected = plan_narrative_groups(normalized, 80, protected_spans=mapped[1],protect_section_opening=True)
    assert rows(actual) == rows(expected)
    assert ''.join(group.text for group in actual) == normalized
    assert len(actual) == 3
    assert parts[1] in actual[1].text and parts[2] not in actual[1].text


@pytest.mark.parametrize('spans', [None, 'unknown', [None], [(0,)], [(0,999)], [(-1,3)],
    [(2,2)], [(3,1)], [(False,3)], [(0,3),(2,4)], [(5,8),(0,3)]])
@pytest.mark.parametrize('dialog', [False,True])
def test_invalid_or_unknown_ranges_fail_closed_without_changing_source(spans,dialog):
    source = 'The meadow shone.\n\n"Stay here."\n\nThe river rose.'
    actual = PhraseGrouper.text_to_groups(source, 40, SegmentationStrategy.NARRATIVE,
        dialog_segmentation=dialog,narrative_heading_spans=spans)
    expected = PhraseGrouper.text_to_groups(source, 40, SegmentationStrategy.BALANCED,
        dialog_segmentation=dialog)
    assert rows(actual) == rows(expected)


def test_heading_range_removed_entirely_by_normalization_falls_back():
    raw = '\r\n \t\r\n'+BODY
    assert PhraseGrouper._normalize_narrative_heading_spans(raw,[(0,2)]) is None
    assert rows(PhraseGrouper.text_to_groups(raw,40,SegmentationStrategy.NARRATIVE,
        narrative_heading_spans=[(0,2)])) == rows(PhraseGrouper.text_to_groups(raw,40,SegmentationStrategy.BALANCED))


def test_duplicate_heading_text_uses_exact_ranges_without_searching():
    heading = 'The repeated heading is deliberately a complete sentence.'
    opening = 'An opening sentence.\n\n'
    source = opening+heading+'\n\n'+BODY+'\n\n'+heading+'\n\n'+BODY
    start = len(opening+heading+'\n\n'+BODY+'\n\n')
    groups = PhraseGrouper.text_to_groups(source,100,SegmentationStrategy.NARRATIVE,
        narrative_heading_spans=[(start,start+len(heading))])
    assert len(groups) == 4
    assert groups[0].text == opening
    assert groups[1].text == source[len(opening):start]
    assert groups[2].text == heading+'\n\n'
    assert groups[3].text == BODY


@pytest.mark.parametrize('dialog',[False,True])
def test_product_dispatch_retains_dialogue_scene_barriers_and_phrase_reasons(dialog):
    source = 'The meadow shone.\n\n"Stay here."\n\nThe river rose.\n\n◇\n\nThe wind settled.'
    groups = PhraseGrouper.text_to_groups(source,80,SegmentationStrategy.NARRATIVE,
        dialog_segmentation=dialog,narrative_heading_spans=[])
    assert ''.join(group.text for group in groups) == source
    assert not any('meadow' in group.text and 'Stay' in group.text for group in groups)
    assert not any('river' in group.text and 'wind' in group.text for group in groups)
    assert all(isinstance(phrase.reason,Reason) for group in groups for phrase in group.phrases)
    assert any(group.voice_index==1 for group in groups) if dialog else all(group.voice_index==-1 for group in groups)


def test_dispatch_roundtrip_persists_only_applied_strategy_and_existing_phrase_metadata(tmp_path):
    groups = PhraseGrouper.text_to_groups(BODY,40,SegmentationStrategy.NARRATIVE,narrative_heading_spans=[])
    book = Book(sections=[BookSection(groups,title='Chapter')],text_source_kind='epub',
        segmentation_settings=BookSegmentationSettings('en',40,SegmentationStrategy.NARRATIVE,False))
    expected = deepcopy(book_to_json_dict(book))
    path = tmp_path/'project_text.json'
    path.write_text(json.dumps(expected))
    loaded = book_from_json_dict(json.loads(path.read_text()))
    assert isinstance(loaded,Book)
    assert loaded.segmentation_settings.strategy is SegmentationStrategy.NARRATIVE
    assert book_to_json_dict(loaded) == expected
    assert rows(loaded.phrase_groups) == rows(groups)
    # Future-import selection is separate; loading/editing old groups never resegments.
    future = BookSegmentationSettings('en',160,SegmentationStrategy.BALANCED,True)
    assert future.strategy is not loaded.segmentation_settings.strategy
    assert book_to_json_dict(loaded) == expected and json.loads(path.read_text()) == expected


def test_non_narrative_strategies_ignore_heading_argument():
    for strategy in SegmentationStrategy:
        if strategy is not SegmentationStrategy.NARRATIVE:
            assert rows(PhraseGrouper.text_to_groups(BODY,40,strategy,narrative_heading_spans=[(-1,999)])) == rows(
                PhraseGrouper.text_to_groups(BODY,40,strategy))
