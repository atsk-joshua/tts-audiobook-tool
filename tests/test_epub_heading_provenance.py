"""CPU-only heading metadata contracts; extraction text stays authoritative."""
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from tts_audiobook_tool.app_types import SegmentationStrategy
from tts_audiobook_tool.app_types.phrase import Phrase, PhraseGroup, Reason
from tts_audiobook_tool.text_ops.epub_extractor import (
    BeautifulSoupEpubChapterTextExtractor as Extractor,
    EpubExtractor, EpubSourceChapter, EpubTextChapter, EpubTextExtractionResult, EpubTextSlice,
)


def source(html, targets=()):
    return EpubSourceChapter("Opening", "Text/story.xhtml", "application/xhtml+xml", html, list(targets))


def target(fragment="", order=0):
    value = EpubExtractor.make_navigation_target("Text/story.xhtml" + ("#" + fragment if fragment else ""),
                                                fragment or "Opening", order)
    assert value is not None
    return value


def heading_texts(value):
    assert value.heading_spans is not None
    return [value.text[start:end] for start, end in value.heading_spans]


@pytest.mark.parametrize("level", range(1, 7))
def test_every_heading_level_has_exact_ranges_without_changing_text(level):
    chapter = source(f"<body><p>Before.</p><h{level}>A <em>long</em> heading.</h{level}><p>After.</p></body>")
    extractor = Extractor()
    result = extractor.extract_text(chapter)
    soup = extractor.import_beautiful_soup()(chapter.html, "html.parser")
    legacy_text = extractor.normalize_output_text(extractor.node_to_text(soup.body))
    assert result.text == legacy_text == "Before.\n\nA long heading.\n\nAfter."
    assert heading_texts(result) == ["A long heading."]
    assert heading_texts(result.slices[0]) == ["A long heading."]


def test_default_extractor_verifies_empty_and_skipped_heading_content():
    result = Extractor().extract_text(source("<body><script><h1>Hidden</h1></script><h2> </h2><p>Body.</p></body>"))
    assert result.text == "Body."
    assert result.heading_spans == result.slices[0].heading_spans == []


def test_optional_fields_preserve_existing_positional_constructors_as_unknown():
    assert EpubTextChapter("Title", "file", "Body.").heading_spans is None
    assert EpubTextSlice("Body.", None).heading_spans is None
    assert EpubTextExtractionResult("Body.", [], [], [], False).heading_spans is None


def test_unicode_entities_whitespace_and_duplicate_text_keep_actual_heading_origins():
    result = Extractor().extract_text(source(
        "<body><p>Étoile &amp; moon.</p><h1> Étoile&nbsp; &amp;\r\n moon. </h1>"
        "<p>Étoile &amp; moon.</p><h2>Étoile &amp; moon.</h2></body>"))
    assert result.text == "Étoile & moon.\n\nÉtoile &\nmoon.\n\nÉtoile & moon.\n\nÉtoile & moon."
    assert heading_texts(result) == ["Étoile &\nmoon.", "Étoile & moon."]
    first, second = result.heading_spans
    assert first[0] == result.text.index("Étoile &\n")
    assert second[0] == result.text.rindex("Étoile & moon.")
    assert second[0] != len(result.text[:second[0]].encode("utf-8"))


def test_inline_navigation_anchor_splits_heading_into_exact_slice_local_fragments():
    result = Extractor().extract_text(source(
        '<body><p>Prelude.</p><h1>First <span id="mid">second</span> final.</h1><p>Body.</p></body>',
        [target("mid")]))
    assert heading_texts(result) == ["First second final."]
    assert [value.text for value in result.slices] == ["Prelude.\n\nFirst", "second final.\n\nBody."]
    assert [heading_texts(value) for value in result.slices] == [["First"], ["second final."]]
    assert result.slices[1].navigation_target.fragment == "mid"


def test_heading_at_navigation_anchor_starts_at_zero_in_new_slice():
    result = Extractor().extract_text(source('<body><p>Before.</p><h3 id="next">Next.</h3><p>Body.</p></body>',
                                            [target("next")]))
    assert result.slices[0].heading_spans == []
    assert result.slices[1].heading_spans == [(0, 5)]


def test_unsafe_normalization_projection_returns_unknown_without_changing_output():
    # An entity split across pieces is deliberately nonlocal normalization.
    text, spans = Extractor.normalize_text_with_heading_spans(["&am", "p;"], [(1, 2)])
    assert text == "&" and spans is None


@pytest.mark.parametrize("bad", [[(-1, 1)], [(0, 99)], [(True, 1)], ["bad"]])
def test_invalid_piece_provenance_is_unknown(bad):
    assert Extractor.normalize_text_with_heading_spans(["Body."], bad) == ("Body.", None)


def test_spine_fallback_copies_full_document_ranges_and_preserves_unknown_custom_extractors():
    known = EpubTextExtractionResult("Heading.\n\nBody.", heading_spans=[(0, 8)])
    unknown = EpubTextExtractionResult("Other body.")
    chapters = EpubExtractor.make_spine_text_chapters([(source(""), known), (source(""), unknown)])
    assert heading_texts(chapters[0]) == ["Heading."]
    assert chapters[0].heading_spans is not known.heading_spans
    assert chapters[1].heading_spans is None


def test_navigation_assembly_shifts_ranges_by_exact_existing_three_lf_separator():
    result = EpubTextExtractionResult("unused", slices=[
        EpubTextSlice("Étoile.", target(), []),
        EpubTextSlice("Heading.\n\nBody.", None, [(0, 8)]),
        EpubTextSlice("Last.", None, []),
    ])
    chapters, used = EpubExtractor.assemble_navigation_text_chapters([(source(""), result)], [], [])
    assert used and len(chapters) == 1
    assert chapters[0].text == "Étoile.\n\n\nHeading.\n\nBody.\n\n\nLast."
    assert chapters[0].heading_spans == [(10, 18)]
    assert heading_texts(chapters[0]) == ["Heading."]


@pytest.mark.parametrize("unknown", [None, [(0, 999)], [(True, 2)], ["bad"]])
def test_one_unknown_or_invalid_included_slice_makes_whole_logical_chapter_unknown(unknown):
    result = EpubTextExtractionResult("unused", slices=[
        EpubTextSlice("Heading.", target(), [(0, 8)]), EpubTextSlice("Body.", None, unknown),
    ])
    chapters, _ = EpubExtractor.assemble_navigation_text_chapters([(source(""), result)], [], [])
    assert chapters[0].text == "Heading.\n\n\nBody."
    assert chapters[0].heading_spans is None


def test_empty_omitted_slice_does_not_poison_known_chapter_and_target_resets_offsets():
    result = EpubTextExtractionResult("unused", slices=[
        EpubTextSlice("First.", target("one"), []), EpubTextSlice("", None, None),
        EpubTextSlice("Heading.", target("two", 1), [(0, 8)]),
    ])
    chapters, _ = EpubExtractor.assemble_navigation_text_chapters([(source(""), result)], [], [])
    assert [value.heading_spans for value in chapters] == [[], [(0, 8)]]


def test_navigation_synthesized_slice_inherits_document_provenance():
    result = EpubTextExtractionResult("Heading.", heading_spans=[(0, 8)])
    chapters, used = EpubExtractor.assemble_navigation_text_chapters([(source("", [target()]), result)], [], [])
    assert used and chapters[0].heading_spans == [(0, 8)]


def test_unresolved_navigation_fallback_uses_full_document_provenance():
    chapter = source('<body><h1>Heading.</h1><p>Body.</p></body>', [target("missing")])
    result = Extractor().extract_text(chapter)
    assert result.has_unresolved_navigation_target
    with patch.object(EpubExtractor, "log_warnings"):
        chapters = EpubExtractor.assemble_text_chapters([(chapter, result)], [], [])
    assert chapters[0].text == result.text
    assert chapters[0].heading_spans == result.heading_spans


@pytest.mark.parametrize("narrative", [False, True])
def test_import_passes_provenance_only_to_narrative_without_changing_legacy_kwargs(narrative):
    chapter = source('<body><h1>Heading.</h1><p>Body.</p></body>')
    strategy = SimpleNamespace(id="narrative") if narrative else SegmentationStrategy.BALANCED
    calls = []

    def group(text, **kwargs):
        calls.append(kwargs)
        return [PhraseGroup([Phrase(text, Reason.PARAGRAPH)])]

    with patch.object(EpubExtractor, "load_source_chapters", return_value=([chapter], "Book", [], [])), \
            patch.object(EpubExtractor, "log_warnings"), \
            patch("tts_audiobook_tool.text_ops.epub_extractor.PhraseGrouper.text_to_groups", side_effect=group):
        result = EpubExtractor.import_epub("unused.epub", 80, strategy, "en")
    assert len(calls) == 1
    if narrative:
        assert calls[0]["narrative_heading_spans"] == result.chapters[0].heading_spans == [(0, 8)]
    else:
        assert "narrative_heading_spans" not in calls[0]
    assert result.phrase_groups[-1].last_reason == Reason.SECTION_BREAK
