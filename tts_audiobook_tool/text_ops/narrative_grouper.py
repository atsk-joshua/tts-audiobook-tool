"""Pack connected prose paragraphs while preserving headings, scenes and voices.

Only verified EPUB imports opt into this strategy. Unknown heading provenance
falls back to paragraph-local grouping in the dispatcher. Paragraphs inside a
packed group use model timing rather than enforced external paragraph pauses.
"""
from __future__ import annotations

from collections.abc import Sequence
import re

from tts_audiobook_tool.app_support import app_text
from tts_audiobook_tool.app_types.phrase import PhraseGroup, Reason
from tts_audiobook_tool.text_ops.dialog_segmenter import DIALOG_VOICE_INDEX, DialogSegmenter
from tts_audiobook_tool.text_ops.phrase_grouper import PhraseGrouper
from tts_audiobook_tool.text_ops.phrase_segmenter import PhraseSegmenter
from tts_audiobook_tool.text_ops.quote_spans import find_quote_spans


def normalize_narrative_source(text: str) -> str:
    """Use the common grouper's normalization; spans address this string."""
    normalized = re.sub(r"[ \t]+\n", "\n", text.replace("\r\n", "\n").replace("\r", "\n"))
    # The common phrase pipeline drops leading structural line feeds before
    # its first vocalizable phrase. Spans must address that same canonical
    # source, rather than shifting or exceeding it after phrase creation.
    return normalized.lstrip("\n")


def _validate_cap(max_words: int) -> None:
    if type(max_words) is not int or max_words <= 0:
        raise ValueError("max_words must be a positive integer")


def _validate_spans(spans: Sequence[tuple[int, int]], length: int) -> None:
    for span in spans:
        if (len(span) != 2 or any(type(value) is not int for value in span)
                or not 0 <= span[0] < span[1] <= length):
            raise ValueError("protected_spans must be nonempty half-open normalized-source ranges")


def _is_protected_paragraph(group: PhraseGroup) -> bool:
    text = group.text
    stripped = text.strip()
    if any(phrase.reason >= Reason.HEADING_END for phrase in group.phrases):
        return True
    if any(line.strip() and not app_text.is_vocalizable(line.strip()) for line in text.splitlines()):
        return True
    # Protect quotations and dialogue-dash paragraphs before merging narration.
    if any(char in text for char in '\"“”«»「」『』') or find_quote_spans(text):
        return True
    if stripped.startswith(("—", "–", "- ", "‘", "‹")):
        return True
    sentence_end = stripped
    # Conservative fallback for titles lacking importer metadata. It intentionally
    # excludes some short narration and cannot identify every possible heading.
    if group.num_words <= 12 and sentence_end and sentence_end[-1] not in ".!?…。！？":
        return True
    return False


def pack_narrative_paragraphs(
    paragraphs: Sequence[PhraseGroup],
    max_words: int,
    *,
    protected_spans: Sequence[tuple[int, int]] = (),
    protect_section_opening: bool = True,
) -> list[PhraseGroup]:
    """Greedily pack complete paragraphs; preserve phrase objects and voices.

    Call once per section. Inputs must be complete paragraphs, not existing
    arbitrary inference groups. Existing HEADING_END reasons take precedence.
    Spans are offsets in the concatenated input text and protect every overlapping
    paragraph on both sides. Long paragraphs use existing BALANCED grouping and
    never join their tail to a neighbor. A pre-existing oversized atomic phrase
    cannot be split without changing metadata, so it is rejected explicitly.
    """
    _validate_cap(max_words)
    paragraphs = list(paragraphs)
    _validate_spans(protected_spans, sum(len(group.text) for group in paragraphs))
    if any(not group.phrases for group in paragraphs):
        raise ValueError("paragraph groups must contain phrases")
    if any(phrase.num_words > max_words for group in paragraphs for phrase in group.phrases):
        raise ValueError("split oversized phrases before packing narrative paragraphs")
    result: list[PhraseGroup] = []
    pending: list[PhraseGroup] = []
    pending_words = 0

    def flush() -> None:
        nonlocal pending_words
        if pending:
            result.append(PhraseGroup([phrase for group in pending for phrase in group.phrases],
                                      voice_index=pending[0].voice_index))
            pending.clear()
        pending_words = 0

    offset = 0
    for index, paragraph in enumerate(paragraphs):
        end = offset + len(paragraph.text)
        protected = ((index == 0 and protect_section_opening)
                     or _is_protected_paragraph(paragraph)
                     or any(start < end and finish > offset for start, finish in protected_spans))
        offset = end
        if protected or paragraph.num_words > max_words:
            flush()
            if paragraph.num_words > max_words:
                pieces = PhraseGrouper.balanced_groups(paragraph.phrases, max_words)
                result.extend(PhraseGroup(piece.phrases, voice_index=paragraph.voice_index) for piece in pieces)
            else:
                result.append(PhraseGroup(paragraph.phrases.copy(), voice_index=paragraph.voice_index))
            continue
        if pending and (pending_words + paragraph.num_words > max_words
                        or pending[-1].voice_index != paragraph.voice_index):
            flush()
        pending.append(paragraph)
        pending_words += paragraph.num_words
    flush()
    return result


def plan_narrative_groups(
    text: str,
    max_words: int,
    pysbd_lang: str = "en",
    dialog_segmentation: bool = False,
    protected_spans: Sequence[tuple[int, int]] = (),
    *,
    protect_section_opening: bool = True,
) -> list[PhraseGroup]:
    """Plan a fresh section without modifying any Book, files or import settings.

    ``protect_section_opening=False`` is appropriate only for a source verified
    to contain no opening heading, such as a controlled narration benchmark.
    """
    _validate_cap(max_words)
    source = normalize_narrative_source(text)
    phrases = PhraseSegmenter.text_to_phrases(source, max_words=max_words, pysbd_lang=pysbd_lang)
    # The legacy reason grouper may append an empty trailing group after a final
    # structural boundary; it is an intermediate artifact, not source content.
    paragraphs = [group for group in PhraseGrouper.phrases_to_groups_by_reason(phrases, Reason.PARAGRAPH)
                  if group.phrases]
    result = pack_narrative_paragraphs(paragraphs, max_words,
                                       protected_spans=protected_spans,
                                       protect_section_opening=protect_section_opening)
    if dialog_segmentation:
        result = DialogSegmenter.segment_groups(result, dialog_voice_index=DIALOG_VOICE_INDEX,
                                                 language_code=pysbd_lang)
    # The dialog pass or a trailing ornament without a final newline can expose
    # a non-vocalizable group. Use the common final cleanup: attach only the
    # ornament, preserve its text, and retain the strong scene boundary.
    return PhraseGrouper.merge_ornamental_groups(result)
