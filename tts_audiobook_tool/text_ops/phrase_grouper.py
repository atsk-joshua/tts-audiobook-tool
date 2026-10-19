import re
from bisect import bisect_right

from tts_audiobook_tool.app_types import SegmentationStrategy
from tts_audiobook_tool.app_types.phrase import PhraseGroup
from tts_audiobook_tool.app_support import app_text
from tts_audiobook_tool.text_ops.phrase_segmenter import Reason, Phrase, PhraseSegmenter
from tts_audiobook_tool.text_ops.dialog_segmenter import (
    DIALOG_VOICE_INDEX,
    DialogSegmenter,
)
from tts_audiobook_tool.text_ops.quote_spans import find_quote_spans
from tts_audiobook_tool.util import printt

class PhraseGrouper:
    """
    Creates PhraseGroups from Phrases or strings
    """

    @staticmethod
    def text_to_groups(
            text: str,
            max_words: int,
            strategy: SegmentationStrategy=SegmentationStrategy.SENTENCE_PLUS,
            pysbd_lang: str="en",
            dialog_segmentation: bool=False,
            target_words: int | None=None,
            narrative_heading_spans: list[tuple[int, int]] | None=None,
    ) -> list[PhraseGroup]:
        """
        Creates PhraseGroups using the passed-in raw source text.
        
        This is the app's main, high-level function for chunking text.
        One PhraseGroup gets transformed into one TTS prompt.

        When dialog_segmentation is enabled, a final pass subdivides groups so
        accepted dialog and narration do not share a group, preassigns dialog
        to voice sample 2, and marks close-quote pieces followed by an
        attribution with reason PHRASE_QUOTE_END (a lowercase continuation, or
        for language code "en" a speaker name followed by a whitelisted verb).
        It does not recombine groups created by the normal segmentation passes.

        NARRATIVE requires verified heading spans in this exact incoming text.
        An empty list certifies no headings; None or invalid ranges conservatively
        uses paragraph-local BALANCED. No source-text search repairs provenance.
        """
        if strategy == SegmentationStrategy.NARRATIVE:
            mapped = PhraseGrouper._normalize_narrative_heading_spans(text, narrative_heading_spans)
            if mapped is not None:
                from tts_audiobook_tool.text_ops.narrative_grouper import plan_narrative_groups
                source, spans = mapped
                return plan_narrative_groups(source, max_words, pysbd_lang,
                    dialog_segmentation=dialog_segmentation, protected_spans=spans,
                    protect_section_opening=True)
            strategy = SegmentationStrategy.BALANCED
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        # This guarantees that imported text has no blank lines containing
        # whitespace, and that no other line ends with spaces or tabs.
        text = re.sub(r"[ \t]+\n", "\n", text)

        if strategy == SegmentationStrategy.BALANCED:
            if max_words <= 0:
                raise ValueError("max_words must be positive")
            if target_words is not None and not 1 <= target_words <= max_words:
                raise ValueError("target_words must be between 1 and max_words")

        phrases = PhraseSegmenter.text_to_phrases(text, max_words=max_words, pysbd_lang=pysbd_lang)

        # First group by either complete sentence or paragraph
        match strategy:
            case SegmentationStrategy.SENTENCE:
                reason_threshold = Reason.SENTENCE
            case SegmentationStrategy.SENTENCE_PLUS:
                reason_threshold = Reason.SENTENCE
            case SegmentationStrategy.MULTI_SENTENCE:
                reason_threshold = Reason.SENTENCE # (will re-combine sentences later)
            case SegmentationStrategy.MAX_LEN:
                reason_threshold = Reason.PARAGRAPH
            case SegmentationStrategy.BALANCED:
                reason_threshold = Reason.PARAGRAPH
        if strategy == SegmentationStrategy.BALANCED:
            groups = PhraseGrouper.balanced_groups(phrases, max_words, target_words)
        else:
            groups = PhraseGrouper.phrases_to_groups_by_reason(phrases, reason_threshold)

        if strategy == SegmentationStrategy.SENTENCE_PLUS:
            # Mitigates tts glitches due to too-short prompts
            SHORT_SENTENCE_NUM_WORDS = 2
            groups = PhraseGrouper.merge_short_sentences(groups, SHORT_SENTENCE_NUM_WORDS, max_words)
        elif strategy == SegmentationStrategy.MULTI_SENTENCE:
            # Piggybacks off same logic
            groups = PhraseGrouper.merge_short_sentences(groups, 9999, max_words)

        # Split group when exceeds max_words
        results = []
        for group in groups:
            split_groups = PhraseGrouper.group_to_groups_by_max_words(group, max_words)
            results.extend(split_groups)
        groups = results

        if dialog_segmentation:
            groups = DialogSegmenter.segment_groups(
                groups,
                dialog_voice_index=DIALOG_VOICE_INDEX,
                language_code=pysbd_lang,
            )

        groups = PhraseGrouper.merge_ornamental_groups(groups)

        return groups

    @staticmethod
    def _normalize_narrative_heading_spans(text, spans):
        """Map verified raw offsets through only the common deleted characters.

        CR-to-LF substitutions retain positions; CRLF removes CR, pre-newline
        horizontal space is removed, and leading normalized LFs are removed.
        Prefix sums over merged deletions translate endpoints without guessing
        heading text. Malformed, overlapping or vanished ranges fail closed.
        """
        if not isinstance(spans, (list, tuple)):
            return None
        checked = []
        prior_end = 0
        for span in spans:
            if (not isinstance(span, (list, tuple)) or len(span) != 2
                    or any(type(value) is not int for value in span)
                    or not 0 <= span[0] < span[1] <= len(text) or span[0] < prior_end):
                return None
            checked.append(tuple(span))
            prior_end = span[1]
        prefix = re.match(r"(?:[ \t]*(?:\r\n|\r|\n))*", text).end()
        deletions = [(0, prefix)] if prefix else []
        for match in re.finditer(r"\r(?=\n)|[ \t]+(?=\r|\n)", text):
            start, end = match.span()
            if deletions and start <= deletions[-1][1]:
                deletions[-1] = (deletions[-1][0], max(end, deletions[-1][1]))
            else:
                deletions.append((start, end))
        ends, removed = [], [0]
        for start, end in deletions:
            ends.append(end)
            removed.append(removed[-1] + end - start)

        def endpoint(point):
            index = bisect_right(ends, point)
            partial = max(0, point - deletions[index][0]) if index < len(deletions) else 0
            return point - removed[index] - partial

        normalized = re.sub(r"[ \t]+\n", "\n", text.replace("\r\n", "\n").replace("\r", "\n")).lstrip("\n")
        mapped = [(endpoint(start), endpoint(end)) for start, end in checked]
        if any(not 0 <= start < end <= len(normalized) for start, end in mapped):
            return None
        return normalized, mapped

    @staticmethod
    def balanced_groups(
            phrases: list[Phrase],
            max_words: int,
            target_words: int | None = None,
    ) -> list[PhraseGroup]:
        """Pack existing phrases deterministically without crossing paragraphs.

        A bounded dynamic program trades prompt count and squared distance from
        the target against weaker cut positions. Costs are expressed in units
        of one target-sized prompt: a phrase cut costs four such units, an
        arbitrary word cut eight, and a cut inside a balanced quote one extra.
        These are structural preferences, not claims about audio quality.

        Text and reasons are retained; dialog classification still runs later.
        A phrase exceeding the cap is retained alone, as in the legacy splitter.
        Normal input has at least one counted word per phrase. The explicit
        predecessor bound also caps work for degenerate zero-word input, giving
        O(P * min(P, max_words)) time and O(P) space per paragraph.
        """
        if max_words <= 0:
            raise ValueError("max_words must be positive")
        target = max(1, max_words * 3 // 4) if target_words is None else target_words
        if not 1 <= target <= max_words:
            raise ValueError("target_words must be between 1 and max_words")

        result: list[PhraseGroup] = []
        paragraphs = PhraseGrouper.phrases_to_groups_by_reason(phrases, Reason.PARAGRAPH)
        for paragraph in paragraphs:
            items = paragraph.phrases
            count = len(items)
            if not count:
                continue
            prefix = [0]
            offsets = [0]
            for phrase in items:
                prefix.append(prefix[-1] + phrase.num_words)
                offsets.append(offsets[-1] + len(phrase.text))

            # Sweep quote ranges once, including nested ranges, rather than
            # rescanning every quote at every candidate cut.
            spans = find_quote_spans("".join(phrase.text for phrase in items))
            quote_index = 0
            quote_end = 0
            cut_costs = [0] * (count + 1)
            unit = target * target
            weak_cut_cost = {
                Reason.UNDEFINED: 8,
                Reason.WORD: 8,
                Reason.PHRASE_QUOTE_END: 6,
                Reason.PHRASE: 4,
            }
            for end in range(1, count):
                offset = offsets[end]
                while quote_index < len(spans) and spans[quote_index].start < offset:
                    quote_end = max(quote_end, spans[quote_index].end)
                    quote_index += 1
                quote_cost = int(offset < quote_end)
                cut_costs[end] = unit * (weak_cut_cost.get(items[end - 1].reason, 0) + quote_cost)

            costs: list[int | None] = [None] * (count + 1)
            sizes = [0] * (count + 1)
            previous = [0] * (count + 1)
            costs[0] = 0
            for end in range(1, count + 1):
                best: tuple[int, int, int] | None = None
                for start in range(end - 1, max(-1, end - max_words - 1), -1):
                    words = prefix[end] - prefix[start]
                    if words > max_words and start != end - 1:
                        break
                    # Oversized indivisible phrases may stand alone only.
                    if words > max_words and end - start != 1:
                        continue
                    prior_cost = costs[start]
                    if prior_cost is None:
                        continue
                    candidate = (
                        prior_cost + unit + (words - target) ** 2 + cut_costs[end],
                        sizes[start] + 1,
                        start,
                    )
                    if best is None or candidate < best:
                        best = candidate
                assert best is not None  # Each individual phrase is admissible.
                costs[end], sizes[end], previous[end] = best

            boundaries = [count]
            while boundaries[-1]:
                boundaries.append(previous[boundaries[-1]])
            boundaries.reverse()
            result.extend(
                PhraseGroup(items[start:end])
                for start, end in zip(boundaries, boundaries[1:])
            )
        return result

    @staticmethod
    def merge_ornamental_groups(groups: list[PhraseGroup]) -> list[PhraseGroup]:
        """
        Folds ornamental-only groups into a vocalizable neighboring group.

        A PhraseGroup is the app's atomic TTS prompt unit, and a group whose
        text has no vocalizable content (a lone dinkus or divider line)
        normalizes to an empty prompt. Segmentation already merges ornamental
        lines into a neighboring phrase, but that invariant can be undone
        later: an ornament trailing the text has no trailing line breaks (so
        its reason stays SENTENCE and it never merges backward), and the
        optional dialog pass re-splits a merged phrase at quote boundaries,
        isolating its ornament again. This final pass restores the invariant
        for every position: mid-text and trailing ornamental-only groups
        merge backward into the previous group, and leading ones merge
        forward into the first vocalizable group.

        Groups are returned unchanged when no group has vocalizable text.
        """

        result: list[PhraseGroup] = []
        leading: list[PhraseGroup] = []

        for group in groups:
            if app_text.is_vocalizable(group.text):
                result.append(group)
                continue
            if not result:
                leading.append(group)
                continue
            # Merge backward, mirroring PhraseSegmenter.merge_ornamental_lines:
            # the ornament joins the previous group's last phrase and promotes
            # its boundary to a space break.
            if result[-1].phrases:
                last_phrase = result[-1].phrases[-1]
                last_phrase.text += group.text
                last_phrase.reason = Reason.SPACE_BREAK
                result[-1].invalidate_presentable_memos()
            else:
                result[-1].phrases.extend(group.phrases)
                result[-1].invalidate_presentable_memos()

        if leading:
            if result:
                # Attach leading ornaments to the first content phrase. The
                # ornament's own break carries no boundary meaning at the
                # start of the text, so the content phrase keeps its reason.
                ornament_text = "".join(group.text for group in leading)
                result[0].phrases[0].text = ornament_text + result[0].phrases[0].text
                result[0].invalidate_presentable_memos()
            else:
                # The text has no vocalizable content at all; leave as-is.
                result = leading

        return result

    @staticmethod
    def group_to_groups_by_max_words(group: PhraseGroup, max_words: int) -> list[PhraseGroup]:
        """
        Breaks up a PhraseGroup into multiple groups if its num_words > max_words.
        A phrase which already exceeds max_words remains in its own non-empty group.
        """
        result = []
        temp_group = PhraseGroup()

        for phrase in group.phrases:

            if temp_group.phrases and temp_group.num_words + phrase.num_words > max_words:
                result.append(temp_group)
                temp_group = PhraseGroup()

            temp_group.phrases.append(phrase)

        if temp_group.phrases:
            result.append(temp_group)

        # TODO: second pass, attempt to balance the num_words of each group maybe
        # eg, 10/10/10, 10, 10 -> 10/10, 10/10, 10/10

        return result

    @staticmethod
    def phrases_to_groups_by_reason(phrases: list[Phrase], reason_threshold: Reason) -> list[PhraseGroup]:
        """
        Groups phrases into PhraseGroups based on reason threshold
        (ie, when a phrase has the given reason's value or greater, it will cut a new group)
        """

        groups: list[PhraseGroup] = []
        group = PhraseGroup()

        for phrase in phrases:
            group.phrases.append(phrase)
            if phrase.reason.level >= reason_threshold.level:
                groups.append(group)
                group = PhraseGroup()
        if group:
            groups.append(group)

        return groups

    @staticmethod
    def merge_short_sentences(
            groups: list[PhraseGroup], 
            shortness_threshold: int,
            max_words: int
    ) -> list[PhraseGroup]:
        """
        Merges short single phrase group with neighbor if doesn't exceed max_words 
        and does not cross paragraph/section boundary. Uses two passes.

        Again, the reason for this operation is to prevent very short prompts,
        which TTS models do not like.
        """
        # Merge with previous
        new_groups = []
        for i, group in enumerate(groups):  
            did_merge = False            
            is_single_short_any = len(group.phrases) == 1 and group.num_words <= shortness_threshold
            if is_single_short_any and len(new_groups) > 0:                
                last_new_group = new_groups[-1]
                can_merge = last_new_group.num_words + group.num_words <= max_words and \
                    last_new_group.last_reason < Reason.PARAGRAPH
                if can_merge:
                    new_groups[-1].phrases.append(group.phrases[0])
                    new_groups[-1].invalidate_presentable_memos()
                    did_merge = True
            if not did_merge:
                new_groups.append(group)

        # Merge with next
        groups = new_groups
        new_groups: list[PhraseGroup] = []
        for i, group in enumerate(groups):            
            did_merge = False
            is_short_sentence = len(group.phrases) == 1 and \
                group.num_words <= shortness_threshold and \
                group.last_reason <= Reason.SENTENCE
            if is_short_sentence and i + 1 < len(groups):
                next_group = groups[i+1]
                can_merge = next_group.num_words + group.num_words <= max_words
                if can_merge:
                    next_group.phrases.insert(0, group.phrases[0])
                    next_group.invalidate_presentable_memos()
                    did_merge = True
            if not did_merge:
                new_groups.append(group)

        return new_groups            

    @staticmethod
    def print_groups(groups: list[PhraseGroup]) -> None:
        """ For debugging """
        for group in groups:
            printt(f"Group: ({group.num_words} words)")
            for i, phrase in enumerate(group.phrases):
                s = f"{phrase.reason} (words: {phrase.num_words})"
                if phrase.reason >= Reason.PARAGRAPH:
                    s += " -----"
                printt(f"  {repr(phrase.text)} {s}")
            printt()
