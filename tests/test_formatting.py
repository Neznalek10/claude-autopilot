"""Unit tests for autopilot.formatting.

Covers Markdown conversion (blocks and inline rules), splitting, the plain
text fallback, duration and time formatting (including conversion from fixed
UTC offsets to local time), run headers and messages, the result document,
the task list, the status reply and the help text.

Every produced chunk is checked with ``assert_valid_telegram_html``: balanced,
properly nested tags from the allowed set only, no raw ``<``, ``>`` or ``&``.
"""

from __future__ import annotations

import random
import re
import unittest
from unittest import mock
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from autopilot.formatting import (
    ATTACHED_NOTE,
    CHUNK_CHARS,
    STATUS_ICONS,
    TELEGRAM_MAX_CHARS,
    escape_html,
    format_duration,
    format_run_messages,
    format_status,
    format_task_list,
    format_time,
    help_text,
    html_to_plain,
    markdown_to_telegram_html,
    result_document,
    run_header,
    run_title,
    split_for_telegram,
)
from autopilot.models import BOT_COMMANDS, CHAT_IDLE_HOURS, QueueSnapshot
from helpers import TZ, make_record, make_task

ALLOWED_TAGS = {"b", "i", "s", "code", "pre", "a"}
# One tag token: optional slash, a name, optional attributes.
TAG_RE = re.compile(r"<(/?)([a-z]+)((?: [^<>]*)?)>")
# An ampersand that is not the start of a valid entity.
BAD_AMPERSAND_RE = re.compile(r"&(?!(?:amp|lt|gt|quot|#\d+);)")
NOW = datetime(2026, 11, 2, 12, 0, tzinfo=TZ)


def assert_valid_telegram_html(test: unittest.TestCase, chunk: str) -> None:
    """Fail the test unless ``chunk`` is valid on its own for Telegram.

    Checks: only allowed tags; every open tag closed in order; no tags inside
    ``pre`` or ``code``; ``<a>`` carries an http(s) href; the text between
    tags has no raw ``<`` or ``>`` and no unescaped ``&``.

    Args:
        test: The running test case, used for assertions.
        chunk: HTML to validate.
    """
    stack: list[str] = []
    position = 0
    for match in TAG_RE.finditer(chunk):
        text = chunk[position : match.start()]
        test.assertNotRegex(text, r"[<>]", f"raw angle bracket in {chunk!r}")
        test.assertIsNone(BAD_AMPERSAND_RE.search(text), f"bare & in {chunk!r}")
        position = match.end()
        closing, name, attributes = match.groups()
        test.assertIn(name, ALLOWED_TAGS, f"tag {name} not allowed in {chunk!r}")
        if closing:
            test.assertTrue(stack and stack[-1] == name, f"unbalanced </{name}> in {chunk!r}")
            stack.pop()
            continue
        test.assertFalse(
            any(open_tag in ("pre", "code") for open_tag in stack),
            f"tag <{name}> inside pre/code in {chunk!r}",
        )
        if name == "a":
            test.assertRegex(attributes, r'^ href="https?://[^"]+"$')
        else:
            test.assertEqual(attributes, "")
        stack.append(name)
    tail = chunk[position:]
    test.assertNotRegex(tail, r"[<>]", f"raw angle bracket in {chunk!r}")
    test.assertIsNone(BAD_AMPERSAND_RE.search(tail), f"bare & in {chunk!r}")
    test.assertEqual(stack, [], f"unclosed tags {stack} in {chunk!r}")


@dataclass(frozen=True)
class StubSchedule:
    """A stand-in for schedule.Schedule: formatting only reads ``description``."""

    description: str


class ValidatorSelfTests(unittest.TestCase):
    """The HTML validator itself must catch the mistakes it guards against."""

    def test_accepts_valid_html(self) -> None:
        """Nested allowed tags and entities pass."""
        assert_valid_telegram_html(
            self, '<b>a <i>b</i></b> <a href="https://x.org?a=1&amp;b=2">l</a> &lt;'
        )

    def test_rejects_broken_html(self) -> None:
        """Unclosed, crossed, forbidden, nested-in-pre and raw characters fail."""
        for bad in (
            "<b>open",
            "<b><i>x</b></i>",
            "<u>x</u>",
            "<pre><b>x</b></pre>",
            "a < b",
            "a & b",
            '<a href="javascript:x">l</a>',
        ):
            with self.subTest(bad=bad), self.assertRaises(AssertionError):
                assert_valid_telegram_html(self, bad)


class EscapeTests(unittest.TestCase):
    """escape_html neutralises markup characters."""

    def test_escapes_ampersand_and_angle_brackets(self) -> None:
        """& < > are escaped, & first so entities are not double-escaped."""
        self.assertEqual(escape_html("a<b>&c"), "a&lt;b&gt;&amp;c")

    def test_quotes_only_when_asked(self) -> None:
        """Double quotes stay unless quote=True."""
        self.assertEqual(escape_html('say "x"'), 'say "x"')
        self.assertEqual(escape_html('say "x"', quote=True), "say &quot;x&quot;")

    def test_unicode_is_untouched(self) -> None:
        """Accented text passes through."""
        self.assertEqual(escape_html("Naïve café résumé ☕"), "Naïve café résumé ☕")


class MarkdownBlockTests(unittest.TestCase):
    """Block-level conversion: fences, tables, headings, bullets, rules."""

    def test_empty_and_blank_input(self) -> None:
        """Nothing in, nothing out."""
        self.assertEqual(markdown_to_telegram_html(""), "")
        self.assertEqual(markdown_to_telegram_html("  \n\n \n"), "")
        self.assertEqual(split_for_telegram(""), [])
        self.assertEqual(split_for_telegram("\n\n"), [])

    def test_plain_text_is_escaped(self) -> None:
        """Markup characters in text never become tags."""
        self.assertEqual(markdown_to_telegram_html("1 < 2 & <b>x</b>"), "1 &lt; 2 &amp; &lt;b&gt;x&lt;/b&gt;")

    def test_fenced_code_becomes_pre_unformatted(self) -> None:
        """Code is escaped but not touched by inline rules; the language is dropped."""
        html = markdown_to_telegram_html("```python\nx = a < b  # **not bold**\n\nprint(1)\n```")
        self.assertEqual(html, "<pre>x = a &lt; b  # **not bold**\n\nprint(1)</pre>")

    def test_unclosed_fence_runs_to_the_end(self) -> None:
        """A missing closing fence swallows the rest of the text as code."""
        html = markdown_to_telegram_html("intro\n```\ncode\nmore")
        self.assertEqual(html, "intro\n\n<pre>code\nmore</pre>")

    def test_empty_fence_is_dropped(self) -> None:
        """A fence with nothing in it produces no empty <pre>."""
        self.assertEqual(markdown_to_telegram_html("a\n```\n```\nb"), "a\n\nb")

    def test_table_becomes_pre_unchanged(self) -> None:
        """Lines starting with | stay as typed inside <pre>."""
        table = "| Name | Qty |\n|---|---|\n| a & b | **2** |"
        self.assertEqual(
            markdown_to_telegram_html(table),
            "<pre>| Name | Qty |\n|---|---|\n| a &amp; b | **2** |</pre>",
        )

    def test_heading_becomes_bold_without_nested_bold(self) -> None:
        """Headings are bold; bold markers inside them are dropped, not nested."""
        self.assertEqual(markdown_to_telegram_html("## Plan **today** #"), "<b>Plan today</b>")
        self.assertEqual(markdown_to_telegram_html("#hashtag"), "#hashtag")

    def test_bullets_use_dots_and_keep_indent(self) -> None:
        """-, * and + bullets become "• " and nesting indent survives."""
        html = markdown_to_telegram_html("- one\n* two\n+ three\n  - nested")
        self.assertEqual(html, "• one\n• two\n• three\n  • nested")

    def test_horizontal_rules(self) -> None:
        """--- and *** lines become a line of box characters."""
        self.assertEqual(markdown_to_telegram_html("a\n\n---\n\n***"), "a\n\n──────\n\n──────")

    def test_blank_lines_collapse_to_one_between_blocks(self) -> None:
        """Several blank lines between paragraphs become exactly one."""
        self.assertEqual(markdown_to_telegram_html("a\n\n\n\nb"), "a\n\nb")

    def test_windows_line_endings(self) -> None:
        """CRLF input is treated like LF."""
        self.assertEqual(markdown_to_telegram_html("- a\r\n- b\r\n"), "• a\n• b")


class MarkdownInlineTests(unittest.TestCase):
    """Inline conversion rules and their protections."""

    def test_inline_code_is_protected(self) -> None:
        """Markers inside code spans are not interpreted and content is escaped."""
        self.assertEqual(
            markdown_to_telegram_html("use `a_b *c* <x>` now"),
            "use <code>a_b *c* &lt;x&gt;</code> now",
        )

    def test_bold_italic_strike(self) -> None:
        """Each marker pair maps to its tag."""
        self.assertEqual(
            markdown_to_telegram_html("**b** __b2__ *i* _i2_ ~~s~~"),
            "<b>b</b> <b>b2</b> <i>i</i> <i>i2</i> <s>s</s>",
        )

    def test_snake_case_and_arithmetic_stay_literal(self) -> None:
        """Underscores inside words and 2*3*4 are not emphasis."""
        for text in ("my_var_name and other_var", "2*3*4", "snake_case_here"):
            with self.subTest(text=text):
                self.assertEqual(markdown_to_telegram_html(text), text)

    def test_italic_inside_bold_nests_properly(self) -> None:
        """Mixed emphasis is nested, never crossed."""
        html = markdown_to_telegram_html("**a *b* c** and *x **y** z*")
        self.assertEqual(html, "<b>a <i>b</i> c</b> and <i>x <b>y</b> z</i>")
        assert_valid_telegram_html(self, html)

    def test_crossing_markers_still_produce_valid_html(self) -> None:
        """Overlapping markers can't produce crossed tags."""
        for text in ("**a *b** c*", "*a **b* c**", "~~a **b~~ c**", "[**a](https://x.org) b**"):
            with self.subTest(text=text):
                assert_valid_telegram_html(self, markdown_to_telegram_html(text))

    def test_link_with_http_and_https(self) -> None:
        """Links become anchors with an escaped URL."""
        html = markdown_to_telegram_html("[site](https://x.org/a?b=1&c=2) and [old](http://y.org)")
        self.assertEqual(
            html,
            '<a href="https://x.org/a?b=1&amp;c=2">site</a> and <a href="http://y.org">old</a>',
        )

    def test_link_url_is_protected_from_emphasis(self) -> None:
        """Underscores and stars in a URL are not turned into tags."""
        html = markdown_to_telegram_html("[x](https://a.org/_p_/**q**)")
        self.assertEqual(html, '<a href="https://a.org/_p_/**q**">x</a>')

    def test_link_label_can_be_formatted(self) -> None:
        """Bold inside a label nests inside the anchor."""
        html = markdown_to_telegram_html("[**big** news](https://x.org)")
        self.assertEqual(html, '<a href="https://x.org"><b>big</b> news</a>')

    def test_link_with_parentheses_in_url(self) -> None:
        """One level of balanced parentheses stays in the URL."""
        html = markdown_to_telegram_html("[w](https://en.wikipedia.org/wiki/Foo_(bar))")
        self.assertEqual(html, '<a href="https://en.wikipedia.org/wiki/Foo_(bar)">w</a>')

    def test_only_http_and_https_links(self) -> None:
        """javascript:, ftp: and relative targets stay as literal text."""
        for target in ("javascript:alert(1)", "ftp://x.org/f", "/relative", "mailto:a@b.cz"):
            with self.subTest(target=target):
                html = markdown_to_telegram_html(f"[x]({target})")
                self.assertNotIn("<a", html)
                assert_valid_telegram_html(self, html)

    def test_quote_in_url_cannot_break_the_attribute(self) -> None:
        """A double quote in a URL is escaped inside href."""
        html = markdown_to_telegram_html('[x](https://a.org/"onclick="y)')
        self.assertIn("&quot;", html)
        assert_valid_telegram_html(self, html)

    def test_unicode_survives(self) -> None:
        """Accented text with formatting converts cleanly."""
        html = markdown_to_telegram_html("**Naïve** café `résumé`")
        self.assertEqual(html, "<b>Naïve</b> café <code>résumé</code>")

    def test_stray_nul_characters_are_removed(self) -> None:
        """NUL (used internally for placeholders) in input can't forge placeholders."""
        html = markdown_to_telegram_html("a\x000\x00 `c`")
        self.assertEqual(html, "a0 <code>c</code>")

    def test_unmatched_markers_stay_literal(self) -> None:
        """Lonely * _ ` ~~ do not produce tags."""
        for text in ("a * b", "a _ b", "a ` b", "a ~~ b", "**", "* "):
            with self.subTest(text=text):
                assert_valid_telegram_html(self, markdown_to_telegram_html(text))


class SplitTests(unittest.TestCase):
    """split_for_telegram packs blocks and keeps every chunk valid and small."""

    def test_short_text_is_one_chunk(self) -> None:
        """Small input is not split."""
        self.assertEqual(split_for_telegram("# A\n\ntext"), ["<b>A</b>\n\ntext"])

    def test_chunks_respect_the_limit_and_start_at_block_edges(self) -> None:
        """Paragraphs are packed greedily and never cut in the middle."""
        paragraphs = [f"paragraph {i} " + "word " * 10 for i in range(20)]
        chunks = split_for_telegram("\n\n".join(paragraphs), limit=200)
        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertLessEqual(len(chunk), 200)
        rebuilt = "\n\n".join(chunks)
        self.assertEqual(rebuilt, "\n\n".join(p.rstrip() for p in paragraphs))

    def test_code_block_stays_whole_when_it_fits(self) -> None:
        """A fence that fits in one chunk is not split across chunks."""
        code = "```\n" + "\n".join(f"line {i}" for i in range(10)) + "\n```"
        chunks = split_for_telegram("a" * 150 + "\n\n" + code, limit=200)
        pre_chunks = [c for c in chunks if "<pre>" in c]
        self.assertEqual(len(pre_chunks), 1)
        self.assertEqual(pre_chunks[0].count("<pre>"), 1)
        self.assertIn("line 9", pre_chunks[0])

    def test_oversized_pre_is_split_by_lines_into_pre_blocks(self) -> None:
        """A giant code block becomes several balanced <pre> blocks with no lost lines."""
        lines = [f"row {i:03d} &" for i in range(200)]
        chunks = split_for_telegram("```\n" + "\n".join(lines) + "\n```", limit=300)
        self.assertGreater(len(chunks), 3)
        joined = ""
        for chunk in chunks:
            self.assertLessEqual(len(chunk), 300)
            assert_valid_telegram_html(self, chunk)
            self.assertTrue(chunk.startswith("<pre>") and chunk.endswith("</pre>"))
            joined += chunk[5:-6] + "\n"
        self.assertEqual(joined.rstrip("\n"), "\n".join(escape_html(line) for line in lines))

    def test_oversized_table_is_split_like_code(self) -> None:
        """Tables get the same treatment as code."""
        table = "\n".join(f"| {i} | value |" for i in range(100))
        chunks = split_for_telegram(table, limit=200)
        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            assert_valid_telegram_html(self, chunk)
            self.assertLessEqual(len(chunk), 200)

    def test_oversized_plain_line_is_cut_without_breaking_entities(self) -> None:
        """A huge line of & characters is cut so no &amp; is split, with no formatting."""
        line = "&" * 500 + " **not bold**"
        chunks = split_for_telegram(line, limit=64)
        self.assertGreater(len(chunks), 5)
        for chunk in chunks:
            self.assertLessEqual(len(chunk), 64)
            assert_valid_telegram_html(self, chunk)
        self.assertNotIn("<b>", "".join(chunks))
        plain = "".join(html_to_plain(c) for c in chunks)
        self.assertEqual(plain, line)

    def test_line_that_grows_past_the_limit_when_escaped_falls_back(self) -> None:
        """A line short in Markdown but long as HTML (many <) is still cut to fit."""
        chunks = split_for_telegram("<" * 40, limit=50)
        for chunk in chunks:
            self.assertLessEqual(len(chunk), 50)
            assert_valid_telegram_html(self, chunk)
        self.assertEqual("".join(html_to_plain(c) for c in chunks), "<" * 40)

    def test_long_paragraph_splits_between_lines(self) -> None:
        """A paragraph of many bullet lines is split on line boundaries, keeping formatting."""
        text = "\n".join(f"- item **{i}** " + "x" * 30 for i in range(40))
        chunks = split_for_telegram(text, limit=300)
        self.assertGreater(len(chunks), 3)
        for chunk in chunks:
            self.assertLessEqual(len(chunk), 300)
            assert_valid_telegram_html(self, chunk)
        self.assertEqual(sum(c.count("<b>") for c in chunks), 40)

    def test_limit_below_minimum_is_rejected(self) -> None:
        """Absurdly small limits fail loudly."""
        with self.assertRaises(ValueError):
            split_for_telegram("x", limit=5)

    def test_random_markdown_always_produces_valid_chunks(self) -> None:
        """Seeded fuzz: any mix of markers yields valid chunks within the limit."""
        pieces = [
            "**", "*", "_", "__", "~~", "`", "```", "\n", "\n\n", "- ", "# ", "|", "---",
            "[a](https://x.org/_q_)", "[b](", ")", "<", ">", "&", '"', "word", " ", "résumé", "snake_case",
        ]
        rng = random.Random(20261102)
        for _ in range(300):
            text = "".join(rng.choice(pieces) for _ in range(rng.randint(1, 60)))
            with self.subTest(text=text):
                assert_valid_telegram_html(self, markdown_to_telegram_html(text))
                for chunk in split_for_telegram(text, limit=48):
                    self.assertLessEqual(len(chunk), 48)
                    assert_valid_telegram_html(self, chunk)


class PlainTextTests(unittest.TestCase):
    """html_to_plain produces text that is safe to send without a parse mode."""

    def test_tags_removed_and_entities_decoded(self) -> None:
        """Formatting disappears and escaped characters come back."""
        self.assertEqual(html_to_plain("<b>a</b> &lt;x&gt; &amp; <code>y</code>"), "a <x> & y")

    def test_links_keep_their_url(self) -> None:
        """An anchor reads "label (url)", or just the URL when label equals it."""
        html = '<a href="https://x.org/?a=1&amp;b=2">site</a> <a href="https://y.org">https://y.org</a>'
        self.assertEqual(html_to_plain(html), "site (https://x.org/?a=1&b=2) https://y.org")

    def test_round_trip_of_converted_markdown(self) -> None:
        """Plain text of a converted document keeps the content and drops the tags."""
        plain = html_to_plain(markdown_to_telegram_html("# T\n- **a** < b\n```\nx & y\n```"))
        self.assertEqual(plain, "T\n• a < b\n\nx & y")


class DurationAndTimeTests(unittest.TestCase):
    """format_duration and format_time."""

    def test_format_duration_ranges(self) -> None:
        """Each range uses its own unit layout."""
        cases = {
            None: "?",
            0: "0 ms",
            850: "850 ms",
            999: "999 ms",
            1000: "1s",
            12_345: "12s",
            59_999: "59s",
            60_000: "1m 00s",
            133_000: "2m 13s",
            3_599_000: "59m 59s",
            3_720_000: "1h 02m",
            90_000_000: "25h 00m",
            -5: "0 ms",
        }
        for ms, expected in cases.items():
            with self.subTest(ms=ms):
                self.assertEqual(format_duration(ms), expected)

    def test_format_time_none_is_never(self) -> None:
        """A missing time reads "never"."""
        self.assertEqual(format_time(None, NOW), "never")

    def test_format_time_same_day(self) -> None:
        """Same calendar day shows only the clock time."""
        self.assertEqual(format_time(datetime(2026, 11, 2, 7, 30, tzinfo=TZ), NOW), "07:30")
        self.assertEqual(format_time(datetime(2026, 11, 2, 23, 59, tzinfo=TZ), NOW), "23:59")

    def test_format_time_within_six_days_uses_weekday(self) -> None:
        """Future and past days within 6 days show the weekday."""
        self.assertEqual(format_time(datetime(2026, 11, 3, 7, 30, tzinfo=TZ), NOW), "Tue 07:30")
        self.assertEqual(format_time(datetime(2026, 11, 8, 18, 0, tzinfo=TZ), NOW), "Sun 18:00")
        self.assertEqual(format_time(datetime(2026, 10, 27, 9, 5, tzinfo=TZ), NOW), "Tue 09:05")

    def test_format_time_further_away_uses_date(self) -> None:
        """Seven or more days away shows the full date."""
        self.assertEqual(format_time(datetime(2026, 11, 14, 7, 30, tzinfo=TZ), NOW), "2026-11-14 07:30")
        self.assertEqual(format_time(datetime(2026, 11, 9, 7, 30, tzinfo=TZ), NOW), "2026-11-09 07:30")

    def test_format_time_converts_fixed_utc_offset_to_local_wall_clock(self) -> None:
        """05:30 UTC on the same day, seen from Berlin (CET, +01:00), reads 06:30."""
        utc_moment = datetime(2026, 11, 2, 5, 30, tzinfo=timezone.utc)
        self.assertEqual(format_time(utc_moment, NOW), "06:30")

    def test_format_time_conversion_can_change_the_day(self) -> None:
        """23:30 UTC the day before is already today in Berlin."""
        moment = datetime(2026, 11, 1, 23, 30, tzinfo=timezone.utc)
        self.assertEqual(format_time(moment, NOW), "00:30")

    def test_format_time_rejects_naive_datetimes(self) -> None:
        """Naive input is a programming error."""
        with self.assertRaises(ValueError):
            format_time(datetime(2026, 11, 2, 7, 30), NOW)
        with self.assertRaises(ValueError):
            format_time(NOW, datetime(2026, 11, 2, 7, 30))


class RunHeaderTests(unittest.TestCase):
    """Titles and headers of run reports."""

    def test_run_title(self) -> None:
        """Chat, Answer and task names."""
        self.assertEqual(run_title(make_record(task_name="ask", trigger="chat")), "Chat")
        self.assertEqual(run_title(make_record(task_name="ask", trigger="manual")), "Answer")
        self.assertEqual(run_title(make_record(task_name="news")), "news")

    def test_success_header(self) -> None:
        """Icon, bold title and duration."""
        self.assertEqual(run_header(make_record(duration_ms=12_000)), "✅ <b>news</b> · 12s")

    def test_retry_marker(self) -> None:
        """Retries are flagged."""
        header = run_header(make_record(duration_ms=1000, retry_of=3))
        self.assertEqual(header, "✅ <b>news</b> · 1s · retry")

    def test_timeout_header_reads_timed_out_after(self) -> None:
        """Timeouts say so and add no duplicate error line."""
        record = make_record(status="timeout", duration_ms=1_800_000, error="Timed out after 30 minutes.")
        self.assertEqual(run_header(record), "⏱ <b>news</b> · timed out after 30m 00s")

    def test_error_header_has_escaped_error_line(self) -> None:
        """Failed runs show the error in italics, escaped."""
        record = make_record(status="error", duration_ms=2000, error="bad <thing> & more")
        self.assertEqual(
            run_header(record), "❌ <b>news</b> · 2s\n<i>bad &lt;thing&gt; &amp; more</i>"
        )

    def test_unknown_duration_and_title_escaping(self) -> None:
        """A missing duration is "?" and titles are escaped."""
        record = make_record(task_name="a&b", duration_ms=None)
        self.assertEqual(run_header(record), "✅ <b>a&amp;b</b> · ?")

    def test_every_known_status_has_an_icon(self) -> None:
        """Each status maps to its icon in the header."""
        for status, icon in STATUS_ICONS.items():
            with self.subTest(status=status):
                self.assertTrue(run_header(make_record(status=status)).startswith(icon))

    def test_unknown_status_gets_a_neutral_icon(self) -> None:
        """An unexpected status doesn't crash."""
        self.assertTrue(run_header(make_record(status="weird")).startswith("•"))

    def test_long_error_is_cut_with_an_ellipsis(self) -> None:
        """Errors longer than the cap are shortened inside the italic tag."""
        header = run_header(make_record(status="error", error="e" * 5000))
        self.assertLess(len(header), 700)
        self.assertTrue(header.endswith("…</i>"))
        assert_valid_telegram_html(self, header)


class RunMessageTests(unittest.TestCase):
    """format_run_messages builds the messages for a finished run."""

    def test_header_blank_line_and_result(self) -> None:
        """First message = header, blank line, converted result."""
        record = make_record(result="**Hi** there", duration_ms=5000)
        messages, needs_file = format_run_messages(record)
        self.assertEqual(messages, ["✅ <b>news</b> · 5s\n\n<b>Hi</b> there"])
        self.assertFalse(needs_file)

    def test_empty_result_placeholder(self) -> None:
        """No text gives an italic placeholder."""
        messages, _ = format_run_messages(make_record(result="", duration_ms=1000))
        self.assertEqual(messages, ["✅ <b>news</b> · 1s\n\n<i>(no text output)</i>"])

    def test_successful_chat_has_no_header(self) -> None:
        """A good chat answer is just the answer."""
        record = make_record(task_name="ask", trigger="chat", result="Sure, **yes**.")
        self.assertEqual(format_run_messages(record), (["Sure, <b>yes</b>."], False))

    def test_failed_chat_keeps_its_header(self) -> None:
        """Only successful chat turns drop the header."""
        record = make_record(task_name="ask", trigger="chat", status="error", error="boom", result="")
        messages, _ = format_run_messages(record)
        self.assertTrue(messages[0].startswith("❌ <b>Chat</b>"))

    def test_blocked_tools_notice_on_last_message(self) -> None:
        """The warning ends the last message, with escaped tool names."""
        record = make_record(result="ok", denied_tools=("Bash", "mcp__x<y>"))
        messages, _ = format_run_messages(record)
        self.assertEqual(len(messages), 1)
        self.assertTrue(
            messages[0].endswith(
                "\n\n⚠️ <i>Blocked tools: Bash, mcp__x&lt;y&gt;. "
                "Add them to allowed_tools in tasks.toml if this task needs them.</i>"
            )
        )
        assert_valid_telegram_html(self, messages[0])

    def test_multiple_messages_when_result_is_long(self) -> None:
        """A result needing up to max_messages chunks is sent in full."""
        result = "\n\n".join(f"para {i} " + "w" * 1000 for i in range(8))
        messages, needs_file = format_run_messages(make_record(result=result, denied_tools=("Bash",)))
        self.assertEqual(len(messages), 3)
        self.assertFalse(needs_file)
        self.assertTrue(messages[0].startswith("✅"))
        self.assertIn("Blocked tools", messages[-1])
        self.assertNotIn("Blocked tools", messages[0])
        for message in messages:
            self.assertLessEqual(len(message), TELEGRAM_MAX_CHARS)
            assert_valid_telegram_html(self, message)

    def test_too_long_result_becomes_one_message_and_a_file_flag(self) -> None:
        """More chunks than max_messages: first chunk plus the attachment note."""
        result = "\n\n".join(f"para {i} " + "w" * 1500 for i in range(30))
        record = make_record(result=result, denied_tools=("Bash",))
        messages, needs_file = format_run_messages(record)
        self.assertTrue(needs_file)
        self.assertEqual(len(messages), 1)
        self.assertIn(ATTACHED_NOTE, messages[0])
        self.assertIn("Blocked tools", messages[0])
        self.assertLessEqual(len(messages[0]), TELEGRAM_MAX_CHARS)
        assert_valid_telegram_html(self, messages[0])

    def test_max_messages_is_respected(self) -> None:
        """A smaller max_messages switches to the file earlier."""
        result = "\n\n".join(f"para {i} " + "w" * 1500 for i in range(4))
        _, needs_file = format_run_messages(make_record(result=result), max_messages=1)
        self.assertTrue(needs_file)
        with self.assertRaises(ValueError):
            format_run_messages(make_record(), max_messages=0)

    def test_worst_case_header_and_notice_still_fit(self) -> None:
        """A max-size error, many blocked tools and a huge result never exceed the cap."""
        record = make_record(
            status="error",
            error="&" * 5000,
            result="&" * 20_000,
            denied_tools=tuple(f"mcp__server__tool_{i}&" for i in range(100)),
        )
        for limit in (1, 2, 3):
            messages, _ = format_run_messages(record, max_messages=limit)
            for message in messages:
                self.assertLessEqual(len(message), TELEGRAM_MAX_CHARS)
                assert_valid_telegram_html(self, message)

    def test_chunk_size_default_is_used_for_plain_results(self) -> None:
        """Without a header or notice overhead, chunks use the full default size."""
        record = make_record(task_name="ask", trigger="chat", result="x" * 3800)
        messages, _ = format_run_messages(record)
        self.assertEqual(messages, ["x" * 3800])
        self.assertLessEqual(3800, CHUNK_CHARS)


class ResultDocumentTests(unittest.TestCase):
    """result_document builds the .md attachment."""

    def test_filename_and_content(self) -> None:
        """Name has task, run id and finish time; content has title, metadata and result."""
        record = make_record(
            id=12,
            result="# Heading\nBody é",
            duration_ms=133_000,
            finished_at=datetime(2026, 11, 2, 7, 45, tzinfo=TZ),
        )
        name, content = result_document(record)
        self.assertEqual(name, "news-run12-20261102-0745.md")
        self.assertEqual(
            content.decode("utf-8"),
            "# news\n\nRun #12 · success · 2m 13s · 2026-11-02 07:45\n\n# Heading\nBody é\n",
        )

    def test_filename_falls_back_to_queued_time_and_sanitises_task(self) -> None:
        """Without finish or start time the queue time is used; odd names are cleaned."""
        record = make_record(id=3, task_name="we ird/name", result="r")
        name, _ = result_document(record)
        self.assertEqual(name, "we-ird-name-run3-20261102-0730.md")

    def test_empty_result_uses_error_then_placeholder(self) -> None:
        """The file never ends up without a body."""
        _, with_error = result_document(make_record(result="", error="Boom"))
        self.assertIn("Boom", with_error.decode())
        _, nothing = result_document(make_record(result=""))
        self.assertIn("(no text output)", nothing.decode())

    def test_chat_documents_are_titled_chat(self) -> None:
        """Ask-name runs use their display title but a safe file name."""
        name, content = result_document(make_record(task_name="ask", trigger="chat"))
        self.assertTrue(name.startswith("ask-run1-"))
        self.assertTrue(content.decode().startswith("# Chat\n"))

    def test_local_tz_converts_fixed_offsets(self) -> None:
        """A UTC-offset time is shown in the zone passed by the caller."""
        record = make_record(finished_at=datetime(2026, 11, 2, 5, 30, tzinfo=timezone.utc))
        name, content = result_document(record, local_tz=TZ)
        self.assertEqual(name, "news-run1-20261102-0630.md")
        self.assertIn("2026-11-02 06:30", content.decode())


class TaskListTests(unittest.TestCase):
    """format_task_list shows one block per task."""

    def test_no_tasks_gives_a_hint(self) -> None:
        """An empty list points the user at tasks.toml."""
        text = format_task_list([], {}, {}, {}, NOW)
        self.assertIn("tasks.toml", text)
        assert_valid_telegram_html(self, text)

    def test_task_blocks(self) -> None:
        """Schedules, next time and the last run are shown per task."""
        news = make_task("news", schedules=(StubSchedule("daily at 07:30"), StubSchedule("Mon at 09:00")))
        last = make_record(
            status="error", finished_at=datetime(2026, 11, 2, 7, 31, tzinfo=timezone.utc)
        )
        text = format_task_list(
            [news],
            {"news": datetime(2026, 11, 3, 7, 30, tzinfo=TZ)},
            {"news": last},
            {},
            NOW,
        )
        self.assertEqual(
            text, "<b>news</b> · daily at 07:30; Mon at 09:00\nnext Tue 07:30 · last ❌ 08:31"
        )
        assert_valid_telegram_html(self, text)

    def test_task_description_is_shown_escaped_under_the_name(self) -> None:
        """A non-empty description is its own escaped line before the next/last line."""
        task = make_task("news", schedules=(StubSchedule("hourly"),), description="Digest <AI> & more")
        text = format_task_list([task], {}, {}, {}, NOW)
        self.assertEqual(
            text, "<b>news</b> · hourly\nDigest &lt;AI&gt; &amp; more\nlast never"
        )
        assert_valid_telegram_html(self, text)

    def test_manual_paused_and_disabled_markers(self) -> None:
        """Manual-only, paused and off tasks are labelled; never-run tasks say so."""
        manual = make_task("manual")
        paused = make_task("paused", schedules=(StubSchedule("hourly"),))
        off = make_task("off", schedules=(StubSchedule("hourly"),), enabled=False)
        text = format_task_list(
            [manual, paused, off], {}, {}, {"paused": True}, NOW
        )
        self.assertEqual(
            text,
            "<b>manual</b> · manual only\nlast never\n\n"
            "<b>paused</b> · hourly · ⏸ paused\nlast never\n\n"
            "<b>off</b> · hourly · off\nlast never",
        )

    def test_names_and_descriptions_are_escaped(self) -> None:
        """Schedule text with markup characters is escaped."""
        task = make_task("a", schedules=(StubSchedule("cron: */5 <x> & y"),))
        text = format_task_list([task], {}, {}, {}, NOW)
        self.assertIn("cron: */5 &lt;x&gt; &amp; y", text)
        assert_valid_telegram_html(self, text)


class StatusTests(unittest.TestCase):
    """format_status summarises the queue."""

    def test_idle_with_upcoming_runs(self) -> None:
        """Idle state lists the next 3 runs, soonest first, skipping missing ones."""
        times = {
            "d": datetime(2026, 11, 5, 8, 0, tzinfo=TZ),
            "a": datetime(2026, 11, 2, 15, 0, tzinfo=TZ),
            "none": None,
            "c": datetime(2026, 11, 4, 8, 0, tzinfo=TZ),
            "b": datetime(2026, 11, 3, 8, 0, tzinfo=TZ),
        }
        text = format_status(QueueSnapshot((), (), None), times, NOW)
        self.assertEqual(
            text,
            "💤 Nothing is running or queued.\n\n"
            "<b>Next runs</b>\n• <b>a</b> · 15:00\n• <b>b</b> · Tue 08:00\n• <b>c</b> · Wed 08:00",
        )
        assert_valid_telegram_html(self, text)

    def test_running_queued_and_limit(self) -> None:
        """Running runs show their start in local time; queue and limit lines appear."""
        running = make_record(
            id=1, status="running", started_at=datetime(2026, 11, 2, 10, 15, tzinfo=timezone.utc)
        )
        queued = tuple(
            make_record(id=i, task_name=name, status="queued")
            for i, name in ((2, "b&c"), (3, "ask"))
        )
        until = datetime(2026, 11, 2, 15, 5, tzinfo=TZ)
        text = format_status(QueueSnapshot((running,), queued, until), {}, NOW)
        self.assertEqual(
            text,
            "▶️ <b>news</b> · started 11:15\n"
            "🕐 2 queued: b&amp;c, Answer\n"
            "⏳ Claude usage limit reached. Runs wait until 15:05.\n\n"
            "No upcoming scheduled runs.",
        )
        assert_valid_telegram_html(self, text)

    def test_long_queue_is_abbreviated(self) -> None:
        """Only the first titles are listed, with a count of the rest."""
        queued = tuple(make_record(id=i, task_name=f"t{i}", status="queued") for i in range(8))
        text = format_status(QueueSnapshot((), queued, None), {}, NOW)
        self.assertIn("8 queued: t0, t1, t2, t3, t4 and 3 more", text)

    def test_next_runs_converted_from_utc(self) -> None:
        """Upcoming times given in UTC display in the local zone."""
        times = {"n": NOW.astimezone(timezone.utc) + timedelta(hours=1)}
        text = format_status(QueueSnapshot((), (), None), times, NOW)
        self.assertIn("• <b>n</b> · 13:00", text)


class HelpTests(unittest.TestCase):
    """help_text explains chat and lists every command."""

    def test_help_is_valid_and_complete(self) -> None:
        """Every command appears, the chat explanation is present, HTML is valid."""
        text = help_text()
        assert_valid_telegram_html(self, text)
        for name, _ in BOT_COMMANDS:
            self.assertIn(f"/{name} ·", text)
        self.assertIn("/run · Run a task now: /run &lt;task&gt;", text)
        self.assertIn("Just type", text)
        self.assertIn("/new", text)
        self.assertIn(f"{CHAT_IDLE_HOURS} quiet hours", text)
        self.assertLess(len(text), TELEGRAM_MAX_CHARS)

    def test_idle_hours_come_from_the_shared_constant(self) -> None:
        """The reset time in the help text follows models.CHAT_IDLE_HOURS."""
        with mock.patch("autopilot.formatting.CHAT_IDLE_HOURS", 9):
            self.assertIn("after 9 quiet hours", help_text())
        with mock.patch("autopilot.formatting.CHAT_IDLE_HOURS", 1):
            self.assertIn("after 1 quiet hour.", help_text())

    def test_commands_come_from_the_shared_list(self) -> None:
        """A command added to models.BOT_COMMANDS shows up in the help text."""
        extra = (*BOT_COMMANDS, ("zap", "Do <zap>"))
        with mock.patch("autopilot.formatting.BOT_COMMANDS", extra):
            self.assertIn("/zap · Do &lt;zap&gt;", help_text())


if __name__ == "__main__":
    unittest.main()
