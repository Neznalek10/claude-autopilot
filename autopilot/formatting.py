"""Text formatting for the Telegram channel: Markdown to Telegram HTML, splitting and message texts.

Responsibility: everything that turns Claude's Markdown answers and the
service's state (runs, tasks, queue) into text a phone can show. This module
is pure: it does no I/O, never reads the clock and imports no other autopilot
module except ``models`` (constants and dataclasses).

Where it fits: ``bot.py`` (TelegramNotifier and TelegramBot) calls the
``format_*`` functions and ``split_for_telegram``; ``html_to_plain`` is the
fallback when Telegram rejects HTML.

Invariants:
    * Output HTML uses only ``b``, ``i``, ``s``, ``code``, ``pre`` and ``a href``.
    * Every chunk returned by ``split_for_telegram`` and every message returned
      by ``format_run_messages`` is valid HTML on its own (balanced, properly
      nested tags, ``& < >`` escaped) and at most ``TELEGRAM_MAX_CHARS`` long.
    * Every time shown to the user is converted to the timezone of the
      ``now`` argument first, because the store returns fixed UTC offsets.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, tzinfo
from html import unescape

from autopilot.models import (
    ASK_TASK_NAME,
    BOT_COMMANDS,
    CHAT_IDLE_HOURS,
    MILLISECONDS_PER_SECOND,
    MINUTES_PER_HOUR,
    SECONDS_PER_MINUTE,
    STATUS_SUCCESS,
    STATUS_TIMEOUT,
    TRIGGER_CHAT,
    QueueSnapshot,
    RunRecord,
    Task,
)

TELEGRAM_MAX_CHARS = 4096
CHUNK_CHARS = 3900
MAX_RESULT_MESSAGES = 3
# Smallest chunk size that still fits a <pre> wrapper and one escaped entity.
MIN_CHUNK_CHARS = 32

STATUS_ICONS = {
    "success": "✅",
    "error": "❌",
    "timeout": "⏱",
    "rate_limited": "⏳",
    "cancelled": "🚫",
    "running": "▶️",
    "queued": "🕐",
}
UNKNOWN_STATUS_ICON = "•"

BULLET = "• "
HORIZONTAL_RULE = "──────"
BLOCK_SEPARATOR = "\n\n"
PRE_OPEN = "<pre>"
PRE_CLOSE = "</pre>"
ELLIPSIS = "…"

# Longest error text (in HTML characters) shown in a run header; keeps headers small.
ERROR_LINE_CHARS = 500
# Longest blocked-tools list (in HTML characters) shown in the warning.
BLOCKED_TOOLS_CHARS = 400
UPCOMING_RUNS_SHOWN = 3
QUEUED_TITLES_SHOWN = 5
RECENT_DAYS = 6
# Separators added around a result chunk in a message: after the header, before the
# attachment note and before the blocked-tools notice.
RESERVED_SEPARATORS = 3

EMPTY_RESULT_HTML = "<i>(no text output)</i>"
ATTACHED_NOTE = "<i>Full result attached as a file.</i>"
NO_TASKS_HINT = (
    "No tasks are defined yet. Add some to <code>tasks.toml</code> "
    "(<code>autopilot init</code> creates an example)."
)

WEEKDAY_ABBREVIATIONS = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
DOCUMENT_TIME_FORMAT = "%Y%m%d-%H%M"
METADATA_TIME_FORMAT = "%Y-%m-%d %H:%M"

# --- Markdown block and line patterns (applied to HTML-escaped text) -------------------

# A line that opens a code fence (``` with an optional language) and one that closes it.
_FENCE_OPEN_RE = re.compile(r"^\s*```")
_FENCE_CLOSE_RE = re.compile(r"^\s*```\s*$")
# A horizontal rule: three or more dashes or asterisks alone on a line.
_RULE_RE = re.compile(r"^\s*(?:-{3,}|\*{3,})\s*$")
# A Markdown heading: up to 3 spaces, 1-6 hashes, a space, the text, optional closing hashes.
_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+(.+?)(?:\s+#+)?\s*$")
# A bullet item: indent, one of - * +, whitespace, the text.
_BULLET_RE = re.compile(r"^(\s*)[-*+]\s+(.*)$")

# --- Inline patterns -------------------------------------------------------------------

# `code` on one line. The content may not hold a placeholder (NUL): leftover lone
# backticks on both sides of finished HTML must not pair up and nest it in <code>.
_CODE_SPAN_RE = re.compile(r"`([^`\n\x00]+)`")
# [label](http(s)://url); the URL may contain one level of balanced parentheses.
_LINK_RE = re.compile(
    r"\[([^\]\n]+)\]\((https?://(?:[^\s()]|\([^\s()]*\))+)\)", re.IGNORECASE
)
# **bold** anywhere; __bold__ only at word boundaries. Content must not start or end with space.
_BOLD_STAR_RE = re.compile(r"\*\*(?=\S)(.+?)(?<=\S)\*\*")
_BOLD_UNDERSCORE_RE = re.compile(r"(?<!\w)__(?=\S)(.+?)(?<=\S)__(?!\w)")
# ~~strikethrough~~.
_STRIKE_RE = re.compile(r"~~(?=\S)(.+?)(?<=\S)~~")
# *italic* and _italic_ only at word boundaries, so snake_case and 2*3*4 stay literal.
_ITALIC_STAR_RE = re.compile(r"(?<![\w*])\*(?=[^\s*])(.+?)(?<=[^\s*])\*(?![\w*])")
_ITALIC_UNDERSCORE_RE = re.compile(r"(?<![\w_])_(?=[^\s_])(.+?)(?<=[^\s_])_(?![\w_])")
# Placeholder for already-rendered inline HTML: NUL, slot number, NUL.
_PLACEHOLDER_RE = re.compile(r"\x00(\d+)\x00")
# An <a href="..."> element, for the plain-text fallback.
_ANCHOR_RE = re.compile(r'<a href="([^"]*)">(.*?)</a>', re.DOTALL)
# Any remaining tag, for the plain-text fallback.
_TAG_RE = re.compile(r"<[^>]+>")
# Characters that may not appear in a downloaded file name.
_UNSAFE_FILENAME_RE = re.compile(r"[^A-Za-z0-9._-]")

_KIND_TEXT = "text"
_KIND_CODE = "code"
_KIND_TABLE = "table"


@dataclass
class _Block:
    """A run of Markdown lines that is rendered and packed as one unit.

    Attributes:
        kind: ``text`` (a paragraph), ``code`` (fenced block) or ``table``.
        lines: The raw Markdown lines, without line breaks.
    """

    kind: str
    lines: list[str] = field(default_factory=list)


# --- Public text helpers ---------------------------------------------------------------


def escape_html(text: str, *, quote: bool = False) -> str:
    """Escape the characters Telegram's HTML parser treats as markup.

    Args:
        text: Raw text.
        quote: Also escape double quotes, for use inside an attribute value.

    Returns:
        The escaped text.

    Example:
        >>> escape_html("a < b & c")
        'a &lt; b &amp; c'
    """
    escaped = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return escaped.replace('"', "&quot;") if quote else escaped


def markdown_to_telegram_html(markdown: str) -> str:
    """Convert Markdown to the HTML subset Telegram accepts.

    Supported: fenced code, tables (kept as monospace), headings, bullets,
    horizontal rules, inline code, links, bold, italic and strikethrough.
    Anything else stays literal text. The result is not length-limited; use
    ``split_for_telegram`` for sending.

    Args:
        markdown: Markdown text, any line endings.

    Returns:
        HTML built only from ``b i s code pre a``; empty for empty input.

    Example:
        >>> markdown_to_telegram_html("# Hi\\n- **a** and `b`")
        '<b>Hi</b>\\n• <b>a</b> and <code>b</code>'
    """
    rendered = (_render_block(block) for block in _parse_blocks(markdown))
    return BLOCK_SEPARATOR.join(part for part in rendered if part)


def split_for_telegram(markdown: str, limit: int = CHUNK_CHARS) -> list[str]:
    """Convert Markdown to HTML chunks that each fit in one Telegram message.

    Blocks (paragraphs, code fences, tables) are converted whole and packed
    greedily into chunks of at most ``limit`` HTML characters. A code block or
    table that is too big is split by lines into several ``<pre>`` blocks. A
    single line that is too big is escaped without inline formatting and cut
    at the limit, never inside an entity.

    Args:
        markdown: Markdown text.
        limit: Maximum chunk length in HTML characters (at least 32).

    Returns:
        Valid, self-contained HTML chunks; an empty list for empty input.

    Raises:
        ValueError: If ``limit`` is below ``MIN_CHUNK_CHARS``.

    Example:
        >>> split_for_telegram("one\\n\\ntwo", limit=100)
        ['one\\n\\ntwo']
    """
    if limit < MIN_CHUNK_CHARS:
        raise ValueError(f"limit must be at least {MIN_CHUNK_CHARS}, got {limit}")
    pieces: list[str] = []
    for block in _parse_blocks(markdown):
        pieces.extend(_block_pieces(block, limit))
    return _pack(pieces, limit)


def html_to_plain(html: str) -> str:
    """Strip Telegram HTML down to plain text for the parse-error fallback.

    Links become ``label (url)``; other tags are removed; entities are decoded.

    Args:
        html: HTML produced by this module.

    Returns:
        Plain text that is safe to send with no parse mode.

    Example:
        >>> html_to_plain('<b>Hi</b> <a href="https://x.org">site</a> &amp; more')
        'Hi site (https://x.org) & more'
    """

    def anchor_to_text(match: re.Match[str]) -> str:
        """Render one anchor as ``label (url)``, or just the URL when they match."""
        url, label = match.group(1), match.group(2)
        return url if url == label else f"{label} ({url})"

    # Tags go first, while literal "<" is still the entity "&lt;"; decode entities last.
    without_anchors = _ANCHOR_RE.sub(anchor_to_text, html)
    return unescape(_TAG_RE.sub("", without_anchors))


def format_duration(ms: int | None) -> str:
    """Format a duration for humans.

    Args:
        ms: Milliseconds, or None when unknown. Negative values count as 0.

    Returns:
        "?" for None, then "850 ms", "12s", "2m 13s" or "1h 02m".

    Example:
        >>> format_duration(133_000)
        '2m 13s'
    """
    if ms is None:
        return "?"
    ms = max(ms, 0)
    if ms < MILLISECONDS_PER_SECOND:
        return f"{ms} ms"
    seconds = ms // MILLISECONDS_PER_SECOND
    if seconds < SECONDS_PER_MINUTE:
        return f"{seconds}s"
    minutes, seconds = divmod(seconds, SECONDS_PER_MINUTE)
    if minutes < MINUTES_PER_HOUR:
        return f"{minutes}m {seconds:02d}s"
    hours, minutes = divmod(minutes, MINUTES_PER_HOUR)
    return f"{hours}h {minutes:02d}m"


def format_time(moment: datetime | None, now: datetime) -> str:
    """Format a moment relative to ``now``, in the timezone of ``now``.

    Args:
        moment: Aware datetime in any timezone or offset, or None.
        now: Aware current time; its timezone decides the wall-clock shown.

    Returns:
        "never" for None; "07:30" on the same day as ``now``; "Tue 07:30" within
        6 days; otherwise "2026-11-14 07:30".

    Raises:
        ValueError: If either datetime is naive.

    Example:
        With ``now`` in Europe/Berlin (+01:00), 05:30 UTC on the same day
        reads "06:30".
    """
    if moment is None:
        return "never"
    local = _to_local(moment, now.tzinfo, now)
    day_gap = abs((local.date() - now.date()).days)
    if day_gap == 0:
        return local.strftime("%H:%M")
    if day_gap <= RECENT_DAYS:
        return f"{WEEKDAY_ABBREVIATIONS[local.weekday()]} {local:%H:%M}"
    return local.strftime(METADATA_TIME_FORMAT)


def run_title(record: RunRecord) -> str:
    """Name a run for headers and lists.

    Args:
        record: The run.

    Returns:
        "Chat" for chat runs, "Answer" for other ask-name runs, else the task name.
    """
    if record.trigger == TRIGGER_CHAT:
        return "Chat"
    if record.task_name == ASK_TASK_NAME:
        return "Answer"
    return record.task_name


def run_header(record: RunRecord) -> str:
    """Build the header line(s) above a run's result.

    Args:
        record: The run.

    Returns:
        ``<icon> <b>title</b> · <duration>``, with `` · retry`` for retries and
        ``timed out after <duration>`` for timeouts. A failed run adds a second
        line with its escaped error in ``<i>`` (cut to ``ERROR_LINE_CHARS``).

    Example:
        A successful run of task "news" that took 12 seconds gives
        ``'✅ <b>news</b> · 12s'``.
    """
    icon = STATUS_ICONS.get(record.status, UNKNOWN_STATUS_ICON)
    duration = format_duration(record.duration_ms)
    timing = f"timed out after {duration}" if record.status == STATUS_TIMEOUT else duration
    parts = [f"{icon} <b>{escape_html(run_title(record))}</b>", timing]
    if record.retry_of is not None:
        parts.append("retry")
    header = " · ".join(parts)
    # A timeout's error text only repeats the header, so it gets no extra line.
    if record.status not in (STATUS_SUCCESS, STATUS_TIMEOUT) and record.error:
        header += f"\n<i>{_shorten_escaped(escape_html(record.error), ERROR_LINE_CHARS)}</i>"
    return header


def format_run_messages(
    record: RunRecord, *, max_messages: int = MAX_RESULT_MESSAGES
) -> tuple[list[str], bool]:
    """Build the Telegram messages that report a finished run.

    The first message is the header, a blank line and the first result chunk.
    Successful chat runs have no header. Blocked tools add a warning to the
    last message. When the result needs more than ``max_messages`` chunks,
    only the first chunk is returned with a note, and the flag is True so the
    caller attaches ``result_document``.

    Args:
        record: The finished run.
        max_messages: Most messages to return (at least 1).

    Returns:
        ``(messages, needs_file)``. Every message is valid HTML of at most
        ``TELEGRAM_MAX_CHARS`` characters.

    Raises:
        ValueError: If ``max_messages`` is below 1.
    """
    if max_messages < 1:
        raise ValueError(f"max_messages must be at least 1, got {max_messages}")
    is_plain_answer = record.status == STATUS_SUCCESS and record.trigger == TRIGGER_CHAT
    header = "" if is_plain_answer else run_header(record)
    notice = _blocked_tools_notice(record.denied_tools)
    # Reserve room for everything added around a chunk so the sum stays under the cap:
    # the header, the attachment note, the blocked-tools notice and their separators.
    reserved = (
        len(header) + len(notice) + len(ATTACHED_NOTE)
        + RESERVED_SEPARATORS * len(BLOCK_SEPARATOR)
    )
    chunks = split_for_telegram(
        record.result, min(CHUNK_CHARS, TELEGRAM_MAX_CHARS - reserved)
    ) or [EMPTY_RESULT_HTML]

    needs_file = len(chunks) > max_messages
    if needs_file:
        chunks = [chunks[0] + BLOCK_SEPARATOR + ATTACHED_NOTE]
    if header:
        chunks[0] = header + BLOCK_SEPARATOR + chunks[0]
    if notice:
        chunks[-1] += BLOCK_SEPARATOR + notice
    return chunks, needs_file


def result_document(
    record: RunRecord, *, local_tz: tzinfo | None = None
) -> tuple[str, bytes]:
    """Build the Markdown file that carries a run's full result.

    Args:
        record: The finished run.
        local_tz: Timezone for the time in the file name and metadata line. The
            store returns fixed UTC offsets, so callers that know the user's
            zone should pass it; None keeps the record's own offset.

    Returns:
        ``(filename, content)``. The name is ``<task>-run<id>-<YYYYMMDD-HHMM>.md``;
        the content is UTF-8 Markdown with a title, a metadata line and the
        full result (or the error when the result is empty).
    """
    moment = record.finished_at or record.started_at or record.queued_at
    if local_tz is not None:
        moment = _to_local(moment, local_tz, moment)
    safe_task = _UNSAFE_FILENAME_RE.sub("-", record.task_name)
    filename = f"{safe_task}-run{record.id}-{moment.strftime(DOCUMENT_TIME_FORMAT)}.md"
    metadata = (
        f"Run #{record.id} · {record.status} · {format_duration(record.duration_ms)}"
        f" · {moment.strftime(METADATA_TIME_FORMAT)}"
    )
    body = record.result or record.error or "(no text output)"
    text = f"# {run_title(record)}\n\n{metadata}\n\n{body}\n"
    return filename, text.encode("utf-8")


def format_task_list(
    tasks: Sequence[Task],
    next_times: Mapping[str, datetime | None],
    last_runs: Mapping[str, RunRecord | None],
    paused: Mapping[str, bool],
    now: datetime,
) -> str:
    """Build the /tasks reply: one block per task.

    Each block reads ``<b>name</b> · <schedules>`` (or "manual only"), with
    "⏸ paused" or "off" appended where they apply, then the task's description
    when it has one, then a line ``next <time> · last <icon> <time>``. The next time is omitted when the
    task has none.

    Args:
        tasks: Tasks in file order. Schedules are read through ``.description``.
        next_times: Next fire time per task name; missing or None means none.
        last_runs: Latest finished run per task name; missing or None means never.
        paused: Pause flag per task name; missing means not paused.
        now: Aware current time, also the timezone for display.

    Returns:
        HTML text; a hint to edit tasks.toml when there are no tasks.
    """
    if not tasks:
        return NO_TASKS_HINT
    blocks = []
    for task in tasks:
        schedule_text = "; ".join(s.description for s in task.schedules) or "manual only"
        title = f"<b>{escape_html(task.name)}</b> · {escape_html(schedule_text)}"
        if not task.enabled:
            title += " · off"
        if paused.get(task.name):
            title += " · ⏸ paused"
        # The task's own description (from tasks.toml) sits under the name line.
        details = []
        next_time = next_times.get(task.name)
        if next_time is not None:
            details.append(f"next {format_time(next_time, now)}")
        details.append(_last_run_text(last_runs.get(task.name), now))
        lines = [title]
        if task.description:
            lines.append(escape_html(task.description))
        lines.append(" · ".join(details))
        blocks.append("\n".join(lines))
    return BLOCK_SEPARATOR.join(blocks)


def format_status(
    snapshot: QueueSnapshot,
    next_times: Mapping[str, datetime | None],
    now: datetime,
) -> str:
    """Build the /status reply.

    Args:
        snapshot: Running and queued runs plus the usage-limit gate.
        next_times: Next fire time per task name; None entries are skipped.
        now: Aware current time, also the timezone for display.

    Returns:
        HTML text with the running runs and their start times, the queue, the
        usage-limit line when the gate is closed, and the next 3 upcoming runs.
    """
    lines: list[str] = []
    for record in snapshot.running:
        started = format_time(record.started_at or record.queued_at, now)
        title = escape_html(run_title(record))
        lines.append(f"{STATUS_ICONS['running']} <b>{title}</b> · started {started}")
    if snapshot.queued:
        titles = [escape_html(run_title(r)) for r in snapshot.queued[:QUEUED_TITLES_SHOWN]]
        extra = len(snapshot.queued) - len(titles)
        listing = ", ".join(titles) + (f" and {extra} more" if extra > 0 else "")
        lines.append(f"{STATUS_ICONS['queued']} {len(snapshot.queued)} queued: {listing}")
    if not snapshot.running and not snapshot.queued:
        lines.append("💤 Nothing is running or queued.")
    if snapshot.limited_until is not None:
        lines.append(
            f"{STATUS_ICONS['rate_limited']} Claude usage limit reached. "
            f"Runs wait until {format_time(snapshot.limited_until, now)}."
        )
    lines.append("")
    lines.extend(_upcoming_lines(next_times, now))
    return "\n".join(lines)


def help_text() -> str:
    """Build the /help reply.

    Returns:
        HTML explaining plain-text chat (conversation continues, /new starts
        fresh, resets after ``CHAT_IDLE_HOURS`` quiet hours) and listing every command.
    """
    unit = "hour" if CHAT_IDLE_HOURS == 1 else "hours"
    lines = [
        "<b>Claude Autopilot</b>",
        "Just type a message to chat with Claude. The conversation continues "
        "from message to message; /new starts a fresh one, and it starts fresh "
        f"by itself after {CHAT_IDLE_HOURS} quiet {unit}.",
        "",
        "<b>Commands</b>",
    ]
    lines.extend(f"/{name} · {escape_html(text)}" for name, text in BOT_COMMANDS)
    return "\n".join(lines)


# --- Block parsing and rendering -------------------------------------------------------


def _parse_blocks(markdown: str) -> list[_Block]:
    """Split Markdown into code, table and paragraph blocks.

    Args:
        markdown: Markdown text, any line endings.

    Returns:
        Blocks in order. Blank lines end paragraphs and tables; an unclosed
        code fence runs to the end of the text.
    """
    blocks: list[_Block] = []
    current: _Block | None = None
    for line in markdown.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        if current is not None and current.kind == _KIND_CODE:
            # Inside a fence every line is code until the closing fence.
            if _FENCE_CLOSE_RE.match(line):
                current = _close(blocks, current)
            else:
                current.lines.append(line)
        elif _FENCE_OPEN_RE.match(line):
            _close(blocks, current)
            current = _Block(_KIND_CODE)
        elif not line.strip():
            current = _close(blocks, current)
        else:
            # Lines starting with "|" form a table; a change of kind starts a new block.
            kind = _KIND_TABLE if line.startswith("|") else _KIND_TEXT
            if current is None or current.kind != kind:
                _close(blocks, current)
                current = _Block(kind)
            current.lines.append(line)
    _close(blocks, current)
    return blocks


def _close(blocks: list[_Block], current: _Block | None) -> None:
    """Append the block being collected, if any.

    Args:
        blocks: Finished blocks, extended in place.
        current: The block being collected, or None.

    Returns:
        None, so callers can write ``current = _close(blocks, current)``.
    """
    if current is not None:
        blocks.append(current)
    return None


def _render_block(block: _Block) -> str:
    """Render one block to HTML.

    Args:
        block: A parsed block.

    Returns:
        HTML; empty for a code block with no content.
    """
    if block.kind == _KIND_TEXT:
        return "\n".join(_render_line(line) for line in block.lines)
    return _render_pre(block.lines)


def _render_pre(lines: Sequence[str]) -> str:
    """Render raw lines as a ``<pre>`` block, escaped but otherwise unchanged.

    Args:
        lines: Raw lines.

    Returns:
        The ``<pre>`` element, or an empty string when the lines are blank.
    """
    text = "\n".join(lines)
    if not text.strip():
        return ""
    return f"{PRE_OPEN}{escape_html(text)}{PRE_CLOSE}"


def _render_line(line: str) -> str:
    """Render one paragraph line: rule, heading, bullet or plain text.

    Args:
        line: Raw Markdown line.

    Returns:
        HTML for the line, with inline formatting applied.
    """
    # Escape first so every later rule only sees safe text; NUL is reserved for placeholders.
    escaped = escape_html(line.replace("\x00", "").rstrip())
    if _RULE_RE.match(escaped):
        return HORIZONTAL_RULE
    heading = _HEADING_RE.match(escaped)
    if heading:
        return f"<b>{_render_inline(heading.group(1), bold=False)}</b>"
    bullet = _BULLET_RE.match(escaped)
    if bullet:
        return f"{bullet.group(1)}{BULLET}{_render_inline(bullet.group(2))}"
    return _render_inline(escaped)


def _render_inline(text: str, *, bold: bool = True) -> str:
    """Apply inline formatting to one escaped line.

    Args:
        text: HTML-escaped text without line breaks.
        bold: False inside headings, where bold markers are dropped silently.

    Returns:
        HTML with ``code``, ``a``, ``b``, ``s`` and ``i`` elements.
    """
    slots: list[str] = []
    return _restore(_apply_inline_rules(text, slots, bold=bold), slots)


def _apply_inline_rules(text: str, slots: list[str], *, bold: bool) -> str:
    """Replace inline Markdown by placeholders for finished HTML.

    Each rule stores its finished HTML in ``slots`` and leaves a placeholder in
    the text. Later rules cannot reach inside or across a placeholder, which
    guarantees properly nested tags and protects code and URLs from other rules.

    Args:
        text: HTML-escaped text.
        slots: Finished HTML fragments; extended in place.
        bold: Whether ``**x**`` produces ``<b>``.

    Returns:
        Text with placeholders; pass it to ``_restore``.
    """
    text = _CODE_SPAN_RE.sub(lambda m: _stash(slots, f"<code>{m.group(1)}</code>"), text)
    text = _LINK_RE.sub(lambda m: _render_link(m, slots, bold), text)
    rules = (
        (_BOLD_STAR_RE, "b"),
        (_BOLD_UNDERSCORE_RE, "b"),
        (_STRIKE_RE, "s"),
        (_ITALIC_STAR_RE, "i"),
        (_ITALIC_UNDERSCORE_RE, "i"),
    )
    for pattern, tag in rules:
        text = pattern.sub(
            lambda m, tag=tag: _wrap(m.group(1), tag, slots, bold), text
        )
    return text


def _wrap(inner: str, tag: str, slots: list[str], bold: bool) -> str:
    """Wrap matched text in an element, formatting its content recursively.

    Args:
        inner: The text between the markers.
        tag: ``b``, ``i`` or ``s``.
        slots: Finished HTML fragments; extended in place.
        bold: False drops ``b`` wrapping (headings are already bold).

    Returns:
        A placeholder for the element, or the formatted content alone when
        bold is suppressed.
    """
    # Inner text never produces bold again, so <b> can't nest inside <b>.
    content = _apply_inline_rules(inner, slots, bold=False)
    if tag == "b" and not bold:
        return content
    return _stash(slots, f"<{tag}>{content}</{tag}>")


def _render_link(match: re.Match[str], slots: list[str], bold: bool) -> str:
    """Render ``[label](url)`` as an anchor placeholder.

    Args:
        match: A ``_LINK_RE`` match on escaped text.
        slots: Finished HTML fragments; extended in place.
        bold: Passed on to the label's formatting.

    Returns:
        A placeholder for the finished ``<a>`` element.
    """
    label = _apply_inline_rules(match.group(1), slots, bold=bold)
    # The text is already escaped, so only the attribute quote is left to handle.
    url = match.group(2).replace('"', "&quot;")
    return _stash(slots, f'<a href="{url}">{label}</a>')


def _stash(slots: list[str], fragment: str) -> str:
    """Store finished HTML and return its placeholder.

    Args:
        slots: Finished HTML fragments; extended in place.
        fragment: The HTML to protect.

    Returns:
        ``NUL<index>NUL``.
    """
    slots.append(fragment)
    return f"\x00{len(slots) - 1}\x00"


def _restore(text: str, slots: Sequence[str]) -> str:
    """Replace placeholders by their HTML, including placeholders nested inside it.

    Args:
        text: Text containing placeholders.
        slots: Finished HTML fragments, indexed by placeholder number.

    Returns:
        The final HTML.
    """
    # Fragments may contain placeholders of earlier rules, so repeat until none remain.
    while "\x00" in text:
        text = _PLACEHOLDER_RE.sub(lambda m: slots[int(m.group(1))], text)
    return text


# --- Splitting helpers -----------------------------------------------------------------


def _block_pieces(block: _Block, limit: int) -> list[str]:
    """Render a block as one or more HTML pieces that each fit in ``limit``.

    Args:
        block: A parsed block.
        limit: Maximum piece length in HTML characters.

    Returns:
        Non-empty pieces in order; empty for an empty block.
    """
    if block.kind == _KIND_TEXT:
        return _paragraph_pieces(block.lines, limit)
    whole = _render_pre(block.lines)
    if len(whole) <= limit:
        return [whole] if whole else []
    return _pre_pieces(block.lines, limit)


def _paragraph_pieces(lines: Sequence[str], limit: int) -> list[str]:
    """Pack paragraph lines into pieces, falling back to plain text for huge lines.

    Args:
        lines: Raw lines of one paragraph.
        limit: Maximum piece length in HTML characters.

    Returns:
        Pieces joined by single line breaks, each at most ``limit`` long.
    """
    rendered: list[str] = []
    for line in lines:
        # Formatting only ever lengthens a line, so a raw line over the limit is hopeless.
        html_line = _render_line(line) if len(line) <= limit else ""
        if html_line and len(html_line) <= limit:
            rendered.append(html_line)
        else:
            # Oversized line: drop the formatting and cut the escaped text at the limit.
            rendered.extend(_split_escaped(escape_html(line.rstrip()), limit))
    return _pack(rendered, limit, separator="\n")


def _pre_pieces(lines: Sequence[str], limit: int) -> list[str]:
    """Split an oversized code block or table into several ``<pre>`` blocks.

    Args:
        lines: Raw lines of the block.
        limit: Maximum piece length including the ``<pre>`` wrapper.

    Returns:
        ``<pre>`` elements, each at most ``limit`` long; over-long lines are cut.
    """
    budget = limit - len(PRE_OPEN) - len(PRE_CLOSE)
    parts: list[str] = []
    for line in lines:
        parts.extend(_split_escaped(escape_html(line), budget))
    bodies = _pack(parts, budget, separator="\n")
    return [f"{PRE_OPEN}{body}{PRE_CLOSE}" for body in bodies if body.strip()]


def _pack(pieces: Sequence[str], limit: int, *, separator: str = BLOCK_SEPARATOR) -> list[str]:
    """Join consecutive pieces greedily while the result fits in ``limit``.

    Args:
        pieces: Strings that each fit in ``limit``.
        limit: Maximum joined length.
        separator: Text placed between joined pieces.

    Returns:
        The packed strings, in order.
    """
    packed: list[str] = []
    current: list[str] = []
    size = 0
    for piece in pieces:
        joined_size = size + len(separator) + len(piece)
        if current and joined_size <= limit:
            current.append(piece)
            size = joined_size
        else:
            if current:
                packed.append(separator.join(current))
            current = [piece]
            size = len(piece)
    if current:
        packed.append(separator.join(current))
    return packed


def _cut_point(escaped: str, size: int) -> int:
    """Find where to cut escaped text so no entity is split.

    Args:
        escaped: HTML-escaped text longer than ``size``.
        size: Preferred maximum length of the head.

    Returns:
        An index at most ``size``; earlier when ``size`` falls inside ``&...;``.
    """
    ampersand = escaped.rfind("&", 0, size)
    # Every "&" in escaped text starts an entity that ends at the next ";".
    if ampersand > 0 and escaped.find(";", ampersand) >= size:
        return ampersand
    return size


def _split_escaped(escaped: str, size: int) -> list[str]:
    """Cut escaped text into parts of at most ``size`` without splitting entities.

    Args:
        escaped: HTML-escaped text.
        size: Maximum part length.

    Returns:
        At least one part; short text comes back whole.
    """
    parts = []
    while len(escaped) > size:
        cut = _cut_point(escaped, size)
        parts.append(escaped[:cut])
        escaped = escaped[cut:]
    parts.append(escaped)
    return parts


def _shorten_escaped(escaped: str, size: int) -> str:
    """Shorten escaped text to ``size`` characters plus an ellipsis.

    Args:
        escaped: HTML-escaped text.
        size: Maximum length before the ellipsis is added.

    Returns:
        The text unchanged when it fits, else its head and "…".
    """
    if len(escaped) <= size:
        return escaped
    return escaped[: _cut_point(escaped, size)] + ELLIPSIS


# --- Message building helpers ----------------------------------------------------------


def _blocked_tools_notice(denied_tools: Sequence[str]) -> str:
    """Build the warning about tools Claude was not allowed to use.

    Args:
        denied_tools: Tool names that permission checks blocked.

    Returns:
        The italic warning with a hint, or an empty string when none.
    """
    if not denied_tools:
        return ""
    tools = _shorten_escaped(escape_html(", ".join(denied_tools)), BLOCKED_TOOLS_CHARS)
    return (
        f"⚠️ <i>Blocked tools: {tools}. "
        "Add them to allowed_tools in tasks.toml if this task needs them.</i>"
    )


def _last_run_text(record: RunRecord | None, now: datetime) -> str:
    """Describe a task's latest run for the task list.

    Args:
        record: The latest finished run, or None.
        now: Aware current time.

    Returns:
        ``last <icon> <time>``, or ``last never``.
    """
    if record is None:
        return "last never"
    icon = STATUS_ICONS.get(record.status, UNKNOWN_STATUS_ICON)
    moment = record.finished_at or record.started_at or record.queued_at
    return f"last {icon} {format_time(moment, now)}"


def _upcoming_lines(next_times: Mapping[str, datetime | None], now: datetime) -> list[str]:
    """Build the "next runs" section of the status reply.

    Args:
        next_times: Next fire time per task name; None entries are skipped.
        now: Aware current time.

    Returns:
        A heading plus up to ``UPCOMING_RUNS_SHOWN`` bullet lines, soonest first,
        or a single line saying nothing is scheduled.
    """
    upcoming = sorted(
        ((when, name) for name, when in next_times.items() if when is not None),
        key=lambda item: item[0],
    )[:UPCOMING_RUNS_SHOWN]
    if not upcoming:
        return ["No upcoming scheduled runs."]
    lines = ["<b>Next runs</b>"]
    lines.extend(
        f"{BULLET}<b>{escape_html(name)}</b> · {format_time(when, now)}"
        for when, name in upcoming
    )
    return lines


def _to_local(moment: datetime, zone: tzinfo | None, now: datetime) -> datetime:
    """Convert an aware datetime to the wall-clock of another timezone.

    Args:
        moment: Aware datetime in any timezone or fixed offset.
        zone: Target timezone.
        now: Aware reference datetime, checked only to reject naive input.

    Returns:
        ``moment`` expressed in ``zone``.

    Raises:
        ValueError: If ``moment`` or ``now`` is naive.
    """
    if moment.utcoffset() is None or now.utcoffset() is None:
        raise ValueError("Expected timezone-aware datetimes, got a naive one")
    return moment.astimezone(zone)
