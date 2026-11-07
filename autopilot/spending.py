"""Money log: parse a `/spend` message and add it to a Markdown balance file.

The balance file has a ``Current balance: **<number> <CUR>**`` line (any 3-letter currency code) under a heading and one
``## <Month> <Year>`` section per month, each with a table, newest entry first. ``parse_entry``
turns free text into an ``Entry``; ``add_entry`` inserts it and updates the balance.
"""

from __future__ import annotations

import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from datetime import date
from decimal import Decimal, InvalidOperation
from typing import NamedTuple
from pathlib import Path

MAX_AMOUNT = Decimal("1000000")
MAX_WHERE_LENGTH = 60
MAX_NOTE_LENGTH = 120
MONTH_NAMES = (
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
)
BALANCE_RE = re.compile(r"^(Current balance: \*\*)(-?\d+(?:\.\d+)?)( ([A-Z]{3})\*\*)[ \t]*$", re.M)
AMOUNT_RE = re.compile(r"^[+-]?[0-9]+(?:[.,][0-9]{1,2})?$")
TABLE_SEPARATOR = "|------------|------|---------------|------------"
USAGE = (
    "Usage: /spend <amount> <where>[, <note>]\n"
    "Examples: /spend 120 Cafe, snack after school\n"
    "Income: /spend +200 allowance"
)


class SpendingError(ValueError):
    """The message or the balance file cannot be used; the text says what to fix."""


class Balance(NamedTuple):
    """The balance after an entry and the currency code used in the balance file."""

    amount: Decimal
    currency: str


@dataclass(frozen=True)
class Entry:
    """One money movement. ``amount`` is negative for spending and positive for income."""

    amount: Decimal
    where: str
    note: str


def _clean(text: str) -> str:
    """Collapse whitespace, drop control characters and the table separator so an entry stays on one row."""
    printable = "".join(char for char in text if char.isprintable() or char.isspace())
    return " ".join(printable.replace("|", " ").split())


def parse_entry(text: str) -> Entry:
    """Parse ``<amount> <where>[, <note>]`` (the amount may be anywhere in the text).

    A leading ``+`` marks income; otherwise the amount is spending. The first comma or
    ``|`` splits the place from the note.

    Raises:
        SpendingError: If there is no amount, no place, or a value is too large or long.
    """
    tokens = text.split()
    index = next((i for i, token in enumerate(tokens) if AMOUNT_RE.match(token)), None)
    if index is None:
        raise SpendingError("I could not find an amount.\n" + USAGE)
    raw = tokens.pop(index)
    amount = Decimal(raw.lstrip("+-").replace(",", "."))
    if amount <= 0 or amount > MAX_AMOUNT:
        raise SpendingError(f"The amount must be between 0 and {MAX_AMOUNT:,.0f}.")
    where, note = _split_where_note(" ".join(tokens))
    where, note = _clean(where), _clean(note)
    if not where:
        raise SpendingError("Tell me where the money went.\n" + USAGE)
    if len(where) > MAX_WHERE_LENGTH or len(note) > MAX_NOTE_LENGTH:
        raise SpendingError(f"Keep the place under {MAX_WHERE_LENGTH} and the note under {MAX_NOTE_LENGTH} characters.")
    return Entry(amount if raw.startswith("+") else -amount, where, note)


def _split_where_note(text: str) -> tuple[str, str]:
    """Split on the first comma or pipe: ``("Cafe", "snack")`` for ``"Cafe, snack"``."""
    match = re.search(r"[,|]", text)
    if match is None:
        return text, ""
    return text[: match.start()], text[match.end():]


def format_amount(value: Decimal, *, signed: bool = False) -> str:
    """Format without trailing zeros: ``-35``, ``+200``, ``12.5``."""
    text = f"{value:f}"
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return f"+{text}" if signed and value > 0 else text


def _month_heading(day: date) -> str:
    return f"## {MONTH_NAMES[day.month - 1]} {day.year}"


def _row(entry: Entry, day: date) -> str:
    return f"| {day.isoformat()} | {format_amount(entry.amount, signed=True)} | {entry.where} | {entry.note}".rstrip()


def _table_header(currency: str) -> str:
    return f"|    Date    | Amount ({currency}) | Where | Circumstances "


def _insert_row(lines: list[str], row: str, day: date, currency: str) -> list[str]:
    """Insert ``row`` at the top of the month's table, creating the section or its table if needed."""
    heading = _month_heading(day)
    if heading in lines:
        start = lines.index(heading) + 1
        end = next((i for i in range(start, len(lines)) if lines[i].startswith("## ")), len(lines))
        separator = next((i for i in range(start, end) if lines[i].startswith("|-")), None)
        if separator is not None:
            return lines[: separator + 1] + [row] + lines[separator + 1 :]
        return lines[:start] + ["", _table_header(currency), TABLE_SEPARATOR, row] + lines[start:]
    balance_at = next(i for i, line in enumerate(lines) if BALANCE_RE.match(line))
    section = ["", heading, "", _table_header(currency), TABLE_SEPARATOR, row]
    return lines[: balance_at + 1] + section + lines[balance_at + 1 :]


def add_entry(path: Path, entry: Entry, day: date) -> Balance:
    """Add ``entry`` to the balance file and return the new balance with its currency.

    The file is replaced atomically, so a crash never leaves it half written.

    Raises:
        SpendingError: If the file is missing, unreadable, or has no balance line.
    """
    target = path.resolve()
    try:
        with open(target, encoding="utf-8", newline="") as stream:
            raw = stream.read()
    except (OSError, UnicodeDecodeError) as err:
        reason = err.strerror if isinstance(err, OSError) else "it is not UTF-8 text"
        raise SpendingError(f"Cannot read the balance file {path}: {reason}") from err
    crlf = "\r\n" in raw
    text = raw.replace("\r\n", "\n")
    match = BALANCE_RE.search(text)
    if match is None:
        raise SpendingError(f"The balance file {path} has no 'Current balance: **<number> <CUR>**' line.")
    try:
        balance = Decimal(match.group(2)) + entry.amount
    except InvalidOperation as err:
        raise SpendingError(f"The balance in {path} is not a number.") from err
    updated = BALANCE_RE.sub(lambda m: f"{m.group(1)}{format_amount(balance)}{m.group(3)}", text, count=1)
    lines = _insert_row(updated.split("\n"), _row(entry, day), day, match.group(4))
    content = "\n".join(lines)
    try:
        _write_atomic(target, content.replace("\n", "\r\n") if crlf else content)
    except OSError as err:
        raise SpendingError(f"Cannot write the balance file {path}: {err.strerror}") from err
    return Balance(balance, match.group(4))


def _write_atomic(target: Path, content: str) -> None:
    """Write via a temp file in the same folder, keeping the original file mode."""
    handle, temp_name = tempfile.mkstemp(dir=target.parent, prefix=".balance-", suffix=".tmp")
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="") as stream:
            stream.write(content)
        shutil.copymode(target, temp_name)
        os.replace(temp_name, target)
    except BaseException:
        if os.path.exists(temp_name):
            os.unlink(temp_name)
        raise
