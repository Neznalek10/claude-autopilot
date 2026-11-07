"""Tests for the money log: message parsing and balance file updates."""

import os
import stat
import tempfile
import unittest
from datetime import date
from decimal import Decimal
from pathlib import Path

from autopilot.spending import Entry, SpendingError, add_entry, format_amount, parse_entry

BALANCE_FILE = """# Balance

Current balance: **221 EUR**

## October 2026

|    Date    | Amount (EUR) | Where | Circumstances 
|------------|------|---------------|------------
| 2026-10-16 | +2000 | allowance | income
| 2026-10-15 | -338 | Market        | groceries
"""


class ParseEntryTests(unittest.TestCase):
    """parse_entry understands loose text."""

    def test_amount_place_and_note(self) -> None:
        entry = parse_entry("120 Cafe, snack after school")
        self.assertEqual(entry, Entry(Decimal("-120"), "Cafe", "snack after school"))

    def test_amount_can_come_last_and_note_is_optional(self) -> None:
        self.assertEqual(parse_entry("Cafe 35"), Entry(Decimal("-35"), "Cafe", ""))

    def test_plus_is_income(self) -> None:
        self.assertEqual(parse_entry("+200 allowance | pocket money").amount, Decimal("200"))

    def test_decimal_comma(self) -> None:
        self.assertEqual(parse_entry("12,5 lunch"), Entry(Decimal("-12.5"), "lunch", ""))

    def test_pipes_cannot_break_the_table(self) -> None:
        self.assertEqual(parse_entry("5 a|b|c").note, "b c")

    def test_unicode_digits_are_not_amounts_and_control_characters_are_dropped(self) -> None:
        with self.assertRaises(SpendingError):
            parse_entry("\u0663 x")
        self.assertEqual(parse_entry("5 x\x00y").where, "xy")

    def test_rejects_missing_amount_place_and_bad_size(self) -> None:
        for text in ("Cafe", "120", "0 Cafe", "9999999 Cafe", "-0 x"):
            with self.subTest(text=text), self.assertRaises(SpendingError):
                parse_entry(text)

    def test_rejects_overlong_text(self) -> None:
        with self.assertRaises(SpendingError):
            parse_entry("5 " + "x" * 61)
        with self.assertRaises(SpendingError):
            parse_entry("5 shop, " + "n" * 121)


class FormatAmountTests(unittest.TestCase):
    def test_formats(self) -> None:
        self.assertEqual(format_amount(Decimal("-35.00")), "-35")
        self.assertEqual(format_amount(Decimal("200"), signed=True), "+200")
        self.assertEqual(format_amount(Decimal("-12.50"), signed=True), "-12.5")


class AddEntryTests(unittest.TestCase):
    """add_entry edits the balance file in place."""

    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.path = Path(tmp.name) / "balance.md"
        self.path.write_text(BALANCE_FILE, encoding="utf-8")

    def lines(self) -> list[str]:
        return self.path.read_text(encoding="utf-8").split("\n")

    def test_adds_row_at_top_of_existing_month_and_updates_balance(self) -> None:
        balance = add_entry(self.path, Entry(Decimal("-120"), "Cafe", "snack"), date(2026, 10, 17))
        self.assertEqual(balance, (Decimal("101"), "EUR"))
        lines = self.lines()
        self.assertIn("Current balance: **101 EUR**", lines)
        self.assertEqual(lines[lines.index("|------------|------|---------------|------------") + 1],
                         "| 2026-10-17 | -120 | Cafe | snack")
        self.assertIn("| 2026-10-15 | -338 | Market        | groceries", lines)

    def test_new_month_section_goes_right_after_the_balance(self) -> None:
        add_entry(self.path, Entry(Decimal("200"), "allowance", ""), date(2026, 11, 7))
        lines = self.lines()
        self.assertEqual(lines[2:5], ["Current balance: **421 EUR**", "", "## November 2026"])
        self.assertIn("| 2026-11-07 | +200 | allowance |", lines)
        self.assertLess(lines.index("## November 2026"), lines.index("## October 2026"))
        self.assertEqual(lines[lines.index("## October 2026") - 1], "")

    def test_two_entries_keep_newest_on_top(self) -> None:
        add_entry(self.path, Entry(Decimal("-10"), "a", ""), date(2026, 11, 7))
        add_entry(self.path, Entry(Decimal("-20"), "b", ""), date(2026, 11, 7))
        lines = self.lines()
        self.assertLess(lines.index("| 2026-11-07 | -20 | b |"), lines.index("| 2026-11-07 | -10 | a |"))

    def test_decimal_amounts_keep_the_balance_exact(self) -> None:
        self.assertEqual(add_entry(self.path, Entry(Decimal("-0.1"), "x", ""), date(2026, 10, 17)).amount, Decimal("220.9"))
        self.assertIn("Current balance: **220.9 EUR**", self.lines())

    def test_keeps_the_file_mode(self) -> None:
        os.chmod(self.path, 0o640)
        add_entry(self.path, Entry(Decimal("-1"), "x", ""), date(2026, 10, 17))
        self.assertEqual(stat.S_IMODE(self.path.stat().st_mode), 0o640)
        self.assertEqual([p.name for p in self.path.parent.iterdir()], ["balance.md"])

    def test_month_heading_without_a_table_gets_its_own_table(self) -> None:
        self.path.write_text(BALANCE_FILE.replace("## October 2026", "## November 2026\n\n## October 2026"), encoding="utf-8")
        add_entry(self.path, Entry(Decimal("-5"), "a", ""), date(2026, 11, 7))
        lines = self.lines()
        november = lines.index("## November 2026")
        self.assertEqual(lines[november + 1 : november + 5],
                         ["", "|    Date    | Amount (EUR) | Where | Circumstances ", "|------------|------|---------------|------------", "| 2026-11-07 | -5 | a |"])
        self.assertEqual(lines[lines.index("## October 2026") - 1], "")
        self.assertNotIn("| 2026-11-07 | -5 | a |", lines[lines.index("## October 2026"):])

    def test_month_heading_at_the_end_of_the_file(self) -> None:
        self.path.write_text("# Balance\n\nCurrent balance: **1 EUR**\n\n## November 2026", encoding="utf-8")
        add_entry(self.path, Entry(Decimal("-1"), "a", ""), date(2026, 11, 7))
        self.assertEqual(self.lines()[-3:], ["|    Date    | Amount (EUR) | Where | Circumstances ", "|------------|------|---------------|------------", "| 2026-11-07 | -1 | a |"])

    def test_crlf_file_stays_crlf(self) -> None:
        self.path.write_bytes(BALANCE_FILE.replace("\n", "\r\n").encode("utf-8"))
        add_entry(self.path, Entry(Decimal("-5"), "a", ""), date(2026, 10, 17))
        data = self.path.read_bytes()
        self.assertNotIn(b"\n", data.replace(b"\r\n", b""))
        self.assertIn(b"| 2026-10-17 | -5 | a |\r\n", data)

    def test_non_utf8_and_unwritable_folder_are_clear_errors(self) -> None:
        self.path.write_bytes(b"\xff\xfe not text")
        with self.assertRaises(SpendingError):
            add_entry(self.path, Entry(Decimal("-1"), "x", ""), date(2026, 10, 17))
        self.path.write_text(BALANCE_FILE, encoding="utf-8")
        os.chmod(self.path.parent, 0o500)
        self.addCleanup(os.chmod, self.path.parent, 0o700)
        if os.access(self.path.parent, os.W_OK):
            self.skipTest("running as a user that ignores folder permissions")
        with self.assertRaises(SpendingError):
            add_entry(self.path, Entry(Decimal("-1"), "x", ""), date(2026, 10, 17))

    def test_missing_file_and_missing_balance_line_are_errors(self) -> None:
        with self.assertRaises(SpendingError):
            add_entry(self.path.with_name("nope.md"), Entry(Decimal("-1"), "x", ""), date(2026, 10, 17))
        self.path.write_text("# Balance\n", encoding="utf-8")
        with self.assertRaises(SpendingError):
            add_entry(self.path, Entry(Decimal("-1"), "x", ""), date(2026, 10, 17))
        self.assertEqual(self.path.read_text(encoding="utf-8"), "# Balance\n")


if __name__ == "__main__":
    unittest.main()
