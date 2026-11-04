"""Unit tests for autopilot.runner and the fake Claude CLI in tests/fake_claude.py.

Covers binary discovery, command and environment building, usage-limit detection,
output parsing, and ClaudeRunner end to end against the fake CLI (including the
timeout path, which must kill the whole process group).

Everything is deterministic: temp folders, a fixed clock, no network, no real Claude.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import threading
import time
import unittest
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from pathlib import Path
from unittest import mock

from autopilot import runner
from autopilot.errors import ClaudeNotFoundError
from autopilot.models import (
    STATUS_ERROR,
    STATUS_RATE_LIMITED,
    STATUS_SUCCESS,
    STATUS_TIMEOUT,
)
from autopilot.runner import (
    MAX_ERROR_CHARS,
    RESET_BUFFER,
    ClaudeRunner,
    build_command,
    child_environment,
    detect_rate_limit,
    find_claude_binary,
    parse_output,
)
from helpers import FAKE_CLAUDE, TZ, FakeClock, make_options

NOW = datetime(2026, 11, 2, 15, 0, tzinfo=TZ)
CZECH_SENTENCE = "Příliš žluťoučký kůň"
POLL_SECONDS = 3.0
POLL_INTERVAL = 0.05


def make_executable(path: Path, content: str = "#!/bin/sh\nexit 0\n") -> Path:
    """Create an executable file (and its parent folders).

    Args:
        path: File to create.
        content: File content; a harmless shell script by default.

    Returns:
        ``path``.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    path.chmod(0o755)
    return path


def result_json(**overrides: object) -> str:
    """Build the stdout of a successful `claude -p --output-format json` run.

    Args:
        **overrides: Keys to replace or add in the result object.

    Returns:
        The object serialized as one JSON line.
    """
    payload: dict[str, object] = {
        "type": "result",
        "subtype": "success",
        "is_error": False,
        "result": "All done",
        "session_id": "sess-1",
        "num_turns": 2,
        "duration_ms": 1234,
        "total_cost_usd": 0.05,
        "permission_denials": [],
        "api_error_status": None,
    }
    payload.update(overrides)
    return json.dumps(payload)


def pids_with_cwd(directory: Path) -> list[int]:
    """List the live processes whose working directory is ``directory`` (Linux /proc).

    Args:
        directory: A unique folder that only the process under test uses as its cwd.

    Returns:
        Matching process ids. Zombies and unreadable entries are ignored.
    """
    target = os.path.realpath(directory)
    found = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            if os.readlink(entry / "cwd") == target:
                found.append(int(entry.name))
        except OSError:
            continue
    return found


class FindClaudeBinaryTests(unittest.TestCase):
    """find_claude_binary: lookup order, skipping broken candidates, helpful failure."""

    def setUp(self) -> None:
        """Create an empty fake home folder."""
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.home = Path(temp.name)

    @staticmethod
    def no_path(_name: str) -> str | None:
        """A PATH lookup that never finds anything."""
        return None

    def find(self, configured: str | None = None, which=None) -> Path:
        """Call find_claude_binary against the fake home."""
        return find_claude_binary(configured, home=self.home, which=which or self.no_path)

    def test_configured_executable_wins_over_everything(self) -> None:
        """A valid CLAUDE_BIN is returned even when other candidates exist."""
        configured = make_executable(self.home / "custom" / "claude")
        make_executable(self.home / ".local/bin/claude")
        self.assertEqual(self.find(str(configured)), configured)

    def test_configured_missing_names_the_path(self) -> None:
        """An unusable CLAUDE_BIN raises an error that names it, with no fallback."""
        make_executable(self.home / ".local/bin/claude")
        with self.assertRaises(ClaudeNotFoundError) as caught:
            self.find(str(self.home / "nope"))
        self.assertIn(str(self.home / "nope"), str(caught.exception))

    def test_configured_unknown_user_home_is_a_clean_error(self) -> None:
        """A CLAUDE_BIN like ~nosuchuser/claude gives ClaudeNotFoundError naming the value."""
        value = "~nosuchuser-autopilot/claude"
        with self.assertRaises(ClaudeNotFoundError) as caught:
            self.find(value)
        self.assertIn(value, str(caught.exception))

    def test_configured_non_executable_is_rejected(self) -> None:
        """A CLAUDE_BIN that exists but isn't executable is rejected."""
        plain = self.home / "claude"
        plain.write_text("x", encoding="utf-8")
        plain.chmod(0o644)
        with self.assertRaises(ClaudeNotFoundError):
            self.find(str(plain))

    def test_path_lookup_comes_before_home_locations(self) -> None:
        """`claude` found on PATH beats ~/.local/bin/claude."""
        on_path = make_executable(self.home / "bin" / "claude")
        make_executable(self.home / ".local/bin/claude")
        self.assertEqual(self.find(which=lambda _name: str(on_path)), on_path)

    def test_local_bin_before_claude_local(self) -> None:
        """~/.local/bin/claude is preferred over ~/.claude/local/claude."""
        local_bin = make_executable(self.home / ".local/bin/claude")
        make_executable(self.home / ".claude/local/claude")
        self.assertEqual(self.find(), local_bin)

    def test_claude_local_used_when_local_bin_missing(self) -> None:
        """~/.claude/local/claude is the next fallback."""
        legacy = make_executable(self.home / ".claude/local/claude")
        self.assertEqual(self.find(), legacy)

    def test_broken_symlink_is_skipped(self) -> None:
        """A dangling ~/.local/bin/claude symlink is skipped in favour of the next candidate."""
        link = self.home / ".local/bin/claude"
        link.parent.mkdir(parents=True)
        link.symlink_to(self.home / "missing-target")
        legacy = make_executable(self.home / ".claude/local/claude")
        self.assertEqual(self.find(), legacy)

    def test_non_executable_candidate_is_skipped(self) -> None:
        """A non-executable file at a known location is not returned."""
        plain = self.home / ".local/bin/claude"
        plain.parent.mkdir(parents=True)
        plain.write_text("x", encoding="utf-8")
        plain.chmod(0o644)
        legacy = make_executable(self.home / ".claude/local/claude")
        self.assertEqual(self.find(), legacy)

    def test_path_hit_that_is_not_executable_is_skipped(self) -> None:
        """A PATH lookup result that isn't an executable file is ignored."""
        legacy = make_executable(self.home / ".claude/local/claude")
        self.assertEqual(self.find(which=lambda _name: str(self.home / "ghost")), legacy)

    def test_vscode_extension_newest_numeric_version_wins(self) -> None:
        """Extension folders are compared by number: 2.1.10 is newer than 2.1.9."""
        suffix = "resources/native-binary/claude"
        extensions = self.home / ".vscode/extensions"
        make_executable(extensions / "anthropic.claude-code-2.1.9-linux-x64" / suffix)
        newest = make_executable(extensions / "anthropic.claude-code-2.1.10-linux-x64" / suffix)
        self.assertEqual(self.find(), newest)

    def test_vscode_newest_broken_falls_back_to_older(self) -> None:
        """If the newest extension's binary is unusable, an older working one is used."""
        suffix = "resources/native-binary/claude"
        extensions = self.home / ".vscode/extensions"
        older = make_executable(extensions / "anthropic.claude-code-2.0.1-linux-x64" / suffix)
        broken = extensions / "anthropic.claude-code-2.5.0-linux-x64" / suffix
        broken.parent.mkdir(parents=True)
        broken.symlink_to(self.home / "missing-target")
        self.assertEqual(self.find(), older)

    def test_vscode_server_used_after_vscode(self) -> None:
        """~/.vscode-server extensions are searched only after ~/.vscode ones."""
        suffix = "resources/native-binary/claude"
        server = make_executable(
            self.home / ".vscode-server/extensions/anthropic.claude-code-9.0.0-linux-x64" / suffix
        )
        self.assertEqual(self.find(), server)
        desktop = make_executable(
            self.home / ".vscode/extensions/anthropic.claude-code-1.0.0-linux-x64" / suffix
        )
        self.assertEqual(self.find(), desktop)

    def test_nothing_found_lists_places_and_fix(self) -> None:
        """The failure message lists every place checked and says how to fix it."""
        with self.assertRaises(ClaudeNotFoundError) as caught:
            self.find()
        message = str(caught.exception)
        self.assertIn("PATH", message)
        self.assertIn(str(self.home / ".local/bin/claude"), message)
        self.assertIn(str(self.home / ".claude/local/claude"), message)
        self.assertIn(".vscode/extensions", message)
        self.assertIn(".vscode-server/extensions", message)
        self.assertIn("claude install", message)
        self.assertIn("CLAUDE_BIN", message)


class BuildCommandTests(unittest.TestCase):
    """build_command: fixed flags, optional flags, ordering and the prompt-never-in-argv rule."""

    BINARY = Path("/opt/claude")

    def test_minimal_command(self) -> None:
        """Default options give exactly the fixed headless flags."""
        command = build_command(self.BINARY, make_options())
        self.assertEqual(
            command,
            [
                "/opt/claude", "-p", "--output-format", "json",
                "--permission-prompts", "none", "--permission-mode", "dontAsk",
            ],
        )  # fmt: skip

    def test_all_options_in_contract_order(self) -> None:
        """Single-value flags come first; variadic flags come last in a fixed order."""
        options = make_options(
            model="sonnet",
            effort="high",
            max_budget_usd=1.5,
            append_system_prompt="Be brief.",
            permission_mode="acceptEdits",
            add_dirs=(Path("/a"), Path("/b")),
            allowed_tools=("Read", "Bash(git log *)"),
            disallowed_tools=("WebFetch",),
        )
        command = build_command(self.BINARY, options, "sess-9")
        self.assertEqual(
            command[1:],
            [
                "-p", "--output-format", "json", "--permission-prompts", "none",
                "--permission-mode", "acceptEdits",
                "--model", "sonnet", "--effort", "high", "--max-budget-usd", "1.5",
                "--append-system-prompt", "Be brief.", "--resume", "sess-9",
                "--add-dir", "/a", "/b",
                "--allowedTools", "Read", "Bash(git log *)",
                "--disallowedTools", "WebFetch",
            ],
        )  # fmt: skip

    def test_never_uses_bare_and_omits_unset_flags(self) -> None:
        """--bare and unset optional flags never appear."""
        command = build_command(self.BINARY, make_options())
        for flag in ("--bare", "--model", "--effort", "--resume", "--add-dir", "--allowedTools"):
            self.assertNotIn(flag, command)

    def test_zero_budget_is_still_passed(self) -> None:
        """A budget of 0.0 is a value, not 'unset'."""
        command = build_command(self.BINARY, make_options(max_budget_usd=0.0))
        self.assertEqual(command[command.index("--max-budget-usd") + 1], "0.0")


class ChildEnvironmentTests(unittest.TestCase):
    """child_environment: credentials and parent-session variables must not leak."""

    def test_strips_api_credentials_and_parent_session(self) -> None:
        """API keys, CLAUDECODE and unrelated CLAUDE_* variables are removed."""
        base = {
            "PATH": "/usr/bin",
            "HOME": "/home/user",
            "ANTHROPIC_API_KEY": "secret",
            "ANTHROPIC_AUTH_TOKEN": "secret",
            "CLAUDECODE": "1",
            "CLAUDE_CODE_ENTRYPOINT": "cli",
            "CLAUDE_CODE_SSE_PORT": "123",
        }
        self.assertEqual(child_environment(base), {"PATH": "/usr/bin", "HOME": "/home/user"})

    def test_keeps_login_variables(self) -> None:
        """CLAUDE_CODE_OAUTH_TOKEN and CLAUDE_CONFIG_DIR survive."""
        base = {"CLAUDE_CODE_OAUTH_TOKEN": "tok", "CLAUDE_CONFIG_DIR": "/cfg", "CLAUDE_X": "1"}
        self.assertEqual(
            child_environment(base),
            {"CLAUDE_CODE_OAUTH_TOKEN": "tok", "CLAUDE_CONFIG_DIR": "/cfg"},
        )

    def test_strips_every_telegram_variable(self) -> None:
        """TELEGRAM_* variables (the bot token and chat id) never reach the child."""
        base = {
            "TELEGRAM_BOT_TOKEN": "123456:secret",
            "TELEGRAM_CHAT_ID": "42",
            "TELEGRAM_ANYTHING": "x",
            "TELEGRAMS": "kept: not a TELEGRAM_ variable",
            "PATH": "/usr/bin",
        }
        self.assertEqual(
            child_environment(base),
            {"TELEGRAMS": "kept: not a TELEGRAM_ variable", "PATH": "/usr/bin"},
        )

    def test_does_not_modify_the_input(self) -> None:
        """The base mapping is left untouched."""
        base = {"ANTHROPIC_API_KEY": "secret"}
        child_environment(base)
        self.assertEqual(base, {"ANTHROPIC_API_KEY": "secret"})


class DetectRateLimitTests(unittest.TestCase):
    """detect_rate_limit: phrase matching and reset-time reading."""

    def test_recognises_documented_phrases(self) -> None:
        """Each known limit phrasing is detected, case-insensitively."""
        phrases = [
            "You've hit your session limit",
            "Claude AI usage limit reached",
            "5-hour limit reached",
            "Weekly limit exceeded",
            "You have hit your limit",
            "You're out of extra usage",
            "RATE LIMIT hit",
        ]
        for phrase in phrases:
            with self.subTest(phrase=phrase):
                self.assertTrue(detect_rate_limit(phrase, NOW)[0])

    def test_ordinary_text_is_not_a_limit(self) -> None:
        """Unrelated text, including empty text and Czech, is not a limit."""
        for text in ("", "All good", CZECH_SENTENCE, "unlimited usage"):
            with self.subTest(text=text):
                self.assertEqual(detect_rate_limit(text, NOW), (False, None))

    def test_limit_without_reset_time(self) -> None:
        """A limit message with no time reports no retry_at."""
        self.assertEqual(detect_rate_limit("usage limit reached", NOW), (True, None))

    def test_epoch_reset_time_gets_buffer(self) -> None:
        """The |epoch form is used exactly, plus RESET_BUFFER."""
        epoch = int((NOW + timedelta(hours=2)).timestamp())
        limited, retry_at = detect_rate_limit(f"usage limit reached|{epoch}", NOW)
        self.assertTrue(limited)
        self.assertEqual(retry_at, NOW + timedelta(hours=2) + RESET_BUFFER)
        self.assertEqual(retry_at.utcoffset(), NOW.utcoffset())

    def test_epoch_wins_over_clock_time(self) -> None:
        """When both forms appear, the exact epoch is used."""
        epoch = int((NOW + timedelta(hours=1)).timestamp())
        text = f"You've hit your session limit · resets 11pm|{epoch}"
        self.assertEqual(detect_rate_limit(text, NOW)[1], NOW + timedelta(hours=1) + RESET_BUFFER)

    def test_clock_time_later_today(self) -> None:
        """'resets 5pm' at 15:00 means today 17:00 plus the buffer."""
        _, retry_at = detect_rate_limit("You've hit your session limit · resets 5pm", NOW)
        self.assertEqual(retry_at, datetime(2026, 11, 2, 17, 2, tzinfo=TZ))

    def test_clock_time_already_passed_means_tomorrow(self) -> None:
        """'resets 9am' at 15:00 means tomorrow 09:00 plus the buffer."""
        _, retry_at = detect_rate_limit("session limit reached, resets 9am", NOW)
        self.assertEqual(retry_at, datetime(2026, 11, 3, 9, 2, tzinfo=TZ))

    def test_clock_time_with_minutes_and_spacing(self) -> None:
        """'resets at 5:30 pm' and 'resets 5:30pm' both read as 17:30."""
        for text in ("usage limit, resets at 5:30 pm", "usage limit, resets 5:30pm"):
            with self.subTest(text=text):
                self.assertEqual(
                    detect_rate_limit(text, NOW)[1], datetime(2026, 11, 2, 17, 32, tzinfo=TZ)
                )

    def test_twelve_hour_edge_cases(self) -> None:
        """12am is midnight and 12pm is noon."""
        midnight = detect_rate_limit("usage limit resets 12am", NOW)[1]
        self.assertEqual(midnight, datetime(2026, 11, 3, 0, 2, tzinfo=TZ))
        noon = detect_rate_limit("usage limit resets 12pm", NOW.replace(hour=9))[1]
        self.assertEqual(noon, datetime(2026, 11, 2, 12, 2, tzinfo=TZ))

    def test_twenty_four_hour_clock(self) -> None:
        """Without am/pm the hour is on the 24-hour clock."""
        self.assertEqual(
            detect_rate_limit("usage limit resets 17:00", NOW)[1],
            datetime(2026, 11, 2, 17, 2, tzinfo=TZ),
        )

    def test_real_weekly_wording_with_zone_name(self) -> None:
        """The wording seen in practice, with a trailing zone name, still reads 22:20."""
        text = "You've hit your session limit · resets 10:20pm (Europe/Prague)"
        self.assertEqual(
            detect_rate_limit(text, NOW)[1], datetime(2026, 11, 2, 22, 22, tzinfo=TZ)
        )

    def test_zone_named_in_the_message_wins_over_autopilot_zone(self) -> None:
        """'resets 10:20pm (Europe/Prague)' means Prague time even when autopilot runs in New York."""
        new_york_noon = datetime(2026, 11, 3, 12, 0, tzinfo=ZoneInfo("America/New_York"))
        text = "You've hit your session limit · resets 10:20pm (Europe/Prague)"
        reset = detect_rate_limit(text, new_york_noon)[1]
        self.assertEqual(reset, datetime(2026, 11, 3, 22, 22, tzinfo=ZoneInfo("Europe/Prague")))

    def test_unknown_zone_name_falls_back_to_autopilot_zone(self) -> None:
        """A name that isn't a known zone is ignored; the time is read in the given now's zone."""
        text = "usage limit reached, resets 5pm (Mars/Olympus_Base)"
        self.assertEqual(detect_rate_limit(text, NOW)[1], datetime(2026, 11, 2, 17, 2, tzinfo=TZ))

    def test_month_and_day_before_the_time(self) -> None:
        """'resets Nov 7, 9am' and 'resets Nov 7 at 9:30am' read the named date."""
        cases = {
            "weekly limit reached, resets Nov 7, 9am": datetime(2026, 11, 7, 9, 2, tzinfo=TZ),
            "weekly limit reached, resets Nov 7 at 9:30am": datetime(2026, 11, 7, 9, 32, tzinfo=TZ),
            "weekly limit reached, resets November 7th 9am": datetime(2026, 11, 7, 9, 2, tzinfo=TZ),
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                self.assertEqual(detect_rate_limit(text, NOW), (True, expected))

    def test_weekday_before_the_time(self) -> None:
        """'resets Thu 9am' is the next Thursday 09:00 (NOW is a Monday)."""
        _, retry_at = detect_rate_limit("weekly limit reached, resets Thu 9am", NOW)
        self.assertEqual(retry_at, datetime(2026, 11, 5, 9, 2, tzinfo=TZ))

    def test_same_weekday_with_time_passed_means_next_week(self) -> None:
        """'resets Mon 9am' at Monday 15:00 is seven days ahead, still inside the 8-day cap."""
        _, retry_at = detect_rate_limit("weekly limit reached, resets Mon 9am", NOW)
        self.assertEqual(retry_at, datetime(2026, 11, 9, 9, 2, tzinfo=TZ))

    def test_same_weekday_with_time_ahead_means_today(self) -> None:
        """'resets Mon 5pm' at Monday 15:00 is today."""
        _, retry_at = detect_rate_limit("weekly limit reached, resets Mon 5pm", NOW)
        self.assertEqual(retry_at, datetime(2026, 11, 2, 17, 2, tzinfo=TZ))

    def test_named_date_beyond_eight_days_gives_no_time(self) -> None:
        """A date more than 8 days away, or one that already passed this year, gives no time."""
        for text in ("usage limit resets Nov 20, 9am", "usage limit resets Nov 1, 9am"):
            with self.subTest(text=text):
                self.assertEqual(detect_rate_limit(text, NOW), (True, None))

    def test_month_day_rolls_into_next_year(self) -> None:
        """'resets Jan 2, 9am' on 30 December means 2 January of the next year."""
        late_december = datetime(2026, 12, 30, 12, 0, tzinfo=TZ)
        _, retry_at = detect_rate_limit("weekly limit reached, resets Jan 2, 9am", late_december)
        self.assertEqual(retry_at, datetime(2027, 1, 2, 9, 2, tzinfo=TZ))

    def test_impossible_date_gives_no_time(self) -> None:
        """'resets Feb 30, 9am' is a limit without a usable time."""
        self.assertEqual(detect_rate_limit("usage limit resets Feb 30, 9am", NOW), (True, None))

    def test_invalid_clock_values_give_no_time(self) -> None:
        """Impossible hours or minutes are reported as a limit without a retry time."""
        texts = ("usage limit resets 13pm", "usage limit resets 25:00", "usage limit resets 5:75")
        for text in texts:
            with self.subTest(text=text):
                self.assertEqual(detect_rate_limit(text, NOW), (True, None))

    def test_clock_time_across_spring_forward(self) -> None:
        """Wall-clock times follow the local zone's DST change (Prague, 29 March 2026)."""
        before_change = datetime(2026, 3, 28, 23, 0, tzinfo=TZ)
        _, retry_at = detect_rate_limit("usage limit resets 5pm", before_change)
        self.assertEqual(retry_at.astimezone(TZ).hour, 17)
        self.assertEqual(retry_at.utcoffset(), timedelta(hours=2))

    def test_past_epoch_gives_no_time(self) -> None:
        """A reset epoch in the past is ignored."""
        epoch = int((NOW - timedelta(hours=1)).timestamp())
        self.assertEqual(detect_rate_limit(f"usage limit reached|{epoch}", NOW), (True, None))

    def test_far_future_epoch_gives_no_time(self) -> None:
        """A reset more than 8 days away is treated as misread."""
        epoch = int((NOW + timedelta(days=9)).timestamp())
        self.assertEqual(detect_rate_limit(f"usage limit reached|{epoch}", NOW), (True, None))

    def test_naive_now_is_rejected(self) -> None:
        """A naive 'now' raises ValueError."""
        with self.assertRaises(ValueError):
            detect_rate_limit("usage limit", datetime(2026, 11, 2, 15, 0))


class ParseOutputTests(unittest.TestCase):
    """parse_output: success, failures, rate limits and unreadable output."""

    def parse(self, stdout: str, stderr: str = "", exit_code: int = 0, duration_ms: int = 77):
        """Call parse_output with the fixed NOW."""
        return parse_output(stdout, stderr, exit_code, duration_ms, NOW)

    def test_success_maps_every_field(self) -> None:
        """A clean result becomes a success outcome carrying all reported fields."""
        stdout = result_json(
            permission_denials=[{"tool_name": "Bash"}, {"tool_name": "Edit"}, {"tool_name": "Bash"}]
        )
        outcome = self.parse(stdout)
        self.assertEqual(outcome.status, STATUS_SUCCESS)
        self.assertEqual(outcome.result, "All done")
        self.assertEqual(outcome.session_id, "sess-1")
        self.assertEqual(outcome.num_turns, 2)
        self.assertEqual(outcome.cost_usd, 0.05)
        self.assertEqual(outcome.duration_ms, 1234)
        self.assertEqual(outcome.denied_tools, ("Bash", "Edit"))
        self.assertIsNone(outcome.error)

    def test_duration_falls_back_to_argument(self) -> None:
        """Without duration_ms in the JSON, the measured duration is used."""
        payload = json.loads(result_json())
        del payload["duration_ms"]
        self.assertEqual(self.parse(json.dumps(payload)).duration_ms, 77)

    def test_unicode_result_is_preserved(self) -> None:
        """Czech text survives parsing."""
        self.assertEqual(self.parse(result_json(result=CZECH_SENTENCE)).result, CZECH_SENTENCE)

    def test_json_on_last_line_after_noise(self) -> None:
        """A warning line before the JSON object doesn't stop it from being read."""
        stdout = f"warning: something odd\n{result_json()}\n"
        self.assertEqual(self.parse(stdout).status, STATUS_SUCCESS)

    def test_missing_optional_fields_become_none(self) -> None:
        """A bare success object yields None for session, turns and cost."""
        outcome = self.parse(json.dumps({"type": "result", "subtype": "success", "result": "x"}))
        self.assertEqual(outcome.status, STATUS_SUCCESS)
        self.assertIsNone(outcome.session_id)
        self.assertIsNone(outcome.num_turns)
        self.assertIsNone(outcome.cost_usd)
        self.assertEqual(outcome.denied_tools, ())

    def test_success_text_about_limits_is_not_a_rate_limit(self) -> None:
        """A successful answer that mentions 'usage limit' stays a success."""
        outcome = self.parse(result_json(result="Explain the usage limit and rate limit."))
        self.assertEqual(outcome.status, STATUS_SUCCESS)

    def test_is_error_result_text_becomes_the_error(self) -> None:
        """A failed result keeps Claude's own message and reports status error."""
        outcome = self.parse(result_json(is_error=True, result="Something broke"), exit_code=1)
        self.assertEqual(outcome.status, STATUS_ERROR)
        self.assertEqual(outcome.error, "Something broke")

    def test_clean_json_but_nonzero_exit_is_an_error(self) -> None:
        """A success-looking object with a non-zero exit code is not trusted."""
        outcome = self.parse(result_json(), exit_code=2)
        self.assertEqual(outcome.status, STATUS_ERROR)

    def test_empty_result_uses_subtype_message(self) -> None:
        """Each known subtype maps to its own message; unknown ones are named."""
        expected = {
            "error_max_turns": "Stopped: hit the maximum number of turns.",
            "error_during_execution": "Claude Code failed during execution.",
            "error_max_budget_usd": "Stopped: reached this task's max_budget_usd.",
            "error_other": "Claude Code stopped: error_other.",
        }
        for subtype, message in expected.items():
            with self.subTest(subtype=subtype):
                outcome = self.parse(
                    result_json(is_error=True, subtype=subtype, result=""), exit_code=1
                )
                self.assertEqual(outcome.status, STATUS_ERROR)
                self.assertEqual(outcome.error, message)

    def test_error_includes_stderr_tail(self) -> None:
        """Stderr is appended to the error, cut to MAX_ERROR_CHARS from the end."""
        stderr = "x" * 5000 + "THE END"
        outcome = self.parse(
            result_json(is_error=True, result="Boom"), stderr=stderr, exit_code=1
        )
        lines = outcome.error.split("\n")
        self.assertEqual(lines[0], "Boom")
        self.assertEqual(len(lines[1]), MAX_ERROR_CHARS)
        self.assertTrue(lines[1].endswith("THE END"))

    def test_rate_limit_in_result_text(self) -> None:
        """A failed result with limit text becomes rate_limited with its reset time."""
        text = "You've hit your session limit · resets 5pm"
        outcome = self.parse(result_json(is_error=True, result=text), exit_code=1)
        self.assertEqual(outcome.status, STATUS_RATE_LIMITED)
        self.assertEqual(outcome.retry_at, datetime(2026, 11, 2, 17, 2, tzinfo=TZ))
        self.assertEqual(outcome.error, text)

    def test_rate_limit_in_stderr(self) -> None:
        """A limit notice on stderr alone is enough."""
        outcome = self.parse(
            result_json(is_error=True, result=""), stderr="usage limit reached", exit_code=1
        )
        self.assertEqual(outcome.status, STATUS_RATE_LIMITED)
        self.assertIsNone(outcome.retry_at)

    def test_rate_limit_from_api_status_429(self) -> None:
        """api_error_status 429 marks a rate limit even without matching text."""
        outcome = self.parse(
            result_json(is_error=True, result="Too many requests", api_error_status=429),
            exit_code=1,
        )
        self.assertEqual(outcome.status, STATUS_RATE_LIMITED)

    def test_no_json_nonzero_exit(self) -> None:
        """Without JSON, a non-zero exit is an error quoting the stderr tail."""
        outcome = self.parse("", stderr="kaboom", exit_code=3)
        self.assertEqual(outcome.status, STATUS_ERROR)
        self.assertEqual(outcome.error, "Claude CLI exited with code 3: kaboom")
        self.assertEqual(outcome.duration_ms, 77)

    def test_no_json_nonzero_exit_uses_stdout_when_stderr_empty(self) -> None:
        """The stdout tail is quoted when stderr has nothing."""
        outcome = self.parse("plain failure", exit_code=1)
        self.assertEqual(outcome.error, "Claude CLI exited with code 1: plain failure")

    def test_no_json_rate_limit(self) -> None:
        """A limit notice in raw output is recognised without JSON."""
        outcome = self.parse("Claude usage limit reached, resets 5pm", exit_code=1)
        self.assertEqual(outcome.status, STATUS_RATE_LIMITED)
        self.assertEqual(outcome.retry_at, datetime(2026, 11, 2, 17, 2, tzinfo=TZ))

    def test_exit_zero_with_unreadable_output(self) -> None:
        """Exit 0 with garbage output is an error that shows the garbage."""
        outcome = self.parse("not json")
        self.assertEqual(outcome.status, STATUS_ERROR)
        self.assertEqual(
            outcome.error, "Claude CLI returned output autopilot couldn't read: not json"
        )

    def test_exit_zero_empty_output(self) -> None:
        """Exit 0 with no output at all is an unreadable-output error."""
        outcome = self.parse("")
        self.assertEqual(outcome.status, STATUS_ERROR)
        self.assertTrue(outcome.error.startswith("Claude CLI returned output"))

    def test_json_of_other_type_is_unreadable(self) -> None:
        """A JSON object that is not a result (or not an object) is treated as unreadable."""
        for stdout in ('{"type": "system"}', "[1, 2]", '"text"'):
            with self.subTest(stdout=stdout):
                self.assertEqual(self.parse(stdout).status, STATUS_ERROR)

    def test_tail_is_limited(self) -> None:
        """Quoted process output never exceeds MAX_ERROR_CHARS."""
        outcome = self.parse("y" * 5000)
        self.assertEqual(len(outcome.error.split(": ", 1)[1]), MAX_ERROR_CHARS)

    def test_naive_now_is_rejected(self) -> None:
        """A naive 'now' raises ValueError."""
        with self.assertRaises(ValueError):
            parse_output(result_json(), "", 0, 1, datetime(2026, 11, 2, 15, 0))


class ClaudeRunnerTests(unittest.TestCase):
    """ClaudeRunner.run against tests/fake_claude.py."""

    def setUp(self) -> None:
        """Create a work folder, a call log and a runner wired to the fake CLI."""
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.workdir = self.root / "work"
        self.workdir.mkdir()
        self.log = self.root / "calls.jsonl"
        self.clock = FakeClock(NOW)
        self.env = {**os.environ, "FAKE_CLAUDE_LOG": str(self.log)}
        self.runner = ClaudeRunner(lambda: FAKE_CLAUDE, clock=self.clock, base_env=self.env)

    def options(self, **overrides: object):
        """RunOptions that start in the test's work folder."""
        return make_options(self.workdir, **overrides)

    def logged_calls(self) -> list[dict]:
        """Return the JSON lines the fake CLI logged."""
        lines = self.log.read_text(encoding="utf-8").splitlines()
        return [json.loads(line) for line in lines]

    def assert_no_leftover_processes(self) -> None:
        """Fail unless every process started in the work folder is gone (waits briefly)."""
        deadline = time.monotonic() + POLL_SECONDS
        while pids_with_cwd(self.workdir) and time.monotonic() < deadline:
            time.sleep(POLL_INTERVAL)
        self.assertEqual(pids_with_cwd(self.workdir), [])

    def test_fake_claude_is_executable(self) -> None:
        """The fake CLI has an exec bit and a python3 shebang."""
        self.assertTrue(os.access(FAKE_CLAUDE, os.X_OK))
        self.assertEqual(
            FAKE_CLAUDE.read_text(encoding="utf-8").splitlines()[0], "#!/usr/bin/env python3"
        )

    def test_success_sends_prompt_on_stdin(self) -> None:
        """The prompt reaches Claude on stdin, not in argv, and the answer comes back."""
        outcome = self.runner.run(f"-hello {CZECH_SENTENCE}", self.options())
        self.assertEqual(outcome.status, STATUS_SUCCESS)
        self.assertEqual(outcome.result, f"echo: -hello {CZECH_SENTENCE}")
        self.assertEqual(outcome.num_turns, 1)
        self.assertEqual(outcome.cost_usd, 0.001)
        self.assertTrue(outcome.session_id.startswith("fake-session-"))
        call = self.logged_calls()[0]
        self.assertEqual(call["prompt"], f"-hello {CZECH_SENTENCE}")
        self.assertNotIn(f"-hello {CZECH_SENTENCE}", call["argv"])
        self.assertEqual(call["argv"][0], "-p")
        self.assertEqual(os.path.realpath(call["cwd"]), os.path.realpath(self.workdir))

    def test_resume_session_is_passed_and_echoed(self) -> None:
        """--resume reaches the CLI and the same session id comes back."""
        outcome = self.runner.run("again", self.options(), resume_session="sess-42")
        self.assertEqual(outcome.session_id, "sess-42")
        argv = self.logged_calls()[0]["argv"]
        self.assertEqual(argv[argv.index("--resume") + 1], "sess-42")

    def test_api_credentials_never_reach_the_child(self) -> None:
        """ANTHROPIC_API_KEY and ANTHROPIC_AUTH_TOKEN in the parent are removed."""
        for name in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"):
            with self.subTest(variable=name):
                guarded = ClaudeRunner(
                    lambda: FAKE_CLAUDE, clock=self.clock, base_env={**self.env, name: "secret"}
                )
                guarded.run("hi", self.options())
                self.assertFalse(self.logged_calls()[-1]["has_api_key"])

    def test_default_environment_is_os_environ(self) -> None:
        """Without base_env the runner reads os.environ at run time."""
        default_runner = ClaudeRunner(lambda: FAKE_CLAUDE, clock=self.clock)
        with mock.patch.dict(os.environ, {"FAKE_CLAUDE_LOG": str(self.log)}):
            outcome = default_runner.run("hi", self.options())
        self.assertEqual(outcome.status, STATUS_SUCCESS)
        self.assertEqual(len(self.logged_calls()), 1)

    def test_denied_tools_are_reported(self) -> None:
        """permission_denials from the CLI end up in denied_tools."""
        outcome = self.runner.run("FAKE_DENIED please", self.options())
        self.assertEqual(outcome.status, STATUS_SUCCESS)
        self.assertEqual(outcome.denied_tools, ("Bash",))

    def test_usage_limit_gives_rate_limited_with_epoch_reset(self) -> None:
        """A usage-limit reply is rate_limited and retry_at is the epoch plus the buffer."""
        reset = NOW + timedelta(hours=3)
        env = {**self.env, "FAKE_RESET_EPOCH": str(int(reset.timestamp()))}
        limited_runner = ClaudeRunner(lambda: FAKE_CLAUDE, clock=self.clock, base_env=env)
        outcome = limited_runner.run("FAKE_LIMIT", self.options())
        self.assertEqual(outcome.status, STATUS_RATE_LIMITED)
        self.assertEqual(outcome.retry_at, reset + RESET_BUFFER)
        self.assertIn("session limit", outcome.error)

    def test_usage_limit_default_reset_is_an_hour_ahead(self) -> None:
        """Without FAKE_RESET_EPOCH the fake limit resets about an hour from the real now."""
        self.clock.set(datetime.now(TZ))
        outcome = self.runner.run("FAKE_LIMIT", self.options())
        self.assertEqual(outcome.status, STATUS_RATE_LIMITED)
        wait = outcome.retry_at - self.clock()
        self.assertGreater(wait, timedelta(minutes=60))
        self.assertLess(wait, timedelta(minutes=64))

    def test_error_result(self) -> None:
        """An execution error becomes an error outcome with the subtype message."""
        outcome = self.runner.run("FAKE_ERROR", self.options())
        self.assertEqual(outcome.status, STATUS_ERROR)
        self.assertEqual(outcome.error, "Claude Code failed during execution.")

    def test_garbage_output(self) -> None:
        """Unreadable output with exit 0 becomes an error that shows it."""
        outcome = self.runner.run("FAKE_GARBAGE", self.options())
        self.assertEqual(outcome.status, STATUS_ERROR)
        self.assertIn("not json", outcome.error)

    def test_duration_is_measured(self) -> None:
        """Failure outcomes without JSON duration carry the measured wall time."""
        outcome = self.runner.run("FAKE_GARBAGE", self.options())
        self.assertGreater(outcome.duration_ms, 0)

    def test_missing_binary_becomes_error_outcome(self) -> None:
        """ClaudeNotFoundError from the resolver is reported as an error outcome."""

        def missing() -> Path:
            """A resolver that always fails, like find_claude_binary on an empty machine."""
            raise ClaudeNotFoundError("Claude Code CLI not found. Checked: nowhere.")

        outcome = ClaudeRunner(missing, clock=self.clock, base_env=self.env).run(
            "hi", self.options()
        )
        self.assertEqual(outcome.status, STATUS_ERROR)
        self.assertEqual(outcome.error, "Claude Code CLI not found. Checked: nowhere.")

    def test_start_failure_becomes_error_outcome(self) -> None:
        """An OSError while starting the process is reported, not raised."""
        outcome = self.runner.run("hi", make_options(self.root / "no-such-folder"))
        self.assertEqual(outcome.status, STATUS_ERROR)
        self.assertTrue(outcome.error.startswith("Could not start the Claude CLI: "))

    def test_binary_that_cannot_execute(self) -> None:
        """A resolved path that can't be executed is reported as a start failure."""
        broken = self.root / "claude"
        broken.write_text("#!/nonexistent/interpreter\n", encoding="utf-8")
        broken.chmod(0o755)
        outcome = ClaudeRunner(lambda: broken, clock=self.clock, base_env=self.env).run(
            "hi", self.options()
        )
        self.assertEqual(outcome.status, STATUS_ERROR)
        self.assertTrue(outcome.error.startswith("Could not start the Claude CLI: "))

    def test_timeout_kills_the_process(self) -> None:
        """A run past its timeout is reported as timeout and the process is gone."""
        started = time.monotonic()
        outcome = self.runner.run("FAKE_SLEEP 30", self.options(timeout_seconds=1))
        self.assertLess(time.monotonic() - started, 15)
        self.assertEqual(outcome.status, STATUS_TIMEOUT)
        self.assertEqual(outcome.error, "Timed out after 1 seconds.")
        self.assertGreaterEqual(outcome.duration_ms, 900)
        self.assert_no_leftover_processes()

    def test_timeout_message_uses_minutes_for_whole_minutes(self) -> None:
        """Whole-minute timeouts read as minutes."""
        with mock.patch.object(subprocess.Popen, "communicate") as communicate:
            communicate.side_effect = subprocess.TimeoutExpired("claude", 120)
            outcome = self.runner.run("hi", self.options(timeout_seconds=120))
        self.assertEqual(outcome.error, "Timed out after 2 minutes.")
        self.assert_no_leftover_processes()

    def test_timeout_kills_children_that_ignore_sigterm(self) -> None:
        """A grandchild that ignores SIGTERM is still killed (SIGKILL after the grace period)."""
        stubborn = make_executable(
            self.root / "stubborn.sh",
            "#!/bin/bash\ntrap '' TERM\nsleep 60 &\nwait\n",
        )
        stubborn_runner = ClaudeRunner(lambda: stubborn, clock=self.clock, base_env=self.env)
        with mock.patch.object(runner, "KILL_GRACE_SECONDS", 1):
            outcome = stubborn_runner.run("hi", self.options(timeout_seconds=1))
        self.assertEqual(outcome.status, STATUS_TIMEOUT)
        self.assert_no_leftover_processes()

    def test_keyboard_interrupt_kills_the_group_and_propagates(self) -> None:
        """An interrupt while waiting kills the process group and is re-raised."""
        with mock.patch.object(subprocess.Popen, "communicate", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                self.runner.run("hi", self.options())
        self.assert_no_leftover_processes()

    def start_in_background(self, active_runner: ClaudeRunner, prompt: str, **option_overrides):
        """Run a prompt on a thread and wait until its process exists.

        Args:
            active_runner: The runner to use.
            prompt: Prompt text, e.g. ``FAKE_SLEEP 30``.
            **option_overrides: RunOptions fields, e.g. ``timeout_seconds=60``.

        Returns:
            ``(thread, outcomes)``; the run's outcome is appended to ``outcomes`` when it ends.
        """
        outcomes = []
        thread = threading.Thread(
            target=lambda: outcomes.append(
                active_runner.run(prompt, self.options(**option_overrides))
            )
        )
        thread.start()
        self.addCleanup(thread.join, 30)
        deadline = time.monotonic() + POLL_SECONDS
        while not pids_with_cwd(self.workdir) and time.monotonic() < deadline:
            time.sleep(POLL_INTERVAL)
        self.assertTrue(pids_with_cwd(self.workdir), "the Claude process did not start")
        return thread, outcomes

    def test_terminate_all_stops_in_flight_runs(self) -> None:
        """terminate_all kills the live process group, counts it, and the run returns."""
        thread, outcomes = self.start_in_background(
            self.runner, "FAKE_SLEEP 30", timeout_seconds=60
        )
        self.assertEqual(self.runner.terminate_all(), 1)
        thread.join(15)
        self.assertFalse(thread.is_alive())
        self.assertEqual(outcomes[0].status, STATUS_ERROR)
        self.assert_no_leftover_processes()

    def test_terminate_all_kills_children_that_ignore_sigterm(self) -> None:
        """terminate_all falls back to SIGKILL for groups that ignore SIGTERM."""
        stubborn = make_executable(
            self.root / "stubborn.sh", "#!/bin/bash\ntrap '' TERM\nsleep 60 &\nwait\n"
        )
        stubborn_runner = ClaudeRunner(lambda: stubborn, clock=self.clock, base_env=self.env)
        with mock.patch.object(runner, "KILL_GRACE_SECONDS", 1):
            thread, _ = self.start_in_background(stubborn_runner, "hi", timeout_seconds=60)
            self.assertEqual(stubborn_runner.terminate_all(), 1)
        thread.join(15)
        self.assertFalse(thread.is_alive())
        self.assert_no_leftover_processes()

    def test_terminate_all_with_nothing_running_returns_zero(self) -> None:
        """An idle runner, or one whose runs have all finished, has nothing to terminate."""
        self.assertEqual(self.runner.terminate_all(), 0)
        self.runner.run("hi", self.options())
        self.assertEqual(self.runner.terminate_all(), 0)

    def test_run_after_terminate_all_refuses_to_start(self) -> None:
        """Once terminate_all has run, run() starts nothing and says Autopilot is stopping."""
        self.runner.terminate_all()
        outcome = self.runner.run("hi", self.options())
        self.assertEqual(outcome.status, STATUS_ERROR)
        self.assertEqual(outcome.error, "Autopilot is stopping, so this run was not started.")
        self.assertFalse(self.log.exists())

    def test_start_racing_with_terminate_all_is_not_left_running(self) -> None:
        """A start that begins while terminate_all holds the lock waits, then is refused."""
        entered_start = threading.Event()
        real_start = runner._start_process

        def slow_start(*args, **kwargs):
            """Start a process after signalling that the launch lock is held."""
            entered_start.set()
            time.sleep(0.3)
            return real_start(*args, **kwargs)

        outcomes = []
        with mock.patch.object(runner, "_start_process", side_effect=slow_start):
            starter = threading.Thread(
                target=lambda: outcomes.append(
                    self.runner.run("FAKE_SLEEP 30", self.options(timeout_seconds=60))
                )
            )
            starter.start()
            self.addCleanup(starter.join, 30)
            self.assertTrue(entered_start.wait(POLL_SECONDS))
            # The launch holds the lock, so this snapshot waits for it and then sees the child.
            self.assertEqual(self.runner.terminate_all(), 1)
        starter.join(15)
        self.assertFalse(starter.is_alive())
        self.assert_no_leftover_processes()

    def test_finished_runs_are_unregistered_even_after_timeout(self) -> None:
        """A timed-out run leaves no entry behind, so terminate_all won't signal a dead group."""
        self.runner.run("FAKE_SLEEP 30", self.options(timeout_seconds=1))
        self.assertEqual(self.runner.terminate_all(), 0)

    def test_child_does_not_receive_telegram_variables(self) -> None:
        """TELEGRAM_BOT_TOKEN in the parent environment is invisible to the Claude process."""
        marker_file = self.root / "env.txt"
        script = make_executable(
            self.root / "printenv.sh",
            f"#!/bin/bash\nenv | grep -c '^TELEGRAM_' > '{marker_file}'\n"
            'echo \'{"type": "result", "subtype": "success", "result": "ok"}\'\n',
        )
        guarded = ClaudeRunner(
            lambda: script,
            clock=self.clock,
            base_env={**self.env, "TELEGRAM_BOT_TOKEN": "123456:secret"},
        )
        self.assertEqual(guarded.run("hi", self.options()).status, STATUS_SUCCESS)
        self.assertEqual(marker_file.read_text(encoding="utf-8").strip(), "0")


class FakeClaudeInfoCommandTests(unittest.TestCase):
    """The fake CLI's non-prompt commands, used by doctor tests."""

    def run_fake(self, *args: str) -> subprocess.CompletedProcess[str]:
        """Run the fake CLI with arguments and no stdin."""
        return subprocess.run(
            [str(FAKE_CLAUDE), *args],
            capture_output=True,
            text=True,
            stdin=subprocess.DEVNULL,
            timeout=30,
            check=False,
        )

    def test_version(self) -> None:
        """--version prints the fixed version string."""
        result = self.run_fake("--version")
        self.assertEqual(result.stdout.strip(), "9.9.9 (Claude Code)")
        self.assertEqual(result.returncode, 0)

    def test_auth_status(self) -> None:
        """auth status --json reports a logged-in Pro subscription."""
        result = self.run_fake("auth", "status", "--json")
        self.assertEqual(
            json.loads(result.stdout),
            {"loggedIn": True, "authMethod": "claude.ai", "subscriptionType": "pro"},
        )


if __name__ == "__main__":
    unittest.main()
