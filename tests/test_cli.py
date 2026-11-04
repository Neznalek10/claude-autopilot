"""Unit tests for autopilot.cli.

Drives ``main()`` end to end with a temp AUTOPILOT_HOME, the executable fake
Claude CLI (tests/fake_claude.py) and no Telegram settings. The OS-touching
collaborators (``run_command``, ``default_unit_dir``, ``run_service``,
``telegram_setup``, ``run_doctor``) are patched as attributes of
``autopilot.cli``, so no test calls systemctl, the network or the real Claude CLI.
"""

from __future__ import annotations

import contextlib
import dataclasses
import io
import logging
import os
import re
import signal
import sqlite3
import subprocess
import sys
import tempfile
import time
import unittest
from datetime import datetime
from pathlib import Path
from unittest import mock

from autopilot import __version__
from autopilot import cli
from autopilot.app import HANDLER_MARKER
from autopilot.doctor import CheckResult
from autopilot.errors import AutopilotError
from autopilot.models import STATUS_ERROR, TRIGGER_SCHEDULE
from autopilot.store import Store
from helpers import FAKE_CLAUDE, TZ

TASKS_TOML = """\
[[task]]
name = "news"
prompt = "Say hi"
schedule = "weekdays 08:00"

[[task]]
name = "broken"
prompt = "FAKE_ERROR"

[[task]]
name = "off-task"
prompt = "Unused"
schedule = "daily 07:30"
enabled = false
"""

SUBCOMMANDS = (
    "init", "start", "run", "ask", "list", "history", "show", "pause", "resume",
    "next", "validate", "doctor", "telegram-setup", "install-service", "uninstall-service",
)


class CliTestCase(unittest.TestCase):
    """Base class: temp home, patched environment and a ``run_cli`` helper."""

    def setUp(self) -> None:
        """Create a temp AUTOPILOT_HOME and point the environment at it."""
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.home = Path(tmp.name)
        self.tasks_file = self.home / "tasks.toml"
        # Empty strings count as unset, which hides the developer's real settings.
        env = {
            "AUTOPILOT_HOME": str(self.home),
            "AUTOPILOT_TASKS_FILE": "",
            "AUTOPILOT_DATA_DIR": "",
            "AUTOPILOT_TIMEZONE": "Europe/Prague",
            "AUTOPILOT_LOG_LEVEL": "",
            "AUTOPILOT_MAX_PARALLEL": "",
            "CLAUDE_BIN": str(FAKE_CLAUDE),
            "TELEGRAM_BOT_TOKEN": "",
            "TELEGRAM_CHAT_ID": "",
        }
        patcher = mock.patch.dict(os.environ, env, clear=False)
        patcher.start()
        self.addCleanup(patcher.stop)
        root = logging.getLogger()
        previous_level = root.level
        self.addCleanup(self._restore_logging, previous_level)

    @staticmethod
    def _restore_logging(level: int) -> None:
        """Remove the handlers configure_logging added and restore the level."""
        root = logging.getLogger()
        for handler in [h for h in root.handlers if getattr(h, HANDLER_MARKER, False)]:
            root.removeHandler(handler)
            handler.close()
        root.setLevel(level)

    def write_tasks(self, text: str = TASKS_TOML) -> None:
        """Write tasks.toml into the temp home."""
        self.tasks_file.write_text(text, encoding="utf-8")

    def run_cli(self, *argv: str) -> tuple[int, str, str]:
        """Run ``main(argv)`` and return ``(exit code, stdout, stderr)``."""
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            try:
                code = cli.main(list(argv))
            except SystemExit as exit_request:
                code = exit_request.code if isinstance(exit_request.code, int) else 1
        return code, out.getvalue(), err.getvalue()

    def store(self) -> Store:
        """Open the database the CLI uses."""
        return Store(self.home / "data" / "autopilot.db")


class ParserTests(CliTestCase):
    """Global options and help texts."""

    def test_version_prints_package_version(self) -> None:
        """--version shows the package version and exits 0."""
        code, out, _ = self.run_cli("--version")
        self.assertEqual(code, 0)
        self.assertIn(__version__, out)

    def test_missing_command_is_usage_error(self) -> None:
        """No subcommand exits 2 like any argparse error."""
        code, _, err = self.run_cli()
        self.assertEqual(code, 2)
        self.assertIn("usage:", err)

    def test_every_subcommand_help_shows_an_example(self) -> None:
        """Each subcommand's --help ends with an example that names the command."""
        for name in SUBCOMMANDS:
            with self.subTest(command=name):
                code, out, _ = self.run_cli(name, "--help")
                self.assertEqual(code, 0)
                self.assertIn("Example:", out)
                self.assertIn(f"autopilot {name}", out)

    def test_top_level_help_lists_every_subcommand(self) -> None:
        """The top-level help mentions each command and has an example."""
        _, out, _ = self.run_cli("--help")
        self.assertIn("Example:", out)
        for name in SUBCOMMANDS:
            self.assertIn(name, out)

    def test_debug_flag_accepted_after_the_subcommand(self) -> None:
        """--debug works before and after the subcommand, with the same effect."""
        for argv in (("--debug", "validate"), ("validate", "--debug")):
            with self.subTest(argv=argv):
                code, _, err = self.run_cli(*argv)
                self.assertEqual(code, 1)
                self.assertIn("Traceback", err)

    def test_debug_flag_after_a_command_with_arguments(self) -> None:
        """`run <task> --debug` parses and keeps the other arguments."""
        args = cli.build_parser().parse_args(["run", "news", "--debug"])
        self.assertTrue(cli._debug_requested(args))
        self.assertEqual(args.task, "news")
        self.assertFalse(cli._debug_requested(cli.build_parser().parse_args(["run", "news"])))

    def test_non_positive_count_is_usage_error(self) -> None:
        """-n 0 is rejected by argparse with exit 2."""
        code, _, err = self.run_cli("next", "daily 07:30", "-n", "0")
        self.assertEqual(code, 2)
        self.assertIn("at least 1", err)


class ErrorHandlingTests(CliTestCase):
    """The shared error path in main()."""

    def test_expected_error_is_one_line_without_traceback(self) -> None:
        """A missing tasks.toml prints one friendly line and exits 1."""
        code, out, err = self.run_cli("validate")
        self.assertEqual(code, 1)
        self.assertEqual(out, "")
        self.assertTrue(err.startswith("autopilot: tasks.toml not found"))
        self.assertEqual(len(err.strip().splitlines()), 1)
        self.assertNotIn("Traceback", err)

    def test_debug_flag_shows_traceback(self) -> None:
        """--debug adds the traceback after the friendly line."""
        code, _, err = self.run_cli("--debug", "validate")
        self.assertEqual(code, 1)
        self.assertIn("autopilot: tasks.toml not found", err)
        self.assertIn("Traceback", err)

    def test_keyboard_interrupt_exits_130(self) -> None:
        """Ctrl-C ends the command with exit code 130."""
        with mock.patch.object(cli, "run_doctor", side_effect=KeyboardInterrupt):
            code, _, _ = self.run_cli("doctor")
        self.assertEqual(code, 130)

    def test_database_error_is_one_friendly_line(self) -> None:
        """A sqlite3 failure names the database and exits 1 without a traceback."""
        with mock.patch.object(Store, "list_runs", side_effect=sqlite3.DatabaseError("file is not a database")):
            code, _, err = self.run_cli("history")
        self.assertEqual(code, 1)
        self.assertIn("autopilot: database error in", err)
        self.assertIn("autopilot.db", err)
        self.assertIn("file is not a database", err)
        self.assertNotIn("Traceback", err)

    def test_data_dir_that_is_a_file_is_a_friendly_error(self) -> None:
        """AUTOPILOT_DATA_DIR pointing at a file gives a one-line error."""
        blocker = self.home / "blocker"
        blocker.write_text("x", encoding="utf-8")
        self.write_tasks()
        with mock.patch.dict(os.environ, {"AUTOPILOT_DATA_DIR": str(blocker)}):
            code, _, err = self.run_cli("list")
        self.assertEqual(code, 1)
        self.assertTrue(err.startswith("autopilot:"))
        self.assertNotIn("Traceback", err)

    def test_corrupt_database_is_a_friendly_error(self) -> None:
        """A garbage autopilot.db is reported as a database error."""
        data = self.home / "data"
        data.mkdir()
        (data / "autopilot.db").write_bytes(b"this is not a sqlite file" * 100)
        code, _, err = self.run_cli("history")
        self.assertEqual(code, 1)
        self.assertIn("database error in", err)
        self.assertNotIn("Traceback", err)

    def test_broken_pipe_exits_quietly(self) -> None:
        """A reader that closes early ends the command with exit 1 and no output noise."""
        with mock.patch.object(cli, "run_doctor", side_effect=BrokenPipeError):
            code, _, err = self.run_cli("doctor")
        self.assertEqual(code, 1)
        self.assertEqual(err, "")

    def test_bad_environment_is_a_friendly_error(self) -> None:
        """An invalid setting stops the command with the settings message."""
        with mock.patch.dict(os.environ, {"AUTOPILOT_MAX_PARALLEL": "99"}):
            code, _, err = self.run_cli("list")
        self.assertEqual(code, 1)
        self.assertIn("AUTOPILOT_MAX_PARALLEL", err)


class SubprocessTests(CliTestCase):
    """Behaviour that needs a real process: signals and closed pipes."""

    def start_cli(self, *argv: str, **popen: object) -> subprocess.Popen:
        """Start ``python -m autopilot`` with the patched test environment."""
        return subprocess.Popen(
            [sys.executable, "-m", "autopilot", *argv],
            env=os.environ.copy(),
            text=True,
            **popen,  # type: ignore[arg-type]
        )

    @staticmethod
    def child_pids(parent: int) -> list[int]:
        """Return the pids whose parent is ``parent`` (read from /proc)."""
        found = []
        for entry in Path("/proc").iterdir():
            if not entry.name.isdigit():
                continue
            try:
                stat = (entry / "stat").read_text(encoding="utf-8")
            except OSError:
                continue
            # The command name sits in parentheses and may contain spaces.
            if int(stat.rsplit(")", 1)[1].split()[1]) == parent:
                found.append(int(entry.name))
        return found

    def wait_for_child(self, parent: int) -> int:
        """Wait up to 15 s for the first child of ``parent`` and return its pid."""
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            children = self.child_pids(parent)
            if children:
                return children[0]
            time.sleep(0.05)
        self.fail("autopilot never started the Claude process")

    def test_sigterm_and_sighup_end_a_run_cleanly(self) -> None:
        """Termination signals kill Claude's process group, finish the run row and exit 130."""
        for signum in (signal.SIGTERM, signal.SIGHUP):
            with self.subTest(signal=signum.name):
                self.write_tasks('[[task]]\nname = "slow"\nprompt = "FAKE_SLEEP 60"\n')
                process = self.start_cli("run", "slow", "--no-notify", stderr=subprocess.DEVNULL)
                self.addCleanup(process.kill)
                child = self.wait_for_child(process.pid)
                group = os.getpgid(child)
                process.send_signal(signum)
                self.assertEqual(process.wait(timeout=30), 130)
                deadline = time.monotonic() + 10
                while time.monotonic() < deadline:
                    try:
                        os.killpg(group, 0)
                    except ProcessLookupError:
                        break
                    time.sleep(0.05)
                else:
                    self.fail("Claude's process group is still alive")
                record = self.store().list_runs(task_name="slow")[0]
                self.assertTrue(record.is_finished, record.status)

    def test_closed_pipe_exits_quietly(self) -> None:
        """`autopilot show <id> | head -1` leaves no traceback and no interpreter warning."""
        run_id = self.store().create_run(
            task_name="news", trigger=TRIGGER_SCHEDULE, prompt="p", queued_at=datetime.now(TZ)
        )
        self.store().finish_run(
            run_id, status="success", finished_at=datetime.now(TZ), result="line\n" * 400_000
        )
        process = self.start_cli("show", str(run_id), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.addCleanup(process.kill)
        assert process.stdout is not None and process.stderr is not None
        process.stdout.readline()
        process.stdout.close()
        self.assertEqual(process.wait(timeout=30), 1)
        self.assertEqual(process.stderr.read(), "")
        process.stderr.close()


    def test_reader_gone_before_any_output_exits_quietly(self) -> None:
        """A pipe closed before the first write (`autopilot --version | true`) prints no warning."""
        for argv in (("--version",), ("--help",), ("next", "daily 07:30")):
            with self.subTest(argv=argv):
                read_end, write_end = os.pipe()
                os.close(read_end)
                process = self.start_cli(*argv, stdout=write_end, stderr=subprocess.PIPE)
                os.close(write_end)
                self.addCleanup(process.kill)
                assert process.stderr is not None
                code = process.wait(timeout=30)
                self.assertEqual(process.stderr.read(), "")
                process.stderr.close()
                self.assertIn(code, (0, 1))


class InitTests(CliTestCase):
    """`autopilot init`."""

    def test_init_creates_files_and_prints_next_steps(self) -> None:
        """A fresh home gets tasks.toml and a private .env."""
        code, out, _ = self.run_cli("init")
        self.assertEqual(code, 0)
        self.assertTrue(self.tasks_file.is_file())
        self.assertEqual((self.home / ".env").stat().st_mode & 0o777, 0o600)
        self.assertIn("Created", out)
        self.assertIn("autopilot telegram-setup", out)

    def test_init_twice_reports_nothing_to_do(self) -> None:
        """A second run does not overwrite anything."""
        self.run_cli("init")
        self.tasks_file.write_text("# mine\n", encoding="utf-8")
        code, out, _ = self.run_cli("init")
        self.assertEqual(code, 0)
        self.assertIn("Nothing to do", out)
        self.assertEqual(self.tasks_file.read_text(encoding="utf-8"), "# mine\n")


class ValidateTests(CliTestCase):
    """`autopilot validate`."""

    def test_valid_file_lists_tasks(self) -> None:
        """A valid file prints the count and one line per task."""
        self.write_tasks()
        code, out, _ = self.run_cli("validate")
        self.assertEqual(code, 0)
        self.assertIn("tasks.toml OK: 3 tasks", out)
        self.assertIn("news: weekdays at 08:00", out)
        self.assertIn("broken: manual only", out)
        self.assertIn("off-task: daily at 07:30 (disabled)", out)

    def test_invalid_file_names_task_and_field(self) -> None:
        """A bad field exits 1 and the message names the task and the field."""
        self.write_tasks('[[task]]\nname = "bad"\nprompt = "x"\nschedule = "sometimes"\n')
        code, _, err = self.run_cli("validate")
        self.assertEqual(code, 1)
        self.assertIn("bad", err)
        self.assertIn("sometimes", err)


class NextTests(CliTestCase):
    """`autopilot next`."""

    def test_next_prints_description_and_five_times(self) -> None:
        """Default count is 5 weekday times, in the "Mon 2026-11-02 07:30" shape."""
        code, out, _ = self.run_cli("next", "weekdays 08:00")
        self.assertEqual(code, 0)
        lines = out.strip().splitlines()
        self.assertEqual(lines[0], "Schedule: weekdays at 08:00")
        times = lines[1:]
        self.assertEqual(len(times), 5)
        for line in times:
            self.assertRegex(line, r"^  (Mon|Tue|Wed|Thu|Fri) \d{4}-\d{2}-\d{2} 08:00$")

    def test_next_count_option(self) -> None:
        """-n changes how many times are shown."""
        _, out, _ = self.run_cli("next", "every 15m", "-n", "3")
        self.assertEqual(len(out.strip().splitlines()), 4)

    def test_next_weekday_matches_date(self) -> None:
        """The weekday name agrees with the printed date."""
        _, out, _ = self.run_cli("next", "daily 07:30", "-n", "2")
        for line in out.strip().splitlines()[1:]:
            match = re.match(r"^  (\w{3}) (\d{4}-\d{2}-\d{2}) 07:30$", line)
            assert match is not None
            day = datetime.strptime(match.group(2), "%Y-%m-%d")
            self.assertEqual(match.group(1), ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")[day.weekday()])

    def test_next_bad_schedule_is_friendly_error(self) -> None:
        """An unparseable expression exits 1 and quotes it."""
        code, out, err = self.run_cli("next", "whenever")
        self.assertEqual(code, 1)
        self.assertEqual(out, "")
        self.assertIn("autopilot:", err)
        self.assertIn("whenever", err)

    def test_next_never_firing_schedule_prints_nothing_to_stdout(self) -> None:
        """A schedule that never fires reports the error without a half-printed preview."""
        code, out, err = self.run_cli("next", "0 0 30 2 *")
        self.assertEqual(code, 1)
        self.assertEqual(out, "")
        self.assertIn("autopilot:", err)

    def test_next_needs_no_tasks_file(self) -> None:
        """The preview works without tasks.toml."""
        self.assertFalse(self.tasks_file.exists())
        code, _, _ = self.run_cli("next", "@daily")
        self.assertEqual(code, 0)


class RunAndAskTests(CliTestCase):
    """`autopilot run` and `autopilot ask`."""

    def test_run_prints_result_and_stores_run(self) -> None:
        """A successful run prints the result on stdout and exits 0."""
        self.write_tasks()
        code, out, err = self.run_cli("run", "news")
        self.assertEqual(code, 0)
        self.assertIn("echo: Say hi", out)
        self.assertIn("Running news", err)
        record = self.store().list_runs(task_name="news")[0]
        self.assertEqual(record.status, "success")
        self.assertEqual(record.trigger, "manual")

    def test_run_failure_exits_1_with_error_on_stderr(self) -> None:
        """A failed run reports its error on stderr and exits 1."""
        self.write_tasks()
        code, out, err = self.run_cli("run", "broken")
        self.assertEqual(code, 1)
        self.assertEqual(out, "")
        self.assertIn("error", err)

    def test_run_unknown_task(self) -> None:
        """An unknown task lists the known names."""
        self.write_tasks()
        code, _, err = self.run_cli("run", "nope")
        self.assertEqual(code, 1)
        self.assertIn("Unknown task 'nope'", err)
        self.assertIn("news", err)

    def test_run_without_tasks_file(self) -> None:
        """A missing tasks.toml points the user to `autopilot init`."""
        code, _, err = self.run_cli("run", "news")
        self.assertEqual(code, 1)
        self.assertIn("autopilot init", err)

    def test_run_refuses_a_task_that_is_already_active(self) -> None:
        """A queued run of the same task blocks a second one."""
        self.write_tasks()
        self.store().create_run(
            task_name="news", trigger=TRIGGER_SCHEDULE, prompt="Say hi", queued_at=datetime.now(TZ)
        )
        code, _, err = self.run_cli("run", "news")
        self.assertEqual(code, 1)
        self.assertIn("already queued or running", err)

    def test_run_no_notify_never_builds_a_telegram_client(self) -> None:
        """--no-notify skips Telegram even when it is configured."""
        self.write_tasks()
        env = {"TELEGRAM_BOT_TOKEN": "123456789:" + "B" * 35, "TELEGRAM_CHAT_ID": "42"}
        with mock.patch.dict(os.environ, env), mock.patch(
            "autopilot.app.TelegramClient", side_effect=AssertionError("client built")
        ):
            code, _, _ = self.run_cli("run", "news", "--no-notify")
        self.assertEqual(code, 0)

    def test_ask_prints_answer(self) -> None:
        """A one-off prompt is answered on stdout and stored as an ask run."""
        self.write_tasks()
        code, out, _ = self.run_cli("ask", "Příliš žluťoučký kůň")
        self.assertEqual(code, 0)
        self.assertIn("echo: Příliš žluťoučký kůň", out)
        self.assertEqual(self.store().list_runs(task_name="ask")[0].trigger, "manual")

    def test_ask_empty_prompt_is_friendly_error(self) -> None:
        """A blank prompt exits 1 with a one-line message."""
        self.write_tasks()
        code, _, err = self.run_cli("ask", "   ")
        self.assertEqual(code, 1)
        self.assertTrue(err.startswith("autopilot:"))

    def test_ask_failure_exits_1(self) -> None:
        """A failed answer exits 1."""
        self.write_tasks()
        code, _, _ = self.run_cli("ask", "FAKE_ERROR")
        self.assertEqual(code, 1)


class ListHistoryShowTests(CliTestCase):
    """`autopilot list`, `history` and `show`."""

    def test_list_shows_columns_and_states(self) -> None:
        """Each task gets a row with its schedule, next run and state."""
        self.write_tasks()
        code, out, _ = self.run_cli("list")
        self.assertEqual(code, 0)
        lines = {line.split()[0]: line for line in out.strip().splitlines()}
        for column in ("NAME", "SCHEDULE", "NEXT", "LAST", "STATE"):
            self.assertIn(column, lines["NAME"])
        self.assertIn("weekdays at 08:00", lines["news"])
        self.assertTrue(lines["news"].endswith("on"))
        self.assertIn("manual", lines["broken"])
        self.assertTrue(lines["off-task"].endswith("off"))

    def test_list_shows_last_run_icon(self) -> None:
        """After a run the LAST column carries the status icon."""
        self.write_tasks()
        self.run_cli("run", "news")
        _, out, _ = self.run_cli("list")
        news = next(line for line in out.splitlines() if line.startswith("news"))
        self.assertIn("✅", news)

    def test_list_without_tasks(self) -> None:
        """A file with no tasks prints a hint instead of an empty table."""
        self.write_tasks("# nothing yet\n")
        code, out, _ = self.run_cli("list")
        self.assertEqual(code, 0)
        self.assertIn("No tasks defined", out)

    def test_list_missing_file(self) -> None:
        """A missing tasks.toml is an expected error."""
        code, _, err = self.run_cli("list")
        self.assertEqual(code, 1)
        self.assertIn("tasks.toml not found", err)

    def test_history_empty(self) -> None:
        """No runs yet prints a plain message."""
        code, out, _ = self.run_cli("history")
        self.assertEqual(code, 0)
        self.assertIn("No runs yet.", out)

    def test_history_lists_runs_and_filters_by_task(self) -> None:
        """Runs appear newest first; a task argument filters them."""
        self.write_tasks()
        self.run_cli("run", "news")
        self.run_cli("run", "broken")
        _, out, _ = self.run_cli("history")
        lines = out.strip().splitlines()
        self.assertIn("ID", lines[0])
        self.assertIn("broken", lines[1])
        self.assertIn("❌ error", lines[1])
        self.assertIn("news", lines[2])
        self.assertIn("echo: Say hi", lines[2])
        _, filtered, _ = self.run_cli("history", "news")
        self.assertEqual(len(filtered.strip().splitlines()), 2)
        _, limited, _ = self.run_cli("history", "-n", "1")
        self.assertEqual(len(limited.strip().splitlines()), 2)

    def test_history_truncates_long_first_line(self) -> None:
        """The summary column keeps at most 60 characters."""
        self.write_tasks('[[task]]\nname = "long"\nprompt = "' + "x" * 100 + '"\n')
        self.run_cli("run", "long")
        _, out, _ = self.run_cli("history")
        summary = out.strip().splitlines()[1].split("  ")[-1]
        self.assertEqual(len(summary), 60)
        self.assertTrue(summary.endswith("…"))

    def test_show_prints_metadata_and_result(self) -> None:
        """show prints the facts of the run and its full result."""
        self.write_tasks()
        self.run_cli("run", "news")
        run_id = self.store().list_runs()[0].id
        code, out, _ = self.run_cli("show", str(run_id))
        self.assertEqual(code, 0)
        self.assertIn(f"#{run_id}", out)
        self.assertIn("Task:", out)
        self.assertIn("✅ success", out)
        self.assertIn("Session:", out)
        self.assertIn("echo: Say hi", out)

    def test_show_failed_run_prints_error(self) -> None:
        """A failed run shows its error text instead of a result."""
        self.write_tasks()
        self.run_cli("run", "broken")
        record = self.store().list_runs()[0]
        self.assertEqual(record.status, STATUS_ERROR)
        _, out, _ = self.run_cli("show", str(record.id))
        self.assertIn(record.error or "", out)

    def test_show_labels_are_followed_by_a_space(self) -> None:
        """Even the longest label ("Blocked tools:") is separated from its value."""
        self.write_tasks()
        self.run_cli("run", "news")
        run_id = self.store().list_runs()[0].id
        record = self.store().get_run(run_id)
        assert record is not None
        blocked = dataclasses.replace(record, denied_tools=("Bash",))
        with mock.patch.object(Store, "get_run", return_value=blocked):
            _, out, _ = self.run_cli("show", str(run_id))
        self.assertIn("Blocked tools: Bash", out)

    def test_show_unknown_id(self) -> None:
        """An unknown id exits 1."""
        code, _, err = self.run_cli("show", "999")
        self.assertEqual(code, 1)
        self.assertIn("No run with id 999", err)

    def test_show_rejects_non_number(self) -> None:
        """A non-numeric id is a usage error."""
        code, _, _ = self.run_cli("show", "abc")
        self.assertEqual(code, 2)


class PauseResumeTests(CliTestCase):
    """`autopilot pause` and `resume`."""

    def test_pause_then_resume(self) -> None:
        """pause flips the stored flag and list shows it; resume undoes both."""
        self.write_tasks()
        code, out, _ = self.run_cli("pause", "news")
        self.assertEqual(code, 0)
        self.assertIn("Paused news", out)
        self.assertTrue(self.store().is_paused("news"))
        _, listing, _ = self.run_cli("list")
        self.assertTrue(next(l for l in listing.splitlines() if l.startswith("news")).endswith("paused"))
        code, out, _ = self.run_cli("resume", "news")
        self.assertEqual(code, 0)
        self.assertIn("Resumed news", out)
        self.assertFalse(self.store().is_paused("news"))

    def test_pause_unknown_task(self) -> None:
        """An unknown task is rejected and nothing is stored."""
        self.write_tasks()
        code, _, err = self.run_cli("pause", "nope")
        self.assertEqual(code, 1)
        self.assertIn("Unknown task 'nope'", err)
        self.assertFalse(self.store().is_paused("nope"))

    def test_resume_unknown_task(self) -> None:
        """resume validates the name too."""
        self.write_tasks()
        code, _, err = self.run_cli("resume", "nope")
        self.assertEqual(code, 1)
        self.assertIn("Unknown task", err)


class StartTests(CliTestCase):
    """`autopilot start` (the service loop itself is patched out)."""

    def test_start_runs_service_and_warns_without_telegram(self) -> None:
        """Without Telegram settings a warning is printed, then the service runs."""
        self.write_tasks()
        with mock.patch.object(cli, "run_service") as run_service:
            code, _, err = self.run_cli("start")
        self.assertEqual(code, 0)
        self.assertIn("Telegram is not configured", err)
        run_service.assert_called_once()
        app = run_service.call_args.args[0]
        self.assertEqual(set(app.engine.config.tasks), {"news", "broken", "off-task"})
        self.assertTrue((self.home / "data" / "autopilot.log").is_file())

    def test_start_without_tasks_file_fails_loudly(self) -> None:
        """A missing tasks.toml stops `start` before the service runs."""
        with mock.patch.object(cli, "run_service") as run_service:
            code, _, err = self.run_cli("start")
        self.assertEqual(code, 1)
        self.assertIn("tasks.toml not found", err)
        run_service.assert_not_called()

    def test_start_without_claude_fails_loudly(self) -> None:
        """A missing Claude binary stops `start` with the install hint."""
        self.write_tasks()
        with mock.patch.dict(os.environ, {"CLAUDE_BIN": str(self.home / "missing")}):
            code, _, err = self.run_cli("start")
        self.assertEqual(code, 1)
        self.assertIn("CLAUDE_BIN", err)


class DoctorTests(CliTestCase):
    """`autopilot doctor`."""

    def test_doctor_exit_codes(self) -> None:
        """Exit 1 when any check failed; warnings alone exit 0."""
        cases = (
            ([CheckResult("A", "ok", "fine"), CheckResult("B", "warn", "meh")], 0),
            ([CheckResult("A", "ok", "fine"), CheckResult("B", "fail", "broken", "fix it")], 1),
        )
        for results, expected in cases:
            with self.subTest(expected=expected), mock.patch.object(
                cli, "run_doctor", return_value=results
            ):
                code, out, _ = self.run_cli("doctor")
                self.assertEqual(code, expected)
                self.assertIn("A: fine", out)

    def test_doctor_live_flag_is_passed(self) -> None:
        """--live reaches run_doctor."""
        with mock.patch.object(cli, "run_doctor", return_value=[]) as run_doctor:
            self.run_cli("doctor", "--live")
        self.assertTrue(run_doctor.call_args.kwargs["live"])


class SetupAndServiceTests(CliTestCase):
    """`telegram-setup`, `install-service` and `uninstall-service`."""

    def test_telegram_setup_passes_wait_and_settings(self) -> None:
        """--wait is forwarded; the setup function does the work."""
        with mock.patch.object(cli, "telegram_setup", return_value=42) as setup:
            code, _, _ = self.run_cli("telegram-setup", "--wait", "300")
        self.assertEqual(code, 0)
        self.assertEqual(setup.call_args.kwargs["wait_seconds"], 300)
        self.assertEqual(setup.call_args.args[0].home, self.home)

    def test_telegram_setup_error_is_friendly(self) -> None:
        """A setup failure becomes one stderr line and exit 1."""
        with mock.patch.object(cli, "telegram_setup", side_effect=AutopilotError("No message arrived")):
            code, _, err = self.run_cli("telegram-setup")
        self.assertEqual(code, 1)
        self.assertEqual(err.strip(), "autopilot: No message arrived")

    def test_install_and_uninstall_service_use_patched_collaborators(self) -> None:
        """The unit lands in the patched folder and systemctl is never really called."""
        unit_dir = self.home / "units"
        calls: list[tuple[str, ...]] = []

        def fake_run(args, timeout: float = 20.0) -> tuple[int, str, str]:
            """Record the command and report success."""
            calls.append(tuple(args))
            return 0, "Linger=yes\n", ""

        with mock.patch.object(cli, "run_command", fake_run), mock.patch.object(
            cli, "default_unit_dir", return_value=unit_dir
        ):
            code, out, _ = self.run_cli("install-service")
            self.assertEqual(code, 0)
            unit = (unit_dir / "autopilot.service").read_text(encoding="utf-8")
            self.assertIn(f"AUTOPILOT_HOME={self.home}", unit)
            self.assertIn("-m autopilot start", unit)
            self.assertTrue(out.strip())
            self.assertIn(("systemctl", "--user", "daemon-reload"), calls)

            code, out, _ = self.run_cli("uninstall-service")
            self.assertEqual(code, 0)
            self.assertFalse((unit_dir / "autopilot.service").exists())
            self.assertTrue(out.strip())


if __name__ == "__main__":
    unittest.main()
