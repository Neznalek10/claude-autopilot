"""End-to-end tests: the real modules wired together, with a fake Claude CLI and a fake Telegram.

One test class per acceptance criterion (AC1 to AC15 in .forge/plan.md). Every
class drives real code: tasks.toml parsing, Engine, Store, ClaudeRunner (a real
subprocess running tests/fake_claude.py), TelegramNotifier and TelegramBot.
Only the two outside edges are fake: Claude is ``tests/fake_claude.py`` (or a
tiny wrapper script around it) and Telegram is ``FakeTelegramClient``.

Where it fits: ``tests/helpers.py`` supplies FakeClock and FakeTelegramClient;
``autopilot.app.build_app`` wires the object graph the same way the service does.
The CLI tests (AC10) run ``python -m autopilot`` as a subprocess and are skipped
until ``autopilot/cli.py`` exists.

Invariants:
    * Deterministic: time comes from FakeClock, files live in temp folders and
      nothing touches the network or the real Claude login.
    * The child environment handed to Claude always contains a fake
      ANTHROPIC_API_KEY, so every run doubles as a "no API key leaks" check.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import logging
import os
import re
import socket
import stat
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from unittest import mock

from dotenv import dotenv_values

from autopilot.app import App, build_app, configure_logging, run_service, telegram_setup
from autopilot.bot import BOT_COMMANDS, OFFSET_KEY, TelegramNotifier
from autopilot.doctor import format_report, run_doctor
from autopilot.engine import CHAT_SESSION_KEY, LIMITED_UNTIL_KEY, TICK_SECONDS, Engine
from autopilot.errors import ConfigError, TelegramError
from autopilot.models import (
    ASK_TASK_NAME,
    STATUS_CANCELLED,
    STATUS_ERROR,
    STATUS_RATE_LIMITED,
    STATUS_SUCCESS,
    STATUS_TIMEOUT,
    TRIGGER_CHAT,
    TRIGGER_MANUAL,
    TRIGGER_SCHEDULE,
    TRIGGER_TELEGRAM,
)
from autopilot.runner import ClaudeRunner
from autopilot.schedule import parse_schedule
from autopilot.service import UNIT_NAME, install_service, uninstall_service
from autopilot.settings import Settings, load_settings
from autopilot.store import INTERRUPTED_MESSAGE, Store
from autopilot.system import run_command
from autopilot.tasks import PROTECTED_READ_RULES, load_task_config
from autopilot.telegram_api import TelegramClient
from helpers import (
    FAKE_BOT_USERNAME,
    FAKE_CLAUDE,
    TZ,
    FakeClock,
    FakeTelegramClient,
    make_config,
    make_options,
    make_task,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TOKEN = "123456789:" + "B" * 35
CHAT_ID = 4711
SETUP_CODE = "482913"
FOREIGN_CHAT_ID = 999
START = datetime(2026, 11, 2, 7, 0, tzinfo=TZ)
WAIT_SECONDS = 10.0
POLL_SECONDS = 0.01
# First modification time given to tasks.toml; each rewrite adds one second so
# the engine's mtime comparison always sees an edit, whatever the file system's clock resolution.
BASE_MTIME_NS = 1_700_000_000 * 10**9
NS_PER_SECOND = 10**9

NEWS_TOML = """
[[task]]
name = "news"
prompt = "Say hi"
schedule = "daily 07:30"
"""
TWO_MANUAL_TASKS_TOML = """
[[task]]
name = "alpha"
prompt = "first job"

[[task]]
name = "beta"
prompt = "second job"
"""
LIMIT_WRAPPER = """#!{python}
import os
import subprocess
import sys

FAKE = {fake!r}
args = [sys.executable, FAKE, *sys.argv[1:]]
if os.path.exists(os.environ["LIMIT_FLAG_FILE"]) and "--version" not in sys.argv:
    prompt = sys.stdin.read()
    done = subprocess.run(args, input=prompt + "\\nFAKE_LIMIT", text=True)
    sys.exit(done.returncode)
os.execv(sys.executable, args)
"""
SPAWNING_WRAPPER = """#!{python}
import os
import subprocess
import sys
import time

child = subprocess.Popen(["sleep", "300"])
with open(os.environ["SPAWNED_PID_FILE"], "w") as pid_file:
    pid_file.write(str(child.pid))
time.sleep(300)
"""


class SleepyTelegramClient(FakeTelegramClient):
    """FakeTelegramClient whose empty polls pause briefly, so a polling thread does not spin."""

    def get_updates(self, offset: int | None = None, timeout: int = 50) -> list[dict[str, Any]]:
        """Return the next batch, sleeping a moment first when none is queued.

        Args:
            offset: Ignored.
            timeout: Ignored.

        Returns:
            The oldest queued batch, or an empty list.
        """
        batch = super().get_updates(offset, timeout)
        if not batch:
            time.sleep(0.02)
        return batch


class IntegrationCase(unittest.TestCase):
    """Base class: temp home, fake clock, fake Telegram, a fake child environment and wait helpers."""

    def setUp(self) -> None:
        """Create the temp home, clock, Telegram fake and the environment given to Claude."""
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.home = Path(tmp.name).resolve()
        self.clock = FakeClock(START)
        self.client = self.make_client()
        self.claude_log = self.home / "claude-calls.jsonl"
        # The fake credentials prove that autopilot strips them from the child.
        self.child_env: dict[str, str] = {
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "HOME": str(self.home),
            "FAKE_CLAUDE_LOG": str(self.claude_log),
            "ANTHROPIC_API_KEY": "sk-fake-key-for-tests",
            "ANTHROPIC_AUTH_TOKEN": "fake-token-for-tests",
        }
        self.mtime_ns = BASE_MTIME_NS
        self.update_counter = 0

    def make_client(self) -> FakeTelegramClient:
        """Create the Telegram fake used by this test (overridable)."""
        return FakeTelegramClient()

    # ------------------------------------------------------------------ build

    def make_settings(self, *, telegram: bool = True, max_parallel: int = 1, claude: Path = FAKE_CLAUDE) -> Settings:
        """Build Settings for the temp home, pointing CLAUDE_BIN at ``claude``.

        Args:
            telegram: Whether the bot token and chat id are configured.
            max_parallel: Worker count.
            claude: Executable standing in for the Claude CLI.

        Returns:
            Resolved settings (no real environment, no .env file).
        """
        env = {
            "CLAUDE_BIN": str(claude),
            "AUTOPILOT_TIMEZONE": "Europe/Prague",
            "AUTOPILOT_MAX_PARALLEL": str(max_parallel),
        }
        if telegram:
            env.update({"TELEGRAM_BOT_TOKEN": TOKEN, "TELEGRAM_CHAT_ID": str(CHAT_ID)})
        return load_settings(env=env, home=self.home, read_env_file=False)

    def write_tasks(self, text: str, settings: Settings | None = None) -> Path:
        """Write tasks.toml and give it a strictly newer modification time than before.

        Args:
            text: TOML content.
            settings: Settings whose tasks file to write; defaults to ``home/tasks.toml``.

        Returns:
            The file path.
        """
        path = settings.tasks_file if settings else self.home / "tasks.toml"
        path.write_text(text, encoding="utf-8")
        self.mtime_ns += NS_PER_SECOND
        os.utime(path, ns=(self.mtime_ns, self.mtime_ns))
        return path

    def build(
        self,
        tasks_toml: str = NEWS_TOML,
        *,
        claude: Path = FAKE_CLAUDE,
        max_parallel: int = 1,
        workers: bool = False,
    ) -> App:
        """Write tasks.toml and wire a full App with the fake Claude and fake Telegram.

        Args:
            tasks_toml: Content of tasks.toml.
            claude: Executable standing in for the Claude CLI.
            max_parallel: Worker count.
            workers: Start the worker threads right away.

        Returns:
            The App; its engine is stopped when the test ends.
        """
        settings = self.make_settings(max_parallel=max_parallel, claude=claude)
        self.write_tasks(tasks_toml, settings)
        runner = ClaudeRunner(lambda: claude, clock=self.clock, base_env=self.child_env)
        app = build_app(settings, runner=runner, client=self.client, clock=self.clock)
        self.addCleanup(app.engine.stop)
        if workers:
            app.engine.start_workers()
        return app

    # ------------------------------------------------------------------ drive

    def send(
        self, app: App, text: str, *, chat_id: int = CHAT_ID, sender_id: int | None = None
    ) -> None:
        """Deliver one Telegram text message to the bot.

        Args:
            app: App with a bot.
            text: Message text.
            chat_id: Chat the message comes from.
            sender_id: User who sent it; defaults to ``chat_id`` (a private chat).
        """
        assert app.bot is not None
        self.update_counter += 1
        app.bot.handle_update(
            {
                "update_id": self.update_counter,
                "message": {
                    "message_id": self.update_counter,
                    "chat": {"id": chat_id},
                    "from": {"id": chat_id if sender_id is None else sender_id},
                    "text": text,
                },
            }
        )

    def wait_until(self, predicate: Callable[[], bool], what: str) -> None:
        """Poll until ``predicate`` is true.

        Args:
            predicate: Condition to wait for.
            what: Description used in the failure message.
        """
        deadline = time.monotonic() + WAIT_SECONDS
        while time.monotonic() < deadline:
            if predicate():
                return
            time.sleep(POLL_SECONDS)
        self.fail(f"Timed out waiting for {what}")

    def wait_for_messages(self, count: int) -> None:
        """Wait until at least ``count`` messages reached the fake Telegram."""
        self.wait_until(lambda: len(self.client.sent) >= count, f"{count} Telegram messages")

    def texts(self) -> list[str]:
        """Return the text of every delivered message, in order."""
        return [message["text"] for message in self.client.sent]

    def claude_calls(self) -> list[dict[str, Any]]:
        """Return the fake Claude's call log: one dict per invocation that read a prompt."""
        if not self.claude_log.exists():
            return []
        return [json.loads(line) for line in self.claude_log.read_text(encoding="utf-8").splitlines()]

    def write_script(self, name: str, template: str, **values: str) -> Path:
        """Write an executable Python script into the temp home.

        Args:
            name: File name.
            template: Script text with ``{python}`` and other ``str.format`` fields.
            **values: Extra format values.

        Returns:
            The script path.
        """
        path = self.home / name
        path.write_text(template.format(python=sys.executable, **values), encoding="utf-8")
        path.chmod(path.stat().st_mode | stat.S_IXUSR)
        return path


def argument_after(argv: list[str], flag: str) -> str:
    """Return the argv entry that follows ``flag``.

    Args:
        argv: Command-line arguments passed to the fake Claude.
        flag: The option to look up.

    Returns:
        The value after the flag.
    """
    return argv[argv.index(flag) + 1]


class AC1TasksFileTests(IntegrationCase):
    """AC1: tasks.toml defines tasks; errors name the task and field; the list shows schedules in words."""

    def test_every_documented_option_is_loaded(self) -> None:
        """A task using all options loads with each value in the right place."""
        workdir = self.home / "work"
        workdir.mkdir()
        path = self.write_tasks(
            f"""
[[task]]
name = "full"
description = "Everything set"
prompt = "Do it"
schedule = ["daily 07:30", "weekdays 18:00"]
workdir = "{workdir}"
model = "sonnet"
effort = "high"
allowed_tools = ["Read", "Bash(git log *)"]
disallowed_tools = ["WebFetch"]
permission_mode = "acceptEdits"
max_budget_usd = 1.5
timeout_minutes = 7
notify = "errors"
enabled = false
"""
        )
        task = load_task_config(path).tasks["full"]
        self.assertEqual([s.description for s in task.schedules], ["daily at 07:30", "weekdays at 18:00"])
        self.assertEqual(task.options.workdir, workdir)
        self.assertEqual((task.options.model, task.options.effort), ("sonnet", "high"))
        self.assertEqual(task.options.allowed_tools, ("Read", "Bash(git log *)"))
        self.assertEqual(task.options.disallowed_tools[0], "WebFetch")
        self.assertEqual(task.options.disallowed_tools[1:], PROTECTED_READ_RULES)
        self.assertEqual(task.options.permission_mode, "acceptEdits")
        self.assertEqual(task.options.max_budget_usd, 1.5)
        self.assertEqual(task.options.timeout_seconds, 7 * 60)
        self.assertEqual((task.notify, task.enabled), ("errors", False))

    def test_error_names_the_task_and_the_field(self) -> None:
        """A bad value is reported with the task name and the field, and build_app fails loudly."""
        bad = '[[task]]\nname = "broken"\nprompt = "x"\npermission_mode = "nonsense"\n'
        settings = self.make_settings()
        self.write_tasks(bad, settings)
        with self.assertRaises(ConfigError) as caught:
            load_task_config(settings.tasks_file)
        self.assertIn("broken", str(caught.exception))
        self.assertIn("permission_mode", str(caught.exception))
        with self.assertRaises(ConfigError):
            build_app(settings, runner=ClaudeRunner(lambda: FAKE_CLAUDE, clock=self.clock), clock=self.clock)

    def test_typo_in_a_key_is_reported(self) -> None:
        """A misspelled key (shedule) is an error instead of a silently manual task."""
        path = self.write_tasks('[[task]]\nname = "typo"\nprompt = "x"\nshedule = "daily 07:30"\n')
        with self.assertRaises(ConfigError) as caught:
            load_task_config(path)
        self.assertIn("shedule", str(caught.exception))

    def test_task_list_shows_schedule_in_words_and_next_run(self) -> None:
        """/tasks lists each task with its schedule described in words and its next run time."""
        app = self.build()
        self.send(app, "/tasks")
        text = self.texts()[-1]
        self.assertIn("news", text)
        self.assertIn("daily at 07:30", text)
        self.assertIn("07:30", text)


class AC2ScheduleTests(unittest.TestCase):
    """AC2: cron, aliases and friendly forms parse; next fire times are DST correct."""

    def test_friendly_forms_describe_themselves(self) -> None:
        """Each friendly form yields its plain-English description."""
        expected = {
            "every 15m": "every 15 minutes",
            "every 2h": "every 2 hours",
            "hourly": "hourly",
            "daily 07:30": "daily at 07:30",
            "weekdays 08:00": "weekdays at 08:00",
            "weekends 10:00": "weekends at 10:00",
            "mon,wed,fri 18:00": "Mon, Wed, Fri at 18:00",
            "mon-fri 07:00": "weekdays at 07:00",
            "30 7 * * 1-5": "weekdays at 07:30",
        }
        for expression, description in expected.items():
            with self.subTest(expression):
                self.assertEqual(parse_schedule(expression).description, description)

    def test_alias_and_cron_next_times(self) -> None:
        """Cron weekdays skip the weekend and @hourly fires on the next full hour."""
        friday = datetime(2026, 11, 6, 8, 0, tzinfo=TZ)
        self.assertEqual(parse_schedule("30 7 * * 1-5").next_after(friday), datetime(2026, 11, 9, 7, 30, tzinfo=TZ))
        self.assertEqual(parse_schedule("@hourly").next_after(friday), datetime(2026, 11, 6, 9, 0, tzinfo=TZ))
        self.assertEqual(parse_schedule("every 15m").next_after(friday.replace(minute=1)), friday.replace(minute=15))

    def test_preview_of_five_fire_times(self) -> None:
        """Five successive fire times of a weekday schedule skip Saturday and Sunday."""
        schedule = parse_schedule("weekdays 07:00")
        moment = datetime(2026, 11, 6, 7, 0, tzinfo=TZ)
        days = []
        for _ in range(5):
            moment = schedule.next_after(moment)
            days.append((moment.month, moment.day, moment.hour))
        self.assertEqual(days, [(11, 9, 7), (11, 10, 7), (11, 11, 7), (11, 12, 7), (11, 13, 7)])

    def test_spring_forward_gap_is_skipped(self) -> None:
        """02:30 does not exist on 2026-03-29 in Prague, so that day has no fire."""
        schedule = parse_schedule("daily 02:30")
        self.assertEqual(
            schedule.next_after(datetime(2026, 3, 28, 12, 0, tzinfo=TZ)),
            datetime(2026, 3, 30, 2, 30, tzinfo=TZ),
        )

    def test_fall_back_hour_fires_once(self) -> None:
        """02:30 happens twice on 2026-10-25 but fires once; the next fire is 25 real hours later."""
        schedule = parse_schedule("daily 02:30")
        first = schedule.next_after(datetime(2026, 10, 24, 12, 0, tzinfo=TZ))
        second = schedule.next_after(first)
        self.assertEqual((first.month, first.day), (10, 25))
        self.assertEqual((second.month, second.day), (10, 26))
        elapsed = second.astimezone(timezone.utc) - first.astimezone(timezone.utc)
        self.assertEqual(elapsed, timedelta(hours=25))


class AC3RunTests(IntegrationCase):
    """AC3: a due slot runs `claude -p` on the subscription and the run is stored."""

    def test_scheduled_slot_runs_and_result_reaches_telegram(self) -> None:
        """A slot becoming due runs the task and its result arrives with a status header."""
        app = self.build(workers=True)
        self.assertEqual(app.engine.tick(), [])  # first sight of the task: wait for the next slot
        self.clock.set(datetime(2026, 11, 2, 7, 31, tzinfo=TZ))
        (run_id,) = app.engine.tick()
        self.wait_for_messages(1)

        message = self.texts()[0]
        self.assertTrue(message.startswith("✅ <b>news</b>"), message)
        self.assertIn("echo: Say hi", message)
        record = app.store.get_run(run_id)
        assert record is not None
        self.assertEqual((record.status, record.trigger), (STATUS_SUCCESS, TRIGGER_SCHEDULE))
        self.assertEqual(record.result, "echo: Say hi")
        self.assertEqual(record.num_turns, 1)
        self.assertTrue((record.session_id or "").startswith("fake-session-"))
        self.assertIsNotNone(record.duration_ms)
        self.assertEqual(record.scheduled_for, datetime(2026, 11, 2, 7, 30, tzinfo=TZ))

    def test_command_line_prompt_on_stdin_and_no_api_key(self) -> None:
        """Claude gets the prompt on stdin, the explicit flags and workdir, and never the API key."""
        workdir = self.home / "project"
        workdir.mkdir()
        app = self.build(
            f'[[task]]\nname = "news"\nprompt = "Say hi"\nworkdir = "{workdir}"\nmodel = "sonnet"\n'
        )
        run_id = app.engine.enqueue_task("news", TRIGGER_MANUAL, dispatch=False)
        assert run_id is not None
        app.engine.execute_run(run_id)

        (call,) = self.claude_calls()
        argv = call["argv"]
        self.assertEqual(call["prompt"], "Say hi")
        self.assertNotIn("Say hi", argv)
        self.assertIn("-p", argv)
        self.assertEqual(argument_after(argv, "--output-format"), "json")
        self.assertEqual(argument_after(argv, "--permission-prompts"), "none")
        self.assertEqual(argument_after(argv, "--permission-mode"), "dontAsk")
        self.assertEqual(argument_after(argv, "--model"), "sonnet")
        self.assertEqual(call["cwd"], str(workdir))
        self.assertFalse(call["has_api_key"])

    def test_secrets_are_denied_and_a_task_without_workdir_runs_in_the_workspace(self) -> None:
        """The real argv denies reading secrets (incl. .env) and the default cwd is the empty workspace."""
        app = self.build()
        run_id = app.engine.enqueue_task("news", TRIGGER_MANUAL, dispatch=False)
        assert run_id is not None
        app.engine.execute_run(run_id)

        (call,) = self.claude_calls()
        argv = call["argv"]
        denied = argv[argv.index("--disallowedTools") + 1 :]
        for rule in PROTECTED_READ_RULES:
            self.assertIn(rule, denied)
        self.assertIn(f"Read(/{self.home / '.env'})", denied)
        self.assertIn(f"Read(/{self.home / 'data' / 'autopilot.db'})", denied)
        workspace = self.home / "workspace"
        self.assertEqual(call["cwd"], str(workspace))
        self.assertEqual(stat.S_IMODE(workspace.stat().st_mode), 0o700)

    def test_default_runner_resolves_claude_bin_and_strips_the_api_key(self) -> None:
        """build_app without a custom runner finds CLAUDE_BIN and still keeps API keys out of the child."""
        settings = self.make_settings()
        self.write_tasks(NEWS_TOML, settings)
        parent_env = {
            "FAKE_CLAUDE_LOG": str(self.claude_log),
            "ANTHROPIC_API_KEY": "sk-fake-key-for-tests",
            "ANTHROPIC_AUTH_TOKEN": "fake-token-for-tests",
        }
        with mock.patch.dict(os.environ, parent_env):
            app = build_app(settings, client=self.client, clock=self.clock)
            self.addCleanup(app.engine.stop)
            run_id = app.engine.enqueue_task("news", TRIGGER_MANUAL, dispatch=False)
            assert run_id is not None
            record = app.engine.execute_run(run_id)
        assert record is not None
        self.assertEqual(record.status, STATUS_SUCCESS)
        (call,) = self.claude_calls()
        self.assertFalse(call["has_api_key"])

    def test_blocked_tools_are_stored_and_listed_with_a_hint(self) -> None:
        """Tools Claude was denied are saved on the run and reported to the phone."""
        app = self.build('[[task]]\nname = "news"\nprompt = "FAKE_DENIED please"\n')
        run_id = app.engine.enqueue_task("news", TRIGGER_MANUAL, dispatch=False)
        assert run_id is not None
        record = app.engine.execute_run(run_id)
        assert record is not None
        self.assertEqual(record.denied_tools, ("Bash",))
        self.assertIn("Blocked tools: Bash", self.texts()[0])
        self.assertIn("allowed_tools", self.texts()[0])


class AC4SerializationTests(IntegrationCase):
    """AC4: runs are serialized, never queued twice, and a run over its timeout is killed."""

    def test_second_run_waits_and_duplicates_are_refused(self) -> None:
        """With one worker the second task stays queued while the first runs; a task is not queued twice."""
        app = self.build(
            '[[task]]\nname = "alpha"\nprompt = "FAKE_SLEEP 0.5 a"\n'
            '[[task]]\nname = "beta"\nprompt = "FAKE_SLEEP 0.5 b"\n',
            workers=True,
        )
        engine = app.engine
        self.assertIsNotNone(engine.enqueue_task("alpha", TRIGGER_MANUAL))
        self.assertIsNotNone(engine.enqueue_task("beta", TRIGGER_MANUAL))
        self.wait_until(lambda: len(engine.snapshot().running) == 1, "the first run to start")
        snapshot = engine.snapshot()
        self.assertEqual([r.task_name for r in snapshot.running], ["alpha"])
        self.assertEqual([r.task_name for r in snapshot.queued], ["beta"])
        self.assertIsNone(engine.enqueue_task("alpha", TRIGGER_MANUAL))
        self.wait_for_messages(2)
        self.assertEqual(len(app.store.list_runs(limit=10)), 2)

    def test_timeout_kills_the_whole_process_group(self) -> None:
        """A run over its timeout is recorded as timeout and its child processes die too."""
        pid_file = self.home / "spawned.pid"
        self.child_env["SPAWNED_PID_FILE"] = str(pid_file)
        wrapper = self.write_script("claude-spawner", SPAWNING_WRAPPER)
        task = make_task("slow", "go", options=make_options(self.home, timeout_seconds=2))
        engine = Engine(
            store=Store(self.home / "data" / "autopilot.db"),
            runner=ClaudeRunner(lambda: wrapper, clock=self.clock, base_env=self.child_env),
            notifier=TelegramNotifier(self.client, CHAT_ID, local_tz=TZ),
            config_loader=lambda: make_config(task),
            clock=self.clock,
        )
        engine.load_config()
        run_id = engine.enqueue_task("slow", TRIGGER_MANUAL, dispatch=False)
        assert run_id is not None
        record = engine.execute_run(run_id)

        assert record is not None
        self.assertEqual(record.status, STATUS_TIMEOUT)
        self.assertTrue((record.error or "").startswith("Timed out"))
        self.assertIn("timed out", self.texts()[0])
        self.wait_until(lambda: self.process_is_gone(int(pid_file.read_text())), "the spawned child to die")

    def test_stopping_during_a_run_records_an_interrupted_error_silently(self) -> None:
        """Engine.stop() ends a running Claude call; the run is an interrupted error and nothing is sent."""
        app = self.build('[[task]]\nname = "slow"\nprompt = "FAKE_SLEEP 30"\n', workers=True)
        run_id = app.engine.enqueue_task("slow", TRIGGER_MANUAL)
        assert run_id is not None
        self.wait_until(lambda: len(app.engine.snapshot().running) == 1, "the run to start")
        self.wait_until(lambda: len(self.claude_calls()) == 1, "Claude to be called")
        app.engine.stop()

        record = app.store.get_run(run_id)
        assert record is not None
        self.assertEqual(record.status, STATUS_ERROR)
        self.assertEqual(record.error, INTERRUPTED_MESSAGE)
        self.assertEqual(self.client.sent, [])

    @staticmethod
    def process_is_gone(pid: int) -> bool:
        """Return True when ``pid`` no longer exists or is only an unreaped zombie."""
        try:
            fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        except OSError:
            return True
        return fields[0] == "Z"


class AC5SlotTests(IntegrationCase):
    """AC5: slightly late slots run, old ones are skipped without a burst, paused tasks stay quiet."""

    def prepare(self, toml: str = NEWS_TOML) -> App:
        """Build the app with workers and let the first tick register the task."""
        app = self.build(toml, workers=True)
        app.engine.tick()
        return app

    def test_slot_four_minutes_late_still_runs(self) -> None:
        """A restart a few minutes after the slot does not lose the run."""
        app = self.prepare()
        self.clock.set(datetime(2026, 11, 2, 7, 34, tzinfo=TZ))
        self.assertEqual(len(app.engine.tick()), 1)
        self.wait_for_messages(1)

    def test_old_slot_is_skipped_and_logged(self) -> None:
        """A slot ten minutes late is dropped (with a log line); the next day's slot runs normally."""
        app = self.prepare()
        self.clock.set(datetime(2026, 11, 2, 7, 40, tzinfo=TZ))
        with self.assertLogs("autopilot.engine", logging.INFO) as logs:
            self.assertEqual(app.engine.tick(), [])
        self.assertTrue(any("Skipped the missed slot" in line for line in logs.output))
        self.assertEqual(app.store.list_runs(), [])
        self.clock.set(datetime(2026, 11, 3, 7, 31, tzinfo=TZ))
        self.assertEqual(len(app.engine.tick()), 1)

    def test_long_gap_runs_the_latest_slot_once_not_a_burst(self) -> None:
        """Two hours of missed 15-minute slots produce one run, not eight."""
        app = self.prepare('[[task]]\nname = "news"\nprompt = "Say hi"\nschedule = "every 15m"\n')
        self.clock.set(datetime(2026, 11, 2, 9, 3, tzinfo=TZ))
        self.assertEqual(len(app.engine.tick()), 1)
        self.wait_for_messages(1)
        self.assertEqual(len(app.store.list_runs()), 1)

    def test_paused_and_disabled_tasks_skip_the_schedule_but_run_manually(self) -> None:
        """A paused or disabled task is silent on schedule yet /run still works."""
        toml = NEWS_TOML + '\n[[task]]\nname = "off"\nprompt = "Say hi"\nschedule = "daily 07:30"\nenabled = false\n'
        app = self.prepare(toml)
        app.store.set_paused("news", True)
        self.clock.set(datetime(2026, 11, 2, 7, 31, tzinfo=TZ))
        self.assertEqual(app.engine.tick(), [])
        self.send(app, "/run off")
        self.wait_for_messages(2)  # "Queued" reply plus the result
        self.assertTrue(any("echo: Say hi" in text for text in self.texts()))
        self.assertEqual([r.trigger for r in app.store.list_runs()], [TRIGGER_TELEGRAM])


class AC6UsageLimitTests(IntegrationCase):
    """AC6: one notice, queued runs wait without calling Claude, and runs retry after the reset."""

    def test_limit_holds_the_queue_and_retries_after_the_reset(self) -> None:
        """Full flow: limit hit, other run deferred unseen by Claude, retry after the reset succeeds."""
        flag = self.home / "limit.flag"
        flag.write_text("limited")
        self.child_env["LIMIT_FLAG_FILE"] = str(flag)
        reset = self.clock() + timedelta(hours=1)
        self.child_env["FAKE_RESET_EPOCH"] = str(int(reset.timestamp()))
        wrapper = self.write_script("claude-limit", LIMIT_WRAPPER, fake=str(FAKE_CLAUDE))
        app = self.build(TWO_MANUAL_TASKS_TOML, claude=wrapper)
        engine = app.engine
        first = engine.enqueue_task("alpha", TRIGGER_MANUAL)
        second = engine.enqueue_task("beta", TRIGGER_MANUAL)
        engine.start_workers()
        self.wait_until(lambda: not app.store.active_runs(), "both runs to be held")

        # Both runs are rate limited, but Claude was called only for the first.
        for run_id in (first, second):
            record = app.store.get_run(run_id)
            assert record is not None
            self.assertEqual(record.status, STATUS_RATE_LIMITED)
            self.assertIsNotNone(record.retry_at)
        self.assertEqual(len(self.claude_calls()), 1)
        self.assertEqual(len(self.texts()), 1)
        self.assertIn("usage limit", self.texts()[0])
        self.assertIn("08:02", self.texts()[0])  # reset 08:00 + the 2 minute safety buffer, in the user's zone
        self.assertEqual(engine.snapshot().limited_until is not None, True)

        # Before the reset a tick retries nothing.
        self.clock.advance(minutes=30)
        self.assertEqual(engine.tick(), [])

        # After the reset the limit is gone and both runs are retried automatically.
        flag.unlink()
        self.clock.set(reset + timedelta(minutes=3))
        retries = engine.tick()
        self.assertEqual(len(retries), 2)
        self.wait_for_messages(3)
        self.wait_until(lambda: not app.store.active_runs(), "the retries to finish")
        self.wait_for_messages(3)
        finished = [app.store.get_run(run_id) for run_id in retries]
        self.assertEqual([r.status for r in finished if r], [STATUS_SUCCESS, STATUS_SUCCESS])
        self.assertEqual([r.retry_of for r in finished if r], [first, second])
        self.assertEqual(len(self.claude_calls()), 3)
        self.assertEqual(sum("usage limit" in text for text in self.texts()), 1)
        self.assertTrue(any("retry" in text for text in self.texts()[1:]))
        self.assertIsNone(engine.snapshot().limited_until)
        self.assertIsNotNone(app.store.get_value(LIMITED_UNTIL_KEY))

    def test_chat_message_during_the_limit_is_deferred_not_sent_to_claude(self) -> None:
        """A chat turn sent while the gate is closed waits for the reset without a Claude call."""
        app = self.build()
        app.store.set_value(LIMITED_UNTIL_KEY, (self.clock() + timedelta(hours=2)).isoformat())
        run_id = app.engine.enqueue_prompt("hello", TRIGGER_CHAT, dispatch=False)
        record = app.engine.execute_run(run_id)
        assert record is not None
        self.assertEqual(record.status, STATUS_RATE_LIMITED)
        self.assertEqual(self.claude_calls(), [])
        self.assertEqual(self.client.sent, [])


class AC7NotificationTests(IntegrationCase):
    """AC7: results reach Telegram per notify policy, long ones as a file, with a plain-text fallback."""

    def run_task(self, app: App, name: str) -> None:
        """Queue a manual run of ``name`` and execute it synchronously."""
        run_id = app.engine.enqueue_task(name, TRIGGER_MANUAL, dispatch=False)
        assert run_id is not None
        app.engine.execute_run(run_id)

    def test_long_result_arrives_as_a_document(self) -> None:
        """A result longer than three messages is announced once and attached as a .md file."""
        long_prompt = "\n".join(f"line {i:03d} alpha beta gamma delta epsilon" for i in range(600))
        app = self.build(f'[[task]]\nname = "news"\nprompt = {json.dumps(long_prompt)}\n')
        self.run_task(app, "news")

        (message,) = self.client.sent
        self.assertIn("Full result attached as a file", message["text"])
        (document,) = self.client.documents
        self.assertRegex(document["filename"], r"^news-run1-\d{8}-\d{4}\.md$")
        content = document["content"].decode("utf-8")
        self.assertIn("line 000", content)
        self.assertIn("line 599", content)

    def test_notify_policies(self) -> None:
        """`never` sends nothing, `errors` sends only failures, `always` sends everything."""
        app = self.build(
            """
[[task]]
name = "quiet"
prompt = "ok"
notify = "never"

[[task]]
name = "errors-ok"
prompt = "ok"
notify = "errors"

[[task]]
name = "errors-bad"
prompt = "FAKE_ERROR"
notify = "errors"

[[task]]
name = "loud"
prompt = "ok"
"""
        )
        for name in ("quiet", "errors-ok"):
            self.run_task(app, name)
        self.assertEqual(self.client.sent, [])
        self.run_task(app, "errors-bad")
        self.assertEqual(len(self.client.sent), 1)
        self.assertTrue(self.texts()[0].startswith("❌ <b>errors-bad</b>"))
        self.assertIn("Claude Code failed during execution.", self.texts()[0])
        self.run_task(app, "loud")
        self.assertEqual(len(self.client.sent), 2)

    def test_message_telegram_cannot_parse_is_resent_as_plain_text(self) -> None:
        """When Telegram rejects the HTML the same content is delivered as plain text."""
        app = self.build()
        self.client.fail_html_once = True
        self.run_task(app, "news")
        (message,) = self.client.sent
        self.assertIsNone(message["parse_mode"])
        self.assertNotIn("<b>", message["text"])
        self.assertIn("echo: Say hi", message["text"])

    def test_failed_run_names_the_error(self) -> None:
        """A run that ends in an error shows the error in its header message."""
        app = self.build('[[task]]\nname = "news"\nprompt = "FAKE_GARBAGE"\n')
        self.run_task(app, "news")
        self.assertTrue(self.texts()[0].startswith("❌"))
        self.assertIn("couldn't read", self.texts()[0])


class AC8BotTests(IntegrationCase):
    """AC8: the bot answers only the configured chat and supports the documented commands."""

    def test_foreign_chat_is_ignored_and_logged_with_its_id(self) -> None:
        """Messages and commands from another chat get no reply, start nothing and are logged."""
        app = self.build()
        with self.assertLogs("autopilot.bot", logging.WARNING) as logs:
            self.send(app, "hello", chat_id=FOREIGN_CHAT_ID)
            self.send(app, "/run news", chat_id=FOREIGN_CHAT_ID)
        self.assertTrue(all(str(FOREIGN_CHAT_ID) in line for line in logs.output))
        self.assertEqual(self.client.sent, [])
        self.assertEqual(self.client.chat_actions, [])
        self.assertEqual(app.store.list_runs(), [])

    def test_group_chat_and_foreign_sender_in_the_owner_chat_are_ignored(self) -> None:
        """Only the owner's private chat is accepted: a group, or another user posting there, gets no answer."""
        app = self.build()
        with self.assertLogs("autopilot.bot", logging.WARNING):
            self.send(app, "/run news", chat_id=-100123)  # a group chat id
            self.send(app, "/run news", chat_id=CHAT_ID, sender_id=FOREIGN_CHAT_ID)
        self.assertEqual(self.client.sent, [])
        self.assertEqual(app.store.list_runs(), [])

    def test_commands(self) -> None:
        """Each command produces its documented reply or side effect."""
        app = self.build()
        self.send(app, "/help")
        self.send(app, "/start")
        self.assertEqual(self.texts()[0], self.texts()[1])
        self.assertIn("/new", self.texts()[0])

        self.send(app, "/Help@" + FAKE_BOT_USERNAME)  # group-style suffix, any case
        self.assertEqual(self.texts()[2], self.texts()[0])

        self.send(app, "/nonsense")
        self.assertIn("Unknown command", self.texts()[-1])

        self.send(app, "/run missing")
        self.assertIn("Unknown task 'missing'", self.texts()[-1])
        self.send(app, "/run news")
        self.assertIn("Queued <b>news</b>", self.texts()[-1])
        self.send(app, "/run news")
        self.assertIn("already queued", self.texts()[-1])
        self.send(app, "/status")
        self.assertIn("news", self.texts()[-1])

        self.send(app, "/pause news")
        self.assertTrue(app.store.is_paused("news"))
        self.send(app, "/resume news")
        self.assertFalse(app.store.is_paused("news"))

        self.send(app, "/last")
        self.assertIn("No finished runs yet", self.texts()[-1])

    def test_last_resends_the_newest_finished_run(self) -> None:
        """/last delivers the stored result of the latest finished run."""
        app = self.build()
        run_id = app.engine.enqueue_task("news", TRIGGER_MANUAL, dispatch=False)
        assert run_id is not None
        app.engine.execute_run(run_id)
        self.client.sent.clear()
        self.send(app, "/last news")
        self.assertIn("echo: Say hi", self.texts()[0])

    def test_run_command_end_to_end(self) -> None:
        """/run queues the task, a worker executes it and the result is pushed back."""
        app = self.build(workers=True)
        self.send(app, "/run news")
        self.wait_for_messages(2)
        self.assertIn("Queued", self.texts()[0])
        self.assertIn("echo: Say hi", self.texts()[1])

    def test_service_polls_telegram_answers_and_announces_itself(self) -> None:
        """run_service starts the bot thread: it registers commands, greets, answers and stops cleanly."""
        app = self.build()
        leftover = app.store.create_run(task_name="news", trigger=TRIGGER_MANUAL, prompt="old", queued_at=self.clock())
        stop = threading.Event()
        self.addCleanup(stop.set)
        service = threading.Thread(target=run_service, args=(app, stop), daemon=True)
        service.start()
        self.wait_for_messages(1)
        self.assertIn("Autopilot is running", self.texts()[0])
        interrupted = app.store.get_run(leftover)
        assert interrupted is not None
        self.assertEqual(interrupted.status, STATUS_ERROR)  # a run left by a previous process is closed at startup
        self.wait_until(lambda: self.client.commands is not None, "the command menu to be registered")
        self.assertEqual(self.client.commands, list(BOT_COMMANDS))

        self.client.updates.append(
            [{"update_id": 41, "message": {"chat": {"id": CHAT_ID}, "from": {"id": CHAT_ID}, "text": "ping"}}]
        )
        self.wait_until(lambda: "echo: ping" in self.texts(), "the chat answer")
        self.wait_until(lambda: app.store.get_value(OFFSET_KEY) == "42", "the poll offset to be stored")
        stop.set()
        service.join(WAIT_SECONDS)
        self.assertFalse(service.is_alive())

    def make_client(self) -> FakeTelegramClient:
        """Use a client whose idle polls sleep, so the service thread in the test does not spin."""
        return SleepyTelegramClient()


class AC9HotReloadTests(IntegrationCase):
    """AC9: tasks.toml edits apply without a restart; a broken edit keeps the old tasks and warns once."""

    def test_edit_adds_a_task_and_a_broken_edit_warns_once(self) -> None:
        """Valid edit applies on the next tick; an invalid one keeps the tasks and sends one warning."""
        app = self.build(workers=True)
        engine = app.engine
        self.write_tasks(NEWS_TOML + '\n[[task]]\nname = "extra"\nprompt = "more"\n')
        engine.tick()
        self.assertEqual(list(engine.config.tasks), ["news", "extra"])
        self.send(app, "/tasks")
        self.assertIn("extra", self.texts()[-1])

        broken = '[[task]]\nname = "news"\nprompt = "x"\nshedule = "daily 07:30"\n'
        self.write_tasks(broken)
        engine.tick()
        warnings = [text for text in self.texts() if "tasks.toml has an error" in text]
        self.assertEqual(len(warnings), 1)
        self.assertIn("shedule", warnings[0])
        self.assertEqual(list(engine.config.tasks), ["news", "extra"])

        # The same broken content saved again, and further ticks, stay quiet.
        self.write_tasks(broken)
        engine.tick()
        engine.tick()
        self.assertEqual(len([t for t in self.texts() if "tasks.toml has an error" in t]), 1)

        # The previous tasks keep working while the file is broken.
        self.send(app, "/run extra")
        self.wait_until(lambda: "echo: more" in "".join(self.texts()), "the run with the old tasks")

        # Fixing the file applies the new content.
        self.write_tasks('[[task]]\nname = "only"\nprompt = "fixed"\n')
        engine.tick()
        self.assertEqual(list(engine.config.tasks), ["only"])

    def test_tick_interval_is_within_the_promised_30_seconds(self) -> None:
        """The service loop wakes at least every 30 seconds, which bounds the reload delay."""
        self.assertLessEqual(TICK_SECONDS, 30)

    def test_run_of_a_removed_task_is_cancelled(self) -> None:
        """A task deleted while its run was queued ends as cancelled without calling Claude."""
        app = self.build()
        run_id = app.engine.enqueue_task("news", TRIGGER_MANUAL, dispatch=False)
        assert run_id is not None
        self.write_tasks('[[task]]\nname = "other"\nprompt = "x"\n')
        app.engine.tick()
        record = app.engine.execute_run(run_id)
        assert record is not None
        self.assertEqual(record.status, STATUS_CANCELLED)
        self.assertEqual(self.claude_calls(), [])


class AC10CliTests(IntegrationCase):
    """AC10: the command line works end to end (skipped until autopilot/cli.py exists)."""

    def autopilot(self, *args: str) -> subprocess.CompletedProcess[str]:
        """Run ``python -m autopilot`` in the temp home with the fake Claude.

        Args:
            *args: Command-line arguments.

        Returns:
            The finished process with captured text output.
        """
        env = {
            **self.child_env,
            "AUTOPILOT_HOME": str(self.home),
            "AUTOPILOT_TIMEZONE": "Europe/Prague",
            "CLAUDE_BIN": str(FAKE_CLAUDE),
            "PYTHONPATH": str(PROJECT_ROOT),
        }
        return subprocess.run(
            [sys.executable, "-m", "autopilot", *args],
            capture_output=True,
            text=True,
            env=env,
            cwd=self.home,
            timeout=60,
            check=False,
        )

    @unittest.skipUnless(importlib.util.find_spec("autopilot.cli"), "autopilot.cli is not written yet")
    def test_validate_list_next_run_history_show(self) -> None:
        """validate, list, next, run, history and show behave as documented."""
        self.write_tasks(NEWS_TOML)
        validated = self.autopilot("validate")
        self.assertEqual(validated.returncode, 0, validated.stderr)
        self.assertIn("tasks.toml OK: 1 task", validated.stdout)
        self.assertIn("daily at 07:30", self.autopilot("list").stdout)
        preview = self.autopilot("next", "weekdays 07:00")
        self.assertEqual(preview.returncode, 0, preview.stderr)
        self.assertEqual(len(re.findall(r"\d{4}-\d{2}-\d{2} 07:00", preview.stdout)), 5)

        ran = self.autopilot("run", "news", "--no-notify")
        self.assertEqual(ran.returncode, 0, ran.stderr)
        self.assertIn("echo: Say hi", ran.stdout)
        self.assertFalse(self.claude_calls()[0]["has_api_key"])
        self.assertIn("news", self.autopilot("history").stdout)
        self.assertIn("echo: Say hi", self.autopilot("show", "1").stdout)

    @unittest.skipUnless(importlib.util.find_spec("autopilot.cli"), "autopilot.cli is not written yet")
    def test_expected_errors_are_one_friendly_line_and_exit_1(self) -> None:
        """A bad tasks.toml and an unknown task print one line to stderr, no traceback, exit 1."""
        self.write_tasks('[[task]]\nname = "broken"\nprompt = "x"\neffort = "huge"\n')
        invalid = self.autopilot("validate")
        self.assertEqual(invalid.returncode, 1)
        self.assertIn("broken", invalid.stderr)
        self.assertIn("effort", invalid.stderr)
        self.assertNotIn("Traceback", invalid.stderr)

        self.write_tasks(NEWS_TOML)
        unknown = self.autopilot("run", "missing", "--no-notify")
        self.assertEqual(unknown.returncode, 1)
        self.assertIn("Unknown task", unknown.stderr)
        self.assertNotIn("Traceback", unknown.stderr)


class AC11DoctorTests(IntegrationCase):
    """AC11: doctor checks the Claude CLI, login, API key warning, tasks, Telegram, data folder and service."""

    def command_runner(self, args: Any, timeout: float = 20.0) -> tuple[int, str, str]:
        """Run the fake Claude for real and pretend systemd is unavailable."""
        if str(args[0]) == str(FAKE_CLAUDE):
            return run_command(args, timeout)
        return 127, "", "not available in tests"

    def test_report_for_a_healthy_setup(self) -> None:
        """Claude CLI, login (plan only), tasks, Telegram and data folder are ok; the API key is a warning."""
        settings = self.make_settings()
        self.write_tasks(NEWS_TOML, settings)
        results = run_doctor(
            settings,
            run=self.command_runner,
            env={"ANTHROPIC_API_KEY": "sk-fake-key-for-tests"},
            client_factory=lambda _token: self.client,
            user="tester",
        )
        by_name = {result.name: result for result in results}
        self.assertEqual(by_name["Claude CLI"].status, "ok")
        self.assertIn("9.9.9", by_name["Claude CLI"].detail)
        self.assertEqual(by_name["Claude login"].status, "ok")
        self.assertIn("pro plan", by_name["Claude login"].detail)
        self.assertEqual(by_name["API key"].status, "warn")
        self.assertEqual(by_name["tasks.toml"].status, "ok")
        self.assertEqual(by_name["Telegram"].status, "ok")
        report = format_report(results)
        self.assertNotIn("sk-fake-key-for-tests", report)
        self.assertNotIn("@", report.replace(f"@{FAKE_BOT_USERNAME}", ""))

    def test_live_check_makes_one_tiny_call(self) -> None:
        """--live runs one real (fake) Claude call and reports it."""
        settings = self.make_settings(telegram=False)
        self.write_tasks(NEWS_TOML, settings)
        with mock.patch.dict(os.environ, {"FAKE_CLAUDE_LOG": str(self.claude_log)}):
            results = run_doctor(settings, live=True, run=self.command_runner, env={}, user="tester")
        self.assertEqual({r.name: r.status for r in results}["Live call"], "ok")
        self.assertEqual(len(self.claude_calls()), 1)


class AC12TelegramSetupTests(IntegrationCase):
    """AC12: telegram-setup validates the token, learns the chat, saves both and sends a test message."""

    def test_setup_saves_credentials_and_confirms(self) -> None:
        """The first message to the bot yields the chat id; token and id land in a private .env."""
        settings = self.make_settings(telegram=False)
        private = {"id": 555, "type": "private", "first_name": "Example"}
        self.client.updates = [
            # Drained first: a message sent before the setup started must not pick the chat.
            [{"update_id": 3, "message": {"chat": {"id": 111, "type": "private"}, "text": "old"}}],
            [
                {"update_id": 6, "message": {"chat": {"id": -100, "type": "group"}, "text": SETUP_CODE}},
                {"update_id": 7, "message": {"chat": private, "from": {"id": 555}, "text": "hi"}},
                {"update_id": 8, "message": {"chat": private, "from": {"id": 555}, "text": f" {SETUP_CODE} "}},
            ],
        ]
        out = io.StringIO()
        chat_id = telegram_setup(
            settings,
            secret_input=lambda _prompt: TOKEN,
            out=out,
            client_factory=lambda _token: self.client,
            wait_seconds=5,
            confirm=lambda _prompt: "y",
            code_factory=lambda: SETUP_CODE,
        )

        self.assertEqual(chat_id, 555)
        saved = dotenv_values(settings.env_file)
        self.assertEqual(saved["TELEGRAM_BOT_TOKEN"], TOKEN)
        self.assertEqual(saved["TELEGRAM_CHAT_ID"], "555")
        self.assertEqual(stat.S_IMODE(settings.env_file.stat().st_mode), 0o600)
        self.assertIn(f"https://t.me/{FAKE_BOT_USERNAME}", out.getvalue())
        self.assertNotIn(TOKEN, out.getvalue())
        self.assertIn("Autopilot is connected", self.texts()[-1])
        # The setup message is skipped by the service, so the bot does not answer it again.
        self.assertEqual(Store(settings.db_path).get_value(OFFSET_KEY), "9")


class AC13ServiceTests(IntegrationCase):
    """AC13: install-service writes and enables a systemd user unit; uninstall removes it."""

    def test_install_then_uninstall(self) -> None:
        """The unit starts at boot, restarts on failure, keeps PATH, and disappears on uninstall."""
        calls: list[tuple[str, ...]] = []

        def fake_systemctl(args: Any, timeout: float = 20.0) -> tuple[int, str, str]:
            """Record the systemctl/loginctl call and report success with linger on."""
            calls.append(tuple(args))
            return 0, "Linger=yes\n", ""

        unit_dir = self.home / "units"
        steps = install_service(
            python=Path(sys.executable),
            project_home=self.home,
            path_env="/opt/tools/bin:/usr/bin",
            run=fake_systemctl,
            unit_dir=unit_dir,
        )
        unit = (unit_dir / UNIT_NAME).read_text(encoding="utf-8")
        self.assertTrue(steps)
        self.assertIn("-m autopilot start", unit)
        self.assertIn("Restart=on-failure", unit)
        self.assertIn("WantedBy=default.target", unit)
        self.assertIn("PATH=/opt/tools/bin:/usr/bin", unit)
        self.assertIn(("systemctl", "--user", "enable", "--now", UNIT_NAME), calls)

        uninstall_service(run=fake_systemctl, unit_dir=unit_dir)
        self.assertFalse((unit_dir / UNIT_NAME).exists())
        self.assertTrue(any("disable" in call for call in calls))


class AC14SecretsTests(IntegrationCase):
    """AC14: the bot token stays out of logs and errors; secrets and data are git-ignored."""

    def test_token_is_redacted_from_the_log_file(self) -> None:
        """A log line that contains the token is written with *** instead."""
        settings = self.make_settings()
        root = logging.getLogger()
        saved_handlers, saved_level = list(root.handlers), root.level

        def restore() -> None:
            """Remove handlers the test added and put the root log level back."""
            for handler in list(root.handlers):
                if handler not in saved_handlers:
                    root.removeHandler(handler)
                    handler.close()
            root.setLevel(saved_level)

        self.addCleanup(restore)
        with contextlib.redirect_stderr(io.StringIO()) as stderr:
            configure_logging(settings, to_file=True)
            logging.getLogger("autopilot.integration").warning("calling https://api.telegram.org/bot%s/getMe", TOKEN)
            for handler in root.handlers:
                handler.flush()
        written = settings.log_file.read_text(encoding="utf-8")
        self.assertNotIn(TOKEN, written)
        self.assertNotIn(TOKEN, stderr.getvalue())
        self.assertIn("***", written)

    def test_network_error_message_has_no_token_or_url(self) -> None:
        """A failed Bot API call raises an error without the token or request URL."""

        def unreachable(*_args: Any, **_kwargs: Any) -> Any:
            """Stand in for urlopen when DNS fails, so no request leaves the machine."""
            raise urllib.error.URLError(socket.gaierror(-2, "Name or service not known"))

        client = TelegramClient(TOKEN, urlopen=unreachable)
        with self.assertRaises(TelegramError) as caught:
            client.get_me()
        self.assertNotIn(TOKEN, str(caught.exception))
        self.assertNotIn("api.telegram.org", str(caught.exception))

    def test_secret_files_are_git_ignored(self) -> None:
        """.env, tasks.toml and data/ are listed in .gitignore."""
        lines = (PROJECT_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
        for entry in (".env", "tasks.toml", "data/"):
            self.assertIn(entry, lines)


class AC15ChatTests(IntegrationCase):
    """AC15: plain text is a chat turn that resumes the conversation; /new, idle time and errors reset it."""

    def chat(self, app: App, text: str, answers: int) -> None:
        """Send ``text`` and wait until ``answers`` messages were delivered in total."""
        self.send(app, text)
        self.wait_for_messages(answers)

    def test_second_turn_resumes_the_first_session_and_new_drops_it(self) -> None:
        """Turn two passes --resume <first session>; /new starts a conversation without --resume."""
        app = self.build(workers=True)
        self.chat(app, "Příliš žluťoučký kůň", 1)
        self.assertEqual(self.texts()[0], "echo: Příliš žluťoučký kůň")  # chat answers have no header
        self.assertEqual(self.client.chat_actions, [(CHAT_ID, "typing")])
        first_session = app.store.get_value(CHAT_SESSION_KEY)
        self.assertTrue((first_session or "").startswith("fake-session-"))

        self.chat(app, "and again", 2)
        first, second = self.claude_calls()
        self.assertNotIn("--resume", first["argv"])
        self.assertEqual(argument_after(second["argv"], "--resume"), first_session)
        self.assertEqual(second["prompt"], "and again")

        self.send(app, "/new")
        self.assertIn("New conversation", self.texts()[-1])
        self.assertIsNone(app.store.get_value(CHAT_SESSION_KEY))
        self.clock.advance(seconds=1)  # a turn queued at the moment of /new never stores its session
        self.chat(app, "fresh start", 4)
        self.assertNotIn("--resume", self.claude_calls()[2]["argv"])
        self.assertNotEqual(app.store.get_value(CHAT_SESSION_KEY), first_session)

    def test_ask_and_new_with_text_are_chat_turns(self) -> None:
        """/ask <text> continues the conversation; /new <text> resets and then asks."""
        app = self.build(workers=True)
        self.chat(app, "/ask one", 1)
        self.chat(app, "/ask two", 2)
        self.assertIn("--resume", self.claude_calls()[1]["argv"])
        self.chat(app, "/new three", 3)  # no reset confirmation is sent when text follows
        self.assertNotIn("--resume", self.claude_calls()[2]["argv"])
        self.assertEqual({r.task_name for r in app.store.list_runs()}, {ASK_TASK_NAME})

    def test_conversation_idle_for_six_hours_starts_fresh(self) -> None:
        """A turn after more than six quiet hours does not resume the old session."""
        app = self.build(workers=True)
        self.chat(app, "morning", 1)
        self.clock.advance(hours=7)
        self.chat(app, "evening", 2)
        self.assertNotIn("--resume", self.claude_calls()[1]["argv"])

    def test_failed_resumed_turn_resets_the_conversation(self) -> None:
        """An error on a resumed turn tells the user and the next message starts a new conversation."""
        app = self.build(workers=True)
        self.chat(app, "start", 1)
        self.chat(app, "FAKE_ERROR now", 2)
        self.assertIn("conversation was reset", self.texts()[-1])
        self.assertIsNone(app.store.get_value(CHAT_SESSION_KEY))
        self.chat(app, "retry", 3)
        self.assertNotIn("--resume", self.claude_calls()[2]["argv"])

    def test_queue_position_is_reported_instead_of_typing(self) -> None:
        """While another run is queued the bot says how many runs are ahead and sends no typing action."""
        app = self.build()  # no workers: runs stay queued
        self.send(app, "first")
        self.assertEqual(self.client.chat_actions, [(CHAT_ID, "typing")])
        self.send(app, "second")
        self.assertIn("1 run(s) ahead", self.texts()[-1])
        self.assertEqual(len(self.client.chat_actions), 1)

    def test_chat_uses_the_ask_options_from_tasks_toml(self) -> None:
        """The [ask] table sets the workdir and tools for chat turns."""
        workdir = self.home / "chatdir"
        workdir.mkdir()
        app = self.build(
            f'[ask]\nworkdir = "{workdir}"\nallowed_tools = ["Read"]\n' + NEWS_TOML,
            workers=True,
        )
        self.chat(app, "hello", 1)
        (call,) = self.claude_calls()
        self.assertEqual(call["cwd"], str(workdir))
        self.assertEqual(argument_after(call["argv"], "--allowedTools"), "Read")
        self.assertFalse(call["has_api_key"])
        record = app.store.list_runs(limit=1)[0]
        self.assertEqual((record.task_name, record.trigger), (ASK_TASK_NAME, TRIGGER_CHAT))

    def test_one_off_ask_from_the_command_line_does_not_touch_the_conversation(self) -> None:
        """A manual one-off ask runs without --resume and leaves the chat session unset."""
        app = self.build()
        run_id = app.engine.enqueue_prompt("one-off", TRIGGER_MANUAL, dispatch=False)
        record = app.engine.execute_run(run_id)
        assert record is not None
        self.assertEqual(record.status, STATUS_SUCCESS)
        self.assertNotIn("--resume", self.claude_calls()[0]["argv"])
        self.assertIsNone(app.store.get_value(CHAT_SESSION_KEY))


if __name__ == "__main__":
    unittest.main()
