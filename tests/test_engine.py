"""Unit tests for autopilot.engine.

The engine is tested with a scripted fake runner, a recording fake notifier,
a real Store in a temporary folder and a FakeClock. Covered: scheduled
enqueueing and markers, missed slots, paused and disabled tasks, deduplication,
notify policies, the usage-limit gate with retries, removed tasks, runner
crashes, hot reload, chat sessions (resume, idle reset, reset, failed turns,
serialization), workers and the service loop.
"""

from __future__ import annotations

import logging
import os
import tempfile
import threading
import time
import unittest
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

from autopilot.engine import (
    CHAT_IDLE_RESET,
    CHAT_RESET_KEY,
    CHAT_SESSION_KEY,
    CHAT_UPDATED_KEY,
    DEFAULT_RETRY_DELAY,
    LIMITED_UNTIL_KEY,
    TICK_SECONDS,
    Engine,
    LogNotifier,
)
from autopilot.errors import ConfigError, TaskNotFoundError
from autopilot.models import (
    ASK_TASK_NAME,
    NOTIFY_ALWAYS,
    NOTIFY_ERRORS,
    NOTIFY_NEVER,
    STATUS_CANCELLED,
    STATUS_ERROR,
    STATUS_QUEUED,
    STATUS_RATE_LIMITED,
    STATUS_RUNNING,
    STATUS_SUCCESS,
    STATUS_TIMEOUT,
    TRIGGER_CHAT,
    TRIGGER_MANUAL,
    TRIGGER_SCHEDULE,
    TRIGGER_TELEGRAM,
    RunOptions,
    RunOutcome,
    RunRecord,
    TaskConfig,
)
from autopilot.schedule import parse_schedule
from autopilot.store import INTERRUPTED_MESSAGE, Store
from helpers import TZ, FakeClock, make_config, make_options, make_task

START = datetime(2026, 11, 2, 7, 30, tzinfo=TZ)  # Monday 07:30
WAIT_SECONDS = 5.0

# Expected warnings (rejected reloads) would otherwise be printed to stderr by logging's last-resort handler.
logging.getLogger("autopilot.engine").addHandler(logging.NullHandler())


def success(**overrides: object) -> RunOutcome:
    """Build a successful outcome.

    Args:
        **overrides: Any RunOutcome field.

    Returns:
        A RunOutcome with status success and result "done".
    """
    fields: dict[str, object] = {"status": STATUS_SUCCESS, "result": "done", "duration_ms": 12}
    fields.update(overrides)
    return RunOutcome(**fields)


@dataclass
class RunCall:
    """One recorded call of the fake runner."""

    prompt: str
    options: RunOptions
    resume_session: str | None


class ScriptedRunner:
    """Fake Runner that returns queued outcomes and records every call.

    Attributes:
        calls: Every call, in order.
        on_run: Optional hook called inside ``run`` before it answers.
    """

    def __init__(self, *outcomes: RunOutcome | BaseException) -> None:
        """Queue outcomes (or exceptions to raise); an empty queue answers with success.

        Args:
            *outcomes: Items returned or raised by successive calls.
        """
        self._outcomes: deque[RunOutcome | BaseException] = deque(outcomes)
        self.calls: list[RunCall] = []
        self.on_run: Callable[[], None] | None = None

    def queue(self, *outcomes: RunOutcome | BaseException) -> None:
        """Add more scripted outcomes.

        Args:
            *outcomes: Items returned or raised by later calls.
        """
        self._outcomes.extend(outcomes)

    def run(self, prompt: str, options: RunOptions, resume_session: str | None = None) -> RunOutcome:
        """Record the call, run the hook and answer from the script.

        Args:
            prompt: Prompt text.
            options: Run options.
            resume_session: Session id to resume.

        Returns:
            The next scripted outcome.

        Raises:
            BaseException: When the next scripted item is an exception.
        """
        self.calls.append(RunCall(prompt, options, resume_session))
        if self.on_run is not None:
            self.on_run()
        item = self._outcomes.popleft() if self._outcomes else success()
        if isinstance(item, BaseException):
            raise item
        return item


class RecordingNotifier:
    """Fake Notifier that records finished runs and notices.

    Attributes:
        runs: Records passed to notify_run.
        texts: Strings passed to notify_text.
        fail: When True both methods raise RuntimeError after recording.
    """

    def __init__(self) -> None:
        """Start with nothing recorded."""
        self.runs: list[RunRecord] = []
        self.texts: list[str] = []
        self.fail = False

    def notify_run(self, record: RunRecord) -> None:
        """Record a finished run.

        Args:
            record: The run.

        Raises:
            RuntimeError: When ``fail`` is set.
        """
        self.runs.append(record)
        if self.fail:
            raise RuntimeError("telegram is down")

    def notify_text(self, text: str) -> None:
        """Record a notice.

        Args:
            text: The notice.

        Raises:
            RuntimeError: When ``fail`` is set.
        """
        self.texts.append(text)
        if self.fail:
            raise RuntimeError("telegram is down")


class RecordingStop:
    """Stand-in for threading.Event that records wait() timeouts and stops after N waits."""

    def __init__(self, stop_after_waits: int = 1) -> None:
        """Create the stop object.

        Args:
            stop_after_waits: How many wait() calls happen before is_set() turns True.
        """
        self.waits: list[float] = []
        self._stop_after = stop_after_waits

    def is_set(self) -> bool:
        """Tell whether the loop should end.

        Returns:
            True once enough waits were recorded.
        """
        return len(self.waits) >= self._stop_after

    def wait(self, timeout: float) -> bool:
        """Record the requested sleep instead of sleeping.

        Args:
            timeout: Seconds the loop wants to sleep.

        Returns:
            The current is_set() value.
        """
        self.waits.append(timeout)
        return self.is_set()


def wait_until(condition: Callable[[], bool], timeout: float = WAIT_SECONDS) -> bool:
    """Poll a condition until it holds or the timeout passes.

    Args:
        condition: Callable returning True when satisfied.
        timeout: Seconds to wait at most.

    Returns:
        The final value of the condition.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.01)
    return condition()


class EngineTestCase(unittest.TestCase):
    """Base class: a real Store, fake runner, fake notifier and an engine on a fixed clock."""

    def setUp(self) -> None:
        """Create the temp folder, store, collaborators and an engine with one daily task."""
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)
        self.store = Store(self.tmp / "data" / "autopilot.db")
        self.clock = FakeClock(START)
        self.runner = ScriptedRunner()
        self.notifier = RecordingNotifier()
        self.ask_options = make_options(model="ask-model")
        self.news = make_task("news", "Say news", (parse_schedule("daily 07:30"),))
        self.current_config: TaskConfig = make_config(self.news, ask_options=self.ask_options)
        self.engine = self.make_engine()
        self.engine.load_config()

    def make_engine(self, **overrides: object) -> Engine:
        """Build an engine wired to this test's collaborators.

        Args:
            **overrides: Engine keyword arguments replacing the defaults.

        Returns:
            A new Engine whose config loader returns ``self.current_config``.
        """
        arguments: dict[str, object] = {
            "store": self.store,
            "runner": self.runner,
            "notifier": self.notifier,
            "config_loader": lambda: self.current_config,
            "clock": self.clock,
        }
        arguments.update(overrides)
        return Engine(**arguments)

    def set_tasks(self, *tasks: object) -> None:
        """Replace the configuration the loader returns and apply it to the engine.

        Args:
            *tasks: Task objects for the new configuration.
        """
        self.current_config = make_config(*tasks, ask_options=self.ask_options)
        self.engine.load_config()

    def run_now(self, run_id: int | None) -> RunRecord:
        """Execute a run synchronously and return its finished record.

        Args:
            run_id: Id from an enqueue call.

        Returns:
            The record returned by execute_run.
        """
        assert run_id is not None
        record = self.engine.execute_run(run_id)
        assert record is not None
        return record

    def chat(self, text: str = "hello") -> RunRecord:
        """Queue and execute one chat turn without workers.

        Args:
            text: The chat message.

        Returns:
            The finished record.
        """
        return self.run_now(self.engine.enqueue_prompt(text, dispatch=False))

    def run_news(self) -> RunRecord:
        """Queue and execute the "news" task without workers.

        Returns:
            The finished record.
        """
        return self.run_now(self.engine.enqueue_task("news", TRIGGER_MANUAL, dispatch=False))


class ConfigTests(EngineTestCase):
    """Loading and reading the configuration."""

    def test_config_before_load_raises(self) -> None:
        """Reading the config before load_config is a programming error."""
        engine = self.make_engine()
        with self.assertRaises(RuntimeError):
            engine.config

    def test_load_config_returns_and_stores_config(self) -> None:
        """load_config returns the loaded config and makes it current."""
        engine = self.make_engine()
        self.assertIs(engine.load_config(), self.current_config)
        self.assertIs(engine.config, self.current_config)

    def test_load_config_error_propagates(self) -> None:
        """A broken config at startup fails loudly."""
        def broken() -> TaskConfig:
            """Fail like a loader that meets an invalid tasks.toml."""
            raise ConfigError("tasks.toml: nope")

        engine = self.make_engine(config_loader=broken)
        with self.assertRaises(ConfigError):
            engine.load_config()


class EnqueueTests(EngineTestCase):
    """enqueue_task, enqueue_prompt and reset_chat."""

    def test_enqueue_task_creates_queued_run_with_current_prompt(self) -> None:
        """A run is stored as queued with the task's prompt, trigger and slot."""
        slot = START - timedelta(minutes=1)
        run_id = self.engine.enqueue_task("news", TRIGGER_TELEGRAM, scheduled_for=slot, dispatch=False)
        record = self.store.get_run(run_id)
        self.assertEqual(record.status, STATUS_QUEUED)
        self.assertEqual((record.task_name, record.prompt), ("news", "Say news"))
        self.assertEqual(record.trigger, TRIGGER_TELEGRAM)
        self.assertEqual(record.scheduled_for, slot)
        self.assertEqual(record.queued_at, START)

    def test_enqueue_unknown_task_raises(self) -> None:
        """An unknown name raises TaskNotFoundError listing known tasks."""
        with self.assertRaises(TaskNotFoundError) as caught:
            self.engine.enqueue_task("ghost", TRIGGER_MANUAL)
        self.assertIn("news", str(caught.exception))

    def test_active_run_blocks_a_second_run_of_the_same_task(self) -> None:
        """A task is never queued twice; once finished it can be queued again."""
        first = self.engine.enqueue_task("news", TRIGGER_MANUAL, dispatch=False)
        self.assertIsNone(self.engine.enqueue_task("news", TRIGGER_TELEGRAM, dispatch=False))
        self.assertEqual(len(self.store.active_runs()), 1)
        self.run_now(first)
        self.assertIsNotNone(self.engine.enqueue_task("news", TRIGGER_MANUAL, dispatch=False))

    def test_enqueue_prompt_creates_ask_run(self) -> None:
        """A chat prompt is stored under the ask name with the chat trigger by default."""
        run_id = self.engine.enqueue_prompt("  What is 2+2?  ", dispatch=False)
        record = self.store.get_run(run_id)
        self.assertEqual((record.task_name, record.trigger), (ASK_TASK_NAME, TRIGGER_CHAT))
        self.assertEqual(record.prompt, "What is 2+2?")

    def test_enqueue_prompt_manual_trigger_and_no_dedupe(self) -> None:
        """One-off asks keep the manual trigger and can be queued repeatedly."""
        first = self.engine.enqueue_prompt("a", TRIGGER_MANUAL, dispatch=False)
        second = self.engine.enqueue_prompt("a", TRIGGER_MANUAL, dispatch=False)
        self.assertNotEqual(first, second)
        self.assertEqual(self.store.get_run(first).trigger, TRIGGER_MANUAL)

    def test_enqueue_prompt_rejects_empty_text(self) -> None:
        """Empty or whitespace-only prompts raise ValueError."""
        for text in ("", "   \n"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                self.engine.enqueue_prompt(text)

    def test_reset_chat_deletes_session_keys(self) -> None:
        """reset_chat removes the stored session and its timestamp."""
        self.store.set_value(CHAT_SESSION_KEY, "s1")
        self.store.set_value(CHAT_UPDATED_KEY, START.isoformat())
        self.engine.reset_chat()
        self.assertIsNone(self.store.get_value(CHAT_SESSION_KEY))
        self.assertIsNone(self.store.get_value(CHAT_UPDATED_KEY))


class TickTests(EngineTestCase):
    """Scheduled enqueueing, markers, skips and per-task isolation."""

    def test_new_task_gets_a_marker_but_does_not_run(self) -> None:
        """The first tick only records the starting marker; the first run waits for a slot."""
        self.assertEqual(self.engine.tick(), [])
        self.assertEqual(self.store.get_last_scheduled("news"), START)
        self.assertEqual(self.store.active_runs(), [])

    def test_due_slot_is_enqueued_once_and_marked(self) -> None:
        """A slot that has passed creates one scheduled run and moves the marker."""
        self.store.set_last_scheduled("news", START - timedelta(days=1))
        self.clock.advance(minutes=1)
        ids = self.engine.tick()
        self.assertEqual(len(ids), 1)
        record = self.store.get_run(ids[0])
        self.assertEqual((record.trigger, record.status), (TRIGGER_SCHEDULE, STATUS_QUEUED))
        self.assertEqual(record.scheduled_for, START)
        self.assertEqual(self.store.get_last_scheduled("news"), START)
        self.assertEqual(self.engine.tick(), [])

    def test_missed_slot_is_skipped_and_logged(self) -> None:
        """A slot older than the grace is not run, the marker moves on and the skip is logged."""
        self.store.set_last_scheduled("news", START - timedelta(days=1))
        self.clock.advance(minutes=10)
        with self.assertLogs("autopilot.engine", level="INFO") as logs:
            ids = self.engine.tick()
        self.assertEqual(ids, [])
        self.assertEqual(self.store.get_last_scheduled("news"), START)
        self.assertTrue(any("missed slot" in line and "news" in line for line in logs.output))

    def test_paused_task_skips_slot_without_catch_up(self) -> None:
        """Paused tasks drop their slots; resuming does not replay them."""
        self.store.set_last_scheduled("news", START - timedelta(days=1))
        self.store.set_paused("news", True)
        self.clock.advance(minutes=1)
        self.assertEqual(self.engine.tick(), [])
        self.assertEqual(self.store.get_last_scheduled("news"), START)
        self.store.set_paused("news", False)
        self.clock.advance(minutes=1)
        self.assertEqual(self.engine.tick(), [])

    def test_disabled_task_skips_slot(self) -> None:
        """Disabled tasks do not run on schedule."""
        self.set_tasks(make_task("news", "Say news", (parse_schedule("daily 07:30"),), enabled=False))
        self.store.set_last_scheduled("news", START - timedelta(days=1))
        self.clock.advance(minutes=1)
        self.assertEqual(self.engine.tick(), [])
        self.assertEqual(self.store.get_last_scheduled("news"), START)

    def test_active_run_deduplicates_scheduled_slot(self) -> None:
        """A due slot is dropped when the task already has an active run."""
        self.engine.enqueue_task("news", TRIGGER_MANUAL, dispatch=False)
        self.store.set_last_scheduled("news", START - timedelta(days=1))
        self.clock.advance(minutes=1)
        self.assertEqual(self.engine.tick(), [])
        self.assertEqual(len(self.store.active_runs()), 1)

    def test_slot_across_dst_change_uses_store_fixed_offset_marker(self) -> None:
        """The store returns fixed-offset markers; the next slot is still found in local time."""
        self.clock.set(datetime(2026, 10, 26, 7, 31, tzinfo=TZ))  # after clocks went back
        self.store.set_last_scheduled("news", datetime(2026, 10, 25, 7, 30, tzinfo=TZ))
        ids = self.engine.tick()
        self.assertEqual(len(ids), 1)
        self.assertEqual(self.store.get_run(ids[0]).scheduled_for, datetime(2026, 10, 26, 7, 30, tzinfo=TZ))

    def test_error_in_one_task_does_not_stop_the_others(self) -> None:
        """An exception for one task is logged and the remaining tasks still run."""
        second = make_task("plan", "Plan", (parse_schedule("daily 07:30"),))
        self.set_tasks(self.news, second)
        for name in ("news", "plan"):
            self.store.set_last_scheduled(name, START - timedelta(days=1))
        self.clock.advance(minutes=1)
        real = self.store.is_paused

        def flaky(name: str) -> bool:
            """Raise for the "news" task only, like a database hiccup mid-tick."""
            if name == "news":
                raise RuntimeError("db hiccup")
            return real(name)

        with mock.patch.object(self.store, "is_paused", side_effect=flaky):
            with self.assertLogs("autopilot.engine", level="ERROR"):
                ids = self.engine.tick()
        self.assertEqual([self.store.get_run(i).task_name for i in ids], ["plan"])

    def test_tick_reloads_changed_tasks_file(self) -> None:
        """tick applies a new tasks.toml before evaluating schedules."""
        path = self.tmp / "tasks.toml"
        path.write_text("v1")
        engine = self.make_engine(tasks_file=path)
        engine.load_config()
        replacement = make_task("plan", "Plan", ())
        self.current_config = make_config(replacement, ask_options=self.ask_options)
        bump_mtime(path)
        engine.tick()
        self.assertEqual(list(engine.config.tasks), ["plan"])


def bump_mtime(path: Path) -> None:
    """Move a file's modification time forward, so reload detection never depends on timer resolution.

    Args:
        path: The file to touch.
    """
    stat = path.stat()
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 2_000_000_000))


class ReloadTests(EngineTestCase):
    """Hot reload of tasks.toml."""

    def setUp(self) -> None:
        """Add a watched tasks file and an engine that watches it."""
        super().setUp()
        self.path = self.tmp / "tasks.toml"
        self.path.write_text("v1")
        self.loader_result: TaskConfig | ConfigError = self.current_config
        self.engine = self.make_engine(tasks_file=self.path, config_loader=self.load_from_script)
        self.engine.load_config()

    def load_from_script(self) -> TaskConfig:
        """Return or raise whatever the test scripted.

        Returns:
            The scripted TaskConfig.

        Raises:
            ConfigError: When the test scripted an error.
        """
        if isinstance(self.loader_result, ConfigError):
            raise self.loader_result
        return self.loader_result

    def test_unchanged_file_is_not_reloaded(self) -> None:
        """Without a new mtime the loader is not consulted and False is returned."""
        self.loader_result = make_config(make_task("other"))
        self.assertFalse(self.engine.reload_if_changed())
        self.assertIn("news", self.engine.config.tasks)

    def test_changed_file_applies_new_config(self) -> None:
        """A new mtime and a valid file swap the config and return True."""
        self.loader_result = make_config(make_task("other"))
        bump_mtime(self.path)
        with self.assertLogs("autopilot.engine", level="INFO") as logs:
            self.assertTrue(self.engine.reload_if_changed())
        self.assertEqual(list(self.engine.config.tasks), ["other"])
        self.assertTrue(any("Reloaded 1 tasks" in line for line in logs.output))
        self.assertEqual(self.notifier.texts, [])

    def test_no_tasks_file_never_reloads(self) -> None:
        """An engine without a watched file ignores reloads."""
        self.assertFalse(self.make_engine().reload_if_changed())

    def test_broken_edit_keeps_old_tasks_and_warns_once(self) -> None:
        """An invalid edit keeps the previous tasks and sends one warning per distinct message."""
        self.loader_result = ConfigError("tasks.toml: task 'x': bad")
        bump_mtime(self.path)
        self.assertFalse(self.engine.reload_if_changed())
        self.assertIn("news", self.engine.config.tasks)
        self.assertEqual(
            self.notifier.texts,
            ["⚠️ tasks.toml has an error, keeping the previous tasks:\ntasks.toml: task 'x': bad"],
        )
        # The same file is not parsed or announced again.
        self.assertFalse(self.engine.reload_if_changed())
        # Saving again with the same error stays quiet.
        bump_mtime(self.path)
        self.assertFalse(self.engine.reload_if_changed())
        self.assertEqual(len(self.notifier.texts), 1)
        # A different error is announced.
        self.loader_result = ConfigError("tasks.toml: other problem")
        bump_mtime(self.path)
        self.engine.reload_if_changed()
        self.assertEqual(len(self.notifier.texts), 2)

    def test_error_is_announced_again_after_a_successful_reload(self) -> None:
        """After a fix, the same error appearing again later is announced again."""
        error = ConfigError("tasks.toml: bad")
        self.loader_result = error
        bump_mtime(self.path)
        self.engine.reload_if_changed()
        self.loader_result = make_config(make_task("news"))
        bump_mtime(self.path)
        self.assertTrue(self.engine.reload_if_changed())
        self.loader_result = error
        bump_mtime(self.path)
        self.engine.reload_if_changed()
        self.assertEqual(len(self.notifier.texts), 2)

    def test_notifier_failure_while_warning_is_swallowed(self) -> None:
        """A failing notifier cannot break the reload check."""
        self.notifier.fail = True
        self.loader_result = ConfigError("tasks.toml: bad")
        bump_mtime(self.path)
        with self.assertLogs("autopilot.engine", level="ERROR"):
            self.assertFalse(self.engine.reload_if_changed())

    def test_deleted_file_is_reported_as_error(self) -> None:
        """A removed tasks file goes through the loader's ConfigError path."""
        self.path.unlink()
        self.loader_result = ConfigError("tasks.toml not found")
        self.assertFalse(self.engine.reload_if_changed())
        self.assertEqual(len(self.notifier.texts), 1)


class NotifyPolicyTests(EngineTestCase):
    """Which finished runs reach the notifier."""

    def test_task_notify_policies(self) -> None:
        """always notifies everything, errors only failures, never nothing."""
        cases = [
            (NOTIFY_ALWAYS, STATUS_SUCCESS, True),
            (NOTIFY_ALWAYS, STATUS_ERROR, True),
            (NOTIFY_ERRORS, STATUS_SUCCESS, False),
            (NOTIFY_ERRORS, STATUS_ERROR, True),
            (NOTIFY_ERRORS, STATUS_TIMEOUT, True),
            (NOTIFY_NEVER, STATUS_SUCCESS, False),
            (NOTIFY_NEVER, STATUS_ERROR, False),
        ]
        for policy, status, expected in cases:
            with self.subTest(policy=policy, status=status):
                self.notifier.runs.clear()
                self.set_tasks(make_task("news", "p", (), notify=policy))
                self.runner.queue(RunOutcome(status=status, error=None if status == STATUS_SUCCESS else "boom"))
                self.run_news()
                self.assertEqual(len(self.notifier.runs), 1 if expected else 0)

    def test_ask_runs_always_notify(self) -> None:
        """Chat and one-off ask runs notify even though they have no policy."""
        self.chat("hi")
        self.run_now(self.engine.enqueue_prompt("one-off", TRIGGER_MANUAL, dispatch=False))
        self.assertEqual([r.trigger for r in self.notifier.runs], [TRIGGER_CHAT, TRIGGER_MANUAL])

    def test_notifier_exception_is_logged_not_raised(self) -> None:
        """A failing notifier does not affect the stored run."""
        self.notifier.fail = True
        with self.assertLogs("autopilot.engine", level="ERROR"):
            record = self.run_news()
        self.assertEqual(self.store.get_run(record.id).status, STATUS_SUCCESS)

    def test_notified_record_is_the_finished_record(self) -> None:
        """The notifier receives the stored, finished record with its result."""
        self.runner.queue(success(result="hello", denied_tools=("Bash",), session_id="s", num_turns=2))
        self.run_news()
        sent = self.notifier.runs[0]
        self.assertEqual((sent.status, sent.result, sent.denied_tools), (STATUS_SUCCESS, "hello", ("Bash",)))


class ExecuteTests(EngineTestCase):
    """execute_run: the plain success path, options and error handling."""

    def test_success_is_recorded_and_runner_gets_task_prompt_and_options(self) -> None:
        """The runner is called with the stored prompt and the task's options, without a session."""
        self.runner.queue(success(result="out", session_id="abc", num_turns=3, cost_usd=0.5, duration_ms=99))
        self.clock.advance(minutes=2)
        record = self.run_news()
        call = self.runner.calls[0]
        self.assertEqual((call.prompt, call.options, call.resume_session), ("Say news", self.news.options, None))
        self.assertEqual(record.status, STATUS_SUCCESS)
        self.assertEqual((record.result, record.session_id, record.num_turns), ("out", "abc", 3))
        self.assertEqual((record.cost_usd, record.duration_ms), (0.5, 99))
        self.assertEqual(record.started_at, self.clock())
        self.assertEqual(record.finished_at, self.clock())

    def test_unknown_run_returns_none(self) -> None:
        """An unknown id is ignored."""
        self.assertIsNone(self.engine.execute_run(12345))

    def test_run_that_is_not_queued_is_ignored(self) -> None:
        """Executing a finished run again returns it unchanged without calling the runner."""
        record = self.run_news()
        again = self.engine.execute_run(record.id)
        self.assertEqual(again.status, STATUS_SUCCESS)
        self.assertEqual(len(self.runner.calls), 1)

    def test_run_is_marked_running_while_the_runner_works(self) -> None:
        """During the Claude call the stored status is running and shows up in the snapshot."""
        seen: list[tuple[int, int]] = []

        def peek() -> None:
            """Record the queue snapshot taken while the runner is busy."""
            snap = self.engine.snapshot()
            seen.append((len(snap.running), len(snap.queued)))

        self.runner.on_run = peek
        self.run_news()
        self.assertEqual(seen, [(1, 0)])

    def test_runner_exception_becomes_error_run(self) -> None:
        """An unexpected runner exception is recorded as an error with its message and notified."""
        self.runner.queue(RuntimeError("kaboom"))
        with self.assertLogs("autopilot.engine", level="ERROR"):
            record = self.run_news()
        self.assertEqual((record.status, record.error), (STATUS_ERROR, "kaboom"))
        self.assertEqual(len(self.notifier.runs), 1)

    def test_runner_exception_without_message_uses_type_name(self) -> None:
        """An exception with an empty message still yields a readable error."""
        self.runner.queue(KeyError())
        with self.assertLogs("autopilot.engine", level="ERROR"):
            record = self.run_news()
        self.assertEqual(record.error, "KeyError")

    def test_task_options_come_from_the_current_config(self) -> None:
        """Options are looked up at execution time, so edits apply to queued runs."""
        run_id = self.engine.enqueue_task("news", TRIGGER_MANUAL, dispatch=False)
        edited = make_task("news", "Say news", (), options=make_options(model="new-model"))
        self.set_tasks(edited)
        self.run_now(run_id)
        self.assertEqual(self.runner.calls[0].options.model, "new-model")

    def test_removed_task_is_cancelled_without_calling_claude_or_notifying(self) -> None:
        """A run whose task vanished from tasks.toml is cancelled silently."""
        run_id = self.engine.enqueue_task("news", TRIGGER_MANUAL, dispatch=False)
        self.set_tasks(make_task("other"))
        record = self.run_now(run_id)
        self.assertEqual((record.status, record.error), (STATUS_CANCELLED, "Task was removed from tasks.toml."))
        self.assertEqual(self.runner.calls, [])
        self.assertEqual((self.notifier.runs, self.notifier.texts), ([], []))


class RateLimitTests(EngineTestCase):
    """The usage-limit gate, retries and giving up."""

    def limit(self, retry_at: datetime | None) -> RunOutcome:
        """Build a rate-limited outcome.

        Args:
            retry_at: Reset time reported by Claude, or None when unknown.

        Returns:
            A rate_limited RunOutcome.
        """
        return RunOutcome(status=STATUS_RATE_LIMITED, error="You've hit your limit", retry_at=retry_at)

    def test_limit_stores_retry_closes_gate_and_notifies_once(self) -> None:
        """A limited run keeps its retry time, sets the gate and sends one notice, no result message."""
        reset = START + timedelta(hours=1)
        self.runner.queue(self.limit(reset))
        record = self.run_news()
        self.assertEqual((record.status, record.retry_at), (STATUS_RATE_LIMITED, reset))
        self.assertEqual(self.store.get_value(LIMITED_UNTIL_KEY), reset.isoformat())
        self.assertEqual(
            self.notifier.texts,
            ["⏳ Claude usage limit reached. All runs wait until 08:30, then retry automatically."],
        )
        self.assertEqual(self.notifier.runs, [])

    def test_notice_shows_local_time_for_a_utc_reset(self) -> None:
        """A reset time given in UTC is shown in the clock's zone (CET, +1 in November)."""
        reset = datetime(2026, 11, 2, 7, 30, tzinfo=timezone.utc)
        self.runner.queue(self.limit(reset))
        self.run_news()
        self.assertIn("08:30", self.notifier.texts[0])

    def test_unknown_reset_defaults_to_one_hour(self) -> None:
        """Without a reset time the retry waits DEFAULT_RETRY_DELAY."""
        self.runner.queue(self.limit(None))
        record = self.run_news()
        self.assertEqual(record.retry_at, START + DEFAULT_RETRY_DELAY)

    def test_closed_gate_defers_later_runs_without_calling_claude(self) -> None:
        """While the gate is closed, runs are deferred until the reset and nobody is notified again."""
        reset = START + timedelta(hours=1)
        self.runner.queue(self.limit(reset))
        self.run_news()
        self.set_tasks(self.news, make_task("plan", "Plan", ()))
        deferred = self.run_now(self.engine.enqueue_task("plan", TRIGGER_MANUAL, dispatch=False))
        chat = self.chat("hello")
        self.assertEqual(len(self.runner.calls), 1)
        for record in (deferred, chat):
            self.assertEqual((record.status, record.retry_at), (STATUS_RATE_LIMITED, reset))
            self.assertEqual(record.error, "Waiting for the Claude usage limit to reset at 08:30.")
        self.assertEqual(len(self.notifier.texts), 1)
        self.assertEqual(self.notifier.runs, [])

    def test_snapshot_reports_the_gate_until_it_opens(self) -> None:
        """snapshot.limited_until is set while the limit holds and None afterwards."""
        reset = START + timedelta(hours=1)
        self.runner.queue(self.limit(reset))
        self.run_news()
        self.assertEqual(self.engine.snapshot().limited_until, reset)
        self.clock.advance(hours=1, minutes=1)
        self.assertIsNone(self.engine.snapshot().limited_until)

    def test_retries_are_queued_after_the_reset_and_call_claude_again(self) -> None:
        """After the reset every deferred run is retried once, linked to its original."""
        reset = START + timedelta(hours=1)
        self.runner.queue(self.limit(reset))
        first = self.run_news()
        self.set_tasks(self.news, make_task("plan", "Plan", ()))
        second = self.run_now(self.engine.enqueue_task("plan", TRIGGER_MANUAL, dispatch=False))
        self.assertEqual(self.engine.tick(), [])  # not yet
        self.clock.advance(hours=1, minutes=1)
        ids = self.engine.tick()
        self.assertEqual(len(ids), 2)
        retries = [self.store.get_run(i) for i in ids]
        self.assertEqual([r.retry_of for r in retries], [first.id, second.id])
        self.assertEqual([(r.task_name, r.prompt, r.trigger) for r in retries],
                         [("news", "Say news", TRIGGER_MANUAL), ("plan", "Plan", TRIGGER_MANUAL)])
        self.assertEqual(self.engine.tick(), [])  # retried exactly once
        self.assertEqual(self.store.due_retries(self.clock()), [])
        done = self.run_now(ids[0])
        self.assertEqual(done.status, STATUS_SUCCESS)
        self.assertEqual(len(self.runner.calls), 2)

    def test_retry_of_removed_task_is_dropped(self) -> None:
        """A retry whose task no longer exists is cleared and not queued."""
        self.runner.queue(self.limit(START + timedelta(minutes=10)))
        self.run_news()
        self.set_tasks(make_task("other"))
        self.clock.advance(minutes=11)
        self.assertEqual(self.engine.tick(), [])
        self.assertEqual(self.store.due_retries(self.clock()), [])

    def test_retry_of_chat_turn_stays_a_chat_turn(self) -> None:
        """Ask-name retries do not need a task and keep the chat trigger."""
        self.runner.queue(self.limit(START + timedelta(minutes=10)))
        original = self.chat("hello")
        self.clock.advance(minutes=11)
        (retry_id,) = self.engine.tick()
        retry = self.store.get_run(retry_id)
        self.assertEqual((retry.task_name, retry.trigger, retry.retry_of), (ASK_TASK_NAME, TRIGGER_CHAT, original.id))

    def test_retry_is_skipped_when_the_task_already_has_an_active_run(self) -> None:
        """A retry never duplicates an active run of the same task."""
        self.runner.queue(self.limit(START + timedelta(minutes=10)))
        self.run_news()
        self.clock.advance(minutes=11)
        self.engine.enqueue_task("news", TRIGGER_MANUAL, dispatch=False)
        self.assertEqual(self.engine.tick(), [])
        self.assertEqual(len(self.store.active_runs()), 1)

    def test_gives_up_after_three_retries(self) -> None:
        """The fourth limited run in a chain is stored without retry, with a final error, and notified."""
        original = self.store.create_run(task_name="news", trigger=TRIGGER_MANUAL, prompt="p", queued_at=START)
        previous = original
        for _ in range(3):
            self.store.mark_running(previous, START)
            self.store.finish_run(previous, status=STATUS_RATE_LIMITED, finished_at=START)
            previous = self.store.create_run(
                task_name="news", trigger=TRIGGER_MANUAL, prompt="p", queued_at=START, retry_of=previous
            )
        self.runner.queue(self.limit(START + timedelta(hours=1)))
        record = self.run_now(previous)
        self.assertEqual(self.store.retry_depth(previous), 3)
        self.assertEqual(record.status, STATUS_RATE_LIMITED)
        self.assertIsNone(record.retry_at)
        self.assertEqual(record.error, "Gave up after 3 retries because of the usage limit.")
        self.assertEqual(len(self.notifier.runs), 1)
        self.assertEqual(self.notifier.texts, [])
        self.assertIsNone(self.store.get_value(LIMITED_UNTIL_KEY))
        self.clock.advance(days=1)
        self.assertEqual(self.store.due_retries(self.clock()), [])

    def test_gate_deferrals_do_not_use_up_the_retries(self) -> None:
        """Links created by gate deferrals never called Claude, so they don't count towards giving up."""
        previous = self.store.create_run(task_name="news", trigger=TRIGGER_MANUAL, prompt="p", queued_at=START)
        # One real attempt followed by four deferrals: depth 5, but only one started run.
        self.store.mark_running(previous, START)
        for _ in range(5):
            self.store.finish_run(previous, status=STATUS_RATE_LIMITED, finished_at=START)
            previous = self.store.create_run(
                task_name="news", trigger=TRIGGER_MANUAL, prompt="p", queued_at=START, retry_of=previous
            )
        self.runner.queue(self.limit(START + timedelta(hours=1)))
        record = self.run_now(previous)
        self.assertEqual(self.store.retry_depth(previous), 5)
        self.assertEqual(len(self.runner.calls), 1)
        self.assertIsNotNone(record.retry_at)
        self.assertNotIn("Gave up", record.error or "")

    def test_deferred_run_in_a_long_chain_still_waits_instead_of_giving_up(self) -> None:
        """A run behind a closed gate is deferred without Claude and never gives up."""
        self.store.set_value(LIMITED_UNTIL_KEY, (START + timedelta(hours=1)).isoformat())
        previous = self.store.create_run(task_name="news", trigger=TRIGGER_MANUAL, prompt="p", queued_at=START)
        for _ in range(4):
            self.store.finish_run(previous, status=STATUS_RATE_LIMITED, finished_at=START)
            previous = self.store.create_run(
                task_name="news", trigger=TRIGGER_MANUAL, prompt="p", queued_at=START, retry_of=previous
            )
        record = self.run_now(previous)
        self.assertEqual(self.runner.calls, [])
        self.assertEqual(record.retry_at, START + timedelta(hours=1))

    def test_third_retry_is_still_allowed(self) -> None:
        """A run at retry depth 2 is limited again and still schedules another retry."""
        previous = self.store.create_run(task_name="news", trigger=TRIGGER_MANUAL, prompt="p", queued_at=START)
        for _ in range(2):
            self.store.mark_running(previous, START)
            self.store.finish_run(previous, status=STATUS_RATE_LIMITED, finished_at=START)
            previous = self.store.create_run(
                task_name="news", trigger=TRIGGER_MANUAL, prompt="p", queued_at=START, retry_of=previous
            )
        self.runner.queue(self.limit(START + timedelta(hours=1)))
        record = self.run_now(previous)
        self.assertIsNotNone(record.retry_at)

    def test_expired_gate_lets_runs_through(self) -> None:
        """Once the reset time has passed the next run calls Claude normally."""
        self.store.set_value(LIMITED_UNTIL_KEY, (START - timedelta(minutes=1)).isoformat())
        self.assertEqual(self.run_news().status, STATUS_SUCCESS)

    def test_unreadable_gate_value_is_ignored(self) -> None:
        """A corrupt limited_until value does not block runs forever."""
        self.store.set_value(LIMITED_UNTIL_KEY, "garbage")
        with self.assertLogs("autopilot.engine", level="WARNING"):
            self.assertEqual(self.run_news().status, STATUS_SUCCESS)

    def test_gate_survives_a_restart(self) -> None:
        """A new Engine on the same store still honours the persisted gate."""
        self.runner.queue(self.limit(START + timedelta(hours=1)))
        self.run_news()
        restarted = self.make_engine()
        restarted.load_config()
        run_id = restarted.enqueue_task("news", TRIGGER_MANUAL, dispatch=False)
        record = restarted.execute_run(run_id)
        self.assertEqual(record.status, STATUS_RATE_LIMITED)
        self.assertEqual(len(self.runner.calls), 1)


class ChatTests(EngineTestCase):
    """Chat turns: session resume, idle reset, reset_chat, failures and serialization."""

    def test_first_turn_starts_a_session_and_second_turn_resumes_it(self) -> None:
        """Turn 1 has no session; its session id is stored and resumed by turn 2 with the ask options."""
        self.runner.queue(success(session_id="s1"), success(session_id="s1"))
        self.chat("one")
        self.chat("two")
        first, second = self.runner.calls
        self.assertIsNone(first.resume_session)
        self.assertEqual(second.resume_session, "s1")
        self.assertEqual(second.options, self.ask_options)
        self.assertEqual(self.store.get_value(CHAT_SESSION_KEY), "s1")
        self.assertEqual(self.store.get_value(CHAT_UPDATED_KEY), START.isoformat())

    def test_stored_session_follows_the_latest_session_id(self) -> None:
        """If Claude answers with a different session id, that one is used next."""
        self.runner.queue(success(session_id="s1"), success(session_id="s2"))
        self.chat()
        self.chat()
        self.chat()
        self.assertEqual(self.runner.calls[2].resume_session, "s2")

    def test_idle_conversation_starts_fresh_after_six_hours(self) -> None:
        """A session idle for more than 6 hours is discarded before the turn."""
        self.runner.queue(success(session_id="s1"), success())
        self.chat()
        self.clock.advance(hours=6, minutes=1)
        self.chat()
        self.assertIsNone(self.runner.calls[1].resume_session)
        self.assertIsNone(self.store.get_value(CHAT_SESSION_KEY))

    def test_conversation_exactly_at_the_idle_limit_is_resumed(self) -> None:
        """Idle for exactly CHAT_IDLE_RESET still counts as within the window."""
        self.runner.queue(success(session_id="s1"))
        self.chat()
        self.clock.advance(seconds=CHAT_IDLE_RESET.total_seconds())
        self.chat()
        self.assertEqual(self.runner.calls[1].resume_session, "s1")

    def test_activity_extends_the_idle_window(self) -> None:
        """Each successful turn refreshes the timestamp, so a long chat never expires mid-conversation."""
        self.runner.queue(success(session_id="s1"), success(session_id="s1"))
        for _ in range(3):
            self.chat()
            self.clock.advance(hours=5)
        self.assertEqual([c.resume_session for c in self.runner.calls], [None, "s1", "s1"])

    def test_session_with_unreadable_timestamp_is_dropped(self) -> None:
        """A stored session without a valid timestamp is not resumed."""
        self.store.set_value(CHAT_SESSION_KEY, "s1")
        self.store.set_value(CHAT_UPDATED_KEY, "garbage")
        self.chat()
        self.assertIsNone(self.runner.calls[0].resume_session)

    def test_idle_reset_value_is_six_hours(self) -> None:
        """CHAT_IDLE_RESET follows the shared CHAT_IDLE_HOURS constant (6 hours)."""
        self.assertEqual(CHAT_IDLE_RESET, timedelta(hours=6))

    def test_reset_chat_records_the_reset_time(self) -> None:
        """reset_chat stores chat_reset_at = now next to clearing the session."""
        self.store.set_value(CHAT_SESSION_KEY, "s1")
        self.engine.reset_chat()
        self.assertEqual(self.store.get_value(CHAT_RESET_KEY), START.isoformat())

    def test_turn_queued_before_new_never_stores_its_session(self) -> None:
        """A turn queued before /new must not bring the old conversation back.

        Chat A is queued, the user sends /new, chat B is queued; A and B then
        run in queue order. B must start fresh and the turn after it must
        resume B's session, not A's.
        """
        self.runner.queue(success(session_id="old"), success(session_id="new"), success(session_id="new"))
        turn_a = self.engine.enqueue_prompt("A", dispatch=False)
        self.clock.advance(seconds=1)
        self.engine.reset_chat()
        self.clock.advance(seconds=1)
        turn_b = self.engine.enqueue_prompt("B", dispatch=False)
        self.run_now(turn_a)
        self.run_now(turn_b)
        self.chat("C")
        self.assertEqual([c.resume_session for c in self.runner.calls], [None, None, "new"])
        self.assertEqual(self.store.get_value(CHAT_SESSION_KEY), "new")

    def test_late_old_turn_cannot_overwrite_the_new_conversation(self) -> None:
        """A turn from before /new that finishes after the new conversation started is ignored."""
        self.runner.queue(success(session_id="new"), success(session_id="old"))
        turn_a = self.engine.enqueue_prompt("A", dispatch=False)
        self.clock.advance(seconds=1)
        self.engine.reset_chat()
        self.clock.advance(seconds=1)
        turn_b = self.engine.enqueue_prompt("B", dispatch=False)
        self.run_now(turn_b)
        self.run_now(turn_a)
        self.assertEqual(self.store.get_value(CHAT_SESSION_KEY), "new")

    def test_turn_queued_at_the_same_instant_as_new_stores_its_session(self) -> None:
        """With a frozen clock, /new followed by a message in the same instant starts the new conversation."""
        self.runner.queue(success(session_id="s1"))
        self.engine.reset_chat()
        self.chat()
        self.assertEqual(self.store.get_value(CHAT_SESSION_KEY), "s1")

    def test_retried_turn_from_before_new_stays_cut_off(self) -> None:
        """A pre-/new turn that hit the usage limit is retried as a copy; it must not revive the old chat."""
        turn_a = self.engine.enqueue_prompt("A", dispatch=False)
        self.clock.advance(seconds=1)
        self.engine.reset_chat()
        self.clock.advance(seconds=1)
        turn_b = self.engine.enqueue_prompt("B", dispatch=False)
        self.runner.queue(
            success(session_id="new"),
            RunOutcome(status=STATUS_RATE_LIMITED, error="limit", retry_at=self.clock() + timedelta(minutes=10)),
            success(session_id="old"),
        )
        self.run_now(turn_b)
        self.assertEqual(self.run_now(turn_a).status, STATUS_RATE_LIMITED)
        self.clock.advance(minutes=11)
        (retry_id,) = self.engine.tick()
        retry = self.store.get_run(retry_id)
        # The copy is queued after the reset; only its chain root predates it.
        self.assertGreater(retry.queued_at, START + timedelta(seconds=2))
        self.assertEqual(self.run_now(retry_id).status, STATUS_SUCCESS)
        self.assertEqual(self.store.get_value(CHAT_SESSION_KEY), "new")

    def test_turn_queued_after_new_stores_its_session(self) -> None:
        """Only turns queued at or before the reset are affected; later ones store normally."""
        self.engine.reset_chat()
        self.clock.advance(seconds=1)
        self.runner.queue(success(session_id="s1"))
        self.chat()
        self.assertEqual(self.store.get_value(CHAT_SESSION_KEY), "s1")

    def test_idle_expiry_and_failed_turns_are_not_user_resets(self) -> None:
        """Automatic resets don't write chat_reset_at, so queued turns still store their sessions."""
        self.runner.queue(success(session_id="s1"), success(session_id="s2"),
                          RunOutcome(status=STATUS_ERROR, error="x"))
        self.chat()
        self.clock.advance(hours=7)
        self.chat()  # idle expiry, then stores s2
        self.assertEqual(self.store.get_value(CHAT_SESSION_KEY), "s2")
        self.chat()  # resumed turn fails and clears the session
        self.assertIsNone(self.store.get_value(CHAT_SESSION_KEY))
        self.assertIsNone(self.store.get_value(CHAT_RESET_KEY))

    def test_unreadable_reset_marker_is_ignored(self) -> None:
        """A corrupt chat_reset_at value does not stop sessions from being stored."""
        self.store.set_value(CHAT_RESET_KEY, "garbage")
        self.runner.queue(success(session_id="s1"))
        self.chat()
        self.assertEqual(self.store.get_value(CHAT_SESSION_KEY), "s1")

    def test_reset_chat_starts_a_new_conversation(self) -> None:
        """After reset_chat the next turn has no session."""
        self.runner.queue(success(session_id="s1"))
        self.chat()
        self.engine.reset_chat()
        self.chat()
        self.assertIsNone(self.runner.calls[1].resume_session)

    def test_failed_resumed_turn_resets_the_chat_and_says_so(self) -> None:
        """An error on a resumed session clears it and tells the user the next message starts fresh."""
        self.runner.queue(success(session_id="s1"), RunOutcome(status=STATUS_ERROR, error="session not found"))
        self.chat()
        failed = self.chat()
        self.assertEqual(failed.status, STATUS_ERROR)
        self.assertEqual(
            failed.error,
            "session not found The conversation was reset, so your next message starts a new one.",
        )
        self.assertIsNone(self.store.get_value(CHAT_SESSION_KEY))
        self.chat()
        self.assertIsNone(self.runner.calls[2].resume_session)

    def test_failed_resumed_turn_without_error_text(self) -> None:
        """The reset note stands alone when the runner gave no error text."""
        self.runner.queue(success(session_id="s1"), RunOutcome(status=STATUS_ERROR))
        self.chat()
        self.assertEqual(
            self.chat().error,
            "The conversation was reset, so your next message starts a new one.",
        )

    def test_failed_first_turn_has_no_reset_note(self) -> None:
        """A failure without a resumed session is reported as is."""
        self.runner.queue(RunOutcome(status=STATUS_ERROR, error="boom"))
        self.assertEqual(self.chat().error, "boom")

    def test_timeout_keeps_the_session(self) -> None:
        """Only errors reset the conversation; a timeout leaves it intact."""
        self.runner.queue(success(session_id="s1"), RunOutcome(status=STATUS_TIMEOUT, error="slow"))
        self.chat()
        self.chat()
        self.assertEqual(self.store.get_value(CHAT_SESSION_KEY), "s1")

    def test_task_and_one_off_runs_never_resume_the_chat(self) -> None:
        """Scheduled/manual task runs and CLI asks always start fresh sessions."""
        self.runner.queue(success(session_id="s1"))
        self.chat()
        self.run_news()
        self.run_now(self.engine.enqueue_prompt("one-off", TRIGGER_MANUAL, dispatch=False))
        self.assertEqual([c.resume_session for c in self.runner.calls[1:]], [None, None])

    def test_task_and_one_off_runs_do_not_touch_the_chat_session(self) -> None:
        """A task run's session id must not replace the chat session."""
        self.runner.queue(success(session_id="chat"), success(session_id="task"))
        self.chat()
        self.run_news()
        self.assertEqual(self.store.get_value(CHAT_SESSION_KEY), "chat")

    def test_chat_turns_are_serialized_under_a_lock(self) -> None:
        """A second turn waits for the first, so it resumes the session the first created."""
        engine = self.make_engine(max_parallel=2)
        engine.load_config()
        first_started = threading.Event()
        release_first = threading.Event()

        def block_first_call() -> None:
            """Hold the first runner call until the test releases it."""
            if len(self.runner.calls) == 1:
                first_started.set()
                release_first.wait(WAIT_SECONDS)

        self.runner.on_run = block_first_call
        self.runner.queue(success(session_id="s1"), success(session_id="s1"))
        first_id = engine.enqueue_prompt("one", dispatch=False)
        second_id = engine.enqueue_prompt("two", dispatch=False)
        threads = [threading.Thread(target=engine.execute_run, args=(i,)) for i in (first_id, second_id)]
        threads[0].start()
        self.assertTrue(first_started.wait(WAIT_SECONDS))
        threads[1].start()
        time.sleep(0.2)  # give the second turn every chance to overtake
        self.assertEqual(len(self.runner.calls), 1)
        release_first.set()
        for thread in threads:
            thread.join(WAIT_SECONDS)
        self.assertEqual([c.prompt for c in self.runner.calls], ["one", "two"])
        self.assertEqual([c.resume_session for c in self.runner.calls], [None, "s1"])


class StoppingTests(EngineTestCase):
    """Runs whose Claude call ends because the service is stopping."""

    def stop_during_the_call(self) -> None:
        """Make the fake runner call engine.stop() while "Claude" is working."""
        self.runner.on_run = self.engine.stop

    def test_run_finished_by_stop_is_interrupted_and_silent(self) -> None:
        """A killed call is recorded as interrupted, whatever it returned, and nobody is notified."""
        self.stop_during_the_call()
        self.runner.queue(RunOutcome(status=STATUS_ERROR, error="Claude CLI exited with code -15"))
        record = self.run_news()
        self.assertEqual((record.status, record.error), (STATUS_ERROR, INTERRUPTED_MESSAGE))
        self.assertEqual((self.notifier.runs, self.notifier.texts), ([], []))
        self.assertEqual(self.store.get_run(record.id).error, INTERRUPTED_MESSAGE)

    def test_interrupted_resumed_chat_turn_keeps_the_conversation(self) -> None:
        """A resumed turn killed by stop() must not wipe the session like a real failed turn does."""
        self.runner.queue(success(session_id="s1"))
        self.chat()
        self.notifier.runs.clear()
        self.stop_during_the_call()
        self.runner.queue(RunOutcome(status=STATUS_ERROR, error="exited with code -15"))
        record = self.chat()
        self.assertEqual(record.error, INTERRUPTED_MESSAGE)
        self.assertEqual(self.store.get_value(CHAT_SESSION_KEY), "s1")
        self.assertEqual(self.notifier.runs, [])

    def test_successful_chat_turn_returning_while_stopping_does_not_store_a_session(self) -> None:
        """Nothing about a stopping service's last call touches the chat session."""
        self.stop_during_the_call()
        self.runner.queue(success(session_id="late"))
        self.chat()
        self.assertIsNone(self.store.get_value(CHAT_SESSION_KEY))

    def test_rate_limit_returning_while_stopping_does_not_close_the_gate(self) -> None:
        """A limit reported by a killed call neither notifies nor holds later runs."""
        self.stop_during_the_call()
        self.runner.queue(RunOutcome(status=STATUS_RATE_LIMITED, error="limit", retry_at=START + timedelta(hours=1)))
        record = self.run_news()
        self.assertEqual(record.error, INTERRUPTED_MESSAGE)
        self.assertIsNone(record.retry_at)
        self.assertIsNone(self.store.get_value(LIMITED_UNTIL_KEY))
        self.assertEqual(self.notifier.texts, [])


class InformationTests(EngineTestCase):
    """next_fire_times and snapshot."""

    def test_next_fire_times_for_every_kind_of_task(self) -> None:
        """Scheduled tasks get a time; manual-only, disabled and paused tasks get None."""
        schedule = (parse_schedule("daily 09:00"),)
        self.set_tasks(
            make_task("daily", "p", schedule),
            make_task("manual", "p", ()),
            make_task("off", "p", schedule, enabled=False),
            make_task("paused", "p", schedule),
        )
        self.store.set_paused("paused", True)
        times = self.engine.next_fire_times()
        self.assertEqual(list(times), ["daily", "manual", "off", "paused"])
        self.assertEqual(times["daily"], datetime(2026, 11, 2, 9, 0, tzinfo=TZ))
        self.assertEqual((times["manual"], times["off"], times["paused"]), (None, None, None))

    def test_next_fire_time_moves_past_the_marker(self) -> None:
        """A slot that already ran (marker) is not offered again, even if the clock is slightly behind."""
        self.store.set_last_scheduled("news", START)
        self.assertEqual(self.engine.next_fire_times()["news"], datetime(2026, 11, 3, 7, 30, tzinfo=TZ))

    def test_next_fire_time_is_in_local_time_across_dst(self) -> None:
        """With a fixed-offset marker from the store the result still follows DST."""
        self.clock.set(datetime(2026, 10, 24, 12, 0, tzinfo=TZ))
        self.store.set_last_scheduled("news", datetime(2026, 10, 24, 7, 30, tzinfo=TZ))
        self.assertEqual(self.engine.next_fire_times()["news"], datetime(2026, 10, 25, 7, 30, tzinfo=TZ))
        self.clock.set(datetime(2026, 10, 25, 12, 0, tzinfo=TZ))
        self.assertEqual(self.engine.next_fire_times()["news"].utcoffset(), timedelta(hours=1))

    def test_snapshot_splits_running_and_queued_oldest_first(self) -> None:
        """The snapshot lists running and queued runs separately, oldest first."""
        running = self.store.create_run(task_name="a", trigger=TRIGGER_MANUAL, prompt="p", queued_at=START)
        self.store.mark_running(running, START)
        queued_one = self.store.create_run(task_name="b", trigger=TRIGGER_MANUAL, prompt="p", queued_at=START)
        queued_two = self.store.create_run(task_name="c", trigger=TRIGGER_MANUAL, prompt="p", queued_at=START)
        snap = self.engine.snapshot()
        self.assertEqual([r.id for r in snap.running], [running])
        self.assertEqual([r.id for r in snap.queued], [queued_one, queued_two])
        self.assertIsNone(snap.limited_until)
        self.assertTrue(all(r.status in (STATUS_RUNNING, STATUS_QUEUED) for r in snap.running + snap.queued))


class WorkerTests(EngineTestCase):
    """Worker threads and the service loop."""

    def test_workers_execute_dispatched_runs_and_skip_undispatched_ones(self) -> None:
        """Dispatched runs are executed by workers; dispatch=False runs stay queued."""
        self.set_tasks(self.news, make_task("plan", "Plan", ()))
        held = self.engine.enqueue_task("plan", TRIGGER_MANUAL, dispatch=False)
        dispatched = self.engine.enqueue_task("news", TRIGGER_MANUAL)
        self.engine.start_workers()
        self.addCleanup(self.engine.stop)
        self.assertTrue(wait_until(lambda: self.store.get_run(dispatched).status == STATUS_SUCCESS))
        self.assertEqual(self.store.get_run(held).status, STATUS_QUEUED)

    def test_worker_survives_a_failing_run(self) -> None:
        """An exception inside execute_run is logged and the worker takes the next run."""
        self.set_tasks(self.news, make_task("plan", "Plan", ()))
        first = self.engine.enqueue_task("news", TRIGGER_MANUAL)
        second = self.engine.enqueue_task("plan", TRIGGER_MANUAL)
        real = self.engine.execute_run

        def flaky(run_id: int) -> RunRecord | None:
            """Raise for the first run and delegate every other run to the real method."""
            if run_id == first:
                raise RuntimeError("store exploded")
            return real(run_id)

        self.addCleanup(self.engine.stop)
        with mock.patch.object(self.engine, "execute_run", side_effect=flaky):
            with self.assertLogs("autopilot.engine", level="ERROR"):
                self.engine.start_workers()
                self.assertTrue(wait_until(lambda: self.store.get_run(second).status == STATUS_SUCCESS))

    def test_start_workers_twice_starts_one_set(self) -> None:
        """Calling start_workers again does not add threads."""
        engine = self.make_engine(max_parallel=2)
        engine.load_config()
        engine.start_workers()
        engine.start_workers()
        names = [t.name for t in threading.enumerate() if t.name.startswith("engine-worker-")]
        engine.stop()
        self.assertEqual(len(names), 2)

    def test_stop_ends_the_workers(self) -> None:
        """After stop() no worker thread is alive and queued runs are left for the next start."""
        engine = self.make_engine(max_parallel=2)
        engine.load_config()
        engine.start_workers()
        engine.stop()
        self.assertEqual([t for t in threading.enumerate() if t.name.startswith("engine-worker-")], [])
        run_id = engine.enqueue_task("news", TRIGGER_MANUAL)
        self.assertEqual(self.store.get_run(run_id).status, STATUS_QUEUED)

    def test_recover_interrupted_fails_leftover_runs_and_returns_the_count(self) -> None:
        """recover_interrupted marks queued and running leftovers as errors and counts them."""
        running = self.store.create_run(task_name="news", trigger=TRIGGER_MANUAL, prompt="p", queued_at=START)
        self.store.mark_running(running, START)
        queued = self.store.create_run(task_name="plan", trigger=TRIGGER_MANUAL, prompt="p", queued_at=START)
        with self.assertLogs("autopilot.engine", level="WARNING"):
            self.assertEqual(self.engine.recover_interrupted(), 2)
        self.assertEqual(self.store.get_run(running).status, STATUS_ERROR)
        self.assertEqual(self.store.get_run(queued).status, STATUS_ERROR)
        self.assertEqual(self.engine.recover_interrupted(), 0)

    def test_stop_terminates_in_flight_claude_processes_when_the_runner_can(self) -> None:
        """stop() asks a runner that offers terminate_all to kill its live processes."""

        class TerminatingRunner(ScriptedRunner):
            """ScriptedRunner that also offers ClaudeRunner's terminate_all."""

            terminated = 0

            def terminate_all(self) -> int:
                """Count the call and report one killed process."""
                self.terminated += 1
                return 1

        runner = TerminatingRunner()
        engine = self.make_engine(runner=runner)
        engine.load_config()
        engine.start_workers()
        with self.assertLogs("autopilot.engine", level="INFO"):
            engine.stop()
        self.assertEqual(runner.terminated, 1)

    def test_stop_without_terminate_all_on_the_runner_still_works(self) -> None:
        """A runner without terminate_all (only the Runner protocol) is simply not asked."""
        self.engine.start_workers()
        self.engine.stop()
        self.assertEqual([t for t in threading.enumerate() if t.name.startswith("engine-worker-")], [])

    def test_run_forever_does_not_recover_runs_queued_before_it_starts(self) -> None:
        """Recovery is the caller's job: a run queued before the loop starts (for example by the
        bot thread) must survive and be executed, not be marked interrupted."""
        run_id = self.engine.enqueue_prompt("early chat message")
        stop = threading.Event()
        thread = threading.Thread(target=self.engine.run_forever, args=(stop,))
        thread.start()
        try:
            self.assertTrue(wait_until(lambda: self.store.get_run(run_id).status == STATUS_SUCCESS))
        finally:
            stop.set()
            thread.join(WAIT_SECONDS)
        self.assertEqual(len(self.runner.calls), 1)

    def test_run_forever_ticks_executes_and_stops(self) -> None:
        """After recovery the loop queues due slots, runs them and ends on stop."""
        stale = self.store.create_run(task_name="news", trigger=TRIGGER_MANUAL, prompt="p", queued_at=START)
        self.store.mark_running(stale, START)
        self.store.set_last_scheduled("news", START - timedelta(days=1))
        self.engine.recover_interrupted()
        stop = threading.Event()
        thread = threading.Thread(target=self.engine.run_forever, args=(stop,))
        thread.start()
        try:
            self.assertTrue(wait_until(lambda: len(self.runner.calls) == 1))
        finally:
            stop.set()
            thread.join(WAIT_SECONDS)
        self.assertFalse(thread.is_alive())
        self.assertEqual(self.store.get_run(stale).status, STATUS_ERROR)
        scheduled = [r for r in self.store.list_runs() if r.trigger == TRIGGER_SCHEDULE]
        self.assertEqual(len(scheduled), 1)
        self.assertEqual([t for t in threading.enumerate() if t.name.startswith("engine-worker-")], [])

    def test_run_forever_returns_immediately_when_already_stopped(self) -> None:
        """A set stop event ends the loop without ticking."""
        stop = threading.Event()
        stop.set()
        with mock.patch.object(self.engine, "tick") as tick:
            self.engine.run_forever(stop)
        tick.assert_not_called()

    def test_run_forever_survives_a_failing_tick(self) -> None:
        """An exception in tick is logged and the loop keeps going until stopped."""
        stop = threading.Event()
        failed = threading.Event()

        def boom() -> list[int]:
            """Signal that tick was reached, then fail."""
            failed.set()
            raise RuntimeError("tick broke")

        thread = threading.Thread(target=self.engine.run_forever, args=(stop,))
        with mock.patch.object(self.engine, "tick", side_effect=boom):
            with self.assertLogs("autopilot.engine", level="ERROR"):
                thread.start()
                self.assertTrue(failed.wait(WAIT_SECONDS))
                self.assertTrue(thread.is_alive())
                stop.set()
                thread.join(WAIT_SECONDS)
        self.assertFalse(thread.is_alive())

    def test_run_forever_sleeps_until_the_next_fire_time(self) -> None:
        """The loop sleeps TICK_SECONDS at most, less when a slot is near, never under 1 second."""
        self.store.set_last_scheduled("news", START)
        cases = [
            (timedelta(hours=1), TICK_SECONDS),  # slot tomorrow: capped at the tick length
            (timedelta(hours=23, minutes=59, seconds=50), 10),  # 10 s to the next slot
            (timedelta(hours=23, minutes=59, seconds=59, milliseconds=500), 1),  # below the minimum
        ]
        for offset, expected in cases:
            with self.subTest(offset=offset):
                self.clock.set(START + offset)
                stop = RecordingStop()
                self.engine.run_forever(stop)
                self.assertEqual(stop.waits, [expected])
                self.store.set_last_scheduled("news", START)

    def test_run_forever_waits_a_full_tick_without_schedules(self) -> None:
        """With only manual tasks the loop sleeps TICK_SECONDS between ticks."""
        self.set_tasks(make_task("manual", "p", ()))
        stop = RecordingStop()
        self.engine.run_forever(stop)
        self.assertEqual(stop.waits, [TICK_SECONDS])


class LogNotifierTests(unittest.TestCase):
    """LogNotifier writes log lines instead of sending messages."""

    def test_notify_run_logs_one_line(self) -> None:
        """A finished run is logged with its id, task and status."""
        record = RunRecord(
            id=7, task_name="news", trigger=TRIGGER_SCHEDULE, status=STATUS_SUCCESS, prompt="p", queued_at=START
        )
        with self.assertLogs("autopilot.engine", level="INFO") as logs:
            LogNotifier().notify_run(record)
        self.assertEqual(len(logs.output), 1)
        self.assertIn("#7", logs.output[0])
        self.assertIn("news", logs.output[0])
        self.assertIn(STATUS_SUCCESS, logs.output[0])

    def test_notify_text_logs_first_line_only(self) -> None:
        """A multi-line notice is logged as its first line."""
        with self.assertLogs("autopilot.engine", level="INFO") as logs:
            LogNotifier().notify_text("⚠️ headline\ndetails")
        self.assertEqual(len(logs.output), 1)
        self.assertIn("headline", logs.output[0])
        self.assertNotIn("details", logs.output[0])


if __name__ == "__main__":
    unittest.main()
