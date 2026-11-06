"""Unit tests for autopilot.models and autopilot.errors, plus the shared test helpers.

Covers the constants other waves depend on (status partitions, trigger order,
defaults), the frozen dataclasses, the error hierarchy and messages, and the
behaviour of the factories and fakes in helpers.py that every later test uses.
"""

from __future__ import annotations

import dataclasses
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from autopilot.errors import (
    AutopilotError,
    ClaudeNotFoundError,
    ConfigError,
    ScheduleError,
    TaskNotFoundError,
    TelegramError,
)
from autopilot.models import (
    ACTIVE_STATUSES,
    ASK_TASK_NAME,
    DEFAULT_PERMISSION_MODE,
    FINISHED_STATUSES,
    NOTIFY_ALWAYS,
    NOTIFY_POLICIES,
    PERMISSION_MODES,
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
    TRIGGERS,
    RunOptions,
    Task,
)
from helpers import (
    FAKE_BOT_USERNAME,
    TZ,
    FakeClock,
    FakeTelegramClient,
    make_config,
    make_options,
    make_record,
    make_task,
)


class StatusConstantsTest(unittest.TestCase):
    """The status sets partition every run status into active and finished."""

    def test_every_status_is_either_active_or_finished(self) -> None:
        """Every status is active or finished, never both, so the engine never loses a run."""
        all_statuses = {
            STATUS_QUEUED,
            STATUS_RUNNING,
            STATUS_SUCCESS,
            STATUS_ERROR,
            STATUS_TIMEOUT,
            STATUS_RATE_LIMITED,
            STATUS_CANCELLED,
        }
        self.assertEqual(ACTIVE_STATUSES | FINISHED_STATUSES, all_statuses)
        self.assertEqual(ACTIVE_STATUSES & FINISHED_STATUSES, frozenset())

    def test_active_statuses_are_queued_and_running(self) -> None:
        """Only queued and running runs count as active."""
        self.assertEqual(ACTIVE_STATUSES, {STATUS_QUEUED, STATUS_RUNNING})


class ChoiceConstantsTest(unittest.TestCase):
    """Fixed choice lists that validation and the CLI depend on."""

    def test_triggers_keep_their_documented_order(self) -> None:
        """Triggers are listed in the order the docs use: schedule, manual, telegram, chat."""
        self.assertEqual(
            TRIGGERS, (TRIGGER_SCHEDULE, TRIGGER_MANUAL, TRIGGER_TELEGRAM, TRIGGER_CHAT)
        )

    def test_ask_task_name_is_the_reserved_word(self) -> None:
        """The reserved name used for chat and ask runs is "ask"."""
        self.assertEqual(ASK_TASK_NAME, "ask")

    def test_notify_policies_are_always_errors_never(self) -> None:
        """The three notification policies from tasks.toml are exactly these."""
        self.assertEqual(NOTIFY_POLICIES, ("always", "errors", "never"))

    def test_default_permission_mode_is_a_known_mode(self) -> None:
        """The default permission mode must be one of the accepted modes."""
        self.assertIn(DEFAULT_PERMISSION_MODE, PERMISSION_MODES)
        self.assertEqual(DEFAULT_PERMISSION_MODE, "dontAsk")


class RunOptionsTest(unittest.TestCase):
    """RunOptions defaults and immutability."""

    def test_defaults_match_the_contract(self) -> None:
        """Options left unset get the documented defaults: no model, 30 minute timeout, dontAsk."""
        options = RunOptions(workdir=Path("/work"))
        self.assertIsNone(options.model)
        self.assertIsNone(options.effort)
        self.assertEqual(options.allowed_tools, ())
        self.assertEqual(options.disallowed_tools, ())
        self.assertEqual(options.permission_mode, "dontAsk")
        self.assertEqual(options.add_dirs, ())
        self.assertIsNone(options.append_system_prompt)
        self.assertIsNone(options.max_budget_usd)
        self.assertEqual(options.timeout_seconds, 1800)

    def test_fields_cannot_be_reassigned(self) -> None:
        """A RunOptions instance is frozen, so a shared default can't be changed by one caller."""
        options = make_options()
        with self.assertRaises(dataclasses.FrozenInstanceError):
            options.model = "sonnet"  # type: ignore[misc]


class TaskTest(unittest.TestCase):
    """Task defaults and schedule handling."""

    def test_defaults_are_enabled_and_notify_always(self) -> None:
        """A new task is enabled, has no description and notifies on every run."""
        task = Task(name="news", prompt="Say hi", schedules=(), options=make_options())
        self.assertTrue(task.enabled)
        self.assertEqual(task.description, "")
        self.assertEqual(task.notify, NOTIFY_ALWAYS)

    def test_schedules_are_stored_as_given(self) -> None:
        """make_task keeps the schedules tuple it receives, so tests can pass real Schedule objects."""
        schedules = ("first", "second")
        task = make_task(schedules=schedules)  # type: ignore[arg-type]
        self.assertIs(task.schedules, schedules)


class RunRecordTest(unittest.TestCase):
    """RunRecord.is_finished tells active runs apart from finished ones."""

    def test_is_finished_is_true_for_every_final_status(self) -> None:
        """Each final status makes is_finished True."""
        for status in FINISHED_STATUSES:
            with self.subTest(status=status):
                self.assertTrue(make_record(status=status).is_finished)

    def test_is_finished_is_false_while_queued_or_running(self) -> None:
        """Queued and running runs are not finished yet."""
        for status in ACTIVE_STATUSES:
            with self.subTest(status=status):
                self.assertFalse(make_record(status=status).is_finished)


class ErrorHierarchyTest(unittest.TestCase):
    """Every expected error is an AutopilotError, and the subclasses nest correctly."""

    def test_config_and_schedule_errors_are_autopilot_errors(self) -> None:
        """ConfigError and ClaudeNotFoundError are AutopilotErrors; ScheduleError is a ConfigError."""
        self.assertTrue(issubclass(ConfigError, AutopilotError))
        self.assertTrue(issubclass(ScheduleError, ConfigError))
        self.assertTrue(issubclass(ClaudeNotFoundError, AutopilotError))

    def test_telegram_and_task_errors_are_autopilot_errors(self) -> None:
        """TelegramError and TaskNotFoundError are AutopilotErrors, so the CLI catches them."""
        self.assertTrue(issubclass(TelegramError, AutopilotError))
        self.assertTrue(issubclass(TaskNotFoundError, AutopilotError))

    def test_message_passes_through_str(self) -> None:
        """A plain AutopilotError prints exactly the message it was built with."""
        self.assertEqual(str(AutopilotError("Run init first.")), "Run init first.")


class TaskNotFoundErrorTest(unittest.TestCase):
    """TaskNotFoundError names the task and lists the known ones."""

    def test_message_lists_known_tasks_in_given_order(self) -> None:
        """The message names the unknown task and lists the known ones in the order given."""
        error = TaskNotFoundError("x", ["a", "b"])
        self.assertEqual(str(error), "Unknown task 'x'. Known tasks: a, b")

    def test_message_says_when_no_tasks_exist(self) -> None:
        """With no tasks defined, the message says so instead of listing nothing."""
        error = TaskNotFoundError("x", [])
        self.assertEqual(str(error), "Unknown task 'x'. No tasks are defined.")

    def test_name_attribute_keeps_the_unknown_name(self) -> None:
        """The unknown name is kept on the error for callers that need it."""
        self.assertEqual(TaskNotFoundError("nightly", ["news"]).name, "nightly")

    def test_known_names_may_come_from_a_one_shot_iterator(self) -> None:
        """Known names are read once, so a generator works as well as a list."""
        error = TaskNotFoundError("x", (name for name in ["a"]))
        self.assertEqual(str(error), "Unknown task 'x'. Known tasks: a")

    def test_unicode_task_names_are_kept_intact(self) -> None:
        """Non-ASCII names are shown exactly, with no escaping or mangling."""
        error = TaskNotFoundError("Naïve", ["café"])
        expected = "Unknown task 'Naïve'. Known tasks: café"
        self.assertEqual(str(error), expected)


class TelegramErrorTest(unittest.TestCase):
    """TelegramError carries the HTTP status and retry hint."""

    def test_status_and_retry_after_default_to_none(self) -> None:
        """A network failure has neither a status nor a retry hint."""
        error = TelegramError("Telegram sendMessage failed: network error")
        self.assertIsNone(error.status)
        self.assertIsNone(error.retry_after)

    def test_status_and_retry_after_are_kept(self) -> None:
        """Status and retry_after given by the caller are stored on the error."""
        error = TelegramError("Too many requests", status=429, retry_after=7)
        self.assertEqual(error.status, 429)
        self.assertEqual(error.retry_after, 7)
        self.assertEqual(str(error), "Too many requests")


class ClaudeNotFoundErrorTest(unittest.TestCase):
    """ClaudeNotFoundError keeps the message it is given."""

    def test_message_is_kept_unchanged(self) -> None:
        """The message the runner builds is passed to the user unchanged."""
        message = "Claude Code CLI not found. Set CLAUDE_BIN in .env."
        self.assertEqual(str(ClaudeNotFoundError(message)), message)


class FakeClockTest(unittest.TestCase):
    """FakeClock moves only when the test says so and refuses naive times."""

    def test_returns_start_until_moved(self) -> None:
        """Calling the clock returns the start moment until advance or set is used."""
        start = datetime(2026, 11, 2, 7, 30, tzinfo=TZ)
        clock = FakeClock(start)
        self.assertEqual(clock(), start)
        self.assertEqual(clock(), start)

    def test_advance_moves_forward_and_returns_new_moment(self) -> None:
        """advance adds the given duration and returns the new moment."""
        clock = FakeClock(datetime(2026, 11, 2, 7, 30, tzinfo=TZ))
        moved = clock.advance(hours=1)
        self.assertEqual(moved, clock())
        self.assertEqual(moved, datetime(2026, 11, 2, 8, 30, tzinfo=TZ))

    def test_set_jumps_to_an_exact_moment(self) -> None:
        """set replaces the current moment with the given one."""
        clock = FakeClock(datetime(2026, 11, 2, 7, 30, tzinfo=TZ))
        target = datetime(2026, 12, 1, 0, 0, tzinfo=timezone.utc)
        clock.set(target)
        self.assertEqual(clock(), target)

    def test_naive_start_is_rejected(self) -> None:
        """A naive start time raises ValueError, because naive times are ambiguous."""
        with self.assertRaises(ValueError):
            FakeClock(datetime(2026, 11, 2, 7, 30))

    def test_naive_set_is_rejected_and_keeps_the_old_moment(self) -> None:
        """A naive set raises ValueError and leaves the clock where it was."""
        start = datetime(2026, 11, 2, 7, 30, tzinfo=TZ)
        clock = FakeClock(start)
        with self.assertRaises(ValueError):
            clock.set(datetime(2026, 11, 2, 8, 0))
        self.assertEqual(clock(), start)

    def test_negative_advance_moves_backwards(self) -> None:
        """A negative duration moves the clock back, for testing slots in the past."""
        clock = FakeClock(datetime(2026, 11, 2, 7, 30, tzinfo=TZ))
        moved = clock.advance(minutes=-30)
        self.assertEqual(moved, datetime(2026, 11, 2, 7, 0, tzinfo=TZ))


class MakeConfigTest(unittest.TestCase):
    """make_config builds a TaskConfig and refuses duplicate task names."""

    def test_tasks_keep_the_given_order(self) -> None:
        """Tasks are keyed by name in the order they were passed, matching file order."""
        config = make_config(make_task("b"), make_task("a"))
        self.assertEqual(list(config.tasks), ["b", "a"])

    def test_default_ask_options_are_built(self) -> None:
        """Without ask_options, the config gets default RunOptions."""
        config = make_config(make_task())
        self.assertEqual(config.ask_options, make_options())

    def test_duplicate_names_are_rejected(self) -> None:
        """Two tasks with one name raise ValueError instead of silently overwriting."""
        with self.assertRaises(ValueError):
            make_config(make_task("news"), make_task("news"))

    def test_source_defaults_to_none(self) -> None:
        """An in-memory config has no source file."""
        self.assertIsNone(make_config().source)


class MakeRecordTest(unittest.TestCase):
    """make_record defaults describe a valid finished run and accept overrides."""

    def test_defaults_describe_a_successful_scheduled_run(self) -> None:
        """Default record: id 1, task news, scheduled trigger, success, result Hello, aware time."""
        record = make_record()
        self.assertEqual(record.id, 1)
        self.assertEqual(record.task_name, "news")
        self.assertEqual(record.trigger, TRIGGER_SCHEDULE)
        self.assertEqual(record.status, STATUS_SUCCESS)
        self.assertEqual(record.prompt, "Say hi")
        self.assertEqual(record.result, "Hello")
        self.assertEqual(record.queued_at, datetime(2026, 11, 2, 7, 30, tzinfo=TZ))
        self.assertIsNotNone(record.queued_at.utcoffset())

    def test_overrides_replace_defaults(self) -> None:
        """Keyword overrides replace the matching default fields."""
        record = make_record(id=7, status=STATUS_ERROR, error="boom")
        self.assertEqual(record.id, 7)
        self.assertEqual(record.status, STATUS_ERROR)
        self.assertEqual(record.error, "boom")


class FakeTelegramClientTest(unittest.TestCase):
    """FakeTelegramClient records calls and simulates the failure switches."""

    def test_send_message_records_text_and_returns_ids_in_sequence(self) -> None:
        """Each sent message is recorded and gets the next id, starting at 100."""
        client = FakeTelegramClient()
        first = client.send_message(1, "hi")
        second = client.send_message(1, "again", parse_mode=None)
        self.assertEqual(first["message_id"], 100)
        self.assertEqual(second["message_id"], 101)
        self.assertEqual([entry["text"] for entry in client.sent], ["hi", "again"])
        self.assertIsNone(client.sent[1]["parse_mode"])

    def test_fail_html_once_raises_one_parse_error_then_recovers(self) -> None:
        """The armed failure raises a 400 "can't parse entities" error once, then HTML sends work again."""
        client = FakeTelegramClient()
        client.fail_html_once = True
        with self.assertRaises(TelegramError) as raised:
            client.send_message(1, "<b>broken")
        self.assertEqual(raised.exception.status, 400)
        self.assertIn("can't parse entities", str(raised.exception))
        self.assertEqual(client.sent, [])
        client.send_message(1, "<b>fixed</b>")
        self.assertEqual(len(client.sent), 1)
        self.assertFalse(client.fail_html_once)

    def test_fail_html_once_does_not_affect_plain_text(self) -> None:
        """The armed parse failure only hits HTML messages, so a plain-text resend succeeds."""
        client = FakeTelegramClient()
        client.fail_html_once = True
        client.send_message(1, "plain", parse_mode=None)
        self.assertEqual(len(client.sent), 1)
        self.assertTrue(client.fail_html_once)

    def test_get_updates_returns_batches_in_order_then_empty(self) -> None:
        """Each get_updates call pops the next queued batch; once none are left it returns an empty list."""
        client = FakeTelegramClient()
        client.updates = [[{"update_id": 1}], [{"update_id": 2}]]
        self.assertEqual(client.get_updates(), [{"update_id": 1}])
        self.assertEqual(client.get_updates(), [{"update_id": 2}])
        self.assertEqual(client.get_updates(), [])

    def test_get_updates_raises_the_armed_error_on_every_call(self) -> None:
        """An armed updates error is raised again on each poll, so the bot's backoff is exercised."""
        client = FakeTelegramClient()
        client.raise_on_updates = TelegramError("network down")
        for attempt in range(2):
            with self.subTest(attempt=attempt):
                with self.assertRaises(TelegramError):
                    client.get_updates()

    def test_send_document_records_content_and_takes_an_id(self) -> None:
        """A document is recorded with its bytes and filename and shares the message id sequence."""
        client = FakeTelegramClient()
        client.send_message(1, "text")
        result = client.send_document(1, "run.md", b"# Title", caption="cap")
        self.assertEqual(result["message_id"], 101)
        self.assertEqual(client.documents[0]["filename"], "run.md")
        self.assertEqual(client.documents[0]["content"], b"# Title")
        self.assertEqual(client.documents[0]["caption"], "cap")

    def test_set_my_commands_and_chat_actions_are_recorded(self) -> None:
        """Command lists and chat actions are kept for later assertions."""
        client = FakeTelegramClient()
        self.assertIsNone(client.commands)
        client.set_my_commands([("help", "How to use this bot")])
        client.send_chat_action(5, "typing")
        self.assertEqual(client.commands, [("help", "How to use this bot")])
        self.assertEqual(client.chat_actions, [(5, "typing")])

    def test_get_me_reports_the_fake_bot_username(self) -> None:
        """get_me returns the fake bot identity used across the tests."""
        self.assertEqual(FakeTelegramClient().get_me()["username"], FAKE_BOT_USERNAME)


class MakeOptionsTest(unittest.TestCase):
    """make_options defaults the workdir to the system temp folder."""

    def test_default_workdir_is_the_system_temp_folder(self) -> None:
        """Without an explicit workdir, options point at the system temp folder."""
        self.assertEqual(make_options().workdir, Path(tempfile.gettempdir()))

    def test_explicit_workdir_and_overrides_are_used(self) -> None:
        """An explicit workdir and other overrides are applied."""
        options = make_options(Path("/projects/x"), timeout_seconds=5)
        self.assertEqual(options.workdir, Path("/projects/x"))
        self.assertEqual(options.timeout_seconds, 5)


if __name__ == "__main__":
    unittest.main()
