"""Unit tests for autopilot.bot: TelegramNotifier and TelegramBot.

Uses a real Store in a temp folder, the FakeTelegramClient and a small
recording FakeEngine, so no network, Claude or engine thread is involved.
"""

from __future__ import annotations

import tempfile
import threading
import unittest
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from autopilot.bot import (
    OFFSET_KEY,
    POLL_BACKOFF_MAX,
    POLL_BACKOFF_START,
    TelegramBot,
    TelegramNotifier,
)
from autopilot.errors import TaskNotFoundError, TelegramError
from autopilot.formatting import TELEGRAM_MAX_CHARS
from autopilot.models import (
    ASK_TASK_NAME,
    BOT_COMMANDS,
    STATUS_ERROR,
    STATUS_SUCCESS,
    TRIGGER_CHAT,
    TRIGGER_TELEGRAM,
    QueueSnapshot,
    TaskConfig,
)
from autopilot.store import Store
from helpers import (
    TZ,
    FakeClock,
    FakeTelegramClient,
    make_config,
    make_record,
    make_task,
)

CHAT_ID = 777
OTHER_CHAT_ID = 999
NOW = datetime(2026, 11, 2, 9, 0, tzinfo=TZ)


class FakeEngine:
    """Records bot calls and creates real queued runs in the store.

    Attributes:
        calls: ``(method, args)`` tuples in call order.
        duplicate: When True, enqueue_task reports an already active run.
    """

    def __init__(self, store: Store, config: TaskConfig) -> None:
        """Remember the store used to create runs and the scripted config."""
        self.store = store
        self._config = config
        self.calls: list[tuple[str, tuple[Any, ...]]] = []
        self.duplicate = False
        self.snapshot_value = QueueSnapshot(running=(), queued=(), limited_until=None)
        self.fire_times: dict[str, datetime | None] = {}

    @property
    def config(self) -> TaskConfig:
        """The scripted config."""
        return self._config

    def enqueue_task(self, name: str, trigger: str, **_: Any) -> int | None:
        """Record the call and create a queued run unless ``duplicate`` is set."""
        self.calls.append(("enqueue_task", (name, trigger)))
        if name not in self._config.tasks:
            raise TaskNotFoundError(name, self._config.tasks)
        if self.duplicate:
            return None
        return self.store.create_run(
            task_name=name, trigger=trigger, prompt="p", queued_at=NOW
        )

    def enqueue_prompt(self, prompt: str, trigger: str = TRIGGER_CHAT, **_: Any) -> int:
        """Record the call and create a queued ask run."""
        self.calls.append(("enqueue_prompt", (prompt, trigger)))
        return self.store.create_run(
            task_name=ASK_TASK_NAME, trigger=trigger, prompt=prompt, queued_at=NOW
        )

    def reset_chat(self) -> None:
        """Record the call."""
        self.calls.append(("reset_chat", ()))

    def next_fire_times(self) -> dict[str, datetime | None]:
        """Return the scripted fire times."""
        return self.fire_times

    def snapshot(self) -> QueueSnapshot:
        """Return the scripted snapshot."""
        return self.snapshot_value

    def methods(self) -> list[str]:
        """Return just the recorded method names."""
        return [name for name, _ in self.calls]


def message_update(
    text: str | None,
    chat_id: int = CHAT_ID,
    update_id: int = 1,
    sender_id: int | None = None,
) -> dict[str, Any]:
    """Build a Telegram message update; the sender defaults to the chat id (a private chat)."""
    sender = chat_id if sender_id is None else sender_id
    message: dict[str, Any] = {"chat": {"id": chat_id}, "from": {"id": sender}}
    if text is not None:
        message["text"] = text
    return {"update_id": update_id, "message": message}


class BotTestCase(unittest.TestCase):
    """Shared fixture: store, client, engine, notifier and bot."""

    def setUp(self) -> None:
        """Build the bot with two tasks, one of them with a finished run."""
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.store = Store(Path(tmp.name) / "db.sqlite")
        self.client = FakeTelegramClient()
        config = make_config(make_task("news"), make_task("plan", description="Day plan"))
        self.engine = FakeEngine(self.store, config)
        self.notifier = TelegramNotifier(self.client, CHAT_ID, local_tz=TZ)
        self.bot = TelegramBot(
            self.client, self.engine, self.store, self.notifier,
            chat_id=CHAT_ID, clock=FakeClock(NOW),
        )

    def send(self, text: str) -> None:
        """Feed one message from the allowed chat to the bot."""
        self.bot.handle_update(message_update(text))

    def texts(self) -> list[str]:
        """Return the texts of all messages sent so far."""
        return [m["text"] for m in self.client.sent]

    def finish_run(self, task: str, result: str, status: str = STATUS_SUCCESS) -> int:
        """Create and finish a run in the store, returning its id."""
        run_id = self.store.create_run(
            task_name=task, trigger=TRIGGER_TELEGRAM, prompt="p", queued_at=NOW
        )
        self.store.finish_run(run_id, status=status, finished_at=NOW, result=result)
        return run_id


class NotifierTests(BotTestCase):
    """TelegramNotifier delivery behavior."""

    def test_notify_run_sends_header_and_result(self) -> None:
        """A successful task run arrives as one HTML message."""
        self.notifier.notify_run(make_record(result="Hello **world**"))
        self.assertEqual(len(self.client.sent), 1)
        self.assertIn("<b>world</b>", self.client.sent[0]["text"])
        self.assertEqual(self.client.sent[0]["chat_id"], CHAT_ID)

    def test_notify_run_attaches_document_for_long_result(self) -> None:
        """A result too long for 3 messages is also sent as a .md file."""
        long_result = "\n\n".join(f"Paragraph {i} " + "x" * 3000 for i in range(8))
        self.notifier.notify_run(make_record(result=long_result))
        self.assertEqual(len(self.client.sent), 1)
        self.assertEqual(len(self.client.documents), 1)
        document = self.client.documents[0]
        self.assertTrue(document["filename"].endswith(".md"))
        self.assertIn("Paragraph 7", document["content"].decode("utf-8"))

    def test_notify_run_without_document_for_short_result(self) -> None:
        """Short results send no file."""
        self.notifier.notify_run(make_record())
        self.assertEqual(self.client.documents, [])

    def test_notify_text_is_plain(self) -> None:
        """notify_text sends without a parse mode so markup characters stay literal."""
        self.notifier.notify_text("a <b> & c")
        self.assertIsNone(self.client.sent[0]["parse_mode"])
        self.assertEqual(self.client.sent[0]["text"], "a <b> & c")

    def test_notify_text_truncates_long_text(self) -> None:
        """A text over Telegram's limit is cut to fit and ends with a note."""
        self.notifier.notify_text("x" * (TELEGRAM_MAX_CHARS + 500))
        sent = self.client.sent[0]["text"]
        self.assertEqual(len(sent), TELEGRAM_MAX_CHARS)
        self.assertTrue(sent.endswith("…(truncated)"))

    def test_notify_text_keeps_text_at_the_limit(self) -> None:
        """A text exactly at the limit is sent unchanged."""
        text = "y" * TELEGRAM_MAX_CHARS
        self.notifier.notify_text(text)
        self.assertEqual(self.client.sent[0]["text"], text)

    def test_send_html_falls_back_to_plain_on_parse_error(self) -> None:
        """A parse error resends the text without tags and without a parse mode."""
        self.client.fail_html_once = True
        with self.assertLogs("autopilot.bot", level="WARNING"):
            self.notifier.send_html("<b>Bold</b> &amp; more")
        self.assertEqual(len(self.client.sent), 1)
        self.assertIsNone(self.client.sent[0]["parse_mode"])
        self.assertEqual(self.client.sent[0]["text"], "Bold & more")

    def test_send_html_reraises_other_errors(self) -> None:
        """Errors that are not parse errors are not swallowed by the fallback."""

        class Failing(FakeTelegramClient):
            """Client whose sends always fail with a 500."""

            def send_message(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
                """Always fail with a 500."""
                raise TelegramError("Telegram sendMessage failed: boom", status=500)

        notifier = TelegramNotifier(Failing(), CHAT_ID)
        with self.assertRaises(TelegramError):
            notifier.send_html("hi")


class FlakyClient(FakeTelegramClient):
    """Client whose next sends raise the queued errors, then behave normally."""

    def __init__(self, errors: list[TelegramError]) -> None:
        """Queue the errors to raise, one per send_message or send_document call."""
        super().__init__()
        self.errors = errors
        self.attempts = 0

    def send_message(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """Raise the next queued error, if any, else record the message."""
        self.attempts += 1
        if self.errors:
            raise self.errors.pop(0)
        return super().send_message(*args, **kwargs)

    def send_document(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """Raise the next queued error, if any, else record the document."""
        self.attempts += 1
        if self.errors:
            raise self.errors.pop(0)
        return super().send_document(*args, **kwargs)


class RetryTests(unittest.TestCase):
    """TelegramNotifier retries one transient send failure."""

    def build(self, errors: list[TelegramError]) -> tuple[TelegramNotifier, FlakyClient, list[float]]:
        """Build a notifier over a flaky client with a recording sleep."""
        client = FlakyClient(errors)
        sleeps: list[float] = []
        return TelegramNotifier(client, CHAT_ID, sleep=sleeps.append), client, sleeps

    def test_network_error_is_retried_once(self) -> None:
        """A network failure (no status) is retried after 5 s and then delivered."""
        notifier, client, sleeps = self.build([TelegramError("network error")])
        with self.assertLogs("autopilot.bot", level="WARNING"):
            notifier.notify_text("hi")
        self.assertEqual(sleeps, [5])
        self.assertEqual(len(client.sent), 1)

    def test_rate_limit_waits_retry_after_capped(self) -> None:
        """A 429 waits retry_after, capped at 30 s."""
        notifier, client, sleeps = self.build([TelegramError("slow", status=429, retry_after=120)])
        with self.assertLogs("autopilot.bot", level="WARNING"):
            notifier.notify_text("hi")
        self.assertEqual(sleeps, [30])
        self.assertEqual(len(client.sent), 1)

    def test_rate_limit_without_retry_after_uses_default(self) -> None:
        """A 429 without retry_after waits the 5 s default."""
        notifier, _, sleeps = self.build([TelegramError("slow", status=429)])
        with self.assertLogs("autopilot.bot", level="WARNING"):
            notifier.notify_text("hi")
        self.assertEqual(sleeps, [5])

    def test_api_error_is_not_retried(self) -> None:
        """A 403 fails immediately without a second attempt or a sleep."""
        notifier, client, sleeps = self.build([TelegramError("forbidden", status=403)])
        with self.assertRaises(TelegramError):
            notifier.notify_text("hi")
        self.assertEqual((client.attempts, sleeps), (1, []))

    def test_second_failure_propagates(self) -> None:
        """Only one retry happens; a second failure reaches the caller."""
        notifier, client, _ = self.build([TelegramError("down"), TelegramError("down")])
        with self.assertLogs("autopilot.bot", level="WARNING"), self.assertRaises(TelegramError):
            notifier.notify_text("hi")
        self.assertEqual(client.attempts, 2)

    def test_document_is_retried(self) -> None:
        """The result file upload gets the same single retry."""
        notifier, client, sleeps = self.build([])
        long_result = "\n\n".join(f"P{i} " + "x" * 3000 for i in range(8))
        client.errors = []
        original = client.send_document
        failed: list[bool] = []

        def flaky_document(*args: Any, **kwargs: Any) -> dict[str, Any]:
            """Fail the first upload with a network error."""
            if not failed:
                failed.append(True)
                raise TelegramError("network error")
            return original(*args, **kwargs)

        client.send_document = flaky_document  # type: ignore[method-assign]
        with self.assertLogs("autopilot.bot", level="WARNING"):
            notifier.notify_run(make_record(result=long_result))
        self.assertEqual(sleeps, [5])
        self.assertEqual(len(client.documents), 1)

    def test_parse_error_fallback_still_works(self) -> None:
        """A parse error is resent as plain text without any sleep."""
        notifier, client, sleeps = self.build([TelegramError("Bad Request: can't parse entities", status=400)])
        with self.assertLogs("autopilot.bot", level="WARNING"):
            notifier.send_html("<b>x</b>")
        self.assertEqual(sleeps, [])
        self.assertEqual(client.sent[0]["text"], "x")


class CommandTests(BotTestCase):
    """Every command and its replies."""

    def test_help_and_start(self) -> None:
        """/help and /start both show the help text."""
        self.send("/help")
        self.send("/start")
        self.assertEqual(len(self.client.sent), 2)
        self.assertEqual(self.texts()[0], self.texts()[1])
        self.assertIn("/tasks", self.texts()[0])

    def test_bot_name_suffix_and_case(self) -> None:
        """'/HELP@MyBot' is treated as /help."""
        self.send("/HELP@MyBot")
        self.assertIn("/tasks", self.texts()[0])

    def test_tasks_lists_each_task(self) -> None:
        """/tasks mentions every task name."""
        self.send("/tasks")
        self.assertIn("news", self.texts()[0])
        self.assertIn("plan", self.texts()[0])

    def test_tasks_shows_paused_and_last_run(self) -> None:
        """/tasks reflects the paused flag from the store."""
        self.store.set_paused("news", True)
        self.finish_run("plan", "done")
        self.send("/tasks")
        self.assertIn("paused", self.texts()[0])

    def test_run_queues_task(self) -> None:
        """/run enqueues with the telegram trigger and confirms the run id."""
        self.send("/run news")
        self.assertEqual(self.engine.calls, [("enqueue_task", ("news", TRIGGER_TELEGRAM))])
        self.assertIn("Queued <b>news</b> (run #1, 0 ahead)", self.texts()[0])

    def test_run_counts_runs_ahead(self) -> None:
        """/run reports active runs created before the new one."""
        self.store.create_run(task_name="plan", trigger="schedule", prompt="p", queued_at=NOW)
        self.send("/run news")
        self.assertIn("1 ahead", self.texts()[0])

    def test_run_without_name_shows_usage(self) -> None:
        """/run alone explains usage and lists tasks without enqueueing."""
        self.send("/run")
        self.assertIn("Usage: /run", self.texts()[0])
        self.assertIn("news, plan", self.texts()[0])
        self.assertEqual(self.engine.calls, [])

    def test_run_unknown_task(self) -> None:
        """/run with an unknown name replies with the TaskNotFoundError message."""
        self.send("/run nope")
        self.assertIn("Unknown task 'nope'. Known tasks: news, plan", self.texts()[0])
        self.assertEqual(self.engine.calls, [])

    def test_run_duplicate(self) -> None:
        """/run on a task that is already active says so."""
        self.engine.duplicate = True
        self.send("/run news")
        self.assertIn("news is already queued or running", self.texts()[0])

    def test_pause_and_resume(self) -> None:
        """/pause and /resume flip the stored flag and confirm."""
        self.send("/pause news")
        self.assertTrue(self.store.is_paused("news"))
        self.send("/resume news")
        self.assertFalse(self.store.is_paused("news"))
        self.assertIn("Paused", self.texts()[0])
        self.assertIn("Resumed", self.texts()[1])

    def test_pause_unknown_task_changes_nothing(self) -> None:
        """/pause with an unknown task replies with the error and stores nothing."""
        self.send("/pause nope")
        self.assertIn("Unknown task 'nope'", self.texts()[0])
        self.assertFalse(self.store.is_paused("nope"))

    def test_pause_without_name_shows_usage(self) -> None:
        """/resume alone explains usage."""
        self.send("/resume")
        self.assertIn("Usage: /resume", self.texts()[0])

    def test_status_uses_snapshot(self) -> None:
        """/status renders the engine snapshot."""
        self.send("/status")
        self.assertIn("Nothing is running", self.texts()[0])

    def test_last_without_runs(self) -> None:
        """/last with no finished runs says so."""
        self.send("/last")
        self.assertEqual(self.texts(), ["No finished runs yet."])

    def test_last_sends_newest_finished_run(self) -> None:
        """/last resends the newest finished run of any task."""
        self.finish_run("news", "old result")
        self.finish_run("plan", "new result")
        self.send("/last")
        self.assertIn("new result", self.texts()[0])

    def test_last_for_task(self) -> None:
        """/last <task> restricts to that task, and errors on unknown tasks."""
        self.finish_run("news", "news result")
        self.finish_run("plan", "plan result")
        self.send("/last news")
        self.assertIn("news result", self.texts()[0])
        self.send("/last nope")
        self.assertIn("Unknown task 'nope'", self.texts()[1])

    def test_last_failed_run_includes_error(self) -> None:
        """/last also resends failed runs, with their error line."""
        run_id = self.store.create_run(task_name="news", trigger="schedule", prompt="p", queued_at=NOW)
        self.store.finish_run(run_id, status=STATUS_ERROR, finished_at=NOW, error="kaboom")
        self.send("/last")
        self.assertIn("kaboom", self.texts()[0])

    def test_unknown_command(self) -> None:
        """An unrecognised command gets the help hint."""
        self.send("/frobnicate")
        self.assertEqual(self.texts(), ["Unknown command. Send /help."])


class SpendCommandTests(BotTestCase):
    """The /spend command writes to the configured balance file."""

    BALANCE = "# Balance\n\nCurrent balance: **100 EUR**\n\n## November 2026\n\n|    Date    | Amount (EUR) | Where | Circumstances \n|------------|------|---------------|------------\n"

    def setUp(self) -> None:
        super().setUp()
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.balance = Path(tmp.name) / "balance.md"
        self.balance.write_text(self.BALANCE, encoding="utf-8")
        self.bot.balance_file = self.balance

    def test_logs_entry_and_confirms_the_new_balance(self) -> None:
        """A valid entry lands in the file under today's month and the reply shows the balance."""
        self.send("/spend 35 Cafe, snack")
        self.assertIn("Logged -35 EUR, Cafe (snack).\nBalance: 65 EUR", self.texts()[-1])
        text = self.balance.read_text(encoding="utf-8")
        self.assertIn("| 2026-11-02 | -35 | Cafe | snack", text)
        self.assertIn("Current balance: **65 EUR**", text)

    def test_income_with_plus(self) -> None:
        """A leading plus adds money."""
        self.send("/spend +200 allowance")
        self.assertIn("Balance: 300 EUR", self.texts()[-1])

    def test_reply_escapes_html(self) -> None:
        """User text cannot inject markup into the Telegram reply."""
        self.send("/spend 5 <b>x</b>")
        self.assertIn("&lt;b&gt;x&lt;/b&gt;", self.texts()[-1])

    def test_bad_input_gets_help_and_leaves_the_file_alone(self) -> None:
        """Missing amount, missing place and an empty command all explain the format."""
        for text in ("/spend", "/spend Cafe", "/spend 50"):
            with self.subTest(text=text):
                self.send(text)
                self.assertIn("/spend", self.texts()[-1])
        self.assertEqual(self.balance.read_text(encoding="utf-8"), self.BALANCE)

    def test_missing_balance_file_is_reported(self) -> None:
        """A balance file that disappeared gives a clear error, not a crash."""
        self.balance.unlink()
        self.send("/spend 5 Cafe")
        self.assertIn("Cannot read the balance file", self.texts()[-1])

    def test_off_without_a_balance_file(self) -> None:
        """Without AUTOPILOT_BALANCE_FILE the command explains how to enable it."""
        self.bot.balance_file = None
        self.send("/spend 5 Cafe")
        self.assertIn("AUTOPILOT_BALANCE_FILE", self.texts()[-1])

    def test_other_chats_cannot_log_money(self) -> None:
        """Only the owner's private chat reaches the handler."""
        self.bot.handle_update(message_update("/spend 5 Cafe", chat_id=CHAT_ID + 1, sender_id=CHAT_ID + 1))
        self.assertEqual(self.balance.read_text(encoding="utf-8"), self.BALANCE)
        self.assertEqual(self.texts(), [])


class ChatTests(BotTestCase):
    """Plain text, /ask and /new."""

    def test_plain_text_enqueues_chat_and_shows_typing(self) -> None:
        """Text with nothing ahead starts a chat run and the typing indicator."""
        self.send("Naïve café résumé ☕")
        self.assertEqual(self.engine.calls, [("enqueue_prompt", ("Naïve café résumé ☕", TRIGGER_CHAT))])
        self.assertEqual(self.client.chat_actions, [(CHAT_ID, "typing")])
        self.assertEqual(self.client.sent, [])

    def test_chat_with_runs_ahead_replies_with_count(self) -> None:
        """Active runs ahead produce the 'N run(s) ahead' reply and no typing."""
        self.store.create_run(task_name="news", trigger="schedule", prompt="p", queued_at=NOW)
        self.store.create_run(task_name="plan", trigger="schedule", prompt="p", queued_at=NOW)
        self.send("hello")
        self.assertEqual(
            self.texts(),
            ["🕐 Got it. 2 run(s) ahead of you, I'll answer when they finish."],
        )
        self.assertEqual(self.client.chat_actions, [])

    def test_chat_while_usage_limited_tells_the_reset_time(self) -> None:
        """With the gate closed the user gets the local reset time instead of typing."""
        until = datetime(2026, 11, 2, 12, 30, tzinfo=timezone.utc)
        self.engine.snapshot_value = QueueSnapshot(running=(), queued=(), limited_until=until)
        self.send("hello")
        self.assertEqual(self.engine.methods(), ["enqueue_prompt"])
        self.assertEqual(self.client.chat_actions, [])
        self.assertEqual(
            self.texts(),
            ["⏳ Claude's usage limit is reached until 13:30. Your message is queued and I'll answer then."],
        )

    def test_finished_runs_are_not_ahead(self) -> None:
        """Only queued or running runs count as ahead."""
        self.finish_run("news", "done")
        self.send("hello")
        self.assertEqual(self.client.chat_actions, [(CHAT_ID, "typing")])

    def test_typing_error_is_ignored(self) -> None:
        """A failing typing indicator never breaks the chat flow."""

        def boom(chat_id: int, action: str = "typing") -> None:
            """Fail like a Telegram outage."""
            raise TelegramError("nope", status=500)

        self.client.send_chat_action = boom  # type: ignore[method-assign]
        self.send("hello")
        self.assertEqual(self.engine.methods(), ["enqueue_prompt"])

    def test_ask_command_is_chat(self) -> None:
        """/ask <text> behaves like plain text."""
        self.send("/ask what is up")
        self.assertEqual(self.engine.calls, [("enqueue_prompt", ("what is up", TRIGGER_CHAT))])

    def test_command_argument_split_on_any_whitespace(self) -> None:
        """A newline or tab between command and argument still separates them."""
        self.send("/ask\nhello there")
        self.send("/run\tnews")
        self.assertEqual(
            self.engine.calls,
            [("enqueue_prompt", ("hello there", TRIGGER_CHAT)), ("enqueue_task", ("news", TRIGGER_TELEGRAM))],
        )

    def test_ask_without_text_shows_usage(self) -> None:
        """/ask alone explains usage and queues nothing."""
        self.send("/ask")
        self.assertIn("Usage: /ask", self.texts()[0])
        self.assertEqual(self.engine.calls, [])

    def test_new_without_text(self) -> None:
        """/new resets the conversation and greets."""
        self.send("/new")
        self.assertEqual(self.engine.methods(), ["reset_chat"])
        self.assertEqual(self.texts(), ["🆕 New conversation. What's on your mind?"])

    def test_new_with_text(self) -> None:
        """/new <text> resets first, then sends the text as a chat turn."""
        self.send("/new tell me a joke")
        self.assertEqual(self.engine.methods(), ["reset_chat", "enqueue_prompt"])
        self.assertEqual(self.engine.calls[1][1], ("tell me a joke", TRIGGER_CHAT))


class FilteringTests(BotTestCase):
    """Which updates are ignored."""

    def test_foreign_chat_is_ignored_and_logged(self) -> None:
        """Another chat gets no reply, no engine call, and its id is logged."""
        with self.assertLogs("autopilot.bot", level="WARNING") as logs:
            self.bot.handle_update(message_update("/run news", chat_id=OTHER_CHAT_ID))
        self.assertIn(str(OTHER_CHAT_ID), logs.output[0])
        self.assertEqual(self.client.sent, [])
        self.assertEqual(self.engine.calls, [])

    def test_group_chat_is_ignored(self) -> None:
        """A group (negative chat id) never commands the runner, even from the owner."""
        with self.assertLogs("autopilot.bot", level="WARNING") as logs:
            self.bot.handle_update(message_update("/run news", chat_id=-100123, sender_id=CHAT_ID))
        self.assertIn("-100123", logs.output[0])
        self.assertEqual(self.client.sent, [])
        self.assertEqual(self.engine.calls, [])

    def test_foreign_sender_in_owner_chat_is_ignored(self) -> None:
        """A sender other than the owner is ignored even inside the configured chat id."""
        with self.assertLogs("autopilot.bot", level="WARNING"):
            self.bot.handle_update(message_update("hello", chat_id=CHAT_ID, sender_id=OTHER_CHAT_ID))
        self.assertEqual(self.client.sent, [])
        self.assertEqual(self.engine.calls, [])

    def test_message_without_sender_is_ignored(self) -> None:
        """A message with no 'from' field cannot be attributed to the owner."""
        update = message_update("hello")
        del update["message"]["from"]
        with self.assertLogs("autopilot.bot", level="WARNING"):
            self.bot.handle_update(update)
        self.assertEqual(self.engine.calls, [])

    def test_updates_without_text_are_ignored(self) -> None:
        """Photos, edits and non-message updates produce nothing."""
        self.bot.handle_update(message_update(None))
        self.bot.handle_update({"update_id": 2, "edited_message": {"text": "x"}})
        self.bot.handle_update(message_update("   "))
        self.assertEqual(self.client.sent, [])
        self.assertEqual(self.engine.calls, [])


class StopAfterWaits(threading.Event):
    """Event whose wait() records the delay and sets itself after N waits."""

    def __init__(self, stop_after: int) -> None:
        """Remember how many waits are allowed before stopping."""
        super().__init__()
        self.stop_after = stop_after
        self.delays: list[float] = []

    def wait(self, timeout: float | None = None) -> bool:
        """Record the delay instead of sleeping; stop once enough waits happened."""
        self.delays.append(timeout or 0)
        if len(self.delays) >= self.stop_after:
            self.set()
        return self.is_set()


class PollingTests(BotTestCase):
    """poll_forever: commands, offsets, errors and backoff."""

    def test_registers_commands_and_handles_updates(self) -> None:
        """Commands are registered once and each update is handled."""
        stop = threading.Event()
        self.client.updates = [[message_update("/help", update_id=10)]]
        original = self.bot.handle_update

        def handle_then_stop(update: dict[str, Any]) -> None:
            """Handle the update normally, then end the poll loop."""
            original(update)
            stop.set()

        self.bot.handle_update = handle_then_stop  # type: ignore[method-assign]
        self.bot.poll_forever(stop)
        self.assertEqual(self.client.commands, list(BOT_COMMANDS))
        self.assertEqual(len(self.client.sent), 1)

    def test_offset_persisted_per_update(self) -> None:
        """The offset is stored after each update, even if a later one fails."""
        stop = threading.Event()
        seen: list[str | None] = []

        def handle(update: dict[str, Any]) -> None:
            """Record the stored offset, then fail on update 6 after stopping."""
            seen.append(self.store.get_value(OFFSET_KEY))
            if update["update_id"] == 6:
                stop.set()
                raise RuntimeError("bad update")

        self.client.updates = [[message_update("a", update_id=5), message_update("b", update_id=6)]]
        self.bot.handle_update = handle  # type: ignore[method-assign]
        with self.assertLogs("autopilot.bot", level="ERROR"):
            self.bot.poll_forever(stop)
        self.assertEqual(seen, [None, "6"])
        self.assertEqual(self.store.get_value(OFFSET_KEY), "7")

    def test_handler_failure_replies_and_still_stores_offset(self) -> None:
        """A crashing handler gets a best-effort apology and the update is not replayed."""
        stop = threading.Event()
        self.engine.enqueue_prompt = self._raise_runtime  # type: ignore[method-assign]
        self.client.updates = [[message_update("hello", update_id=8)]]
        original_get = self.client.get_updates

        def get_updates(offset: int | None = None, timeout: int = 50) -> list[dict[str, Any]]:
            """Serve the scripted batch, then stop the loop on the next call."""
            batch = original_get(offset, timeout)
            if not batch:
                stop.set()
            return batch

        self.client.get_updates = get_updates  # type: ignore[method-assign]
        with self.assertLogs("autopilot.bot", level="ERROR"):
            self.bot.poll_forever(stop)
        self.assertEqual(self.texts(), ["⚠️ Something went wrong with that message. Please try again."])
        self.assertEqual(self.store.get_value(OFFSET_KEY), "9")

    def test_failed_apology_is_ignored(self) -> None:
        """If the apology itself cannot be sent, the loop still stores the offset."""
        self.engine.enqueue_prompt = self._raise_runtime  # type: ignore[method-assign]
        self.client.send_message = self._raise_telegram  # type: ignore[method-assign]
        with self.assertLogs("autopilot.bot", level="WARNING"):
            self.bot._process(message_update("hello", update_id=3))
        self.assertEqual(self.store.get_value(OFFSET_KEY), "4")

    @staticmethod
    def _raise_runtime(*args: Any, **kwargs: Any) -> int:
        """Stand-in for an engine call that crashes."""
        raise RuntimeError("boom")

    @staticmethod
    def _raise_telegram(*args: Any, **kwargs: Any) -> dict[str, Any]:
        """Stand-in for a send that fails with a non-retried API error."""
        raise TelegramError("forbidden", status=403)

    def test_stored_offset_is_passed_to_get_updates(self) -> None:
        """A persisted offset is used for the next poll."""
        self.store.set_value(OFFSET_KEY, "41")
        stop = threading.Event()
        offsets: list[int | None] = []

        def get_updates(offset: int | None = None, timeout: int = 50) -> list[dict[str, Any]]:
            """Record the offset, stop the loop and return no updates."""
            offsets.append(offset)
            stop.set()
            return []

        self.client.get_updates = get_updates  # type: ignore[method-assign]
        self.bot.poll_forever(stop)
        self.assertEqual(offsets, [41])

    def test_backoff_doubles_up_to_max(self) -> None:
        """Repeated Telegram errors wait 5, 10, 20, 40, 60, 60 seconds."""
        self.client.raise_on_updates = TelegramError("down", status=502)
        stop = StopAfterWaits(6)
        with self.assertLogs("autopilot.bot", level="WARNING"):
            self.bot.poll_forever(stop)
        self.assertEqual(stop.delays, [POLL_BACKOFF_START, 10, 20, 40, POLL_BACKOFF_MAX, POLL_BACKOFF_MAX])

    def test_retry_after_wins_and_success_resets_backoff(self) -> None:
        """retry_after overrides the backoff; a good poll restarts it at 5."""
        errors = [
            TelegramError("slow down", status=429, retry_after=17),
            None,
            TelegramError("down", status=502),
        ]
        stop = StopAfterWaits(2)

        def get_updates(offset: int | None = None, timeout: int = 50) -> list[dict[str, Any]]:
            """Raise or return the next scripted outcome."""
            error = errors.pop(0)
            if error is not None:
                raise error
            return []

        self.client.get_updates = get_updates  # type: ignore[method-assign]
        with self.assertLogs("autopilot.bot", level="WARNING"):
            self.bot.poll_forever(stop)
        self.assertEqual(stop.delays, [17, POLL_BACKOFF_START])

    def test_set_my_commands_error_is_not_fatal(self) -> None:
        """A failing command registration is logged and polling still runs."""

        def boom(commands: Any) -> None:
            """Fail like a rejected setMyCommands call."""
            raise TelegramError("no", status=500)

        self.client.set_my_commands = boom  # type: ignore[method-assign]
        stop = threading.Event()
        stop.set()
        with self.assertLogs("autopilot.bot", level="WARNING"):
            self.bot.poll_forever(stop)

    def test_store_error_does_not_end_polling(self) -> None:
        """A transient store error is logged, backed off, and polling resumes."""
        waits = StopAfterWaits(5)
        real_get = self.store.get_value
        calls: list[int] = []

        def flaky_get(key: str) -> str | None:
            """Fail the first read like a locked database, then work."""
            calls.append(1)
            if len(calls) == 1:
                raise RuntimeError("database is locked")
            waits.set()
            return real_get(key)

        self.store.get_value = flaky_get  # type: ignore[method-assign]
        with self.assertLogs("autopilot.bot", level="WARNING"):
            self.bot.poll_forever(waits)
        self.assertEqual(waits.delays, [POLL_BACKOFF_START])
        self.assertGreaterEqual(len(calls), 2)

    def test_offset_write_failure_does_not_replay_update(self) -> None:
        """If storing the offset fails, the next poll still asks past that update."""
        stop = StopAfterWaits(99)
        offsets: list[int | None] = []
        handled: list[int] = []
        batches = [[message_update("a", update_id=5)]]

        def get_updates(offset: int | None = None, timeout: int = 50) -> list[dict[str, Any]]:
            """Record the requested offset and serve the scripted batches."""
            offsets.append(offset)
            if len(offsets) == 2:
                stop.set()
            return batches.pop(0) if batches else []

        def failing_set(key: str, value: str | None) -> None:
            """Fail like a locked database."""
            raise RuntimeError("database is locked")

        self.client.get_updates = get_updates  # type: ignore[method-assign]
        self.bot.handle_update = lambda update: handled.append(update["update_id"])  # type: ignore[method-assign]
        self.store.set_value = failing_set  # type: ignore[method-assign]
        with self.assertLogs("autopilot.bot", level="WARNING"):
            self.bot.poll_forever(stop)
        self.assertEqual(handled, [5])
        self.assertEqual(offsets[1], 6)

    def test_unreadable_stored_offset_is_ignored(self) -> None:
        """A corrupt offset value falls back to no offset."""
        self.store.set_value(OFFSET_KEY, "abc")
        stop = threading.Event()
        offsets: list[int | None] = []

        def get_updates(offset: int | None = None, timeout: int = 50) -> list[dict[str, Any]]:
            """Record the offset, stop the loop and return no updates."""
            offsets.append(offset)
            stop.set()
            return []

        self.client.get_updates = get_updates  # type: ignore[method-assign]
        with self.assertLogs("autopilot.bot", level="WARNING"):
            self.bot.poll_forever(stop)
        self.assertEqual(offsets, [None])


if __name__ == "__main__":
    unittest.main()
