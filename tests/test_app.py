"""Unit tests for autopilot.app.

Covers log redaction and logging setup, build_app wiring (with and without
Telegram), run_service on a worker thread, the telegram-setup flow (happy path,
timeout, 409 conflict, bad token) and init_project. Nothing touches the real
network, systemd, ~/.config or the real Claude CLI: Claude is tests/fake_claude.py,
Telegram is FakeTelegramClient and all paths live in temp folders.
"""

from __future__ import annotations

import contextlib
import io
import logging
import os
import re
import signal
import stat
import sys
import tempfile
import threading
import time
import unittest
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

from dotenv import dotenv_values

from autopilot.app import (
    App,
    RedactFilter,
    build_app,
    configure_logging,
    init_project,
    run_service,
    telegram_setup,
)
from autopilot.bot import OFFSET_KEY, TelegramNotifier
from autopilot.engine import LogNotifier
from autopilot.errors import AutopilotError, ClaudeNotFoundError, ConfigError, TelegramError
from autopilot.runner import ClaudeRunner
from autopilot.settings import Settings, load_settings
from autopilot.store import Store
from autopilot.telegram_api import TelegramClient
from helpers import FAKE_CLAUDE, TZ, FakeClock, FakeTelegramClient

TOKEN = "123456789:" + "B" * 35
SETUP_CODE = "493021"
VALID_TASKS = '[[task]]\nname = "news"\nprompt = "Say hi"\nschedule = "daily 07:30"\n'
WAIT_SECONDS = 10.0


class AppTestCase(unittest.TestCase):
    """Base class with a temp home, a settings factory and a tasks file helper."""

    def setUp(self) -> None:
        """Create a temp home removed after the test."""
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.home = Path(tmp.name)

    def make_settings(self, **env: str) -> Settings:
        """Build Settings in the temp home using the fake Claude binary."""
        values = {"CLAUDE_BIN": str(FAKE_CLAUDE), "AUTOPILOT_TIMEZONE": "Europe/Prague"}
        values.update(env)
        return load_settings(env=values, home=self.home, read_env_file=False)

    def write_tasks(self, settings: Settings, text: str = VALID_TASKS) -> None:
        """Write tasks.toml for the settings."""
        settings.tasks_file.write_text(text, encoding="utf-8")


class RedactFilterTests(unittest.TestCase):
    """RedactFilter."""

    def record(self, msg: str, *args: object, exc_info: object = None) -> logging.LogRecord:
        """Build a log record."""
        return logging.LogRecord("t", logging.INFO, __file__, 1, msg, args or None, exc_info)  # type: ignore[arg-type]

    def test_secret_in_message_and_args_is_replaced(self) -> None:
        """A secret is hidden whether it is in the format string or in its arguments."""
        flt = RedactFilter([TOKEN])
        record = self.record("token %s and " + TOKEN, TOKEN)
        self.assertTrue(flt.filter(record))
        self.assertEqual(record.getMessage(), "token *** and ***")

    def test_secret_in_traceback_is_replaced(self) -> None:
        """A secret inside an exception message does not survive in the traceback text."""
        try:
            raise RuntimeError(f"failed for {TOKEN}")
        except RuntimeError:
            record = self.record("oops", exc_info=sys.exc_info())
        RedactFilter([TOKEN]).filter(record)
        text = logging.Formatter("%(message)s").format(record)
        self.assertNotIn(TOKEN, text)
        self.assertIn("RuntimeError", text)

    def test_none_and_empty_secrets_are_ignored(self) -> None:
        """None or empty secrets never blank out the message."""
        record = self.record("plain text")
        RedactFilter([None, ""]).filter(record)
        self.assertEqual(record.getMessage(), "plain text")


class ConfigureLoggingTests(AppTestCase):
    """configure_logging."""

    def setUp(self) -> None:
        """Remember the root logger state and restore it after the test."""
        super().setUp()
        root = logging.getLogger()
        saved_handlers, saved_level = list(root.handlers), root.level

        def restore() -> None:
            """Remove handlers the test added and put the old level back."""
            for handler in list(root.handlers):
                if handler not in saved_handlers:
                    root.removeHandler(handler)
                    handler.close()
            root.setLevel(saved_level)

        self.addCleanup(restore)

    def test_token_never_reaches_the_log_file(self) -> None:
        """A log line containing the token is written with *** instead."""
        settings = self.make_settings(TELEGRAM_BOT_TOKEN=TOKEN, TELEGRAM_CHAT_ID="5")
        # The stderr handler binds sys.stderr at creation; keep the test output clean.
        with contextlib.redirect_stderr(io.StringIO()):
            configure_logging(settings)
        logging.getLogger("autopilot.test").error("request failed for %s", TOKEN)
        for handler in logging.getLogger().handlers:
            handler.flush()
        content = settings.log_file.read_text(encoding="utf-8")
        self.assertIn("request failed for ***", content)
        self.assertNotIn(TOKEN, content)

    def test_log_file_and_data_folder_are_private(self) -> None:
        """The log file is 0600 and a data folder created for it is 0700."""
        settings = self.make_settings()
        with contextlib.redirect_stderr(io.StringIO()):
            configure_logging(settings)
        self.assertEqual(stat.S_IMODE(settings.log_file.stat().st_mode), 0o600)
        self.assertEqual(stat.S_IMODE(settings.data_dir.stat().st_mode), 0o700)

    def test_debug_flag_and_level(self) -> None:
        """The configured level applies, and debug=True overrides it."""
        settings = self.make_settings(AUTOPILOT_LOG_LEVEL="WARNING")
        configure_logging(settings, to_file=False)
        self.assertEqual(logging.getLogger().level, logging.WARNING)
        configure_logging(settings, debug=True, to_file=False)
        self.assertEqual(logging.getLogger().level, logging.DEBUG)

    def test_second_call_does_not_duplicate_handlers(self) -> None:
        """Calling twice leaves one stderr handler and one file handler."""
        settings = self.make_settings()
        configure_logging(settings)
        before = len(logging.getLogger().handlers)
        configure_logging(settings)
        self.assertEqual(len(logging.getLogger().handlers), before)

    def test_no_file_when_disabled(self) -> None:
        """to_file=False writes no log file."""
        settings = self.make_settings()
        configure_logging(settings, to_file=False)
        self.assertFalse(settings.log_file.exists())


class BuildAppTests(AppTestCase):
    """build_app."""

    def test_without_telegram_uses_log_notifier(self) -> None:
        """No Telegram settings: log notifier, no client, no bot; the default runner is Claude."""
        settings = self.make_settings()
        self.write_tasks(settings)
        app = build_app(settings)
        self.assertIsInstance(app.notifier, LogNotifier)
        self.assertIsNone(app.client)
        self.assertIsNone(app.bot)
        self.assertIsInstance(app.engine._runner, ClaudeRunner)
        self.assertEqual(list(app.engine.config.tasks), ["news"])
        self.assertTrue(settings.db_path.exists())

    def test_env_file_is_protected_from_claude(self) -> None:
        """Every loaded task denies reading this installation's .env by absolute path."""
        settings = self.make_settings()
        self.write_tasks(settings)
        rule = f"Read(/{settings.env_file})"
        app = build_app(settings)
        self.assertIn(rule, app.engine.config.tasks["news"].options.disallowed_tools)
        self.assertIn(rule, app.engine.config.ask_options.disallowed_tools)
        self.assertIn(f"Read(/{settings.db_path})", app.engine.config.ask_options.disallowed_tools)

    def test_with_telegram_builds_notifier_and_bot(self) -> None:
        """Telegram settings: a notifier with the local zone, plus the bot."""
        settings = self.make_settings(TELEGRAM_BOT_TOKEN=TOKEN, TELEGRAM_CHAT_ID="77")
        self.write_tasks(settings)
        fake = FakeTelegramClient()
        app = build_app(settings, client=fake)  # type: ignore[arg-type]
        self.assertIsInstance(app.notifier, TelegramNotifier)
        self.assertEqual(app.notifier.chat_id, 77)
        self.assertEqual(app.notifier.local_tz, settings.timezone)
        self.assertIs(app.client, fake)
        self.assertIsNotNone(app.bot)

    def test_real_client_is_built_from_the_token(self) -> None:
        """Without an injected client a TelegramClient is created (no call is made)."""
        settings = self.make_settings(TELEGRAM_BOT_TOKEN=TOKEN, TELEGRAM_CHAT_ID="5")
        self.write_tasks(settings)
        self.assertIsInstance(build_app(settings).client, TelegramClient)

    def test_notify_false_skips_telegram(self) -> None:
        """notify=False keeps results local even when Telegram is configured."""
        settings = self.make_settings(TELEGRAM_BOT_TOKEN=TOKEN, TELEGRAM_CHAT_ID="5")
        self.write_tasks(settings)
        app = build_app(settings, notify=False, client=FakeTelegramClient())  # type: ignore[arg-type]
        self.assertIsInstance(app.notifier, LogNotifier)
        self.assertIsNone(app.bot)
        self.assertIsNone(app.client)

    def test_missing_claude_fails_up_front(self) -> None:
        """A missing Claude binary stops build_app before anything runs."""
        settings = self.make_settings(CLAUDE_BIN=str(self.home / "nope"))
        self.write_tasks(settings)
        with self.assertRaises(ClaudeNotFoundError):
            build_app(settings)

    def test_invalid_tasks_file_propagates(self) -> None:
        """A missing or broken tasks.toml raises ConfigError."""
        settings = self.make_settings()
        with self.assertRaises(ConfigError):
            build_app(settings)

    def test_injected_runner_and_clock_are_used(self) -> None:
        """An injected runner skips the binary lookup and the clock reaches the engine."""
        settings = self.make_settings(CLAUDE_BIN=str(self.home / "nope"))
        self.write_tasks(settings)
        runner = object()
        clock = FakeClock(datetime(2026, 11, 2, 7, 30, tzinfo=TZ))
        app = build_app(settings, runner=runner, clock=clock)  # type: ignore[arg-type]
        self.assertIs(app.engine._runner, runner)
        self.assertIs(app.engine._clock, clock)


class RunServiceTests(AppTestCase):
    """run_service."""

    def start_in_thread(self, app: App, stop: threading.Event) -> threading.Thread:
        """Run the service on a worker thread."""
        thread = threading.Thread(target=run_service, args=(app, stop), daemon=True)
        thread.start()
        return thread

    def wait_for(self, condition: Callable[[], bool]) -> None:
        """Poll until the condition holds or the test times out."""
        deadline = time.monotonic() + WAIT_SECONDS
        while not condition() and time.monotonic() < deadline:
            time.sleep(0.02)
        self.assertTrue(condition(), "timed out waiting")

    def test_announces_start_runs_bot_and_stops(self) -> None:
        """The service sends the running notice, starts the bot and ends when stop is set."""
        settings = self.make_settings(TELEGRAM_BOT_TOKEN=TOKEN, TELEGRAM_CHAT_ID="5")
        self.write_tasks(settings)
        fake = FakeTelegramClient()
        app = build_app(settings, client=fake)  # type: ignore[arg-type]
        stop = threading.Event()
        thread = self.start_in_thread(app, stop)
        try:
            self.wait_for(lambda: bool(fake.sent) and fake.commands is not None)
        finally:
            stop.set()
            thread.join(WAIT_SECONDS)
        self.assertFalse(thread.is_alive())
        text = fake.sent[0]["text"]
        self.assertTrue(text.startswith("🟢 Autopilot is running · 1 tasks · next: news at "), text)

    def test_interrupted_runs_are_recovered_at_start(self) -> None:
        """A run left queued by a previous process is marked failed when the service starts."""
        settings = self.make_settings()
        self.write_tasks(settings)
        app = build_app(settings)
        run_id = app.store.create_run(
            task_name="news", trigger="manual", prompt="x", queued_at=datetime(2026, 11, 2, 7, 0, tzinfo=TZ)
        )
        stop = threading.Event()
        stop.set()
        self.start_in_thread(app, stop).join(WAIT_SECONDS)
        record = app.store.get_run(run_id)
        self.assertEqual(record.status, "error")
        self.assertIn("Interrupted", record.error)

    def test_without_telegram_or_next_run(self) -> None:
        """A manual-only task list works with the log notifier and no 'next' part."""
        settings = self.make_settings()
        self.write_tasks(settings, '[[task]]\nname = "a"\nprompt = "x"\n')
        app = build_app(settings)
        stop = threading.Event()
        stop.set()
        thread = self.start_in_thread(app, stop)
        thread.join(WAIT_SECONDS)
        self.assertFalse(thread.is_alive())

    def test_failing_notice_does_not_stop_the_service(self) -> None:
        """An error while sending the start notice is logged and the loop still runs."""
        settings = self.make_settings(TELEGRAM_BOT_TOKEN=TOKEN, TELEGRAM_CHAT_ID="5")
        self.write_tasks(settings)
        fake = FakeTelegramClient()
        app = build_app(settings, client=fake)  # type: ignore[arg-type]

        def explode(text: str) -> None:
            """Fail like a Telegram outage."""
            raise TelegramError("down")

        app.notifier.notify_text = explode  # type: ignore[method-assign]
        stop = threading.Event()
        stop.set()
        with self.assertLogs("autopilot.app", level="ERROR"):
            thread = self.start_in_thread(app, stop)
            thread.join(WAIT_SECONDS)
        self.assertFalse(thread.is_alive())

    def test_main_thread_installs_and_restores_signal_handlers(self) -> None:
        """On the main thread SIGTERM sets stop during the run and the old handler returns after."""
        settings = self.make_settings()
        self.write_tasks(settings)
        app = build_app(settings)
        before = signal.getsignal(signal.SIGTERM)
        stop = threading.Event()
        timer = threading.Timer(0.3, os.kill, args=(os.getpid(), signal.SIGTERM))
        timer.start()
        run_service(app, stop)
        timer.join()
        self.assertTrue(stop.is_set())
        self.assertEqual(signal.getsignal(signal.SIGTERM), before)


class TickingClock:
    """Monotonic clock fake that advances by a fixed step on every call."""

    def __init__(self, step: float) -> None:
        """Start at zero."""
        self.value = 0.0
        self.step = step

    def __call__(self) -> float:
        """Return the current value, then advance."""
        current = self.value
        self.value += self.step
        return current


class TelegramSetupTests(AppTestCase):
    """telegram_setup."""

    def message_update(
        self, update_id: int, chat_id: int, chat_type: str = "private", first_name: str = "Alex", text: str = SETUP_CODE
    ) -> dict:
        """Build a message update from a user whose private chat id equals the user id."""
        return {
            "update_id": update_id,
            "message": {
                "chat": {"id": chat_id, "type": chat_type},
                "from": {"id": chat_id, "first_name": first_name, "username": "alex_example"},
                "text": text,
            },
        }

    def run_setup(
        self, settings: Settings, fake: FakeTelegramClient, **kwargs: object
    ) -> tuple[int, str]:
        """Call telegram_setup with injected collaborators (answering "y"); return (chat id, output)."""
        out = io.StringIO()
        defaults: dict[str, object] = {
            "secret_input": lambda prompt: TOKEN,
            "out": out,
            "client_factory": lambda token: fake,
            "wait_seconds": 120,
            "monotonic": TickingClock(1.0),
            "confirm": lambda prompt: "y",
            "code_factory": lambda: SETUP_CODE,
        }
        defaults.update(kwargs)
        chat_id = telegram_setup(settings, **defaults)  # type: ignore[arg-type]
        return chat_id, out.getvalue()

    def test_happy_path_saves_everything(self) -> None:
        """Token and chat id land in a 0600 .env, the offset is stored and a test message is sent."""
        settings = self.make_settings()
        fake = FakeTelegramClient()
        fake.updates = [[], [self.message_update(7, 555)]]
        chat_id, output = self.run_setup(settings, fake)
        self.assertEqual(chat_id, 555)
        values = dotenv_values(settings.env_file)
        self.assertEqual(values["TELEGRAM_BOT_TOKEN"], TOKEN)
        self.assertEqual(values["TELEGRAM_CHAT_ID"], "555")
        self.assertEqual(stat.S_IMODE(settings.env_file.stat().st_mode), 0o600)
        self.assertEqual(Store(settings.db_path).get_value(OFFSET_KEY), "8")
        self.assertEqual(fake.sent[0]["chat_id"], 555)
        self.assertIn("connected", fake.sent[0]["text"])
        self.assertIn("https://t.me/autopilot_test_bot", output)
        self.assertIn("Alex (@alex_example), chat id 555", output)
        self.assertIn(f"send the code {SETUP_CODE}", output)
        self.assertNotIn(TOKEN, output)

    def test_stale_pending_update_is_skipped(self) -> None:
        """A message that was waiting before setup started is never offered or saved."""
        settings = self.make_settings()
        fake = FakeTelegramClient()
        fake.updates = [[self.message_update(3, 111, first_name="Stranger")], [self.message_update(4, 222)]]
        asked: list[str] = []
        chat_id, output = self.run_setup(settings, fake, confirm=lambda prompt: asked.append(prompt) or "y")
        self.assertEqual(chat_id, 222)
        self.assertEqual(len(asked), 1)
        self.assertNotIn("Stranger", output)
        self.assertEqual(Store(settings.db_path).get_value(OFFSET_KEY), "5")

    def test_message_without_the_code_is_skipped_and_name_sanitized(self) -> None:
        """A stranger's message is never offered; its name is printed without escape codes."""
        settings = self.make_settings()
        fake = FakeTelegramClient()
        evil = self.message_update(5, 900, first_name="\x1b[2JEvil\x07", text="hi")
        fake.updates = [[], [evil], [self.message_update(6, 42, text=f" {SETUP_CODE} \n")]]
        asked: list[str] = []
        chat_id, output = self.run_setup(settings, fake, confirm=lambda prompt: asked.append(prompt) or "y")
        self.assertEqual(chat_id, 42)
        self.assertEqual(len(asked), 1)
        self.assertIn("without the code from [2JEvil (@alex_example)", output)
        self.assertNotIn("\x1b", output)
        self.assertNotIn("\x07", output)

    def test_default_code_is_six_random_digits(self) -> None:
        """Without code_factory the printed code is six digits and is what the bot waits for."""
        settings = self.make_settings()
        fake = FakeTelegramClient()
        out = io.StringIO()
        match_holder: list[str] = []

        def confirm(prompt: str) -> str:
            """Record the code shown to the user, then agree."""
            match_holder.extend(re.findall(r"send the code (\d{6}) ", out.getvalue()))
            return "y"

        fake.updates = [[], []]
        original = fake.get_updates
        calls = {"n": 0}

        def get_updates(offset: int | None = None, timeout: int = 50) -> list[dict]:
            """Return the code message once the code has been printed."""
            calls["n"] += 1
            shown = re.findall(r"send the code (\d{6}) ", out.getvalue())
            if shown and calls["n"] > 1:
                return [self.message_update(9, 77, text=shown[0])]
            return original(offset, timeout)

        fake.get_updates = get_updates  # type: ignore[method-assign]
        chat_id = telegram_setup(
            settings,
            secret_input=lambda prompt: TOKEN,
            out=out,
            client_factory=lambda token: fake,
            monotonic=TickingClock(1.0),
            confirm=confirm,
        )
        self.assertEqual(chat_id, 77)
        self.assertEqual(len(match_holder), 1)

    def test_no_terminal_gives_a_friendly_error(self) -> None:
        """EOF on the confirmation prompt becomes an AutopilotError, not a traceback."""
        fake = FakeTelegramClient()
        fake.updates = [[], [self.message_update(7, 555)]]

        def no_terminal(prompt: str) -> str:
            """Behave like input() with stdin closed."""
            raise EOFError

        with self.assertRaises(AutopilotError) as caught:
            self.run_setup(self.make_settings(), fake, confirm=no_terminal)
        self.assertIn("interactive terminal", str(caught.exception))

    def test_group_chat_is_refused_with_a_note(self) -> None:
        """A group message is never offered; a note is printed and waiting continues."""
        settings = self.make_settings()
        fake = FakeTelegramClient()
        fake.updates = [[], [self.message_update(5, -900, chat_type="group")], [self.message_update(6, 42)]]
        chat_id, output = self.run_setup(settings, fake)
        self.assertEqual(chat_id, 42)
        self.assertIn("group or channel", output)
        self.assertNotIn("-900", output)

    def test_answer_no_does_not_save_and_times_out(self) -> None:
        """Answering "n" saves no chat id; with nothing else arriving the setup times out."""
        settings = self.make_settings()
        fake = FakeTelegramClient()
        fake.updates = [[], [self.message_update(7, 555)]]
        with self.assertRaises(AutopilotError):
            self.run_setup(settings, fake, confirm=lambda prompt: "n", monotonic=TickingClock(10.0))
        self.assertNotIn("TELEGRAM_CHAT_ID", dotenv_values(settings.env_file))
        self.assertEqual(fake.sent, [])

    def test_no_then_yes_uses_the_second_message(self) -> None:
        """After a refusal the next confirmed message is the one saved."""
        settings = self.make_settings()
        fake = FakeTelegramClient()
        fake.updates = [[], [self.message_update(7, 555)], [self.message_update(8, 666)]]
        answers = iter(["n", "YES"])
        chat_id, _ = self.run_setup(settings, fake, confirm=lambda prompt: next(answers))
        self.assertEqual(chat_id, 666)

    def test_default_output_is_resolved_at_call_time(self) -> None:
        """Without `out`, progress goes to the sys.stdout in force when the function runs."""
        settings = self.make_settings()
        fake = FakeTelegramClient()
        fake.updates = [[], [self.message_update(1, 5)]]
        captured = io.StringIO()
        with contextlib.redirect_stdout(captured):
            telegram_setup(
                settings,
                secret_input=lambda prompt: TOKEN,
                client_factory=lambda token: fake,
                monotonic=TickingClock(1.0),
                confirm=lambda prompt: "y",
                code_factory=lambda: SETUP_CODE,
            )
        self.assertIn("https://t.me/", captured.getvalue())

    def test_existing_token_is_not_prompted(self) -> None:
        """A token already in the settings is used without asking."""
        settings = self.make_settings(TELEGRAM_BOT_TOKEN=TOKEN)
        fake = FakeTelegramClient()
        fake.updates = [[], [self.message_update(1, 100)]]

        def refuse(prompt: str) -> str:
            """Fail the test if the token is asked for."""
            raise AssertionError("must not prompt")

        chat_id, _ = self.run_setup(settings, fake, secret_input=refuse)
        self.assertEqual(chat_id, 100)

    def test_updates_without_a_message_are_skipped(self) -> None:
        """An update that is not a message is passed over and the next batch is read."""
        settings = self.make_settings()
        fake = FakeTelegramClient()
        fake.updates = [[], [{"update_id": 3, "edited_message": {}}], [self.message_update(4, 9)]]
        chat_id, _ = self.run_setup(settings, fake)
        self.assertEqual(chat_id, 9)
        self.assertEqual(Store(settings.db_path).get_value(OFFSET_KEY), "5")

    def test_timeout_raises_and_keeps_chat_id_unset(self) -> None:
        """No message in time gives a friendly error and no chat id in .env."""
        settings = self.make_settings()
        with self.assertRaises(AutopilotError) as caught:
            self.run_setup(settings, FakeTelegramClient(), wait_seconds=30, monotonic=TickingClock(10.0))
        self.assertIn("30 s", str(caught.exception))
        self.assertNotIn("TELEGRAM_CHAT_ID", dotenv_values(settings.env_file))

    def test_conflict_explains_the_other_poller(self) -> None:
        """HTTP 409 means another process polls the bot: the message says how to stop it."""
        fake = FakeTelegramClient()
        fake.raise_on_updates = TelegramError("conflict", status=409)
        with self.assertRaises(AutopilotError) as caught:
            self.run_setup(self.make_settings(), fake)
        self.assertIn("systemctl --user stop autopilot", str(caught.exception))

    def test_other_telegram_errors_propagate(self) -> None:
        """Errors other than 409 reach the caller unchanged."""
        fake = FakeTelegramClient()
        fake.raise_on_updates = TelegramError("boom", status=500)
        with self.assertRaises(TelegramError):
            self.run_setup(self.make_settings(), fake)

    def test_bad_token_is_rejected_before_saving(self) -> None:
        """A malformed token fails validation and is never written to .env."""
        settings = self.make_settings()
        with self.assertRaises(ConfigError):
            self.run_setup(
                settings,
                FakeTelegramClient(),
                secret_input=lambda prompt: "not-a-token",
                client_factory=TelegramClient,
            )
        self.assertFalse(settings.env_file.exists())


class InitProjectTests(AppTestCase):
    """init_project."""

    def test_creates_both_files_with_private_env(self) -> None:
        """tasks.toml and .env are created and .env is mode 600."""
        created = init_project(self.home)
        self.assertEqual(created, ["tasks.toml", ".env"])
        self.assertTrue((self.home / "tasks.toml").read_text().strip())
        self.assertEqual(stat.S_IMODE((self.home / ".env").stat().st_mode), 0o600)

    def test_env_file_is_never_world_readable_while_created(self) -> None:
        """.env is created with mode 0600 from the start (umask 0 would expose a later chmod gap)."""
        old = os.umask(0)
        self.addCleanup(os.umask, old)
        init_project(self.home)
        self.assertEqual(stat.S_IMODE((self.home / ".env").stat().st_mode), 0o600)

    def test_never_overwrites(self) -> None:
        """Existing files keep their content and are not reported as created."""
        (self.home / "tasks.toml").write_text("mine")
        created = init_project(self.home)
        self.assertEqual(created, [".env"])
        self.assertEqual((self.home / "tasks.toml").read_text(), "mine")
        self.assertEqual(init_project(self.home), [])

    def test_created_tasks_file_validates(self) -> None:
        """The copied example loads with the real parser."""
        init_project(self.home)
        settings = self.make_settings()
        self.assertEqual(
            list(build_app(settings, runner=object()).engine.config.tasks),  # type: ignore[arg-type]
            ["ai-news", "day-plan", "weekly-review"],
        )

    def test_creates_missing_home_folder(self) -> None:
        """A home folder that does not exist yet is created."""
        target = self.home / "new" / "home"
        self.assertEqual(init_project(target), ["tasks.toml", ".env"])


if __name__ == "__main__":
    unittest.main()
