"""Application wiring: logging, the service loop, Telegram setup and project init.

Responsibility: build the object graph (store, runner, engine, notifier, bot)
from Settings, run it as a service, and provide the two one-time setup flows
(`telegram-setup`, `init`).

Where it fits: ``cli.py`` calls everything here; the integration tests build
an ``App`` directly. It uses settings, store, runner, engine, bot,
telegram_api and formatting.

Invariants:
    * The bot token is redacted from every log line (``RedactFilter``).
    * ``build_app`` fails loudly (ClaudeNotFoundError, ConfigError) so a
      broken setup stops `autopilot start` instead of failing at the first run.
    * Signal handlers are installed only on the main thread and restored on exit.
    * `init_project` never overwrites an existing file.
"""

from __future__ import annotations

import getpass
import logging
import logging.handlers
import os
import shutil
import signal
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TextIO

from autopilot.bot import OFFSET_KEY, TelegramBot, TelegramNotifier
from autopilot.engine import Engine, LogNotifier
from autopilot.errors import AutopilotError, TelegramError
from autopilot.formatting import format_time
from autopilot.models import Notifier, Runner
from autopilot.runner import ClaudeRunner, find_claude_binary
from autopilot.settings import (
    DEFAULT_TASKS_FILE_NAME,
    ENV_FILE_MODE,
    ENV_FILE_NAME,
    Settings,
    project_root,
    set_env_value,
)
from autopilot.store import Store
from autopilot.tasks import load_task_config
from autopilot.telegram_api import TelegramClient

logger = logging.getLogger(__name__)

LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
LOG_MAX_BYTES = 1_000_000
LOG_BACKUP_COUNT = 3
REDACTED = "***"
# Marks handlers installed by configure_logging so a second call replaces them.
HANDLER_MARKER = "_autopilot_handler"

SETUP_POLL_SECONDS = 25
DEFAULT_SETUP_WAIT_SECONDS = 120
# getUpdates returns at most this many updates per call.
UPDATES_BATCH_LIMIT = 100
HTTP_CONFLICT = 409
PRIVATE_CHAT_TYPE = "private"
CONFIRM_PROMPT = "Use this chat? [y/N] "
CONFIRM_YES = ("y", "yes")
DATA_DIR_MODE = 0o700
LOG_FILE_MODE = 0o600
PUBLIC_FILE_MODE = 0o644
TOKEN_PROMPT = "Paste the bot token from @BotFather: "
SETUP_CONFLICT_MESSAGE = (
    "Another autopilot is polling this bot. Stop it first: systemctl --user stop autopilot"
)
SETUP_DONE_TEXT = "✅ Autopilot is connected. Send /help to see what I can do."

# (source file in the project folder, file created in the home folder, private mode?)
INIT_FILES = (
    ("tasks.example.toml", DEFAULT_TASKS_FILE_NAME, False),
    (".env.example", ENV_FILE_NAME, True),
)


class RedactFilter(logging.Filter):
    """Replaces given secrets in every log record with ***.

    Example:
        handler.addFilter(RedactFilter([settings.telegram_token]))
    """

    def __init__(self, secrets: list[str | None]) -> None:
        """Remember the secrets to hide.

        Args:
            secrets: Strings to redact; None and empty entries are ignored.
        """
        super().__init__()
        self._secrets = [secret for secret in secrets if secret]

    def filter(self, record: logging.LogRecord) -> bool:
        """Rewrite the record's message and traceback text without the secrets.

        Args:
            record: The record about to be emitted.

        Returns:
            Always True: the record is kept, only edited.
        """
        if not self._secrets:
            return True
        # Render the message once so secrets inside %-arguments are caught too.
        message = record.getMessage()
        record.msg = self._redact(message)
        record.args = None
        if record.exc_info and not record.exc_text:
            record.exc_text = logging.Formatter().formatException(record.exc_info)
        if record.exc_text:
            record.exc_text = self._redact(record.exc_text)
        return True

    def _redact(self, text: str) -> str:
        """Replace every secret in ``text``."""
        for secret in self._secrets:
            text = text.replace(secret, REDACTED)
        return text


def configure_logging(settings: Settings, *, debug: bool = False, to_file: bool = True) -> None:
    """Send logs to stderr and, optionally, a rotating file, with the token redacted.

    Calling it again replaces the handlers from the earlier call.

    Args:
        settings: Provides the log level, log file path and the token to redact.
        debug: Force DEBUG level regardless of the configured level.
        to_file: Also write to ``settings.log_file`` (1 MB, 3 backups); the
            data folder is created when needed.
    """
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]
    if to_file:
        settings.log_file.parent.mkdir(parents=True, exist_ok=True, mode=DATA_DIR_MODE)
        handlers.append(
            _PrivateRotatingFileHandler(
                settings.log_file,
                maxBytes=LOG_MAX_BYTES,
                backupCount=LOG_BACKUP_COUNT,
                encoding="utf-8",
            )
        )
    redactor = RedactFilter([settings.telegram_token])
    formatter = logging.Formatter(LOG_FORMAT)
    root = logging.getLogger()
    # Drop handlers from a previous call so lines are not duplicated.
    for old in [h for h in root.handlers if getattr(h, HANDLER_MARKER, False)]:
        root.removeHandler(old)
        old.close()
    for handler in handlers:
        setattr(handler, HANDLER_MARKER, True)
        handler.setFormatter(formatter)
        # Filters on handlers (not on the root logger) see records from every module.
        handler.addFilter(redactor)
        root.addHandler(handler)
    root.setLevel(logging.DEBUG if debug else settings.log_level)


class _PrivateRotatingFileHandler(logging.handlers.RotatingFileHandler):
    """Rotating log file that only its owner can read (logs may hold personal data)."""

    def __init__(self, filename: Path, **kwargs: object) -> None:
        """Open the log file and restrict it to mode 0o600.

        Args:
            filename: Log file path.
            **kwargs: Passed to ``RotatingFileHandler``.
        """
        super().__init__(filename, **kwargs)  # type: ignore[arg-type]
        os.chmod(filename, LOG_FILE_MODE)

    def doRollover(self) -> None:
        """Rotate the files, then restrict the fresh log file again."""
        super().doRollover()
        os.chmod(self.baseFilename, LOG_FILE_MODE)


@dataclass
class App:
    """The wired object graph.

    Attributes:
        settings: Resolved settings.
        store: SQLite persistence.
        engine: Queue, workers and scheduler.
        notifier: Telegram notifier, or a log-only one.
        client: Telegram client; None when Telegram is not in use.
        bot: Telegram bot; None when Telegram is not in use.
    """

    settings: Settings
    store: Store
    engine: Engine
    notifier: Notifier
    client: TelegramClient | None
    bot: TelegramBot | None


def build_app(
    settings: Settings,
    *,
    notify: bool = True,
    runner: Runner | None = None,
    client: TelegramClient | None = None,
    clock: Callable[[], datetime] | None = None,
) -> App:
    """Create the store, runner, engine, notifier and (when configured) bot.

    Args:
        settings: Resolved settings.
        notify: False sends nothing to Telegram (used by `run --no-notify`); the
            client and bot are then not created.
        runner: Replaces the Claude runner (tests). When None the Claude binary
            is located now, so a missing install fails at startup.
        client: Replaces the Telegram client (tests); used only when Telegram is in use.
        clock: Replaces the clock; defaults to now in ``settings.timezone``.

    Returns:
        The wired App with tasks.toml already loaded.

    Raises:
        ClaudeNotFoundError: No Claude binary and no runner was given.
        ConfigError: tasks.toml is missing or invalid.
    """
    now = clock or (lambda: datetime.now(settings.timezone))
    if runner is None:
        find_claude_binary(settings.claude_bin)
        runner = ClaudeRunner(lambda: find_claude_binary(settings.claude_bin), clock=now)
    store = Store(settings.db_path)

    telegram_client: TelegramClient | None = None
    notifier: Notifier = LogNotifier()
    token, chat_id = settings.telegram_token, settings.telegram_chat_id
    telegram_notifier: TelegramNotifier | None = None
    # Both values are checked here (not via telegram_enabled) so their types narrow.
    if notify and token is not None and chat_id is not None:
        telegram_client = client or TelegramClient(token)
        telegram_notifier = TelegramNotifier(telegram_client, chat_id, local_tz=settings.timezone)
        notifier = telegram_notifier

    engine = Engine(
        store=store,
        runner=runner,
        notifier=notifier,
        # The .env file holds the bot token: Claude must never be allowed to read it.
        config_loader=lambda: load_task_config(
            settings.tasks_file, protected_paths=(settings.env_file,)
        ),
        clock=now,
        tasks_file=settings.tasks_file,
        max_parallel=settings.max_parallel,
    )
    engine.load_config()

    bot: TelegramBot | None = None
    if telegram_client is not None and telegram_notifier is not None and chat_id is not None:
        bot = TelegramBot(telegram_client, engine, store, telegram_notifier, chat_id=chat_id, clock=now)
    return App(settings, store, engine, notifier, telegram_client, bot)


def run_service(app: App, stop: threading.Event) -> None:
    """Run the service until ``stop`` is set (or SIGTERM/SIGINT arrives).

    Args:
        app: The wired application.
        stop: Event that ends the loop; set by the signal handlers on the main thread.
    """
    on_main_thread = threading.current_thread() is threading.main_thread()
    previous: dict[int, object] = {}
    if on_main_thread:
        for signum in (signal.SIGTERM, signal.SIGINT):
            previous[signum] = signal.signal(signum, lambda _signum, _frame: stop.set())
    try:
        # Before the bot thread starts, or fresh runs would be marked interrupted too.
        app.engine.recover_interrupted()
        _announce_start(app)
        if app.bot is not None:
            threading.Thread(
                target=app.bot.poll_forever, args=(stop,), name="telegram-bot", daemon=True
            ).start()
        app.engine.run_forever(stop)
    finally:
        # Put the original handlers back so embedding code is unaffected.
        for signum, handler in previous.items():
            signal.signal(signum, handler)  # type: ignore[arg-type]
        logger.info("Autopilot stopped")


def _announce_start(app: App) -> None:
    """Send the "running" notice; any failure is logged, never fatal."""
    try:
        config = app.engine.config
        now = datetime.now(app.settings.timezone)
        upcoming = [
            (moment, name) for name, moment in app.engine.next_fire_times().items() if moment
        ]
        text = f"🟢 Autopilot is running · {len(config.tasks)} tasks"
        if upcoming:
            moment, name = min(upcoming, key=lambda pair: pair[0].timestamp())
            text += f" · next: {name} at {format_time(moment, now)}"
        app.notifier.notify_text(text)
    except Exception:  # A failed greeting must not stop the service, so only log it.
        logger.exception("Could not send the startup notice")


def telegram_setup(
    settings: Settings,
    *,
    secret_input: Callable[[str], str] = getpass.getpass,
    out: TextIO | None = None,
    client_factory: Callable[[str], TelegramClient] = TelegramClient,
    wait_seconds: int = DEFAULT_SETUP_WAIT_SECONDS,
    monotonic: Callable[[], float] = time.monotonic,
    confirm: Callable[[str], str] = input,
) -> int:
    """Connect the bot: validate the token, learn the chat id, save both, send a test message.

    Messages that were pending before the setup started are ignored, only
    private chats are offered, and the user must confirm the chat shown.

    Args:
        settings: Provides an existing token, the .env path and the database path.
        secret_input: Prompt for the token without echo.
        out: Where progress lines are written; None means ``sys.stdout`` at call time.
        client_factory: Builds the Telegram client from the token.
        wait_seconds: How long to wait for a confirmed chat.
        monotonic: Clock used for the wait deadline.
        confirm: Asks the yes/no question and returns the answer.

    Returns:
        The chat id that was saved.

    Raises:
        AutopilotError: Bad token, no confirmed chat in time, or another poller holds the bot.
        TelegramError: Telegram rejected a call.
    """
    out = sys.stdout if out is None else out
    token = settings.telegram_token or secret_input(TOKEN_PROMPT).strip()
    client = client_factory(token)
    username = client.get_me().get("username", "")
    # Saved only after getMe accepted it, so a typo never lands in .env.
    set_env_value(settings.env_file, "TELEGRAM_BOT_TOKEN", token)

    # Old messages (possibly from strangers) must not be mistaken for the user's first one.
    offset = _drain_pending_updates(client)
    out.write(
        f"Open https://t.me/{username} and send any message (waiting up to {wait_seconds} s)...\n"
    )
    out.flush()

    update_id, chat_id = _wait_for_confirmed_chat(client, offset, wait_seconds, monotonic, confirm, out)
    set_env_value(settings.env_file, "TELEGRAM_CHAT_ID", str(chat_id))
    # Skip this message in the service so the bot does not answer it a second time.
    Store(settings.db_path).set_value(OFFSET_KEY, str(update_id + 1))
    client.send_message(chat_id, SETUP_DONE_TEXT, parse_mode=None)
    out.write(f"Connected. Saved TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID to {settings.env_file}.\n")
    return chat_id


def _get_updates(client: TelegramClient, offset: int | None, timeout: int) -> list[dict]:
    """Call getUpdates, turning a polling conflict (HTTP 409) into a friendly error."""
    try:
        return client.get_updates(offset=offset, timeout=timeout)
    except TelegramError as err:
        if err.status == HTTP_CONFLICT:
            raise AutopilotError(SETUP_CONFLICT_MESSAGE) from None
        raise


def _drain_pending_updates(client: TelegramClient) -> int | None:
    """Read (without waiting) and skip every update already waiting at Telegram.

    Args:
        client: Telegram client.

    Returns:
        The offset just past the last pending update, or None when none were pending.
    """
    offset: int | None = None
    while True:
        batch = _get_updates(client, offset, 0)
        if batch:
            offset = batch[-1]["update_id"] + 1
        if len(batch) < UPDATES_BATCH_LIMIT:
            return offset


def _describe_sender(message: dict) -> str:
    """Return "<first name> (@username), chat id <id>" for a private message (parts may be absent)."""
    sender = message.get("from") or message.get("chat") or {}
    parts = [str(sender.get("first_name") or "").strip()]
    if sender.get("username"):
        parts.append(f"(@{sender['username']})")
    label = " ".join(part for part in parts if part)
    return f"{label}, chat id {message['chat']['id']}" if label else f"chat id {message['chat']['id']}"


def _wait_for_confirmed_chat(
    client: TelegramClient,
    offset: int | None,
    wait_seconds: int,
    monotonic: Callable[[], float],
    confirm: Callable[[str], str],
    out: TextIO,
) -> tuple[int, int]:
    """Poll new messages until the user confirms a private chat.

    Args:
        client: Telegram client.
        offset: First update id to read (past the drained ones).
        wait_seconds: Total time allowed.
        monotonic: Clock for the deadline.
        confirm: Asks the yes/no question.
        out: Where notes are written.

    Returns:
        ``(update_id, chat_id)`` of the confirmed message.

    Raises:
        AutopilotError: On timeout or when another process polls the same bot (HTTP 409).
    """
    deadline = monotonic() + wait_seconds
    while True:
        remaining = deadline - monotonic()
        if remaining <= 0:
            raise AutopilotError(
                f"No confirmed chat within {wait_seconds} s. Run `autopilot telegram-setup` "
                "again, send your bot a message and answer y."
            )
        timeout = min(SETUP_POLL_SECONDS, max(1, int(remaining)))
        for update in _get_updates(client, offset, timeout):
            offset = update["update_id"] + 1
            message = update.get("message") or {}
            chat = message.get("chat") or {}
            if not isinstance(chat.get("id"), int):
                continue
            if chat.get("type") != PRIVATE_CHAT_TYPE:
                out.write("Ignored a message from a group or channel: send it in a private chat with the bot.\n")
                continue
            out.write(f"Message from {_describe_sender(message)}\n")
            if confirm(CONFIRM_PROMPT).strip().lower() in CONFIRM_YES:
                return update["update_id"], chat["id"]
            out.write("Not saved. Waiting for another message...\n")


def init_project(home: Path) -> list[str]:
    """Create tasks.toml and .env from the shipped examples when they are missing.

    Args:
        home: Project folder to create the files in; created if absent.

    Returns:
        Names of the files created (empty when everything already existed).

    Raises:
        AutopilotError: A shipped example file cannot be found.
    """
    home.mkdir(parents=True, exist_ok=True)
    created: list[str] = []
    for source_name, target_name, private in INIT_FILES:
        target = home / target_name
        if target.exists():
            continue
        source = project_root() / source_name
        if not source.is_file():
            raise AutopilotError(f"Cannot find {source}. Reinstall autopilot or create {target_name} by hand.")
        # O_EXCL fails instead of overwriting if the file appears meanwhile; the mode is
        # set at creation so a private file is never readable by others, even briefly.
        mode = ENV_FILE_MODE if private else PUBLIC_FILE_MODE
        try:
            descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
        except FileExistsError:
            continue
        with source.open("rb") as reader, os.fdopen(descriptor, "wb") as writer:
            shutil.copyfileobj(reader, writer)
        created.append(target_name)
    return created
