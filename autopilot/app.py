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
        settings.log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(
            logging.handlers.RotatingFileHandler(
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
        config_loader=lambda: load_task_config(settings.tasks_file),
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
    except Exception:  # noqa: BLE001 - a failed greeting must not stop the service
        logger.exception("Could not send the startup notice")


def telegram_setup(
    settings: Settings,
    *,
    secret_input: Callable[[str], str] = getpass.getpass,
    out: TextIO = sys.stdout,
    client_factory: Callable[[str], TelegramClient] = TelegramClient,
    wait_seconds: int = 120,
    monotonic: Callable[[], float] = time.monotonic,
) -> int:
    """Connect the bot: validate the token, learn the chat id, save both, send a test message.

    Args:
        settings: Provides an existing token, the .env path and the database path.
        secret_input: Prompt for the token without echo.
        out: Where progress lines are written.
        client_factory: Builds the Telegram client from the token.
        wait_seconds: How long to wait for the user's first message.
        monotonic: Clock used for the wait deadline.

    Returns:
        The chat id that was saved.

    Raises:
        AutopilotError: Bad token, no message in time, or another poller holds the bot.
        TelegramError: Telegram rejected a call.
    """
    token = settings.telegram_token or secret_input(TOKEN_PROMPT).strip()
    client = client_factory(token)
    username = client.get_me().get("username", "")
    # Saved only after getMe accepted it, so a typo never lands in .env.
    set_env_value(settings.env_file, "TELEGRAM_BOT_TOKEN", token)
    out.write(
        f"Open https://t.me/{username} and send any message (waiting up to {wait_seconds} s)...\n"
    )
    out.flush()

    update_id, chat_id = _wait_for_first_message(client, wait_seconds, monotonic)
    set_env_value(settings.env_file, "TELEGRAM_CHAT_ID", str(chat_id))
    # Skip this message in the service so the bot does not answer it a second time.
    Store(settings.db_path).set_value(OFFSET_KEY, str(update_id + 1))
    client.send_message(chat_id, SETUP_DONE_TEXT, parse_mode=None)
    out.write(f"Connected. Saved TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID to {settings.env_file}.\n")
    return chat_id


def _wait_for_first_message(
    client: TelegramClient, wait_seconds: int, monotonic: Callable[[], float]
) -> tuple[int, int]:
    """Poll until a message arrives.

    Args:
        client: Telegram client.
        wait_seconds: Total time allowed.
        monotonic: Clock for the deadline.

    Returns:
        ``(update_id, chat_id)`` of the first message.

    Raises:
        AutopilotError: On timeout or when another process polls the same bot (HTTP 409).
    """
    deadline = monotonic() + wait_seconds
    offset: int | None = None
    while True:
        remaining = deadline - monotonic()
        if remaining <= 0:
            raise AutopilotError(
                f"No message arrived within {wait_seconds} s. Run `autopilot telegram-setup` "
                "again and send your bot a message."
            )
        try:
            updates = client.get_updates(
                offset=offset, timeout=min(SETUP_POLL_SECONDS, max(1, int(remaining)))
            )
        except TelegramError as err:
            if err.status == 409:
                raise AutopilotError(SETUP_CONFLICT_MESSAGE) from None
            raise
        for update in updates:
            offset = update["update_id"] + 1
            chat = (update.get("message") or {}).get("chat") or {}
            if isinstance(chat.get("id"), int):
                return update["update_id"], chat["id"]


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
        # "x" mode fails instead of overwriting if the file appears meanwhile.
        try:
            with source.open("rb") as reader, open(target, "xb") as writer:
                shutil.copyfileobj(reader, writer)
        except FileExistsError:
            continue
        if private:
            os.chmod(target, ENV_FILE_MODE)
        created.append(target_name)
    return created
