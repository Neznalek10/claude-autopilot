"""Telegram side of Claude Autopilot: result delivery and the remote-control bot.

Two classes live here:

* ``TelegramNotifier`` sends finished runs and plain notices to the one
  configured chat. The engine calls it through the ``Notifier`` protocol.
* ``TelegramBot`` long-polls Telegram, ignores every chat except the
  configured one, and turns commands and plain text into engine calls.

Fits in the architecture between ``engine.py`` (which it drives through the
``EngineAPI`` protocol defined here, so this file never imports the engine),
``store.py`` (pause flags, finished runs, the persisted poll offset),
``formatting.py`` (message texts) and ``telegram_api.py`` (the HTTP client).

Invariants:
    * Only messages from ``chat_id`` are ever answered; others are logged.
    * The poll offset is stored after every update, so a crash never replays
      an update that was already handled.
    * ``handle_update`` may raise; ``poll_forever`` never lets that kill the loop.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from datetime import datetime, tzinfo
from typing import Any, Protocol

from autopilot.errors import TaskNotFoundError, TelegramError
from autopilot.formatting import (
    escape_html,
    format_run_messages,
    format_status,
    format_task_list,
    help_text,
    html_to_plain,
    result_document,
)
from autopilot.models import (
    ASK_TASK_NAME,
    BOT_COMMANDS,
    TRIGGER_CHAT,
    TRIGGER_TELEGRAM,
    QueueSnapshot,
    RunRecord,
    TaskConfig,
)
from autopilot.store import Store
from autopilot.telegram_api import TelegramClient, is_parse_error

logger = logging.getLogger(__name__)

OFFSET_KEY = "telegram_offset"
POLL_BACKOFF_START = 5
POLL_BACKOFF_MAX = 60

COMMAND_PREFIX = "/"
BOT_NAME_SEPARATOR = "@"
TYPING_ACTION = "typing"

REPLY_NEW_CONVERSATION = "🆕 New conversation. What's on your mind?"
REPLY_UNKNOWN_COMMAND = "Unknown command. Send /help."
REPLY_NO_FINISHED_RUNS = "No finished runs yet."
REPLY_ASK_USAGE = "Usage: /ask &lt;text&gt; (or just type your message)."


class EngineAPI(Protocol):
    """The slice of ``Engine`` the bot uses; tests supply a small fake."""

    @property
    def config(self) -> TaskConfig:
        """The currently loaded tasks.

        Returns:
            The TaskConfig parsed from tasks.toml.
        """
        ...

    def enqueue_task(
        self,
        name: str,
        trigger: str,
        *,
        scheduled_for: datetime | None = None,
        retry_of: int | None = None,
        dispatch: bool = True,
    ) -> int | None:
        """Queue a run of a task from tasks.toml.

        Args:
            name: Task name.
            trigger: What started the run (bot commands use TRIGGER_TELEGRAM).
            scheduled_for: Schedule slot the run belongs to, if any.
            retry_of: Id of the run this one retries, if any.
            dispatch: Put the run on the worker queue right away.

        Returns:
            The new run id, or None when the task already has an active run.

        Raises:
            TaskNotFoundError: If no task has that name.
        """
        ...

    def enqueue_prompt(
        self,
        prompt: str,
        trigger: str = TRIGGER_CHAT,
        *,
        retry_of: int | None = None,
        dispatch: bool = True,
    ) -> int:
        """Queue a chat turn or one-off prompt.

        Args:
            prompt: The text to send to Claude.
            trigger: TRIGGER_CHAT continues the conversation; TRIGGER_MANUAL is one-off.
            retry_of: Id of the run this one retries, if any.
            dispatch: Put the run on the worker queue right away.

        Returns:
            The new run id.

        Raises:
            ValueError: If the prompt is empty.
        """
        ...

    def reset_chat(self) -> None:
        """Forget the current conversation so the next turn starts fresh."""
        ...

    def next_fire_times(self) -> dict[str, datetime | None]:
        """Next scheduled run per task name.

        Returns:
            Task name mapped to its next fire time, or None for manual-only,
            disabled or paused tasks.
        """
        ...

    def snapshot(self) -> QueueSnapshot:
        """Running and queued runs plus the usage-limit gate.

        Returns:
            A QueueSnapshot taken from the store at call time.
        """
        ...


class TelegramNotifier:
    """Delivers run results and notices to the configured Telegram chat.

    Example:
        notifier = TelegramNotifier(client, chat_id, local_tz=settings.timezone)
        notifier.notify_text("Autopilot is running")
    """

    def __init__(
        self, client: TelegramClient, chat_id: int, *, local_tz: tzinfo | None = None
    ) -> None:
        """Store the client and destination.

        Args:
            client: Telegram Bot API client.
            chat_id: The only chat that receives messages (may be negative).
            local_tz: Zone used for times in result documents; None keeps the
                record's own offset.
        """
        self.client = client
        self.chat_id = chat_id
        self.local_tz = local_tz

    def notify_run(self, record: RunRecord) -> None:
        """Send a finished run: its messages in order, then a file if needed.

        Args:
            record: The finished run.

        Raises:
            TelegramError: If Telegram rejects a send; the engine logs it.
        """
        messages, needs_document = format_run_messages(record)
        for message in messages:
            self.send_html(message)
        if needs_document:
            filename, content = result_document(record, local_tz=self.local_tz)
            self.client.send_document(self.chat_id, filename, content)

    def notify_text(self, text: str) -> None:
        """Send a plain-text notice (no formatting is interpreted).

        Args:
            text: The notice text.

        Raises:
            TelegramError: If Telegram rejects the send.
        """
        self.client.send_message(self.chat_id, text, parse_mode=None)

    def send_html(self, html: str) -> dict[str, Any]:
        """Send HTML, falling back to plain text if Telegram can't parse it.

        Args:
            html: Message text using Telegram's HTML subset.

        Returns:
            The API result of the message that was delivered.

        Raises:
            TelegramError: For any failure other than a parse error, or if
                the plain-text resend fails too.
        """
        try:
            return self.client.send_message(self.chat_id, html)
        except TelegramError as err:
            if not is_parse_error(err):
                raise
            # Telegram refused the markup; the text is still worth delivering.
            logger.warning("Telegram could not parse an HTML message, resending as plain text")
            return self.client.send_message(self.chat_id, html_to_plain(html), parse_mode=None)


class TelegramBot:
    """Long-polls Telegram and maps commands and chat text to engine calls."""

    def __init__(
        self,
        client: TelegramClient,
        engine: EngineAPI,
        store: Store,
        notifier: TelegramNotifier,
        *,
        chat_id: int,
        clock: Callable[[], datetime],
    ) -> None:
        """Wire the bot to its collaborators.

        Args:
            client: Telegram Bot API client.
            engine: Engine (or a fake) that queues runs.
            store: Persistence for pause flags, runs and the poll offset.
            notifier: Sends all replies to ``chat_id``.
            chat_id: The only chat the bot answers.
            clock: Returns the aware current time.
        """
        self.client = client
        self.engine = engine
        self.store = store
        self.notifier = notifier
        self.chat_id = chat_id
        self.clock = clock
        self._handlers: dict[str, Callable[[str], None]] = {
            "start": self._cmd_help,
            "help": self._cmd_help,
            "tasks": self._cmd_tasks,
            "run": self._cmd_run,
            "ask": self._cmd_ask,
            "new": self._cmd_new,
            "last": self._cmd_last,
            "status": self._cmd_status,
            "pause": self._cmd_pause,
            "resume": self._cmd_resume,
        }

    # --- Update handling ---------------------------------------------------------------

    def handle_update(self, update: dict[str, Any]) -> None:
        """Process one Telegram update.

        Updates without text, and messages from any other chat, are ignored
        (the latter are logged with their chat id and get no reply).

        Args:
            update: One item of the ``getUpdates`` result.

        Raises:
            TelegramError: If sending a reply fails.
        """
        message = update.get("message")
        if not isinstance(message, dict):
            return
        text = message.get("text")
        chat_id = (message.get("chat") or {}).get("id")
        sender_id = (message.get("from") or {}).get("id")
        if not isinstance(text, str) or not text.strip():
            return
        # Only the owner's private chat may command the runner: in a private chat
        # chat.id equals the user id, so a group (or another sender inside the
        # configured chat id) fails one of the two checks.
        if chat_id != self.chat_id or sender_id != self.chat_id:
            logger.warning(
                "Ignoring a message from unauthorized chat %s (sender %s)", chat_id, sender_id
            )
            return
        text = text.strip()
        if not text.startswith(COMMAND_PREFIX):
            self._chat(text)
            return
        parts = text.split(None, 1)
        command = parts[0]
        argument = parts[1] if len(parts) > 1 else ""
        # "/Run@MyBot" -> "run": group chats append the bot name to commands.
        name = command[1:].split(BOT_NAME_SEPARATOR, 1)[0].lower()
        handler = self._handlers.get(name)
        if handler is None:
            self._reply(REPLY_UNKNOWN_COMMAND)
            return
        handler(argument.strip())

    def poll_forever(self, stop: threading.Event) -> None:
        """Poll Telegram until ``stop`` is set.

        Registers the command menu once, then fetches updates starting at the
        stored offset, handles each one and stores ``update_id + 1`` after it.
        Telegram errors back off from 5 to 60 seconds (``retry_after`` wins).

        Args:
            stop: Set by the service to end the loop; also used for the
                backoff wait so shutdown is immediate.
        """
        try:
            self.client.set_my_commands(BOT_COMMANDS)
        except TelegramError as err:
            logger.warning("Could not register the bot commands: %s", err)
        backoff = POLL_BACKOFF_START
        while not stop.is_set():
            try:
                updates = self.client.get_updates(offset=self._stored_offset())
            except TelegramError as err:
                delay = err.retry_after if err.retry_after else backoff
                logger.warning("Telegram polling failed, retrying in %s s: %s", delay, err)
                backoff = min(backoff * 2, POLL_BACKOFF_MAX)
                stop.wait(delay)
                continue
            backoff = POLL_BACKOFF_START
            for update in updates:
                self._process(update)

    def _process(self, update: dict[str, Any]) -> None:
        """Handle one update without letting any error escape, then store its offset."""
        try:
            self.handle_update(update)
        # Broad on purpose: one bad update must never stop the polling loop.
        except Exception:
            logger.exception("Failed to handle Telegram update %s", update.get("update_id"))
        # Stored even after a failure: replaying a poison update would loop forever.
        update_id = update.get("update_id")
        if isinstance(update_id, int):
            self.store.set_value(OFFSET_KEY, str(update_id + 1))

    def _stored_offset(self) -> int | None:
        """Return the persisted offset, or None when absent or unreadable."""
        raw = self.store.get_value(OFFSET_KEY)
        if raw is None:
            return None
        try:
            return int(raw)
        except ValueError:
            logger.warning("Ignoring unreadable stored Telegram offset %r", raw)
            return None

    # --- Replies -----------------------------------------------------------------------

    def _reply(self, html: str) -> None:
        """Send an HTML reply to the configured chat."""
        self.notifier.send_html(html)

    def _task_names(self) -> list[str]:
        """Return the names of the currently defined tasks, in file order."""
        return list(self.engine.config.tasks)

    def _reply_if_unknown_task(self, name: str) -> bool:
        """Reply with the TaskNotFoundError message when ``name`` isn't a task.

        Args:
            name: Task name typed by the user.

        Returns:
            True when the task is unknown (the reply was sent and the caller
            should stop), False when it exists.
        """
        if name in self.engine.config.tasks:
            return False
        error = TaskNotFoundError(name, self._task_names())
        self._reply(escape_html(str(error)))
        return True

    def _ahead_of(self, run_id: int) -> int:
        """Count active runs created before ``run_id`` (the ones it waits for)."""
        return sum(1 for record in self.store.active_runs() if record.id < run_id)

    # --- Chat --------------------------------------------------------------------------

    def _chat(self, text: str) -> None:
        """Queue a chat turn and tell the user whether they must wait."""
        run_id = self.engine.enqueue_prompt(text, TRIGGER_CHAT)
        ahead = self._ahead_of(run_id)
        if ahead:
            self._reply(
                f"🕐 Got it. {ahead} run(s) ahead of you, I'll answer when they finish."
            )
            return
        try:
            self.client.send_chat_action(self.chat_id, TYPING_ACTION)
        except TelegramError as err:
            # The indicator is cosmetic; the answer will still arrive.
            logger.debug("Could not send the typing indicator: %s", err)

    # --- Commands ----------------------------------------------------------------------

    def _cmd_help(self, argument: str) -> None:
        """/help and /start: show the usage text."""
        self._reply(help_text())

    def _cmd_tasks(self, argument: str) -> None:
        """/tasks: list tasks with next and last runs."""
        now = self.clock()
        tasks = list(self.engine.config.tasks.values())
        last_runs: dict[str, RunRecord | None] = {}
        for task in tasks:
            finished = self.store.list_runs(task_name=task.name, limit=1, finished_only=True)
            last_runs[task.name] = finished[0] if finished else None
        paused = {task.name: self.store.is_paused(task.name) for task in tasks}
        self._reply(
            format_task_list(tasks, self.engine.next_fire_times(), last_runs, paused, now)
        )

    def _cmd_run(self, argument: str) -> None:
        """/run <task>: queue a task now."""
        if not argument:
            names = ", ".join(self._task_names()) or "none defined"
            self._reply(f"Usage: /run &lt;task&gt;\nTasks: {escape_html(names)}")
            return
        if self._reply_if_unknown_task(argument):
            return
        run_id = self.engine.enqueue_task(argument, TRIGGER_TELEGRAM)
        if run_id is None:
            self._reply(f"⏳ {escape_html(argument)} is already queued or running.")
            return
        self._reply(
            f"🕐 Queued <b>{escape_html(argument)}</b> "
            f"(run #{run_id}, {self._ahead_of(run_id)} ahead)."
        )

    def _cmd_ask(self, argument: str) -> None:
        """/ask <text>: same as typing the text."""
        if not argument:
            self._reply(REPLY_ASK_USAGE)
            return
        self._chat(argument)

    def _cmd_new(self, argument: str) -> None:
        """/new [text]: start a fresh conversation, optionally with a first message."""
        self.engine.reset_chat()
        if argument:
            self._chat(argument)
            return
        self._reply(REPLY_NEW_CONVERSATION)

    def _cmd_last(self, argument: str) -> None:
        """/last [task]: resend the newest finished run."""
        task_name = argument or None
        if task_name is not None and task_name != ASK_TASK_NAME:
            if self._reply_if_unknown_task(task_name):
                return
        finished = self.store.list_runs(task_name=task_name, limit=1, finished_only=True)
        if not finished:
            self._reply(REPLY_NO_FINISHED_RUNS)
            return
        self.notifier.notify_run(finished[0])

    def _cmd_status(self, argument: str) -> None:
        """/status: show running and queued runs and upcoming schedules."""
        self._reply(
            format_status(self.engine.snapshot(), self.engine.next_fire_times(), self.clock())
        )

    def _cmd_pause(self, argument: str) -> None:
        """/pause <task>: stop a task's schedule."""
        self._set_paused(argument, True)

    def _cmd_resume(self, argument: str) -> None:
        """/resume <task>: restart a task's schedule."""
        self._set_paused(argument, False)

    def _set_paused(self, name: str, paused: bool) -> None:
        """Validate the task, store its paused flag and confirm to the user."""
        verb = "pause" if paused else "resume"
        if not name:
            names = ", ".join(self._task_names()) or "none defined"
            self._reply(f"Usage: /{verb} &lt;task&gt;\nTasks: {escape_html(names)}")
            return
        if self._reply_if_unknown_task(name):
            return
        self.store.set_paused(name, paused)
        icon, word = ("⏸", "Paused") if paused else ("▶️", "Resumed")
        self._reply(f"{icon} {word} <b>{escape_html(name)}</b>.")
