"""Shared factories and fakes for the unit tests.

Every test module in this folder imports this file as ``helpers``. That works
because ``unittest discover -s tests`` puts this folder on ``sys.path``, so
there is deliberately no ``tests/__init__.py``.

Contents:
    * ``FakeClock``: a clock the test moves by hand; inject it wherever code asks for "now".
    * ``make_*`` factories: build valid dataclass instances with sane defaults.
    * ``FakeTelegramClient``: records Bot API calls in memory, no network.
    * ``FAKE_CLAUDE``: path to the executable stand-in for the Claude CLI.

Invariants:
    * Every datetime built here is timezone-aware and uses ``TZ``.
    * Factories pass schedules through unchanged, so tests can use real Schedule objects.
"""

from __future__ import annotations

import tempfile
from collections.abc import Sequence
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any
from zoneinfo import ZoneInfo

from autopilot.errors import TelegramError
from autopilot.models import (
    STATUS_SUCCESS,
    TRIGGER_SCHEDULE,
    RunOptions,
    RunRecord,
    Task,
    TaskConfig,
)

if TYPE_CHECKING:
    from autopilot.schedule import Schedule

# The time zone used by every fixed test moment. A zone with DST, so DST logic gets exercised.
TZ = ZoneInfo("Europe/Prague")

# Executable stand-in for the Claude CLI; its behaviour is documented in fake_claude.py.
FAKE_CLAUDE = Path(__file__).with_name("fake_claude.py")

# Mirrors autopilot.telegram_api.LONG_POLL_SECONDS. Copied rather than imported so this
# file loads before Wave 1 creates telegram_api.py; keep the two values equal.
LONG_POLL_SECONDS = 50

# Values the FakeTelegramClient reports or assigns. They mirror what the real API returns.
FAKE_BOT_ID = 4242
FAKE_BOT_USERNAME = "autopilot_test_bot"
FIRST_MESSAGE_ID = 100
FAKE_PARSE_ERROR_MESSAGE = (
    "Telegram sendMessage failed: Bad Request: "
    "can't parse entities: Unsupported start tag"
)


def _require_aware(moment: datetime) -> datetime:
    """Return ``moment`` unchanged after checking that it carries a UTC offset.

    Args:
        moment: The datetime to check.

    Returns:
        The same datetime object.

    Raises:
        ValueError: If ``moment`` is naive. Naive times are ambiguous, so tests must not use them.
    """
    if moment.utcoffset() is None:
        raise ValueError(f"Expected a timezone-aware datetime, got naive {moment!r}")
    return moment


class FakeClock:
    """A clock that the test moves by hand, so time-dependent code is deterministic.

    Pass the instance itself wherever a ``Callable[[], datetime]`` is expected.

    Example:
        clock = FakeClock(datetime(2026, 11, 2, 7, 30, tzinfo=TZ))
        clock.advance(minutes=5)  # now 07:35
    """

    def __init__(self, start: datetime) -> None:
        """Start the clock at a fixed moment.

        Args:
            start: The initial moment. Must be timezone-aware.

        Raises:
            ValueError: If ``start`` is naive.
        """
        self._now = _require_aware(start)

    def __call__(self) -> datetime:
        """Return the current fake moment.

        Returns:
            The moment set by the last ``advance`` or ``set`` call, or the start moment.
        """
        return self._now

    def advance(self, **timedelta_kwargs: float) -> datetime:
        """Move the clock by a duration.

        Args:
            **timedelta_kwargs: Keyword arguments for ``datetime.timedelta``, e.g. ``minutes=5``.
                Negative values move the clock backwards.

        Returns:
            The new current moment.
        """
        self._now = self._now + timedelta(**timedelta_kwargs)
        return self._now

    def set(self, moment: datetime) -> None:
        """Jump the clock to an exact moment.

        Args:
            moment: The new current moment. Must be timezone-aware.

        Raises:
            ValueError: If ``moment`` is naive.
        """
        self._now = _require_aware(moment)


def make_options(workdir: Path | None = None, **overrides: Any) -> RunOptions:
    """Build RunOptions that are valid by default.

    Args:
        workdir: Folder the run starts in. Defaults to the system temp folder.
        **overrides: Any other RunOptions field, e.g. ``timeout_seconds=5``.

    Returns:
        A frozen RunOptions instance.
    """
    resolved_workdir = Path(tempfile.gettempdir()) if workdir is None else workdir
    return RunOptions(workdir=resolved_workdir, **overrides)


def make_task(
    name: str = "news",
    prompt: str = "Say hi",
    schedules: tuple[Schedule, ...] = (),
    **overrides: Any,
) -> Task:
    """Build a Task that is valid by default.

    Args:
        name: Task name.
        prompt: Prompt text.
        schedules: Schedules passed through exactly as given (same tuple object).
        **overrides: Any other Task field, e.g. ``enabled=False`` or ``options=make_options(...)``.

    Returns:
        A frozen Task instance.
    """
    fields: dict[str, Any] = {
        "name": name,
        "prompt": prompt,
        "schedules": schedules,
        "options": make_options(),
    }
    fields.update(overrides)
    return Task(**fields)


def make_config(
    *tasks: Task,
    ask_options: RunOptions | None = None,
    **overrides: Any,
) -> TaskConfig:
    """Build a TaskConfig from tasks, keeping the order they are given in.

    Args:
        *tasks: Tasks to include. Their names must be unique.
        ask_options: Options for chat and ask runs. Defaults to ``make_options()``.
        **overrides: Any other TaskConfig field, e.g. ``source=Path(...)``.

    Returns:
        A frozen TaskConfig instance.

    Raises:
        ValueError: If two tasks share a name. A silent overwrite would hide a test bug.
    """
    by_name: dict[str, Task] = {}
    for task in tasks:
        if task.name in by_name:
            raise ValueError(f"make_config got two tasks named {task.name!r}")
        by_name[task.name] = task
    resolved_ask_options = make_options() if ask_options is None else ask_options
    return TaskConfig(tasks=by_name, ask_options=resolved_ask_options, **overrides)


def make_record(**overrides: Any) -> RunRecord:
    """Build a finished, successful RunRecord that is valid by default.

    Args:
        **overrides: Any RunRecord field, e.g. ``status="error"`` or ``id=7``.

    Returns:
        A frozen RunRecord instance with id 1, task "news", result "Hello".
    """
    fields: dict[str, Any] = {
        "id": 1,
        "task_name": "news",
        "trigger": TRIGGER_SCHEDULE,
        "status": STATUS_SUCCESS,
        "prompt": "Say hi",
        "queued_at": datetime(2026, 11, 2, 7, 30, tzinfo=TZ),
        "result": "Hello",
    }
    fields.update(overrides)
    return RunRecord(**fields)


class FakeTelegramClient:
    """In-memory stand-in for TelegramClient with the same public methods.

    Every call is recorded instead of sent. Tests inspect the recorded lists
    and set the failure switches below to simulate Telegram misbehaving.

    Attributes:
        sent: One dict per delivered message, in send order.
        documents: One dict per delivered document, in send order.
        chat_actions: ``(chat_id, action)`` pairs from send_chat_action.
        commands: The last list passed to set_my_commands; None until called.
        updates: Batches returned by get_updates, one batch per call. An empty list is returned after the last batch.
        fail_html_once: When True, the next HTML send_message raises a parse error (400), then resets.
        raise_on_updates: When set, get_updates raises this error on every call.
        next_message_id: Id given to the next delivered message or document.
    """

    def __init__(self) -> None:
        """Create an empty fake with no failures armed."""
        self.sent: list[dict[str, Any]] = []
        self.documents: list[dict[str, Any]] = []
        self.chat_actions: list[tuple[int, str]] = []
        self.commands: list[tuple[str, str]] | None = None
        self.updates: list[list[dict[str, Any]]] = []
        self.fail_html_once: bool = False
        self.raise_on_updates: TelegramError | None = None
        self.next_message_id: int = FIRST_MESSAGE_ID

    def get_me(self) -> dict[str, Any]:
        """Return a fake bot identity.

        Returns:
            A dict with the fake bot's id, is_bot flag and username.
        """
        return {"id": FAKE_BOT_ID, "is_bot": True, "username": FAKE_BOT_USERNAME}

    def get_updates(
        self, offset: int | None = None, timeout: int = LONG_POLL_SECONDS
    ) -> list[dict[str, Any]]:
        """Return the next queued batch of updates.

        Args:
            offset: Accepted for signature compatibility and ignored.
            timeout: Accepted for signature compatibility and ignored.

        Returns:
            The oldest queued batch, or an empty list when none are queued.

        Raises:
            TelegramError: The ``raise_on_updates`` error, if one is set.
        """
        if self.raise_on_updates is not None:
            raise self.raise_on_updates
        if not self.updates:
            return []
        return self.updates.pop(0)

    def send_message(
        self,
        chat_id: int,
        text: str,
        *,
        parse_mode: str | None = "HTML",
        reply_markup: dict[str, Any] | None = None,
        disable_notification: bool = False,
    ) -> dict[str, Any]:
        """Record a text message, or fail once with a parse error if armed.

        Args:
            chat_id: Destination chat id.
            text: Message text.
            parse_mode: "HTML", None for plain text.
            reply_markup: Optional keyboard markup, recorded as given.
            disable_notification: Recorded as given.

        Returns:
            A minimal Message-shaped dict with the new message id.

        Raises:
            TelegramError: Status 400 with a "can't parse entities" message when
                ``fail_html_once`` is armed and the message is HTML.
        """
        if parse_mode == "HTML" and self.fail_html_once:
            self.fail_html_once = False
            raise TelegramError(FAKE_PARSE_ERROR_MESSAGE, status=400)
        message_id = self._take_message_id()
        self.sent.append(
            {
                "message_id": message_id,
                "chat_id": chat_id,
                "text": text,
                "parse_mode": parse_mode,
                "reply_markup": reply_markup,
                "disable_notification": disable_notification,
            }
        )
        return {"message_id": message_id, "chat": {"id": chat_id}, "text": text}

    def send_document(
        self,
        chat_id: int,
        filename: str,
        content: bytes,
        *,
        caption: str | None = None,
        parse_mode: str | None = "HTML",
        reply_markup: dict[str, Any] | None = None,
        disable_notification: bool = False,
    ) -> dict[str, Any]:
        """Record a document upload.

        Args:
            chat_id: Destination chat id.
            filename: Name shown to the recipient.
            content: File bytes, recorded as given.
            caption: Optional caption, recorded as given.
            parse_mode: Caption parse mode, recorded as given.
            reply_markup: Optional keyboard markup, recorded as given.
            disable_notification: Recorded as given.

        Returns:
            A minimal Message-shaped dict with the new message id.
        """
        message_id = self._take_message_id()
        self.documents.append(
            {
                "message_id": message_id,
                "chat_id": chat_id,
                "filename": filename,
                "content": content,
                "caption": caption,
                "parse_mode": parse_mode,
                "reply_markup": reply_markup,
                "disable_notification": disable_notification,
            }
        )
        return {"message_id": message_id, "chat": {"id": chat_id}, "document": {}}

    def set_my_commands(self, commands: Sequence[tuple[str, str]]) -> None:
        """Record the bot command list.

        Args:
            commands: ``(command, description)`` pairs, stored as a list.
        """
        self.commands = list(commands)

    def send_chat_action(self, chat_id: int, action: str = "typing") -> None:
        """Record a chat action such as "typing".

        Args:
            chat_id: Chat that should see the action.
            action: The action name.
        """
        self.chat_actions.append((chat_id, action))

    def _take_message_id(self) -> int:
        """Return the next message id and advance the counter.

        Returns:
            The id to assign to the message being recorded.
        """
        message_id = self.next_message_id
        self.next_message_id += 1
        return message_id
