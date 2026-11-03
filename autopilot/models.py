"""Shared constants, dataclasses and protocols used across the autopilot package.

This module is the common vocabulary: every other module imports its status
values, trigger names, options and records from here. It imports nothing at
runtime except the standard library, so it can never create an import cycle.

Invariants:
    * All datetimes stored in these models are timezone-aware.
    * Dataclasses are frozen; use ``dataclasses.replace`` to change one.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    # Type-only import: keeps this module free of a runtime dependency on the schedule parser.
    from autopilot.schedule import Schedule

# Run lifecycle states. A run is active while queued or running; every other state is final.
STATUS_QUEUED = "queued"
STATUS_RUNNING = "running"
STATUS_SUCCESS = "success"
STATUS_ERROR = "error"
STATUS_TIMEOUT = "timeout"
STATUS_RATE_LIMITED = "rate_limited"
STATUS_CANCELLED = "cancelled"
ACTIVE_STATUSES = frozenset({STATUS_QUEUED, STATUS_RUNNING})
FINISHED_STATUSES = frozenset(
    {
        STATUS_SUCCESS,
        STATUS_ERROR,
        STATUS_TIMEOUT,
        STATUS_RATE_LIMITED,
        STATUS_CANCELLED,
    }
)

# Why a run was created. Stored with every run so history can show its origin.
TRIGGER_SCHEDULE = "schedule"  # a scheduled slot
# CLI `autopilot run` or `autopilot ask`: one-off, fresh session.
TRIGGER_MANUAL = "manual"
TRIGGER_TELEGRAM = "telegram"  # /run in Telegram
# Telegram text or /ask: continues the current chat conversation.
TRIGGER_CHAT = "chat"
TRIGGERS = (TRIGGER_SCHEDULE, TRIGGER_MANUAL, TRIGGER_TELEGRAM, TRIGGER_CHAT)
# A retry after a usage limit keeps the original trigger and sets retry_of.

# task_name stored for chat and `autopilot ask` runs. tasks.toml must not use it as a task name.
ASK_TASK_NAME = "ask"

# Notification policies a task can choose from in tasks.toml.
NOTIFY_ALWAYS = "always"
NOTIFY_ERRORS = "errors"
NOTIFY_NEVER = "never"
NOTIFY_POLICIES = (NOTIFY_ALWAYS, NOTIFY_ERRORS, NOTIFY_NEVER)

# Values accepted by `claude --permission-mode`.
PERMISSION_MODES = (
    "acceptEdits",
    "auto",
    "bypassPermissions",
    "manual",
    "dontAsk",
    "plan",
)
EFFORT_LEVELS = ("low", "medium", "high", "xhigh", "max")
# Never prompts, so an unattended run can't hang waiting for an answer.
DEFAULT_PERMISSION_MODE = "dontAsk"

# Default run timeout when tasks.toml sets none: 30 minutes.
DEFAULT_TIMEOUT_SECONDS = 1800

# A chat conversation idle for this many hours starts fresh with the next message.
CHAT_IDLE_HOURS = 6

# Telegram bot commands with their one-line help texts. bot.py registers them
# with Telegram and formatting.py lists them in the /help text, so both read
# this single copy.
BOT_COMMANDS = (
    ("tasks", "List tasks and next runs"),
    ("run", "Run a task now: /run <task>"),
    ("ask", "Chat with Claude: /ask <text> (or just type)"),
    ("new", "Start a new conversation: /new [text]"),
    ("last", "Latest result: /last [task]"),
    ("status", "What is running and queued"),
    ("pause", "Pause a task's schedule"),
    ("resume", "Resume a task's schedule"),
    ("help", "How to use this bot"),
)


@dataclass(frozen=True)
class RunOptions:
    """How one `claude -p` invocation is configured.

    Attributes:
        workdir: Folder the Claude process starts in. Must exist.
        model: Model alias or full name passed as ``--model``; None uses the CLI default.
        effort: Effort level from EFFORT_LEVELS passed as ``--effort``; None uses the CLI default.
        allowed_tools: Tool rules passed as ``--allowedTools``, e.g. ``Bash(git log *)``.
        disallowed_tools: Tool rules passed as ``--disallowedTools``.
        permission_mode: One of PERMISSION_MODES, passed as ``--permission-mode``.
        add_dirs: Extra folders Claude may read, passed as ``--add-dir``.
        append_system_prompt: Text appended to the system prompt; None adds nothing.
        max_budget_usd: Spending cap per run in USD; None means no cap.
        timeout_seconds: Wall-clock limit for the whole run, in seconds.
    """

    workdir: Path
    model: str | None = None
    effort: str | None = None
    allowed_tools: tuple[str, ...] = ()
    disallowed_tools: tuple[str, ...] = ()
    permission_mode: str = DEFAULT_PERMISSION_MODE
    add_dirs: tuple[Path, ...] = ()
    append_system_prompt: str | None = None
    max_budget_usd: float | None = None
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS


@dataclass(frozen=True)
class Task:
    """One task from tasks.toml: a prompt, when it runs and how Claude runs it.

    Attributes:
        name: Unique task name matching TASK_NAME_RE in tasks.py. Used on the CLI and in Telegram.
        prompt: Stripped, non-empty prompt sent to Claude on stdin.
        schedules: Schedules that fire this task. Empty means manual only.
        options: How the Claude process is configured for this task.
        description: Short human-readable text shown in task lists.
        enabled: False keeps the task defined but stops its schedule.
        notify: One of NOTIFY_POLICIES, deciding when results reach Telegram.
    """

    name: str
    prompt: str
    schedules: tuple[Schedule, ...]  # empty = manual only
    options: RunOptions
    description: str = ""
    enabled: bool = True
    notify: str = NOTIFY_ALWAYS


@dataclass(frozen=True)
class TaskConfig:
    """A validated snapshot of tasks.toml.

    Attributes:
        tasks: Tasks keyed by name, in the order they appear in the file.
        ask_options: Options used by chat and `autopilot ask` runs.
        source: The file the config was loaded from; None for in-memory configs.
    """

    tasks: dict[str, Task]  # file order
    ask_options: RunOptions
    source: Path | None = None


@dataclass(frozen=True)
class RunOutcome:
    """The result of one Claude call, as reported by the runner before it is stored.

    Attributes:
        status: One of success, error, timeout or rate_limited.
        result: Text Claude returned; empty when there is none.
        error: Human-readable failure message; None on success.
        duration_ms: Wall-clock time of the call in milliseconds.
        session_id: Claude session id, used to resume chat conversations.
        num_turns: Number of model turns Claude used.
        cost_usd: API-equivalent cost reported by Claude Code. Informational on a subscription.
        denied_tools: Tool names Claude wanted to use but was not allowed to.
        retry_at: For rate_limited only: when the limit resets, if known.
    """

    status: str  # success | error | timeout | rate_limited
    result: str = ""
    error: str | None = None
    duration_ms: int = 0
    session_id: str | None = None
    num_turns: int | None = None
    # API-equivalent cost reported by Claude Code; informational on a subscription.
    cost_usd: float | None = None
    denied_tools: tuple[str, ...] = ()
    # rate_limited only: when the limit resets, if known.
    retry_at: datetime | None = None


@dataclass(frozen=True)
class RunRecord:
    """A run as stored in the database: one request to Claude and its outcome.

    Attributes:
        id: Database id of the run.
        task_name: Task that was run, or ASK_TASK_NAME for chat and ask runs.
        trigger: One of TRIGGERS.
        status: One of the STATUS_* values.
        prompt: The prompt that was sent.
        queued_at: When the run was created.
        scheduled_for: The schedule slot this run serves; None for non-scheduled runs.
        started_at: When the Claude process started; None until the run starts.
        finished_at: When the run reached a final status; None while active.
        duration_ms: Wall-clock duration in milliseconds; None until finished.
        result: Text Claude returned; empty when there is none.
        error: Failure message for non-success runs; None otherwise.
        session_id: Claude session id, if Claude reported one.
        num_turns: Number of model turns, if reported.
        cost_usd: API-equivalent cost, if reported.
        denied_tools: Tool names Claude was not allowed to use.
        retry_of: Id of the run this one retries after a usage limit; None for originals.
        retry_at: For rate_limited runs: when the run will be retried.
    """

    id: int
    task_name: str
    trigger: str
    status: str
    prompt: str
    queued_at: datetime
    scheduled_for: datetime | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None
    duration_ms: int | None = None
    result: str = ""
    error: str | None = None
    session_id: str | None = None
    num_turns: int | None = None
    cost_usd: float | None = None
    denied_tools: tuple[str, ...] = ()
    retry_of: int | None = None
    retry_at: datetime | None = None

    @property
    def is_finished(self) -> bool:
        """Whether this run has reached a final status.

        Returns:
            True when ``status`` is in FINISHED_STATUSES, False while queued or running.
        """
        return self.status in FINISHED_STATUSES


@dataclass(frozen=True)
class DueRun:
    """A scheduled slot that the engine must enqueue.

    Attributes:
        task_name: Name of the task to run.
        scheduled_for: The slot time the run serves, timezone-aware.
    """

    task_name: str
    scheduled_for: datetime


@dataclass(frozen=True)
class QueueSnapshot:
    """A point-in-time view of the work queue, for status displays.

    Attributes:
        running: Runs currently executing.
        queued: Runs waiting for a worker, oldest first.
        limited_until: When the usage-limit gate reopens; None while the gate is open.
    """

    running: tuple[RunRecord, ...]
    queued: tuple[RunRecord, ...]  # oldest first
    limited_until: datetime | None  # set while the usage-limit gate is closed


class Runner(Protocol):
    """Anything that can execute one Claude prompt and report its outcome."""

    def run(
        self,
        prompt: str,
        options: RunOptions,
        resume_session: str | None = None,
    ) -> RunOutcome:
        """Execute one prompt.

        Args:
            prompt: Text sent to Claude on stdin.
            options: How the Claude process is configured.
            resume_session: Session id to continue, or None to start a new session.

        Returns:
            The outcome. Implementations report failures as an outcome, not by raising.
        """
        ...


class Notifier(Protocol):
    """Anything that can deliver run results and plain messages to the user."""

    def notify_run(self, record: RunRecord) -> None:
        """Deliver a finished run's result.

        Args:
            record: The finished run to report.
        """
        ...

    def notify_text(self, text: str) -> None:
        """Deliver a plain text message.

        Args:
            text: The message to deliver.
        """
        ...
