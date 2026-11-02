"""Exception hierarchy for Claude Autopilot.

Every error the user can cause or fix derives from AutopilotError. The CLI
catches that base class once, prints its message as one line on stderr and
exits 1, so each message must say what went wrong and what to do about it.
Other exceptions are bugs and reach the top-level handler with a traceback.

Used by: every module that validates input or talks to an external tool.
Depends on: the standard library only.
"""

from __future__ import annotations

from collections.abc import Iterable


class AutopilotError(Exception):
    """An expected, user-facing error.

    The CLI prints ``str(err)`` to stderr and exits 1. Messages must not
    contain secrets such as the Telegram bot token.

    Example:
        Raise it with the fix included, e.g.
        ``raise AutopilotError("tasks.toml not found. Run `autopilot init`.")``.
    """


class ConfigError(AutopilotError):
    """Invalid environment, .env or tasks.toml content."""


class ScheduleError(ConfigError):
    """A schedule expression can't be parsed or never fires."""


class TaskNotFoundError(AutopilotError):
    """A command named a task that tasks.toml doesn't define.

    Attributes:
        name: The unknown task name exactly as the user typed it.
    """

    def __init__(self, name: str, known: Iterable[str]) -> None:
        """Build the error and its message, listing the tasks that do exist.

        Args:
            name: The unknown task name.
            known: Names of the defined tasks, in file order. Consumed once.

        Example:
            >>> str(TaskNotFoundError("x", ["a", "b"]))
            "Unknown task 'x'. Known tasks: a, b"
        """
        self.name = name
        known_names = list(known)
        if known_names:
            message = f"Unknown task '{name}'. Known tasks: {', '.join(known_names)}"
        else:
            message = f"Unknown task '{name}'. No tasks are defined."
        super().__init__(message)


class ClaudeNotFoundError(AutopilotError):
    """The Claude Code CLI binary can't be located."""


class TelegramError(AutopilotError):
    """A Telegram Bot API call failed.

    Attributes:
        status: HTTP status code returned by Telegram, or None for network failures.
        retry_after: Seconds Telegram asked us to wait before retrying, or None.
    """

    def __init__(
        self,
        message: str,
        *,
        status: int | None = None,
        retry_after: int | None = None,
    ) -> None:
        """Build the error. The message must not contain the bot token or a request URL.

        Args:
            message: Human-readable description of the failure.
            status: HTTP status code, if Telegram answered.
            retry_after: Seconds to wait before retrying, if Telegram said so.
        """
        super().__init__(message)
        self.status = status
        self.retry_after = retry_after
