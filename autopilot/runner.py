"""Runs one prompt through the Claude Code CLI (`claude -p`) and turns the answer into a RunOutcome.

Responsibilities:
    * Locate the Claude binary (``find_claude_binary``), per run, because VS Code
      extension updates move the bundled binary.
    * Build the command line and the child environment (``build_command``,
      ``child_environment``). The prompt always travels on stdin, never in argv.
    * Run the process in its own session, enforce the timeout and kill the whole
      process group when it is exceeded (``ClaudeRunner``).
    * Read the JSON the CLI prints and detect usage limits (``parse_output``,
      ``detect_rate_limit``).

Where it fits: ``autopilot.engine`` calls ``ClaudeRunner.run`` (it satisfies the
``models.Runner`` protocol); ``autopilot.doctor`` and ``autopilot.app`` use
``find_claude_binary``. Depends on ``autopilot.models`` and ``autopilot.errors``.

Invariants:
    * The subscription login is used: ANTHROPIC_API_KEY and ANTHROPIC_AUTH_TOKEN
      never reach the child process. Neither do TELEGRAM_* variables (the bot token).
    * Every Claude process group this module starts is tracked while it runs, so
      ``ClaudeRunner.terminate_all`` can stop them on shutdown.
    * Rate-limit detection never runs on a successful result, so a task that merely
      talks about "usage limits" is never mistaken for one.
    * All datetimes are timezone-aware; "now" comes from an injected clock.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import signal
import subprocess
import threading
import time
from collections.abc import Callable, Mapping
from datetime import UTC, date, datetime, time as day_time, timedelta
from pathlib import Path
from typing import Any

from autopilot.errors import ClaudeNotFoundError
from autopilot.models import (
    STATUS_ERROR,
    STATUS_RATE_LIMITED,
    STATUS_SUCCESS,
    STATUS_TIMEOUT,
    RunOptions,
    RunOutcome,
)
from autopilot.schedule import MONTH_NAMES, WEEKDAY_NAMES

logger = logging.getLogger(__name__)

# Credentials that would make Claude Code bill an API account instead of the subscription.
STRIP_ENV = ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN")
# CLAUDE_* variables the child still needs: the subscription token and the config folder.
KEEP_CLAUDE_ENV = ("CLAUDE_CODE_OAUTH_TOKEN", "CLAUDE_CONFIG_DIR")
# Set by a parent Claude Code session; passing it down would confuse the child.
PARENT_SESSION_ENV = "CLAUDECODE"
CLAUDE_ENV_PREFIX = "CLAUDE_"
# Variables with these prefixes never reach Claude: a prompt-injected task could read the
# Telegram bot token from its environment and leak it.
STRIPPED_ENV_PREFIXES = ("TELEGRAM_",)

# Seconds between SIGTERM and SIGKILL when a run exceeds its timeout.
KILL_GRACE_SECONDS = 10
# Extra wait after a reported reset time, so the first retry doesn't land a second too early.
RESET_BUFFER = timedelta(minutes=2)
# A reset time further away than this is a parsing accident, not a real reset.
MAX_RESET_AHEAD = timedelta(days=8)
# Longest slice of CLI output kept in an error message.
MAX_ERROR_CHARS = 1000

# Where find_claude_binary looks, relative to the home folder, in this order.
LOCAL_BINARY_RELATIVE_PATHS = (Path(".local/bin/claude"), Path(".claude/local/claude"))
VSCODE_EXTENSION_DIRS = (Path(".vscode/extensions"), Path(".vscode-server/extensions"))
EXTENSION_GLOB = "anthropic.claude-code-*"
EXTENSION_BINARY_RELATIVE_PATH = Path("resources/native-binary/claude")
# Pulls "2.1.314" out of "anthropic.claude-code-2.1.314-linux-x64".
EXTENSION_VERSION_RE = re.compile(r"claude-code-(\d+(?:\.\d+)*)")
INSTALL_HINT = (
    "Fix: install it with `claude install` from any working copy, "
    "or set CLAUDE_BIN in .env to the full path of the claude binary."
)

# Matches the phrases Claude Code uses when the subscription's usage limit is hit,
# e.g. "You've hit your session limit", "usage limit reached", "out of extra usage".
RATE_LIMIT_RE = re.compile(
    r"\b(usage|rate|session|weekly|5-hour|five-hour)[ -]limit\b"
    r"|\blimit (reached|exceeded|hit)\b"
    r"|\bhit your limit\b"
    r"|\bout of (extra )?usage\b",
    re.IGNORECASE,
)
# Matches the machine-readable reset time "...limit reached|1760000000" (epoch seconds).
RESET_EPOCH_RE = re.compile(r"\|(\d{10})\b")
# Matches a human reset time such as "resets 5pm", "resets at 5:30 pm", "reset 17:00",
# optionally preceded by a month and day ("resets Nov 7, 9am", "resets Nov 7 at 9:30am")
# or by a weekday ("resets Thu 9am"), as weekly-limit messages are worded.
RESET_CLOCK_RE = re.compile(
    r"resets?\s+"
    rf"(?:(?P<month>{'|'.join(MONTH_NAMES)})[a-z]*\.?\s+(?P<day>\d{{1,2}})(?:st|nd|rd|th)?,?\s+"
    rf"|(?P<weekday>{'|'.join(WEEKDAY_NAMES)})[a-z]*\.?,?\s+)?"
    r"(?:at\s+)?(?P<hour>\d{1,2})(?::(?P<minute>\d{2}))?\s*(?P<meridiem>[ap]\.?m\.?)?",
    re.IGNORECASE,
)
DAYS_IN_WEEK = 7
HTTP_TOO_MANY_REQUESTS = 429
MAX_HOUR_24 = 23
MAX_MINUTE = 59
MAX_HOUR_12 = 12
HOURS_PER_HALF_DAY = 12

SECONDS_PER_MINUTE = 60
MILLISECONDS_PER_SECOND = 1000

# Messages for results that carry no text, keyed by the JSON "subtype".
SUBTYPE_MESSAGES = {
    "error_max_turns": "Stopped: hit the maximum number of turns.",
    "error_during_execution": "Claude Code failed during execution.",
    "error_max_budget_usd": "Stopped: reached this task's max_budget_usd.",
}
UNREADABLE_OUTPUT_MESSAGE = "Claude CLI returned output autopilot couldn't read"


def _is_executable_file(path: Path) -> bool:
    """Tell whether ``path`` is an existing, executable regular file.

    Args:
        path: Candidate binary. A broken symlink counts as not existing.

    Returns:
        True when the file exists (symlinks followed) and the user may execute it.
    """
    return path.is_file() and os.access(path, os.X_OK)


def _extension_version_key(folder: Path) -> tuple[int, ...]:
    """Return a sortable version tuple for an extension folder name.

    Args:
        folder: Folder such as ``anthropic.claude-code-2.1.314-linux-x64``.

    Returns:
        The numeric version, e.g. ``(2, 1, 314)``; an empty tuple (the oldest) when
        the name carries no version. Comparing numbers keeps 2.1.10 newer than 2.1.9.
    """
    match = EXTENSION_VERSION_RE.search(folder.name)
    if match is None:
        return ()
    return tuple(int(part) for part in match.group(1).split("."))


def _extension_binaries(extensions_dir: Path) -> list[Path]:
    """List the bundled Claude binaries of VS Code extensions, newest version first.

    Args:
        extensions_dir: A VS Code ``extensions`` folder; it may not exist.

    Returns:
        Candidate binary paths (not yet checked for being executable).
    """
    folders = sorted(extensions_dir.glob(EXTENSION_GLOB), key=_extension_version_key, reverse=True)
    return [folder / EXTENSION_BINARY_RELATIVE_PATH for folder in folders]


def find_claude_binary(
    configured: str | None = None,
    *,
    home: Path | None = None,
    which: Callable[[str], str | None] = shutil.which,
) -> Path:
    """Locate the Claude Code CLI binary.

    Order: the configured path (CLAUDE_BIN), ``claude`` on PATH, ``~/.local/bin/claude``,
    ``~/.claude/local/claude``, the newest bundled VS Code extension binary, then the
    same under ``~/.vscode-server``. Broken symlinks and non-executables are skipped.

    Args:
        configured: Explicit path from CLAUDE_BIN. When given it must be usable;
            there is no fallback, so a typo is reported instead of hidden.
        home: Home folder to search under. Defaults to the current user's home.
        which: PATH lookup function, replaceable in tests.

    Returns:
        The path of an executable Claude binary.

    Raises:
        ClaudeNotFoundError: If the configured path is unusable, or nothing was found.
            The message lists every place checked and how to fix it.

    Example:
        >>> binary = find_claude_binary()  # doctest: +SKIP
    """
    if configured:
        try:
            configured_path = Path(configured).expanduser()
        except RuntimeError as err:
            # "~nosuchuser/claude": the home folder of that user can't be determined.
            raise ClaudeNotFoundError(
                f"CLAUDE_BIN points to '{configured}', whose home folder can't be "
                f"resolved ({err}). "
                "Fix the path in .env or remove CLAUDE_BIN to search automatically."
            ) from err
        if not _is_executable_file(configured_path):
            raise ClaudeNotFoundError(
                f"CLAUDE_BIN points to '{configured}', which is not an executable file. "
                "Fix the path in .env or remove CLAUDE_BIN to search automatically."
            )
        return configured_path

    home_dir = Path.home() if home is None else home
    checked: list[str] = []

    # PATH comes first so a user's own install wins over bundled copies.
    on_path = which("claude")
    checked.append("claude on PATH")
    if on_path and _is_executable_file(Path(on_path)):
        return Path(on_path)

    candidates = [home_dir / relative for relative in LOCAL_BINARY_RELATIVE_PATHS]
    for extensions_dir in VSCODE_EXTENSION_DIRS:
        candidates.extend(_extension_binaries(home_dir / extensions_dir))
    for candidate in candidates:
        if _is_executable_file(candidate):
            return candidate

    checked.extend(str(home_dir / relative) for relative in LOCAL_BINARY_RELATIVE_PATHS)
    checked.extend(
        str(home_dir / extensions_dir / EXTENSION_GLOB / EXTENSION_BINARY_RELATIVE_PATH)
        for extensions_dir in VSCODE_EXTENSION_DIRS
    )
    raise ClaudeNotFoundError(
        f"Claude Code CLI not found. Checked: {', '.join(checked)}. {INSTALL_HINT}"
    )


def build_command(
    binary: Path, options: RunOptions, resume_session: str | None = None
) -> list[str]:
    """Build the argument list for one headless Claude run.

    The prompt is not part of the list: it is written to stdin, so a prompt that
    starts with "-" can't be mistaken for a flag. Variadic flags (``--add-dir``,
    ``--allowedTools``, ``--disallowedTools``) come last so they can't swallow
    other arguments. ``--bare`` is never used because it ignores the subscription login.

    Args:
        binary: Path of the Claude binary.
        options: Per-run settings.
        resume_session: Session id to continue (chat turns), or None for a fresh session.

    Returns:
        The argv list, for ``subprocess.Popen`` without a shell.

    Example:
        >>> build_command(Path("/bin/claude"), RunOptions(workdir=Path("/tmp")))[:3]
        ['/bin/claude', '-p', '--output-format']
    """
    command = [
        str(binary),
        "-p",
        "--output-format",
        "json",
        "--permission-prompts",
        "none",
        "--permission-mode",
        options.permission_mode,
    ]
    # Single-value flags first, each only when the option is set.
    if options.model:
        command += ["--model", options.model]
    if options.effort:
        command += ["--effort", options.effort]
    if options.max_budget_usd is not None:
        command += ["--max-budget-usd", str(options.max_budget_usd)]
    if options.append_system_prompt:
        command += ["--append-system-prompt", options.append_system_prompt]
    if resume_session:
        command += ["--resume", resume_session]
    # Variadic flags last: they consume every following argument up to the next flag.
    if options.add_dirs:
        command += ["--add-dir", *(str(directory) for directory in options.add_dirs)]
    if options.allowed_tools:
        command += ["--allowedTools", *options.allowed_tools]
    if options.disallowed_tools:
        command += ["--disallowedTools", *options.disallowed_tools]
    return command


def child_environment(base: Mapping[str, str]) -> dict[str, str]:
    """Build the environment for the Claude child process.

    Removes API credentials (so the subscription login is used), every TELEGRAM_*
    variable (the bot token must not be readable by a prompted tool call) and everything
    a parent Claude Code session exported, except the variables Claude needs to log in.

    Args:
        base: The environment to start from, usually ``os.environ``.

    Returns:
        A new dict; ``base`` is not modified.
    """
    environment: dict[str, str] = {}
    for key, value in base.items():
        if key in STRIP_ENV or key == PARENT_SESSION_ENV:
            continue
        if key.startswith(STRIPPED_ENV_PREFIXES):
            continue
        if key.startswith(CLAUDE_ENV_PREFIX) and key not in KEEP_CLAUDE_ENV:
            continue
        environment[key] = value
    return environment


def _require_aware(moment: datetime) -> datetime:
    """Return ``moment`` after checking that it carries a UTC offset.

    Args:
        moment: Datetime to check.

    Returns:
        The same datetime.

    Raises:
        ValueError: If ``moment`` is naive.
    """
    if moment.utcoffset() is None:
        raise ValueError(f"Expected a timezone-aware datetime, got naive {moment!r}")
    return moment


def _time_of_day(match: re.Match[str]) -> day_time | None:
    """Read the hour, minute and optional am/pm of a reset-time match.

    Args:
        match: A match of RESET_CLOCK_RE.

    Returns:
        The time of day, or None when the numbers are not valid (e.g. "13pm", "25:00").
    """
    hour = int(match.group("hour"))
    minute = int(match.group("minute") or 0)
    meridiem = (match.group("meridiem") or "").lower()
    if meridiem:
        # 12-hour clock: 12am is midnight, 12pm is noon.
        if not 1 <= hour <= MAX_HOUR_12:
            return None
        hour %= HOURS_PER_HALF_DAY
        if meridiem.startswith("p"):
            hour += HOURS_PER_HALF_DAY
    if hour > MAX_HOUR_24 or minute > MAX_MINUTE:
        return None
    return day_time(hour, minute)


def _candidate_days(match: re.Match[str], now: datetime) -> list[date]:
    """List, soonest first, the calendar days a reset-time match could refer to.

    Args:
        match: A match of RESET_CLOCK_RE.
        now: Current time, aware; its tzinfo defines which calendar day "today" is.

    Returns:
        The named month-day (this year and next), the next matching weekdays (today
        included), or just today and tomorrow when the text names no date.
    """
    today = now.date()
    if match.group("month"):
        month = MONTH_NAMES.index(match.group("month").lower()) + 1
        day = int(match.group("day"))
        days = []
        for year in (today.year, today.year + 1):
            try:
                days.append(date(year, month, day))
            except ValueError:
                # Impossible date such as "Feb 30", or Feb 29 in a non-leap year.
                continue
        return days
    if match.group("weekday"):
        weekday = WEEKDAY_NAMES.index(match.group("weekday").lower())
        week = [today + timedelta(days=offset) for offset in range(DAYS_IN_WEEK + 1)]
        return [day for day in week if day.weekday() == weekday]
    return [today, today + timedelta(days=1)]


def _next_wall_time(match: re.Match[str], now: datetime) -> datetime | None:
    """Turn a "resets 5pm" style match into the next such wall-clock moment after ``now``.

    Args:
        match: A match of RESET_CLOCK_RE.
        now: Current time, aware; its tzinfo defines the wall clock.

    Returns:
        The first moment strictly after ``now`` that has the named time of day (on the
        named date or weekday, when there is one), or None when the numbers or the
        date are not valid.
    """
    time_of_day = _time_of_day(match)
    if time_of_day is None:
        return None
    # Compare as real instants: aware datetimes sharing a tzinfo compare by wall time only.
    now_utc = now.astimezone(UTC)
    for day in _candidate_days(match, now):
        candidate = datetime.combine(day, time_of_day, tzinfo=now.tzinfo)
        if candidate.astimezone(UTC) > now_utc:
            return candidate
    return None


def _parse_reset_time(text: str, now: datetime) -> datetime | None:
    """Find the usage-limit reset moment in ``text`` (without the safety buffer).

    Args:
        text: Output of a limited run.
        now: Current time, aware.

    Returns:
        The reset moment, or None when the text names none. The epoch form wins
        over the wall-clock form because it is exact.
    """
    epoch_match = RESET_EPOCH_RE.search(text)
    if epoch_match:
        return datetime.fromtimestamp(int(epoch_match.group(1)), tz=now.tzinfo)
    clock_match = RESET_CLOCK_RE.search(text)
    if clock_match:
        return _next_wall_time(clock_match, now)
    return None


def detect_rate_limit(text: str, now: datetime) -> tuple[bool, datetime | None]:
    """Tell whether ``text`` reports a Claude usage limit, and when it resets.

    Args:
        text: Result text, stderr or stdout of a failed run.
        now: Current time, aware; wall-clock reset times are read in its time zone.

    Returns:
        ``(is_limited, retry_at)``. ``retry_at`` is the reset moment plus
        RESET_BUFFER, or None when no time is named, the time is already past,
        or it is more than 8 days away (probably misread).

    Raises:
        ValueError: If ``now`` is naive.

    Example:
        >>> detect_rate_limit("You've hit your session limit", now)[0]  # doctest: +SKIP
        True
    """
    _require_aware(now)
    if not RATE_LIMIT_RE.search(text):
        return False, None
    reset = _parse_reset_time(text, now)
    if reset is None or reset.astimezone(UTC) <= now.astimezone(UTC):
        return True, None
    retry_at = reset + RESET_BUFFER
    if retry_at.astimezone(UTC) - now.astimezone(UTC) > MAX_RESET_AHEAD:
        return True, None
    return True, retry_at


def _tail(text: str) -> str:
    """Return the last MAX_ERROR_CHARS characters of ``text``, stripped.

    Args:
        text: Raw process output.

    Returns:
        The stripped tail; the end is kept because errors usually come last.
    """
    return text.strip()[-MAX_ERROR_CHARS:]


def _extract_json_object(stdout: str) -> dict[str, Any] | None:
    """Find the JSON object the CLI printed.

    Args:
        stdout: Everything the process wrote to stdout.

    Returns:
        The object parsed from the whole output, else from the last non-empty line
        when it starts with "{" (tolerates warnings printed before it); None if neither parses.
    """
    candidates = [stdout.strip()]
    lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    if lines and lines[-1].startswith("{"):
        candidates.append(lines[-1])
    for candidate in candidates:
        try:
            data = json.loads(candidate)
        except ValueError:
            continue
        if isinstance(data, dict):
            return data
    return None


def _number(value: Any) -> int | float | None:
    """Return ``value`` when it is a real number (not a bool), else None.

    Args:
        value: A JSON value.

    Returns:
        The number, or None.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return value


def _denied_tools(payload: Mapping[str, Any]) -> tuple[str, ...]:
    """List the tools Claude was not allowed to use, without duplicates, in order.

    Args:
        payload: The parsed result object.

    Returns:
        Unique ``tool_name`` values from ``permission_denials``.
    """
    denials = payload.get("permission_denials")
    if not isinstance(denials, list):
        return ()
    names = (entry.get("tool_name") for entry in denials if isinstance(entry, dict))
    return tuple(dict.fromkeys(name for name in names if isinstance(name, str) and name))


def _failure_message(payload: Mapping[str, Any], result_text: str, stderr: str) -> str:
    """Explain a failed result: its text, else a message for its subtype, plus the stderr tail.

    Args:
        payload: The parsed result object.
        result_text: The ``result`` text, possibly empty.
        stderr: Standard error of the process.

    Returns:
        A message of one or more lines for the run's ``error`` field.
    """
    message = result_text.strip()
    if not message:
        subtype = payload.get("subtype")
        message = SUBTYPE_MESSAGES.get(
            subtype if isinstance(subtype, str) else "",
            f"Claude Code stopped: {subtype or 'unknown'}.",
        )
    stderr_tail = _tail(stderr)
    return f"{message}\n{stderr_tail}" if stderr_tail else message


def _outcome_from_result(
    payload: Mapping[str, Any], stderr: str, exit_code: int, duration_ms: int, now: datetime
) -> RunOutcome:
    """Convert a ``type == "result"`` JSON object into a RunOutcome.

    Args:
        payload: The parsed result object.
        stderr: Standard error of the process.
        exit_code: Process exit code.
        duration_ms: Wall-clock duration to use when the JSON has none.
        now: Current time, aware.

    Returns:
        A success, rate_limited or error outcome.
    """
    raw_result = payload.get("result")
    result_text = raw_result if isinstance(raw_result, str) else ""
    reported_duration = _number(payload.get("duration_ms"))
    turns = _number(payload.get("num_turns"))
    session_id = payload.get("session_id")
    # Fields shared by every result outcome, whatever its status.
    common: dict[str, Any] = {
        "duration_ms": duration_ms if reported_duration is None else int(reported_duration),
        "session_id": session_id if isinstance(session_id, str) else None,
        "num_turns": None if turns is None else int(turns),
        "cost_usd": _number(payload.get("total_cost_usd")),
        "denied_tools": _denied_tools(payload),
    }

    succeeded = (
        not payload.get("is_error") and payload.get("subtype") == "success" and exit_code == 0
    )
    if succeeded:
        return RunOutcome(status=STATUS_SUCCESS, result=result_text, **common)

    message = _failure_message(payload, result_text, stderr)
    limited, retry_at = detect_rate_limit(f"{result_text}\n{stderr}", now)
    if limited or payload.get("api_error_status") == HTTP_TOO_MANY_REQUESTS:
        return RunOutcome(status=STATUS_RATE_LIMITED, error=message, retry_at=retry_at, **common)
    return RunOutcome(status=STATUS_ERROR, error=message, **common)


def parse_output(
    stdout: str, stderr: str, exit_code: int, duration_ms: int, now: datetime
) -> RunOutcome:
    """Turn the output of a finished `claude -p --output-format json` run into a RunOutcome.

    Args:
        stdout: Standard output of the process.
        stderr: Standard error of the process.
        exit_code: Process exit code.
        duration_ms: Measured duration, used when the JSON does not report one.
        now: Current time, aware; used to read usage-limit reset times.

    Returns:
        status ``success`` for a clean result; ``rate_limited`` when Claude reports a
        usage limit (with ``retry_at`` when the reset time is known); otherwise ``error``
        with a message that says what happened. Error text is capped at MAX_ERROR_CHARS
        for process output tails.

    Raises:
        ValueError: If ``now`` is naive.
    """
    _require_aware(now)
    payload = _extract_json_object(stdout)
    if payload is not None and payload.get("type") == "result":
        return _outcome_from_result(payload, stderr, exit_code, duration_ms, now)

    if exit_code != 0:
        # Without JSON the limit notice may be anywhere in the raw output.
        limited, retry_at = detect_rate_limit(f"{stdout}\n{stderr}", now)
        message = f"Claude CLI exited with code {exit_code}: {_tail(stderr) or _tail(stdout)}"
        if limited:
            return RunOutcome(
                status=STATUS_RATE_LIMITED,
                error=message,
                duration_ms=duration_ms,
                retry_at=retry_at,
            )
        return RunOutcome(status=STATUS_ERROR, error=message, duration_ms=duration_ms)

    message = f"{UNREADABLE_OUTPUT_MESSAGE}: {_tail(stdout) or _tail(stderr)}"
    return RunOutcome(status=STATUS_ERROR, error=message, duration_ms=duration_ms)


def _signal_group(process_group_id: int, signal_number: signal.Signals) -> None:
    """Send a signal to a whole process group, ignoring a group that no longer exists.

    Args:
        process_group_id: Group id; equals the leader's pid because the child
            was started with ``start_new_session=True``.
        signal_number: Signal to deliver.
    """
    try:
        os.killpg(process_group_id, signal_number)
    except ProcessLookupError:
        # The group already exited (normally, or killed a moment ago): nothing left to signal.
        pass


def _kill_process_group(process: subprocess.Popen[str]) -> None:
    """Stop a process and everything it spawned: SIGTERM, a grace period, then SIGKILL.

    SIGKILL is sent even when the leader exited during the grace period, so
    children that ignore SIGTERM can't survive. Afterwards the leader is reaped.

    Args:
        process: The child started in its own session.

    Side effects:
        Blocks up to KILL_GRACE_SECONDS.
    """
    _signal_group(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=KILL_GRACE_SECONDS)
    except subprocess.TimeoutExpired:
        logger.warning("Claude did not stop after SIGTERM; sending SIGKILL to its process group")
    _signal_group(process.pid, signal.SIGKILL)
    process.wait()


def _describe_timeout(seconds: int) -> str:
    """Phrase a timeout for humans.

    Args:
        seconds: The timeout in seconds.

    Returns:
        "N minute(s)" for whole minutes, else "N seconds".
    """
    minutes, remainder = divmod(seconds, SECONDS_PER_MINUTE)
    if remainder == 0:
        return f"{minutes} minute{'' if minutes == 1 else 's'}"
    return f"{seconds} seconds"


def _start_process(
    command: list[str], options: RunOptions, environment: Mapping[str, str]
) -> subprocess.Popen[str]:
    """Start the Claude CLI with piped stdin, stdout and stderr in a new session.

    Args:
        command: The argv from ``build_command``.
        options: Per-run settings; the process starts in ``options.workdir``.
        environment: The child's environment, from ``child_environment``.

    Returns:
        The running process. It leads its own process group, so the whole tree can be killed.

    Raises:
        OSError: If the process can't start (missing folder, unusable binary).
    """
    return subprocess.Popen(
        command,
        cwd=options.workdir,
        env=dict(environment),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        start_new_session=True,
    )


class ClaudeRunner:
    """Runs prompts through the Claude CLI; implements the ``models.Runner`` protocol.

    Example:
        runner = ClaudeRunner(lambda: find_claude_binary(), clock=lambda: datetime.now(UTC))
        outcome = runner.run("Summarize today's news", RunOptions(workdir=Path.home()))
    """

    def __init__(
        self,
        resolve_binary: Callable[[], Path],
        *,
        clock: Callable[[], datetime],
        base_env: Mapping[str, str] | None = None,
    ) -> None:
        """Create a runner.

        Args:
            resolve_binary: Called before every run to find the binary, so an
                updated or moved installation is picked up without a restart.
            clock: Returns the current aware time; used to read usage-limit reset times.
            base_env: Environment to derive the child's from; defaults to ``os.environ``
                at the time of each run.
        """
        self._resolve_binary = resolve_binary
        self._clock = clock
        self._base_env = base_env
        # Live children by pid, so terminate_all can find them from another thread.
        self._live: dict[int, subprocess.Popen[str]] = {}
        self._lock = threading.Lock()

    def run(
        self, prompt: str, options: RunOptions, resume_session: str | None = None
    ) -> RunOutcome:
        """Run one prompt to completion and report what happened.

        Never raises for problems with Claude itself (missing binary, start failure,
        timeout, bad output): those become outcomes.

        Args:
            prompt: Text written to the CLI's stdin.
            options: Per-run settings, including the working directory and timeout.
            resume_session: Session id to continue, or None for a fresh session.

        Returns:
            The RunOutcome; status ``timeout`` after ``options.timeout_seconds``.

        Raises:
            BaseException: Whatever interrupted the wait (KeyboardInterrupt, SystemExit or a
                bug) is re-raised unchanged, after the process group has been killed.
        """
        try:
            binary = self._resolve_binary()
        except ClaudeNotFoundError as err:
            return RunOutcome(status=STATUS_ERROR, error=str(err))

        command = build_command(binary, options, resume_session)
        environment = child_environment(os.environ if self._base_env is None else self._base_env)
        logger.debug("Starting Claude in %s: %s", options.workdir, command)
        started = time.monotonic()
        try:
            process = _start_process(command, options, environment)
        except OSError as err:
            return RunOutcome(status=STATUS_ERROR, error=f"Could not start the Claude CLI: {err}")
        return self._supervise(process, prompt, options, started)

    def terminate_all(self) -> int:
        """Stop every Claude process group this runner has started and not yet seen finish.

        Used on shutdown so Ctrl-C or SIGTERM doesn't leave Claude running. Sends SIGTERM
        to each group, waits up to KILL_GRACE_SECONDS in total, then SIGKILLs the groups.
        The ``run`` calls waiting on those processes return normally (as failed runs).

        Returns:
            The number of process groups signalled; 0 when nothing was running.

        Side effects:
            Blocks up to KILL_GRACE_SECONDS.
        """
        with self._lock:
            live = list(self._live.values())
        for process in live:
            _signal_group(process.pid, signal.SIGTERM)
        deadline = time.monotonic() + KILL_GRACE_SECONDS
        for process in live:
            try:
                process.wait(timeout=max(0.0, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                logger.warning("Claude (pid %s) ignored SIGTERM; sending SIGKILL", process.pid)
        # Always SIGKILL: a leader that exited may have left children that ignore SIGTERM.
        for process in live:
            _signal_group(process.pid, signal.SIGKILL)
        return len(live)

    def _supervise(
        self, process: subprocess.Popen[str], prompt: str, options: RunOptions, started: float
    ) -> RunOutcome:
        """Feed the prompt to a started process, wait for it and build the outcome.

        Args:
            process: The running child (its own session and process group).
            prompt: Text written to its stdin.
            options: Per-run settings; ``timeout_seconds`` bounds the wait.
            started: ``time.monotonic()`` reading from just before the start.

        Returns:
            The parsed outcome, or a ``timeout`` outcome after killing the group.

        Raises:
            BaseException: Any other interruption, re-raised after the group is killed.
        """
        with self._lock:
            self._live[process.pid] = process
        try:
            with process:
                try:
                    stdout, stderr = process.communicate(prompt, timeout=options.timeout_seconds)
                except subprocess.TimeoutExpired:
                    _kill_process_group(process)
                    return RunOutcome(
                        status=STATUS_TIMEOUT,
                        error=f"Timed out after {_describe_timeout(options.timeout_seconds)}.",
                        duration_ms=self._elapsed_ms(started),
                    )
                except BaseException:
                    # Includes KeyboardInterrupt: never leave a Claude tree running behind us.
                    _kill_process_group(process)
                    raise
        finally:
            with self._lock:
                self._live.pop(process.pid, None)
        return parse_output(
            stdout, stderr, process.returncode, self._elapsed_ms(started), self._clock()
        )

    @staticmethod
    def _elapsed_ms(started: float) -> int:
        """Return the milliseconds since a ``time.monotonic()`` reading.

        Args:
            started: Value of ``time.monotonic()`` taken at the start.

        Returns:
            Elapsed whole milliseconds.
        """
        return int((time.monotonic() - started) * MILLISECONDS_PER_SECOND)
