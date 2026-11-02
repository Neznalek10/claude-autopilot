"""Parsing and validation of tasks.toml.

Responsible for turning the TOML data into a ``TaskConfig`` and rejecting every
mistake with a message that names the task and the field. Nothing is silently
ignored: unknown keys (typos such as ``shedule``) are errors.

Used by: engine.py (hot reload), app.py, doctor.py and cli.py (validate, list).
Uses: models.py, schedule.py and errors.py.

Invariants:
    * Every error is a ``ConfigError`` starting with ``tasks.toml: `` followed
      by ``task 'name': ``, ``task #N: `` (name missing) or ``[defaults]: `` /
      ``[ask]: ``, then the field and what is expected.
    * Merge order is built-in defaults, then ``[defaults]``, then the task.
    * A schedule that can never fire (``0 0 30 2 *``) fails at load time.
"""

from __future__ import annotations

import math
import re
import tomllib
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from autopilot.errors import ConfigError, ScheduleError
from autopilot.models import (
    ASK_TASK_NAME,
    DEFAULT_PERMISSION_MODE,
    EFFORT_LEVELS,
    NOTIFY_ALWAYS,
    NOTIFY_POLICIES,
    PERMISSION_MODES,
    RunOptions,
    Task,
    TaskConfig,
)
from autopilot.schedule import Schedule, parse_schedule

# Task names are used in Telegram commands and file names: short, lower case, no spaces.
TASK_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,39}$")
# A tool rule: Read, WebSearch, mcp__srv__tool, or a pattern such as Bash(git log *).
TOOL_RULE_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(\(.+\))?$")
# Model alias or id such as sonnet, claude-sonnet-4-5 or opus[1m].
MODEL_RE = re.compile(r"^[A-Za-z0-9._\[\]-]+$")

NAME_HINT = "1-40 characters: lower-case letters, digits, '-' or '_', starting with a letter or digit"
TOOL_RULE_HINT = "a tool name such as Read, or a rule such as Bash(git log *)"
MODEL_HINT = "a model alias or id such as sonnet or claude-sonnet-4-5"

DEFAULT_ALLOWED_TOOLS = ("Read", "Glob", "Grep", "WebSearch", "WebFetch")
DEFAULT_TIMEOUT_MINUTES = 30
DEFAULT_ASK_TIMEOUT_MINUTES = 15
MAX_TIMEOUT_MINUTES = 720
SECONDS_PER_MINUTE = 60

TOP_LEVEL_KEYS = ("defaults", "ask", "task")
OPTION_KEYS = (
    "workdir",
    "model",
    "effort",
    "allowed_tools",
    "disallowed_tools",
    "permission_mode",
    "add_dirs",
    "append_system_prompt",
    "max_budget_usd",
    "timeout_minutes",
)
BEHAVIOUR_KEYS = ("enabled", "notify")
TASK_ONLY_KEYS = ("name", "prompt", "schedule", "description")

DEFAULTS_KEYS = OPTION_KEYS + BEHAVIOUR_KEYS
ASK_KEYS = OPTION_KEYS
TASK_KEYS = TASK_ONLY_KEYS + OPTION_KEYS + BEHAVIOUR_KEYS

DEFAULTS_LABEL = "tasks.toml: [defaults]: "
ASK_LABEL = "tasks.toml: [ask]: "


def _fail(label: str, message: str) -> ConfigError:
    """Build a ConfigError whose text starts with the context ``label``."""
    return ConfigError(f"{label}{message}")


def _check_keys(table: Mapping[str, Any], allowed: tuple[str, ...], label: str) -> None:
    """Reject keys outside ``allowed`` so typos are reported instead of ignored.

    Raises:
        ConfigError: Naming the first unknown key and listing the allowed ones.
    """
    for key in table:
        if key not in allowed:
            raise _fail(label, f"unknown key '{key}'. Allowed keys: {', '.join(allowed)}.")


def _as_table(value: Any, label: str, what: str) -> Mapping[str, Any]:
    """Return ``value`` if it is a TOML table, else raise ConfigError mentioning ``what``."""
    if not isinstance(value, dict):
        raise _fail(label, f"{what} must be a table.")
    return value


def _parse_text(
    value: Any,
    label: str,
    key: str,
    pattern: re.Pattern[str] | None = None,
    expected: str = "",
) -> str:
    """Validate a non-empty string, optionally against ``pattern``; return it stripped.

    ``expected`` describes the accepted format and is appended to the error for pattern mismatches.
    """
    if not isinstance(value, str) or not value.strip():
        raise _fail(label, f"'{key}' must be a non-empty string.")
    text = value.strip()
    if pattern is not None and not pattern.match(text):
        raise _fail(label, f"'{key}' has an invalid value '{text}'. Expected {expected}.")
    return text


def _parse_choice(value: Any, label: str, key: str, choices: tuple[str, ...]) -> str:
    """Validate that ``value`` is one of ``choices``."""
    if not isinstance(value, str) or value not in choices:
        raise _fail(label, f"'{key}' must be one of {', '.join(choices)}.")
    return value


def _parse_bool(value: Any, label: str, key: str) -> bool:
    """Validate a TOML boolean (``true`` or ``false``, no quotes)."""
    if not isinstance(value, bool):
        raise _fail(label, f"'{key}' must be true or false (without quotes).")
    return value


def _parse_directory(value: Any, label: str, key: str, base_dir: Path) -> Path:
    """Expand ``~``, resolve against ``base_dir`` and require an existing directory."""
    raw = _parse_text(value, label, key)
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    path = path.resolve()
    if not path.is_dir():
        raise _fail(label, f"'{key}' must be an existing folder, but {path} is not one.")
    return path


def _parse_tool_list(value: Any, label: str, key: str) -> tuple[str, ...]:
    """Validate an array of tool rules such as ``["Read", "Bash(git log *)"]``."""
    if not isinstance(value, list):
        raise _fail(label, f"'{key}' must be an array of tool names, e.g. [\"Read\", \"Bash(git log *)\"].")
    return tuple(_parse_text(item, label, key, TOOL_RULE_RE, TOOL_RULE_HINT) for item in value)


def _parse_directory_list(value: Any, label: str, key: str, base_dir: Path) -> tuple[Path, ...]:
    """Validate an array of existing folders."""
    if not isinstance(value, list):
        raise _fail(label, f"'{key}' must be an array of folder paths.")
    return tuple(_parse_directory(item, label, key, base_dir) for item in value)


def _parse_budget(value: Any, label: str, key: str) -> float:
    """Validate a positive, finite number (booleans are not numbers here)."""
    is_number = isinstance(value, (int, float)) and not isinstance(value, bool)
    if not is_number or not math.isfinite(value) or value <= 0:
        raise _fail(label, f"'{key}' must be a positive number (US dollars), e.g. 0.50.")
    return float(value)


def _parse_timeout_seconds(value: Any, label: str, key: str) -> int:
    """Validate whole minutes in 1..MAX_TIMEOUT_MINUTES and convert to seconds."""
    is_int = isinstance(value, int) and not isinstance(value, bool)
    if not is_int or not 1 <= value <= MAX_TIMEOUT_MINUTES:
        raise _fail(label, f"'{key}' must be a whole number of minutes from 1 to {MAX_TIMEOUT_MINUTES}.")
    return value * SECONDS_PER_MINUTE


def _optional(raw: Mapping[str, Any], key: str, parse: Callable[[Any], Any], default: Any = None) -> Any:
    """Return ``parse(raw[key])`` when the key is present, else ``default``."""
    return parse(raw[key]) if key in raw else default


def _build_options(
    raw: Mapping[str, Any], label: str, base_dir: Path, default_timeout_minutes: int
) -> RunOptions:
    """Validate the option keys in ``raw`` (already merged) and build RunOptions.

    Args:
        raw: Merged option values from TOML; keys outside OPTION_KEYS are ignored.
        label: Error prefix naming the task (or [defaults] / [ask]).
        base_dir: Folder against which relative paths resolve; also the default workdir.
        default_timeout_minutes: Timeout used when ``timeout_minutes`` is absent.

    Raises:
        ConfigError: On the first invalid value.
    """
    workdir = _optional(
        raw, "workdir", lambda v: _parse_directory(v, label, "workdir", base_dir), base_dir.resolve()
    )
    timeout_seconds = _optional(
        raw,
        "timeout_minutes",
        lambda v: _parse_timeout_seconds(v, label, "timeout_minutes"),
        default_timeout_minutes * SECONDS_PER_MINUTE,
    )
    return RunOptions(
        workdir=workdir,
        model=_optional(raw, "model", lambda v: _parse_text(v, label, "model", MODEL_RE, MODEL_HINT)),
        effort=_optional(raw, "effort", lambda v: _parse_choice(v, label, "effort", EFFORT_LEVELS)),
        allowed_tools=_optional(
            raw, "allowed_tools", lambda v: _parse_tool_list(v, label, "allowed_tools"), DEFAULT_ALLOWED_TOOLS
        ),
        disallowed_tools=_optional(
            raw, "disallowed_tools", lambda v: _parse_tool_list(v, label, "disallowed_tools"), ()
        ),
        permission_mode=_optional(
            raw,
            "permission_mode",
            lambda v: _parse_choice(v, label, "permission_mode", PERMISSION_MODES),
            DEFAULT_PERMISSION_MODE,
        ),
        add_dirs=_optional(
            raw, "add_dirs", lambda v: _parse_directory_list(v, label, "add_dirs", base_dir), ()
        ),
        append_system_prompt=_optional(
            raw, "append_system_prompt", lambda v: _parse_text(v, label, "append_system_prompt")
        ),
        max_budget_usd=_optional(raw, "max_budget_usd", lambda v: _parse_budget(v, label, "max_budget_usd")),
        timeout_seconds=timeout_seconds,
    )


def _parse_schedules(value: Any, label: str, now: datetime) -> tuple[Schedule, ...]:
    """Parse ``schedule`` (a string or an array of strings) and prove each one can fire.

    Args:
        value: The TOML value; ``None`` or an empty array means manual only.
        label: Error prefix naming the task.
        now: Aware moment used to check that the schedule fires at least once.

    Raises:
        ScheduleError: If an expression is invalid or never fires.
        ConfigError: If the value has the wrong type.
    """
    if value is None:
        return ()
    items = [value] if isinstance(value, str) else value
    if not isinstance(items, list) or not all(isinstance(item, str) for item in items):
        raise _fail(label, "'schedule' must be a string or an array of strings, e.g. \"daily 07:30\".")

    schedules: list[Schedule] = []
    for item in items:
        try:
            schedule = parse_schedule(item)
            # parse_schedule accepts impossible dates; next_after is what proves it can fire.
            schedule.next_after(now)
        except ScheduleError as err:
            raise ScheduleError(f"{label}'schedule': {err}") from err
        schedules.append(schedule)
    return tuple(schedules)


def _task_label(entry: Mapping[str, Any], index: int) -> str:
    """Return the error prefix for a task: by name when it is a usable string, else by position."""
    name = entry.get("name")
    if isinstance(name, str) and name.strip():
        return f"tasks.toml: task '{name.strip()}': "
    return f"tasks.toml: task #{index}: "


def _parse_task(
    entry: Mapping[str, Any],
    index: int,
    defaults: Mapping[str, Any],
    base_dir: Path,
    seen: set[str],
    now: datetime,
) -> Task:
    """Validate one ``[[task]]`` table merged over ``[defaults]`` and build a Task.

    Args:
        entry: The task's TOML table.
        index: 1-based position in the file (used when the name is missing).
        defaults: The raw ``[defaults]`` table.
        base_dir: Folder for relative paths.
        seen: Names already used; this task's name is added on success.
        now: Aware moment for the schedule check.

    Raises:
        ConfigError: On any invalid or missing field.
    """
    label = _task_label(entry, index)
    _check_keys(entry, TASK_KEYS, label)

    if "name" not in entry:
        raise _fail(label, "'name' is required.")
    name = _parse_text(entry["name"], label, "name", TASK_NAME_RE, NAME_HINT)
    if "prompt" not in entry:
        raise _fail(label, "'prompt' is required.")
    if name == ASK_TASK_NAME:
        raise _fail(label, f"'name' cannot be '{ASK_TASK_NAME}': that name is reserved for chat answers.")
    if name in seen:
        raise _fail(label, "'name' is already used by an earlier task. Names must be unique.")

    prompt = _parse_text(entry["prompt"], label, "prompt")
    merged = {**defaults, **entry}
    notify = _optional(merged, "notify", lambda v: _parse_choice(v, label, "notify", NOTIFY_POLICIES), NOTIFY_ALWAYS)
    enabled = _optional(merged, "enabled", lambda v: _parse_bool(v, label, "enabled"), True)
    description = _optional(entry, "description", lambda v: _parse_text(v, label, "description"), "")

    task = Task(
        name=name,
        prompt=prompt,
        schedules=_parse_schedules(entry.get("schedule"), label, now),
        options=_build_options(merged, label, base_dir, DEFAULT_TIMEOUT_MINUTES),
        description=description,
        enabled=enabled,
        notify=notify,
    )
    seen.add(name)
    return task


def parse_task_config(data: Mapping[str, Any], base_dir: Path, *, source: Path | None = None) -> TaskConfig:
    """Validate parsed TOML data and build a TaskConfig.

    Args:
        data: The dict produced by ``tomllib`` (top-level keys defaults, ask, task).
        base_dir: Folder against which relative workdir and add_dirs resolve;
            also the default workdir.
        source: The file the data came from, stored on the result.

    Returns:
        The validated config with tasks in file order.

    Raises:
        ConfigError: On the first problem found; the message names the task
            and the field. ScheduleError (a ConfigError) for bad schedules.

    Example:
        >>> config = parse_task_config(
        ...     {"task": [{"name": "news", "prompt": "Summarise.", "schedule": "daily 07:30"}]},
        ...     Path("."),
        ... )
        >>> list(config.tasks)
        ['news']
    """
    _check_keys(data, TOP_LEVEL_KEYS, "tasks.toml: ")
    defaults = _as_table(data.get("defaults", {}), DEFAULTS_LABEL, "[defaults]")
    ask = _as_table(data.get("ask", {}), ASK_LABEL, "[ask]")
    _check_keys(defaults, DEFAULTS_KEYS, DEFAULTS_LABEL)
    _check_keys(ask, ASK_KEYS, ASK_LABEL)

    # Validate [defaults] on its own so a bad default is blamed on [defaults], not on every task.
    _build_options(defaults, DEFAULTS_LABEL, base_dir, DEFAULT_TIMEOUT_MINUTES)
    _optional(defaults, "notify", lambda v: _parse_choice(v, DEFAULTS_LABEL, "notify", NOTIFY_POLICIES))
    _optional(defaults, "enabled", lambda v: _parse_bool(v, DEFAULTS_LABEL, "enabled"))

    # Ask options inherit the option keys from [defaults] but keep their own timeout default.
    default_options = {key: value for key, value in defaults.items() if key in OPTION_KEYS}
    ask_options = _build_options({**default_options, **ask}, ASK_LABEL, base_dir, DEFAULT_ASK_TIMEOUT_MINUTES)

    raw_tasks = data.get("task", [])
    if not isinstance(raw_tasks, list):
        raise _fail("tasks.toml: ", "'task' must be an array of tables written as [[task]].")

    now = datetime.now(timezone.utc)
    tasks: dict[str, Task] = {}
    seen: set[str] = set()
    for index, entry in enumerate(raw_tasks, start=1):
        table = _as_table(entry, f"tasks.toml: task #{index}: ", "each [[task]] entry")
        task = _parse_task(table, index, defaults, base_dir, seen, now)
        tasks[task.name] = task
    return TaskConfig(tasks=tasks, ask_options=ask_options, source=source)


def load_task_config(path: Path) -> TaskConfig:
    """Read, parse and validate a tasks.toml file.

    Args:
        path: Location of tasks.toml; relative paths in it resolve against its folder.

    Returns:
        The validated TaskConfig with ``source`` set to ``path``.

    Raises:
        ConfigError: If the file is missing, unreadable, not valid TOML (the
            message carries tomllib's line and column) or fails validation.

    Example:
        >>> config = load_task_config(Path("tasks.toml"))
    """
    if not path.is_file():
        raise ConfigError(f"tasks.toml not found at {path}. Run `autopilot init` to create one.")
    try:
        text = path.read_text(encoding="utf-8")
        data = tomllib.loads(text)
    except tomllib.TOMLDecodeError as err:
        # tomllib's message already ends with "(at line L, column C)".
        raise ConfigError(f"tasks.toml: invalid TOML in {path}: {err}") from err
    except (OSError, UnicodeDecodeError) as err:
        raise ConfigError(f"tasks.toml: cannot read {path}: {err}") from err
    return parse_task_config(data, path.parent, source=path)
