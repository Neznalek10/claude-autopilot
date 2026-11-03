"""Settings for Claude Autopilot: paths, timezone and Telegram secrets.

Responsible for turning the process environment and the project's ``.env``
file into one immutable ``Settings`` object, and for writing single keys back
into ``.env`` (used by ``autopilot telegram-setup``).

Used by: app.py, doctor.py and cli.py. Uses: errors.py and python-dotenv.

Invariants:
    * ``os.environ`` is never modified: ``.env`` is read with ``dotenv_values``.
    * The real environment wins over ``.env``; empty strings count as unset.
    * ``AUTOPILOT_HOME`` is read from the real environment only, because it
      decides where ``.env`` lives.
    * Error messages name the variable and the expected format, and never
      echo the Telegram bot token.
"""

from __future__ import annotations

import logging
import os
import re
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from dotenv import dotenv_values

from autopilot.errors import ConfigError

logger = logging.getLogger(__name__)

ENV_FILE_NAME = ".env"
DEFAULT_TASKS_FILE_NAME = "tasks.toml"
DEFAULT_DATA_DIR_NAME = "data"
DB_FILE_NAME = "autopilot.db"
LOG_FILE_NAME = "autopilot.log"

MIN_PARALLEL = 1
MAX_PARALLEL_LIMIT = 4
LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR")
DEFAULT_LOG_LEVEL = "INFO"

ENV_FILE_MODE = 0o600

# Marker in the /etc/localtime symlink target, e.g. /usr/share/zoneinfo/Europe/Prague.
ZONEINFO_MARKER = "zoneinfo/"

# Names accepted by set_env_value: upper-case letters, digits and underscores, starting with a letter.
ENV_KEY_RE = re.compile(r"^[A-Z][A-Z0-9_]*$")
# Values made only of these characters are written unquoted; anything else is single-quoted.
SAFE_ENV_VALUE_RE = re.compile(r"^[A-Za-z0-9_.,:/@+=-]*$")
# Matches an existing "KEY=..." line (optionally "export KEY=...") so it can be replaced.
ENV_LINE_RE = re.compile(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=")


@dataclass(frozen=True)
class Settings:
    """Everything the service needs to know about its environment.

    Attributes:
        home: Project folder; relative paths in the environment resolve against it.
        tasks_file: Path of tasks.toml.
        data_dir: Folder for the database and the log file.
        db_path: SQLite database (``data_dir / "autopilot.db"``).
        log_file: Log file (``data_dir / "autopilot.log"``).
        env_file: The ``.env`` file (``home / ".env"``).
        timezone: Zone used to show times and to evaluate schedules.
        claude_bin: Configured Claude CLI path (``CLAUDE_BIN``), if any.
        telegram_token: Bot token, if configured. Never log it.
        telegram_chat_id: The one chat the bot answers, if configured.
        max_parallel: Number of runs executed at once (1-4).
        log_level: One of DEBUG, INFO, WARNING, ERROR.
    """

    home: Path
    tasks_file: Path
    data_dir: Path
    db_path: Path
    log_file: Path
    env_file: Path
    timezone: ZoneInfo
    claude_bin: str | None
    telegram_token: str | None
    telegram_chat_id: int | None
    max_parallel: int
    log_level: str

    @property
    def telegram_enabled(self) -> bool:
        """True when both the bot token and the chat id are set."""
        return self.telegram_token is not None and self.telegram_chat_id is not None


def project_root() -> Path:
    """Return the folder that contains the ``autopilot`` package.

    Returns:
        Absolute path of the project folder.
    """
    return Path(__file__).resolve().parent.parent


def _non_empty(values: Mapping[str, str | None]) -> dict[str, str]:
    """Return a copy of ``values`` without unset or blank entries (they count as unset)."""
    return {key: value for key, value in values.items() if value is not None and value.strip() != ""}


def _resolve_path(raw: str, home: Path, variable: str) -> Path:
    """Expand ``~`` and resolve a relative path against ``home``.

    Raises:
        ConfigError: Naming ``variable`` if a ``~user`` prefix can't be expanded.
    """
    try:
        path = Path(raw.strip()).expanduser()
    except RuntimeError:
        # expanduser raises RuntimeError for "~user" when that user does not exist.
        raise ConfigError(f"{variable} uses '{raw}', but that home folder cannot be found.") from None
    if not path.is_absolute():
        path = home / path
    return path.resolve()


def _parse_max_parallel(raw: str | None) -> int:
    """Parse AUTOPILOT_MAX_PARALLEL (integer 1-4, default 1).

    Raises:
        ConfigError: If the value is not an integer in range.
    """
    if raw is None:
        return MIN_PARALLEL
    try:
        value = int(raw.strip())
    except ValueError:
        value = 0
    if not MIN_PARALLEL <= value <= MAX_PARALLEL_LIMIT:
        raise ConfigError(
            f"AUTOPILOT_MAX_PARALLEL must be a whole number from {MIN_PARALLEL} to "
            f"{MAX_PARALLEL_LIMIT}, got '{raw}'."
        )
    return value


def _parse_log_level(raw: str | None) -> str:
    """Parse AUTOPILOT_LOG_LEVEL (case-insensitive, default INFO).

    Raises:
        ConfigError: If the value is not one of LOG_LEVELS.
    """
    if raw is None:
        return DEFAULT_LOG_LEVEL
    level = raw.strip().upper()
    if level not in LOG_LEVELS:
        raise ConfigError(f"AUTOPILOT_LOG_LEVEL must be one of {', '.join(LOG_LEVELS)}, got '{raw}'.")
    return level


def _parse_chat_id(raw: str | None) -> int | None:
    """Parse TELEGRAM_CHAT_ID, the owner's positive Telegram user id.

    Raises:
        ConfigError: If the value is not a positive integer (group chats have
            negative ids and are not supported).
    """
    if raw is None:
        return None
    try:
        chat_id = int(raw.strip())
    except ValueError:
        chat_id = 0
    if chat_id <= 0:
        raise ConfigError(
            f"TELEGRAM_CHAT_ID must be your own Telegram user id, a positive whole number such as "
            f"123456789, got '{raw}'. Group chats are not supported. "
            "Run `autopilot telegram-setup` to set it."
        )
    return chat_id


def load_settings(
    env: Mapping[str, str] | None = None,
    *,
    home: Path | None = None,
    read_env_file: bool = True,
) -> Settings:
    """Build Settings from the environment and the ``.env`` file.

    Args:
        env: Environment mapping; defaults to ``os.environ``. It is only read.
        home: Project folder. Defaults to ``AUTOPILOT_HOME`` from ``env``,
            else ``project_root()``.
        read_env_file: Read ``<home>/.env`` for values missing from ``env``.

    Returns:
        The resolved Settings.

    Raises:
        ConfigError: If a variable has a bad value (the message names it and
            the expected format).

    Example:
        >>> settings = load_settings({}, home=Path("/srv/autopilot"), read_env_file=False)
        >>> settings.db_path
        PosixPath('/srv/autopilot/data/autopilot.db')
    """
    real_env = _non_empty(os.environ if env is None else env)
    if home is None:
        raw_home = real_env.get("AUTOPILOT_HOME")
        home = _resolve_path(raw_home, Path.cwd(), "AUTOPILOT_HOME") if raw_home else project_root()
    else:
        home = home.expanduser().resolve()

    # The real environment wins: .env only fills in what is missing.
    merged = dict(real_env)
    env_file = home / ENV_FILE_NAME
    if read_env_file and env_file.is_file():
        for key, value in _non_empty(dotenv_values(env_file)).items():
            merged.setdefault(key, value)

    tasks_file = _resolve_path(merged.get("AUTOPILOT_TASKS_FILE", DEFAULT_TASKS_FILE_NAME), home, "AUTOPILOT_TASKS_FILE")
    data_dir = _resolve_path(merged.get("AUTOPILOT_DATA_DIR", DEFAULT_DATA_DIR_NAME), home, "AUTOPILOT_DATA_DIR")
    return Settings(
        home=home,
        tasks_file=tasks_file,
        data_dir=data_dir,
        db_path=data_dir / DB_FILE_NAME,
        log_file=data_dir / LOG_FILE_NAME,
        env_file=env_file,
        timezone=resolve_timezone(merged.get("AUTOPILOT_TIMEZONE"), env_tz=real_env.get("TZ")),
        claude_bin=merged.get("CLAUDE_BIN"),
        telegram_token=merged.get("TELEGRAM_BOT_TOKEN"),
        telegram_chat_id=_parse_chat_id(merged.get("TELEGRAM_CHAT_ID")),
        max_parallel=_parse_max_parallel(merged.get("AUTOPILOT_MAX_PARALLEL")),
        log_level=_parse_log_level(merged.get("AUTOPILOT_LOG_LEVEL")),
    )


def _zone_or_none(name: str) -> ZoneInfo | None:
    """Return the ZoneInfo called ``name`` or None when it is not a valid IANA zone."""
    try:
        return ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError, OSError):
        # OSError covers names such as "Europe" that point at a folder, not a zone.
        return None


def _zone_from_localtime(localtime: Path) -> ZoneInfo | None:
    """Read the zone from the ``/etc/localtime`` symlink target, if it is one."""
    if not localtime.is_symlink():
        return None
    target = os.readlink(localtime)
    _, marker, name = target.partition(ZONEINFO_MARKER)
    return _zone_or_none(name) if marker and name else None


def _zone_from_file(timezone_file: Path) -> ZoneInfo | None:
    """Read the zone name from a ``/etc/timezone`` style file, if present and valid."""
    try:
        name = timezone_file.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeDecodeError):
        return None
    return _zone_or_none(name) if name else None


def resolve_timezone(
    name: str | None,
    *,
    env_tz: str | None = None,
    localtime: Path = Path("/etc/localtime"),
    timezone_file: Path = Path("/etc/timezone"),
) -> ZoneInfo:
    """Pick the time zone: explicit name, then ``TZ``, then the system setting.

    Args:
        name: Explicit IANA name (``AUTOPILOT_TIMEZONE``), or None.
        env_tz: Value of the ``TZ`` variable; used only when it is a valid zone.
        localtime: Path of the system's localtime symlink.
        timezone_file: Path of the plain-text system timezone file.

    Returns:
        The zone. Falls back to UTC (with a logged warning) when nothing works.

    Raises:
        ConfigError: If ``name`` is given but is not a known IANA zone.
    """
    if name:
        zone = _zone_or_none(name.strip())
        if zone is None:
            raise ConfigError(
                f"AUTOPILOT_TIMEZONE '{name}' is not a known time zone. "
                "Use an IANA name such as Europe/Prague or America/New_York."
            )
        return zone

    # TZ may carry a leading colon (":Europe/Prague"), which POSIX allows.
    candidates = (
        _zone_or_none(env_tz.strip().lstrip(":")) if env_tz else None,
        _zone_from_localtime(localtime),
        _zone_from_file(timezone_file),
    )
    for zone in candidates:
        if zone is not None:
            return zone
    logger.warning("Could not detect the system time zone, using UTC. Set AUTOPILOT_TIMEZONE in .env.")
    return ZoneInfo("UTC")


def _format_env_value(value: str) -> str:
    """Render a value for a .env line, single-quoting it when it has special characters."""
    if SAFE_ENV_VALUE_RE.match(value):
        return value
    escaped = value.replace("\\", "\\\\").replace("'", "\\'")
    return f"'{escaped}'"


def set_env_value(env_file: Path, key: str, value: str) -> None:
    """Create or update ``KEY=value`` in a .env file, keeping all other lines.

    The file is replaced atomically (temp file in the same folder, then
    rename) and ends up with mode 0o600 because it can hold secrets.

    Args:
        env_file: The .env path. Created when missing.
        key: Variable name, matching ``^[A-Z][A-Z0-9_]*$``.
        value: New value, a single line.

    Raises:
        ValueError: If the key is malformed or the value has a newline or any other
            non-printable character.
        OSError: If the file cannot be read or written.

    Example:
        >>> set_env_value(Path(".env"), "TELEGRAM_CHAT_ID", "123456789")
    """
    if not ENV_KEY_RE.match(key):
        raise ValueError(f"Invalid environment variable name '{key}': use A-Z, 0-9 and underscores.")
    # str.isprintable is False for newlines, other line separators (U+2028, U+0085) and control codes.
    if not value.isprintable():
        raise ValueError(f"The value for {key} must be one line of printable characters.")

    new_line = f"{key}={_format_env_value(value)}"
    lines = env_file.read_text(encoding="utf-8").split("\n") if env_file.exists() else []

    # Split on "\n" only (not splitlines) so exotic separators can't be mistaken for line ends.
    if lines and lines[-1] == "":
        lines.pop()

    # Replace the first assignment of the key (and drop duplicates); append when absent.
    updated: list[str] = []
    replaced = False
    for line in lines:
        match = ENV_LINE_RE.match(line)
        if match and match.group(1) == key:
            if not replaced:
                updated.append(new_line)
                replaced = True
            continue
        updated.append(line)
    if not replaced:
        updated.append(new_line)

    _atomic_write(env_file, "\n".join(updated) + "\n")


def _atomic_write(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` through a temp file and rename, with mode 0o600."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    temp_path = Path(temp_name)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as temp_file:
            temp_file.write(text)
        temp_path.chmod(ENV_FILE_MODE)
        os.replace(temp_path, path)
    except BaseException:
        # Do not leave a half-written secrets file behind, then let the error propagate.
        temp_path.unlink(missing_ok=True)
        raise
