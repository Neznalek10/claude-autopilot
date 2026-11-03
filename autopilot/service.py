"""Installs, removes and inspects the systemd user service that keeps autopilot running.

The service starts `python -m autopilot start` at boot, restarts it on failure
and keeps the PATH from install time, so the Claude CLI found by the installer
is also found by the service. All systemctl and loginctl calls go through an
injected CommandRunner, so tests never touch the real systemd.

Used by: cli.py (install-service, uninstall-service) and doctor.py (linger check).
Depends on: errors.py (AutopilotError) and system.py (CommandRunner type).

Invariants:
    * Values written to the unit file are escaped: ``%`` becomes ``%%`` (systemd
      would otherwise expand it as a specifier), and quotes and backslashes are
      escaped inside quoted values.
    * A unit file is replaced atomically, so a crash never leaves a half-written unit.
"""

from __future__ import annotations

import contextlib
import getpass
import os
from pathlib import Path

from autopilot.errors import AutopilotError
from autopilot.system import CommandRunner

# Name of the systemd user unit this package installs.
UNIT_NAME = "autopilot.service"
# Seconds systemctl may take to reload or enable. Starting is quick because the unit is Type=simple.
SYSTEMCTL_TIMEOUT_SECONDS = 30.0
# Seconds loginctl may take to answer a property query.
LOGINCTL_TIMEOUT_SECONDS = 10.0
# Seconds systemd waits before restarting a failed service.
RESTART_DELAY_SECONDS = 15
# Window and attempts for systemd's start rate limit, so a crash loop stops instead of spinning.
START_LIMIT_INTERVAL_SECONDS = 300
START_LIMIT_BURST = 5
# systemd's stderr wording when `disable --now` finds nothing to stop; both mean "already gone".
TOLERATED_DISABLE_FAILURES = ("not loaded", "does not exist")
# Characters that force a word in ExecStart to be quoted so systemd splits it correctly.
EXEC_WORD_SPECIAL_CHARACTERS = frozenset('"\\')


def default_unit_dir() -> Path:
    """Return the folder where systemd looks for user units.

    Returns:
        ``~/.config/systemd/user`` for the current user. The path is not checked for existence.

    Example:
        >>> default_unit_dir().name
        'user'
    """
    return Path.home() / ".config" / "systemd" / "user"


def render_unit(*, python: Path, project_home: Path, path_env: str) -> str:
    """Build the text of the systemd user unit file.

    Args:
        python: Interpreter that runs ``-m autopilot start``; use the venv python.
        project_home: Folder the service works in; also set as AUTOPILOT_HOME.
        path_env: The PATH value to keep in the service environment, so tools
            such as the Claude CLI resolve the same way they did at install time.

    Returns:
        The complete unit file, ending with a newline.

    Raises:
        ValueError: If any value contains a newline, which would break the unit file.

    Example:
        >>> text = render_unit(
        ...     python=Path("/srv/venv/bin/python"),
        ...     project_home=Path("/srv/ap"),
        ...     path_env="/usr/bin",
        ... )
        >>> "ExecStart=/srv/venv/bin/python -m autopilot start" in text
        True
    """
    home_text = str(project_home)
    python_text = str(python)
    _reject_newlines(home_text, python_text, path_env)
    return "\n".join(
        [
            "[Unit]",
            "Description=Claude Autopilot: scheduled Claude Code tasks",
            "After=network-online.target",
            "Wants=network-online.target",
            f"StartLimitIntervalSec={START_LIMIT_INTERVAL_SECONDS}",
            f"StartLimitBurst={START_LIMIT_BURST}",
            "",
            "[Service]",
            "Type=simple",
            # WorkingDirectory takes the rest of the line literally, so it is not quoted.
            f"WorkingDirectory={_escape_specifiers(home_text)}",
            f"Environment={_quote(f'AUTOPILOT_HOME={home_text}')}",
            f"Environment={_quote(f'PATH={path_env}')}",
            f"Environment={_quote('PYTHONUNBUFFERED=1')}",
            f"ExecStart={_exec_word(python_text)} -m autopilot start",
            "Restart=on-failure",
            f"RestartSec={RESTART_DELAY_SECONDS}",
            "",
            "[Install]",
            "WantedBy=default.target",
            "",
        ]
    )


def install_service(
    *,
    python: Path,
    project_home: Path,
    path_env: str,
    run: CommandRunner,
    unit_dir: Path,
) -> list[str]:
    """Write the unit file, reload systemd and enable the service now.

    Args:
        python: Interpreter for the unit's ExecStart.
        project_home: Working folder and AUTOPILOT_HOME for the service.
        path_env: PATH value to keep in the service environment.
        run: Command runner used for systemctl and loginctl.
        unit_dir: Folder that receives ``autopilot.service``; created when missing.

    Returns:
        Readable step lines describing what was done. When linger is known to be
        off, the last line tells the user how to keep the service running without a login.

    Raises:
        AutopilotError: When the unit file can't be written, or when systemctl fails.
            The message contains systemctl's own error output.

    Side effects:
        Writes ``unit_dir/autopilot.service``, runs ``systemctl --user daemon-reload``
        and ``systemctl --user enable --now autopilot.service``.

    Example:
        >>> install_service(  # doctest: +SKIP
        ...     python=venv_python, project_home=home, path_env=path,
        ...     run=run_command, unit_dir=default_unit_dir(),
        ... )
    """
    content = render_unit(python=python, project_home=project_home, path_env=path_env)
    unit_path = _write_unit_file(unit_dir, content)
    steps = [f"Wrote {unit_path}"]
    _check_systemctl(
        "daemon-reload",
        run(["systemctl", "--user", "daemon-reload"], SYSTEMCTL_TIMEOUT_SECONDS),
    )
    steps.append("Reloaded the systemd user manager.")
    _check_systemctl(
        "enable --now",
        run(["systemctl", "--user", "enable", "--now", UNIT_NAME], SYSTEMCTL_TIMEOUT_SECONDS),
    )
    steps.append(f"Enabled and started {UNIT_NAME}: it starts at boot and restarts on failure.")
    steps.extend(_linger_hint(run))
    return steps


def uninstall_service(*, run: CommandRunner, unit_dir: Path) -> list[str]:
    """Stop and disable the service, remove its unit file and reload systemd.

    Args:
        run: Command runner used for systemctl.
        unit_dir: Folder that holds ``autopilot.service``.

    Returns:
        Readable step lines describing what was done.

    Raises:
        AutopilotError: When systemctl fails for a reason other than "not loaded"
            (or "does not exist"), or when the unit file can't be removed. The unit
            file is left in place in the first case, so the user can retry.

    Side effects:
        Runs ``systemctl --user disable --now autopilot.service``, deletes the unit
        file when present, and runs ``systemctl --user daemon-reload``.

    Example:
        >>> uninstall_service(run=run_command, unit_dir=default_unit_dir())  # doctest: +SKIP
    """
    steps = [_disable_now(run)]
    steps.append(_remove_unit_file(unit_dir))
    _check_systemctl(
        "daemon-reload",
        run(["systemctl", "--user", "daemon-reload"], SYSTEMCTL_TIMEOUT_SECONDS),
    )
    steps.append("Reloaded the systemd user manager.")
    return steps


def linger_enabled(run: CommandRunner, user: str) -> bool | None:
    """Ask systemd whether the user's services keep running after logout.

    Args:
        run: Command runner used for loginctl.
        user: Login name to query, e.g. the value of ``getpass.getuser()``.

    Returns:
        True when linger is on, False when it is off, None when loginctl fails or
        its answer can't be read.

    Example:
        >>> linger_enabled(run_command, "alice")  # doctest: +SKIP
        False
    """
    code, stdout, _ = run(
        ["loginctl", "show-user", user, "--property=Linger"],
        LOGINCTL_TIMEOUT_SECONDS,
    )
    if code != 0:
        return None
    return _parse_linger(stdout)


def _reject_newlines(*values: str) -> None:
    """Refuse values that would split a unit file line.

    Args:
        *values: Text that will be written into the unit file.

    Raises:
        ValueError: If any value contains a line break.
    """
    for value in values:
        if "\n" in value or "\r" in value:
            raise ValueError(f"Unit file values must be single-line, got {value!r}")


def _escape_specifiers(text: str) -> str:
    """Escape systemd specifiers so ``%`` is taken literally.

    Args:
        text: Raw value, e.g. a path that may contain ``%``.

    Returns:
        The value with every ``%`` doubled.
    """
    return text.replace("%", "%%")


def _quote(text: str) -> str:
    """Wrap a value in double quotes for a unit file assignment.

    Inside the quotes systemd reads backslash escapes, so backslashes and double
    quotes are escaped first. ``%`` is escaped as well, because specifiers still apply.

    Args:
        text: The raw assignment, e.g. ``PATH=/usr/bin:/opt/my bin``.

    Returns:
        The quoted, escaped text, e.g. ``"PATH=/usr/bin:/opt/my bin"``.
    """
    escaped = _escape_specifiers(text).replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _exec_word(text: str) -> str:
    """Format one word of ExecStart so systemd passes it as a single argument.

    Words without spaces or special characters stay bare, so the unit reads
    naturally. Anything else is quoted with the same escaping as assignments.
    ExecStart also expands ``$NAME`` and ``${NAME}`` from the environment, so every
    ``$`` is doubled; ``$$`` is systemd's literal dollar sign.

    Args:
        text: One command word, e.g. the interpreter path.

    Returns:
        The word as it should appear in ExecStart.
    """
    needs_quotes = any(char.isspace() or char in EXEC_WORD_SPECIAL_CHARACTERS for char in text)
    literal_dollars = text.replace("$", "$$")
    if needs_quotes:
        return _quote(literal_dollars)
    return _escape_specifiers(literal_dollars)


def _write_unit_file(unit_dir: Path, content: str) -> Path:
    """Write the unit file atomically: a temp file in the same folder, then rename.

    Args:
        unit_dir: Folder for the unit; created when missing.
        content: The complete unit file text.

    Returns:
        The path of the written unit file.

    Raises:
        AutopilotError: When the folder can't be created or the file can't be written.
    """
    target = unit_dir / UNIT_NAME
    temp = unit_dir / f"{UNIT_NAME}.tmp"
    try:
        unit_dir.mkdir(parents=True, exist_ok=True)
        temp.write_text(content, encoding="utf-8")
        os.replace(temp, target)
    except OSError as err:
        # Remove a half-written temp file; a failure here must not hide the original error.
        with contextlib.suppress(OSError):
            temp.unlink(missing_ok=True)
        raise AutopilotError(
            f"Could not write {target}: {err}. Check that {unit_dir} is writable."
        ) from err
    return target


def _check_systemctl(action: str, result: tuple[int, str, str]) -> None:
    """Raise when a systemctl call failed, quoting its own output.

    Args:
        action: The systemctl action for the message, e.g. "enable --now".
        result: The ``(exit_code, stdout, stderr)`` tuple from the command runner.

    Raises:
        AutopilotError: When the exit code is not 0.
    """
    code, stdout, stderr = result
    if code != 0:
        raise _systemctl_error(action, code, stdout, stderr)


def _systemctl_error(action: str, code: int, stdout: str, stderr: str) -> AutopilotError:
    """Build the error for a failed systemctl call.

    Args:
        action: The systemctl action for the message, e.g. "enable --now".
        code: The non-zero exit code.
        stdout: Standard output, used when stderr is empty.
        stderr: Standard error, which systemctl normally uses for its explanation.

    Returns:
        An AutopilotError naming the action, the exit code, systemctl's own words
        and a command to check the service.
    """
    detail = (stderr or stdout).strip() or "no error output"
    return AutopilotError(
        f"systemctl --user {action} failed (exit {code}): {detail}. "
        f"Check the service with `systemctl --user status {UNIT_NAME}`."
    )


def _disable_now(run: CommandRunner) -> str:
    """Stop and disable the service, accepting the case where systemd doesn't know it.

    Args:
        run: Command runner used for systemctl.

    Returns:
        A step line: stopped and disabled, or not loaded so nothing to stop.

    Raises:
        AutopilotError: When systemctl fails for any other reason.
    """
    code, stdout, stderr = run(
        ["systemctl", "--user", "disable", "--now", UNIT_NAME],
        SYSTEMCTL_TIMEOUT_SECONDS,
    )
    if code == 0:
        return f"Stopped and disabled {UNIT_NAME}."
    # systemd words "nothing to disable" differently across versions; match both phrasings.
    message = (stderr or stdout).lower()
    if any(phrase in message for phrase in TOLERATED_DISABLE_FAILURES):
        return f"{UNIT_NAME} was not loaded, so there was nothing to stop."
    raise _systemctl_error("disable --now", code, stdout, stderr)


def _remove_unit_file(unit_dir: Path) -> str:
    """Delete the unit file if it exists.

    Args:
        unit_dir: Folder that holds the unit file.

    Returns:
        A step line saying whether a file was removed.

    Raises:
        AutopilotError: When the file exists but can't be deleted.
    """
    unit_path = unit_dir / UNIT_NAME
    if not unit_path.exists():
        return f"No unit file at {unit_path}; nothing to remove."
    try:
        unit_path.unlink()
    except OSError as err:
        raise AutopilotError(f"Could not remove {unit_path}: {err}.") from err
    return f"Removed {unit_path}."


def _linger_hint(run: CommandRunner) -> list[str]:
    """Explain how to keep the service running without a login, when linger is off.

    Args:
        run: Command runner used for loginctl.

    Returns:
        An empty list when linger is on or unknown; otherwise one hint line.
    """
    user = getpass.getuser()
    if linger_enabled(run, user) is not False:
        return []
    return [
        "Linger is off, so the service stops when you log out. "
        f"To keep it running without a login, run: sudo loginctl enable-linger {user}"
    ]


def _parse_linger(output: str) -> bool | None:
    """Read the ``Linger=`` line from ``loginctl show-user`` output.

    Args:
        output: The stdout of ``loginctl show-user <user> --property=Linger``,
            e.g. ``"Linger=yes\\n"``.

    Returns:
        True for ``yes``, False for ``no``, None when no recognizable line is present.
    """
    for line in output.splitlines():
        key, separator, value = line.partition("=")
        if separator and key.strip() == "Linger":
            answer = value.strip()
            if answer == "yes":
                return True
            if answer == "no":
                return False
    return None
