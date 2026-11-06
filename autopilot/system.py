"""Runs operating-system commands and reports their outcome without raising.

This is the one place that starts external programs for system-level work
(systemctl, loginctl). Callers receive a plain ``(exit_code, stdout, stderr)``
tuple, so they can decide what a failure means and tests can replace the whole
function with a fake that records calls.

Used by: service.py (systemd user unit install, uninstall and linger check),
    and doctor.py for its service checks.
Depends on: the standard library only.

Invariants:
    * Commands are argument lists, never shell strings, so input is never interpreted by a shell.
    * ``run_command`` never raises. Missing programs, timeouts and OS errors come back
      as exit codes with a message on stderr.
"""

from __future__ import annotations

import subprocess
from collections.abc import Callable, Sequence

# Exit code used by shells for "command not found"; reused for a missing program.
EXIT_NOT_FOUND = 127
# Exit code used by shells for "found but not executable"; reused for other OS errors.
EXIT_NOT_EXECUTABLE = 126
# Exit code GNU `timeout` uses when it kills a command; reused for timeouts.
EXIT_TIMED_OUT = 124
# Default wall-clock limit for one command, in seconds.
DEFAULT_COMMAND_TIMEOUT_SECONDS = 20.0

# A callable that runs a command and returns (exit_code, stdout, stderr).
# Tests pass a fake with this signature; production code passes run_command.
CommandRunner = Callable[[Sequence[str], float], tuple[int, str, str]]


def run_command(
    args: Sequence[str],
    timeout: float = DEFAULT_COMMAND_TIMEOUT_SECONDS,
) -> tuple[int, str, str]:
    """Run one program with an argument list and capture its output.

    Args:
        args: The program followed by its arguments, e.g. ``["systemctl", "--user", "is-enabled"]``.
            The first element is looked up on PATH; nothing goes through a shell.
        timeout: Wall-clock limit in seconds. Must be positive. When it expires the
            child process is killed and the call returns exit code 124.

    Returns:
        A tuple ``(exit_code, stdout, stderr)``. Output is decoded as UTF-8 with
        replacement, so binary noise never raises. Failures before the program
        runs use the codes 127 (not found) and 126 (other OS error) with the
        reason in stderr; a timeout uses 124 with "timed out" in stderr.

    Example:
        >>> run_command(["systemctl", "--user", "is-enabled", "x.service"])  # doctest: +SKIP
        (0, "enabled\\n", "")
    """
    try:
        completed = subprocess.run(
            list(args),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as err:
        # On a timeout the partial output arrives as bytes even in text mode, so decode it here.
        partial_stdout = _as_text(err.stdout)
        return EXIT_TIMED_OUT, partial_stdout, f"Command timed out after {timeout:g} s."
    except FileNotFoundError as err:
        return EXIT_NOT_FOUND, "", f"Program not found: {args[0]} ({err.strerror})."
    except OSError as err:
        return EXIT_NOT_EXECUTABLE, "", f"Could not run {args[0]}: {err}."
    return completed.returncode, completed.stdout, completed.stderr


def _as_text(value: bytes | str | None) -> str:
    """Convert captured output from a timed-out process to text.

    Args:
        value: Bytes, text, or None when the process produced no output before the timeout.

    Returns:
        The output as a str. Undecodable bytes become U+FFFD.
    """
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value
