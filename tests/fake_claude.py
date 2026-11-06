#!/usr/bin/env python3
"""Executable stand-in for the Claude Code CLI, used by the tests (standard library only).

Tests point ClaudeRunner at this file instead of the real ``claude`` binary, so
no network, login or subscription is needed.

Behaviour:
    * ``--version`` prints ``9.9.9 (Claude Code)``.
    * ``auth status --json`` prints a logged-in subscription account.
    * Otherwise the prompt is read from stdin and one JSON result object is printed,
      shaped like the real ``claude -p --output-format json`` output.
    * When ``FAKE_CLAUDE_LOG`` names a file, one JSON line is appended per call:
      ``{"argv", "prompt", "cwd", "has_api_key"}``. ``has_api_key`` is true when
      ANTHROPIC_API_KEY or ANTHROPIC_AUTH_TOKEN reached this process.

Prompt markers (substrings of the prompt), checked in this order:
    * ``FAKE_SLEEP <seconds>``: sleep first, then continue with the markers below.
    * ``FAKE_LIMIT``: usage-limit error "You've hit your session limit", exit 1. The
      reset epoch comes from the FAKE_RESET_EPOCH environment variable, else now + 1 hour.
    * ``FAKE_ERROR``: error result with subtype ``error_during_execution``, exit 1.
    * ``FAKE_GARBAGE``: prints ``not json`` and exits 0.
    * ``FAKE_DENIED``: success whose permission_denials list the Bash tool.
    * Anything else: success with result ``echo: <prompt>``.

The session id is the ``--resume`` value when given, else ``fake-session-<8 hex>``.
"""

import json
import os
import re
import secrets
import sys
import time

VERSION_TEXT = "9.9.9 (Claude Code)"
AUTH_STATUS = {"loggedIn": True, "authMethod": "claude.ai", "subscriptionType": "pro"}
SLEEP_MARKER_RE = re.compile(r"FAKE_SLEEP\s+([0-9]+(?:\.[0-9]+)?)")
SECONDS_PER_HOUR = 3600
FAKE_DURATION_MS = 5
FAKE_COST_USD = 0.001
FAKE_TURNS = 1
SESSION_HEX_BYTES = 4
CREDENTIAL_VARIABLES = ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN")


def append_log(prompt: str) -> None:
    """Append one JSON line describing this call to FAKE_CLAUDE_LOG, if it is set.

    Args:
        prompt: The text read from stdin.
    """
    log_path = os.environ.get("FAKE_CLAUDE_LOG")
    if not log_path:
        return
    entry = {
        "argv": sys.argv[1:],
        "prompt": prompt,
        "cwd": os.getcwd(),
        "has_api_key": any(name in os.environ for name in CREDENTIAL_VARIABLES),
    }
    with open(log_path, "a", encoding="utf-8") as log_file:
        log_file.write(json.dumps(entry, ensure_ascii=False) + "\n")


def session_id_from_argv() -> str:
    """Return the session id: the value after ``--resume``, else a fresh random one.

    Returns:
        The id to report in the result.
    """
    argv = sys.argv[1:]
    if "--resume" in argv and argv.index("--resume") + 1 < len(argv):
        return argv[argv.index("--resume") + 1]
    return f"fake-session-{secrets.token_hex(SESSION_HEX_BYTES)}"


def result_object(
    result: str,
    *,
    is_error: bool = False,
    subtype: str = "success",
    denials: list[dict[str, str]] | None = None,
) -> dict:
    """Build a result object with the keys of the real CLI output.

    Args:
        result: The answer or error text.
        is_error: Whether Claude Code reports the run as failed.
        subtype: The result subtype, e.g. ``success`` or ``error_during_execution``.
        denials: Entries for ``permission_denials``; None means no denials.

    Returns:
        A JSON-serialisable dict.
    """
    return {
        "type": "result",
        "subtype": subtype,
        "is_error": is_error,
        "result": result,
        "session_id": session_id_from_argv(),
        "num_turns": FAKE_TURNS,
        "duration_ms": FAKE_DURATION_MS,
        "total_cost_usd": FAKE_COST_USD,
        "permission_denials": denials or [],
        "api_error_status": None,
        "terminal_reason": "completed",
    }


def reset_epoch() -> int:
    """Return the epoch (seconds) at which the fake usage limit resets.

    Returns:
        FAKE_RESET_EPOCH when set, else one hour from now.
    """
    configured = os.environ.get("FAKE_RESET_EPOCH")
    return int(configured) if configured else int(time.time()) + SECONDS_PER_HOUR


def answer(prompt: str) -> int:
    """Print the fake answer for ``prompt`` according to its markers.

    Args:
        prompt: The text read from stdin.

    Returns:
        The process exit code.
    """
    sleep_match = SLEEP_MARKER_RE.search(prompt)
    if sleep_match:
        time.sleep(float(sleep_match.group(1)))

    if "FAKE_LIMIT" in prompt:
        text = f"You've hit your session limit · resets 5pm|{reset_epoch()}"
        print(json.dumps(result_object(text, is_error=True)))
        return 1
    if "FAKE_ERROR" in prompt:
        print(json.dumps(result_object("", is_error=True, subtype="error_during_execution")))
        return 1
    if "FAKE_GARBAGE" in prompt:
        print("not json")
        return 0
    denials = [{"tool_name": "Bash"}] if "FAKE_DENIED" in prompt else None
    print(json.dumps(result_object(f"echo: {prompt}", denials=denials), ensure_ascii=False))
    return 0


def main() -> int:
    """Entry point: handle the info sub-commands, otherwise answer the prompt.

    Returns:
        The process exit code.
    """
    sys.stdout.reconfigure(encoding="utf-8")
    argv = sys.argv[1:]
    if "--version" in argv:
        print(VERSION_TEXT)
        return 0
    if argv[:2] == ["auth", "status"]:
        print(json.dumps(AUTH_STATUS))
        return 0

    sys.stdin.reconfigure(encoding="utf-8")
    prompt = sys.stdin.read()
    append_log(prompt)
    return answer(prompt)


if __name__ == "__main__":
    sys.exit(main())
