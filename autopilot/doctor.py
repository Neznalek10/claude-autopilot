"""Health checks behind `autopilot doctor`.

Responsibility: inspect the setup (Claude CLI, subscription login, API key
variables, tasks.toml, Telegram, data folder, systemd service) and report each
finding as a ``CheckResult`` with a plain-language fix. Nothing here changes
the system; the only side effect of ``--live`` is one tiny real Claude call.

Where it fits: ``cli.py`` calls ``run_doctor`` and prints ``format_report``.
It uses settings, tasks, runner, telegram_api, service and system.

Invariants:
    * The login check shows the plan only. The account email and organisation
      fields from `claude auth status` are never copied into a result.
    * The bot token never appears in any result.
    * Every command, client and clock is injected, so tests need no real
      network, systemd or Claude CLI.
    * A missing Claude binary turns the checks that need it into "warn: skipped".
"""

from __future__ import annotations

import getpass
import json
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from autopilot.errors import AutopilotError, ClaudeNotFoundError
from autopilot.formatting import format_duration
from autopilot.models import STATUS_RATE_LIMITED, STATUS_SUCCESS, RunOptions
from autopilot.runner import ClaudeRunner, find_claude_binary
from autopilot.service import UNIT_NAME, linger_enabled
from autopilot.settings import Settings
from autopilot.system import CommandRunner, run_command
from autopilot.tasks import load_task_config
from autopilot.telegram_api import TelegramClient

STATUS_OK = "ok"
STATUS_WARN = "warn"
STATUS_FAIL = "fail"
STATUS_MARKS = {STATUS_OK: "✔", STATUS_WARN: "!", STATUS_FAIL: "✘"}

COMMAND_TIMEOUT_SECONDS = 20.0
LIVE_TIMEOUT_SECONDS = 120
LIVE_MODEL = "haiku"
LIVE_PROMPT = "Reply with the single word OK."
AUTH_METHOD_SUBSCRIPTION = "claude.ai"
# Environment variables that would make Claude bill an API account.
API_KEY_VARIABLES = ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN")
FIX_INDENT = "    "

FIX_INSTALL_CLAUDE = "Install Claude Code (`claude install`) or set CLAUDE_BIN in .env."
FIX_LOGIN = "Run `claude auth login` and sign in with your Claude subscription."
FIX_TASKS = "Edit tasks.toml, then run `autopilot validate`."
FIX_TELEGRAM = "Run `autopilot telegram-setup`."
FIX_SERVICE = "Run `autopilot install-service` to start it at boot."
FIX_LINGER = "Run `sudo loginctl enable-linger $USER` to keep it running without a login."


@dataclass(frozen=True)
class CheckResult:
    """The outcome of one health check.

    Attributes:
        name: Short check name, e.g. "Claude CLI".
        status: "ok", "warn" or "fail".
        detail: One line saying what was found.
        fix: What the user can do about a warning or failure; empty when none is needed.
    """

    name: str
    status: str
    detail: str
    fix: str = ""


def _skipped(name: str) -> CheckResult:
    """Build the result for a check that needs the Claude binary when it is missing."""
    return CheckResult(name, STATUS_WARN, "skipped (Claude CLI not found)")


def check_claude_binary(settings: Settings, run: CommandRunner) -> tuple[CheckResult, Path | None]:
    """Locate the Claude CLI and read its version.

    Args:
        settings: Provides the optional CLAUDE_BIN path.
        run: Command runner used for `claude --version`.

    Returns:
        The result and the binary path (None when it could not be found).
    """
    name = "Claude CLI"
    try:
        binary = find_claude_binary(settings.claude_bin)
    except ClaudeNotFoundError as err:
        return CheckResult(name, STATUS_FAIL, str(err), FIX_INSTALL_CLAUDE), None
    code, stdout, stderr = run([str(binary), "--version"], COMMAND_TIMEOUT_SECONDS)
    if code != 0:
        reason = (stderr or stdout).strip() or f"exit code {code}"
        return CheckResult(name, STATUS_FAIL, f"{binary} did not run: {reason}", FIX_INSTALL_CLAUDE), binary
    version = stdout.strip() or "unknown version"
    return CheckResult(name, STATUS_OK, f"{binary} ({version})"), binary


def check_claude_login(binary: Path, run: CommandRunner) -> CheckResult:
    """Check that Claude is signed in with a subscription, not an API account.

    Only the plan is reported: email and organisation fields are never read out.

    Args:
        binary: Path of the Claude CLI.
        run: Command runner used for `claude auth status --json`.

    Returns:
        ok for a claude.ai login, warn for another auth method or unreadable
        output, fail when logged out.
    """
    name = "Claude login"
    code, stdout, stderr = run([str(binary), "auth", "status", "--json"], COMMAND_TIMEOUT_SECONDS)
    status = _parse_json_object(stdout)
    if status is None:
        reason = (stderr or stdout).strip()[:200] or f"exit code {code}"
        return CheckResult(name, STATUS_WARN, f"could not read the login status ({reason})", FIX_LOGIN)
    if not status.get("loggedIn"):
        return CheckResult(name, STATUS_FAIL, "not logged in", FIX_LOGIN)
    method = status.get("authMethod")
    if method != AUTH_METHOD_SUBSCRIPTION:
        return CheckResult(
            name,
            STATUS_WARN,
            "logged in with an API account (runs would bill it, not your subscription)",
            FIX_LOGIN,
        )
    plan = status.get("subscriptionType")
    detail = f"{plan} plan" if isinstance(plan, str) and plan else "subscription login"
    return CheckResult(name, STATUS_OK, detail)


def _parse_json_object(text: str) -> dict[str, Any] | None:
    """Parse ``text`` as a JSON object; return None when it is not one."""
    try:
        data = json.loads(text)
    except ValueError:
        return None
    return data if isinstance(data, dict) else None


def check_api_key(env: Mapping[str, str]) -> CheckResult:
    """Warn when an Anthropic API credential is present in the environment.

    Autopilot strips these from every Claude run, so the warning is about the
    user's other tools and about surprises, not about autopilot billing.

    Args:
        env: Environment to inspect.

    Returns:
        ok when none is set, warn naming the variables otherwise (values are never shown).
    """
    name = "API key"
    present = [variable for variable in API_KEY_VARIABLES if env.get(variable)]
    if not present:
        return CheckResult(name, STATUS_OK, "no Anthropic API key in the environment")
    return CheckResult(
        name,
        STATUS_WARN,
        f"{', '.join(present)} is set; autopilot removes it from Claude runs so your subscription is used",
        "Unset it if you want plain `claude` in this shell to use your subscription too.",
    )


def check_tasks(settings: Settings) -> CheckResult:
    """Validate tasks.toml.

    Args:
        settings: Provides the tasks file path.

    Returns:
        ok with the task counts, or fail with the validation message.
    """
    name = "tasks.toml"
    try:
        config = load_task_config(settings.tasks_file)
    except AutopilotError as err:
        return CheckResult(name, STATUS_FAIL, str(err), FIX_TASKS)
    enabled = sum(1 for task in config.tasks.values() if task.enabled)
    return CheckResult(name, STATUS_OK, f"{len(config.tasks)} tasks ({enabled} enabled)")


def check_telegram(settings: Settings, client_factory: Callable[[str], TelegramClient]) -> CheckResult:
    """Check the Telegram token and chat id.

    Args:
        settings: Provides the token and chat id.
        client_factory: Builds a client from the token (replaced by a fake in tests).

    Returns:
        warn when not configured, fail when Telegram rejects the token, else ok
        with the bot username. The token is never included.
    """
    name = "Telegram"
    if settings.telegram_token is None and settings.telegram_chat_id is None:
        return CheckResult(name, STATUS_WARN, "not configured (results stay in the log)", FIX_TELEGRAM)
    if settings.telegram_token is None:
        return CheckResult(name, STATUS_FAIL, "TELEGRAM_CHAT_ID is set but TELEGRAM_BOT_TOKEN is missing", FIX_TELEGRAM)
    if settings.telegram_chat_id is None:
        return CheckResult(name, STATUS_FAIL, "TELEGRAM_BOT_TOKEN is set but TELEGRAM_CHAT_ID is missing", FIX_TELEGRAM)
    try:
        bot = client_factory(settings.telegram_token).get_me()
    except AutopilotError as err:
        return CheckResult(name, STATUS_FAIL, str(err), FIX_TELEGRAM)
    username = bot.get("username", "?")
    return CheckResult(name, STATUS_OK, f"bot @{username}, chat id configured")


def check_data_dir(settings: Settings) -> CheckResult:
    """Check that the data folder exists and is writable, or can be created.

    Args:
        settings: Provides the data folder.

    Returns:
        ok when writable, warn when it will be created on first start, fail otherwise.
    """
    name = "Data folder"
    data_dir = settings.data_dir
    if data_dir.exists():
        if data_dir.is_dir() and os.access(data_dir, os.W_OK | os.X_OK):
            return CheckResult(name, STATUS_OK, f"{data_dir} is writable")
        return CheckResult(name, STATUS_FAIL, f"{data_dir} is not a writable folder", f"Fix its permissions: chmod u+rwx {data_dir}")
    # Not created yet: look for the nearest existing parent to see whether creation can work.
    parent = next((p for p in data_dir.parents if p.exists()), None)
    if parent is not None and os.access(parent, os.W_OK | os.X_OK):
        return CheckResult(name, STATUS_WARN, f"{data_dir} does not exist yet, it will be created on first start")
    return CheckResult(name, STATUS_FAIL, f"{data_dir} cannot be created", "Set AUTOPILOT_DATA_DIR to a folder you can write to.")


def check_service(run: CommandRunner, user: str) -> CheckResult:
    """Report the systemd user service and linger state.

    Args:
        run: Command runner used for systemctl and loginctl.
        user: Login name for the linger query.

    Returns:
        ok when enabled and active (and linger is on or unknown), warn for
        every other state: the service is optional, `autopilot start` also works.
    """
    name = "Service"
    code, stdout, _ = run(["systemctl", "--user", "is-enabled", UNIT_NAME], COMMAND_TIMEOUT_SECONDS)
    if code in (124, 126, 127):
        return CheckResult(name, STATUS_WARN, "systemd is not available here, run `autopilot start` by hand")
    enabled = stdout.strip() == "enabled"
    if not enabled:
        return CheckResult(name, STATUS_WARN, "not installed (fine if you run `autopilot start` yourself)", FIX_SERVICE)
    active_code, active_out, _ = run(["systemctl", "--user", "is-active", UNIT_NAME], COMMAND_TIMEOUT_SECONDS)
    is_active = active_code == 0 and active_out.strip() == "active"
    linger = linger_enabled(run, user)
    state = f"enabled, {'running' if is_active else 'not running'}"
    if linger is False:
        return CheckResult(name, STATUS_WARN, f"{state}, linger off (stops when you log out)", FIX_LINGER)
    if not is_active:
        return CheckResult(name, STATUS_WARN, state, f"Start it with `systemctl --user start {UNIT_NAME}`.")
    return CheckResult(name, STATUS_OK, state + (", linger on" if linger else ""))


def check_live(binary: Path, clock: Callable[[], datetime], workdir: Path) -> CheckResult:
    """Make one tiny real Claude call (small model, no tools) to prove the whole path works.

    Args:
        binary: Path of the Claude CLI.
        clock: Returns the aware current time for the runner.
        workdir: Folder the call runs in.

    Returns:
        ok with the duration, warn when the usage limit is reached, else fail.
    """
    name = "Live call"
    options = RunOptions(
        workdir=workdir,
        model=LIVE_MODEL,
        allowed_tools=(),
        timeout_seconds=LIVE_TIMEOUT_SECONDS,
    )
    outcome = ClaudeRunner(lambda: binary, clock=clock).run(LIVE_PROMPT, options)
    if outcome.status == STATUS_SUCCESS:
        return CheckResult(name, STATUS_OK, f"Claude answered in {format_duration(outcome.duration_ms)}")
    if outcome.status == STATUS_RATE_LIMITED:
        return CheckResult(name, STATUS_WARN, "the Claude usage limit is reached right now", "Try again after the limit resets.")
    return CheckResult(name, STATUS_FAIL, outcome.error or f"run ended as {outcome.status}", FIX_LOGIN)


def run_doctor(
    settings: Settings,
    *,
    live: bool = False,
    run: CommandRunner = run_command,
    env: Mapping[str, str] | None = None,
    client_factory: Callable[[str], TelegramClient] = TelegramClient,
    user: str | None = None,
) -> list[CheckResult]:
    """Run every check in a fixed order.

    Args:
        settings: Resolved settings.
        live: Also make one real Claude call.
        run: Command runner for the CLI and systemd checks.
        env: Environment for the API key check; defaults to ``os.environ``.
        client_factory: Builds the Telegram client.
        user: Login name for the linger check; defaults to the current user.

    Returns:
        The results, in report order.

    Example:
        >>> results = run_doctor(load_settings())  # doctest: +SKIP
        >>> print(format_report(results))  # doctest: +SKIP
    """
    binary_result, binary = check_claude_binary(settings, run)
    results = [binary_result]
    results.append(check_claude_login(binary, run) if binary else _skipped("Claude login"))
    results.append(check_api_key(os.environ if env is None else env))
    results.append(check_tasks(settings))
    results.append(check_telegram(settings, client_factory))
    results.append(check_data_dir(settings))
    results.append(check_service(run, user or getpass.getuser()))
    if live:
        if binary:
            results.append(check_live(binary, lambda: datetime.now(settings.timezone), settings.home))
        else:
            results.append(_skipped("Live call"))
    return results


def format_report(results: Sequence[CheckResult]) -> str:
    """Render results as a text report with a summary line.

    Args:
        results: Check results in display order.

    Returns:
        Lines like "✔ name: detail", each fix indented below, then a summary.
    """
    lines: list[str] = []
    for result in results:
        lines.append(f"{STATUS_MARKS[result.status]} {result.name}: {result.detail}")
        if result.fix:
            lines.append(f"{FIX_INDENT}Fix: {result.fix}")
    counts = {status: sum(1 for r in results if r.status == status) for status in STATUS_MARKS}
    lines.append("")
    lines.append(
        f"{counts[STATUS_OK]} ok, {counts[STATUS_WARN]} warnings, {counts[STATUS_FAIL]} failed"
    )
    return "\n".join(lines)
