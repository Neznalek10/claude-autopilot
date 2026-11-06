"""Command line interface: `autopilot <command>`.

Responsibility: parse arguments, load settings, call the right collaborator
(app wiring, store, doctor, service installer) and turn the result into
terminal output and an exit code. It holds no business logic.

Where it fits: the `autopilot` console script and `python -m autopilot` end
in ``main``. It uses settings, app, store, engine (through ``build_app``),
doctor, service, system, schedule and formatting.

Invariants:
    * Results and tables go to stdout; progress, warnings and errors go to stderr.
    * Expected errors (``AutopilotError``) print one line ``autopilot: <message>``
      and exit 1; a traceback is shown only with ``--debug``.
    * Exit codes: 0 success, 1 failure, 2 usage error (argparse), 130 Ctrl-C.
    * Every time shown to the user is converted to ``settings.timezone``.
    * Commands that only read or edit state never need the Claude CLI; only
      `start`, `run` and `ask` look for it.
"""

from __future__ import annotations

import argparse
import dataclasses
import contextlib
import io
import os
import signal
import sqlite3
import sys
import threading
import traceback
from collections.abc import Callable, Iterator, Sequence
from datetime import datetime
from pathlib import Path

from autopilot import __version__
from autopilot.app import App, build_app, configure_logging, init_project, run_service, telegram_setup
from autopilot.doctor import format_report, run_doctor
from autopilot.errors import AutopilotError, TaskNotFoundError
from autopilot.formatting import METADATA_TIME_FORMAT, STATUS_ICONS, format_duration, format_time
from autopilot.models import (
    ASK_TASK_NAME,
    STATUS_SUCCESS,
    TRIGGER_MANUAL,
    RunOptions,
    RunOutcome,
    RunRecord,
)
from autopilot.schedule import parse_schedule
from autopilot.service import default_unit_dir, install_service, uninstall_service
from autopilot.settings import Settings, load_settings
from autopilot.store import Store
from autopilot.system import run_command
from autopilot.tasks import load_task_config

PROGRAM = "autopilot"
EXIT_OK = 0
EXIT_FAILURE = 1
EXIT_INTERRUPTED = 130

DEFAULT_HISTORY_LIMIT = 10
DEFAULT_NEXT_COUNT = 5
DEFAULT_SETUP_WAIT_SECONDS = 120
SUMMARY_WIDTH = 60

# argparse has no public name for the object add_subparsers() returns.
SubParsers = argparse._SubParsersAction
TABLE_GAP = "  "
NO_VALUE = "-"
# Level for commands that are not the service: only problems reach the terminal.
QUIET_LOG_LEVEL = "WARNING"
UNKNOWN_ICON = "?"
SHOW_TIME_FORMAT = "%Y-%m-%d %H:%M:%S"

# A handler gets the parsed arguments and the settings and returns the exit code.
Handler = Callable[[argparse.Namespace, Settings], int]


class _UnusedRunner:
    """Runner for commands that open the app but never execute Claude."""

    def run(self, prompt: str, options: RunOptions, resume_session: str | None = None) -> RunOutcome:
        """Always fails: reaching this is a programming error.

        Args:
            prompt: Ignored.
            options: Ignored.
            resume_session: Ignored.

        Raises:
            RuntimeError: Always.
        """
        raise RuntimeError("This command must not run Claude.")


# ---------------------------------------------------------------- parser


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser with every subcommand.

    ``--debug`` is accepted before or after the subcommand.

    Returns:
        The parser; ``parse_args`` sets ``args.handler`` to the command function.

    Example:
        >>> build_parser().parse_args(["next", "daily 07:30"]).expression
        'daily 07:30'
    """
    # SUPPRESS keeps a subparser from overwriting a --debug given before the command.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--debug",
        action="store_true",
        default=argparse.SUPPRESS,
        help="debug logging and full tracebacks on errors",
    )
    parser = argparse.ArgumentParser(
        prog=PROGRAM,
        parents=[common],
        description="Run Claude Code prompt tasks on schedules and control them from Telegram.",
        epilog="Example:\n  autopilot init\n  autopilot validate\n  autopilot start",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"{PROGRAM} {__version__}")
    commands = parser.add_subparsers(dest="command", required=True, metavar="<command>")
    for add_group in (
        _add_run_commands,
        _add_inspect_commands,
        _add_schedule_commands,
        _add_setup_commands,
    ):
        add_group(commands, common)
    return parser


def _add_command(
    commands: SubParsers,
    common: argparse.ArgumentParser,
    name: str,
    handler: Handler,
    summary: str,
    example: str,
) -> argparse.ArgumentParser:
    """Register one subcommand whose --help ends with a usage example."""
    sub = commands.add_parser(
        name,
        parents=[common],
        help=summary,
        description=summary,
        epilog=f"Example:\n  {example}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub.set_defaults(handler=handler)
    return sub


def _add_run_commands(commands: SubParsers, common: argparse.ArgumentParser) -> None:
    """Register `init`, `start`, `run` and `ask`."""
    _add_command(commands, common, "init", _cmd_init,
                 "create tasks.toml and .env from the examples", "autopilot init")
    _add_command(commands, common, "start", _cmd_start,
                 "run the scheduler and the Telegram bot (blocks)", "autopilot start")
    run = _add_command(commands, common, "run", _cmd_run,
                       "run one task now and print the result", "autopilot run ai-news --no-notify")
    run.add_argument("task", help="task name from tasks.toml")
    run.add_argument("--no-notify", action="store_true", help="do not send the result to Telegram")
    ask = _add_command(commands, common, "ask", _cmd_ask,
                       "send a one-off prompt to Claude", 'autopilot ask "Summarize today\'s news"')
    ask.add_argument("prompt", help="the prompt text")
    ask.add_argument("--no-notify", action="store_true", help="do not send the answer to Telegram")


def _add_inspect_commands(commands: SubParsers, common: argparse.ArgumentParser) -> None:
    """Register `list`, `history` and `show`."""
    _add_command(commands, common, "list", _cmd_list,
                 "show tasks, schedules and next runs", "autopilot list")
    history = _add_command(commands, common, "history", _cmd_history,
                           "show recent runs", "autopilot history ai-news -n 20")
    history.add_argument("task", nargs="?", help="only runs of this task")
    history.add_argument(
        "-n", dest="limit", type=_positive_int, default=DEFAULT_HISTORY_LIMIT,
        help=f"how many runs to show (default {DEFAULT_HISTORY_LIMIT})",
    )
    show = _add_command(commands, common, "show", _cmd_show,
                        "show one run with its full result", "autopilot show 42")
    show.add_argument("run_id", type=int, help="run id from `autopilot history`")


def _add_schedule_commands(commands: SubParsers, common: argparse.ArgumentParser) -> None:
    """Register `pause`, `resume`, `next` and `validate`."""
    pause = _add_command(commands, common, "pause", _cmd_pause,
                         "stop a task's schedule", "autopilot pause ai-news")
    pause.add_argument("task", help="task name from tasks.toml")
    resume = _add_command(commands, common, "resume", _cmd_resume,
                          "resume a task's schedule", "autopilot resume ai-news")
    resume.add_argument("task", help="task name from tasks.toml")
    preview = _add_command(commands, common, "next", _cmd_next,
                           "preview when a schedule fires", 'autopilot next "weekdays 08:00"')
    preview.add_argument("expression", help='schedule, e.g. "daily 07:30" or "30 7 * * 1-5"')
    preview.add_argument(
        "-n", dest="count", type=_positive_int, default=DEFAULT_NEXT_COUNT,
        help=f"how many fire times to show (default {DEFAULT_NEXT_COUNT})",
    )
    _add_command(commands, common, "validate", _cmd_validate,
                 "check tasks.toml and list its tasks", "autopilot validate")


def _add_setup_commands(commands: SubParsers, common: argparse.ArgumentParser) -> None:
    """Register `doctor`, `telegram-setup`, `install-service` and `uninstall-service`."""
    doctor = _add_command(commands, common, "doctor", _cmd_doctor,
                          "check the whole setup", "autopilot doctor --live")
    doctor.add_argument("--live", action="store_true", help="also make one tiny real Claude call")
    setup = _add_command(commands, common, "telegram-setup", _cmd_telegram_setup,
                         "connect the Telegram bot and save its token and chat id",
                         "autopilot telegram-setup --wait 300")
    setup.add_argument(
        "--wait", type=_positive_int, default=DEFAULT_SETUP_WAIT_SECONDS,
        help=f"seconds to wait for your first message (default {DEFAULT_SETUP_WAIT_SECONDS})",
    )
    _add_command(commands, common, "install-service", _cmd_install_service,
                 "install and start the systemd user service", "autopilot install-service")
    _add_command(commands, common, "uninstall-service", _cmd_uninstall_service,
                 "stop and remove the systemd user service", "autopilot uninstall-service")


def _positive_int(text: str) -> int:
    """argparse type: an integer of at least 1."""
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"'{text}' is not a whole number") from None
    if value < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return value


# ---------------------------------------------------------------- entry point


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI.

    Args:
        argv: Arguments without the program name; defaults to ``sys.argv[1:]``.

    Returns:
        The process exit code: 0 on success, 1 on an expected error or failed
        run, 130 after Ctrl-C. Invalid usage exits 2 through argparse.

    Example:
        >>> main(["next", "daily 07:30", "-n", "2"])  # doctest: +SKIP
        0
    """
    try:
        args = build_parser().parse_args(argv)
    except SystemExit:
        # --help, --version and usage errors print through argparse and exit here.
        _flush_stdout()
        raise
    try:
        code = _execute(args)
        return code if _flush_stdout() else EXIT_FAILURE
    except AutopilotError as err:
        print(f"{PROGRAM}: {err}", file=sys.stderr)
        # Tracebacks are for bug reports, so they are opt-in.
        if _debug_requested(args):
            traceback.print_exc(file=sys.stderr)
        return EXIT_FAILURE
    except BrokenPipeError:
        # The reader (for example `head`) went away: stop quietly.
        _silence_stdout()
        return EXIT_FAILURE
    except KeyboardInterrupt:
        print(file=sys.stderr)
        return EXIT_INTERRUPTED


def _flush_stdout() -> bool:
    """Write buffered output now, while a closed pipe can still be handled quietly.

    Without this, the interpreter flushes stdout at exit, outside every handler,
    and a reader that already went away (``autopilot list | true``) turns that
    flush into an "Exception ignored" warning and exit code 120.

    Returns:
        False when the reader has gone away; stdout then points at devnull.
    """
    try:
        sys.stdout.flush()
    except BrokenPipeError:
        _silence_stdout()
        return False
    return True


def _execute(args: argparse.Namespace) -> int:
    """Load settings, configure logging and run the chosen command.

    Raises:
        AutopilotError: Expected failures, including database problems
            (``sqlite3.Error`` is converted so the user sees one friendly line).
    """
    settings = load_settings()
    _configure_cli_logging(settings, args)
    try:
        return args.handler(args, settings)
    except sqlite3.Error as err:
        raise AutopilotError(f"database error in {settings.db_path}: {err}") from err


def _silence_stdout() -> None:
    """Point stdout at /dev/null so the interpreter's exit flush cannot fail again.

    This is the approach the Python documentation recommends after a
    BrokenPipeError. It does nothing when stdout has no file descriptor
    (for example when it is redirected to a StringIO).
    """
    try:
        descriptor = sys.stdout.fileno()
    except (AttributeError, ValueError, io.UnsupportedOperation):
        return
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, descriptor)
    os.close(devnull)


@contextlib.contextmanager
def _interrupt_on_termination() -> Iterator[None]:
    """Turn SIGTERM and SIGHUP into KeyboardInterrupt while the block runs.

    That lets the runner kill Claude's process group and the engine record the
    run as interrupted, exactly as for Ctrl-C. Previous handlers are restored
    afterwards. Handlers can only be installed on the main thread, so elsewhere
    this does nothing.
    """
    if threading.current_thread() is not threading.main_thread():
        yield
        return

    def raise_interrupt(signum: int, frame: object) -> None:
        """Signal handler: behave like Ctrl-C."""
        raise KeyboardInterrupt

    previous = {
        signum: signal.signal(signum, raise_interrupt)
        for signum in (signal.SIGTERM, signal.SIGHUP)
    }
    try:
        yield
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


def _debug_requested(args: argparse.Namespace) -> bool:
    """True when --debug was given (the option is absent from the namespace otherwise)."""
    return bool(getattr(args, "debug", False))


def _configure_cli_logging(settings: Settings, args: argparse.Namespace) -> None:
    """Set up logging: full (file and stderr) for `start`, quiet for everything else."""
    if args.command == "start":
        configure_logging(settings, debug=_debug_requested(args))
        return
    quiet = dataclasses.replace(settings, log_level=QUIET_LOG_LEVEL)
    configure_logging(quiet, debug=_debug_requested(args), to_file=False)


# ---------------------------------------------------------------- helpers


def _open_app(settings: Settings) -> App:
    """Build the app for commands that never run Claude (no Claude CLI needed)."""
    return build_app(settings, notify=False, runner=_UnusedRunner())


def _now(settings: Settings) -> datetime:
    """Current time in the configured timezone."""
    return datetime.now(settings.timezone)


def _format_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    """Render left-aligned columns separated by two spaces.

    Args:
        headers: Column titles.
        rows: Cell texts, one sequence per row, same length as ``headers``.

    Returns:
        The table as text without trailing spaces.
    """
    widths = [max(len(cell) for cell in column) for column in zip(headers, *rows, strict=True)]
    lines = [headers, *rows]
    return "\n".join(
        TABLE_GAP.join(cell.ljust(width) for cell, width in zip(line, widths, strict=True)).rstrip()
        for line in lines
    )


def _first_line(text: str, width: int = SUMMARY_WIDTH) -> str:
    """Return the first non-empty line of ``text``, shortened to ``width`` characters."""
    line = next((part.strip() for part in text.splitlines() if part.strip()), "")
    return line if len(line) <= width else line[: width - 1] + "…"


def _status_text(status: str) -> str:
    """Status name with its icon, e.g. "✅ success"."""
    return f"{STATUS_ICONS.get(status, UNKNOWN_ICON)} {status}"


def _last_run_text(record: RunRecord, now: datetime) -> str:
    """Status icon plus the time a finished run ended, e.g. "✅ 07:30"."""
    icon = STATUS_ICONS.get(record.status, UNKNOWN_ICON)
    return f"{icon} {format_time(record.finished_at or record.queued_at, now)}"


def _require_task(app: App, name: str) -> None:
    """Raise TaskNotFoundError unless ``name`` is a task in the loaded tasks.toml."""
    known = app.engine.config.tasks
    if name not in known:
        raise TaskNotFoundError(name, known)


def _finish_one_off(app: App, run_id: int, label: str) -> int:
    """Execute a queued run in this process and report its outcome.

    Args:
        app: Wired application.
        run_id: Id of a queued, undispatched run.
        label: Name shown in the progress line.

    Returns:
        0 when the run succeeded, otherwise 1.

    Raises:
        AutopilotError: The run record disappeared.
    """
    print(f"Running {label}...", file=sys.stderr)
    with _interrupt_on_termination():
        record = app.engine.execute_run(run_id)
    if record is None:
        raise AutopilotError(f"Run #{run_id} disappeared from the database.")
    if record.status == STATUS_SUCCESS:
        print(record.result)
        return EXIT_OK
    reason = record.error or f"finished with status {record.status}"
    print(f"{PROGRAM}: run #{record.id} {record.status}: {reason}", file=sys.stderr)
    return EXIT_FAILURE


# ---------------------------------------------------------------- commands


def _cmd_init(args: argparse.Namespace, settings: Settings) -> int:
    """`init`: create tasks.toml and .env and print the next steps."""
    created = init_project(settings.home)
    if created:
        print(f"Created in {settings.home}: {', '.join(created)}")
    else:
        print(f"Nothing to do: tasks.toml and .env already exist in {settings.home}.")
    print("Next steps:")
    print("  1. Edit tasks.toml, then check it with: autopilot validate")
    print("  2. Connect your phone with: autopilot telegram-setup")
    print("  3. Check everything with: autopilot doctor")
    print("  4. Start it with: autopilot start (or autopilot install-service)")
    return EXIT_OK


def _cmd_start(args: argparse.Namespace, settings: Settings) -> int:
    """`start`: run the service until it is stopped."""
    if not settings.telegram_enabled:
        print(
            f"{PROGRAM}: Telegram is not configured, results are only logged. "
            "Run `autopilot telegram-setup` to connect your phone.",
            file=sys.stderr,
        )
    app = build_app(settings)
    run_service(app, threading.Event())
    return EXIT_OK


def _cmd_run(args: argparse.Namespace, settings: Settings) -> int:
    """`run <task>`: run a task now, in this process."""
    app = build_app(settings, notify=not args.no_notify)
    run_id = app.engine.enqueue_task(args.task, TRIGGER_MANUAL, dispatch=False)
    if run_id is None:
        raise AutopilotError(f"Task '{args.task}' is already queued or running.")
    return _finish_one_off(app, run_id, args.task)


def _cmd_ask(args: argparse.Namespace, settings: Settings) -> int:
    """`ask "<prompt>"`: one-off prompt with the [ask] options."""
    app = build_app(settings, notify=not args.no_notify)
    try:
        run_id = app.engine.enqueue_prompt(args.prompt, TRIGGER_MANUAL, dispatch=False)
    except ValueError as err:
        raise AutopilotError(str(err)) from None
    return _finish_one_off(app, run_id, ASK_TASK_NAME)


def _cmd_list(args: argparse.Namespace, settings: Settings) -> int:
    """`list`: one table row per task."""
    app = _open_app(settings)
    now = _now(settings)
    next_times = app.engine.next_fire_times()
    rows: list[list[str]] = []
    for name, task in app.engine.config.tasks.items():
        paused = app.store.is_paused(name)
        last = app.store.list_runs(task_name=name, limit=1, finished_only=True)
        rows.append(
            [
                name,
                "; ".join(s.description for s in task.schedules) or "manual",
                format_time(next_times[name], now) if next_times[name] else NO_VALUE,
                _last_run_text(last[0], now) if last else NO_VALUE,
                "off" if not task.enabled else "paused" if paused else "on",
            ]
        )
    if not rows:
        print("No tasks defined. Add [[task]] entries to tasks.toml.")
        return EXIT_OK
    print(_format_table(["NAME", "SCHEDULE", "NEXT", "LAST", "STATE"], rows))
    return EXIT_OK


def _cmd_history(args: argparse.Namespace, settings: Settings) -> int:
    """`history [task] [-n N]`: recent runs, newest first."""
    store = Store(settings.db_path)
    now = _now(settings)
    records = store.list_runs(task_name=args.task, limit=args.limit)
    if not records:
        print("No runs yet.")
        return EXIT_OK
    rows = [
        [
            str(record.id),
            format_time(record.queued_at, now),
            record.task_name,
            _status_text(record.status),
            format_duration(record.duration_ms),
            _first_line(record.result or record.error or ""),
        ]
        for record in records
    ]
    print(_format_table(["ID", "WHEN", "TASK", "STATUS", "DURATION", "RESULT"], rows))
    return EXIT_OK


def _cmd_show(args: argparse.Namespace, settings: Settings) -> int:
    """`show <id>`: metadata plus the full result or error of one run."""
    record = Store(settings.db_path).get_run(args.run_id)
    if record is None:
        raise AutopilotError(f"No run with id {args.run_id}. See `autopilot history`.")
    print(_run_details(record, settings))
    print()
    body = record.result if record.status == STATUS_SUCCESS else (record.error or record.result)
    print(body or "(no output)")
    return EXIT_OK


def _run_details(record: RunRecord, settings: Settings) -> str:
    """Build the metadata block shown by `show` (times in the configured timezone)."""

    def stamp(moment: datetime | None) -> str:
        """Local date and time, or "-" when the moment is unknown."""
        if moment is None:
            return NO_VALUE
        return moment.astimezone(settings.timezone).strftime(SHOW_TIME_FORMAT)

    fields = [
        ("Run", f"#{record.id}"),
        ("Task", record.task_name),
        ("Trigger", record.trigger),
        ("Status", _status_text(record.status)),
        ("Queued", stamp(record.queued_at)),
        ("Started", stamp(record.started_at)),
        ("Finished", stamp(record.finished_at)),
        ("Duration", format_duration(record.duration_ms)),
    ]
    # Optional facts only appear when the run produced them.
    if record.num_turns is not None:
        fields.append(("Turns", str(record.num_turns)))
    if record.cost_usd is not None:
        fields.append(("Cost", f"${record.cost_usd:.4f} (API-equivalent)"))
    if record.session_id:
        fields.append(("Session", record.session_id))
    if record.denied_tools:
        fields.append(("Blocked tools", ", ".join(record.denied_tools)))
    if record.retry_of is not None:
        fields.append(("Retry of", f"#{record.retry_of}"))
    return "\n".join(f"{label + ':':<15}{value}" for label, value in fields)


def _cmd_pause(args: argparse.Namespace, settings: Settings) -> int:
    """`pause <task>`: stop scheduling the task."""
    return _set_paused(args.task, settings, paused=True)


def _cmd_resume(args: argparse.Namespace, settings: Settings) -> int:
    """`resume <task>`: schedule the task again."""
    return _set_paused(args.task, settings, paused=False)


def _set_paused(name: str, settings: Settings, *, paused: bool) -> int:
    """Validate the task, store its paused flag and confirm."""
    app = _open_app(settings)
    _require_task(app, name)
    app.store.set_paused(name, paused)
    print(f"{'Paused' if paused else 'Resumed'} {name}.")
    return EXIT_OK


def _cmd_next(args: argparse.Namespace, settings: Settings) -> int:
    """`next "<expr>"`: preview the next fire times in the configured timezone."""
    schedule = parse_schedule(args.expression)
    # Compute everything first so a schedule that never fires prints nothing.
    moments: list[datetime] = []
    moment = _now(settings)
    for _ in range(args.count):
        moment = schedule.next_after(moment)
        moments.append(moment)
    print(f"Schedule: {schedule.description}")
    for moment in moments:
        print(f"  {_weekday(moment)} {moment.strftime(METADATA_TIME_FORMAT)}")
    return EXIT_OK


def _weekday(moment: datetime) -> str:
    """English weekday abbreviation, independent of the system locale."""
    return ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")[moment.weekday()]


def _cmd_validate(args: argparse.Namespace, settings: Settings) -> int:
    """`validate`: load tasks.toml and list what it defines."""
    config = load_task_config(settings.tasks_file)
    count = len(config.tasks)
    print(f"tasks.toml OK: {count} {'task' if count == 1 else 'tasks'}")
    for name, task in config.tasks.items():
        schedules = "; ".join(s.description for s in task.schedules) or "manual only"
        suffix = "" if task.enabled else " (disabled)"
        print(f"  {name}: {schedules}{suffix}")
    return EXIT_OK


def _cmd_doctor(args: argparse.Namespace, settings: Settings) -> int:
    """`doctor [--live]`: print the health report; exit 1 when a check failed."""
    results = run_doctor(settings, live=args.live)
    print(format_report(results))
    return EXIT_FAILURE if any(result.status == "fail" for result in results) else EXIT_OK


def _cmd_telegram_setup(args: argparse.Namespace, settings: Settings) -> int:
    """`telegram-setup [--wait N]`: connect the bot."""
    # `out` is passed explicitly so output follows the current sys.stdout.
    telegram_setup(settings, out=sys.stdout, wait_seconds=args.wait)
    return EXIT_OK


def _cmd_install_service(args: argparse.Namespace, settings: Settings) -> int:
    """`install-service`: write and enable the systemd user unit."""
    steps = install_service(
        python=Path(sys.executable),
        project_home=settings.home,
        path_env=os.environ.get("PATH", ""),
        run=run_command,
        unit_dir=default_unit_dir(),
    )
    print("\n".join(steps))
    return EXIT_OK


def _cmd_uninstall_service(args: argparse.Namespace, settings: Settings) -> int:
    """`uninstall-service`: disable and remove the systemd user unit."""
    steps = uninstall_service(run=run_command, unit_dir=default_unit_dir())
    print("\n".join(steps))
    return EXIT_OK
