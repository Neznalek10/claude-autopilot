"""The run engine: queue, workers, tick loop, usage-limit gate and notifications.

The engine is the heart of the service. Every 30 seconds (or sooner, when a
schedule fires) ``tick`` reloads ``tasks.toml`` if it changed, queues the runs
that are due and queues retries of runs that hit the Claude usage limit.
Worker threads take queued run ids and execute them through the injected
``Runner``, store the outcome and tell the ``Notifier`` according to the task's
notify policy.

Used by: app.py (wiring and the service loop), bot.py (``enqueue_*``,
``snapshot``, ``reset_chat``, ``next_fire_times``) and cli.py (one-off runs).
Depends on: models.py, scheduler.py, store.py and errors.py. It knows the
runner and the notifier only through the protocols in models.py, and gets its
task configuration from an injected ``config_loader``.

Invariants:
    * "now" always comes from the injected clock; all datetimes are aware.
    * The usage-limit gate (``limited_until``) lives in the store, so it
      survives restarts. While it is closed no run calls Claude.
    * A task never has two active (queued or running) runs.
    * Chat turns run under one lock, so their order and session handling are
      deterministic even with several workers.
    * Workers never die: every exception inside a run is logged and recorded.
"""

from __future__ import annotations

import dataclasses
import logging
import queue
import threading
import time
from collections.abc import Callable
from datetime import datetime, timedelta
from pathlib import Path

from autopilot.errors import ConfigError, TaskNotFoundError
from autopilot.models import (
    ASK_TASK_NAME,
    CHAT_IDLE_HOURS,
    DEFAULT_TIMEOUT_SECONDS,
    NOTIFY_ALWAYS,
    NOTIFY_ERRORS,
    NOTIFY_NEVER,
    STATUS_CANCELLED,
    STATUS_ERROR,
    STATUS_QUEUED,
    STATUS_RATE_LIMITED,
    STATUS_RUNNING,
    STATUS_SUCCESS,
    TRIGGER_CHAT,
    TRIGGER_SCHEDULE,
    Notifier,
    QueueSnapshot,
    RunOptions,
    RunOutcome,
    RunRecord,
    Runner,
    TaskConfig,
)
from autopilot.scheduler import evaluate, next_fire_time, to_utc
from autopilot.store import INTERRUPTED_MESSAGE, Store

logger = logging.getLogger(__name__)

TICK_SECONDS = 30
MIN_WAIT_SECONDS = 1
DEFAULT_RETRY_DELAY = timedelta(minutes=60)
MAX_RATE_LIMIT_RETRIES = 3
LIMITED_UNTIL_KEY = "limited_until"
CHAT_SESSION_KEY = "chat_session"
CHAT_UPDATED_KEY = "chat_updated_at"
CHAT_RESET_KEY = "chat_reset_at"
CHAT_IDLE_RESET = timedelta(hours=CHAT_IDLE_HOURS)
# Time formats: "HH:MM" in user notices, a full date and time in log lines.
CLOCK_TIME_FORMAT = "%H:%M"
LOG_SLOT_TIME_FORMAT = "%Y-%m-%d %H:%M"
# A run still "running" this long after its timeout was abandoned by a dead process.
STALE_RUN_GRACE = timedelta(minutes=5)
# How long stop() waits for workers to finish their current run before giving up on them.
WORKER_JOIN_SECONDS = 5

REMOVED_TASK_MESSAGE = "Task was removed from tasks.toml."
CHAT_RESET_NOTE = "The conversation was reset, so your next message starts a new one."
RELOAD_WARNING = "⚠️ tasks.toml has an error, keeping the previous tasks:\n{message}"
LIMIT_NOTICE = (
    "⏳ Claude usage limit reached. All runs wait until {time}, then retry automatically."
)
LIMIT_WAIT_MESSAGE = "Waiting for the Claude usage limit to reset at {time}."
GIVE_UP_MESSAGE = "Gave up after {retries} retries because of the usage limit."


def _clock_time(moment: datetime, now: datetime) -> str:
    """Format a moment as local "HH:MM" in the clock's time zone.

    Store datetimes carry a fixed UTC offset, so they are converted first.

    Args:
        moment: Any aware datetime.
        now: The current moment; its tzinfo is the zone shown to the user.

    Returns:
        The wall-clock time, e.g. "08:30".
    """
    return moment.astimezone(now.tzinfo).strftime(CLOCK_TIME_FORMAT)


class LogNotifier:
    """Notifier used when Telegram is not configured: writes one log line per call."""

    def notify_run(self, record: RunRecord) -> None:
        """Log that a run finished.

        Args:
            record: The finished run.
        """
        logger.info("Run #%d (%s) finished: %s", record.id, record.task_name, record.status)

    def notify_text(self, text: str) -> None:
        """Log a notice that would have gone to the phone.

        Args:
            text: The notice text; only the first line is logged.
        """
        logger.info("Notice: %s", text.splitlines()[0] if text else "")


class Engine:
    """Queues, executes and records Claude runs.

    Example:
        engine = Engine(store=store, runner=runner, notifier=notifier,
                        config_loader=lambda: load_task_config(path),
                        clock=lambda: datetime.now(tz), tasks_file=path)
        engine.load_config()
        engine.run_forever(stop_event)
    """

    def __init__(
        self,
        *,
        store: Store,
        runner: Runner,
        notifier: Notifier,
        config_loader: Callable[[], TaskConfig],
        clock: Callable[[], datetime],
        tasks_file: Path | None = None,
        max_parallel: int = 1,
    ) -> None:
        """Wire the engine to its collaborators; nothing runs until a method is called.

        Args:
            store: Persistence for runs, markers and the usage-limit gate.
            runner: Executes one prompt (the Claude CLI wrapper, or a fake).
            notifier: Delivers finished runs and notices to the user.
            config_loader: Returns a freshly parsed TaskConfig; raises ConfigError when invalid.
            clock: Returns the current aware datetime.
            tasks_file: File watched for changes (by modification time) to hot-reload tasks.
            max_parallel: Number of worker threads.
        """
        self._store = store
        self._runner = runner
        self._notifier = notifier
        self._config_loader = config_loader
        self._clock = clock
        self._tasks_file = tasks_file
        self._max_parallel = max_parallel
        self._config: TaskConfig | None = None
        self._tasks_mtime_ns: int | None = None
        self._last_reload_error: str | None = None
        self._queue: queue.Queue[int | None] = queue.Queue()
        self._workers: list[threading.Thread] = []
        self._stopping = threading.Event()
        # Held while checking for an active run and creating the new one, so two
        # threads cannot both queue the same task.
        self._enqueue_lock = threading.Lock()
        # Held for a whole chat turn: session lookup, Claude call and session save.
        self._chat_lock = threading.Lock()
        # Held while opening the usage-limit gate, so simultaneous limit hits send one notice.
        self._limit_lock = threading.Lock()

    # ----------------------------------------------------------- configuration

    @property
    def config(self) -> TaskConfig:
        """The task configuration currently in force.

        Returns:
            The loaded TaskConfig.

        Raises:
            RuntimeError: If ``load_config`` has not been called yet.
        """
        if self._config is None:
            raise RuntimeError("Task configuration is not loaded yet. Call load_config() first.")
        return self._config

    def load_config(self) -> TaskConfig:
        """Load the configuration for the first time (or force a reload).

        Returns:
            The new TaskConfig, which is now in force.

        Raises:
            ConfigError: If the configuration is invalid; startup should fail loudly.
        """
        # The mtime is read before parsing: a save during the parse triggers another reload.
        self._tasks_mtime_ns = self._read_tasks_mtime()
        self._config = self._config_loader()
        self._last_reload_error = None
        return self._config

    def reload_if_changed(self) -> bool:
        """Apply ``tasks.toml`` edits when the file's modification time changed.

        An invalid edit keeps the previous tasks and sends one warning per
        distinct error message.

        Returns:
            True only when a new configuration was applied.
        """
        mtime_ns = self._read_tasks_mtime()
        if self._tasks_file is None or mtime_ns == self._tasks_mtime_ns:
            return False
        # Remember the new mtime even on failure, so a broken file is parsed once, not every tick.
        self._tasks_mtime_ns = mtime_ns
        try:
            config = self._config_loader()
        except ConfigError as err:
            self._report_reload_error(str(err))
            return False
        self._config = config
        self._last_reload_error = None
        logger.info("Reloaded %d tasks", len(config.tasks))
        return True

    def _read_tasks_mtime(self) -> int | None:
        """Read the modification time of the watched file.

        Returns:
            ``st_mtime_ns``, or None when no file is watched or it does not exist.
        """
        if self._tasks_file is None:
            return None
        try:
            return self._tasks_file.stat().st_mtime_ns
        except OSError:
            return None

    def _report_reload_error(self, message: str) -> None:
        """Log a failed reload and warn the user once per distinct message.

        Args:
            message: The ConfigError text.
        """
        logger.warning("Keeping the previous tasks, tasks.toml is invalid: %s", message)
        if message == self._last_reload_error:
            return
        self._last_reload_error = message
        self._notify_text(RELOAD_WARNING.format(message=message))

    # ---------------------------------------------------------------- enqueue

    def enqueue_task(
        self,
        name: str,
        trigger: str,
        *,
        scheduled_for: datetime | None = None,
        retry_of: int | None = None,
        dispatch: bool = True,
    ) -> int | None:
        """Queue a run of a configured task, using its current prompt.

        Args:
            name: Task name.
            trigger: One of the TRIGGER_* constants.
            scheduled_for: The schedule slot this run serves, if any.
            retry_of: Id of the rate-limited run this one retries.
            dispatch: Hand the run to the workers. False leaves it queued in the
                store for the caller to execute (the CLI does this).

        Returns:
            The new run id, or None when the task already has an active run.

        Raises:
            TaskNotFoundError: If ``name`` is not a configured task.
        """
        config = self.config
        task = config.tasks.get(name)
        if task is None:
            raise TaskNotFoundError(name, config.tasks)
        return self._create_run(
            task_name=name,
            trigger=trigger,
            prompt=task.prompt,
            scheduled_for=scheduled_for,
            retry_of=retry_of,
            dispatch=dispatch,
            dedupe=True,
        )

    def enqueue_prompt(
        self,
        prompt: str,
        trigger: str = TRIGGER_CHAT,
        *,
        retry_of: int | None = None,
        dispatch: bool = True,
    ) -> int:
        """Queue a free-text prompt (chat turn or one-off ask).

        Args:
            prompt: The text to send to Claude.
            trigger: TRIGGER_CHAT (continues the conversation) or TRIGGER_MANUAL (one-off).
            retry_of: Id of the rate-limited run this one retries.
            dispatch: Hand the run to the workers (see ``enqueue_task``).

        Returns:
            The new run id. Prompts are never deduplicated.

        Raises:
            ValueError: If the prompt is empty or only whitespace.
        """
        text = prompt.strip()
        if not text:
            raise ValueError("The prompt is empty. Write what you want to ask Claude.")
        run_id = self._create_run(
            task_name=ASK_TASK_NAME,
            trigger=trigger,
            prompt=text,
            scheduled_for=None,
            retry_of=retry_of,
            dispatch=dispatch,
            dedupe=False,
        )
        # Without dedupe a run is always created.
        assert run_id is not None
        return run_id

    def _create_run(
        self,
        *,
        task_name: str,
        trigger: str,
        prompt: str,
        scheduled_for: datetime | None,
        retry_of: int | None,
        dispatch: bool,
        dedupe: bool,
    ) -> int | None:
        """Create a queued run record and optionally hand it to the workers.

        Args:
            task_name: Task name or ASK_TASK_NAME.
            trigger: One of the TRIGGER_* constants.
            prompt: Prompt text stored with the run.
            scheduled_for: The schedule slot served, if any.
            retry_of: Id of the run this one retries.
            dispatch: Put the id on the worker queue.
            dedupe: Refuse to create the run when the task already has an active one.

        Returns:
            The new run id, or None when deduplicated.
        """
        with self._enqueue_lock:
            if dedupe and any(r.task_name == task_name for r in self._store.active_runs()):
                logger.info("Task '%s' is already queued or running, not queuing it again", task_name)
                return None
            run_id = self._store.create_run(
                task_name=task_name,
                trigger=trigger,
                prompt=prompt,
                queued_at=self._clock(),
                scheduled_for=scheduled_for,
                retry_of=retry_of,
            )
        if dispatch:
            self._queue.put(run_id)
        return run_id

    def reset_chat(self) -> None:
        """Start a new conversation (the user's /new).

        Forgets the stored session and records the reset time. A chat turn that
        was queued at or before that time (it may still be waiting or running)
        belongs to the old conversation, so it never stores its session and
        cannot bring the old conversation back.
        """
        self._clear_chat_session()
        self._store.set_value(CHAT_RESET_KEY, self._clock().isoformat())

    def _clear_chat_session(self) -> None:
        """Delete the stored session and its timestamp, without marking a user reset."""
        self._store.set_value(CHAT_SESSION_KEY, None)
        self._store.set_value(CHAT_UPDATED_KEY, None)

    def _was_chat_reset_since(self, queued_at: datetime) -> bool:
        """Tell whether the user reset the conversation after a chat turn was queued.

        The comparison is strict, so a turn queued at the very instant of the
        reset (a frozen test clock, or two events in one microsecond) counts as
        belonging to the new conversation.

        Args:
            queued_at: When the turn was queued.

        Returns:
            True when a reset happened after ``queued_at``.
        """
        raw = self._store.get_value(CHAT_RESET_KEY)
        if raw is None:
            return False
        try:
            reset_at = datetime.fromisoformat(raw)
        except ValueError:
            return False
        if reset_at.utcoffset() is None:
            return False
        return to_utc(queued_at) < to_utc(reset_at)

    # ------------------------------------------------------------------- tick

    def tick(self) -> list[int]:
        """Reload the configuration, clear abandoned runs, queue due runs and due retries.

        An exception while handling one task or retry is logged and does not
        stop the others.

        Returns:
            Ids of the runs created by this tick.
        """
        now = self._clock()
        self.reload_if_changed()
        # Before scheduling, so a task freed by the sweep can be queued in this same tick.
        self._finish_stale_runs(now)
        new_ids: list[int] = []
        for name in list(self.config.tasks):
            try:
                run_id = self._tick_task(name, now)
            # Deliberately broad: a bug in one task's schedule must not stop the other tasks.
            except Exception:
                logger.exception("Could not evaluate the schedule of task '%s'", name)
                continue
            if run_id is not None:
                new_ids.append(run_id)
        new_ids.extend(self._queue_due_retries(now))
        return new_ids

    def _finish_stale_runs(self, now: datetime) -> None:
        """Finish runs stuck in ``running`` long past their timeout as interrupted.

        A process killed without cleanup (``kill -9`` on a CLI run) leaves its
        row ``running``, which would block the task through deduplication. The
        runner enforces each run's timeout, so a run older than the timeout
        plus STALE_RUN_GRACE cannot still be alive.

        Args:
            now: The current moment.
        """
        for record in self._store.active_runs():
            if record.status != STATUS_RUNNING or record.started_at is None:
                continue
            limit = timedelta(seconds=self._timeout_seconds(record)) + STALE_RUN_GRACE
            if to_utc(now) - to_utc(record.started_at) > limit:
                logger.warning("Run #%d (%s) has been running too long, marking it interrupted",
                               record.id, record.task_name)
                self._store.finish_run(
                    record.id, status=STATUS_ERROR, finished_at=now, error=INTERRUPTED_MESSAGE
                )

    def _timeout_seconds(self, record: RunRecord) -> int:
        """Look up the timeout a run was started with, from the current configuration.

        Args:
            record: A run.

        Returns:
            The ask timeout for ask-name runs, the task's timeout for task runs, or
            DEFAULT_TIMEOUT_SECONDS when the task no longer exists.
        """
        options = self._options_for(record)
        return DEFAULT_TIMEOUT_SECONDS if options is None else options.timeout_seconds

    def _tick_task(self, name: str, now: datetime) -> int | None:
        """Evaluate one task's schedule and queue its due slot.

        Args:
            name: Task name.
            now: The current moment.

        Returns:
            The new run id, or None when nothing was queued.
        """
        task = self.config.tasks[name]
        skip = not task.enabled or self._store.is_paused(name)
        last_scheduled = self._store.get_last_scheduled(name)
        due, marker = evaluate(task, last_scheduled, now, skip=skip)
        # The marker is stored before queuing: a crash in between loses one slot
        # instead of running it twice.
        if marker is not None:
            self._store.set_last_scheduled(name, marker)
        if due is not None:
            return self.enqueue_task(name, TRIGGER_SCHEDULE, scheduled_for=due.scheduled_for)
        if marker is not None and not skip and last_scheduled is not None:
            logger.info(
                "Skipped the missed slot of task '%s' at %s (too late to run)",
                name,
                marker.astimezone(now.tzinfo).strftime(LOG_SLOT_TIME_FORMAT),
            )
        return None

    def _queue_due_retries(self, now: datetime) -> list[int]:
        """Queue a copy of every rate-limited run whose retry time has come.

        Args:
            now: The current moment.

        Returns:
            Ids of the copies created.
        """
        new_ids: list[int] = []
        for original in self._store.due_retries(now):
            try:
                run_id = self._queue_retry(original)
            # Deliberately broad: a failure on one retry must not stop the other retries.
            except Exception:
                logger.exception("Could not queue the retry of run #%d", original.id)
                continue
            if run_id is not None:
                new_ids.append(run_id)
        return new_ids

    def _queue_retry(self, original: RunRecord) -> int | None:
        """Queue one retry and mark the original as handled.

        Args:
            original: A rate-limited run whose retry is due.

        Returns:
            The new run id, or None when the retry is not needed.
        """
        # Cleared first so the retry can never be queued twice.
        self._store.clear_retry(original.id)
        is_task_run = original.task_name != ASK_TASK_NAME
        if is_task_run and original.task_name not in self.config.tasks:
            logger.info("Not retrying run #%d: task '%s' no longer exists", original.id, original.task_name)
            return None
        # The retry keeps the original prompt, trigger and task; task runs stay deduplicated.
        run_id = self._create_run(
            task_name=original.task_name,
            trigger=original.trigger,
            prompt=original.prompt,
            scheduled_for=original.scheduled_for,
            retry_of=original.id,
            dispatch=True,
            dedupe=is_task_run,
        )
        if run_id is not None:
            logger.info("Queued run #%d as the retry of run #%d", run_id, original.id)
        return run_id

    # ------------------------------------------------------------ information

    def next_fire_times(self) -> dict[str, datetime | None]:
        """Compute the next scheduled run of every task.

        Returns:
            Task name to next fire time (in the clock's time zone). None for
            manual-only, disabled and paused tasks.
        """
        now = self._clock()
        times: dict[str, datetime | None] = {}
        for name, task in self.config.tasks.items():
            if not task.enabled or not task.schedules or self._store.is_paused(name):
                times[name] = None
                continue
            marker = self._store.get_last_scheduled(name)
            start = now if marker is None else max(now, marker, key=to_utc)
            # Fixed-offset store datetimes must become zoned first, or DST is ignored.
            times[name] = next_fire_time(task, start.astimezone(now.tzinfo))
        return times

    def snapshot(self) -> QueueSnapshot:
        """Describe what is running, what is waiting and whether the usage-limit gate is closed.

        Returns:
            Active runs split by status (oldest first) plus ``limited_until``.
        """
        active = self._store.active_runs()
        return QueueSnapshot(
            running=tuple(r for r in active if r.status == STATUS_RUNNING),
            queued=tuple(r for r in active if r.status == STATUS_QUEUED),
            limited_until=self._limited_until(self._clock()),
        )

    def _limited_until(self, now: datetime) -> datetime | None:
        """Read the usage-limit gate.

        Args:
            now: The current moment.

        Returns:
            The reset time while it lies in the future, else None (also for unreadable values).
        """
        raw = self._store.get_value(LIMITED_UNTIL_KEY)
        if raw is None:
            return None
        try:
            until = datetime.fromisoformat(raw)
        except ValueError:
            logger.warning("Ignoring unreadable %s value %r", LIMITED_UNTIL_KEY, raw)
            return None
        if until.utcoffset() is None or to_utc(until) <= to_utc(now):
            return None
        return until

    # -------------------------------------------------------------- execution

    def execute_run(self, run_id: int) -> RunRecord | None:
        """Execute one queued run to completion and record the outcome.

        Args:
            run_id: Id of a run created by ``enqueue_*``.

        Returns:
            The record: finished if this call executed it, unchanged if the run
            was not queued, or None when the id is unknown.
        """
        record = self._store.get_run(run_id)
        if record is None or record.status != STATUS_QUEUED:
            return record
        options = self._options_for(record)
        if options is None:
            return self._store.finish_run(
                run_id,
                status=STATUS_CANCELLED,
                finished_at=self._clock(),
                error=REMOVED_TASK_MESSAGE,
            )
        if record.trigger == TRIGGER_CHAT:
            # Turns resolve and save the shared session, so they must not overlap.
            with self._chat_lock:
                return self._execute(record, options)
        return self._execute(record, options)

    def _options_for(self, record: RunRecord) -> RunOptions | None:
        """Pick the run options for a record from the current configuration.

        Args:
            record: The queued run.

        Returns:
            The ask options for ask-name runs, the task's current options for
            task runs, or None when the task no longer exists.
        """
        config = self.config
        if record.task_name == ASK_TASK_NAME:
            return config.ask_options
        task = config.tasks.get(record.task_name)
        return None if task is None else task.options

    def _execute(self, record: RunRecord, options: RunOptions) -> RunRecord:
        """Run one record through the gate, the runner and the bookkeeping.

        Args:
            record: The queued run.
            options: Options for the Claude call.

        Returns:
            The finished record.
        """
        now = self._clock()
        limited_until = self._limited_until(now)
        if limited_until is not None:
            # Deferred without calling Claude; it retries when the limit resets.
            finished = self._store.finish_run(
                record.id,
                status=STATUS_RATE_LIMITED,
                finished_at=now,
                error=LIMIT_WAIT_MESSAGE.format(time=_clock_time(limited_until, now)),
                retry_at=limited_until,
            )
            logger.info("Run #%d deferred until the usage limit resets", record.id)
            return finished
        is_chat = record.trigger == TRIGGER_CHAT
        # A turn queued before the latest /new (or retried from one) belongs to the
        # old conversation: it runs fresh, so it can neither continue nor break the new one.
        is_stale_chat = is_chat and self._was_chat_reset_since(self._first_queued_at(record))
        resume_session = self._chat_session(now) if is_chat and not is_stale_chat else None
        self._store.mark_running(record.id, now)
        outcome = self._call_runner(record, options, resume_session)
        finished_at = self._clock()
        if self._stopping.is_set():
            return self._finish_interrupted(record, finished_at)
        if is_chat:
            outcome = self._apply_chat_outcome(
                record, outcome, resumed=resume_session is not None, now=finished_at
            )
        if outcome.status == STATUS_RATE_LIMITED:
            return self._finish_rate_limited(record, outcome, finished_at)
        finished = self._store_outcome(record.id, outcome, finished_at)
        self._notify_run(finished)
        return finished

    def _finish_interrupted(self, record: RunRecord, now: datetime) -> RunRecord:
        """Record a run whose Claude call ended because the service is stopping.

        Whatever the call returned is the product of stop() killing the process
        (for example "exited with code -15"), so it is neither reported to the
        user nor allowed to change the chat session.

        Args:
            record: The run being executed.
            now: The current moment.

        Returns:
            The finished record, with status error and the interrupted message.
        """
        logger.info("Run #%d was interrupted by the service stopping", record.id)
        return self._store.finish_run(
            record.id, status=STATUS_ERROR, finished_at=now, error=INTERRUPTED_MESSAGE
        )

    def _call_runner(
        self, record: RunRecord, options: RunOptions, resume_session: str | None
    ) -> RunOutcome:
        """Call the runner, turning an unexpected exception into an error outcome.

        A BaseException (shutdown signal) finishes the run as interrupted and is re-raised.

        Args:
            record: The run being executed.
            options: Options for the Claude call.
            resume_session: Session id to resume, or None.

        Returns:
            The runner's outcome, or an error outcome carrying the exception message.

        Raises:
            BaseException: KeyboardInterrupt, SystemExit and the like, after the run is recorded.
        """
        try:
            return self._runner.run(record.prompt, options, resume_session)
        # Deliberately broad: any runner bug becomes an error run instead of killing the worker.
        except Exception as err:
            logger.exception("Run #%d crashed", record.id)
            return RunOutcome(status=STATUS_ERROR, error=str(err) or type(err).__name__)
        except BaseException:
            # KeyboardInterrupt or SystemExit (Ctrl-C, SIGTERM): record the run as interrupted so it
            # does not stay "running" and block its task, then let the shutdown continue.
            self._finish_interrupted(record, self._clock())
            raise

    def _store_outcome(
        self, run_id: int, outcome: RunOutcome, finished_at: datetime, *, retry_at: datetime | None = None
    ) -> RunRecord:
        """Persist an outcome as the final state of a run.

        Args:
            run_id: The run to finish.
            outcome: What the runner reported.
            finished_at: When the run ended.
            retry_at: When to retry (rate-limited runs only).

        Returns:
            The stored record.
        """
        return self._store.finish_run(
            run_id,
            status=outcome.status,
            finished_at=finished_at,
            result=outcome.result,
            error=outcome.error,
            duration_ms=outcome.duration_ms,
            session_id=outcome.session_id,
            num_turns=outcome.num_turns,
            cost_usd=outcome.cost_usd,
            denied_tools=outcome.denied_tools,
            retry_at=retry_at,
        )

    # ------------------------------------------------------------------- chat

    def _chat_session(self, now: datetime) -> str | None:
        """Look up the session to resume, dropping it when the conversation went idle.

        Args:
            now: The current moment.

        Returns:
            The stored session id, or None when there is none or it is too old.
        """
        session = self._store.get_value(CHAT_SESSION_KEY)
        if session is None:
            return None
        updated_raw = self._store.get_value(CHAT_UPDATED_KEY)
        try:
            updated = None if updated_raw is None else datetime.fromisoformat(updated_raw)
        except ValueError:
            updated = None
        if updated is None or updated.utcoffset() is None or to_utc(now) - to_utc(updated) > CHAT_IDLE_RESET:
            # An expired session is dropped quietly; it is not a user reset.
            self._clear_chat_session()
            return None
        return session

    def _first_queued_at(self, record: RunRecord) -> datetime:
        """Find when the first run of a retry chain was queued.

        A retry is a copy with a fresh ``queued_at``; the conversation it
        belongs to is the one that was current when the original was queued.

        Args:
            record: A run, possibly a retry.

        Returns:
            ``queued_at`` of the root of the ``retry_of`` chain (the record's own if it has none).
        """
        queued_at = record.queued_at
        seen = {record.id}
        parent_id = record.retry_of
        # ``seen`` guards against a corrupt cycle in the chain.
        while parent_id is not None and parent_id not in seen:
            seen.add(parent_id)
            parent = self._store.get_run(parent_id)
            if parent is None:
                break
            queued_at = parent.queued_at
            parent_id = parent.retry_of
        return queued_at

    def _apply_chat_outcome(
        self, record: RunRecord, outcome: RunOutcome, *, resumed: bool, now: datetime
    ) -> RunOutcome:
        """Update the stored conversation after a chat turn.

        A success with a session id becomes the new current session, unless the
        user started a new conversation after this turn was queued. A failed
        resumed turn discards the conversation, because its session may be broken.

        Args:
            record: The chat run that just finished.
            outcome: The runner's outcome for the turn.
            resumed: Whether the turn resumed an existing session.
            now: The current moment, stored as the conversation's last activity.

        Returns:
            The outcome, with the reset note appended to its error when the
            conversation was reset.
        """
        if outcome.status == STATUS_SUCCESS and outcome.session_id:
            if not self._was_chat_reset_since(self._first_queued_at(record)):
                self._store.set_value(CHAT_SESSION_KEY, outcome.session_id)
                self._store.set_value(CHAT_UPDATED_KEY, now.isoformat())
        elif outcome.status == STATUS_ERROR and resumed:
            self._clear_chat_session()
            error = f"{outcome.error} {CHAT_RESET_NOTE}" if outcome.error else CHAT_RESET_NOTE
            return dataclasses.replace(outcome, error=error)
        return outcome

    # ------------------------------------------------------------- rate limit

    def _finish_rate_limited(self, record: RunRecord, outcome: RunOutcome, now: datetime) -> RunRecord:
        """Record a run that hit the usage limit and close the gate, or give up.

        Args:
            record: The run that was limited.
            outcome: The rate_limited outcome from the runner.
            now: The current moment.

        Returns:
            The finished record.
        """
        if self._started_retries(record) >= MAX_RATE_LIMIT_RETRIES:
            giving_up = dataclasses.replace(
                outcome, error=GIVE_UP_MESSAGE.format(retries=MAX_RATE_LIMIT_RETRIES)
            )
            finished = self._store_outcome(record.id, giving_up, now)
            self._notify_run(finished)
            return finished
        retry_at = outcome.retry_at or now + DEFAULT_RETRY_DELAY
        finished = self._store_outcome(record.id, outcome, now, retry_at=retry_at)
        self._close_gate(retry_at, now)
        return finished

    def _started_retries(self, record: RunRecord) -> int:
        """Count the earlier attempts in a retry chain that actually called Claude.

        Runs deferred by the closed gate never started, so they must not use up
        the retries a run is allowed.

        Args:
            record: The run being finished.

        Returns:
            How many runs above it in the ``retry_of`` chain have a start time.
        """
        started = 0
        seen = {record.id}
        parent_id = record.retry_of
        # ``seen`` guards against a corrupt cycle in the chain.
        while parent_id is not None and parent_id not in seen:
            seen.add(parent_id)
            parent = self._store.get_run(parent_id)
            if parent is None:
                break
            if parent.started_at is not None:
                started += 1
            parent_id = parent.retry_of
        return started

    def _close_gate(self, until: datetime, now: datetime) -> None:
        """Hold every run until the usage limit resets and tell the user once.

        Args:
            until: When the limit resets.
            now: The current moment.
        """
        with self._limit_lock:
            already_closed = self._limited_until(now) is not None
            self._store.set_value(LIMITED_UNTIL_KEY, until.isoformat())
            if not already_closed:
                self._notify_text(LIMIT_NOTICE.format(time=_clock_time(until, now)))

    # ---------------------------------------------------------- notifications

    def _notify_run(self, record: RunRecord) -> None:
        """Send a finished run to the user if its notify policy asks for it.

        Ask-name runs (chat and one-off questions) always notify; task runs
        follow the task's ``notify`` setting. Notifier errors are logged only.

        Args:
            record: The finished run.
        """
        if not self._should_notify(record):
            return
        try:
            self._notifier.notify_run(record)
        # Deliberately broad: any notifier failure is logged, never raised into the engine.
        except Exception:
            logger.exception("Could not notify about run #%d", record.id)

    def _should_notify(self, record: RunRecord) -> bool:
        """Apply the notify policy to a finished run.

        Args:
            record: The finished run.

        Returns:
            True when the user should be told.
        """
        if record.task_name == ASK_TASK_NAME:
            return True
        task = self.config.tasks.get(record.task_name)
        # A task removed from the file mid-run still reports its result.
        policy = NOTIFY_ALWAYS if task is None else task.notify
        if policy == NOTIFY_NEVER:
            return False
        return policy != NOTIFY_ERRORS or record.status != STATUS_SUCCESS

    def _notify_text(self, text: str) -> None:
        """Send a plain notice, logging any delivery error.

        Args:
            text: The notice text.
        """
        try:
            self._notifier.notify_text(text)
        # Deliberately broad: any notifier failure is logged, never raised into the engine.
        except Exception:
            logger.exception("Could not send a notice")

    # ---------------------------------------------------------------- workers

    def start_workers(self) -> None:
        """Start the worker threads that execute queued runs; does nothing if already started."""
        if self._workers:
            return
        self._stopping.clear()
        for index in range(self._max_parallel):
            worker = threading.Thread(target=self._worker_loop, name=f"engine-worker-{index}", daemon=True)
            worker.start()
            self._workers.append(worker)

    def _worker_loop(self) -> None:
        """Take run ids from the queue and execute them until told to stop."""
        while True:
            run_id = self._queue.get()
            if run_id is None or self._stopping.is_set():
                return
            try:
                self.execute_run(run_id)
            # Deliberately broad: the worker thread must keep serving later runs whatever fails.
            except Exception:
                logger.exception("Worker failed on run #%d", run_id)

    def stop(self) -> None:
        """Ask the workers to exit, end in-flight Claude processes and wait briefly.

        Queued runs stay in the store; the next start marks them as interrupted.
        Runs whose Claude call returns after this point are recorded as
        interrupted, without a notification.
        A runner that offers ``terminate_all`` (ClaudeRunner does) is asked to
        kill the Claude processes still running, so none outlives the service.
        """
        self._stopping.set()
        for _ in self._workers:
            self._queue.put(None)
        # Optional runner capability: the Runner protocol only requires run().
        terminate_all = getattr(self._runner, "terminate_all", None)
        if callable(terminate_all):
            terminated = terminate_all()
            if terminated:
                logger.info("Terminated %d running Claude process(es)", terminated)
        deadline = time.monotonic() + WORKER_JOIN_SECONDS
        for worker in self._workers:
            worker.join(timeout=max(0.0, deadline - time.monotonic()))
        self._workers = []

    def recover_interrupted(self) -> int:
        """Mark runs left queued or running by a previous process as failed.

        Call once at startup, before anything can queue a run (the bot thread
        included), or fresh runs would be marked interrupted too.

        Returns:
            How many runs were recovered.
        """
        recovered = self._store.recover_interrupted(self._clock())
        if recovered:
            logger.warning("Marked %d interrupted run(s) from the previous session as failed", recovered)
        return recovered

    def run_forever(self, stop: threading.Event) -> None:
        """Run the service loop until ``stop`` is set.

        Starts the workers, then ticks. Call ``recover_interrupted`` first.
        Between ticks it sleeps until the next schedule
        fires, but never longer than TICK_SECONDS (so edits to tasks.toml are
        picked up) and never shorter than MIN_WAIT_SECONDS.

        Args:
            stop: Event set by the caller (for example a signal handler) to end the loop.
        """
        self.start_workers()
        try:
            while not stop.is_set():
                wait = float(TICK_SECONDS)
                try:
                    self.tick()
                    wait = self._seconds_until_wake()
                # Deliberately broad: a failed tick is logged and retried at the next one.
                except Exception:
                    logger.exception("Tick failed")
                stop.wait(wait)
        finally:
            self.stop()

    def _seconds_until_wake(self) -> float:
        """Compute how long the service loop may sleep before the next tick.

        Returns:
            Seconds until the earliest next fire time, limited to
            MIN_WAIT_SECONDS..TICK_SECONDS.
        """
        now = self._clock()
        upcoming = [moment for moment in self.next_fire_times().values() if moment is not None]
        if not upcoming:
            return float(TICK_SECONDS)
        seconds = (to_utc(min(upcoming, key=to_utc)) - to_utc(now)).total_seconds()
        return max(float(MIN_WAIT_SECONDS), min(float(TICK_SECONDS), seconds))
