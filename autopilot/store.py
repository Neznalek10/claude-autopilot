"""SQLite persistence for runs, per-task state and small key/value settings.

The engine, the Telegram bot and the CLI all talk to the database through
``Store``. It is the only module that knows the SQL schema.

Design:
    * One short-lived connection per call, so a ``Store`` can be shared by any
      number of threads and the database by several processes (service + CLI).
    * WAL journal mode with a 10 second busy timeout, so concurrent writers wait
      for each other instead of failing with "database is locked".
    * Datetimes are stored as ISO 8601 text with their UTC offset and come back
      timezone-aware. Naive datetimes are rejected with ValueError.
    * All SQL is parameterized.

Tables: ``runs`` (one row per RunRecord), ``task_state`` (pause flag and last
scheduled slot per task) and ``kv`` (bot offset, chat session, usage-limit gate).
"""

from __future__ import annotations

import logging
import sqlite3
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

from autopilot.models import (
    ACTIVE_STATUSES,
    FINISHED_STATUSES,
    STATUS_ERROR,
    STATUS_QUEUED,
    STATUS_RATE_LIMITED,
    STATUS_RUNNING,
    RunRecord,
)

logger = logging.getLogger(__name__)

BUSY_TIMEOUT_SECONDS = 10
# Prompts, results and chat history can be personal, so the data folder and
# the database stay readable by the owner only, whatever the umask says.
PRIVATE_DIR_MODE = 0o700
PRIVATE_FILE_MODE = 0o600
INTERRUPTED_MESSAGE = "Interrupted: autopilot stopped before this run finished."
DENIED_TOOLS_SEPARATOR = ","

# Column order shared by every INSERT/SELECT on ``runs``; keeps _row_to_record in sync.
RUN_COLUMNS = (
    "id, task_name, trigger, status, prompt, queued_at, scheduled_for, started_at, "
    "finished_at, duration_ms, result, error, session_id, num_turns, cost_usd, "
    "denied_tools, retry_of, retry_at"
)

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    task_name     TEXT NOT NULL,
    trigger       TEXT NOT NULL,
    status        TEXT NOT NULL,
    prompt        TEXT NOT NULL,
    queued_at     TEXT NOT NULL,
    scheduled_for TEXT,
    started_at    TEXT,
    finished_at   TEXT,
    duration_ms   INTEGER,
    result        TEXT NOT NULL DEFAULT '',
    error         TEXT,
    session_id    TEXT,
    num_turns     INTEGER,
    cost_usd      REAL,
    denied_tools  TEXT NOT NULL DEFAULT '',
    retry_of      INTEGER,
    retry_at      TEXT
);
CREATE INDEX IF NOT EXISTS idx_runs_task ON runs (task_name, id);
CREATE INDEX IF NOT EXISTS idx_runs_status ON runs (status);
CREATE TABLE IF NOT EXISTS task_state (
    task_name      TEXT PRIMARY KEY,
    paused         INTEGER NOT NULL DEFAULT 0,
    last_scheduled TEXT
);
CREATE TABLE IF NOT EXISTS kv (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


def _to_text(moment: datetime | None) -> str | None:
    """Serialize an aware datetime to ISO 8601 text, keeping its UTC offset.

    Args:
        moment: The datetime to store, or None.

    Returns:
        ISO text, or None when ``moment`` is None.

    Raises:
        ValueError: If ``moment`` is naive.
    """
    if moment is None:
        return None
    if moment.utcoffset() is None:
        raise ValueError(f"Expected a timezone-aware datetime, got naive {moment!r}")
    return moment.isoformat()


def _from_text(text: str | None) -> datetime | None:
    """Parse ISO 8601 text written by ``_to_text`` back into an aware datetime.

    Args:
        text: Stored text, or None.

    Returns:
        The aware datetime (fixed UTC offset), or None when ``text`` is None.
    """
    return None if text is None else datetime.fromisoformat(text)


def _row_to_record(row: sqlite3.Row) -> RunRecord:
    """Convert a ``runs`` row into a RunRecord.

    Args:
        row: A row selected with RUN_COLUMNS.

    Returns:
        The equivalent frozen RunRecord with aware datetimes.
    """
    denied = row["denied_tools"]
    return RunRecord(
        id=row["id"],
        task_name=row["task_name"],
        trigger=row["trigger"],
        status=row["status"],
        prompt=row["prompt"],
        queued_at=_from_text(row["queued_at"]),
        scheduled_for=_from_text(row["scheduled_for"]),
        started_at=_from_text(row["started_at"]),
        finished_at=_from_text(row["finished_at"]),
        duration_ms=row["duration_ms"],
        result=row["result"],
        error=row["error"],
        session_id=row["session_id"],
        num_turns=row["num_turns"],
        cost_usd=row["cost_usd"],
        # An empty column means "no tools"; split("") would give [""] instead.
        denied_tools=tuple(denied.split(DENIED_TOOLS_SEPARATOR)) if denied else (),
        retry_of=row["retry_of"],
        retry_at=_from_text(row["retry_at"]),
    )


class Store:
    """Thread-safe and process-safe SQLite storage for autopilot.

    Example:
        store = Store(Path("data/autopilot.db"))
        run_id = store.create_run(task_name="news", trigger="manual",
                                  prompt="Hi", queued_at=now)
    """

    def __init__(self, db_path: Path) -> None:
        """Open (and if needed create) the database and its schema.

        Args:
            db_path: Database file. Missing parent folders are created.

        Raises:
            sqlite3.Error: If the file cannot be opened or is not a database.
            OSError: If the folder or file permissions cannot be set.

        Side effects:
            Sets the data folder to mode 0o700 and the database file to 0o600.
        """
        self._db_path = db_path
        db_path.parent.mkdir(mode=PRIVATE_DIR_MODE, parents=True, exist_ok=True)
        # mkdir's mode is filtered by the umask and skipped for an existing folder.
        db_path.parent.chmod(PRIVATE_DIR_MODE)
        with self._connect() as conn:
            # WAL lets readers and one writer work at once; the mode is stored in the file.
            conn.execute("PRAGMA journal_mode=WAL")
            conn.executescript(SCHEMA)
        # SQLite creates the file with the umask; its -wal and -shm files copy this mode.
        db_path.chmod(PRIVATE_FILE_MODE)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        """Open a connection, commit on success, roll back on error, always close.

        Yields:
            A connection whose rows behave like mappings.
        """
        conn = sqlite3.connect(self._db_path, timeout=BUSY_TIMEOUT_SECONDS)
        conn.row_factory = sqlite3.Row
        try:
            with conn:  # commits or rolls back the transaction
                yield conn
        finally:
            conn.close()

    # ------------------------------------------------------------------ runs

    def create_run(
        self,
        *,
        task_name: str,
        trigger: str,
        prompt: str,
        queued_at: datetime,
        scheduled_for: datetime | None = None,
        retry_of: int | None = None,
    ) -> int:
        """Insert a new run with status ``queued``.

        Args:
            task_name: Task the run belongs to (or the reserved ask name).
            trigger: One of the TRIGGER_* constants.
            prompt: The prompt text as it was at queue time.
            queued_at: When the run was queued (aware).
            scheduled_for: The schedule slot this run serves, if any (aware).
            retry_of: Id of the rate-limited run this one retries.

        Returns:
            The new run id.

        Raises:
            ValueError: If a datetime is naive.
        """
        with self._connect() as conn:
            cursor = conn.execute(
                "INSERT INTO runs (task_name, trigger, status, prompt, queued_at, "
                "scheduled_for, retry_of) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    task_name,
                    trigger,
                    STATUS_QUEUED,
                    prompt,
                    _to_text(queued_at),
                    _to_text(scheduled_for),
                    retry_of,
                ),
            )
            return int(cursor.lastrowid)

    def mark_running(self, run_id: int, started_at: datetime) -> None:
        """Set a run to ``running``.

        Args:
            run_id: The run to update.
            started_at: When execution began (aware).

        Raises:
            ValueError: If the run id is unknown or ``started_at`` is naive.
        """
        with self._connect() as conn:
            cursor = conn.execute(
                "UPDATE runs SET status = ?, started_at = ? WHERE id = ?",
                (STATUS_RUNNING, _to_text(started_at), run_id),
            )
            if cursor.rowcount == 0:
                raise ValueError(f"Unknown run id {run_id}.")

    def finish_run(
        self,
        run_id: int,
        *,
        status: str,
        finished_at: datetime,
        result: str = "",
        error: str | None = None,
        duration_ms: int | None = None,
        session_id: str | None = None,
        num_turns: int | None = None,
        cost_usd: float | None = None,
        denied_tools: Sequence[str] = (),
        retry_at: datetime | None = None,
    ) -> RunRecord:
        """Store the outcome of a run and return the updated record.

        Args:
            run_id: The run to finish.
            status: A member of FINISHED_STATUSES.
            finished_at: When the run ended (aware).
            result: Result text.
            error: Error message for failed runs.
            duration_ms: Run duration in milliseconds.
            session_id: Claude session id, if reported.
            num_turns: Number of agent turns, if reported.
            cost_usd: API-equivalent cost, if reported.
            denied_tools: Names of tools Claude was blocked from using.
            retry_at: For rate-limited runs, when to retry (aware).

        Returns:
            The finished RunRecord as stored.

        Raises:
            ValueError: If the status is not a finished status, the run id is
                unknown, or a datetime is naive.
        """
        if status not in FINISHED_STATUSES:
            raise ValueError(
                f"Cannot finish a run with status {status!r}. "
                f"Use one of: {', '.join(sorted(FINISHED_STATUSES))}."
            )
        with self._connect() as conn:
            cursor = conn.execute(
                "UPDATE runs SET status = ?, finished_at = ?, result = ?, error = ?, "
                "duration_ms = ?, session_id = ?, num_turns = ?, cost_usd = ?, "
                "denied_tools = ?, retry_at = ? WHERE id = ?",
                (
                    status,
                    _to_text(finished_at),
                    result,
                    error,
                    duration_ms,
                    session_id,
                    num_turns,
                    cost_usd,
                    DENIED_TOOLS_SEPARATOR.join(denied_tools),
                    _to_text(retry_at),
                    run_id,
                ),
            )
            if cursor.rowcount == 0:
                raise ValueError(f"Unknown run id {run_id}.")
            row = conn.execute(
                f"SELECT {RUN_COLUMNS} FROM runs WHERE id = ?", (run_id,)
            ).fetchone()
        return _row_to_record(row)

    def get_run(self, run_id: int) -> RunRecord | None:
        """Look up one run.

        Args:
            run_id: The run id.

        Returns:
            The record, or None when the id is unknown.
        """
        with self._connect() as conn:
            row = conn.execute(
                f"SELECT {RUN_COLUMNS} FROM runs WHERE id = ?", (run_id,)
            ).fetchone()
        return None if row is None else _row_to_record(row)

    def list_runs(
        self,
        *,
        task_name: str | None = None,
        limit: int = 10,
        finished_only: bool = False,
    ) -> list[RunRecord]:
        """List runs, newest first.

        Args:
            task_name: Only runs of this task; None for all tasks.
            limit: Maximum number of runs to return.
            finished_only: Skip queued and running runs.

        Returns:
            Matching records, newest first.
        """
        clauses: list[str] = []
        params: list[Any] = []
        if task_name is not None:
            clauses.append("task_name = ?")
            params.append(task_name)
        if finished_only:
            # Placeholders are generated; the values themselves are bound parameters.
            placeholders = ", ".join("?" for _ in ACTIVE_STATUSES)
            clauses.append(f"status NOT IN ({placeholders})")
            params.extend(sorted(ACTIVE_STATUSES))
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(limit)
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT {RUN_COLUMNS} FROM runs {where} ORDER BY id DESC LIMIT ?",
                params,
            ).fetchall()
        return [_row_to_record(row) for row in rows]

    def active_runs(self) -> list[RunRecord]:
        """List queued and running runs.

        Returns:
            Active records, oldest first.
        """
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT {RUN_COLUMNS} FROM runs WHERE status IN (?, ?) ORDER BY id",
                (STATUS_QUEUED, STATUS_RUNNING),
            ).fetchall()
        return [_row_to_record(row) for row in rows]

    def recover_interrupted(self, now: datetime) -> int:
        """Fail runs left queued or running by a previous process.

        Args:
            now: Used as the finish time of the recovered runs (aware).

        Returns:
            How many runs were recovered.
        """
        with self._connect() as conn:
            cursor = conn.execute(
                "UPDATE runs SET status = ?, error = ?, finished_at = ? "
                "WHERE status IN (?, ?)",
                (
                    STATUS_ERROR,
                    INTERRUPTED_MESSAGE,
                    _to_text(now),
                    STATUS_QUEUED,
                    STATUS_RUNNING,
                ),
            )
            return cursor.rowcount

    def due_retries(self, now: datetime) -> list[RunRecord]:
        """List rate-limited runs whose retry time has come.

        Args:
            now: The current moment (aware).

        Returns:
            Records with status rate_limited and retry_at <= now, oldest first.
        """
        if now.utcoffset() is None:
            raise ValueError(f"Expected a timezone-aware datetime, got naive {now!r}")
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT {RUN_COLUMNS} FROM runs "
                "WHERE status = ? AND retry_at IS NOT NULL ORDER BY id",
                (STATUS_RATE_LIMITED,),
            ).fetchall()
        # Compared as datetimes here: ISO text with different offsets does not sort by instant.
        records = (_row_to_record(row) for row in rows)
        return [record for record in records if record.retry_at <= now]

    def clear_retry(self, run_id: int) -> None:
        """Remove a run's pending retry so it is not retried twice.

        Args:
            run_id: The rate-limited run.
        """
        with self._connect() as conn:
            conn.execute("UPDATE runs SET retry_at = NULL WHERE id = ?", (run_id,))

    def retry_depth(self, run_id: int) -> int:
        """Count the retry links above a run.

        Args:
            run_id: The run to inspect.

        Returns:
            0 for an original run, 1 for its first retry, and so on.
        """
        depth = 0
        seen = {run_id}
        with self._connect() as conn:
            current: int | None = run_id
            while current is not None:
                row = conn.execute(
                    "SELECT retry_of FROM runs WHERE id = ?", (current,)
                ).fetchone()
                current = None if row is None else row["retry_of"]
                # A cycle cannot be produced by the API, but a corrupt file must not hang us.
                if current is None or current in seen:
                    break
                seen.add(current)
                depth += 1
        return depth

    # ------------------------------------------------------------ task state

    def is_paused(self, task_name: str) -> bool:
        """Tell whether a task's schedule is paused.

        Args:
            task_name: The task name.

        Returns:
            True when paused; False for tasks never paused.
        """
        with self._connect() as conn:
            row = conn.execute(
                "SELECT paused FROM task_state WHERE task_name = ?", (task_name,)
            ).fetchone()
        return row is not None and bool(row["paused"])

    def set_paused(self, task_name: str, paused: bool) -> None:
        """Pause or resume a task's schedule.

        Args:
            task_name: The task name.
            paused: True to pause, False to resume.
        """
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO task_state (task_name, paused) VALUES (?, ?) "
                "ON CONFLICT(task_name) DO UPDATE SET paused = excluded.paused",
                (task_name, int(paused)),
            )

    def get_last_scheduled(self, task_name: str) -> datetime | None:
        """Return the last schedule slot already handled for a task.

        Args:
            task_name: The task name.

        Returns:
            The slot (aware), or None if the task has no marker yet.
        """
        with self._connect() as conn:
            row = conn.execute(
                "SELECT last_scheduled FROM task_state WHERE task_name = ?",
                (task_name,),
            ).fetchone()
        return None if row is None else _from_text(row["last_scheduled"])

    def set_last_scheduled(self, task_name: str, when: datetime) -> None:
        """Record the last schedule slot handled for a task.

        Args:
            task_name: The task name.
            when: The slot or marker moment (aware).

        Raises:
            ValueError: If ``when`` is naive.
        """
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO task_state (task_name, last_scheduled) VALUES (?, ?) "
                "ON CONFLICT(task_name) DO UPDATE SET last_scheduled = excluded.last_scheduled",
                (task_name, _to_text(when)),
            )

    # -------------------------------------------------------------- key/value

    def get_value(self, key: str) -> str | None:
        """Read a small persisted setting.

        Args:
            key: The key, e.g. ``telegram_offset``.

        Returns:
            The value, or None when the key is not set.
        """
        with self._connect() as conn:
            row = conn.execute("SELECT value FROM kv WHERE key = ?", (key,)).fetchone()
        return None if row is None else row["value"]

    def set_value(self, key: str, value: str | None) -> None:
        """Write or delete a small persisted setting.

        Args:
            key: The key.
            value: The new value; None deletes the key.
        """
        with self._connect() as conn:
            if value is None:
                conn.execute("DELETE FROM kv WHERE key = ?", (key,))
                return
            conn.execute(
                "INSERT INTO kv (key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (key, value),
            )
