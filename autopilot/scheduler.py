"""Due-slot decisions for scheduled tasks (pure, no I/O).

Given a task, the last schedule slot already handled and the current moment,
this module answers one question: "does this task have to run now?". The
engine calls ``evaluate`` once per task on every tick and acts on the answer;
nothing here touches the database, the clock or the network.

Used by: engine.py (tick loop, next fire times).
Depends on: models.py (Task, DueRun) and schedule.py (Schedule.next_after).

Invariants:
    * All datetimes are timezone-aware; naive input raises ValueError.
    * Slots are searched in the time zone of ``now``. Datetimes read back from
      the store carry a fixed UTC offset, and walking the calendar in a fixed
      offset would ignore DST changes, so they are converted first.
    * Datetimes are ordered and subtracted in UTC, never by wall clock.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from autopilot.models import DueRun, Task

# A slot at most this late still runs (for example right after a service restart).
ON_TIME_GRACE = timedelta(minutes=5)
# Upper bound on slots examined per schedule, so an "every minute" task left
# unattended for years cannot stall a tick.
MAX_SLOT_SCAN = 100_000
# latest_slot looks at this much recent time first: nearly every tick finds its slot there,
# which keeps dense schedules cheap even after a long outage.
RECENT_WINDOW = timedelta(hours=24)


def _require_aware(moment: datetime, name: str) -> None:
    """Reject a naive datetime, because it would silently be read as system local time.

    Args:
        moment: The datetime to check.
        name: Parameter name used in the error message.

    Raises:
        ValueError: If ``moment`` has no UTC offset.
    """
    if moment.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware, got naive {moment!r}")


def to_utc(moment: datetime) -> datetime:
    """Return ``moment`` as a UTC datetime, for ordering and subtraction.

    The one shared helper for UTC comparisons (engine.py imports it): two
    datetimes with the same ZoneInfo compare by wall clock, which is wrong
    across a DST change.

    Args:
        moment: An aware datetime.

    Returns:
        The same instant with tzinfo UTC.
    """
    return moment.astimezone(timezone.utc)


def next_fire_time(task: Task, after: datetime) -> datetime | None:
    """Find the earliest moment any of the task's schedules fires after ``after``.

    Args:
        task: The task to inspect.
        after: Search start (aware); the result is in ``after``'s time zone.

    Returns:
        The earliest fire time strictly after ``after``, or None when the task
        has no schedules (manual only).

    Raises:
        ValueError: If ``after`` is naive.
        ScheduleError: If a schedule never fires.

    Example:
        next_fire_time(task, datetime(2026, 11, 2, 7, 30, tzinfo=TZ))
    """
    _require_aware(after, "after")
    candidates = [schedule.next_after(after) for schedule in task.schedules]
    if not candidates:
        return None
    return min(candidates, key=to_utc)


def latest_slot(task: Task, after: datetime, now: datetime) -> datetime | None:
    """Find the last schedule slot ``s`` with ``after < s <= now``.

    The last RECENT_WINDOW before ``now`` is scanned first. Only when it holds
    no slot is the whole range from ``after`` scanned, so a dense schedule
    (every minute) over a long gap costs about a day of steps, not the gap.

    Args:
        task: The task whose schedules are examined.
        after: Exclusive lower bound, normally the last handled slot (aware).
        now: Inclusive upper bound (aware); also sets the time zone slots are
            computed in.

    Returns:
        The latest slot in ``now``'s time zone, or None when no slot falls in
        the window. Each schedule is examined for at most MAX_SLOT_SCAN slots
        per scan; if that limit is hit, the last slot found so far counts.

    Raises:
        ValueError: If ``after`` or ``now`` is naive.
    """
    _require_aware(after, "after")
    _require_aware(now, "now")
    # Fixed-offset datetimes from the store would make next_after ignore DST.
    start = after.astimezone(now.tzinfo)
    recent_start = now - RECENT_WINDOW
    if to_utc(recent_start) > to_utc(start):
        # The latest slot overall is also the latest in the recent window, if the window has any.
        recent = _latest_after(task, recent_start, now)
        if recent is not None:
            return recent
    return _latest_after(task, start, now)


def _latest_after(task: Task, start: datetime, now: datetime) -> datetime | None:
    """Scan every schedule from ``start`` and return the latest slot not after ``now``.

    Args:
        task: The task whose schedules are examined.
        start: Exclusive lower bound, already in ``now``'s time zone.
        now: Inclusive upper bound (aware).

    Returns:
        The latest slot found, or None.
    """
    now_utc = to_utc(now)
    latest: datetime | None = None
    for schedule in task.schedules:
        cursor = start
        for _ in range(MAX_SLOT_SCAN):
            slot = schedule.next_after(cursor)
            if to_utc(slot) > now_utc:
                break
            if latest is None or to_utc(slot) > to_utc(latest):
                latest = slot
            cursor = slot
    return latest


def evaluate(
    task: Task,
    last_scheduled: datetime | None,
    now: datetime,
    *,
    skip: bool = False,
    grace: timedelta = ON_TIME_GRACE,
) -> tuple[DueRun | None, datetime | None]:
    """Decide whether a task must be queued now.

    Args:
        task: The task to evaluate.
        last_scheduled: The last slot already handled, or None for a task the
            engine has not seen before.
        now: The current moment (aware).
        skip: True for disabled or paused tasks: their slots are dropped, not queued.
        grace: How late a slot may be and still run.

    Returns:
        ``(run, marker)``. ``run`` is the DueRun to enqueue, or None. ``marker``
        is the new last-scheduled value to store, or None to leave it unchanged.
        A slot that is skipped or missed still advances the marker, so it is
        never examined again and never runs in a burst later.

    Raises:
        ValueError: If ``last_scheduled`` or ``now`` is naive.
    """
    _require_aware(now, "now")
    if not task.schedules:
        return None, None
    if last_scheduled is None:
        # A new task waits for its next slot instead of running immediately.
        return None, now
    latest = latest_slot(task, last_scheduled, now)
    if latest is None:
        return None, None
    if skip:
        return None, latest
    if to_utc(now) - to_utc(latest) <= grace:
        return DueRun(task_name=task.name, scheduled_for=latest), latest
    # The service was stopped or the PC was off: skip the slot, never run it late.
    return None, latest
