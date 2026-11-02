"""Unit tests for autopilot.scheduler.

Covers the earliest fire time across schedules, the latest due slot in a
window, and every branch of ``evaluate`` (new task, nothing due, on time,
late, skipped), including DST changes and fixed-offset store datetimes.
"""

from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone
from unittest import mock

from autopilot import scheduler
from autopilot.models import DueRun
from autopilot.schedule import parse_schedule
from autopilot.scheduler import evaluate, latest_slot, next_fire_time
from helpers import TZ, make_task

NOW = datetime(2026, 11, 2, 7, 30, tzinfo=TZ)  # a Monday


def at(hour: int, minute: int, day: int = 2, month: int = 11) -> datetime:
    """Build a 2026 moment in the test time zone.

    Args:
        hour: Hour of day.
        minute: Minute of the hour.
        day: Day of month.
        month: Month number.

    Returns:
        An aware datetime in TZ.
    """
    return datetime(2026, month, day, hour, minute, tzinfo=TZ)


def daily_task(*expressions: str):
    """Build a task with the given schedule expressions.

    Args:
        *expressions: Schedule expressions such as "daily 07:30".

    Returns:
        A Task with those schedules.
    """
    return make_task(schedules=tuple(parse_schedule(e) for e in expressions))


class NextFireTimeTests(unittest.TestCase):
    """next_fire_time picks the earliest fire time over all schedules."""

    def test_returns_none_for_manual_only_task(self) -> None:
        """A task without schedules never fires on its own."""
        self.assertIsNone(next_fire_time(make_task(), NOW))

    def test_single_schedule(self) -> None:
        """One schedule gives its next slot, strictly after the given moment."""
        task = daily_task("daily 07:30")
        self.assertEqual(next_fire_time(task, NOW), at(7, 30, day=3))

    def test_earliest_of_several_schedules(self) -> None:
        """With several schedules the nearest slot wins, whatever their order."""
        task = daily_task("daily 18:00", "daily 09:15", "daily 07:30")
        self.assertEqual(next_fire_time(task, NOW), at(9, 15))

    def test_naive_datetime_is_rejected(self) -> None:
        """A naive datetime raises ValueError instead of being read as local time."""
        with self.assertRaises(ValueError):
            next_fire_time(daily_task("daily 07:30"), datetime(2026, 11, 2, 7, 30))


class LatestSlotTests(unittest.TestCase):
    """latest_slot returns the last slot in the window (after, now]."""

    def test_no_slot_in_window(self) -> None:
        """An empty window yields None."""
        self.assertIsNone(latest_slot(daily_task("daily 07:30"), at(7, 31), at(23, 0)))

    def test_manual_only_task(self) -> None:
        """A task without schedules has no slots."""
        self.assertIsNone(latest_slot(make_task(), at(0, 0), NOW))

    def test_lower_bound_is_exclusive_and_upper_bound_inclusive(self) -> None:
        """A slot equal to ``after`` is excluded; a slot equal to ``now`` is included."""
        task = daily_task("daily 07:30")
        self.assertIsNone(latest_slot(task, NOW, NOW))
        self.assertEqual(latest_slot(task, at(7, 29), NOW), NOW)

    def test_returns_the_last_of_many_slots(self) -> None:
        """When several slots fall in the window only the latest is returned."""
        task = daily_task("every 15m")
        self.assertEqual(latest_slot(task, at(6, 0), at(7, 40)), at(7, 30))

    def test_latest_across_several_schedules(self) -> None:
        """The latest slot is chosen over all schedules, not per schedule."""
        task = daily_task("daily 07:00", "daily 07:20", "daily 07:10")
        self.assertEqual(latest_slot(task, at(6, 0), at(8, 0)), at(7, 20))

    def test_scan_limit_returns_last_slot_found(self) -> None:
        """After MAX_SLOT_SCAN steps the scan stops and keeps the last slot it found."""
        task = daily_task("every 15m")
        with mock.patch.object(scheduler, "MAX_SLOT_SCAN", 3):
            self.assertEqual(latest_slot(task, at(0, 0), at(23, 0)), at(0, 45))

    def test_fixed_offset_start_is_walked_in_the_zone_of_now(self) -> None:
        """A store-style fixed-offset ``after`` must not freeze DST at the old offset.

        Clocks go back on 2026-10-25. The daily 07:30 slot on the 26th is
        07:30 CET even though ``after`` was written with the summer offset.
        """
        task = daily_task("daily 07:30")
        after = datetime(2026, 10, 24, 7, 30, tzinfo=timezone(timedelta(hours=2)))
        now = datetime(2026, 10, 26, 7, 30, tzinfo=TZ)
        self.assertEqual(latest_slot(task, after, now), now)

    def test_naive_arguments_are_rejected(self) -> None:
        """Naive ``after`` or ``now`` raise ValueError."""
        task = daily_task("daily 07:30")
        with self.assertRaises(ValueError):
            latest_slot(task, datetime(2026, 11, 1), NOW)
        with self.assertRaises(ValueError):
            latest_slot(task, at(0, 0), datetime(2026, 11, 2))


class EvaluateTests(unittest.TestCase):
    """evaluate maps (task, marker, now) to (run to enqueue, new marker)."""

    def test_manual_only_task_changes_nothing(self) -> None:
        """Without schedules there is nothing to run and nothing to remember."""
        self.assertEqual(evaluate(make_task(), at(0, 0), NOW), (None, None))

    def test_new_task_waits_for_its_next_slot(self) -> None:
        """A task without a marker is not run; the marker becomes ``now``."""
        task = daily_task("daily 07:30")
        self.assertEqual(evaluate(task, None, NOW), (None, NOW))

    def test_nothing_due_leaves_the_marker(self) -> None:
        """With no slot since the marker, neither a run nor a new marker is returned."""
        task = daily_task("daily 07:30")
        self.assertEqual(evaluate(task, at(7, 30), at(7, 45)), (None, None))

    def test_due_slot_runs_and_moves_the_marker(self) -> None:
        """A slot that has just passed is run and stored as the marker."""
        task = daily_task("daily 07:30")
        run, marker = evaluate(task, at(7, 0), at(7, 31))
        self.assertEqual(run, DueRun(task_name="news", scheduled_for=at(7, 30)))
        self.assertEqual(marker, at(7, 30))

    def test_exactly_at_the_grace_limit_still_runs(self) -> None:
        """A slot exactly 5 minutes late is still on time."""
        task = daily_task("daily 07:30")
        run, _ = evaluate(task, at(7, 0), at(7, 35))
        self.assertIsNotNone(run)

    def test_late_slot_is_skipped_but_remembered(self) -> None:
        """A slot more than the grace late is skipped, yet the marker moves past it."""
        task = daily_task("daily 07:30")
        run, marker = evaluate(task, at(7, 0), at(7, 36))
        self.assertIsNone(run)
        self.assertEqual(marker, at(7, 30))

    def test_custom_grace(self) -> None:
        """The grace period can be widened."""
        task = daily_task("daily 07:30")
        run, _ = evaluate(task, at(7, 0), at(8, 0), grace=timedelta(hours=1))
        self.assertIsNotNone(run)

    def test_many_missed_slots_run_nothing_in_a_burst(self) -> None:
        """After a long outage only one decision is made and no run is created."""
        task = daily_task("every 15m")
        run, marker = evaluate(task, at(0, 0), at(12, 7))
        self.assertIsNone(run)
        self.assertEqual(marker, at(12, 0))

    def test_skip_drops_due_slots_without_running(self) -> None:
        """Paused or disabled tasks drop their slots: no run, marker advances."""
        task = daily_task("daily 07:30")
        run, marker = evaluate(task, at(7, 0), at(7, 31), skip=True)
        self.assertIsNone(run)
        self.assertEqual(marker, at(7, 30))

    def test_skip_without_due_slot_changes_nothing(self) -> None:
        """A skipped task with no slot in the window keeps its marker."""
        task = daily_task("daily 07:30")
        self.assertEqual(evaluate(task, at(7, 30), at(7, 40), skip=True), (None, None))

    def test_new_task_marker_is_set_even_when_skipped(self) -> None:
        """A paused new task still gets its starting marker, so resuming does not catch up."""
        task = daily_task("daily 07:30")
        self.assertEqual(evaluate(task, None, NOW, skip=True), (None, NOW))

    def test_lateness_is_measured_in_real_time_across_dst(self) -> None:
        """Lateness is a UTC difference, not a wall-clock difference.

        Clocks go back at 03:00 on 2026-10-25, so 02:30 happens twice. The slot
        at 02:30 (first pass, +02:00) is 65 real minutes old at 02:35 on the
        second pass (+01:00), although the wall clocks differ by only 5 minutes.
        """
        task = daily_task("30 2 * * *")
        first_pass = datetime(2026, 10, 25, 2, 30, tzinfo=TZ)
        second_pass_now = datetime(2026, 10, 25, 2, 35, fold=1, tzinfo=TZ)
        run, marker = evaluate(task, first_pass - timedelta(hours=1), second_pass_now)
        self.assertIsNone(run)
        self.assertEqual(marker, first_pass)

    def test_naive_now_is_rejected(self) -> None:
        """A naive ``now`` raises ValueError."""
        with self.assertRaises(ValueError):
            evaluate(daily_task("daily 07:30"), at(7, 0), datetime(2026, 11, 2, 7, 31))


if __name__ == "__main__":
    unittest.main()
