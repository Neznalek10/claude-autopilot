"""Unit tests for autopilot.store.

Covers the run lifecycle, queries, retry bookkeeping, task state, the key/value
table, datetime round trips, unicode text and concurrent writers. Every test
uses its own temporary database folder.
"""

from __future__ import annotations

import tempfile
import threading
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from autopilot.models import (
    STATUS_CANCELLED,
    STATUS_ERROR,
    STATUS_QUEUED,
    STATUS_RATE_LIMITED,
    STATUS_RUNNING,
    STATUS_SUCCESS,
    TRIGGER_MANUAL,
    TRIGGER_SCHEDULE,
)
from autopilot.store import INTERRUPTED_MESSAGE, Store
from helpers import TZ

NOW = datetime(2026, 11, 2, 7, 30, tzinfo=TZ)
UNICODE_TEXT = "Naïve café résumé ☕"


class StoreTestCase(unittest.TestCase):
    """Base class: gives each test a fresh Store in a temporary folder."""

    def setUp(self) -> None:
        """Create the temp folder and an empty store inside a nested path."""
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.db_path = Path(tmp.name) / "nested" / "autopilot.db"
        self.store = Store(self.db_path)

    def new_run(self, task_name: str = "news", **overrides: object) -> int:
        """Create a queued run with default fields.

        Args:
            task_name: Task name for the run.
            **overrides: Extra create_run keyword arguments.

        Returns:
            The new run id.
        """
        fields: dict[str, object] = {
            "task_name": task_name,
            "trigger": TRIGGER_SCHEDULE,
            "prompt": "Say hi",
            "queued_at": NOW,
        }
        fields.update(overrides)
        return self.store.create_run(**fields)  # type: ignore[arg-type]

    def finish(self, run_id: int, status: str = STATUS_SUCCESS, **overrides: object):
        """Finish a run at NOW with the given status.

        Args:
            run_id: Run to finish.
            status: Finished status.
            **overrides: Extra finish_run keyword arguments.

        Returns:
            The finished RunRecord.
        """
        return self.store.finish_run(run_id, status=status, finished_at=NOW, **overrides)  # type: ignore[arg-type]


class RunLifecycleTests(StoreTestCase):
    """Create, start and finish a run."""

    def test_constructor_creates_parent_folders_and_file(self) -> None:
        """A missing database folder is created on construction."""
        self.assertTrue(self.db_path.exists())

    def test_new_run_is_queued_with_given_fields(self) -> None:
        """create_run stores a queued record carrying the provided metadata."""
        slot = NOW - timedelta(minutes=1)
        run_id = self.new_run(scheduled_for=slot)
        record = self.store.get_run(run_id)
        self.assertEqual(record.status, STATUS_QUEUED)
        self.assertEqual(record.task_name, "news")
        self.assertEqual(record.trigger, TRIGGER_SCHEDULE)
        self.assertEqual(record.prompt, "Say hi")
        self.assertEqual(record.queued_at, NOW)
        self.assertEqual(record.scheduled_for, slot)
        self.assertIsNone(record.started_at)
        self.assertEqual(record.denied_tools, ())

    def test_mark_running_sets_status_and_start(self) -> None:
        """mark_running moves the run to running and stores the start time."""
        run_id = self.new_run()
        self.store.mark_running(run_id, NOW)
        record = self.store.get_run(run_id)
        self.assertEqual(record.status, STATUS_RUNNING)
        self.assertEqual(record.started_at, NOW)

    def test_finish_run_returns_and_persists_all_fields(self) -> None:
        """finish_run stores every outcome field and returns the same record."""
        run_id = self.new_run()
        finished = self.finish(
            run_id,
            result="Done",
            error=None,
            duration_ms=1234,
            session_id="abc",
            num_turns=3,
            cost_usd=0.25,
            denied_tools=["Bash", "Write"],
        )
        self.assertEqual(finished, self.store.get_run(run_id))
        self.assertEqual(finished.status, STATUS_SUCCESS)
        self.assertEqual(finished.result, "Done")
        self.assertEqual(finished.duration_ms, 1234)
        self.assertEqual(finished.session_id, "abc")
        self.assertEqual(finished.num_turns, 3)
        self.assertEqual(finished.cost_usd, 0.25)
        self.assertEqual(finished.denied_tools, ("Bash", "Write"))
        self.assertTrue(finished.is_finished)

    def test_finish_run_rejects_unfinished_status(self) -> None:
        """A non-final status such as running cannot be used to finish a run."""
        run_id = self.new_run()
        with self.assertRaises(ValueError):
            self.finish(run_id, STATUS_RUNNING)

    def test_finish_run_rejects_unknown_id(self) -> None:
        """Finishing a run that does not exist fails loudly."""
        with self.assertRaises(ValueError):
            self.finish(999)

    def test_mark_running_rejects_unknown_id(self) -> None:
        """Starting a run that does not exist fails loudly."""
        with self.assertRaises(ValueError):
            self.store.mark_running(999, NOW)

    def test_get_run_unknown_returns_none(self) -> None:
        """Looking up a missing run gives None rather than raising."""
        self.assertIsNone(self.store.get_run(42))

    def test_naive_datetime_is_rejected(self) -> None:
        """Naive datetimes are refused at the boundary."""
        with self.assertRaises(ValueError):
            self.new_run(queued_at=datetime(2026, 11, 2, 7, 30))

    def test_unicode_prompt_result_and_error_round_trip(self) -> None:
        """Accented text survive storage unchanged."""
        run_id = self.new_run(prompt=UNICODE_TEXT)
        self.finish(run_id, STATUS_ERROR, result=UNICODE_TEXT, error=UNICODE_TEXT)
        record = self.store.get_run(run_id)
        self.assertEqual(record.prompt, UNICODE_TEXT)
        self.assertEqual(record.result, UNICODE_TEXT)
        self.assertEqual(record.error, UNICODE_TEXT)

    def test_sql_metacharacters_are_stored_literally(self) -> None:
        """Quotes and SQL fragments in text are data, not code."""
        nasty = "x'); DROP TABLE runs; --"
        run_id = self.new_run(prompt=nasty)
        self.assertEqual(self.store.get_run(run_id).prompt, nasty)


class DatetimeTests(StoreTestCase):
    """Aware datetimes survive storage."""

    def test_non_utc_zone_round_trips_to_same_instant_and_offset(self) -> None:
        """A Berlin time comes back as the same instant with the same offset."""
        summer = datetime(2026, 7, 1, 12, 0, tzinfo=ZoneInfo("Europe/Berlin"))
        run_id = self.new_run(queued_at=summer, scheduled_for=summer)
        record = self.store.get_run(run_id)
        self.assertEqual(record.queued_at, summer)
        self.assertEqual(record.queued_at.utcoffset(), timedelta(hours=2))
        self.assertEqual(record.scheduled_for, summer)

    def test_negative_offset_round_trips(self) -> None:
        """A zone west of UTC keeps its offset."""
        moment = datetime(2026, 1, 5, 9, 15, tzinfo=timezone(timedelta(hours=-8)))
        run_id = self.new_run(queued_at=moment)
        record = self.store.get_run(run_id)
        self.assertEqual(record.queued_at, moment)
        self.assertEqual(record.queued_at.utcoffset(), timedelta(hours=-8))

    def test_last_scheduled_round_trips(self) -> None:
        """The scheduler marker keeps its instant."""
        self.store.set_last_scheduled("news", NOW)
        self.assertEqual(self.store.get_last_scheduled("news"), NOW)

    def test_last_scheduled_naive_rejected(self) -> None:
        """A naive marker is refused."""
        with self.assertRaises(ValueError):
            self.store.set_last_scheduled("news", datetime(2026, 11, 2))


class QueryTests(StoreTestCase):
    """list_runs and active_runs."""

    def test_list_runs_newest_first_with_limit(self) -> None:
        """list_runs returns the latest runs first and honours the limit."""
        ids = [self.new_run() for _ in range(5)]
        listed = self.store.list_runs(limit=3)
        self.assertEqual([r.id for r in listed], ids[::-1][:3])

    def test_list_runs_filters_by_task(self) -> None:
        """Only the requested task's runs are listed."""
        first = self.new_run("a")
        self.new_run("b")
        self.assertEqual([r.id for r in self.store.list_runs(task_name="a")], [first])

    def test_list_runs_finished_only_skips_active(self) -> None:
        """finished_only hides queued and running runs."""
        done = self.new_run()
        self.finish(done)
        running = self.new_run()
        self.store.mark_running(running, NOW)
        self.new_run()
        self.assertEqual([r.id for r in self.store.list_runs(finished_only=True)], [done])

    def test_list_runs_empty(self) -> None:
        """An empty database lists nothing."""
        self.assertEqual(self.store.list_runs(), [])

    def test_active_runs_oldest_first_and_excludes_finished(self) -> None:
        """active_runs holds queued and running runs in creation order."""
        first = self.new_run()
        self.store.mark_running(first, NOW)
        second = self.new_run()
        done = self.new_run()
        self.finish(done)
        self.assertEqual([r.id for r in self.store.active_runs()], [first, second])


class RecoveryAndRetryTests(StoreTestCase):
    """recover_interrupted, due_retries, clear_retry and retry_depth."""

    def test_recover_interrupted_fails_active_runs_only(self) -> None:
        """Queued and running runs become errors; finished runs are untouched."""
        queued = self.new_run()
        running = self.new_run()
        self.store.mark_running(running, NOW)
        done = self.new_run()
        self.finish(done)
        later = NOW + timedelta(hours=1)
        self.assertEqual(self.store.recover_interrupted(later), 2)
        for run_id in (queued, running):
            record = self.store.get_run(run_id)
            self.assertEqual(record.status, STATUS_ERROR)
            self.assertEqual(record.error, INTERRUPTED_MESSAGE)
            self.assertEqual(record.finished_at, later)
        self.assertEqual(self.store.get_run(done).status, STATUS_SUCCESS)

    def test_recover_interrupted_with_nothing_to_do(self) -> None:
        """No active runs means a count of zero."""
        self.assertEqual(self.store.recover_interrupted(NOW), 0)

    def test_due_retries_selects_by_status_and_time_oldest_first(self) -> None:
        """Only rate-limited runs whose retry time has passed are due."""
        late = self.new_run("late")
        self.finish(late, STATUS_RATE_LIMITED, retry_at=NOW - timedelta(minutes=1))
        early = self.new_run("early")
        self.finish(early, STATUS_RATE_LIMITED, retry_at=NOW - timedelta(hours=1))
        future = self.new_run("future")
        self.finish(future, STATUS_RATE_LIMITED, retry_at=NOW + timedelta(minutes=1))
        wrong_status = self.new_run("other")
        self.finish(wrong_status, STATUS_ERROR, retry_at=NOW - timedelta(hours=1))
        self.assertEqual([r.id for r in self.store.due_retries(NOW)], [late, early])

    def test_due_retries_compares_instants_across_offsets(self) -> None:
        """A retry time in another zone is compared as an instant, not as text."""
        # 06:30 UTC is 07:30 in Berlin (winter): exactly due, as retry_at <= now.
        run_id = self.new_run()
        self.finish(
            run_id,
            STATUS_RATE_LIMITED,
            retry_at=datetime(2026, 11, 2, 6, 30, tzinfo=timezone.utc),
        )
        self.assertEqual([r.id for r in self.store.due_retries(NOW)], [run_id])
        self.assertEqual(self.store.due_retries(NOW - timedelta(seconds=1)), [])

    def test_clear_retry_removes_run_from_due_list(self) -> None:
        """After clear_retry the run is no longer retried."""
        run_id = self.new_run()
        self.finish(run_id, STATUS_RATE_LIMITED, retry_at=NOW)
        self.store.clear_retry(run_id)
        self.assertEqual(self.store.due_retries(NOW), [])
        self.assertIsNone(self.store.get_run(run_id).retry_at)

    def test_retry_depth_over_three_link_chain(self) -> None:
        """Depth counts retry_of links: original 0, then 1, 2 and 3."""
        original = self.new_run()
        first = self.new_run(trigger=TRIGGER_MANUAL, retry_of=original)
        second = self.new_run(retry_of=first)
        third = self.new_run(retry_of=second)
        depths = [self.store.retry_depth(i) for i in (original, first, second, third)]
        self.assertEqual(depths, [0, 1, 2, 3])

    def test_retry_depth_unknown_run_is_zero(self) -> None:
        """A missing run has no links above it."""
        self.assertEqual(self.store.retry_depth(12345), 0)

    def test_finish_with_cancelled_status_is_allowed(self) -> None:
        """cancelled is a valid finished status."""
        run_id = self.new_run()
        self.assertEqual(self.finish(run_id, STATUS_CANCELLED).status, STATUS_CANCELLED)


class TaskStateTests(StoreTestCase):
    """Pause flags and scheduler markers."""

    def test_unknown_task_defaults(self) -> None:
        """A task never seen is not paused and has no marker."""
        self.assertFalse(self.store.is_paused("news"))
        self.assertIsNone(self.store.get_last_scheduled("news"))

    def test_pause_and_resume(self) -> None:
        """set_paused toggles the flag both ways."""
        self.store.set_paused("news", True)
        self.assertTrue(self.store.is_paused("news"))
        self.store.set_paused("news", False)
        self.assertFalse(self.store.is_paused("news"))

    def test_pause_and_marker_do_not_overwrite_each_other(self) -> None:
        """Updating one task_state column keeps the other."""
        self.store.set_paused("news", True)
        self.store.set_last_scheduled("news", NOW)
        self.assertTrue(self.store.is_paused("news"))
        self.store.set_last_scheduled("news", NOW + timedelta(days=1))
        self.store.set_paused("news", False)
        self.assertEqual(self.store.get_last_scheduled("news"), NOW + timedelta(days=1))

    def test_state_is_per_task(self) -> None:
        """Pausing one task leaves the others alone."""
        self.store.set_paused("a", True)
        self.assertFalse(self.store.is_paused("b"))


class KeyValueTests(StoreTestCase):
    """The kv table."""

    def test_missing_key_is_none(self) -> None:
        """An unset key reads as None."""
        self.assertIsNone(self.store.get_value("telegram_offset"))

    def test_set_get_and_overwrite(self) -> None:
        """Values can be written and replaced."""
        self.store.set_value("k", "1")
        self.store.set_value("k", "2")
        self.assertEqual(self.store.get_value("k"), "2")

    def test_set_none_deletes(self) -> None:
        """Writing None removes the key; deleting a missing key is fine."""
        self.store.set_value("k", "1")
        self.store.set_value("k", None)
        self.assertIsNone(self.store.get_value("k"))
        self.store.set_value("never-set", None)

    def test_empty_string_is_kept_distinct_from_none(self) -> None:
        """An empty string is a real value, not a deletion."""
        self.store.set_value("k", "")
        self.assertEqual(self.store.get_value("k"), "")

    def test_unicode_value(self) -> None:
        """Unicode values round trip."""
        self.store.set_value("k", UNICODE_TEXT)
        self.assertEqual(self.store.get_value("k"), UNICODE_TEXT)


class PersistenceAndConcurrencyTests(StoreTestCase):
    """Data outlives a Store object and survives parallel writers."""

    def test_data_visible_to_second_store_on_same_file(self) -> None:
        """A second Store (like the CLI next to the service) sees earlier writes."""
        run_id = self.new_run()
        other = Store(self.db_path)
        self.assertEqual(other.get_run(run_id).id, run_id)

    def test_two_threads_writing_runs_both_succeed(self) -> None:
        """Concurrent writers each create their runs without lock errors."""
        per_thread = 25
        errors: list[BaseException] = []
        ids: list[list[int]] = [[], []]
        barrier = threading.Barrier(2)

        def worker(index: int) -> None:
            """Create runs in a loop, recording ids and any exception."""
            try:
                barrier.wait()
                for _ in range(per_thread):
                    ids[index].append(self.new_run(f"task{index}", prompt=UNICODE_TEXT))
            except BaseException as err:  # noqa: BLE001 - surfaced by the assertion below
                errors.append(err)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        self.assertEqual(errors, [])
        all_ids = ids[0] + ids[1]
        self.assertEqual(len(set(all_ids)), 2 * per_thread)
        self.assertEqual(len(self.store.list_runs(limit=1000)), 2 * per_thread)



class PermissionTests(unittest.TestCase):
    """The data folder and database are private to the owner."""

    def test_new_folder_and_database_are_owner_only(self) -> None:
        """A fresh store creates its folder as 0700 and its database as 0600."""
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "data" / "autopilot.db"
            Store(db_path)
            self.assertEqual(db_path.parent.stat().st_mode & 0o777, 0o700)
            self.assertEqual(db_path.stat().st_mode & 0o777, 0o600)

    def test_existing_folder_keeps_its_mode_but_database_is_tightened(self) -> None:
        """A folder the user chose keeps its mode; a readable database becomes 0600."""
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "shared"
            folder.mkdir()
            folder.chmod(0o755)
            db_path = folder / "autopilot.db"
            Store(db_path)
            db_path.chmod(0o644)
            Store(db_path)
            self.assertEqual(folder.stat().st_mode & 0o777, 0o755)
            self.assertEqual(db_path.stat().st_mode & 0o777, 0o600)


if __name__ == "__main__":
    unittest.main()
