"""Unit tests for autopilot.schedule.

Covers cron parsing (lists, ranges, steps, names, Sunday as 0 and 7, error
cases), aliases, every friendly form, descriptions, the day-of-month /
day-of-week OR rule, next_after in Europe/Berlin including both DST changes of
2026, and impossible schedules.

All times are fixed and aware (TZ from helpers.py); nothing reads the real clock.
"""

from __future__ import annotations

import unittest
from datetime import date, datetime, timedelta, timezone, tzinfo

from autopilot.errors import ConfigError, ScheduleError
from autopilot.schedule import (
    ACCEPTED_FORMS,
    SEARCH_DAYS,
    Schedule,
    parse_schedule,
)
from helpers import TZ

UTC = timezone.utc

# 2026-03-29 02:00 -> 03:00 in Berlin (spring forward); 2026-10-25 03:00 -> 02:00 (fall back).
SPRING_FORWARD_DAY = date(2026, 3, 29)
FALL_BACK_DAY = date(2026, 10, 25)


class CountingZone(tzinfo):
    """A UTC time zone that counts offset lookups, a stand-in for "work done"."""

    def __init__(self) -> None:
        """Start with zero recorded lookups."""
        self.calls = 0

    def utcoffset(self, moment: datetime | None) -> timedelta:
        """Count the lookup and return a zero offset."""
        self.calls += 1
        return timedelta(0)

    def dst(self, moment: datetime | None) -> timedelta:
        """Return no daylight saving offset."""
        return timedelta(0)

    def tzname(self, moment: datetime | None) -> str:
        """Return the zone name."""
        return "COUNT"


def at(year: int, month: int, day: int, hour: int = 0, minute: int = 0, fold: int = 0) -> datetime:
    """Build an aware Berlin datetime for a test.

    Args:
        year: Calendar year.
        month: Calendar month.
        day: Day of month.
        hour: Hour of day.
        minute: Minute of the hour.
        fold: 1 selects the second pass of an ambiguous fall-back time.

    Returns:
        The datetime in Europe/Berlin.
    """
    return datetime(year, month, day, hour, minute, tzinfo=TZ, fold=fold)


def next_times(schedule: Schedule, start: datetime, count: int) -> list[datetime]:
    """Return the next ``count`` fire times, each found from the previous one.

    Args:
        schedule: The schedule to walk.
        start: The moment to search after.
        count: How many fire times to collect.

    Returns:
        The fire times in order.
    """
    times: list[datetime] = []
    moment = start
    for _ in range(count):
        moment = schedule.next_after(moment)
        times.append(moment)
    return times


class CronParsingTests(unittest.TestCase):
    """parse_schedule on 5-field cron expressions."""

    def test_star_fields_expand_to_full_ranges(self) -> None:
        schedule = parse_schedule("* * * * *")
        self.assertEqual(schedule.minutes, frozenset(range(60)))
        self.assertEqual(schedule.hours, frozenset(range(24)))
        self.assertEqual(schedule.days_of_month, frozenset(range(1, 32)))
        self.assertEqual(schedule.months, frozenset(range(1, 13)))
        self.assertEqual(schedule.weekdays, frozenset(range(7)))
        self.assertTrue(schedule.day_of_month_any)
        self.assertTrue(schedule.weekday_any)

    def test_single_values_and_expression_is_stripped(self) -> None:
        schedule = parse_schedule("  30 7 * * *  ")
        self.assertEqual(schedule.expression, "30 7 * * *")
        self.assertEqual(schedule.minutes, frozenset({30}))
        self.assertEqual(schedule.hours, frozenset({7}))

    def test_lists_ranges_and_steps(self) -> None:
        schedule = parse_schedule("0,30 8-10 1,15 */3 *")
        self.assertEqual(schedule.minutes, frozenset({0, 30}))
        self.assertEqual(schedule.hours, frozenset({8, 9, 10}))
        self.assertEqual(schedule.days_of_month, frozenset({1, 15}))
        self.assertEqual(schedule.months, frozenset({1, 4, 7, 10}))

    def test_range_with_step_and_start_with_step(self) -> None:
        schedule = parse_schedule("10-40/10 20/2 * * *")
        self.assertEqual(schedule.minutes, frozenset({10, 20, 30, 40}))
        # "N/S" runs from N to the end of the field.
        self.assertEqual(schedule.hours, frozenset({20, 22}))

    def test_step_over_star_starts_at_field_minimum(self) -> None:
        self.assertEqual(parse_schedule("*/20 * * * *").minutes, frozenset({0, 20, 40}))
        self.assertEqual(parse_schedule("0 0 */10 * *").days_of_month, frozenset({1, 11, 21, 31}))

    def test_month_names_and_ranges(self) -> None:
        schedule = parse_schedule("0 0 1 jan,mar-may,DEC *")
        self.assertEqual(schedule.months, frozenset({1, 3, 4, 5, 12}))

    def test_weekday_names_convert_to_python_numbering(self) -> None:
        schedule = parse_schedule("0 0 * * mon,wed,sun")
        self.assertEqual(schedule.weekdays, frozenset({0, 2, 6}))

    def test_cron_weekday_numbers_convert_to_python_numbering(self) -> None:
        # Cron: 1 = Monday ... 6 = Saturday; Python: 0 = Monday ... 5 = Saturday.
        self.assertEqual(parse_schedule("0 0 * * 1-5").weekdays, frozenset({0, 1, 2, 3, 4}))
        self.assertEqual(parse_schedule("0 0 * * 6").weekdays, frozenset({5}))

    def test_cron_weekday_zero_and_seven_are_both_sunday(self) -> None:
        self.assertEqual(parse_schedule("0 0 * * 0").weekdays, frozenset({6}))
        self.assertEqual(parse_schedule("0 0 * * 7").weekdays, frozenset({6}))
        self.assertEqual(parse_schedule("0 0 * * 5-7").weekdays, frozenset({4, 5, 6}))

    def test_case_and_whitespace_are_tolerated(self) -> None:
        schedule = parse_schedule("0   9\t*  JAN   MON-FRI")
        self.assertEqual(schedule.months, frozenset({1}))
        self.assertEqual(schedule.weekdays, frozenset({0, 1, 2, 3, 4}))

    def test_any_flags_are_set_for_fields_starting_with_a_star(self) -> None:
        # Like Vixie cron: "*/2" counts as unrestricted for the OR rule.
        schedule = parse_schedule("0 0 */2 * 1")
        self.assertTrue(schedule.day_of_month_any)
        self.assertFalse(schedule.weekday_any)
        self.assertTrue(parse_schedule("0 0 1 * */2").weekday_any)

    def test_any_flags_are_clear_for_explicit_values(self) -> None:
        schedule = parse_schedule("0 0 1-31 * 0-6")
        self.assertFalse(schedule.day_of_month_any)
        self.assertFalse(schedule.weekday_any)

    def test_invalid_cron_raises_schedule_error_with_detail(self) -> None:
        cases = {
            "60 * * * *": "out of range",
            "* 24 * * *": "out of range",
            "* * 0 * *": "out of range",
            "* * 32 * *": "out of range",
            "* * * 13 *": "out of range",
            "* * * * 8": "out of range",
            "*/0 * * * *": "step",
            "*/-1 * * * *": "step",
            "*/x * * * *": "step",
            "5-1 * * * *": "backwards",
            "1,,2 * * * *": "empty item",
            ", * * * *": "empty item",
            "* * * foo *": "not a valid month value",
            "* * * * funday": "not a valid day-of-week value",
            "a * * * *": "not a valid minute value",
            "-5 * * * *": "not a valid minute value",
            "* * * *": "got 4",
            "* * * * * *": "got 6",
        }
        for expression, detail in cases.items():
            with self.subTest(expression=expression):
                with self.assertRaises(ScheduleError) as caught:
                    parse_schedule(expression)
                self.assertIn(detail, str(caught.exception))

    def test_error_quotes_expression_and_lists_accepted_forms(self) -> None:
        with self.assertRaises(ScheduleError) as caught:
            parse_schedule("61 * * * *")
        message = str(caught.exception)
        self.assertIn("'61 * * * *'", message)
        self.assertIn(ACCEPTED_FORMS, message)

    def test_schedule_error_is_a_config_error(self) -> None:
        with self.assertRaises(ConfigError):
            parse_schedule("nonsense")

    def test_empty_and_blank_input_is_rejected(self) -> None:
        for text in ("", "   "):
            with self.subTest(text=text):
                with self.assertRaises(ScheduleError) as caught:
                    parse_schedule(text)
                self.assertIn("empty", str(caught.exception))

    def test_unicode_input_is_rejected_cleanly(self) -> None:
        with self.assertRaises(ScheduleError):
            parse_schedule("Naïve café résumé ☕")
        with self.assertRaises(ScheduleError):
            parse_schedule("٣٠ * * * *")  # Arabic-Indic digits are not cron numbers.


class AliasTests(unittest.TestCase):
    """The @-aliases."""

    def test_aliases_map_to_their_cron_sets(self) -> None:
        hourly = parse_schedule("@hourly")
        self.assertEqual((hourly.minutes, hourly.hours), (frozenset({0}), frozenset(range(24))))
        for alias in ("@daily", "@midnight"):
            with self.subTest(alias=alias):
                daily = parse_schedule(alias)
                self.assertEqual((daily.minutes, daily.hours), (frozenset({0}), frozenset({0})))
                self.assertTrue(daily.weekday_any)
        weekly = parse_schedule("@weekly")
        self.assertEqual(weekly.weekdays, frozenset({6}))
        monthly = parse_schedule("@MONTHLY")
        self.assertEqual(monthly.days_of_month, frozenset({1}))
        self.assertFalse(monthly.day_of_month_any)

    def test_alias_keeps_original_text_as_expression(self) -> None:
        self.assertEqual(parse_schedule(" @Daily ").expression, "@Daily")

    def test_unknown_alias_is_rejected(self) -> None:
        with self.assertRaises(ScheduleError) as caught:
            parse_schedule("@yearly")
        self.assertIn("unknown alias", str(caught.exception))

    def test_alias_descriptions(self) -> None:
        self.assertEqual(parse_schedule("@daily").description, "daily at 00:00")
        self.assertEqual(parse_schedule("@midnight").description, "daily at 00:00")
        self.assertEqual(parse_schedule("@hourly").description, "hourly")
        self.assertEqual(parse_schedule("@weekly").description, "Sun at 00:00")
        self.assertEqual(parse_schedule("@monthly").description, "monthly on day 1 at 00:00")


class FriendlyFormTests(unittest.TestCase):
    """Every friendly form and its description."""

    def test_every_n_minutes_spellings(self) -> None:
        for text in ("every 15m", "every 15 m", "every 15min", "every 15 mins",
                     "every 15 minute", "EVERY 15 Minutes"):
            with self.subTest(text=text):
                schedule = parse_schedule(text)
                self.assertEqual(schedule.minutes, frozenset({0, 15, 30, 45}))
                self.assertEqual(schedule.hours, frozenset(range(24)))
                self.assertEqual(schedule.description, "every 15 minutes")

    def test_every_minute_and_every_60_minutes(self) -> None:
        every_minute = parse_schedule("every 1m")
        self.assertEqual(every_minute.minutes, frozenset(range(60)))
        self.assertEqual(every_minute.description, "every minute")
        sixty = parse_schedule("every 60m")
        self.assertEqual(sixty.minutes, frozenset({0}))
        self.assertEqual(sixty.hours, frozenset(range(24)))
        self.assertEqual(sixty.description, "hourly")

    def test_every_n_hours_spellings(self) -> None:
        for text in ("every 2h", "every 2 h", "every 2hour", "every 2 hours"):
            with self.subTest(text=text):
                schedule = parse_schedule(text)
                self.assertEqual(schedule.minutes, frozenset({0}))
                self.assertEqual(schedule.hours, frozenset(range(0, 24, 2)))
                self.assertEqual(schedule.description, "every 2 hours")

    def test_every_one_hour_and_hourly(self) -> None:
        for text in ("every 1h", "hourly", "Hourly"):
            with self.subTest(text=text):
                schedule = parse_schedule(text)
                self.assertEqual(schedule.minutes, frozenset({0}))
                self.assertEqual(schedule.hours, frozenset(range(24)))
                self.assertEqual(schedule.description, "hourly")

    def test_uneven_or_zero_intervals_are_rejected_with_cron_hint(self) -> None:
        for text in ("every 7m", "every 90m", "every 0m", "every 5h", "every 25h", "every 0h"):
            with self.subTest(text=text):
                with self.assertRaises(ScheduleError) as caught:
                    parse_schedule(text)
                self.assertIn("cron expression", str(caught.exception))

    def test_daily_forms(self) -> None:
        for text in ("daily 07:30", "every day 07:30", "Daily 7:30", "  daily   07:30 "):
            with self.subTest(text=text):
                schedule = parse_schedule(text)
                self.assertEqual(schedule.minutes, frozenset({30}))
                self.assertEqual(schedule.hours, frozenset({7}))
                self.assertEqual(schedule.weekdays, frozenset(range(7)))
                self.assertTrue(schedule.weekday_any)
                self.assertEqual(schedule.description, "daily at 07:30")

    def test_weekdays_and_weekends(self) -> None:
        weekdays = parse_schedule("weekdays 08:00")
        self.assertEqual(weekdays.weekdays, frozenset({0, 1, 2, 3, 4}))
        self.assertFalse(weekdays.weekday_any)
        self.assertEqual(weekdays.description, "weekdays at 08:00")
        weekends = parse_schedule("weekends 10:00")
        self.assertEqual(weekends.weekdays, frozenset({5, 6}))
        self.assertEqual(weekends.description, "weekends at 10:00")

    def test_day_lists(self) -> None:
        schedule = parse_schedule("mon,wed,fri 18:00")
        self.assertEqual(schedule.weekdays, frozenset({0, 2, 4}))
        self.assertEqual(schedule.minutes, frozenset({0}))
        self.assertEqual(schedule.hours, frozenset({18}))
        self.assertEqual(schedule.description, "Mon, Wed, Fri at 18:00")

    def test_full_day_names_and_single_day(self) -> None:
        self.assertEqual(parse_schedule("monday,Thursday 09:15").weekdays, frozenset({0, 3}))
        self.assertEqual(parse_schedule("sun 19:00").description, "Sun at 19:00")

    def test_day_ranges(self) -> None:
        weekdays = parse_schedule("mon-fri 07:00")
        self.assertEqual(weekdays.weekdays, frozenset({0, 1, 2, 3, 4}))
        self.assertEqual(weekdays.description, "weekdays at 07:00")
        # Ranges run Monday-first, so a range ending on Sunday works.
        self.assertEqual(parse_schedule("fri-sun 07:00").weekdays, frozenset({4, 5, 6}))
        self.assertEqual(parse_schedule("mon-wed,sat 07:00").weekdays, frozenset({0, 1, 2, 5}))

    def test_friendly_forms_use_every_month_and_day_of_month(self) -> None:
        schedule = parse_schedule("daily 07:30")
        self.assertEqual(schedule.months, frozenset(range(1, 13)))
        self.assertEqual(schedule.days_of_month, frozenset(range(1, 32)))
        self.assertTrue(schedule.day_of_month_any)

    def test_clock_boundaries_are_accepted(self) -> None:
        self.assertEqual(parse_schedule("daily 00:00").description, "daily at 00:00")
        self.assertEqual(parse_schedule("daily 23:59").description, "daily at 23:59")

    def test_bad_clock_times_are_rejected(self) -> None:
        for text in ("daily 24:00", "daily 07:60", "daily 7", "daily 7:5", "daily 07:30:00",
                     "daily noon", "weekdays 123:00"):
            with self.subTest(text=text):
                with self.assertRaises(ScheduleError) as caught:
                    parse_schedule(text)
                self.assertIn("time", str(caught.exception))

    def test_bad_day_lists_are_rejected(self) -> None:
        cases = {
            "funday 08:00": "unknown day",
            "mon,,fri 08:00": "empty item",
            "mon, fri 08:00": None,  # a space splits the words, so this is 3 words
            "sat-mon 08:00": "backwards",
            "mon-tue-wed 08:00": "unknown day",
            "every banana": "every",
        }
        for text, detail in cases.items():
            with self.subTest(text=text):
                with self.assertRaises(ScheduleError) as caught:
                    parse_schedule(text)
                if detail is not None:
                    self.assertIn(detail, str(caught.exception))

    def test_unrecognised_text_is_rejected_with_field_count(self) -> None:
        with self.assertRaises(ScheduleError) as caught:
            parse_schedule("whenever you like")
        self.assertIn("got 3", str(caught.exception))


class CronDescriptionTests(unittest.TestCase):
    """Descriptions of cron expressions."""

    def test_one_minute_one_hour_any_day_is_daily(self) -> None:
        self.assertEqual(parse_schedule("30 7 * * *").description, "daily at 07:30")
        self.assertEqual(parse_schedule("0 0 * * 0-6").description, "daily at 00:00")

    def test_monday_to_friday_is_weekdays(self) -> None:
        self.assertEqual(parse_schedule("30 7 * * 1-5").description, "weekdays at 07:30")
        self.assertEqual(parse_schedule("0 8 * * mon-fri").description, "weekdays at 08:00")

    def test_minute_zero_of_every_hour_is_hourly(self) -> None:
        self.assertEqual(parse_schedule("0 * * * *").description, "hourly")
        self.assertEqual(parse_schedule("0 0-23 * * *").description, "hourly")

    def test_one_time_with_weekday_list_names_the_days(self) -> None:
        self.assertEqual(parse_schedule("0 18 * * mon,wed").description, "Mon, Wed at 18:00")
        self.assertEqual(parse_schedule("15 9 * * 1,3,5").description, "Mon, Wed, Fri at 09:15")
        self.assertEqual(parse_schedule("0 8 * * 0,6").description, "weekends at 08:00")
        self.assertEqual(parse_schedule("0 19 * * 7").description, "Sun at 19:00")

    def test_one_time_on_one_day_of_month_is_monthly(self) -> None:
        self.assertEqual(parse_schedule("0 9 1 * *").description, "monthly on day 1 at 09:00")
        self.assertEqual(parse_schedule("5 14 15 * *").description, "monthly on day 15 at 14:05")

    def test_everything_else_shows_the_cron_text(self) -> None:
        for expression in (
            "*/5 * * * *",
            "30 * * * *",
            "0 8,17 * * *",
            "0,30 8 * * *",
            "0 8 1,15 * *",
            "0 8 * 6 *",
            "0 8 1 6 *",
            "0 9 13 * fri",  # day-of-month and weekday both set: cron OR rule
            "0 9 */2 * *",  # every second day, not every day
            "0 9 1 * 0-6",  # an explicit weekday list triggers the OR rule: fires every day
            "0 9 1-31 * mon",
        ):
            with self.subTest(expression=expression):
                self.assertEqual(parse_schedule(expression).description, f"cron: {expression}")


class MatchesDayTests(unittest.TestCase):
    """Schedule.matches_day, including the cron OR rule."""

    def test_unrestricted_days_always_match(self) -> None:
        schedule = parse_schedule("0 0 * * *")
        self.assertTrue(schedule.matches_day(date(2026, 2, 28)))
        self.assertTrue(schedule.matches_day(date(2026, 12, 31)))

    def test_month_must_match(self) -> None:
        schedule = parse_schedule("0 0 * jun *")
        self.assertTrue(schedule.matches_day(date(2026, 6, 15)))
        self.assertFalse(schedule.matches_day(date(2026, 7, 15)))

    def test_only_day_of_month_restricted(self) -> None:
        schedule = parse_schedule("0 0 15 * *")
        self.assertTrue(schedule.matches_day(date(2026, 11, 15)))  # a Sunday
        self.assertFalse(schedule.matches_day(date(2026, 11, 16)))

    def test_only_weekday_restricted(self) -> None:
        schedule = parse_schedule("0 0 * * fri")
        self.assertTrue(schedule.matches_day(date(2026, 11, 6)))
        self.assertFalse(schedule.matches_day(date(2026, 11, 7)))

    def test_both_restricted_means_either_matches(self) -> None:
        # Cron OR rule: the 13th, or any Friday.
        schedule = parse_schedule("0 0 13 * 5")
        self.assertTrue(schedule.matches_day(date(2026, 11, 13)))  # 13th and a Friday
        self.assertTrue(schedule.matches_day(date(2026, 11, 6)))  # a Friday, not the 13th
        self.assertTrue(schedule.matches_day(date(2026, 12, 13)))  # the 13th, a Sunday
        self.assertFalse(schedule.matches_day(date(2026, 11, 12)))  # neither

    def test_or_rule_in_next_after(self) -> None:
        schedule = parse_schedule("0 9 13 * fri")
        found = next_times(schedule, at(2026, 11, 1), 4)
        self.assertEqual(
            found,
            [at(2026, 11, 6, 9), at(2026, 11, 13, 9), at(2026, 11, 20, 9), at(2026, 11, 27, 9)],
        )

    def test_star_step_day_field_combines_with_weekday_by_and(self) -> None:
        # "*/2" is unrestricted for the OR rule: odd days that are Mondays, not odd days plus Mondays.
        schedule = parse_schedule("0 9 */2 * 1")
        found = next_times(schedule, at(2026, 11, 4), 8)
        for moment in found:
            with self.subTest(moment=moment):
                self.assertEqual(moment.weekday(), 0)
                self.assertEqual(moment.day % 2, 1)
        self.assertEqual(found[:3], [at(2026, 11, 9, 9), at(2026, 11, 23, 9), at(2026, 12, 7, 9)])
        self.assertTrue(schedule.matches_day(date(2026, 11, 9)))  # an odd Monday
        self.assertFalse(schedule.matches_day(date(2026, 11, 16)))  # an even Monday
        self.assertFalse(schedule.matches_day(date(2026, 11, 11)))  # an odd Wednesday

    def test_star_step_weekday_field_combines_with_day_of_month_by_and(self) -> None:
        schedule = parse_schedule("0 9 13 * */2")  # weekday set: Sun, Tue, Thu, Sat
        self.assertFalse(schedule.matches_day(date(2026, 11, 13)))  # the 13th, a Friday
        self.assertTrue(schedule.matches_day(date(2026, 6, 13)))  # the 13th, a Saturday
        self.assertFalse(schedule.matches_day(date(2026, 6, 16)))  # a Tuesday, not the 13th

    def test_month_still_applies_with_the_or_rule(self) -> None:
        schedule = parse_schedule("0 9 13 jun fri")
        self.assertFalse(schedule.matches_day(date(2026, 11, 13)))
        self.assertTrue(schedule.matches_day(date(2026, 6, 5)))  # a Friday in June


class NextAfterTests(unittest.TestCase):
    """Schedule.next_after outside DST changes."""

    def test_result_is_strictly_after(self) -> None:
        schedule = parse_schedule("30 7 * * *")
        self.assertEqual(schedule.next_after(at(2026, 11, 2, 7, 29)), at(2026, 11, 2, 7, 30))
        self.assertEqual(schedule.next_after(at(2026, 11, 2, 7, 30)), at(2026, 11, 3, 7, 30))

    def test_seconds_and_microseconds_are_dropped(self) -> None:
        schedule = parse_schedule("every 1m")
        after = datetime(2026, 11, 2, 7, 30, 59, 999999, tzinfo=TZ)
        result = schedule.next_after(after)
        self.assertEqual(result, at(2026, 11, 2, 7, 31))
        self.assertEqual((result.second, result.microsecond), (0, 0))

    def test_result_keeps_the_timezone_of_the_input(self) -> None:
        schedule = parse_schedule("daily 07:30")
        self.assertIs(schedule.next_after(at(2026, 11, 2)).tzinfo, TZ)
        in_utc = schedule.next_after(datetime(2026, 11, 2, 12, 0, tzinfo=UTC))
        self.assertEqual(in_utc, datetime(2026, 11, 3, 7, 30, tzinfo=UTC))
        self.assertEqual(in_utc.utcoffset(), timedelta(0))

    def test_search_does_not_rebuild_earlier_times_of_the_first_day(self) -> None:
        # Late in the day, a dense schedule must go straight to the next slot
        # instead of walking every minute since 00:00 (about 1400 datetimes).
        zone = CountingZone()
        after = datetime(2026, 11, 2, 23, 57, tzinfo=zone)
        self.assertEqual(
            parse_schedule("* * * * *").next_after(after),
            datetime(2026, 11, 2, 23, 58, tzinfo=zone),
        )
        self.assertLess(zone.calls, 20)

    def test_naive_datetime_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            parse_schedule("daily 07:30").next_after(datetime(2026, 11, 2, 7, 0))

    def test_every_15_minutes_across_midnight(self) -> None:
        schedule = parse_schedule("every 15m")
        self.assertEqual(
            next_times(schedule, at(2026, 11, 2, 23, 40), 3),
            [at(2026, 11, 2, 23, 45), at(2026, 11, 3, 0, 0), at(2026, 11, 3, 0, 15)],
        )

    def test_every_2_hours(self) -> None:
        schedule = parse_schedule("every 2h")
        self.assertEqual(
            next_times(schedule, at(2026, 11, 2, 21, 30), 3),
            [at(2026, 11, 2, 22), at(2026, 11, 3, 0), at(2026, 11, 3, 2)],
        )

    def test_weekdays_skip_the_weekend(self) -> None:
        schedule = parse_schedule("weekdays 08:00")
        # 2026-11-06 is a Friday.
        self.assertEqual(
            next_times(schedule, at(2026, 11, 6, 9), 2),
            [at(2026, 11, 9, 8), at(2026, 11, 10, 8)],
        )

    def test_weekends(self) -> None:
        schedule = parse_schedule("weekends 10:00")
        self.assertEqual(
            next_times(schedule, at(2026, 11, 2), 3),
            [at(2026, 11, 7, 10), at(2026, 11, 8, 10), at(2026, 11, 14, 10)],
        )

    def test_day_list_with_several_times_per_day(self) -> None:
        schedule = parse_schedule("0,30 8,17 * * mon,wed")
        self.assertEqual(
            next_times(schedule, at(2026, 11, 2, 8, 0), 4),  # a Monday
            [at(2026, 11, 2, 8, 30), at(2026, 11, 2, 17, 0), at(2026, 11, 2, 17, 30),
             at(2026, 11, 4, 8, 0)],
        )

    def test_monthly_alias_rolls_over_the_year(self) -> None:
        schedule = parse_schedule("@monthly")
        self.assertEqual(
            next_times(schedule, at(2026, 11, 2), 3),
            [at(2026, 12, 1), at(2027, 1, 1), at(2027, 2, 1)],
        )

    def test_weekly_alias_fires_on_sunday(self) -> None:
        schedule = parse_schedule("@weekly")
        self.assertEqual(schedule.next_after(at(2026, 11, 2)), at(2026, 11, 8))  # a Sunday

    def test_month_restriction_jumps_to_the_right_month(self) -> None:
        schedule = parse_schedule("0 12 1 jun *")
        self.assertEqual(schedule.next_after(at(2026, 11, 2)), at(2027, 6, 1, 12))

    def test_feb_29_waits_for_the_next_leap_year(self) -> None:
        schedule = parse_schedule("0 0 29 2 *")
        self.assertEqual(schedule.next_after(at(2026, 11, 2)), at(2028, 2, 29))

    def test_end_of_year_rollover(self) -> None:
        schedule = parse_schedule("daily 07:30")
        self.assertEqual(schedule.next_after(at(2026, 12, 31, 8)), at(2027, 1, 1, 7, 30))

    def test_impossible_schedule_raises_schedule_error(self) -> None:
        for expression in ("0 0 30 2 *", "0 0 31 4 *", "0 0 31 apr,jun,sep,nov *"):
            with self.subTest(expression=expression):
                schedule = parse_schedule(expression)  # parsing itself succeeds
                with self.assertRaises(ScheduleError) as caught:
                    schedule.next_after(at(2026, 11, 2))
                self.assertIn(expression, str(caught.exception))
                self.assertIn("never fires", str(caught.exception))

    def test_search_window_is_five_years(self) -> None:
        self.assertEqual(SEARCH_DAYS, 366 * 5)


class DaylightSavingTests(unittest.TestCase):
    """next_after around the two 2026 DST changes in Europe/Berlin."""

    def test_spring_forward_skips_a_time_that_does_not_exist(self) -> None:
        schedule = parse_schedule("30 2 * * *")
        # 02:30 does not exist on 2026-03-29, so that day is skipped.
        self.assertEqual(schedule.next_after(at(2026, 3, 28, 3)), at(2026, 3, 30, 2, 30))
        self.assertEqual(schedule.next_after(at(2026, 3, 28, 2, 30)), at(2026, 3, 30, 2, 30))

    def test_spring_forward_day_is_skipped_in_a_daily_sequence(self) -> None:
        schedule = parse_schedule("30 2 * * *")
        found = next_times(schedule, at(2026, 3, 27, 12), 3)
        self.assertEqual(found, [at(2026, 3, 28, 2, 30), at(2026, 3, 30, 2, 30),
                                 at(2026, 3, 31, 2, 30)])
        self.assertNotIn(SPRING_FORWARD_DAY, [moment.date() for moment in found])

    def test_spring_forward_keeps_other_times_of_the_same_day(self) -> None:
        schedule = parse_schedule("0,30 1-4 * * *")
        found = next_times(schedule, at(2026, 3, 29, 1, 0), 4)
        # 02:00 and 02:30 do not exist; 03:00 exists (UTC+2).
        self.assertEqual(found, [at(2026, 3, 29, 1, 30), at(2026, 3, 29, 3, 0),
                                 at(2026, 3, 29, 3, 30), at(2026, 3, 29, 4, 0)])
        self.assertEqual(found[1].utcoffset(), timedelta(hours=2))

    def test_spring_forward_gap_is_skipped_by_interval_schedules(self) -> None:
        # Nothing fires inside the missing 02:00-03:00 hour; the next slot is 03:00 (UTC+2).
        schedule = parse_schedule("every 15m")
        self.assertEqual(schedule.next_after(at(2026, 3, 29, 1, 50)), at(2026, 3, 29, 3, 0))

    def test_spring_forward_result_offsets_are_correct(self) -> None:
        schedule = parse_schedule("daily 07:00")
        before = schedule.next_after(at(2026, 3, 27, 8))
        after = schedule.next_after(before)
        self.assertEqual(before.utcoffset(), timedelta(hours=1))
        self.assertEqual(after.utcoffset(), timedelta(hours=2))
        # Local 07:00 on both days, so the real gap is 23 hours.
        self.assertEqual(after.astimezone(UTC) - before.astimezone(UTC), timedelta(hours=23))

    def test_fall_back_ambiguous_time_fires_once(self) -> None:
        schedule = parse_schedule("30 2 * * *")
        first = schedule.next_after(at(2026, 10, 24, 12))
        self.assertEqual(first, at(2026, 10, 25, 2, 30))
        self.assertEqual(first.fold, 0)
        self.assertEqual(first.utcoffset(), timedelta(hours=2))  # the first pass, summer time
        self.assertEqual(schedule.next_after(first), at(2026, 10, 26, 2, 30))

    def test_fall_back_second_pass_does_not_fire_again(self) -> None:
        schedule = parse_schedule("30 2 * * *")
        second_pass = at(2026, 10, 25, 2, 10, fold=1)  # after the first 02:30 has passed
        self.assertEqual(second_pass.utcoffset(), timedelta(hours=1))
        self.assertEqual(schedule.next_after(second_pass), at(2026, 10, 26, 2, 30))
        # Even from the very end of the second pass.
        self.assertEqual(
            schedule.next_after(at(2026, 10, 25, 2, 59, fold=1)), at(2026, 10, 26, 2, 30)
        )

    def test_fall_back_daily_sequence_has_exactly_one_firing(self) -> None:
        schedule = parse_schedule("30 2 * * *")
        found = next_times(schedule, at(2026, 10, 23, 12), 4)
        self.assertEqual(found, [at(2026, 10, 24, 2, 30), at(2026, 10, 25, 2, 30),
                                 at(2026, 10, 26, 2, 30), at(2026, 10, 27, 2, 30)])
        self.assertEqual(
            len([moment for moment in found if moment.date() == FALL_BACK_DAY]), 1
        )

    def test_fall_back_results_always_move_forward_in_real_time(self) -> None:
        schedule = parse_schedule("*/15 * * * *")
        moment = at(2026, 10, 25, 1, 0)
        for _ in range(12):
            following = schedule.next_after(moment)
            self.assertGreater(following.astimezone(UTC), moment.astimezone(UTC))
            moment = following

    def test_fall_back_hourly_fires_once_per_wall_hour(self) -> None:
        schedule = parse_schedule("hourly")
        found = next_times(schedule, at(2026, 10, 25, 0, 30), 4)
        wall_hours = [moment.hour for moment in found]
        self.assertEqual(wall_hours, [1, 2, 3, 4])  # 02:00 is not repeated

    def test_time_outside_the_ambiguous_hour_is_unaffected(self) -> None:
        schedule = parse_schedule("30 3 * * *")
        self.assertEqual(schedule.next_after(at(2026, 10, 24, 12)), at(2026, 10, 25, 3, 30))
        self.assertEqual(
            at(2026, 10, 25, 3, 30).utcoffset(), timedelta(hours=1)
        )  # winter time again


if __name__ == "__main__":
    unittest.main()
