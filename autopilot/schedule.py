"""Schedule parsing and next-fire-time calculation (pure, no I/O).

A ``Schedule`` is the parsed form of one schedule expression from tasks.toml
or from ``autopilot next "<expr>"``. Three input styles are accepted and all
of them end up as the same five cron sets (minutes, hours, days of month,
months, weekdays):

    * 5-field cron, e.g. ``30 7 * * 1-5`` (names and ranges allowed);
    * aliases: ``@hourly``, ``@daily``, ``@midnight``, ``@weekly``, ``@monthly``;
    * friendly forms: ``every 15m``, ``every 2h``, ``hourly``, ``daily 07:30``,
      ``weekdays 08:00``, ``weekends 10:00``, ``mon,wed,fri 18:00``, ``mon-fri 07:00``.

Used by: tasks.py (validation at load), scheduler.py (due slots), engine.py and
the CLI (next fire times, descriptions).
Depends on: errors.py and the standard library.

Invariants:
    * All datetimes are timezone-aware; naive input raises ValueError.
    * Fire times are wall-clock times in the zone of the datetime passed in.
      A wall time that does not exist (DST spring-forward gap) is skipped; a
      wall time that happens twice (fall back) fires once, on its first pass.
    * Weekdays use Python numbering (0 = Monday) everywhere outside the cron
      parser, which converts from cron numbering (0 and 7 = Sunday).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from datetime import date, datetime, timedelta, timezone

from autopilot.errors import ScheduleError
from autopilot.models import MINUTES_PER_HOUR

# Index = date.weekday().
WEEKDAY_NAMES = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")
# Index + 1 = month number.
MONTH_NAMES = (
    "jan", "feb", "mar", "apr", "may", "jun",
    "jul", "aug", "sep", "oct", "nov", "dec",
)
# How far next_after searches before declaring that a schedule never fires.
SEARCH_DAYS = 366 * 5

# Full English day names accepted by the friendly "<days> HH:MM" form (same order as WEEKDAY_NAMES).
FULL_WEEKDAY_NAMES = (
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
)

# Full sets for every cron field; a "*" field expands to exactly these.
ALL_HOURS = frozenset(range(24))
ALL_DAYS_OF_MONTH = frozenset(range(1, 32))
ALL_MONTHS = frozenset(range(1, 13))
ALL_WEEKDAYS = frozenset(range(7))
MON_TO_FRI = frozenset(range(5))
SAT_AND_SUN = frozenset({5, 6})

HOURS_PER_DAY = 24
LAST_MINUTE = MINUTES_PER_HOUR - 1
LAST_HOUR = HOURS_PER_DAY - 1
CRON_FIELD_COUNT = 5

# Cron shorthand names and the 5-field expression each one stands for.
CRON_ALIASES = {
    "@hourly": "0 * * * *",
    "@daily": "0 0 * * *",
    "@midnight": "0 0 * * *",
    "@weekly": "0 0 * * 0",
    "@monthly": "0 0 1 * *",
}

# Shown at the end of every ScheduleError so the user sees what is allowed.
ACCEPTED_FORMS = (
    "Accepted forms: 5-field cron ('30 7 * * 1-5'), @hourly, @daily, @weekly, @monthly, "
    "'every 15m', 'every 2h', 'hourly', 'daily 07:30', 'weekdays 08:00', "
    "'weekends 10:00', 'mon,wed,fri 18:00', 'mon-fri 07:00'."
)

# "every 15m", "every 2 hours": N, optional space, unit. Matches the whole text.
_INTERVAL_RE = re.compile(r"every ([0-9]+) ?(m|min|mins|minute|minutes|h|hour|hours)")
# "07:30" or "7:30": one or two hour digits, exactly two minute digits.
_CLOCK_RE = re.compile(r"([0-9]{1,2}):([0-9]{2})")
_MINUTE_UNITS = frozenset({"m", "min", "mins", "minute", "minutes"})


class _InvalidSchedule(Exception):
    """Internal: a parse step found a problem; carries only the detail text.

    parse_schedule turns it into a ScheduleError that also quotes the whole
    expression and lists the accepted forms.
    """


@dataclass(frozen=True)
class _FieldSpec:
    """Describes one cron field: its label, allowed range and accepted names.

    Attributes:
        label: Field name used in error messages, e.g. "month".
        low: Smallest allowed value.
        high: Largest allowed value (the weekday field allows 7 for Sunday).
        names: Lower-case name to value, e.g. ``{"jan": 1}``; empty when the field has none.
    """

    label: str
    low: int
    high: int
    names: dict[str, int]


_MINUTE_FIELD = _FieldSpec("minute", 0, LAST_MINUTE, {})
_HOUR_FIELD = _FieldSpec("hour", 0, LAST_HOUR, {})
_DAY_OF_MONTH_FIELD = _FieldSpec("day-of-month", 1, 31, {})
_MONTH_FIELD = _FieldSpec("month", 1, 12, {name: i + 1 for i, name in enumerate(MONTH_NAMES)})
# Cron weekday numbering: sun = 0, mon = 1 ... sat = 6; 7 is accepted as Sunday too.
_WEEKDAY_FIELD = _FieldSpec(
    "day-of-week", 0, 7, {name: (i + 1) % 7 for i, name in enumerate(WEEKDAY_NAMES)}
)


@dataclass(frozen=True)
class Schedule:
    """One parsed schedule: the cron sets plus a readable description.

    Attributes:
        expression: The stripped original text.
        description: Short English, e.g. "daily at 07:30".
        minutes: Allowed minutes, 0-59.
        hours: Allowed hours, 0-23.
        days_of_month: Allowed days of month, 1-31.
        months: Allowed months, 1-12.
        weekdays: Allowed weekdays, 0-6 with 0 = Monday (Python numbering).
        day_of_month_any: True when the day-of-month field started with ``*``
            (``*``, ``*/2``), which cron treats as unrestricted for the OR rule.
        weekday_any: True when the day-of-week field started with ``*``.

    Example:
        >>> schedule = parse_schedule("daily 07:30")
        >>> schedule.next_after(datetime(2026, 11, 2, 8, 0, tzinfo=timezone.utc))
        datetime.datetime(2026, 11, 3, 7, 30, tzinfo=datetime.timezone.utc)
    """

    expression: str
    description: str
    minutes: frozenset[int]
    hours: frozenset[int]
    days_of_month: frozenset[int]
    months: frozenset[int]
    weekdays: frozenset[int]
    day_of_month_any: bool
    weekday_any: bool

    def matches_day(self, day: date) -> bool:
        """Say whether the schedule fires at all on a calendar day.

        Follows the cron rule: when both day-of-month and day-of-week are
        restricted, a day matches if either one matches.

        Args:
            day: The calendar date to test.

        Returns:
            True if at least one time on ``day`` can fire.
        """
        if day.month not in self.months:
            return False
        day_of_month_matches = day.day in self.days_of_month
        weekday_matches = day.weekday() in self.weekdays
        if not self.day_of_month_any and not self.weekday_any:
            # Cron OR rule: "0 0 13 * 5" fires on the 13th and on every Friday.
            return day_of_month_matches or weekday_matches
        # A "*" field holds every value, so requiring both is the same as
        # requiring only the restricted one (or nothing when both are "*").
        return day_of_month_matches and weekday_matches

    def next_after(self, after: datetime) -> datetime:
        """Find the first fire time strictly after a moment.

        Walks the calendar day by day from ``after``'s local date. Wall times
        that do not exist (DST spring-forward gap) are skipped, and wall times
        that occur twice (fall back) fire once, on their first pass.

        Args:
            after: The moment to search from. Must be timezone-aware.

        Returns:
            An aware datetime in ``after.tzinfo`` with second and microsecond 0.

        Raises:
            ValueError: If ``after`` is naive.
            ScheduleError: If nothing matches within SEARCH_DAYS days, e.g. ``0 0 30 2 *``.

        Example:
            >>> parse_schedule("every 15m").next_after(datetime(2026, 1, 1, 0, 1, tzinfo=timezone.utc))
            datetime.datetime(2026, 1, 1, 0, 15, tzinfo=datetime.timezone.utc)
        """
        if after.utcoffset() is None:
            raise ValueError(f"next_after needs a timezone-aware datetime, got naive {after!r}")
        # Compare in UTC: two datetimes sharing one tzinfo compare by wall clock,
        # which misorders the two passes of an ambiguous fall-back hour.
        after_utc = after.astimezone(timezone.utc)
        first_day = after.date()
        for offset in range(SEARCH_DAYS):
            day = first_day + timedelta(days=offset)
            if not self.matches_day(day):
                continue
            slot = self._first_slot_on(day, after, after_utc)
            if slot is not None:
                return slot
        raise ScheduleError(
            f"Schedule '{self.expression}' never fires: no matching date in the next "
            f"{SEARCH_DAYS // 366} years. Check its day-of-month and month fields."
        )

    def _first_slot_on(self, day: date, after: datetime, after_utc: datetime) -> datetime | None:
        """Return the earliest valid fire time on ``day`` that is later than ``after``.

        Args:
            day: A calendar day for which ``matches_day`` is true.
            after: The search start; supplies the time zone of the result.
            after_utc: ``after`` converted to UTC, for ordering comparisons.

        Returns:
            The fire time, or None when no time on ``day`` qualifies.
        """
        is_first_day = day == after.date()
        for hour in sorted(self.hours):
            # On the day of ``after`` nothing earlier than its wall-clock hour
            # and minute can qualify, so skip those without building datetimes.
            if is_first_day and hour < after.hour:
                continue
            for minute in sorted(self.minutes):
                if is_first_day and (hour, minute) < (after.hour, after.minute):
                    continue
                # fold=0 (the default) picks the first pass of an ambiguous time.
                candidate = datetime(day.year, day.month, day.day, hour, minute, tzinfo=after.tzinfo)
                # The UTC comparison stays: it also rejects first-pass times that
                # lie before a second-pass ``after`` during a fall-back hour.
                if candidate.astimezone(timezone.utc) <= after_utc:
                    continue
                if _wall_time_exists(candidate):
                    return candidate
        return None


def _wall_time_exists(candidate: datetime) -> bool:
    """Say whether a wall-clock time exists in its zone.

    A time inside a spring-forward gap converts to UTC and back as a
    different wall time; every real time round-trips unchanged.

    Args:
        candidate: An aware datetime built from wall-clock fields.

    Returns:
        False only for times skipped by a DST jump.
    """
    round_trip = candidate.astimezone(timezone.utc).astimezone(candidate.tzinfo)
    return round_trip.replace(tzinfo=None) == candidate.replace(tzinfo=None)


def parse_schedule(text: str) -> Schedule:
    """Parse a schedule expression (cron, alias or friendly form).

    Case-insensitive and tolerant of extra whitespace.

    Args:
        text: The expression, e.g. ``"30 7 * * 1-5"``, ``"@daily"`` or ``"weekdays 08:00"``.

    Returns:
        The parsed Schedule; ``expression`` holds the stripped original text.

    Raises:
        ScheduleError: If the text is not a valid schedule. The message quotes
            the expression, says what is wrong and lists the accepted forms.

    Example:
        >>> parse_schedule("mon-fri 07:00").description
        'weekdays at 07:00'
    """
    expression = text.strip()
    normalized = " ".join(expression.lower().split())
    try:
        return _parse_normalized(expression, normalized)
    except _InvalidSchedule as problem:
        raise ScheduleError(
            f"Invalid schedule '{expression}': {problem}. {ACCEPTED_FORMS}"
        ) from problem


def _parse_normalized(expression: str, normalized: str) -> Schedule:
    """Pick the right parser for a lower-cased, single-spaced expression.

    Args:
        expression: The stripped original text, stored on the result.
        normalized: ``expression`` lower-cased with whitespace collapsed.

    Returns:
        The parsed Schedule.

    Raises:
        _InvalidSchedule: If the text matches no accepted form or is malformed.
    """
    words = normalized.split()
    if not words:
        raise _InvalidSchedule("the schedule is empty")
    if normalized.startswith("@"):
        if normalized not in CRON_ALIASES:
            raise _InvalidSchedule(f"unknown alias '{normalized}'")
        return _parse_cron(expression, CRON_ALIASES[normalized])
    if normalized == "hourly":
        return _hourly(expression)
    interval = _INTERVAL_RE.fullmatch(normalized)
    if interval:
        return _parse_interval(expression, int(interval.group(1)), interval.group(2))
    if len(words) == 3 and words[:2] == ["every", "day"]:
        return _parse_clock_schedule(expression, "daily", words[2])
    if words[0] == "every":
        raise _InvalidSchedule(
            "unrecognised 'every' form, use 'every 15m', 'every 2h' or 'every day 07:30'"
        )
    if len(words) == 2:
        return _parse_clock_schedule(expression, words[0], words[1])
    if len(words) == CRON_FIELD_COUNT:
        return _parse_cron(expression, normalized)
    raise _InvalidSchedule(
        f"unrecognised schedule (a cron expression needs {CRON_FIELD_COUNT} fields, got {len(words)})"
    )


def _friendly_schedule(
    expression: str,
    description: str,
    *,
    minutes: frozenset[int],
    hours: frozenset[int],
    weekdays: frozenset[int] = ALL_WEEKDAYS,
) -> Schedule:
    """Build a Schedule for a friendly form: every day of every month unless limited by weekday.

    Args:
        expression: The stripped original text.
        description: Readable summary.
        minutes: Allowed minutes.
        hours: Allowed hours.
        weekdays: Allowed weekdays in Python numbering; all days by default.

    Returns:
        The Schedule.
    """
    return Schedule(
        expression=expression,
        description=description,
        minutes=minutes,
        hours=hours,
        days_of_month=ALL_DAYS_OF_MONTH,
        months=ALL_MONTHS,
        weekdays=weekdays,
        day_of_month_any=True,
        weekday_any=weekdays == ALL_WEEKDAYS,
    )


def _hourly(expression: str) -> Schedule:
    """Build the "at minute 0 of every hour" schedule.

    Args:
        expression: The stripped original text.

    Returns:
        A Schedule described as "hourly".
    """
    return _friendly_schedule(expression, "hourly", minutes=frozenset({0}), hours=ALL_HOURS)


def _parse_interval(expression: str, count: int, unit: str) -> Schedule:
    """Build "every N minutes" or "every N hours" (``*/N`` in cron terms).

    Args:
        expression: The stripped original text.
        count: The N from the text.
        unit: The unit word, e.g. "m", "mins" or "hours".

    Returns:
        The Schedule.

    Raises:
        _InvalidSchedule: If N is zero or does not divide the hour (minutes) or day (hours) evenly.
    """
    is_minutes = unit in _MINUTE_UNITS
    cycle = MINUTES_PER_HOUR if is_minutes else HOURS_PER_DAY
    if count < 1 or cycle % count != 0:
        # An uneven step would restart at the top of every hour or day and give
        # a short last gap, so only exact divisors are allowed.
        divisors = ", ".join(str(n) for n in range(1, cycle + 1) if cycle % n == 0)
        raise _InvalidSchedule(
            f"'every {count} {'minutes' if is_minutes else 'hours'}' is not supported: N must be "
            f"one of {divisors}. Use a cron expression for other intervals"
        )
    if is_minutes:
        if count == MINUTES_PER_HOUR:
            return _hourly(expression)
        description = "every minute" if count == 1 else f"every {count} minutes"
        return _friendly_schedule(
            expression, description, minutes=frozenset(range(0, MINUTES_PER_HOUR, count)), hours=ALL_HOURS
        )
    if count == 1:
        return _hourly(expression)
    return _friendly_schedule(
        expression,
        f"every {count} hours",
        minutes=frozenset({0}),
        hours=frozenset(range(0, HOURS_PER_DAY, count)),
    )


def _parse_clock_schedule(expression: str, days_text: str, clock_text: str) -> Schedule:
    """Build a once-a-day schedule such as "weekdays 08:00" or "mon,wed 18:00".

    Args:
        expression: The stripped original text.
        days_text: ``daily``, ``weekdays``, ``weekends`` or a list of day names and ranges.
        clock_text: ``HH:MM``.

    Returns:
        The Schedule.

    Raises:
        _InvalidSchedule: If the days or the time are malformed.
    """
    weekdays = _parse_day_list(days_text)
    hour, minute = _parse_clock(clock_text)
    return _friendly_schedule(
        expression,
        _describe_daily(weekdays, hour, minute),
        minutes=frozenset({minute}),
        hours=frozenset({hour}),
        weekdays=weekdays,
    )


def _parse_clock(text: str) -> tuple[int, int]:
    """Parse ``HH:MM`` (hour 0-23 with 1-2 digits, minute 00-59 with 2 digits).

    Args:
        text: The time text.

    Returns:
        ``(hour, minute)``.

    Raises:
        _InvalidSchedule: If the text is not a valid time of day.
    """
    match = _CLOCK_RE.fullmatch(text)
    if not match:
        raise _InvalidSchedule(f"time '{text}' must look like HH:MM, for example 07:30")
    hour, minute = int(match.group(1)), int(match.group(2))
    if hour > LAST_HOUR or minute > LAST_MINUTE:
        raise _InvalidSchedule(
            f"time '{text}' is out of range (hour 0-{LAST_HOUR}, minute 00-{LAST_MINUTE})"
        )
    return hour, minute


def _parse_day_list(text: str) -> frozenset[int]:
    """Parse the days part of a friendly form into Python weekday numbers.

    Args:
        text: ``daily``, ``weekdays``, ``weekends``, or a comma list of day
            names (``mon`` or ``monday``) and ranges (``mon-fri``).

    Returns:
        The set of weekdays, 0 = Monday.

    Raises:
        _InvalidSchedule: On an unknown day name, an empty item or a backwards range.
    """
    shortcuts = {"daily": ALL_WEEKDAYS, "weekdays": MON_TO_FRI, "weekends": SAT_AND_SUN}
    if text in shortcuts:
        return shortcuts[text]
    days: set[int] = set()
    for item in text.split(","):
        if not item:
            raise _InvalidSchedule(f"the day list '{text}' has an empty item")
        first, dash, last = item.partition("-")
        start = _weekday_number(first)
        end = _weekday_number(last) if dash else start
        if start > end:
            raise _InvalidSchedule(f"the day range '{item}' runs backwards")
        days.update(range(start, end + 1))
    return frozenset(days)


def _weekday_number(name: str) -> int:
    """Convert a 3-letter or full English day name to a Python weekday number.

    Args:
        name: Lower-case day name, e.g. "wed" or "wednesday".

    Returns:
        0 for Monday through 6 for Sunday.

    Raises:
        _InvalidSchedule: If the name is not a day.
    """
    for names in (WEEKDAY_NAMES, FULL_WEEKDAY_NAMES):
        if name in names:
            return names.index(name)
    raise _InvalidSchedule(f"unknown day '{name}' (use mon, tue, wed, thu, fri, sat, sun)")


def _describe_daily(weekdays: frozenset[int], hour: int, minute: int) -> str:
    """Describe a once-a-day schedule in words.

    Args:
        weekdays: Allowed weekdays, 0 = Monday.
        hour: Hour of day.
        minute: Minute of the hour.

    Returns:
        E.g. "daily at 07:30", "weekdays at 08:00", "weekends at 10:00" or "Mon, Wed, Fri at 18:00".
    """
    clock = f"{hour:02d}:{minute:02d}"
    if weekdays == ALL_WEEKDAYS:
        return f"daily at {clock}"
    if weekdays == MON_TO_FRI:
        return f"weekdays at {clock}"
    if weekdays == SAT_AND_SUN:
        return f"weekends at {clock}"
    names = ", ".join(WEEKDAY_NAMES[day].capitalize() for day in sorted(weekdays))
    return f"{names} at {clock}"


def _parse_cron(expression: str, cron_text: str) -> Schedule:
    """Parse five whitespace-separated cron fields.

    Args:
        expression: The stripped original text, stored on the result (may be an alias).
        cron_text: Lower-cased ``minute hour day-of-month month day-of-week``.

    Returns:
        The Schedule.

    Raises:
        _InvalidSchedule: On a wrong field count or any invalid field.
    """
    fields = cron_text.split()
    if len(fields) != CRON_FIELD_COUNT:
        raise _InvalidSchedule(
            f"a cron expression needs {CRON_FIELD_COUNT} fields "
            f"(minute hour day-of-month month day-of-week), got {len(fields)}"
        )
    minute_text, hour_text, day_text, month_text, weekday_text = fields
    cron_weekdays = _parse_field(weekday_text, _WEEKDAY_FIELD)
    schedule = Schedule(
        expression=expression,
        description="",
        minutes=_parse_field(minute_text, _MINUTE_FIELD),
        hours=_parse_field(hour_text, _HOUR_FIELD),
        days_of_month=_parse_field(day_text, _DAY_OF_MONTH_FIELD),
        months=_parse_field(month_text, _MONTH_FIELD),
        # Cron counts 0 or 7 as Sunday and 1 as Monday; Python counts 0 as Monday.
        weekdays=frozenset((value + 6) % 7 for value in cron_weekdays),
        # Like Vixie cron, a field that starts with "*" (even "*/2") counts as
        # unrestricted, so "0 9 */2 * 1" means odd days that are Mondays (AND),
        # not odd days plus every Monday (OR).
        day_of_month_any=day_text.startswith("*"),
        weekday_any=weekday_text.startswith("*"),
    )
    # The description depends on the parsed sets, so it is filled in afterwards.
    return replace(schedule, description=_describe_cron(schedule))


def _describe_cron(schedule: Schedule) -> str:
    """Describe a cron schedule in words when it has a common shape, else show the cron text.

    Common shapes (all need every month):
        * minute 0 of every hour, every day: "hourly";
        * one minute, one hour, any day of month: "daily at HH:MM", "weekdays at
          HH:MM", "weekends at HH:MM" or "Mon, Wed at HH:MM" (by weekday set);
        * one minute, one hour, one day of month, any weekday: "monthly on day D at HH:MM".

    Args:
        schedule: A schedule parsed from cron fields.

    Returns:
        The words, or "cron: <expression>" for any other shape.
    """
    fallback = f"cron: {schedule.expression}"
    if schedule.months != ALL_MONTHS:
        return fallback
    if (
        schedule.minutes == frozenset({0})
        and schedule.hours == ALL_HOURS
        and schedule.day_of_month_any
        and schedule.days_of_month == ALL_DAYS_OF_MONTH
        and schedule.weekdays == ALL_WEEKDAYS
    ):
        return "hourly"
    if len(schedule.minutes) != 1 or len(schedule.hours) != 1:
        return fallback
    (hour,) = schedule.hours
    (minute,) = schedule.minutes
    # Both the flag and the set are checked: "*/2" starts with "*" but still skips
    # days, and an explicit "1-31" is a full set that still triggers the OR rule.
    if schedule.day_of_month_any and schedule.days_of_month == ALL_DAYS_OF_MONTH:
        return _describe_daily(schedule.weekdays, hour, minute)
    if (
        len(schedule.days_of_month) == 1
        and schedule.weekday_any
        and schedule.weekdays == ALL_WEEKDAYS
    ):
        (day,) = schedule.days_of_month
        return f"monthly on day {day} at {hour:02d}:{minute:02d}"
    return fallback


def _parse_field(text: str, spec: _FieldSpec) -> frozenset[int]:
    """Parse one cron field: a comma list of ``*``, ``N``, ``A-B``, ``*/S``, ``A-B/S``, ``N/S``.

    Args:
        text: The lower-cased field text.
        spec: Range and names for this field.

    Returns:
        Every value the field allows, in the field's own numbering.

    Raises:
        _InvalidSchedule: On an empty item, a bad value, a value out of range,
            a step below 1 or a backwards range.
    """
    values: set[int] = set()
    for item in text.split(","):
        if not item:
            raise _InvalidSchedule(f"the {spec.label} field '{text}' has an empty item")
        values.update(_parse_item(item, spec))
    return frozenset(values)


def _parse_item(item: str, spec: _FieldSpec) -> range:
    """Parse one comma-separated item of a cron field.

    Args:
        item: E.g. ``*``, ``5``, ``1-5``, ``*/15``, ``10-40/5`` or ``5/10``.
        spec: Range and names for the field.

    Returns:
        The values the item selects.

    Raises:
        _InvalidSchedule: If the item is malformed or out of range.
    """
    base, slash, step_text = item.partition("/")
    step = 1
    if slash:
        if not (step_text.isascii() and step_text.isdigit()) or int(step_text) < 1:
            raise _InvalidSchedule(
                f"the step in '{item}' ({spec.label} field) must be a whole number of 1 or more"
            )
        step = int(step_text)
    if base == "*":
        start, end = spec.low, spec.high
    elif "-" in base:
        first, _, last = base.partition("-")
        start, end = _parse_value(first, spec, item), _parse_value(last, spec, item)
        if start > end:
            raise _InvalidSchedule(f"the range '{item}' in the {spec.label} field runs backwards")
    else:
        start = _parse_value(base, spec, item)
        # "N/S" means "from N to the end of the field in steps of S".
        end = spec.high if slash else start
    return range(start, end + 1, step)


def _parse_value(token: str, spec: _FieldSpec, item: str) -> int:
    """Convert a number or name to a value and check it against the field's range.

    Args:
        token: A decimal number or one of the field's names.
        spec: Range and names for the field.
        item: The whole item, for error messages.

    Returns:
        The value in the field's own numbering.

    Raises:
        _InvalidSchedule: If the token is not a number or name, or is out of range.
    """
    if token in spec.names:
        return spec.names[token]
    if not (token.isascii() and token.isdigit()):
        raise _InvalidSchedule(f"'{token}' in '{item}' is not a valid {spec.label} value")
    value = int(token)
    if not spec.low <= value <= spec.high:
        raise _InvalidSchedule(
            f"{spec.label} value {value} in '{item}' is out of range ({spec.low}-{spec.high})"
        )
    return value
