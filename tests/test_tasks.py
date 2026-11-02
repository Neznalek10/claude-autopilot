"""Unit tests for autopilot.tasks.

Covers merging of defaults, every validation rule (each error must name the
task and the field), file loading and the shipped tasks.example.toml.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import Any

from autopilot.errors import ConfigError, ScheduleError
from autopilot.models import DEFAULT_PERMISSION_MODE, NOTIFY_ALWAYS
from autopilot.tasks import (
    DEFAULT_ALLOWED_TOOLS,
    load_task_config,
    parse_task_config,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class TaskConfigTestCase(unittest.TestCase):
    """Base class providing a temp base folder and a parse helper."""

    def setUp(self) -> None:
        """Create a temp base folder with a sub folder."""
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.base = Path(self._tmp.name).resolve()
        (self.base / "work").mkdir()

    def parse(self, data: dict[str, Any]):
        """Parse ``data`` against the temp base folder."""
        return parse_task_config(data, self.base)

    def task(self, **fields: Any) -> dict[str, Any]:
        """Return a minimal valid task table updated with ``fields``."""
        return {"name": "news", "prompt": "Say hi.", **fields}

    def assertConfigError(self, data: dict[str, Any], *fragments: str) -> None:
        """Assert parsing ``data`` raises ConfigError containing every fragment."""
        with self.assertRaises(ConfigError) as ctx:
            self.parse(data)
        for fragment in fragments:
            self.assertIn(fragment, str(ctx.exception))


class DefaultsAndMergeTest(TaskConfigTestCase):
    """Built-in defaults, [defaults] and per-task overrides."""

    def test_empty_config(self) -> None:
        """No tasks is valid and ask options use built-in defaults."""
        config = self.parse({})
        self.assertEqual(config.tasks, {})
        self.assertEqual(config.ask_options.timeout_seconds, 15 * 60)
        self.assertEqual(config.ask_options.workdir, self.base)

    def test_minimal_task_gets_builtin_defaults(self) -> None:
        """A task with name and prompt only gets every default."""
        task = self.parse({"task": [self.task(prompt="  Say hi.  ")]}).tasks["news"]
        self.assertEqual(task.prompt, "Say hi.")
        self.assertEqual(task.schedules, ())
        self.assertTrue(task.enabled)
        self.assertEqual(task.notify, NOTIFY_ALWAYS)
        self.assertEqual(task.description, "")
        options = task.options
        self.assertEqual(options.workdir, self.base)
        self.assertEqual(options.allowed_tools, DEFAULT_ALLOWED_TOOLS)
        self.assertEqual(options.disallowed_tools, ())
        self.assertEqual(options.permission_mode, DEFAULT_PERMISSION_MODE)
        self.assertEqual(options.timeout_seconds, 30 * 60)
        self.assertIsNone(options.model)
        self.assertIsNone(options.max_budget_usd)

    def test_defaults_then_task_override(self) -> None:
        """The task overrides [defaults], which override built-ins."""
        config = self.parse({
            "defaults": {"model": "sonnet", "notify": "errors", "enabled": False, "timeout_minutes": 5},
            "task": [self.task(), self.task(name="other", model="opus", notify="never", enabled=True)],
        })
        first, other = config.tasks["news"], config.tasks["other"]
        self.assertEqual((first.options.model, first.notify, first.enabled), ("sonnet", "errors", False))
        self.assertEqual(first.options.timeout_seconds, 300)
        self.assertEqual((other.options.model, other.notify, other.enabled), ("opus", "never", True))

    def test_ask_inherits_defaults_options_and_overrides(self) -> None:
        """[ask] layers over [defaults] option keys; notify/enabled do not apply."""
        config = self.parse({
            "defaults": {"model": "sonnet", "effort": "low", "notify": "never"},
            "ask": {"effort": "high", "timeout_minutes": 3},
        })
        self.assertEqual(config.ask_options.model, "sonnet")
        self.assertEqual(config.ask_options.effort, "high")
        self.assertEqual(config.ask_options.timeout_seconds, 180)

    def test_defaults_timeout_applies_to_ask(self) -> None:
        """A timeout in [defaults] replaces the built-in ask timeout."""
        config = self.parse({"defaults": {"timeout_minutes": 40}})
        self.assertEqual(config.ask_options.timeout_seconds, 40 * 60)

    def test_all_option_values(self) -> None:
        """Every option key is parsed into RunOptions."""
        task = self.parse({"task": [self.task(
            workdir="work", model="claude-sonnet-4-5", effort="max",
            allowed_tools=["Read", "Bash(git log *)", "mcp__srv__tool"], disallowed_tools=["Edit"],
            permission_mode="plan", add_dirs=["work", str(self.base)], append_system_prompt="Be brief.",
            max_budget_usd=2, timeout_minutes=720, description="Daily news",
        )]}).tasks["news"]
        options = task.options
        self.assertEqual(options.workdir, self.base / "work")
        self.assertEqual(options.allowed_tools, ("Read", "Bash(git log *)", "mcp__srv__tool"))
        self.assertEqual(options.disallowed_tools, ("Edit",))
        self.assertEqual(options.add_dirs, (self.base / "work", self.base))
        self.assertEqual(options.max_budget_usd, 2.0)
        self.assertIsInstance(options.max_budget_usd, float)
        self.assertEqual(options.timeout_seconds, 720 * 60)
        self.assertEqual(options.append_system_prompt, "Be brief.")
        self.assertEqual(task.description, "Daily news")

    def test_tilde_workdir(self) -> None:
        """'~' expands to the home folder."""
        task = self.parse({"task": [self.task(workdir="~")]}).tasks["news"]
        self.assertEqual(task.options.workdir, Path.home().resolve())

    def test_task_order_and_unicode_prompt(self) -> None:
        """Tasks keep file order and unicode prompts survive."""
        config = self.parse({"task": [self.task(name="b", prompt="Příliš žluťoučký kůň"), self.task(name="a")]})
        self.assertEqual(list(config.tasks), ["b", "a"])
        self.assertEqual(config.tasks["b"].prompt, "Příliš žluťoučký kůň")


class ScheduleTest(TaskConfigTestCase):
    """The schedule field."""

    def test_string_and_list(self) -> None:
        """A single string and a list both produce Schedule tuples."""
        single = self.parse({"task": [self.task(schedule="daily 07:30")]}).tasks["news"]
        many = self.parse({"task": [self.task(schedule=["daily 07:30", "0 12 * * 1"])]}).tasks["news"]
        self.assertEqual([s.expression for s in single.schedules], ["daily 07:30"])
        self.assertEqual(len(many.schedules), 2)

    def test_empty_list_is_manual_only(self) -> None:
        """An empty array means manual only."""
        self.assertEqual(self.parse({"task": [self.task(schedule=[])]}).tasks["news"].schedules, ())

    def test_invalid_expression_names_task_and_field(self) -> None:
        """A bad expression is a ScheduleError naming the task and 'schedule'."""
        with self.assertRaises(ScheduleError) as ctx:
            self.parse({"task": [self.task(schedule="whenever")]})
        self.assertIn("tasks.toml: task 'news': 'schedule'", str(ctx.exception))
        self.assertIn("whenever", str(ctx.exception))

    def test_impossible_schedule_fails_at_load(self) -> None:
        """A schedule that parses but never fires is rejected."""
        with self.assertRaises(ScheduleError) as ctx:
            self.parse({"task": [self.task(schedule=["daily 07:30", "0 0 30 2 *"])]})
        self.assertIn("'schedule'", str(ctx.exception))

    def test_wrong_types(self) -> None:
        """Non-string schedule values are rejected."""
        for value in (5, [1], ["daily 07:30", 2], {"a": 1}):
            with self.subTest(value=value):
                self.assertConfigError({"task": [self.task(schedule=value)]}, "task 'news'", "'schedule'")


class ValidationTest(TaskConfigTestCase):
    """Field validation and error message shape."""

    def test_field_errors_name_task_and_field(self) -> None:
        """Each invalid value raises a ConfigError naming the task and the field."""
        cases = [
            ("workdir", "missing-folder"), ("workdir", 5), ("workdir", ""),
            ("model", "bad model!"), ("model", 3),
            ("effort", "extreme"),
            ("allowed_tools", "Read"), ("allowed_tools", ["Bad Tool"]), ("allowed_tools", [3]),
            ("disallowed_tools", ["Bash("]),
            ("permission_mode", "yolo"),
            ("add_dirs", "work"), ("add_dirs", ["nope"]),
            ("append_system_prompt", ""), ("append_system_prompt", 1),
            ("max_budget_usd", 0), ("max_budget_usd", -1), ("max_budget_usd", "5"),
            ("max_budget_usd", True), ("max_budget_usd", float("inf")), ("max_budget_usd", float("nan")),
            ("timeout_minutes", 0), ("timeout_minutes", 721), ("timeout_minutes", 1.5),
            ("timeout_minutes", True), ("timeout_minutes", "10"),
            ("enabled", "yes"), ("enabled", 1),
            ("notify", "sometimes"),
            ("description", ""),
        ]
        for field, value in cases:
            with self.subTest(field=field, value=value):
                self.assertConfigError(
                    {"task": [self.task(**{field: value})]}, "tasks.toml: task 'news': ", f"'{field}'"
                )

    def test_boundary_timeouts_accepted(self) -> None:
        """1 and 720 minutes are valid."""
        for minutes in (1, 720):
            options = self.parse({"task": [self.task(timeout_minutes=minutes)]}).tasks["news"].options
            self.assertEqual(options.timeout_seconds, minutes * 60)

    def test_missing_name_and_prompt(self) -> None:
        """Missing required keys are reported; a missing name uses the position."""
        self.assertConfigError({"task": [self.task(), {"prompt": "x"}]}, "task #2: ", "'name' is required")
        self.assertConfigError({"task": [{"name": "news"}]}, "task 'news': ", "'prompt' is required")

    def test_empty_prompt(self) -> None:
        """A blank prompt is rejected."""
        self.assertConfigError({"task": [self.task(prompt="   ")]}, "'prompt'")

    def test_bad_names(self) -> None:
        """Names must match the pattern, be unique and not be 'ask'."""
        for name in ("Upper", "has space", "-lead", "x" * 41, "", 7):
            with self.subTest(name=name):
                self.assertConfigError({"task": [self.task(name=name)]}, "'name'")
        self.assertConfigError({"task": [self.task(name="ask")]}, "task 'ask'", "reserved")
        self.assertConfigError({"task": [self.task(), self.task()]}, "task 'news'", "already used")

    def test_name_of_40_characters_ok(self) -> None:
        """The longest allowed name is accepted."""
        self.assertIn("a" * 40, self.parse({"task": [self.task(name="a" * 40)]}).tasks)

    def test_unknown_keys(self) -> None:
        """Typos are errors in tasks, [defaults], [ask] and at the top level."""
        self.assertConfigError({"task": [self.task(shedule="daily 07:30")]}, "task 'news'", "'shedule'", "schedule")
        self.assertConfigError({"defaults": {"name": "x"}}, "[defaults]", "'name'")
        self.assertConfigError({"ask": {"enabled": True}}, "[ask]", "'enabled'")
        self.assertConfigError({"tasks": []}, "'tasks'", "task")

    def test_defaults_errors_are_blamed_on_defaults(self) -> None:
        """A bad default is reported against [defaults], with the field."""
        self.assertConfigError({"defaults": {"effort": "extreme"}, "task": [self.task()]}, "[defaults]", "'effort'")
        self.assertConfigError({"defaults": {"notify": "x"}}, "[defaults]", "'notify'")
        self.assertConfigError({"defaults": {"enabled": "x"}}, "[defaults]", "'enabled'")

    def test_ask_errors(self) -> None:
        """A bad [ask] value is reported against [ask]."""
        self.assertConfigError({"ask": {"timeout_minutes": 0}}, "[ask]", "'timeout_minutes'")

    def test_structure_errors(self) -> None:
        """Wrongly shaped tables are rejected."""
        self.assertConfigError({"defaults": []}, "[defaults]", "table")
        self.assertConfigError({"ask": "x"}, "[ask]", "table")
        self.assertConfigError({"task": {"name": "x"}}, "[[task]]")
        self.assertConfigError({"task": ["x"]}, "task #1", "table")

    def test_source_is_stored(self) -> None:
        """The source argument ends up on the config."""
        source = self.base / "tasks.toml"
        self.assertEqual(parse_task_config({}, self.base, source=source).source, source)


class LoadTaskConfigTest(TaskConfigTestCase):
    """load_task_config: file handling."""

    def write(self, text: str, name: str = "tasks.toml") -> Path:
        """Write ``text`` into the temp folder and return the path."""
        path = self.base / name
        path.write_text(text, encoding="utf-8")
        return path

    def test_loads_file_and_resolves_relative_to_its_folder(self) -> None:
        """Relative workdir resolves against the file's folder; source is set."""
        path = self.write('[[task]]\nname = "a"\nprompt = "p"\nworkdir = "work"\nschedule = "hourly"\n')
        config = load_task_config(path)
        self.assertEqual(config.source, path)
        self.assertEqual(config.tasks["a"].options.workdir, self.base / "work")

    def test_missing_file(self) -> None:
        """A missing file points to `autopilot init`."""
        with self.assertRaises(ConfigError) as ctx:
            load_task_config(self.base / "nope.toml")
        self.assertIn("not found", str(ctx.exception))
        self.assertIn("autopilot init", str(ctx.exception))

    def test_syntax_error_has_line_and_column(self) -> None:
        """TOML syntax errors carry the position."""
        path = self.write('[[task]]\nname = "a\n')
        with self.assertRaises(ConfigError) as ctx:
            load_task_config(path)
        self.assertIn("line 2", str(ctx.exception))
        self.assertIn("column", str(ctx.exception))

    def test_invalid_utf8(self) -> None:
        """Undecodable bytes are a ConfigError, not a crash."""
        path = self.base / "tasks.toml"
        path.write_bytes(b"\xff\xfe\x00")
        with self.assertRaises(ConfigError):
            load_task_config(path)

    def test_validation_error_propagates(self) -> None:
        """Validation errors from a file name the task and field."""
        path = self.write('[[task]]\nname = "a"\nprompt = "p"\neffort = "x"\n')
        with self.assertRaises(ConfigError) as ctx:
            load_task_config(path)
        self.assertIn("task 'a'", str(ctx.exception))
        self.assertIn("'effort'", str(ctx.exception))


class ExampleFileTest(unittest.TestCase):
    """The shipped tasks.example.toml must be valid on a fresh machine."""

    def test_example_loads(self) -> None:
        """tasks.example.toml validates and contains the three documented tasks."""
        config = load_task_config(PROJECT_ROOT / "tasks.example.toml")
        self.assertEqual(list(config.tasks), ["ai-news", "day-plan", "weekly-review"])
        news, plan, review = config.tasks.values()
        self.assertEqual(news.schedules[0].expression, "daily 07:30")
        self.assertEqual(news.options.model, "sonnet")
        self.assertEqual(plan.schedules[0].expression, "weekdays 06:45")
        self.assertEqual(plan.options.allowed_tools, ("Bash(gws calendar *)", "Bash(gws tasks *)"))
        self.assertEqual(review.schedules[0].expression, "sun 19:00")
        self.assertFalse(review.enabled)
        self.assertEqual(news.options.workdir, Path.home().resolve())


if __name__ == "__main__":
    unittest.main()
