"""Unit tests for autopilot.service: unit rendering, install, uninstall and linger detection.

Every systemctl and loginctl call goes through a FakeRunner that records the
arguments and returns scripted results, and every unit file is written into a
temporary folder. The real ~/.config and the real systemd are never touched.
"""

from __future__ import annotations

import tempfile
import unittest
from collections.abc import Sequence
from pathlib import Path

from autopilot.errors import AutopilotError
from autopilot.service import (
    UNIT_NAME,
    default_unit_dir,
    install_service,
    linger_enabled,
    render_unit,
    uninstall_service,
)

# Interpreter and project folder used by the install and uninstall tests.
TEST_PYTHON = Path("/srv/venv/bin/python")
TEST_HOME = Path("/srv/autopilot")
TEST_PATH = "/usr/bin:/bin"
# Exit code and output triple that a successful systemctl call returns.
SUCCESS = (0, "", "")
# Exit code systemctl uses for a failed call.
FAILURE_EXIT_CODE = 1
# systemctl's stderr when the user manager is unreachable.
BUS_ERROR = "Failed to connect to bus"
# systemctl's stderr when disable --now is asked about a unit it doesn't know.
NOT_LOADED_MESSAGE = "Unit autopilot.service not loaded."


class FakeRunner:
    """A CommandRunner stand-in that records every call and answers from a script.

    Attributes:
        calls: The argument tuples received, in call order.
        timeouts: The timeout passed with each call, same order as ``calls``.
    """

    def __init__(self, scripted: dict[str, tuple[int, str, str]] | None = None) -> None:
        """Create a runner that answers every call with success unless scripted.

        Args:
            scripted: Maps a word such as "disable" or "Linger" to the result returned
                for any call whose argument list contains that word exactly. The first
                matching entry wins.
        """
        self._scripted = scripted or {}
        self.calls: list[tuple[str, ...]] = []
        self.timeouts: list[float] = []

    def __call__(self, args: Sequence[str], timeout: float) -> tuple[int, str, str]:
        """Record the call and return the scripted or default result.

        Args:
            args: The command and its arguments.
            timeout: The timeout the caller asked for.

        Returns:
            The scripted ``(exit_code, stdout, stderr)`` tuple, or success.
        """
        self.calls.append(tuple(args))
        self.timeouts.append(timeout)
        for word, result in self._scripted.items():
            if word in args:
                return result
        return SUCCESS

    def systemctl_calls(self) -> list[tuple[str, ...]]:
        """Return only the systemctl calls, in order.

        Returns:
            The recorded calls whose program is systemctl.
        """
        return [call for call in self.calls if call[0] == "systemctl"]


class RenderUnitTest(unittest.TestCase):
    """The text of the generated systemd unit file."""

    def render(
        self,
        *,
        home: str = "/srv/autopilot",
        python: str = "/srv/venv/bin/python",
        path: str = TEST_PATH,
    ) -> str:
        """Render a unit with test defaults, overriding only the values a test cares about.

        Args:
            home: Project folder value.
            python: Interpreter path value.
            path: PATH value.

        Returns:
            The rendered unit file text.
        """
        return render_unit(python=Path(python), project_home=Path(home), path_env=path)

    def test_unit_contains_every_required_setting(self) -> None:
        """The rendered unit has the sections, restart policy and environment the service needs."""
        lines = set(self.render().splitlines())
        expected = {
            "[Unit]",
            "Description=Claude Autopilot: scheduled Claude Code tasks",
            "After=network-online.target",
            "Wants=network-online.target",
            "StartLimitIntervalSec=300",
            "StartLimitBurst=5",
            "[Service]",
            "Type=simple",
            "WorkingDirectory=/srv/autopilot",
            'Environment="AUTOPILOT_HOME=/srv/autopilot"',
            f'Environment="PATH={TEST_PATH}"',
            'Environment="PYTHONUNBUFFERED=1"',
            "ExecStart=/srv/venv/bin/python -m autopilot start",
            "Restart=on-failure",
            "RestartSec=15",
            "[Install]",
            "WantedBy=default.target",
        }
        self.assertTrue(expected.issubset(lines), f"missing: {expected - lines}")

    def test_unit_ends_with_a_single_newline(self) -> None:
        """The unit file ends with exactly one newline, so systemd and diffs read it cleanly."""
        text = self.render()
        self.assertTrue(text.endswith("WantedBy=default.target\n"))
        self.assertFalse(text.endswith("\n\n"))

    def test_percent_signs_are_doubled_in_every_value(self) -> None:
        """A literal % in any value is escaped, so systemd doesn't expand it as a specifier."""
        text = self.render(home="/srv/100%", python="/srv/%u/python", path="/usr/bin:%h")
        self.assertIn("WorkingDirectory=/srv/100%%", text)
        self.assertIn('Environment="AUTOPILOT_HOME=/srv/100%%"', text)
        self.assertIn('Environment="PATH=/usr/bin:%%h"', text)
        self.assertIn("ExecStart=/srv/%%u/python -m autopilot start", text)

    def test_dollar_signs_in_exec_start_are_doubled_so_systemd_keeps_them_literal(self) -> None:
        """systemd expands $NAME in ExecStart, so a $ in the interpreter path is written as $$."""
        quoted = self.render(python="/home/a b/$HOME/venv/bin/python")
        self.assertIn('ExecStart="/home/a b/$$HOME/venv/bin/python" -m autopilot start', quoted)
        bare = self.render(python="/srv/${NAME}/python")
        self.assertIn("ExecStart=/srv/$${NAME}/python -m autopilot start", bare)

    def test_interpreter_path_with_spaces_is_quoted_in_exec_start(self) -> None:
        """A space in the interpreter path would split ExecStart, so the path is quoted."""
        text = self.render(python="/opt/my venv/bin/python")
        self.assertIn('ExecStart="/opt/my venv/bin/python" -m autopilot start', text)

    def test_path_value_with_spaces_stays_one_quoted_assignment(self) -> None:
        """A PATH entry with a space stays inside one quoted Environment assignment."""
        text = self.render(path="/usr/bin:/home/u/my bin")
        self.assertIn('Environment="PATH=/usr/bin:/home/u/my bin"', text)

    def test_quotes_and_backslashes_are_escaped_inside_quoted_values(self) -> None:
        """Double quotes and backslashes in a quoted value are escaped for systemd's parser."""
        raw_python = '/opt/a "b"\\c d/python'
        text = self.render(python=raw_python)
        self.assertIn('ExecStart="/opt/a \\"b\\"\\\\c d/python" -m autopilot start', text)

    def test_unicode_project_path_is_written_unchanged(self) -> None:
        """A non-ASCII project folder is written as-is, so the service starts in the right place."""
        text = self.render(home="/home/u/Příliš žluťoučký kůň")
        self.assertIn("WorkingDirectory=/home/u/Příliš žluťoučký kůň", text)

    def test_newline_in_a_value_is_rejected(self) -> None:
        """A line break in any value would inject unit directives, so rendering refuses it."""
        with self.assertRaises(ValueError):
            self.render(path="/usr/bin\n[Install]\nWantedBy=multi-user.target")


class DefaultUnitDirTest(unittest.TestCase):
    """Where the user's systemd units live."""

    def test_default_unit_dir_is_the_systemd_user_folder_under_home(self) -> None:
        """The default is ~/.config/systemd/user, the folder systemd reads user units from."""
        self.assertEqual(default_unit_dir(), Path.home() / ".config" / "systemd" / "user")


class InstallServiceTest(unittest.TestCase):
    """install_service writes the unit and enables it through systemctl."""

    def setUp(self) -> None:
        """Create a temporary folder to hold the unit file."""
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.unit_dir = self.root / "systemd" / "user"

    def install(self, run: FakeRunner, unit_dir: Path | None = None) -> list[str]:
        """Install with the test interpreter, home and PATH.

        Args:
            run: The fake command runner to record calls on.
            unit_dir: Folder for the unit; defaults to the test folder.

        Returns:
            The step lines returned by install_service.
        """
        return install_service(
            python=TEST_PYTHON,
            project_home=TEST_HOME,
            path_env=TEST_PATH,
            run=run,
            unit_dir=self.unit_dir if unit_dir is None else unit_dir,
        )

    def test_writes_the_rendered_unit_file_into_the_unit_folder(self) -> None:
        """The unit file on disk is exactly what render_unit produces."""
        self.install(FakeRunner())
        written = (self.unit_dir / UNIT_NAME).read_text(encoding="utf-8")
        expected = render_unit(python=TEST_PYTHON, project_home=TEST_HOME, path_env=TEST_PATH)
        self.assertEqual(written, expected)

    def test_reloads_systemd_then_enables_and_starts_the_service(self) -> None:
        """daemon-reload runs before enable --now, so systemd sees the new unit."""
        run = FakeRunner()
        self.install(run)
        self.assertEqual(
            run.systemctl_calls(),
            [
                ("systemctl", "--user", "daemon-reload"),
                ("systemctl", "--user", "enable", "--now", UNIT_NAME),
            ],
        )

    def test_returns_a_step_line_for_each_action(self) -> None:
        """The result describes the unit written, the reload and the enable, in that order."""
        steps = self.install(FakeRunner(scripted={"show-user": (0, "Linger=yes\n", "")}))
        self.assertEqual(len(steps), 3)
        self.assertIn("Wrote", steps[0])
        self.assertIn("Reloaded", steps[1])
        self.assertIn("Enabled", steps[2])

    def test_creates_a_missing_unit_folder(self) -> None:
        """A user with no ~/.config/systemd/user yet still gets the unit installed."""
        nested = self.root / "a" / "b" / "c"
        self.install(FakeRunner(), unit_dir=nested)
        self.assertTrue((nested / UNIT_NAME).is_file())

    def test_adds_linger_hint_when_linger_is_off(self) -> None:
        """With linger off the service stops at logout, so the output says how to enable it."""
        steps = self.install(FakeRunner(scripted={"show-user": (0, "Linger=no\n", "")}))
        self.assertTrue(any("sudo loginctl enable-linger" in step for step in steps))

    def test_no_linger_hint_when_linger_is_on(self) -> None:
        """With linger on there is nothing to tell the user, so no hint is added."""
        steps = self.install(FakeRunner(scripted={"show-user": (0, "Linger=yes\n", "")}))
        self.assertFalse(any("enable-linger" in step for step in steps))

    def test_daemon_reload_failure_raises_with_systemctl_output(self) -> None:
        """A failed daemon-reload stops the install and quotes systemctl's own message."""
        run = FakeRunner(scripted={"daemon-reload": (FAILURE_EXIT_CODE, "", BUS_ERROR)})
        with self.assertRaises(AutopilotError) as caught:
            self.install(run)
        self.assertIn("daemon-reload", str(caught.exception))
        self.assertIn(BUS_ERROR, str(caught.exception))

    def test_enable_failure_raises_with_systemctl_output(self) -> None:
        """A failed enable --now stops the install and quotes systemctl's own message."""
        run = FakeRunner(scripted={"enable": (FAILURE_EXIT_CODE, "", "Unit file is masked.")})
        with self.assertRaises(AutopilotError) as caught:
            self.install(run)
        self.assertIn("Unit file is masked.", str(caught.exception))

    def test_unwritable_unit_folder_raises_before_calling_systemctl(self) -> None:
        """An unusable unit folder raises before systemctl is called."""
        blocker = self.root / "blocker"
        blocker.write_text("a file, not a folder", encoding="utf-8")
        run = FakeRunner()
        with self.assertRaises(AutopilotError) as caught:
            self.install(run, unit_dir=blocker / "systemd")
        self.assertIn("Could not write", str(caught.exception))
        self.assertEqual(run.calls, [])


class UninstallServiceTest(unittest.TestCase):
    """uninstall_service stops the service, removes the unit and reloads systemd."""

    def setUp(self) -> None:
        """Create a temporary unit folder that already holds an installed unit file."""
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.unit_dir = Path(temp.name) / "user"
        self.unit_dir.mkdir()
        self.unit_path = self.unit_dir / UNIT_NAME
        self.unit_path.write_text("[Unit]\n", encoding="utf-8")

    def test_removes_the_unit_file_and_reloads_systemd(self) -> None:
        """A normal uninstall deletes the unit file and runs daemon-reload afterwards."""
        run = FakeRunner()
        uninstall_service(run=run, unit_dir=self.unit_dir)
        self.assertFalse(self.unit_path.exists())
        self.assertEqual(
            run.systemctl_calls(),
            [
                ("systemctl", "--user", "disable", "--now", UNIT_NAME),
                ("systemctl", "--user", "daemon-reload"),
            ],
        )

    def test_not_loaded_failure_is_tolerated(self) -> None:
        """A "not loaded" answer from disable means the service is gone, so uninstall goes on."""
        run = FakeRunner(scripted={"disable": (FAILURE_EXIT_CODE, "", NOT_LOADED_MESSAGE)})
        steps = uninstall_service(run=run, unit_dir=self.unit_dir)
        self.assertFalse(self.unit_path.exists())
        self.assertTrue(any("not loaded" in step for step in steps))

    def test_already_removed_unit_is_tolerated(self) -> None:
        """A "does not exist" answer from disable, as on a second uninstall, is not an error."""
        message = "Failed to disable unit: Unit file autopilot.service does not exist."
        run = FakeRunner(scripted={"disable": (FAILURE_EXIT_CODE, "", message)})
        uninstall_service(run=run, unit_dir=self.unit_dir)
        self.assertFalse(self.unit_path.exists())

    def test_other_disable_failure_raises_and_keeps_the_unit_file(self) -> None:
        """A real disable failure stops uninstall before the unit is deleted, for a retry."""
        run = FakeRunner(scripted={"disable": (FAILURE_EXIT_CODE, "", "Access denied")})
        with self.assertRaises(AutopilotError) as caught:
            uninstall_service(run=run, unit_dir=self.unit_dir)
        self.assertIn("Access denied", str(caught.exception))
        self.assertTrue(self.unit_path.exists())

    def test_missing_unit_file_is_reported_without_error(self) -> None:
        """When no unit file exists, the step says so and uninstall still succeeds."""
        self.unit_path.unlink()
        steps = uninstall_service(run=FakeRunner(), unit_dir=self.unit_dir)
        self.assertTrue(any("nothing to remove" in step for step in steps))

    def test_daemon_reload_failure_raises(self) -> None:
        """A failed daemon-reload after removal is reported, not hidden."""
        run = FakeRunner(scripted={"daemon-reload": (FAILURE_EXIT_CODE, "", BUS_ERROR)})
        with self.assertRaises(AutopilotError) as caught:
            uninstall_service(run=run, unit_dir=self.unit_dir)
        self.assertIn(BUS_ERROR, str(caught.exception))


class LingerEnabledTest(unittest.TestCase):
    """linger_enabled reads loginctl's Linger property."""

    def test_asks_loginctl_for_the_linger_property_of_the_user(self) -> None:
        """The query names the user and asks only for the Linger property."""
        run = FakeRunner(scripted={"show-user": (0, "Linger=yes\n", "")})
        linger_enabled(run, "alice")
        self.assertEqual(run.calls, [("loginctl", "show-user", "alice", "--property=Linger")])

    def test_reports_true_and_false_from_the_linger_line(self) -> None:
        """A "yes" line means linger is on, a "no" line means it is off."""
        cases = {"Linger=yes\n": True, "Linger=no\n": False}
        for output, expected in cases.items():
            with self.subTest(output=output):
                run = FakeRunner(scripted={"show-user": (0, output, "")})
                self.assertIs(linger_enabled(run, "alice"), expected)

    def test_returns_none_when_loginctl_fails(self) -> None:
        """A failing loginctl gives an unknown answer, not a guess."""
        run = FakeRunner(scripted={"show-user": (FAILURE_EXIT_CODE, "", "No such user")})
        self.assertIsNone(linger_enabled(run, "alice"))

    def test_returns_none_when_the_output_has_no_linger_line(self) -> None:
        """Output without a recognizable Linger line gives an unknown answer."""
        run = FakeRunner(scripted={"show-user": (0, "Something=else\nLinger=maybe\n", "")})
        self.assertIsNone(linger_enabled(run, "alice"))


if __name__ == "__main__":
    unittest.main()
