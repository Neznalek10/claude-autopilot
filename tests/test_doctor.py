"""Unit tests for autopilot.doctor.

Covers every check (happy path, warnings, failures), the skip behaviour when the
Claude binary is missing, the privacy rule that the account email never reaches
a report, and the report format. Nothing touches the real network, systemd,
~/.config or the real Claude CLI: commands go through a scripted runner, the
Telegram client is a fake and Claude is tests/fake_claude.py.
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

from autopilot.doctor import (
    CheckResult,
    check_api_key,
    check_claude_binary,
    check_claude_login,
    check_data_dir,
    check_live,
    check_service,
    check_tasks,
    check_telegram,
    format_report,
    run_doctor,
)
from autopilot.errors import TelegramError
from autopilot.settings import Settings, load_settings
from autopilot.system import run_command
from helpers import FAKE_CLAUDE, TZ, FakeTelegramClient

TOKEN = "123456789:" + "A" * 35
EMAIL = "someone@example.com"
LOGIN_JSON = json.dumps(
    {
        "loggedIn": True,
        "authMethod": "claude.ai",
        "apiProvider": "firstParty",
        "email": EMAIL,
        "orgId": "org-123",
        "orgName": "Example Org",
        "subscriptionType": "pro",
    }
)
VALID_TASKS = '[[task]]\nname = "news"\nprompt = "Say hi"\nschedule = "daily 07:30"\n'


class ScriptedRunner:
    """CommandRunner fake: answers by command, records calls, runs the fake Claude for real."""

    def __init__(self, answers: dict[str, tuple[int, str, str]] | None = None) -> None:
        """Store answers keyed by the space-joined command (matched as a suffix of it)."""
        self.answers = answers or {}
        self.calls: list[list[str]] = []

    def __call__(self, args: Sequence[str], timeout: float) -> tuple[int, str, str]:
        """Return the scripted answer, delegate the fake Claude binary, else "not found"."""
        command = " ".join(args)
        self.calls.append(list(args))
        for suffix, answer in self.answers.items():
            if command.endswith(suffix):
                return answer
        if str(FAKE_CLAUDE) in args:
            return run_command(args, timeout)
        return 127, "", "not scripted"


class DoctorTestCase(unittest.TestCase):
    """Base class giving each test a temp home and a settings factory."""

    def setUp(self) -> None:
        """Create a temp home folder removed after the test."""
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.home = Path(tmp.name)

    def make_settings(self, **env: str) -> Settings:
        """Build Settings in the temp home, with the fake Claude unless overridden."""
        values = {"CLAUDE_BIN": str(FAKE_CLAUDE), "AUTOPILOT_TIMEZONE": "Europe/Prague"}
        values.update(env)
        return load_settings(env=values, home=self.home, read_env_file=False)


class ClaudeBinaryTests(DoctorTestCase):
    """check_claude_binary."""

    def test_ok_with_path_and_version(self) -> None:
        """A working binary is reported with its path and version."""
        result, binary = check_claude_binary(self.make_settings(), ScriptedRunner())
        self.assertEqual(result.status, "ok")
        self.assertIn("9.9.9", result.detail)
        self.assertEqual(binary, FAKE_CLAUDE)

    def test_missing_binary_fails_with_fix(self) -> None:
        """A configured but missing binary fails, returns no path and offers a fix."""
        settings = self.make_settings(CLAUDE_BIN=str(self.home / "nope"))
        result, binary = check_claude_binary(settings, ScriptedRunner())
        self.assertEqual(result.status, "fail")
        self.assertIsNone(binary)
        self.assertIn("CLAUDE_BIN", result.fix)

    def test_version_command_failure_fails(self) -> None:
        """A binary whose --version exits non-zero is a failure that keeps the path."""
        runner = ScriptedRunner({"--version": (1, "", "boom")})
        result, binary = check_claude_binary(self.make_settings(), runner)
        self.assertEqual(result.status, "fail")
        self.assertIn("boom", result.detail)
        self.assertEqual(binary, FAKE_CLAUDE)


class ClaudeLoginTests(DoctorTestCase):
    """check_claude_login."""

    def login(self, answer: tuple[int, str, str]) -> CheckResult:
        """Run the login check against a scripted `auth status` answer."""
        return check_claude_login(FAKE_CLAUDE, ScriptedRunner({"auth status --json": answer}))

    def test_subscription_login_shows_plan_and_never_the_email(self) -> None:
        """The plan is shown; the email and org fields never reach the result or the report."""
        result = self.login((0, LOGIN_JSON, ""))
        self.assertEqual((result.status, result.detail), ("ok", "pro plan"))
        report = format_report([result])
        for secret in (EMAIL, "org-123", "Example Org"):
            self.assertNotIn(secret, report)

    def test_login_check_strips_api_credentials(self) -> None:
        """`auth status` runs through env -u so it sees what real runs see."""
        runner = ScriptedRunner({"auth status --json": (0, LOGIN_JSON, "")})
        check_claude_login(FAKE_CLAUDE, runner)
        self.assertEqual(
            runner.calls[0],
            ["env", "-u", "ANTHROPIC_API_KEY", "-u", "ANTHROPIC_AUTH_TOKEN",
             str(FAKE_CLAUDE), "auth", "status", "--json"],
        )

    def test_real_fake_claude_output_is_ok(self) -> None:
        """The bundled fake CLI reports a logged-in subscription."""
        result = check_claude_login(FAKE_CLAUDE, ScriptedRunner())
        self.assertEqual(result.status, "ok")

    def test_other_auth_method_warns(self) -> None:
        """An API-account login warns that runs would bill it."""
        payload = json.dumps({"loggedIn": True, "authMethod": "api_key", "email": EMAIL})
        result = self.login((0, payload, ""))
        self.assertEqual(result.status, "warn")
        self.assertIn("bill", result.detail)
        self.assertNotIn(EMAIL, format_report([result]))

    def test_logged_out_fails_with_login_fix(self) -> None:
        """Logged out (even with a non-zero exit) fails with `claude auth login`."""
        result = self.login((1, json.dumps({"loggedIn": False}), ""))
        self.assertEqual(result.status, "fail")
        self.assertIn("claude auth login", result.fix)

    def test_unreadable_output_warns(self) -> None:
        """Output that is not JSON is a warning with the reason, not a crash."""
        result = self.login((1, "", "unknown command"))
        self.assertEqual(result.status, "warn")
        self.assertIn("unknown command", result.detail)

    def test_json_that_is_not_an_object_warns(self) -> None:
        """A JSON list is treated as unreadable."""
        self.assertEqual(self.login((0, "[1]", "")).status, "warn")

    def test_missing_plan_still_ok(self) -> None:
        """A subscription login without a plan field is still ok."""
        payload = json.dumps({"loggedIn": True, "authMethod": "claude.ai"})
        self.assertEqual(self.login((0, payload, "")).detail, "subscription login")


class ApiKeyTests(unittest.TestCase):
    """check_api_key."""

    def test_clean_environment_is_ok(self) -> None:
        """No credential variables: ok."""
        self.assertEqual(check_api_key({"PATH": "/bin"}).status, "ok")

    def test_key_warns_without_showing_the_value(self) -> None:
        """A set key warns, names the variable and never prints its value."""
        result = check_api_key({"ANTHROPIC_API_KEY": "sk-secret-value"})
        self.assertEqual(result.status, "warn")
        self.assertIn("ANTHROPIC_API_KEY", result.detail)
        self.assertNotIn("sk-secret-value", result.detail + result.fix)

    def test_several_variables_use_plural_verb(self) -> None:
        """Two variables read "are set", one reads "is set"."""
        both = check_api_key({"ANTHROPIC_API_KEY": "a", "ANTHROPIC_AUTH_TOKEN": "b"})
        self.assertIn("ANTHROPIC_API_KEY, ANTHROPIC_AUTH_TOKEN are set", both.detail)
        self.assertIn("removes them", both.detail)
        self.assertIn("Unset them", both.fix)
        one = check_api_key({"ANTHROPIC_API_KEY": "a"})
        self.assertIn("ANTHROPIC_API_KEY is set", one.detail)
        self.assertIn("removes it", one.detail)
        self.assertIn("Unset it", one.fix)

    def test_auth_token_warns_and_empty_value_is_ignored(self) -> None:
        """The auth token variable warns too; an empty value counts as unset."""
        self.assertEqual(check_api_key({"ANTHROPIC_AUTH_TOKEN": "x"}).status, "warn")
        self.assertEqual(check_api_key({"ANTHROPIC_API_KEY": ""}).status, "ok")


class TasksTests(DoctorTestCase):
    """check_tasks."""

    def test_valid_file_is_ok_with_counts(self) -> None:
        """A valid file reports how many tasks exist and are enabled."""
        settings = self.make_settings()
        settings.tasks_file.write_text(VALID_TASKS + '[[task]]\nname = "b"\nprompt = "x"\nenabled = false\n')
        result = check_tasks(settings)
        self.assertEqual((result.status, result.detail), ("ok", "2 tasks (1 enabled)"))

    def test_missing_file_fails(self) -> None:
        """A missing file fails and the message points to `autopilot init`."""
        result = check_tasks(self.make_settings())
        self.assertEqual(result.status, "fail")
        self.assertNotIn("tasks.toml: tasks.toml", format_report([result]))
        self.assertEqual(result.fix, "Run `autopilot init` to create it.")
        self.assertNotIn("autopilot init", result.detail)

    def test_invalid_file_fails_naming_the_field(self) -> None:
        """A typo in a key is reported with the key name."""
        settings = self.make_settings()
        settings.tasks_file.write_text('[[task]]\nname = "a"\nprompt = "x"\nshedule = "daily 07:30"\n')
        result = check_tasks(settings)
        self.assertEqual(result.status, "fail")
        self.assertIn("shedule", result.detail)
        self.assertNotIn("tasks.toml: tasks.toml", format_report([result]))
        self.assertIn("autopilot validate", result.fix)


class TelegramTests(DoctorTestCase):
    """check_telegram."""

    def test_not_configured_warns(self) -> None:
        """Nothing configured: warn with the setup command; the factory is never called."""
        def factory(token: str) -> FakeTelegramClient:
            """Fail the test if a client is built."""
            raise AssertionError("must not be called")

        result = check_telegram(self.make_settings(), factory)  # type: ignore[arg-type]
        self.assertEqual(result.status, "warn")
        self.assertIn("telegram-setup", result.fix)

    def test_half_configured_fails(self) -> None:
        """A token without a chat id (and the reverse) fails."""
        only_token = self.make_settings(TELEGRAM_BOT_TOKEN=TOKEN)
        only_chat = self.make_settings(TELEGRAM_CHAT_ID="5")
        self.assertEqual(check_telegram(only_token, FakeTelegramClient).status, "fail")  # type: ignore[arg-type]
        self.assertEqual(check_telegram(only_chat, FakeTelegramClient).status, "fail")  # type: ignore[arg-type]

    def test_valid_setup_is_ok_without_the_token(self) -> None:
        """A working bot is ok, shows its username and never the token."""
        settings = self.make_settings(TELEGRAM_BOT_TOKEN=TOKEN, TELEGRAM_CHAT_ID="100")
        seen: list[str] = []

        def factory(token: str) -> FakeTelegramClient:
            """Record the token and return a working fake."""
            seen.append(token)
            return FakeTelegramClient()

        result = check_telegram(settings, factory)  # type: ignore[arg-type]
        self.assertEqual(result.status, "ok")
        self.assertIn("@autopilot_test_bot", result.detail)
        self.assertEqual(seen, [TOKEN])
        self.assertNotIn(TOKEN, result.detail)

    def test_telegram_error_fails(self) -> None:
        """A rejected token fails with Telegram's message."""
        settings = self.make_settings(TELEGRAM_BOT_TOKEN=TOKEN, TELEGRAM_CHAT_ID="5")

        class Rejecting:
            """Client whose token Telegram rejects."""

            def get_me(self) -> dict:
                """Fail like Telegram answering 401."""
                raise TelegramError("Telegram getMe failed: Unauthorized", status=401)

        result = check_telegram(settings, lambda token: Rejecting())  # type: ignore[arg-type,return-value]
        self.assertEqual(result.status, "fail")
        self.assertIn("Unauthorized", result.detail)


class DataDirTests(DoctorTestCase):
    """check_data_dir."""

    def test_existing_writable_folder_is_ok(self) -> None:
        """An existing writable data folder is ok."""
        settings = self.make_settings()
        settings.data_dir.mkdir()
        self.assertEqual(check_data_dir(settings).status, "ok")

    def test_missing_folder_with_writable_parent_is_ok(self) -> None:
        """A folder that does not exist yet is fine, because start creates it."""
        result = check_data_dir(self.make_settings())
        self.assertEqual(result.status, "ok")
        self.assertIn("created", result.detail)

    def test_file_in_place_of_folder_fails(self) -> None:
        """A regular file where the folder should be fails."""
        settings = self.make_settings()
        settings.data_dir.write_text("x")
        result = check_data_dir(settings)
        self.assertEqual(result.status, "fail")
        self.assertIn(f"chmod u+rwx {settings.data_dir}", result.fix)
        self.assertNotIn("Fix its", result.fix)

    @unittest.skipIf(os.geteuid() == 0, "root ignores permission bits")
    def test_read_only_parent_fails(self) -> None:
        """A missing folder under a read-only parent cannot be created: fail."""
        settings = self.make_settings(AUTOPILOT_DATA_DIR=str(self.home / "ro" / "data"))
        (self.home / "ro").mkdir()
        (self.home / "ro").chmod(0o500)
        self.addCleanup((self.home / "ro").chmod, 0o700)
        self.assertEqual(check_data_dir(settings).status, "fail")


class ServiceTests(unittest.TestCase):
    """check_service."""

    @staticmethod
    def runner(enabled: str = "enabled", active: tuple[int, str] = (0, "active"), linger: str = "Linger=yes") -> ScriptedRunner:
        """Script systemctl and loginctl answers."""
        return ScriptedRunner(
            {
                "is-enabled autopilot.service": (0 if enabled == "enabled" else 1, enabled + "\n", ""),
                "is-active autopilot.service": (active[0], active[1] + "\n", ""),
                "--property=Linger": (0, linger + "\n", ""),
            }
        )

    def test_enabled_active_linger_on_is_ok(self) -> None:
        """Everything in place is ok."""
        result = check_service(self.runner(), "user")
        self.assertEqual(result.status, "ok")
        self.assertIn("linger on", result.detail)

    def test_not_installed_warns_with_install_hint(self) -> None:
        """A disabled unit warns and suggests install-service."""
        result = check_service(self.runner(enabled="disabled"), "user")
        self.assertEqual(result.status, "warn")
        self.assertIn("install-service", result.fix)

    def test_linger_off_warns(self) -> None:
        """Linger off warns with the enable-linger command."""
        result = check_service(self.runner(linger="Linger=no"), "user")
        self.assertEqual(result.status, "warn")
        self.assertIn("enable-linger", result.fix)

    def test_enabled_but_stopped_warns(self) -> None:
        """An enabled unit that is not running warns."""
        result = check_service(self.runner(active=(3, "inactive")), "user")
        self.assertEqual(result.status, "warn")
        self.assertIn("not running", result.detail)

    def test_no_systemd_warns(self) -> None:
        """A missing systemctl (exit 127) is a warning, not a failure."""
        result = check_service(ScriptedRunner({"is-enabled autopilot.service": (127, "", "missing")}), "user")
        self.assertEqual(result.status, "warn")
        self.assertIn("not available", result.detail)
        self.assertTrue(result.fix)

    def test_unknown_linger_does_not_block_ok(self) -> None:
        """When loginctl cannot answer, an active service is still ok."""
        runner = self.runner()
        runner.answers["--property=Linger"] = (1, "", "no session")
        self.assertEqual(check_service(runner, "user").status, "ok")


class LiveTests(DoctorTestCase):
    """check_live."""

    clock = staticmethod(lambda: datetime(2026, 11, 2, 7, 30, tzinfo=TZ))

    def test_successful_call_is_ok(self) -> None:
        """The fake Claude answers, so the live check is ok."""
        result = check_live(FAKE_CLAUDE, self.clock, self.home)
        self.assertEqual(result.status, "ok")

    def test_failed_call_fails(self) -> None:
        """A binary that cannot start fails with the reason."""
        result = check_live(self.home / "missing", self.clock, self.home)
        self.assertEqual(result.status, "fail")

    def test_usage_limit_warns(self) -> None:
        """A usage-limit answer is a warning, not a failure."""
        script = self.home / "limit.py"
        script.write_text(
            "#!/usr/bin/env python3\nimport json,sys\nsys.stdin.read()\n"
            'print(json.dumps({"type":"result","is_error":True,"subtype":"success",'
            '"result":"You\'ve hit your session limit"}))\nsys.exit(1)\n'
        )
        script.chmod(0o755)
        self.assertEqual(check_live(script, self.clock, self.home).status, "warn")


class RunDoctorTests(DoctorTestCase):
    """run_doctor and format_report."""

    def run_all(self, settings: Settings, **kwargs: object) -> list[CheckResult]:
        """Run the doctor with injected collaborators."""
        return run_doctor(
            settings,
            run=ScriptedRunner({"is-enabled autopilot.service": (1, "disabled\n", "")}),
            env={},
            client_factory=lambda token: FakeTelegramClient(),  # type: ignore[arg-type,return-value]
            user="user",
            **kwargs,  # type: ignore[arg-type]
        )

    def test_runs_all_checks_in_order(self) -> None:
        """Seven checks run in report order; --live adds an eighth."""
        settings = self.make_settings()
        settings.tasks_file.write_text(VALID_TASKS)
        names = [r.name for r in self.run_all(settings)]
        self.assertEqual(
            names,
            ["Claude CLI", "Claude login", "API key", "tasks.toml", "Telegram", "Data folder", "Service"],
        )
        self.assertEqual(len(self.run_all(settings, live=True)), 8)

    def test_missing_binary_skips_dependent_checks(self) -> None:
        """Without a binary the login and live checks are warnings saying "skipped"."""
        settings = self.make_settings(CLAUDE_BIN=str(self.home / "nope"))
        results = self.run_all(settings, live=True)
        by_name = {r.name: r for r in results}
        for name in ("Claude login", "Live call"):
            self.assertEqual(by_name[name].status, "warn")
            self.assertIn("skipped", by_name[name].detail)
            self.assertIn("Fix the Claude CLI check above first", by_name[name].fix)
        self.assertEqual(by_name["Claude CLI"].status, "fail")

    def test_every_warning_and_failure_has_a_fix(self) -> None:
        """No warn or fail result in a typical broken setup lacks a Fix line."""
        settings = self.make_settings(CLAUDE_BIN=str(self.home / "nope"))
        for result in self.run_all(settings, live=True):
            if result.status != "ok":
                self.assertTrue(result.fix, f"{result.name} has no fix")

    def test_multiline_detail_is_indented_under_the_check(self) -> None:
        """Continuation lines of a detail are indented, without repeating the prefix."""
        settings = self.make_settings()
        settings.tasks_file.write_text('[[task]]\nname = "a"\nprompt = "x"\nbad1 = 1\n')
        result = check_tasks(settings)
        multi = CheckResult("tasks.toml", "fail", "first\nsecond", "fix")
        self.assertEqual(format_report([multi]).splitlines()[:3], ["✘ tasks.toml: first", "    second", "    Fix: fix"])
        self.assertFalse(result.detail.startswith("tasks.toml:"))

    def test_report_format_and_summary(self) -> None:
        """Marks, indented fixes and the summary line follow the contract."""
        report = format_report(
            [
                CheckResult("A", "ok", "fine"),
                CheckResult("B", "warn", "hmm", "do this"),
                CheckResult("C", "fail", "broken", "fix that"),
            ]
        )
        self.assertEqual(
            report.splitlines(),
            [
                "✔ A: fine",
                "! B: hmm",
                "    Fix: do this",
                "✘ C: broken",
                "    Fix: fix that",
                "",
                "1 ok, 1 warnings, 1 failed",
            ],
        )

    def test_full_report_never_contains_email_or_token(self) -> None:
        """Whole-doctor privacy check with a login payload carrying an email."""
        settings = self.make_settings(TELEGRAM_BOT_TOKEN=TOKEN, TELEGRAM_CHAT_ID="5")
        runner = ScriptedRunner({"auth status --json": (0, LOGIN_JSON, "")})
        results = run_doctor(
            settings,
            run=runner,
            env={},
            client_factory=lambda token: FakeTelegramClient(),  # type: ignore[arg-type,return-value]
            user="user",
        )
        report = format_report(results)
        self.assertIn("pro plan", report)
        self.assertNotIn(EMAIL, report)
        self.assertNotIn(TOKEN, report)


if __name__ == "__main__":
    unittest.main()
