"""Unit tests for autopilot.settings.

Covers environment parsing, .env precedence, time zone detection and the
atomic .env writer. All tests use temporary folders and never touch the real
environment or /etc.
"""

from __future__ import annotations

import os
import stat
import tempfile
import unittest
from pathlib import Path
from zoneinfo import ZoneInfo

from autopilot.errors import ConfigError
from autopilot.settings import (
    Settings,
    load_settings,
    project_root,
    resolve_timezone,
    set_env_value,
)


class LoadSettingsTest(unittest.TestCase):
    """load_settings: defaults, overrides, precedence and validation."""

    def setUp(self) -> None:
        """Create a temporary home folder."""
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.home = Path(self._tmp.name).resolve()

    def load(self, env: dict[str, str] | None = None, **kwargs) -> Settings:
        """Load settings for the temp home with an explicit environment."""
        return load_settings(env or {"AUTOPILOT_TIMEZONE": "Europe/Berlin"}, home=self.home, **kwargs)

    def test_defaults(self) -> None:
        """With nothing set, paths derive from home and limits use defaults."""
        settings = self.load()
        self.assertEqual(settings.tasks_file, self.home / "tasks.toml")
        self.assertEqual(settings.data_dir, self.home / "data")
        self.assertEqual(settings.db_path, self.home / "data" / "autopilot.db")
        self.assertEqual(settings.log_file, self.home / "data" / "autopilot.log")
        self.assertEqual(settings.env_file, self.home / ".env")
        self.assertEqual(settings.max_parallel, 1)
        self.assertEqual(settings.log_level, "INFO")
        self.assertIsNone(settings.claude_bin)
        self.assertFalse(settings.telegram_enabled)
        self.assertEqual(settings.timezone, ZoneInfo("Europe/Berlin"))

    def test_relative_and_tilde_paths(self) -> None:
        """Relative values resolve against home and ~ expands."""
        settings = self.load({
            "AUTOPILOT_TIMEZONE": "UTC",
            "AUTOPILOT_TASKS_FILE": "conf/my.toml",
            "AUTOPILOT_DATA_DIR": "~/autopilot-data",
        })
        self.assertEqual(settings.tasks_file, self.home / "conf" / "my.toml")
        self.assertEqual(settings.data_dir, (Path.home() / "autopilot-data").resolve())

    def test_home_from_environment(self) -> None:
        """AUTOPILOT_HOME in the real environment sets home when no argument is given."""
        settings = load_settings({"AUTOPILOT_HOME": str(self.home), "AUTOPILOT_TIMEZONE": "UTC"})
        self.assertEqual(settings.home, self.home)

    def test_home_defaults_to_project_root(self) -> None:
        """Without a home argument or variable, the project root is used."""
        settings = load_settings({"AUTOPILOT_TIMEZONE": "UTC"}, read_env_file=False)
        self.assertEqual(settings.home, project_root())
        self.assertTrue((project_root() / "autopilot").is_dir())

    def test_telegram_values(self) -> None:
        """Token and a positive chat id enable Telegram."""
        settings = self.load({
            "AUTOPILOT_TIMEZONE": "UTC",
            "TELEGRAM_BOT_TOKEN": "tok",
            "TELEGRAM_CHAT_ID": "123456789",
        })
        self.assertEqual(settings.telegram_chat_id, 123456789)
        self.assertTrue(settings.telegram_enabled)

    def test_token_without_chat_is_not_enabled(self) -> None:
        """Telegram needs both values."""
        self.assertFalse(self.load({"AUTOPILOT_TIMEZONE": "UTC", "TELEGRAM_BOT_TOKEN": "tok"}).telegram_enabled)

    def test_env_file_fills_missing_values_and_real_env_wins(self) -> None:
        """.env supplies missing values; the real environment overrides it."""
        (self.home / ".env").write_text(
            "AUTOPILOT_MAX_PARALLEL=3\nAUTOPILOT_LOG_LEVEL=debug\nCLAUDE_BIN=/opt/claude\n",
            encoding="utf-8",
        )
        settings = self.load({"AUTOPILOT_TIMEZONE": "UTC", "AUTOPILOT_LOG_LEVEL": "error"})
        self.assertEqual(settings.max_parallel, 3)
        self.assertEqual(settings.log_level, "ERROR")
        self.assertEqual(settings.claude_bin, "/opt/claude")

    def test_env_file_does_not_touch_os_environ(self) -> None:
        """Reading .env must leave os.environ unchanged."""
        (self.home / ".env").write_text("AUTOPILOT_TEST_ONLY_VAR=1\nTELEGRAM_CHAT_ID=5\n", encoding="utf-8")
        before = dict(os.environ)
        self.load()
        self.assertEqual(dict(os.environ), before)

    def test_env_file_can_be_skipped(self) -> None:
        """read_env_file=False ignores .env."""
        (self.home / ".env").write_text("AUTOPILOT_MAX_PARALLEL=3\n", encoding="utf-8")
        self.assertEqual(self.load(read_env_file=False).max_parallel, 1)

    def test_autopilot_home_in_env_file_is_ignored(self) -> None:
        """AUTOPILOT_HOME comes from the real environment only."""
        (self.home / ".env").write_text("AUTOPILOT_HOME=/elsewhere\n", encoding="utf-8")
        self.assertEqual(self.load().home, self.home)

    def test_empty_values_count_as_unset(self) -> None:
        """Blank real-env values fall through to .env and then to defaults."""
        (self.home / ".env").write_text("AUTOPILOT_MAX_PARALLEL=2\nCLAUDE_BIN=\n", encoding="utf-8")
        settings = self.load({"AUTOPILOT_TIMEZONE": "UTC", "AUTOPILOT_MAX_PARALLEL": "", "TELEGRAM_CHAT_ID": " "})
        self.assertEqual(settings.max_parallel, 2)
        self.assertIsNone(settings.claude_bin)
        self.assertIsNone(settings.telegram_chat_id)

    def test_bad_values_name_the_variable(self) -> None:
        """Each invalid value raises ConfigError that names its variable."""
        cases = {
            "AUTOPILOT_MAX_PARALLEL": ["0", "5", "two"],
            "AUTOPILOT_LOG_LEVEL": ["LOUD"],
            "TELEGRAM_CHAT_ID": ["abc", "12.5"],
            "AUTOPILOT_TIMEZONE": ["Mars/Base"],
        }
        for variable, values in cases.items():
            for value in values:
                with self.subTest(variable=variable, value=value):
                    with self.assertRaises(ConfigError) as ctx:
                        self.load({variable: value})
                    self.assertIn(variable, str(ctx.exception))

    def test_unknown_user_home_is_config_error(self) -> None:
        """'~nosuchuser' in a path variable raises ConfigError naming the variable."""
        for variable in ("AUTOPILOT_TASKS_FILE", "AUTOPILOT_DATA_DIR", "AUTOPILOT_HOME"):
            with self.subTest(variable=variable):
                with self.assertRaises(ConfigError) as ctx:
                    load_settings({variable: "~nosuchuser_zz/x", "AUTOPILOT_TIMEZONE": "UTC"}, home=None if variable == "AUTOPILOT_HOME" else self.home)
                self.assertIn(variable, str(ctx.exception))

    def test_group_and_zero_chat_ids_are_rejected(self) -> None:
        """Negative (group) and zero chat ids fail, saying groups are not supported."""
        for value in ("-100123", "0"):
            with self.subTest(value=value):
                with self.assertRaises(ConfigError) as ctx:
                    self.load({"AUTOPILOT_TIMEZONE": "UTC", "TELEGRAM_CHAT_ID": value})
                self.assertIn("TELEGRAM_CHAT_ID", str(ctx.exception))
                self.assertIn("Group chats are not supported", str(ctx.exception))

    def test_token_never_in_error(self) -> None:
        """A bad chat id error must not echo the bot token."""
        with self.assertRaises(ConfigError) as ctx:
            self.load({"TELEGRAM_BOT_TOKEN": "123456:SECRETSECRET", "TELEGRAM_CHAT_ID": "x"})
        self.assertNotIn("SECRET", str(ctx.exception))


class ResolveTimezoneTest(unittest.TestCase):
    """resolve_timezone: explicit name, TZ, system files and the UTC fallback."""

    def setUp(self) -> None:
        """Create a temp folder holding fake system files."""
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)
        self.localtime = self.dir / "localtime"
        self.timezone_file = self.dir / "timezone"

    def resolve(self, name: str | None = None, env_tz: str | None = None) -> ZoneInfo:
        """Call resolve_timezone against the fake system files."""
        return resolve_timezone(name, env_tz=env_tz, localtime=self.localtime, timezone_file=self.timezone_file)

    def test_explicit_name(self) -> None:
        """An explicit name wins over everything."""
        self.timezone_file.write_text("Asia/Tokyo\n", encoding="utf-8")
        self.assertEqual(self.resolve("America/New_York"), ZoneInfo("America/New_York"))

    def test_unknown_explicit_name(self) -> None:
        """An unknown explicit name is a ConfigError."""
        with self.assertRaises(ConfigError):
            self.resolve("Nowhere/Land")

    def test_folder_name_is_not_a_zone(self) -> None:
        """A bare region such as 'Europe' is rejected, not crashed on."""
        with self.assertRaises(ConfigError):
            self.resolve("Europe")

    def test_tz_variable(self) -> None:
        """A valid TZ value is used, including a leading colon."""
        self.assertEqual(self.resolve(env_tz=":Europe/Berlin"), ZoneInfo("Europe/Berlin"))

    def test_invalid_tz_variable_falls_through(self) -> None:
        """An invalid TZ is skipped in favour of the next source."""
        self.timezone_file.write_text("Asia/Tokyo\n", encoding="utf-8")
        self.assertEqual(self.resolve(env_tz="garbage"), ZoneInfo("Asia/Tokyo"))

    def test_localtime_symlink(self) -> None:
        """The zone is read from the /etc/localtime symlink target."""
        self.localtime.symlink_to("/usr/share/zoneinfo/Europe/Berlin")
        self.assertEqual(self.resolve(), ZoneInfo("Europe/Berlin"))

    def test_timezone_file(self) -> None:
        """/etc/timezone content is used when there is no symlink."""
        self.timezone_file.write_text("Europe/Berlin\n", encoding="utf-8")
        self.assertEqual(self.resolve(), ZoneInfo("Europe/Berlin"))

    def test_utc_fallback_logs_warning(self) -> None:
        """Nothing usable gives UTC and a warning."""
        with self.assertLogs("autopilot.settings", level="WARNING"):
            self.assertEqual(self.resolve(), ZoneInfo("UTC"))


class SetEnvValueTest(unittest.TestCase):
    """set_env_value: create, update, preserve, validate and secure the file."""

    def setUp(self) -> None:
        """Create a temp folder for the .env file."""
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.env_file = Path(self._tmp.name) / ".env"

    def test_creates_file_with_mode_600(self) -> None:
        """A new file holds the line and is private."""
        set_env_value(self.env_file, "TELEGRAM_CHAT_ID", "123")
        self.assertEqual(self.env_file.read_text(encoding="utf-8"), "TELEGRAM_CHAT_ID=123\n")
        self.assertEqual(stat.S_IMODE(self.env_file.stat().st_mode), 0o600)

    def test_updates_in_place_and_keeps_other_lines(self) -> None:
        """An existing key is replaced; comments and other keys stay."""
        self.env_file.write_text("# my settings\nA=1\nTELEGRAM_CHAT_ID=old\n\nB=2\n", encoding="utf-8")
        self.env_file.chmod(0o644)
        set_env_value(self.env_file, "TELEGRAM_CHAT_ID", "-55")
        self.assertEqual(
            self.env_file.read_text(encoding="utf-8"), "# my settings\nA=1\nTELEGRAM_CHAT_ID=-55\n\nB=2\n"
        )
        self.assertEqual(stat.S_IMODE(self.env_file.stat().st_mode), 0o600)

    def test_appends_missing_key(self) -> None:
        """A new key is appended even when the file lacks a trailing newline."""
        self.env_file.write_text("A=1", encoding="utf-8")
        set_env_value(self.env_file, "B", "2")
        self.assertEqual(self.env_file.read_text(encoding="utf-8"), "A=1\nB=2\n")

    def test_value_with_special_characters_round_trips(self) -> None:
        """Values with spaces, quotes and # are quoted so dotenv reads them back."""
        from dotenv import dotenv_values

        value = "it's a #value with \\ and é"
        set_env_value(self.env_file, "CLAUDE_BIN", value)
        self.assertEqual(dotenv_values(self.env_file)["CLAUDE_BIN"], value)

    def test_token_round_trips_unquoted(self) -> None:
        """A typical bot token needs no quoting."""
        set_env_value(self.env_file, "TELEGRAM_BOT_TOKEN", "123456:AA-bb_CC")
        self.assertEqual(self.env_file.read_text(encoding="utf-8"), "TELEGRAM_BOT_TOKEN=123456:AA-bb_CC\n")

    def test_rejects_unicode_line_separators(self) -> None:
        """U+2028, U+0085 and control codes are rejected so rewrites can't split a value."""
        for value in ("a\u2028b=1", "a\u0085b", "a\x00b", "a\tb"):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    set_env_value(self.env_file, "OK", value)

    def test_existing_unicode_separator_does_not_split_lines(self) -> None:
        """A comment holding U+2028 stays one line when the file is rewritten."""
        self.env_file.write_text("# note\u2028X=1\nA=1\n", encoding="utf-8")
        set_env_value(self.env_file, "B", "2")
        self.assertEqual(self.env_file.read_text(encoding="utf-8"), "# note\u2028X=1\nA=1\nB=2\n")

    def test_rejects_bad_key_and_newline(self) -> None:
        """Invalid keys and multi-line values raise ValueError and write nothing."""
        for key, value in (("lower", "x"), ("1BAD", "x"), ("A B", "x"), ("OK", "a\nb"), ("OK", "a\rb")):
            with self.subTest(key=key, value=value):
                with self.assertRaises(ValueError):
                    set_env_value(self.env_file, key, value)
        self.assertFalse(self.env_file.exists())

    def test_leaves_no_temp_files(self) -> None:
        """The atomic write leaves only the final file in the folder."""
        set_env_value(self.env_file, "A", "1")
        self.assertEqual([p.name for p in self.env_file.parent.iterdir()], [".env"])


if __name__ == "__main__":
    unittest.main()
