"""Unit tests for autopilot.system.run_command.

These run real, harmless programs (true, false, sleep, the Python interpreter)
to check the exit codes, captured output, timeout handling and the error codes
used for programs that can't start. No systemctl or loginctl is ever called.
"""

from __future__ import annotations

import sys
import time
import unittest

from autopilot.system import EXIT_NOT_FOUND, EXIT_TIMED_OUT, run_command

# A program name that is guaranteed not to exist on PATH.
MISSING_PROGRAM = "autopilot-test-no-such-program"
# How long the timeout test waits before giving up on `sleep`. Short so the suite stays fast.
SHORT_TIMEOUT_SECONDS = 0.2
# Budget for the partial-output test: the child must start and print within this window.
PARTIAL_OUTPUT_TIMEOUT_SECONDS = 3.0


class RunCommandTest(unittest.TestCase):
    """Behaviour of run_command with real child processes."""

    def test_successful_command_returns_zero_and_its_stdout(self) -> None:
        """A command that exits 0 returns code 0 and the text it printed."""
        code, stdout, stderr = run_command([sys.executable, "-c", "print('hello')"])
        self.assertEqual(code, 0)
        self.assertEqual(stdout, "hello\n")
        self.assertEqual(stderr, "")

    def test_failing_command_returns_its_exit_code(self) -> None:
        """A command that exits non-zero reports that exact code, without raising."""
        code, _, _ = run_command(["false"])
        self.assertEqual(code, 1)

    def test_stderr_is_captured_separately_from_stdout(self) -> None:
        """Text written to stderr comes back in the stderr slot, not in stdout."""
        script = "import sys; sys.stderr.write('warning text')"
        code, stdout, stderr = run_command([sys.executable, "-c", script])
        self.assertEqual(code, 0)
        self.assertEqual(stdout, "")
        self.assertEqual(stderr, "warning text")

    def test_unicode_output_round_trips_unchanged(self) -> None:
        """Non-ASCII output is decoded as UTF-8 regardless of the locale."""
        text = "Příliš žluťoučký kůň"
        script = f"import sys; sys.stdout.buffer.write({text.encode('utf-8')!r})"
        _, stdout, _ = run_command([sys.executable, "-c", script])
        self.assertEqual(stdout, text)

    def test_missing_program_returns_not_found_code_and_names_the_program(self) -> None:
        """A program that isn't installed returns 127 and a stderr message that names it."""
        code, stdout, stderr = run_command([MISSING_PROGRAM])
        self.assertEqual(code, EXIT_NOT_FOUND)
        self.assertEqual(stdout, "")
        self.assertIn(MISSING_PROGRAM, stderr)

    def test_timeout_kills_the_command_and_returns_timeout_code(self) -> None:
        """A command that outlives its timeout returns 124 and says it timed out."""
        started = time.monotonic()
        code, _, stderr = run_command(["sleep", "5"], timeout=SHORT_TIMEOUT_SECONDS)
        elapsed = time.monotonic() - started
        self.assertEqual(code, EXIT_TIMED_OUT)
        self.assertIn("timed out", stderr)
        self.assertLess(elapsed, 4.0)

    def test_output_printed_before_timeout_is_returned_as_text(self) -> None:
        """Stdout written before a timeout comes back as a str, not as bytes."""
        script = (
            "import sys, time; sys.stdout.write('partial'); sys.stdout.flush(); time.sleep(30)"
        )
        code, stdout, _ = run_command(
            [sys.executable, "-c", script],
            timeout=PARTIAL_OUTPUT_TIMEOUT_SECONDS,
        )
        self.assertEqual(code, EXIT_TIMED_OUT)
        self.assertIsInstance(stdout, str)
        self.assertEqual(stdout, "partial")


if __name__ == "__main__":
    unittest.main()
