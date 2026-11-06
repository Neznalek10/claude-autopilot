"""Entry point for `python -m autopilot`.

Delegates to autopilot.cli.main and turns its return value into the process
exit code. Used by the systemd unit (ExecStart) and by developers running the
package from the project folder without installing the console script.
"""

from autopilot.cli import main

raise SystemExit(main())
