"""Claude Autopilot: runs Claude Code prompt tasks on schedules and reports to Telegram.

The package runs `claude -p` headless on the user's Claude subscription, on
schedules defined in tasks.toml and on demand. Results go to a private Telegram
bot, which is also the remote control.

Entry points: the `autopilot` console script and `python -m autopilot`, both of
which end in autopilot.cli.main. Submodules are imported explicitly by their users;
this file imports nothing so that importing the package stays cheap.
"""

# Single source of truth for the version: pyproject.toml reads this attribute.
__version__ = "1.0.0"
