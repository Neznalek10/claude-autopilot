# Changelog

All notable changes to this project are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses [Semantic Versioning](https://semver.org/).

## [1.2.0] - 2026-11-07

### Added

- Read-only web viewer with a phone-first layout: chats as threads, task runs, search, and an optional tab for Claude Code transcripts. Off unless `AUTOPILOT_WEB_TOKEN` is set. Settings: `AUTOPILOT_WEB_TOKEN`, `AUTOPILOT_WEB_HOST`, `AUTOPILOT_WEB_PORT`, `AUTOPILOT_CLAUDE_PROJECTS_DIR`.

## [1.1.0] - 2026-11-07

### Added

- `/spend <amount> <where>[, <note>]` Telegram command that adds an entry to a Markdown balance file and replies with the new balance. A leading `+` marks income. Set the file with `AUTOPILOT_BALANCE_FILE`; without it the command is off.

## [1.0.0] - 2026-11-03

First release.

### Added

- Scheduled and on-demand Claude Code tasks, run headless with `claude -p` on the user's Claude subscription. API keys are removed from every run.
- `tasks.toml` with per-task options: prompt, schedules, workdir, model, effort, allowed and disallowed tools, permission mode, add dirs, system prompt addition, budget cap, timeout, enabled flag and notify policy. `[defaults]` and `[ask]` tables share options.
- Schedule forms: 5-field cron, `@hourly`, `@daily`, `@weekly`, `@monthly`, and friendly forms such as `every 15m`, `daily 07:30`, `weekdays 08:00` and `mon,wed,fri 18:00`. Times follow the configured time zone, including daylight saving changes.
- Live reload of `tasks.toml` within 30 seconds. An invalid edit keeps the previous tasks and sends one warning.
- Run queue with a configurable number of parallel runs, a timeout per run that kills the whole process group, and one active run per task.
- Usage-limit handling: the run is recorded as `rate_limited`, one notice with the reset time is sent, waiting runs are held until the reset, and each run retries automatically up to three times.
- Telegram bot as the remote control: `/tasks`, `/run`, `/ask`, `/new`, `/last`, `/status`, `/pause`, `/resume` and `/help`, plus plain-text chat that continues one conversation. Conversations reset after six quiet hours and on `/new`.
- Results sent to Telegram with a status header, Markdown converted to Telegram HTML, splitting into at most three messages, and a `.md` file for longer results.
- `autopilot telegram-setup`: validates the bot token, requires a one-time code sent from a private chat, asks for confirmation, and saves the token and chat id to `.env`.
- Command line interface: `init`, `start`, `run`, `ask`, `list`, `history`, `show`, `pause`, `resume`, `next`, `validate`, `doctor`, `telegram-setup`, `install-service` and `uninstall-service`. Each command has `--help` with an example. `--debug` works before or after the command.
- `autopilot doctor` checks the Claude CLI and login, an API key warning, the task file, the Telegram settings, the data folder and the service. `--live` adds one small real Claude call.
- systemd user service that starts at boot and restarts after a failure. Runs cut off by a stop are recorded as interrupted.
- Security: the bot answers only the owner's private chat. Tasks run in an empty private `workspace/` folder by default. Claude is always denied reads of credential and secret files, including `.env` files, SSH, cloud and Git tokens, Claude's credentials and the autopilot data folder. Tools are allow-listed per task.
- Private files: `.env`, the data folder and the database and log files are owner-only.
- Robustness: one service per project folder (lock file); Ctrl+C, SIGTERM or SIGHUP during `run`, `ask` or `start` stops Claude and records the run as interrupted; runs stuck after a crash are cleared once they pass their timeout plus five minutes; failed Telegram deliveries are retried once.
- Usage limits: only Claude's own limit wording counts, the reset time is read in the zone Claude names, chat messages during a limit get an immediate reply, and terminal runs are never retried in the background.
- `autopilot validate` lists the first problem of every table, one per line.
