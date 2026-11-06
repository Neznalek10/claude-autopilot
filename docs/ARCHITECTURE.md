# Architecture

This document describes how Claude Autopilot is built: its components, the path of a scheduled run through the code, the data it stores, the reasons behind the main decisions, and where new features belong. For setup and daily use, see the [README](../README.md).

## Overview

```text
 you ──edit──► tasks.toml ──► tasks.py (parse and validate, uses schedule.py)
               .env ────────► settings.py (paths, time zone, Telegram settings)

 ┌──────────────── autopilot start  (cli.py ► app.py) ────────────────────────────┐
 │ Engine (engine.py)                                                             │
 │   tick every ≤30 s: reload tasks.toml ► scheduler.py (due slots)               │
 │                     ► due retries after a usage limit ► run queue              │
 │   worker thread(s): runner.py ► `claude -p` subprocess (subscription login)    │
 │        ├─► store.py   SQLite: runs, task state, bot offset, chat session       │
 │        └─► TelegramNotifier (bot.py + formatting.py) ─► telegram_api.py ───────┼─► Telegram ─► phone
 │ TelegramBot long-poll thread (bot.py) ◄── chat text, /run, /status ────────────┼── Telegram ◄─ phone
 └────────────────────────────────────────────────────────────────────────────────┘
 systemd user service (service.py, system.py) starts it at boot; doctor.py checks the setup
```

The program is one Python process. Two long-running threads share a SQLite store: the engine loop with its worker threads, and the Telegram polling thread. The CLI commands use the same modules in short-lived processes.

## Components

| Module | Responsibility |
|---|---|
| `cli.py` | Argument parsing for the 15 subcommands. Turns expected errors into one line on stderr and exit code 1. |
| `app.py` | Wiring (`build_app`), logging with token redaction, the service loop (`run_service`), `telegram-setup` and `init`. |
| `engine.py` | The run queue, worker threads, tick loop, usage-limit gate, chat sessions and the notify policy. |
| `scheduler.py` | Pure decisions: which slot is due, whether a slot is too late, what to do with paused or disabled tasks. |
| `schedule.py` | Parses schedule expressions (cron, aliases, friendly forms) and computes the next fire time in local time. Pure. |
| `tasks.py` | Parses and validates `tasks.toml`, merges `[defaults]`, `[ask]` and each task, and adds the protected read rules. |
| `runner.py` | Finds the Claude binary, builds the `claude -p` command line and child environment, runs the process, enforces the timeout, detects usage limits and parses the JSON output. |
| `store.py` | SQLite persistence: runs, task state (paused, last scheduled slot) and a small key-value table. |
| `bot.py` | `TelegramNotifier` (sends run results) and `TelegramBot` (commands, chat turns, long polling). |
| `formatting.py` | Markdown to Telegram HTML, message splitting, status text, help text and the result file. Pure. |
| `telegram_api.py` | Thin Telegram Bot API client over `urllib`. Never puts the token in an error message. |
| `settings.py` | Settings from the real environment and `.env`, time zone resolution, the `.env` writer. |
| `service.py` | Renders, installs and removes the systemd user unit. |
| `doctor.py` | Health checks behind `autopilot doctor`. |
| `system.py` | Runs OS commands with a timeout and never raises. |
| `models.py` | Shared dataclasses, status and trigger constants, the bot command list, and protocols. |
| `errors.py` | Exception hierarchy. `AutopilotError` is the expected, user-facing error. |

## Main use case: a scheduled run, from slot to phone

Take the task `ai-news` with `schedule = "daily 07:30"` and Telegram enabled.

1. **Start.** `cli.main` calls `app.build_app`, which resolves the Claude binary once so that a missing binary fails at start. It creates the `Store`, the `TelegramNotifier`, the `TelegramBot` and the `Engine`, and loads `tasks.toml` (a `ConfigError` stops the start). `app.run_service` installs signal handlers, marks runs left over from the last run as interrupted (`recover_interrupted`), starts the bot thread and calls `Engine.run_forever`.
2. **Tick.** Every loop iteration calls `Engine.tick`. It calls `reload_if_changed`, which compares the file's modification time and reloads only when it changed. An invalid file keeps the old tasks and sends one warning.
3. **Slot.** For each task, `_tick_task` reads `skip` (disabled or paused, from `Store.is_paused`) and `last_scheduled`, then calls `scheduler.evaluate`. The function returns a `DueRun` when the latest slot is at most five minutes old. A slot that is older is skipped and logged, never run late. The new marker is stored before the run is queued, so a crash loses one slot instead of running it twice.
4. **Queue.** `enqueue_task` refuses a task that already has an active run (dedupe). `_create_run` writes a `queued` row with `Store.create_run` and puts the run id on the worker queue.
5. **Execute.** A worker calls `Engine.execute_run`. It reads the record, takes the options from the current `tasks.toml` (`_options_for`), and runs `_execute`. The usage-limit gate is checked first. If the gate is closed, the run finishes as `rate_limited` without calling Claude.
6. **Call Claude.** `_execute` marks the run `running` and calls `ClaudeRunner.run`. The runner resolves the binary, builds the command (`build_command`, prompt on stdin, never in argv) and starts the process in its own session with the child environment from `child_environment`. It waits with the task's timeout. On timeout it signals the whole process group, then kills it.
7. **Read the outcome.** `parse_output` reads the JSON result: status, result text, session id, turns, cost, and the tools that were denied. A usage limit found in a failed result gives `rate_limited` with the reset time.
8. **Store.** `_store_outcome` writes the final state with `Store.finish_run`. A `rate_limited` outcome goes to `_finish_rate_limited` instead. It sets the `limited_until` gate and retries later, up to three times.
9. **Notify.** `_notify_run` applies the notify policy (`always`, `errors` or `never`). `TelegramNotifier.notify_run` formats the result with `format_run_messages`, sends the messages in order with `send_html`, and sends a `.md` file when the text does not fit. A Telegram parse error makes `send_html` resend the text as plain text. Notifier errors are logged, never raised into the engine.

A chat message follows the same path with two differences. It enters through `TelegramBot.handle_update`, which checks the owner's private chat first. And it runs under a lock, with `resume_session` set to the stored chat session, so turns stay in order and continue the conversation.

## Data model and storage

Everything lives under the data folder (`data/` in the home folder by default). The folder is created owner-only (mode 700), and the database and log files are set to mode 600.

| Storage | Contents |
|---|---|
| `data/autopilot.db` (SQLite, WAL mode, 10 s busy timeout) | Tables below. |
| `data/autopilot.log` | Rotating log, 1 MB and three backups. Redacts the bot token. |
| `.env` (mode 600) | Bot token, chat id and optional settings. Written by `set_env_value`, atomically. |
| `tasks.toml` | User configuration. Read only, reloaded when its modification time changes. |
| `workspace/` | Default working folder for tasks. Empty, owner-only. |

Tables:

- `runs`: one row per run. Columns: `id`, `task_name`, `trigger` (`schedule`, `manual`, `telegram` or `chat`), `status` (`queued`, `running`, `success`, `error`, `timeout`, `rate_limited`, `cancelled`), `prompt`, `queued_at`, `scheduled_for`, `started_at`, `finished_at`, `duration_ms`, `result`, `error`, `session_id`, `num_turns`, `cost_usd`, `denied_tools` (comma-joined), `retry_of`, `retry_at`.
- `task_state`: `task_name` (primary key), `paused`, `last_scheduled`.
- `kv`: `key` and `value`. Keys: `telegram_offset` (next Telegram update to read), `limited_until` (usage-limit gate), `chat_session` (Claude session id of the conversation), `chat_updated_at`, and `chat_reset_at`.

Datetimes are stored as ISO 8601 text with their UTC offset. The store returns them as aware datetimes with a fixed offset. Display code converts them to the configured time zone before showing them. Each store call opens and closes its own connection, which keeps the engine, the bot thread and the CLI safe to use at the same time.

## Key design decisions

- **Claude runs as a subprocess with `claude -p`.** It is the headless path that uses the subscription login. The Agent SDK and the API would bill API credits. `--bare` ignores the subscription login, so it is not used. The prompt goes on stdin, so a prompt that starts with `-` cannot be read as a flag.
- **Telegram long polling, not webhooks.** The PC needs no open port and no public HTTPS address, so it works behind a home router.
- **`urllib` for the Bot API.** Autopilot uses six endpoints. A Telegram library would add a large dependency for that.
- **`tasks.toml` read with the standard library `tomllib`.** It has comments and multi-line prompts, with no extra dependency. YAML would need PyYAML, and JSON has no comments.
- **Own cron parser on `zoneinfo`.** It is small, testable and DST aware. Its handling of skipped and repeated local times is covered by tests.
- **SQLite with one connection per call.** It is safe across threads and processes. JSON files would have no concurrency control.
- **Slots are never run late in a burst.** A slot up to five minutes old runs. Older slots are skipped and logged, so a woken PC does not fire a backlog.
- **One gate for usage limits, kept in the store.** When Claude reports a limit, one notice goes out and every queued run waits without calling Claude. The gate survives restarts.
- **Chat is one rolling session.** The session id is stored and resumed for each message. It resets on `/new`, after six quiet hours, and after a failed turn.
- **Claude binary found on every run.** Editor extensions update their bundled binary, so the path is searched again each time rather than cached.
- **Reads of secrets are denied by rule, and tools are allow-listed.** Each task gets `disallowed_tools` entries for credential files, and `allowed_tools` defaults to read and web tools. A task that can read your files should not also get `WebSearch` or `WebFetch`.
- **Child processes get a clean environment.** API keys, `TELEGRAM_*` variables and most `CLAUDE_*` variables are removed, so a run cannot read the bot token or switch to API billing.
- **Only the owner can command the bot.** Messages count only when both the chat id and the sender id equal the saved owner id, and `telegram-setup` needs a one-time code and a confirmation.

## How to extend

Put new code next to the module that owns the concern, and add a test file with the same name under `tests/` (`test_<module>.py`).

- **A new CLI command:** add a `add_parser` block and handler in `cli.py`, then a `--help` example in its epilog. Keep the logic in the module it belongs to, so the command stays a thin wrapper.
- **A new task option:** add the key to `OPTION_KEYS` (and `TASK_ONLY_KEYS` or `BEHAVIOUR_KEYS` if needed) and validate it in `tasks.py`. Add the field to `RunOptions` in `models.py`, pass it through `build_command` in `runner.py`, and document it in `tasks.example.toml` and the README table.
- **A new schedule form:** add the parsing branch in `schedule.py` (`parse_schedule`) and a test in `test_schedule.py`. Keep `next_after` as the only place that computes fire times.
- **A new bot command:** add a `_cmd_<name>` method and register it in `TelegramBot._handlers`. Add it to `BOT_COMMANDS` in `models.py`, which feeds both the Telegram command menu and the help text.
- **A new notification:** add a message builder in `formatting.py` (text only, no network). Send it through `TelegramNotifier` in `bot.py`, or through `Notifier.notify_text` if it is a plain notice.
- **A new health check:** add a function in `doctor.py` that returns `CheckResult`, and call it from `run_doctor`. Keep external commands behind the injected `run` argument so tests can fake them.
- **A new environment variable:** read it in `settings.py` (`load_settings`), add the field to `Settings`, and document it in `.env.example` and the README configuration table.
- **A new engine behavior:** change `engine.py` only when it concerns queueing, gating or notification policy. Anything that depends on the clock takes the injected `clock`, so tests can control time.

Two rules apply everywhere. Datetimes are timezone aware, and "now" comes from the injected clock. Code that shows a time to the user converts it to the configured time zone first.
