# Claude Autopilot

Claude Autopilot is a background service for your Linux PC. It runs Claude Code prompt tasks in headless mode (`claude -p`) on the Claude subscription you already pay for, either on schedules you write in `tasks.toml` or when you ask. Every result arrives in a private Telegram chat with your own bot, and the same bot is the remote control: run a task, ask Claude a question, pause a schedule, or check what is running. The service restarts at boot, and it queues or retries runs when Claude reports a usage limit, so you do not have to watch it.

## Requirements

- Linux. The background service uses systemd user services. Running `autopilot start` by hand works without systemd, but the install and status commands in this guide assume systemd.
- Python 3.11 or newer. The project is developed and tested on Python 3.12.
- The Claude Code CLI, installed and logged in with a Claude subscription (for example Pro). Runs use that subscription login. Autopilot removes API keys from the environment of every run, so no run is billed to an API account.
- A Telegram account, to receive results and send commands.

## Quick start

Run these commands in the project folder, in this order:

```bash
python3 -m venv venv
venv/bin/pip install -e .
venv/bin/autopilot init
venv/bin/autopilot telegram-setup
venv/bin/autopilot doctor
venv/bin/autopilot start
```

What each step does:

1. `python3 -m venv venv` and `pip install -e .` create a private Python environment and install the `autopilot` command into it.
2. `autopilot init` creates `tasks.toml` (a copy of `tasks.example.toml`) and `.env` (mode 600). It never overwrites existing files. Its output looks like this:

   ```text
   Created in <project folder>: tasks.toml, .env
   Next steps:
     1. Edit tasks.toml, then check it with: autopilot validate
     2. Connect your phone with: autopilot telegram-setup
     3. Check everything with: autopilot doctor
     4. Start it with: autopilot start (or autopilot install-service)
   ```

3. `autopilot telegram-setup` connects your phone. See [Connecting your phone](#connecting-your-phone).
4. `autopilot doctor` checks the Claude CLI, the login, the task file, the Telegram settings, the data folder and the service. Each line starts with a check mark (ok), an exclamation mark (warning) or a cross (failure). A failure or warning comes with a `Fix:` line.
5. `autopilot start` runs the service in the foreground until you press Ctrl+C. To run it in the background instead, use `venv/bin/autopilot install-service` (see [Running in the background](#running-in-the-background)).

## Connecting your phone

1. In Telegram, open **@BotFather**, send `/newbot`, and answer its two questions: a display name and a username that ends in `bot`.
2. BotFather replies with a token that looks like `123456789:AA...`. Copy it.
3. Run `venv/bin/autopilot telegram-setup`. It asks:

   ```text
   Paste the bot token from @BotFather:
   ```

   The token is typed without echo. Autopilot checks it with Telegram and saves it to `.env`.
4. The command prints a one-time code and asks you to send it to the bot, for example `Open https://t.me/<your_bot> and send the code 483920 (waiting up to 120 s)...`. Open that link, and send the code from your private chat with the bot. Messages sent before the setup started are ignored, so only a new message with the code counts. A message without the code is ignored.
5. The terminal shows who sent the code and asks `Use this chat? [y/N]`. Type `y` and press Enter. Only `y` or `yes` saves the chat. Answer `n` (or press Enter) to wait for another message.
6. Autopilot saves `TELEGRAM_CHAT_ID` to `.env` and sends the bot's first message to you: `✅ Autopilot is connected. Send /help to see what I can do.`
7. Send `/help` to the bot. It should answer with the command list.

The chat id must be your own Telegram user id, a positive number. Group chats are not supported: the bot answers only your private chat with it.

If the wait runs out, run the command again. Use `--wait 300` to allow five minutes instead of two.

## Defining tasks

Tasks live in `tasks.toml`. The file has three kinds of tables:

- `[defaults]` holds values shared by every task.
- `[ask]` holds the options for chat messages and `autopilot ask`.
- `[[task]]` has one table per task.

Keys for `[[task]]` (and for `[defaults]`, where they are shared):

| Key | Meaning | Default |
|---|---|---|
| `name` | Required. Lower case letters, digits, `-` and `_`, up to 40 characters. Unique. `ask` is reserved. | none |
| `prompt` | Required. What Claude is asked to do. | none |
| `schedule` | One schedule, or a list of them. Leave it out for a task that only runs by hand. | manual only |
| `description` | One line shown by `list` and `/tasks`. | empty |
| `workdir` | Folder Claude runs in. `~` and relative paths work. | `workspace/` next to `tasks.toml`, a private empty folder created for you |
| `model` | Model alias or id, for example `sonnet` or `haiku`. | Claude's default |
| `effort` | `low`, `medium`, `high`, `xhigh` or `max`. | Claude's default |
| `allowed_tools` | Tools Claude may use without asking, for example `["Read", "Bash(git log *)"]`. | `Read`, `Glob`, `Grep`, `WebSearch`, `WebFetch` |
| `disallowed_tools` | Tools Claude may never use. | none |
| `permission_mode` | `acceptEdits`, `auto`, `bypassPermissions`, `manual`, `dontAsk` or `plan`. With `dontAsk`, anything not in `allowed_tools` is blocked. | `dontAsk` |
| `add_dirs` | Extra folders Claude may access. | none |
| `append_system_prompt` | Extra instructions added to Claude's system prompt. | none |
| `max_budget_usd` | Stop a run that would cost more than this many dollars. | no cap |
| `timeout_minutes` | Kill a run after this many minutes (1 to 720). | 30 for tasks, 15 for `[ask]` |
| `enabled` | `false` stops the schedule. `autopilot run` and `/run` still work. | `true` |
| `notify` | `always`, `errors` or `never`: when the result goes to Telegram. | `always` |

Schedules accept three kinds of expression:

- Friendly: `every 15m`, `every 2h`, `hourly`, `daily 07:30`, `weekdays 08:00`, `weekends 10:00`, `mon,wed,fri 18:00`, `mon-fri 07:00`.
- Aliases: `@hourly`, `@daily`, `@weekly`, `@monthly`.
- Cron, five fields: `minute hour day-of-month month day-of-week`, for example `30 7 * * 1-5`.

Times use the time zone in `AUTOPILOT_TIMEZONE`. Preview any expression with `venv/bin/autopilot next "weekdays 08:00"`.

A slot that is more than five minutes late is skipped and logged, never run in a burst. This happens when the PC was asleep or the service was stopped at that time. A slot up to five minutes late, for example right after a restart, still runs.

Three short examples:

```toml
[[task]]
name = "ai-news"
description = "Morning digest of AI news"
schedule = "daily 07:30"
model = "sonnet"
allowed_tools = ["WebSearch", "WebFetch"]
prompt = """
Search the web for the most important AI news from the last 24 hours.
Reply with at most 7 bullets, one sentence each, with a link after each bullet.
"""

[[task]]
name = "repo-check"
description = "Commits from the last working day"
schedule = ["30 9 * * 1-5", "weekends 10:00"]
allowed_tools = ["Bash(git log *)", "Read"]
max_budget_usd = 0.50
prompt = "List the commits from the last day in this repository and summarise them in five lines."

[[task]]
name = "weekly-review"
description = "Review of the past week (run by hand)"
enabled = false
allowed_tools = ["Read", "Glob", "Grep"]
prompt = "Look at the files that changed in the last 7 days and suggest three priorities."
```

The `weekly-review` task has no schedule of its own. It is disabled, so it runs only when you start it with `autopilot run weekly-review` or `/run weekly-review`. The file `tasks.example.toml` documents every key and ships three working tasks. Check your file after each edit with `venv/bin/autopilot validate`, which lists every error with the task and field name, and `venv/bin/autopilot list`, which shows the next run of each task.

Edits to `tasks.toml` apply within 30 seconds, with no restart. An invalid edit keeps the previous tasks and sends one warning to your phone.

## Configuration

Settings come from `.env` in the project folder and from the real environment. A real environment variable wins over `.env`. An empty value counts as unset, so the default applies. `autopilot init` creates `.env` from `.env.example`, which lists every variable with a comment.

| Variable | Default | Purpose |
|---|---|---|
| `AUTOPILOT_HOME` | the project folder | Project home folder. Read from the real environment only, not from `.env`. |
| `AUTOPILOT_TASKS_FILE` | `tasks.toml` in the home folder | Path to the task file. |
| `AUTOPILOT_DATA_DIR` | `data/` in the home folder | Folder for `autopilot.db` and `autopilot.log`. |
| `AUTOPILOT_TIMEZONE` | the system time zone | IANA time zone for schedules, for example `Europe/Prague`. |
| `AUTOPILOT_MAX_PARALLEL` | `1` | Runs that may execute at the same time, from 1 to 4. |
| `AUTOPILOT_LOG_LEVEL` | `INFO` | `DEBUG`, `INFO`, `WARNING` or `ERROR`. |
| `CLAUDE_BIN` | found automatically | Full path to the Claude Code binary, if autodetection fails. |
| `TELEGRAM_BOT_TOKEN` | none | Bot token from @BotFather. `telegram-setup` writes it. |
| `TELEGRAM_CHAT_ID` | none | Your own Telegram user id, a positive number. Group chats are not supported. `telegram-setup` writes it. |

If `AUTOPILOT_TIMEZONE` is empty, autopilot uses the `TZ` variable from the real environment, then the system zone. Environment changes take effect when the service starts, so restart it after editing `.env` (`systemctl --user restart autopilot`).

## Using it from the phone

Send a command to your bot, or type a message without a command:

| Command | What it does |
|---|---|
| `/tasks` | Lists tasks, their next run and their last result. |
| `/run <task>` | Queues a run of the task now. |
| `/ask <text>` | Same as typing the text. |
| plain text | A chat message. Claude answers in the same conversation as before. |
| `/new` | Starts a new conversation. `/new <text>` starts one and sends the text as its first message. |
| `/last [task]` | Shows the newest finished result, for all tasks or for one task. |
| `/status` | Shows what is running, what is queued, and the next three runs. |
| `/pause <task>` | Stops a task's schedule. Manual runs still work. |
| `/resume <task>` | Starts a paused task's schedule again. |
| `/help` or `/start` | Shows the help text. |

Chat works like a conversation. Each message continues the last chat session, so Claude remembers what you said before. The conversation starts fresh in two cases: when you send `/new`, and when six hours pass without a message. If a chat turn fails, the conversation is reset too, and the error message says so.

The bot tells you where your message stands:

- If nothing else is queued, the bot shows the "typing" indicator while Claude works.
- If other runs are ahead, the bot replies `🕐 Got it. 2 run(s) ahead of you, I'll answer when they finish.`

Results follow the task's `notify` setting. A result starts with a header that shows the status icon, the task name and the duration, followed by the answer converted to Telegram formatting. Chat answers that succeed have no header. Up to three messages carry the text. A longer result arrives as a `.md` file. If a run was blocked from using a tool, the message lists the blocked tools and says how to allow them.

## CLI reference

Run `venv/bin/autopilot <command> --help` for the full options of any command. `--debug` works before or after the command and shows debug logs and full tracebacks.

| Command | What it does | Example |
|---|---|---|
| `init` | Creates `tasks.toml` and `.env` from the examples. | `venv/bin/autopilot init` |
| `start` | Runs the scheduler and the bot in the foreground. Stops on Ctrl+C. | `venv/bin/autopilot start` |
| `run <task>` | Runs one task now and prints its result. `--no-notify` skips Telegram. | `venv/bin/autopilot run ai-news --no-notify` |
| `ask "<prompt>"` | Sends a one-off prompt to Claude in a new session. | `venv/bin/autopilot ask "Summarize today's news"` |
| `list` | Shows each task with its schedule, next run, last run and state. | `venv/bin/autopilot list` |
| `history [task]` | Shows recent runs. `-n` sets how many (default 10). | `venv/bin/autopilot history ai-news -n 20` |
| `show <id>` | Shows one run with its full result. | `venv/bin/autopilot show 42` |
| `pause <task>` | Stops a task's schedule. | `venv/bin/autopilot pause ai-news` |
| `resume <task>` | Starts a paused task's schedule again. | `venv/bin/autopilot resume ai-news` |
| `next "<expr>"` | Shows the next fire times of a schedule. `-n` sets how many (default 5). | `venv/bin/autopilot next "weekdays 08:00"` |
| `validate` | Checks `tasks.toml` and lists its tasks and schedules. | `venv/bin/autopilot validate` |
| `doctor` | Checks the whole setup. `--live` also makes one tiny real Claude call. | `venv/bin/autopilot doctor` |
| `telegram-setup` | Connects the bot: token, one-time code, then your confirmation. `--wait` sets the seconds to wait. | `venv/bin/autopilot telegram-setup --wait 300` |
| `install-service` | Installs and starts the systemd user service. | `venv/bin/autopilot install-service` |
| `uninstall-service` | Stops and removes the systemd user service. | `venv/bin/autopilot uninstall-service` |

Examples of the output, from a fresh install with the three example tasks:

```text
$ venv/bin/autopilot list
NAME           SCHEDULE           NEXT       LAST  STATE
ai-news        daily at 07:30     Wed 07:30  -     on
day-plan       weekdays at 06:45  Wed 06:45  -     on
weekly-review  Sun at 19:00       -          -     off

$ venv/bin/autopilot next "weekdays 08:00"
Schedule: weekdays at 08:00
  Wed 2026-11-04 08:00
  Thu 2026-11-05 08:00
  Fri 2026-11-06 08:00
  Mon 2026-11-09 08:00
  Tue 2026-11-10 08:00
```

`history` lists recent runs, newest first. This output comes from a test run with a stand-in for Claude, so the results and times on your machine will differ:

```text
$ venv/bin/autopilot history
ID  WHEN   TASK     STATUS     DURATION  RESULT
2   21:52  ask      ✅ success  5 ms      echo: What is 2+2?
1   21:52  ai-news  ✅ success  5 ms      echo: Search the web for the most important AI news from th…
```

## Running in the background

`install-service` writes a systemd user unit, enables it and starts it. The unit starts at boot, restarts after a failure, and keeps the PATH that was active when you installed it.

```bash
venv/bin/autopilot install-service
systemctl --user status autopilot        # is it running?
journalctl --user -u autopilot -f        # follow the log (Ctrl+C to stop)
systemctl --user stop autopilot          # stop it
venv/bin/autopilot uninstall-service     # stop it and remove the unit
```

The service runs only while you are logged in unless you enable lingering. To keep it running after you log out and across reboots without a login, run this once:

```bash
sudo loginctl enable-linger $USER
```

`doctor` reports the linger state. The log file is also kept at `data/autopilot.log` (rotated at 1 MB, three backups). If you move the project folder, run `install-service` again, because the unit stores the project path.

When the service stops, a run that was still going is recorded as interrupted, with status `error`. No notification is sent to your phone for it. The next start does not run it again.

## Usage limits

Your Claude subscription has usage limits. When Claude reports that a limit is reached, Autopilot handles it like this:

- The run that hit the limit is recorded as `rate_limited`.
- You get one message: `⏳ Claude usage limit reached. All runs wait until 22:56, then retry automatically.` The time is the reset time that Claude reported.
- Other runs wait in the queue without calling Claude. Their status says `Waiting for the Claude usage limit to reset at 22:56.`
- After the reset, the waiting runs retry automatically. Each original run gets at most three retries. If a run hits the limit again after its third retry, it ends with `Gave up after 3 retries because of the usage limit.`

Scheduled runs share the plan's limits with your normal Claude use. A busy day of scheduled tasks can use up the limit that you would otherwise have for your own work. For routine tasks, set a smaller model, such as `model = "sonnet"` or `model = "haiku"`, and use `max_budget_usd` to cap the cost of a run.

## Security model

- The bot answers only your private chat, the one `telegram-setup` saved. Messages from other chats, including groups, get no reply and are logged with their chat id. `telegram-setup` only accepts a code sent from a private chat and asks you to confirm it.
- Tasks run in a private `workspace/` folder by default, which starts empty. Point `workdir` somewhere else only when a task needs your files.
- Claude is always blocked from reading credential and secret files, whatever `allowed_tools` says. That covers `.env` files, SSH and cloud keys (`~/.ssh`, `~/.aws`, `~/.gnupg`), `~/.config/gws`, `~/.config/gh`, `~/.config/gcloud` and `~/.kube`, the Git, Docker and npm token files (`~/.git-credentials`, `~/.netrc`, `~/.docker/config.json`, `~/.npmrc`, `~/.pypirc`), Claude's own credentials and config files, and the autopilot `data/` folder (database and log).
- Tools are allow-listed per task. Anything outside `allowed_tools` is denied under the default `dontAsk` mode. Denied tools are listed in the result as blocked, with a hint.
- Every run removes `ANTHROPIC_API_KEY` and `ANTHROPIC_AUTH_TOKEN` from its environment, so the subscription login is used. `TELEGRAM_*` variables and other `CLAUDE_*` variables are removed too, so the bot token never reaches Claude.
- The bot token is redacted from every log line. `.env` is written with mode 600. The `data/` folder is created owner-only (mode 700), and the database and log files are owner-only too.

Web pages can contain hidden instructions. Keep web tools (`WebSearch`, `WebFetch`) away from tasks that can read your files, and give a task access to a folder that holds personal files only when it needs one.

## Troubleshooting

**`Claude Code CLI not found`.** Run `venv/bin/autopilot doctor` to see where Autopilot looked. Install Claude Code, then run `claude install` from any working copy. If the binary is somewhere else, set `CLAUDE_BIN` in `.env` to its full path. Restart the service after the change.

**The bot does not answer.** Run `venv/bin/autopilot doctor`. Check that `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` are set in `.env`, and that the chat id is your own user id. Send your messages from the private chat with the bot, the one `telegram-setup` saved, because messages from other chats are ignored. If the log shows `Telegram polling failed` with a conflict message, another process is polling the same bot (Telegram answers with HTTP 409 in that case). Stop the other one: `systemctl --user stop autopilot` for the service, or stop the `autopilot start` you ran in a terminal. Then start the service again.

**A result says `Blocked tools: ...`.** The task tried to use tools that are not in `allowed_tools`, so Claude was not allowed to finish the job. Add only the tools the task really needs to `allowed_tools` in `tasks.toml`. Use the narrowest rule you can, for example `Bash(git log *)` instead of `Bash`.

**Something failed and the reason is unclear.** Read the log. It is in `data/autopilot.log` in the project folder, and the service also writes it to the journal: `journalctl --user -u autopilot -n 100`. Run the command by hand with `--debug` to see full tracebacks, for example `venv/bin/autopilot --debug run ai-news --no-notify`.

## Running the tests

```bash
venv/bin/python -m unittest discover -s tests -p "test_*.py" -v
```

The tests use temporary folders, a fake clock, a fake Claude CLI (`tests/fake_claude.py`) and a fake Telegram client. They make no network calls and do not use your subscription.

## How it works

- `autopilot start` runs an engine loop. Every 30 seconds it reloads `tasks.toml` if the file changed, and it asks each task's schedule whether a slot is due.
- A due slot becomes a queued run in a SQLite database, and a worker thread picks it up.
- The worker starts `claude -p` in the task's folder with the prompt on stdin, using a clean environment. It reads the JSON result and stores the outcome.
- The finished run goes to Telegram according to its `notify` setting. The bot's long-polling thread turns your messages into chat turns and commands.
- A usage limit closes a gate in the engine. Every later run waits until the reset time, then the gate reopens.

For the components, data flow and design decisions, see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Project structure

```text
claude-autopilot/
├── pyproject.toml          package metadata and the `autopilot` command
├── requirements.txt        pinned runtime dependency (python-dotenv)
├── .env.example            every environment variable, with comments
├── tasks.example.toml      example tasks, copied to tasks.toml by `init`
├── CHANGELOG.md            release history
├── docs/ARCHITECTURE.md    components, data flow and design decisions
├── autopilot/              the package
│   ├── cli.py              command line interface
│   ├── app.py              wiring, logging, service loop, telegram-setup
│   ├── engine.py           queue, workers, tick loop, usage-limit gate
│   ├── runner.py           builds and runs `claude -p`, reads its output
│   ├── bot.py              Telegram notifier and bot commands
│   ├── scheduler.py        due-slot decisions
│   ├── schedule.py         schedule parsing and next fire times
│   ├── tasks.py            tasks.toml parsing and validation
│   ├── settings.py         settings from the environment and .env
│   ├── store.py            SQLite persistence
│   ├── formatting.py       Markdown to Telegram HTML, message texts
│   ├── telegram_api.py     Telegram Bot API client
│   ├── service.py          systemd user unit
│   ├── doctor.py           health checks
│   ├── system.py           runs OS commands
│   ├── models.py           shared dataclasses and constants
│   └── errors.py           exception types
└── tests/                  unit and integration tests, fake Claude CLI
```

`data/`, `workspace/`, `tasks.toml`, `.env` and `venv/` are created locally and are git-ignored.
