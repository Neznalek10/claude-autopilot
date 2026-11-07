"""Read-only web viewer for Autopilot chats, task runs and (optionally) Claude Code transcripts.

The viewer is off unless ``AUTOPILOT_WEB_TOKEN`` is set. It only answers GET and HEAD, needs the
token (as ``?token=`` once, then a cookie), listens on loopback by default, sets strict browser
security headers, escapes everything it shows and hides secrets that look like tokens or keys.
Pages are plain server-rendered HTML with a phone-first layout and no JavaScript.
"""

from __future__ import annotations

import hmac
import html
import json
import logging
import re
import threading
import time
from collections import defaultdict
from datetime import datetime, tzinfo
from http import HTTPStatus
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, urlsplit

from autopilot.models import ASK_TASK_NAME, STATUS_SUCCESS, RunRecord
from autopilot.settings import Settings
from autopilot.store import Store

logger = logging.getLogger(__name__)

COOKIE_NAME = "autopilot_viewer"
COOKIE_MAX_AGE = 30 * 24 * 3600
LOGIN_DELAY_SECONDS = 0.5
REQUEST_TIMEOUT_SECONDS = 15
LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}
PAGE_SIZE = 60
MAX_PAGES = 1000
MAX_THREAD_MESSAGES = 300
MAX_MESSAGE_CHARS = 20000
MAX_SEARCH_FILES_HITS = 30
SAFE_NAME_RE = re.compile(r"^[A-Za-z0-9._-]+$")
SECRET_PATTERNS = (
    re.compile(r"(?<!\d)\d{8,11}:[A-Za-z0-9_-]{30,}"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{16,}"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}"),
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}"),
    re.compile(r"\bAIza[A-Za-z0-9_-]{30,}"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}"),
    re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]{16,}"),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?(?:-----END [A-Z ]*PRIVATE KEY-----|\Z)", re.S),
    re.compile(r"(?i)(token|secret|passw(?:or)?d|api[_-]?key|authorization)([\"']?\s*[=:]\s*[\"']?)[^\s\"',}]{8,}"),
)
FENCE_RE = re.compile(r"```[^\n]*\n(.*?)(?:```|\Z)", re.S)
INLINE_RE = re.compile(r"`([^`\n]+)`")
STYLE = """
:root{--bg:#f6f7f9;--fg:#1c1e21;--muted:#667;--card:#fff;--line:#dde0e5;--accent:#1b6ef3;--on:#fff;--you:#e6f0ff;--bad:#c62828}
@media (prefers-color-scheme:dark){:root{--bg:#121417;--fg:#e8eaed;--muted:#9aa;--card:#1b1e23;--line:#2c3037;--accent:#6aa5ff;--on:#0b1020;--you:#1d2b45;--bad:#ef6a6a}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:16px/1.5 system-ui,sans-serif}
header{position:sticky;top:0;background:var(--card);border-bottom:1px solid var(--line);padding:8px 16px;z-index:1}.wrap{max-width:760px;margin:0 auto}
nav{display:flex;gap:4px;flex-wrap:wrap}nav a{padding:10px 14px;border-radius:8px;color:var(--fg);text-decoration:none}
nav a.on{background:var(--accent);color:var(--on)}form.s{display:flex;gap:8px;margin-top:6px}
input[type=search],input[type=password]{flex:1;min-width:0;padding:10px;border:1px solid var(--line);border-radius:8px;background:var(--bg);color:var(--fg);font:inherit}
button{padding:10px 14px;border:0;border-radius:8px;background:var(--accent);color:var(--on);font:inherit}
main{max-width:760px;margin:0 auto;padding:12px 16px 40px}h1{font-size:1.2rem;margin:8px 0}
.card{display:block;background:var(--card);border:1px solid var(--line);border-radius:12px;padding:12px 14px;margin:10px 0;color:inherit;text-decoration:none}
.card small,.meta{color:var(--muted)}.bad{color:var(--bad)}.t{font-weight:600;overflow-wrap:anywhere}
.msg{border-radius:14px;padding:10px 14px;margin:10px 0;white-space:pre-wrap;overflow-wrap:anywhere}
.you{background:var(--you)}.ai{background:var(--card);border:1px solid var(--line)}.you::before,.ai::before{display:block;font-size:.75rem;color:var(--muted);font-weight:600}.you::before{content:"You"}.ai::before{content:"Claude"}h1 a{color:inherit;display:inline-block;padding:8px 0}main a{color:var(--accent)}
pre{background:var(--bg);border:1px solid var(--line);border-radius:8px;padding:8px;overflow-x:auto;white-space:pre}
code{background:var(--bg);padding:1px 4px;border-radius:4px}.pager{display:flex;justify-content:space-between;margin:14px 0}
"""
TABS = (("chats", "Chats"), ("runs", "Runs"), ("code", "VS Code"))


def redact(text: str) -> str:
    """Hide anything that looks like a token, key or password."""
    for pattern in SECRET_PATTERNS:
        if pattern.groups >= 2:
            text = pattern.sub(lambda m: f"{m.group(1)}{m.group(2)}[hidden]", text)
        else:
            text = pattern.sub("[hidden]", text)
    return text


def render_text(text: str) -> str:
    """Escape ``text`` and turn code fences and inline code into ``pre`` and ``code``."""
    safe = html.escape(redact(text)[:MAX_MESSAGE_CHARS].replace("\x00", ""))
    blocks: list[str] = []

    def stash(match: re.Match[str]) -> str:
        blocks.append(f"<pre>{match.group(1)}</pre>")
        return f"\x00{len(blocks) - 1}\x00"

    safe = FENCE_RE.sub(stash, safe)
    safe = INLINE_RE.sub(lambda m: f"<code>{m.group(1)}</code>", safe)
    return re.sub(r"\x00(\d+)\x00", lambda m: blocks[int(m.group(1))], safe)


def clip(text: str, length: int = 90) -> str:
    """One-line preview of ``text``."""
    line = " ".join(redact(text).replace("```", " ").replace("**", "").split())
    return line if len(line) <= length else line[: length - 1] + "…"


def stamp(moment: datetime | None, zone: tzinfo) -> str:
    return moment.astimezone(zone).strftime("%a %d.%m. %H:%M") if moment else ""


def page(title: str, body: str, active: str = "", query: str = "", *, header: bool = True) -> str:
    tabs = "".join(f'<a href="/{key}"{" class=on" if key == active else ""}>{label}</a>' for key, label in TABS)
    top = (
        f"<header><div class=wrap><nav>{tabs}</nav><form class=s action=/search><input type=search name=q "
        f"aria-label=Search value=\"{html.escape(query)}\" placeholder=Search><button>Go</button></form></div></header>"
        if header else ""
    )
    return (
        "<!doctype html><html lang=en><head><meta charset=utf-8>"
        "<meta name=viewport content='width=device-width,initial-scale=1'><link rel=icon href=\"data:,\">"
        f"<title>{html.escape(title)}</title><style>{STYLE}</style></head><body>{top}"
        f"<main>{body}</main></body></html>"
    )


NOT_FOUND = '<h1>Not found</h1><p><a href="/chats">Back to chats</a></p>'


class CodeTranscripts:
    """Reads Claude Code transcript files (JSONL) under one folder, read only."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self._cache: dict[Path, tuple[float, dict[str, object]]] = {}

    def _files(self) -> list[Path]:
        """Transcript files that really live inside the folder (symlinks pointing outside are skipped)."""
        inside = [p for p in self.root.glob("*/*.jsonl") if p.is_file() and p.resolve().parent.parent == self.root]
        return sorted(inside, key=lambda p: p.stat().st_mtime, reverse=True)

    def resolve(self, project: str, session: str) -> Path | None:
        """Return the transcript path for the two url parts, or None when they are unsafe or missing."""
        if not (SAFE_NAME_RE.match(project) and SAFE_NAME_RE.match(session)):
            return None
        path = (self.root / project / f"{session}.jsonl").resolve()
        return path if path.parent.parent == self.root and path.is_file() else None

    def summary(self, path: Path) -> dict[str, object]:
        """Title, folder and first prompt of one transcript, cached by modification time."""
        mtime = path.stat().st_mtime
        cached = self._cache.get(path)
        if cached and cached[0] == mtime:
            return cached[1]
        title, first, cwd = "", "", ""
        with open(path, encoding="utf-8", errors="replace") as stream:
            for line in stream:
                if '"ai-title"' in line:
                    entry = _loads(line)
                    title = str(entry.get("aiTitle") or title)
                elif not first and _may_be_message(line):
                    entry = _loads(line)
                    for role, text in _entry_messages(entry):
                        if role == "you":
                            first, cwd = text, str(entry.get("cwd") or "")
                            break
        info: dict[str, object] = {"title": title or clip(first, 80) or "(empty)", "cwd": cwd, "mtime": mtime}
        self._cache[path] = (mtime, info)
        return info

    def messages(self, path: Path) -> list[tuple[str, str]]:
        """The visible conversation: your messages and Claude's text, without tool traffic."""
        out: list[tuple[str, str]] = []
        with open(path, encoding="utf-8", errors="replace") as stream:
            for line in stream:
                if _may_be_message(line):
                    out.extend(_entry_messages(_loads(line)))
        return out

    def search(self, needle: str) -> list[tuple[Path, str]]:
        """Transcripts whose visible text contains ``needle`` (case-insensitive), newest first.

        Tool output and thinking are not searched, and snippets are redacted before they are cut.
        """
        wanted = needle.lower()
        hits: list[tuple[Path, str]] = []
        for path in self._files():
            try:
                if wanted.encode() not in path.read_bytes().lower():
                    continue
                messages = self.messages(path)
            except OSError:
                continue
            for _role, text in messages:
                hidden = redact(text)
                at = hidden.lower().find(wanted)
                if at >= 0:
                    hits.append((path, clip(hidden[max(0, at - 60): at + 100], 140)))
                    break
            if len(hits) >= MAX_SEARCH_FILES_HITS:
                break
        return hits


def _may_be_message(line: str) -> bool:
    """Cheap pre-check so only user and assistant lines are parsed as JSON."""
    head = line[:500]
    return '"user"' in head or '"assistant"' in head


def _loads(line: str) -> dict[str, object]:
    try:
        value = json.loads(line)
    except (ValueError, RecursionError):
        return {}
    return value if isinstance(value, dict) else {}


def _entry_messages(entry: dict[str, object]) -> list[tuple[str, str]]:
    """Visible text of one transcript entry as (role, text) pairs; tool calls and thinking are skipped."""
    kind = entry.get("type")
    message = entry.get("message")
    if kind not in ("user", "assistant") or not isinstance(message, dict) or entry.get("isSidechain") or entry.get("isMeta"):
        return []
    content = message.get("content")
    blocks = [{"type": "text", "text": content}] if isinstance(content, str) else content
    role = "you" if kind == "user" else "claude"
    result = []
    for block in blocks if isinstance(blocks, list) else []:
        if isinstance(block, dict) and block.get("type") == "text":
            text = str(block.get("text", "")).strip()
            if text and not text.startswith("<system-reminder>"):
                result.append((role, text))
    return result


class Viewer:
    """Builds every page from the run store and the optional transcripts."""

    def __init__(self, store: Store, zone: tzinfo, transcripts: CodeTranscripts | None) -> None:
        self.store = store
        self.zone = zone
        self.transcripts = transcripts

    def route(self, path: str, query: dict[str, list[str]]) -> tuple[int, str]:
        """Return (status, html) for a path."""
        first = lambda key: (query.get(key) or [""])[0]  # noqa: E731
        if path in ("/", "/chats"):
            return 200, self.chats()
        if path == "/chat":
            return self.chat(first("s"))
        if path == "/runs":
            return 200, self.runs(first("task"), int(first("page")) if first("page").isdigit() and int(first("page")) <= MAX_PAGES else 0)
        if path.startswith("/run/") and path[5:].isdigit():
            return self.run(int(path[5:]))
        if path == "/code":
            return self.code_list()
        if path.startswith("/code/"):
            parts = path.split("/")
            return self.code_thread(parts[2], parts[3]) if len(parts) == 4 else (404, page("Not found", NOT_FOUND))
        if path == "/search":
            return 200, self.search(first("q").replace("\x00", "").strip())
        return 404, page("Not found", NOT_FOUND)

    def _card(self, href: str, title: str, meta: str, bad: bool = False) -> str:
        mark = ' class="t bad"' if bad else ' class=t'
        return f'<a class=card href="{href}"><div{mark}>{html.escape(title)}</div><small>{html.escape(meta)}</small></a>'

    def chats(self) -> str:
        groups: dict[str, list[RunRecord]] = defaultdict(list)
        for run in self.store.list_runs(task_name=ASK_TASK_NAME, limit=500):
            groups[run.session_id or f"run{run.id}"].append(run)
        cards = []
        for session, runs in sorted(groups.items(), key=lambda kv: kv[1][0].id, reverse=True)[:PAGE_SIZE]:
            oldest = runs[-1]
            cards.append(self._card(f"/chat?s={quote(session)}", clip(oldest.prompt),
                                    f"{len(runs)} message{'s' if len(runs) != 1 else ''} · last {stamp(runs[0].queued_at, self.zone)}"))
        return page("Chats", "<h1>Chats</h1>" + ("".join(cards) or "<p class=meta>No chats yet.</p>"), "chats")

    def chat(self, session: str) -> tuple[int, str]:
        runs = self.store.runs_of_session(session)
        if not runs and session.startswith("run") and session[3:].isdigit():
            single = self.store.get_run(int(session[3:]))
            runs = [single] if single and single.task_name == ASK_TASK_NAME and not single.session_id else []
        if not runs:
            return 404, page("Not found", NOT_FOUND, "chats")
        body = ["<h1>Chat</h1>"]
        for run in reversed(runs):
            body.append(f"<div class='msg you'>{render_text(run.prompt)}</div>")
            body.append(f"<div class='msg ai'>{render_text(run.result or run.error or '(no answer)')}</div>")
        return 200, page(clip(runs[-1].prompt, 60), "".join(body), "chats")

    def runs(self, task: str, page_no: int) -> str:
        runs = self.store.list_runs(task_name=task or None, limit=PAGE_SIZE * (page_no + 1))[PAGE_SIZE * page_no:]
        cards = [self._card(f"/run/{r.id}", f"{r.task_name}: {clip(r.result or r.error or r.prompt, 70)}",
                            f"{stamp(r.queued_at, self.zone)} · {r.status}", r.status != STATUS_SUCCESS) for r in runs]
        pager = f'<div class=pager><span></span><a href="/runs?task={quote(task)}&page={page_no + 1}">Older</a></div>' if len(runs) == PAGE_SIZE else ""
        title = f"Runs of {task}" if task else "Runs"
        return page(title, f"<h1>{html.escape(title)}</h1>" + ("".join(cards) or "<p class=meta>No runs.</p>") + pager, "runs")

    def run(self, run_id: int) -> tuple[int, str]:
        run = self.store.get_run(run_id)
        if run is None:
            return 404, page("Not found", NOT_FOUND, "runs")
        meta = f"{stamp(run.queued_at, self.zone)} · {run.status} · {run.trigger}"
        body = (f"<h1><a href='/runs?task={quote(run.task_name)}'>{html.escape(run.task_name)}</a></h1><p class=meta>{html.escape(meta)}</p>"
                f"<div class='msg you'>{render_text(run.prompt)}</div>"
                f"<div class='msg ai'>{render_text(run.result or run.error or '(no result)')}</div>")
        return 200, page(f"{run.task_name} ({run.status})", body, "runs")

    def code_list(self) -> tuple[int, str]:
        if self.transcripts is None:
            note = "<p class=meta>Set AUTOPILOT_CLAUDE_PROJECTS_DIR in .env to browse your Claude Code chats here.</p>"
            return 200, page("VS Code", "<h1>VS Code chats</h1>" + note, "code")
        cards = []
        for path in self.transcripts._files()[:PAGE_SIZE]:
            info = self.transcripts.summary(path)
            when = stamp(datetime.fromtimestamp(float(info["mtime"]), self.zone), self.zone)  # type: ignore[arg-type]
            cards.append(self._card(f"/code/{quote(path.parent.name)}/{quote(path.stem)}", str(info["title"]), f"{when} · {info['cwd']}"))
        return 200, page("VS Code", "<h1>VS Code chats</h1>" + ("".join(cards) or "<p class=meta>No transcripts found.</p>"), "code")

    def code_thread(self, project: str, session: str) -> tuple[int, str]:
        path = self.transcripts.resolve(project, session) if self.transcripts else None
        if path is None:
            return 404, page("Not found", NOT_FOUND, "code")
        messages = self.transcripts.messages(path)  # type: ignore[union-attr]
        note = f"<p class=meta>Showing the last {MAX_THREAD_MESSAGES} of {len(messages)} messages.</p>" if len(messages) > MAX_THREAD_MESSAGES else ""
        body = "".join(f"<div class='msg {'you' if role == 'you' else 'ai'}'>{render_text(text)}</div>" for role, text in messages[-MAX_THREAD_MESSAGES:])
        title = str(self.transcripts.summary(path)["title"])  # type: ignore[union-attr]
        return 200, page(title, f"<h1>{html.escape(title)}</h1>{note}{body}", "code")

    def search(self, needle: str) -> str:
        if not needle:
            return page("Search", "<h1>Search</h1><p class=meta>Type something above.</p>")
        cards = [self._card(f"/run/{r.id}", f"{r.task_name}: {clip(r.prompt, 70)}", f"{stamp(r.queued_at, self.zone)} · {clip(r.result, 100)}")
                 for r in self.store.search_runs(needle, limit=PAGE_SIZE)]
        if self.transcripts is not None:
            for path, snippet in self.transcripts.search(needle):
                cards.append(self._card(f"/code/{quote(path.parent.name)}/{quote(path.stem)}", str(self.transcripts.summary(path)["title"]), snippet))
        return page(f"Search: {needle}", f"<h1>Results for {html.escape(needle)}</h1>" + ("".join(cards) or "<p class=meta>Nothing found.</p>"), "", needle)


def make_handler(viewer: Viewer, token: str, loopback_only: bool) -> type[BaseHTTPRequestHandler]:
    """Create the request handler class bound to a viewer and a token."""

    cookie_value = hmac.new(token.encode(), b"autopilot-viewer-cookie", "sha256").hexdigest()

    class Handler(BaseHTTPRequestHandler):
        timeout = REQUEST_TIMEOUT_SECONDS
        server_version = "AutopilotViewer"
        sys_version = ""

        def log_message(self, format: str, *args: object) -> None:  # noqa: A002 - signature fixed by the base class
            """Silence access logs: urls can carry the login token."""

        def _send(self, status: int, body: str, extra: dict[str, str] | None = None) -> None:
            data = body.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("X-Frame-Options", "DENY")
            self.send_header("Content-Security-Policy", "default-src 'none'; style-src 'unsafe-inline'; form-action 'self'; base-uri 'none'; frame-ancestors 'none'")
            for key, value in (extra or {}).items():
                self.send_header(key, value)
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(data)

        def _authorised(self, supplied: str) -> bool:
            return bool(supplied) and hmac.compare_digest(supplied.encode(), token.encode())

        def _cookie_authorised(self, supplied: str) -> bool:
            return bool(supplied) and hmac.compare_digest(supplied.encode(), cookie_value.encode())

        def _host_ok(self) -> bool:
            if not loopback_only:
                return True
            host = (self.headers.get("Host") or "").rsplit(":", 1)[0].strip("[]")
            return host in LOOPBACK_HOSTS

        def _login_page(self, message: str = "") -> str:
            form = ("<h1>Autopilot</h1>" + (f"<p class=bad>{html.escape(message)}</p>" if message else "") +
                    "<form class=s action=/><input type=password name=token aria-label=Token autocomplete=current-password placeholder=Token autofocus><button>Open</button></form>")
            return page("Sign in", form, header=False)

        def do_GET(self) -> None:  # noqa: N802 - name fixed by the base class
            if not self._host_ok():
                self._send(HTTPStatus.FORBIDDEN, page("Forbidden", "<h1>Forbidden</h1>"))
                return
            parts = urlsplit(self.path)
            query = parse_qs(parts.query)
            supplied = (query.get("token") or [""])[0]
            if supplied:
                if self._authorised(supplied):
                    cookie = f"{COOKIE_NAME}={cookie_value}; HttpOnly; SameSite=Strict; Path=/; Max-Age={COOKIE_MAX_AGE}"
                    self.send_response(HTTPStatus.SEE_OTHER)
                    self.send_header("Location", parts.path or "/")
                    self.send_header("Set-Cookie", cookie)
                    self.send_header("Cache-Control", "no-store")
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                time.sleep(LOGIN_DELAY_SECONDS)
                self._send(HTTPStatus.UNAUTHORIZED, self._login_page("Wrong token."))
                return
            jar = SimpleCookie(self.headers.get("Cookie", ""))
            if COOKIE_NAME not in jar or not self._cookie_authorised(jar[COOKIE_NAME].value):
                self._send(HTTPStatus.UNAUTHORIZED, self._login_page())
                return
            try:
                status, body = viewer.route(parts.path, query)
            except Exception:  # A broken page must not kill the viewer thread, so log it and answer 500.
                logger.exception("Viewer error on %s", parts.path)
                status, body = 500, page("Error", "<h1>Something went wrong</h1>")
            self._send(status, body)

        do_HEAD = do_GET  # noqa: N815

        def _method_not_allowed(self) -> None:
            self._send(HTTPStatus.METHOD_NOT_ALLOWED, page("Not allowed", "<h1>Read only</h1>"), {"Allow": "GET, HEAD"})

        do_POST = do_PUT = do_DELETE = do_PATCH = _method_not_allowed  # noqa: N815

    return Handler


def start_web(settings: Settings, store: Store) -> ThreadingHTTPServer | None:
    """Start the viewer in a daemon thread; None when it is off or the port cannot be used."""
    if not settings.web_token:
        return None
    transcripts = CodeTranscripts(settings.claude_projects_dir) if settings.claude_projects_dir else None
    viewer = Viewer(store, settings.timezone, transcripts)
    handler = make_handler(viewer, settings.web_token, settings.web_host in LOOPBACK_HOSTS)
    try:
        server = ThreadingHTTPServer((settings.web_host, settings.web_port), handler)
    except OSError as err:
        logger.error("Web viewer could not listen on %s:%s: %s", settings.web_host, settings.web_port, err)
        return None
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, name="web-viewer", daemon=True).start()
    logger.info("Web viewer on http://%s:%s", settings.web_host, server.server_address[1])
    return server
