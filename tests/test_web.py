"""Tests for the read-only web viewer: pages, login, security headers and transcript reading."""

import hmac
import http.client
import json
import socket
import tempfile
import threading
import unittest
from datetime import datetime, timezone
from http.server import ThreadingHTTPServer
from pathlib import Path
from unittest import mock
from zoneinfo import ZoneInfo

from autopilot import web
from autopilot.errors import ConfigError
from autopilot.settings import load_settings
from autopilot.store import Store

TOKEN = "t" * 24
COOKIE = hmac.new(TOKEN.encode(), b"autopilot-viewer-cookie", "sha256").hexdigest()
ZONE = ZoneInfo("Europe/Berlin")
NOW = datetime(2026, 11, 7, 10, 0, tzinfo=timezone.utc)


def line(**entry: object) -> str:
    return json.dumps(entry)


def transcript_lines() -> list[str]:
    return [
        line(type="user", cwd="/work/app", message={"content": "<system-reminder>ignore me</system-reminder>"}),
        line(type="user", cwd="/work/app", message={"content": [{"type": "text", "text": "Build the <b>app</b>"}]}),
        line(type="assistant", message={"content": [
            {"type": "thinking", "thinking": "hidden thoughts"},
            {"type": "text", "text": "Sure.\n```py\nprint(1)\n```"},
            {"type": "tool_use", "name": "Bash", "input": {"command": "ls"}},
        ]}),
        line(type="user", message={"content": [{"type": "tool_result", "content": "tool output"}]}),
        line(type="user", isSidechain=True, message={"content": "sub agent text"}),
        line(type="user", isMeta=True, message={"content": "meta text"}),
        line(type="ai-title", aiTitle="Build an app"),
    ]


class RedactAndRenderTests(unittest.TestCase):
    def test_redacts_tokens_keys_and_passwords(self) -> None:
        for secret in ("123456789:" + "A" * 35, "sk-ant-" + "k" * 12, "ghp_" + "a" * 25, "AIza" + "b" * 35):
            self.assertEqual(web.redact(f"x {secret} y"), "x [hidden] y")
        self.assertEqual(web.redact("password = " + "p" * 10), "password = [hidden]")
        self.assertIn("[hidden]", web.redact("-----BEGIN PRIVATE KEY-----\nabc\n-----END PRIVATE KEY-----"))
        self.assertEqual(web.redact("plain text"), "plain text")

    def test_render_escapes_html_and_formats_code(self) -> None:
        rendered = web.render_text("<script>alert(1)</script> `x<y` ```\n<b>\n```")
        self.assertNotIn("<script>", rendered)
        self.assertIn("&lt;script&gt;", rendered)
        self.assertIn("<code>x&lt;y</code>", rendered)
        self.assertIn("<pre>&lt;b&gt;\n</pre>", rendered)

    def test_render_hides_secrets_and_clip_shortens(self) -> None:
        self.assertIn("[hidden]", web.render_text("key sk-ant-" + "k" * 12))
        self.assertEqual(len(web.clip("word " * 100, 20)), 20)


class RedactMoreTests(unittest.TestCase):
    def test_more_secret_forms_are_hidden(self) -> None:
        filler = "x" * 16
        secrets = (
            "curl https://api.telegram.org/bot" + "123456789:" + "A" * 35 + "/getMe",
            "access_token=" + filler, "OPENAI_API_KEY=" + filler, "client_secret: " + filler,
            "GITHUB_TOKEN=" + filler, '{"api_key": "' + filler + '"}', '{"password": "' + filler + '"}',
            "password='" + filler + "'", "sk-proj-" + "y" * 20, "AKIA" + "Z" * 16, "xoxb-" + "1" * 10 + "-" + "z" * 6,
            "Authorization: Bearer " + "q" * 20, "12345678901:" + "A" * 35,
        )
        for secret in secrets:
            with self.subTest(secret=secret):
                self.assertIn("[hidden]", web.redact(secret))
                self.assertNotIn(filler, web.redact(secret))

    def test_render_survives_placeholder_lookalikes(self) -> None:
        self.assertEqual(web.render_text("a \x000\x00 b"), "a 0 b")

    def test_clip_drops_markdown_noise(self) -> None:
        self.assertEqual(web.clip("**Bold** ```py x```"), "Bold py x")


class TranscriptTests(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        (self.root / "proj").mkdir()
        self.file = self.root / "proj" / "abc-123.jsonl"
        self.file.write_text("\n".join(transcript_lines()) + "\nnot json\n", encoding="utf-8")
        self.transcripts = web.CodeTranscripts(self.root)

    def test_messages_keep_only_visible_text(self) -> None:
        messages = self.transcripts.messages(self.file)
        self.assertEqual([role for role, _ in messages], ["you", "claude"])
        self.assertEqual(messages[0][1], "Build the <b>app</b>")
        self.assertNotIn("hidden thoughts", str(messages))
        self.assertNotIn("tool output", str(messages))

    def test_summary_uses_title_folder_and_cache(self) -> None:
        info = self.transcripts.summary(self.file)
        self.assertEqual((info["title"], info["cwd"]), ("Build an app", "/work/app"))
        self.assertIs(self.transcripts.summary(self.file), info)

    def test_summary_falls_back_to_first_prompt(self) -> None:
        self.file.write_text(transcript_lines()[1] + "\n", encoding="utf-8")
        self.assertEqual(self.transcripts.summary(self.file)["title"], "Build the <b>app</b>")

    def test_resolve_rejects_unsafe_names(self) -> None:
        self.assertEqual(self.transcripts.resolve("proj", "abc-123"), self.file.resolve())
        for project, session in (("..", "abc-123"), ("proj", "../proj/abc-123"), ("pro/j", "x"), ("proj", "missing"), ("", "")):
            self.assertIsNone(self.transcripts.resolve(project, session), (project, session))

    def test_symlink_out_of_the_folder_is_refused(self) -> None:
        outside = self.root.parent / f"outside-{self.root.name}.jsonl"
        outside.write_text("{}", encoding="utf-8")
        self.addCleanup(outside.unlink)
        (self.root / "proj" / "link.jsonl").symlink_to(outside)
        self.assertIsNone(self.transcripts.resolve("proj", "link"))

    def test_search_ignores_hidden_content_and_redacts_snippets(self) -> None:
        secret = "AIza" + "C" * 35
        self.file.write_text("\n".join([
            line(type="user", message={"content": f"my key {secret} ok"}),
            line(type="assistant", message={"content": [{"type": "thinking", "thinking": "THINKSECRET"},
                                                         {"type": "tool_use", "name": "x", "input": {"a": "TOOLSECRET"}}]}),
            line(type="user", message={"content": [{"type": "tool_result", "content": "RESULTSECRET"}]}),
        ]) + "\n", encoding="utf-8")
        for hidden in ("THINKSECRET", "TOOLSECRET", "RESULTSECRET", '"type"'):
            self.assertEqual(self.transcripts.search(hidden), [], hidden)
        (_, snippet), = self.transcripts.search("my key")
        self.assertIn("[hidden]", snippet)
        self.assertNotIn("CCCC", snippet)

    def test_symlinks_leaving_the_folder_are_not_listed_or_searched(self) -> None:
        outside = self.root.parent / f"out-{self.root.name}"
        outside.mkdir()
        (outside / "s.jsonl").write_text(transcript_lines()[1] + "\n", encoding="utf-8")
        self.addCleanup(outside.rmdir)
        self.addCleanup((outside / "s.jsonl").unlink)
        (self.root / "linkdir").symlink_to(outside)
        (self.root / "proj" / "linkfile.jsonl").symlink_to(outside / "s.jsonl")
        self.assertEqual([p.name for p in self.transcripts._files()], ["abc-123.jsonl"])
        self.assertEqual([p for p, _ in self.transcripts.search("Build the")], [self.file])

    def test_directory_named_like_a_transcript_and_deep_json_do_not_crash(self) -> None:
        (self.root / "proj" / "dir.jsonl").mkdir()
        self.file.write_text("[" * 100000 + "\n" + transcript_lines()[1] + "\n", encoding="utf-8")
        self.assertEqual([p.name for p in self.transcripts._files()], ["abc-123.jsonl"])
        self.assertEqual(len(self.transcripts.messages(self.file)), 1)

    def test_search_finds_text_case_insensitively(self) -> None:
        self.assertEqual([path for path, _ in self.transcripts.search("BUILD THE")], [self.file])
        self.assertEqual(self.transcripts.search("nothing like this"), [])


class ServerTestCase(unittest.TestCase):
    """A real server on a free port with a store holding two chat runs and one failed task run."""

    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        self.store = Store(root / "db.sqlite")
        for prompt, result, session in (("hello <i>there</i>", "hi! `code`", "s1"), ("second", "answer two", "s1"), ("other chat 50%", "ok", "s2")):
            run = self.store.create_run(task_name="ask", trigger="chat", prompt=prompt, queued_at=NOW)
            self.store.finish_run(run, status="success", finished_at=NOW, result=result, session_id=session)
        run = self.store.create_run(task_name="news", trigger="schedule", prompt="p", queued_at=NOW)
        self.store.finish_run(run, status="error", finished_at=NOW, result="", error="boom")
        (root / "proj").mkdir()
        (root / "proj" / "abc-123.jsonl").write_text("\n".join(transcript_lines()) + "\n", encoding="utf-8")
        viewer = web.Viewer(self.store, ZONE, web.CodeTranscripts(root))
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), web.make_handler(viewer, TOKEN, True))
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        self.port = self.server.server_address[1]

    def get(self, path: str, *, cookie: bool = True, method: str = "GET", host: str | None = None):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        headers = {"Cookie": f"{web.COOKIE_NAME}={COOKIE}"} if cookie else {}
        if host:
            headers["Host"] = host
        conn.request(method, path, headers=headers)
        response = conn.getresponse()
        body = response.read().decode()
        conn.close()
        return response, body


class PageTests(ServerTestCase):
    def test_chats_group_by_session(self) -> None:
        response, body = self.get("/chats")
        self.assertEqual(response.status, 200)
        self.assertIn("2 messages", body)
        self.assertIn("1 message \u00b7", body)

    def test_chat_thread_escapes_and_formats(self) -> None:
        _, body = self.get("/chat?s=s1")
        self.assertIn("hello &lt;i&gt;there&lt;/i&gt;", body)
        self.assertIn("<code>code</code>", body)
        self.assertLess(body.index("hello"), body.index("second"))
        self.assertEqual(self.get("/chat?s=nope")[0].status, 404)

    def test_runs_filter_and_detail(self) -> None:
        _, body = self.get("/runs?task=news")
        self.assertIn("news: boom", body)
        self.assertNotIn("hi!", body)
        self.assertIn("bad", body)
        self.assertEqual(self.get("/run/4")[0].status, 200)
        self.assertEqual(self.get("/run/999")[0].status, 404)

    def test_code_pages(self) -> None:
        _, listing = self.get("/code")
        self.assertIn("/code/proj/abc-123", listing)
        _, thread = self.get("/code/proj/abc-123")
        self.assertIn("Build the &lt;b&gt;app&lt;/b&gt;", thread)
        self.assertNotIn("sub agent text", thread)
        self.assertEqual(self.get("/code/proj/..%2Fx")[0].status, 404)
        self.assertEqual(self.get("/code/%2e%2e/abc-123")[0].status, 404)

    def test_search_covers_runs_and_transcripts_and_escapes(self) -> None:
        _, body = self.get("/search?q=50%25")
        self.assertIn("other chat 50%", body)
        _, body = self.get("/search?q=build+the")
        self.assertIn("Build an app", body)
        _, body = self.get("/search?q=%3Cscript%3E")
        self.assertNotIn("<script>", body)
        self.assertIn("Nothing found", body)

    def test_login_page_has_no_header_and_titles_are_specific(self) -> None:
        _, login = self.get("/", cookie=False)
        self.assertNotIn("<nav>", login)
        self.assertIn('aria-label=Token', login)
        _, chat = self.get("/chat?s=s1")
        self.assertIn("<title>hello &lt;i&gt;there&lt;/i&gt;</title>", chat)
        _, run = self.get("/run/4")
        self.assertIn("<title>news (error)</title>", run)
        self.assertIn('aria-label=Search', chat)
        self.assertIn('rel=icon', chat)

    def test_old_chats_stay_reachable_and_single_runs_work(self) -> None:
        for i in range(520):
            run = self.store.create_run(task_name="ask", trigger="chat", prompt=f"p{i}", queued_at=NOW)
            self.store.finish_run(run, status="success", finished_at=NOW, result="r", session_id=f"bulk{i}")
        self.assertEqual(self.get("/chat?s=s1")[0].status, 200)
        lone = self.store.create_run(task_name="ask", trigger="chat", prompt="lonely", queued_at=NOW)
        self.store.finish_run(lone, status="success", finished_at=NOW, result="r")
        self.assertIn("lonely", self.get(f"/chat?s=run{lone}")[1])
        self.assertEqual(self.get("/chat?s=run3")[0].status, 404)

    def test_nul_and_huge_page_numbers_are_harmless(self) -> None:
        self.assertEqual(self.get("/runs?page=99999999999999999999")[0].status, 200)
        _, body = self.get("/search?q=%00")
        self.assertIn("Type something", body)
        self.assertEqual(self.store.search_runs("a\x00b"), [])

    def test_unknown_path_and_bad_page_number(self) -> None:
        self.assertEqual(self.get("/nope")[0].status, 404)
        self.assertEqual(self.get("/runs?page=abc")[0].status, 200)

    def test_code_page_explains_when_not_configured(self) -> None:
        viewer = web.Viewer(self.store, ZONE, None)
        self.assertIn("AUTOPILOT_CLAUDE_PROJECTS_DIR", viewer.route("/code", {})[1])
        self.assertEqual(viewer.route("/code/a/b", {})[0], 404)


class SecurityTests(ServerTestCase):
    def test_no_cookie_shows_login_and_hides_data(self) -> None:
        response, body = self.get("/chats", cookie=False)
        self.assertEqual(response.status, 401)
        self.assertIn("name=token", body)
        self.assertNotIn("hello", body)

    def test_wrong_token_is_rejected_after_a_delay(self) -> None:
        with mock.patch("autopilot.web.time.sleep") as sleep:
            response, body = self.get("/?token=wrong", cookie=False)
        self.assertEqual(response.status, 401)
        self.assertIn("Wrong token", body)
        sleep.assert_called_once()

    def test_right_token_sets_a_strict_cookie_and_redirects(self) -> None:
        response, _ = self.get(f"/runs?token={TOKEN}", cookie=False)
        self.assertEqual(response.status, 303)
        self.assertEqual(response.getheader("Location"), "/runs")
        cookie = response.getheader("Set-Cookie")
        for part in ("HttpOnly", "SameSite=Strict", f"{web.COOKIE_NAME}={COOKIE}"):
            self.assertIn(part, cookie)
        self.assertNotIn(TOKEN, cookie)

    def test_security_headers(self) -> None:
        response, _ = self.get("/chats")
        self.assertEqual(response.getheader("X-Content-Type-Options"), "nosniff")
        self.assertEqual(response.getheader("Cache-Control"), "no-store")
        self.assertIn("default-src 'none'", response.getheader("Content-Security-Policy"))
        self.assertEqual(response.getheader("X-Frame-Options"), "DENY")

    def test_only_get_and_head(self) -> None:
        for method in ("POST", "PUT", "DELETE", "PATCH"):
            response, _ = self.get("/chats", method=method)
            self.assertEqual((method, response.status, response.getheader("Allow")), (method, 405, "GET, HEAD"))
        response, body = self.get("/chats", method="HEAD")
        self.assertEqual((response.status, body), (200, ""))

    def test_foreign_host_header_is_refused_on_loopback(self) -> None:
        self.assertEqual(self.get("/chats", host="evil.example")[0].status, 403)
        self.assertEqual(self.get("/chats", host="localhost:8765")[0].status, 200)

    def test_a_viewer_error_gives_a_500_not_a_crash(self) -> None:
        with mock.patch.object(web.Viewer, "chats", side_effect=RuntimeError("x")), self.assertLogs("autopilot.web", "ERROR"):
            self.assertEqual(self.get("/chats")[0].status, 500)
        self.assertEqual(self.get("/chats")[0].status, 200)


class SettingsAndStartTests(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.home = Path(tmp.name)

    def load(self, **env: str):
        return load_settings({"AUTOPILOT_TIMEZONE": "UTC", **env}, home=self.home, read_env_file=False)

    def test_defaults_are_off_and_loopback(self) -> None:
        settings = self.load()
        self.assertEqual((settings.web_token, settings.web_host, settings.web_port, settings.claude_projects_dir),
                         (None, "127.0.0.1", 8765, None))

    def test_short_token_and_bad_port_are_config_errors(self) -> None:
        with self.assertRaises(ConfigError):
            self.load(AUTOPILOT_WEB_TOKEN="short")
        for port in ("0", "70000", "abc"):
            with self.subTest(port=port), self.assertRaises(ConfigError):
                self.load(AUTOPILOT_WEB_PORT=port)

    def test_token_with_unsafe_characters_is_a_config_error(self) -> None:
        for token in ("abc def ghi jkl mno pqr", "abc;def-ghi-jkl-mno-pqr", 'a"' * 15, "\u0160" * 25, "t" * 24 + " "):
            with self.subTest(token=token), self.assertRaises(ConfigError):
                self.load(AUTOPILOT_WEB_TOKEN=token)

    def test_values_are_read(self) -> None:
        settings = self.load(AUTOPILOT_WEB_TOKEN=TOKEN, AUTOPILOT_WEB_HOST="0.0.0.0", AUTOPILOT_WEB_PORT="9000",
                             AUTOPILOT_CLAUDE_PROJECTS_DIR="chats")
        self.assertEqual((settings.web_token, settings.web_host, settings.web_port), (TOKEN, "0.0.0.0", 9000))
        self.assertEqual(settings.claude_projects_dir, self.home.resolve() / "chats")

    def test_start_web_is_off_without_a_token(self) -> None:
        self.assertIsNone(web.start_web(self.load(), Store(self.home / "d.db")))

    def test_start_web_serves_and_survives_a_busy_port(self) -> None:
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        settings = self.load(AUTOPILOT_WEB_TOKEN=TOKEN, AUTOPILOT_WEB_PORT=str(port))
        store = Store(self.home / "d.db")
        server = web.start_web(settings, store)
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
        conn.request("GET", "/", headers={"Cookie": f"{web.COOKIE_NAME}={COOKIE}"})
        self.assertEqual(conn.getresponse().status, 200)
        with self.assertLogs("autopilot.web", "ERROR"):
            self.assertIsNone(web.start_web(settings, store))


if __name__ == "__main__":
    unittest.main()
