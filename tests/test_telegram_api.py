"""Unit tests for autopilot.telegram_api.

Covers: token validation, JSON success paths for every method, the request
shape (URL, headers, payload, timeouts), HTTP errors with JSON bodies, network
failures, invalid JSON, the multipart body of send_document, parse-error
detection and the guarantee that the bot token never leaks into an error
message or a log line.

No test touches the network: ``urlopen`` is replaced by ``FakeUrlopen``.
"""

from __future__ import annotations

import io
import json
import socket
import unittest
import urllib.error
import urllib.request
from typing import Any

from autopilot.errors import ConfigError, TelegramError
from autopilot.telegram_api import (
    API_BASE,
    DEFAULT_TIMEOUT,
    LONG_POLL_SECONDS,
    TelegramClient,
    is_parse_error,
)

TOKEN = "123456789:AAExampleTokenWithEnoughCharacters_-1234"


class FakeResponse(io.BytesIO):
    """A urlopen response: readable bytes that work as a context manager."""


class FakeUrlopen:
    """Replacement for urllib.request.urlopen that replays queued outcomes.

    Attributes:
        requests: Every Request received, in order.
        timeouts: The timeout argument of every call.
        outcomes: Queue of bytes (a response body) or exceptions to raise.
    """

    def __init__(self, *outcomes: bytes | BaseException) -> None:
        """Queue the outcomes to replay, one per call.

        Args:
            *outcomes: Response bodies, or exceptions that urlopen should raise.
        """
        self.requests: list[urllib.request.Request] = []
        self.timeouts: list[float] = []
        self.outcomes = list(outcomes)

    def __call__(self, request: urllib.request.Request, timeout: float) -> FakeResponse:
        """Record the call and return or raise the next queued outcome."""
        self.requests.append(request)
        self.timeouts.append(timeout)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return FakeResponse(outcome)

    @property
    def last_payload(self) -> dict[str, Any]:
        """The JSON body of the most recent request."""
        return json.loads(self.requests[-1].data.decode("utf-8"))


def ok_body(result: Any) -> bytes:
    """Encode a successful Bot API response around ``result``."""
    return json.dumps({"ok": True, "result": result}).encode("utf-8")


def http_error(code: int, body: bytes, reason: str = "Bad Request") -> urllib.error.HTTPError:
    """Build the HTTPError urllib raises, with the token-bearing URL like the real one."""
    url = f"{API_BASE}/bot{TOKEN}/sendMessage"
    return urllib.error.HTTPError(url, code, reason, {}, io.BytesIO(body))  # type: ignore[arg-type]


def make_client(*outcomes: bytes | BaseException) -> tuple[TelegramClient, FakeUrlopen]:
    """Build a client wired to a FakeUrlopen with the given outcomes."""
    fake = FakeUrlopen(*outcomes)
    return TelegramClient(TOKEN, urlopen=fake), fake


class ConstructorTests(unittest.TestCase):
    """The client refuses tokens that cannot be real, without echoing them."""

    def test_valid_token_is_accepted(self) -> None:
        """A token in the BotFather format constructs a client."""
        TelegramClient(TOKEN, urlopen=FakeUrlopen())

    def test_bad_tokens_raise_config_error(self) -> None:
        """Empty, colon-less, short-secret and spaced tokens are all rejected."""
        for bad in ("", "nocolon", "123456789:short", f" {TOKEN}", "abc:" + "A" * 40):
            with self.subTest(token=bad), self.assertRaises(ConfigError):
                TelegramClient(bad, urlopen=FakeUrlopen())

    def test_config_error_does_not_contain_the_token(self) -> None:
        """The rejection message never repeats the rejected value."""
        bad = "123456789:short"
        with self.assertRaises(ConfigError) as caught:
            TelegramClient(bad, urlopen=FakeUrlopen())
        self.assertNotIn(bad, str(caught.exception))


class SuccessTests(unittest.TestCase):
    """Each method posts the right JSON and returns the result field."""

    def test_get_me_returns_result_and_posts_to_the_token_url(self) -> None:
        """getMe posts JSON to /bot<token>/getMe with the default timeout."""
        client, fake = make_client(ok_body({"id": 1, "username": "bot"}))
        self.assertEqual(client.get_me(), {"id": 1, "username": "bot"})
        request = fake.requests[0]
        self.assertEqual(request.full_url, f"{API_BASE}/bot{TOKEN}/getMe")
        self.assertEqual(request.get_method(), "POST")
        self.assertEqual(request.get_header("Content-type"), "application/json")
        self.assertEqual(fake.timeouts, [DEFAULT_TIMEOUT])

    def test_get_updates_sends_offset_and_widens_the_http_timeout(self) -> None:
        """getUpdates asks for message updates only and waits timeout + 15 s."""
        client, fake = make_client(ok_body([{"update_id": 7}]))
        self.assertEqual(client.get_updates(offset=7, timeout=20), [{"update_id": 7}])
        self.assertEqual(
            fake.last_payload,
            {"timeout": 20, "allowed_updates": ["message"], "offset": 7},
        )
        self.assertEqual(fake.timeouts, [35])

    def test_get_updates_defaults_omit_offset(self) -> None:
        """Without an offset the payload has none, and the long-poll default applies."""
        client, fake = make_client(ok_body([]))
        self.assertEqual(client.get_updates(), [])
        self.assertNotIn("offset", fake.last_payload)
        self.assertEqual(fake.last_payload["timeout"], LONG_POLL_SECONDS)
        self.assertEqual(fake.timeouts, [LONG_POLL_SECONDS + 15])

    def test_send_message_payload_defaults(self) -> None:
        """send_message uses HTML, turns link previews off and returns the message."""
        client, fake = make_client(ok_body({"message_id": 5}))
        result = client.send_message(-42, "<b>Příliš</b>")
        self.assertEqual(result, {"message_id": 5})
        self.assertEqual(
            fake.last_payload,
            {
                "chat_id": -42,
                "text": "<b>Příliš</b>",
                "parse_mode": "HTML",
                "link_preview_options": {"is_disabled": True},
                "disable_notification": False,
            },
        )

    def test_send_message_plain_with_markup_and_silent(self) -> None:
        """parse_mode None is omitted; markup and silent delivery are forwarded."""
        client, fake = make_client(ok_body({}))
        markup = {"inline_keyboard": [[{"text": "Go", "callback_data": "go"}]]}
        client.send_message(1, "hi", parse_mode=None, reply_markup=markup, disable_notification=True)
        payload = fake.last_payload
        self.assertNotIn("parse_mode", payload)
        self.assertEqual(payload["reply_markup"], markup)
        self.assertTrue(payload["disable_notification"])

    def test_set_my_commands_payload(self) -> None:
        """Commands become a list of command/description objects."""
        client, fake = make_client(ok_body(True))
        self.assertIsNone(client.set_my_commands([("help", "How to use"), ("tasks", "List")]))
        self.assertEqual(
            fake.last_payload,
            {
                "commands": [
                    {"command": "help", "description": "How to use"},
                    {"command": "tasks", "description": "List"},
                ]
            },
        )
        self.assertTrue(fake.requests[0].full_url.endswith("/setMyCommands"))

    def test_send_chat_action_payload(self) -> None:
        """The action defaults to typing."""
        client, fake = make_client(ok_body(True))
        self.assertIsNone(client.send_chat_action(9))
        self.assertEqual(fake.last_payload, {"chat_id": 9, "action": "typing"})

    def test_custom_timeout_is_used(self) -> None:
        """The constructor timeout applies to ordinary calls."""
        fake = FakeUrlopen(ok_body({}))
        TelegramClient(TOKEN, timeout=3, urlopen=fake).get_me()
        self.assertEqual(fake.timeouts, [3])


class DocumentTests(unittest.TestCase):
    """send_document builds a correct multipart/form-data upload."""

    def test_multipart_body_has_fields_and_file(self) -> None:
        """Fields, caption and the raw file bytes appear in the multipart body."""
        client, fake = make_client(ok_body({"message_id": 8}))
        content = "# Výsledek\nPříliš žluťoučký kůň".encode("utf-8") + b"\x00\xff"
        result = client.send_document(
            77, "news-run1.md", content, caption="<b>Full</b>", disable_notification=True
        )
        self.assertEqual(result, {"message_id": 8})
        request = fake.requests[0]
        self.assertTrue(request.full_url.endswith("/sendDocument"))
        content_type = request.get_header("Content-type")
        self.assertTrue(content_type.startswith("multipart/form-data; boundary="))
        boundary = content_type.split("boundary=", 1)[1]
        body = request.data
        self.assertTrue(body.startswith(f"--{boundary}\r\n".encode()))
        self.assertTrue(body.endswith(f"\r\n--{boundary}--\r\n".encode()))
        for expected in (
            b'name="chat_id"\r\n\r\n77\r\n',
            b'name="caption"\r\n\r\n<b>Full</b>\r\n',
            b'name="parse_mode"\r\n\r\nHTML\r\n',
            b'name="disable_notification"\r\n\r\ntrue\r\n',
            b'name="document"; filename="news-run1.md"\r\n',
            b"Content-Type: application/octet-stream\r\n\r\n" + content + b"\r\n",
        ):
            self.assertIn(expected, body)

    def test_optional_fields_are_left_out(self) -> None:
        """Without caption or markup, no caption, parse_mode or reply_markup part is sent."""
        client, fake = make_client(ok_body({}))
        client.send_document(1, "a.md", b"x")
        body = fake.requests[0].data
        for absent in (b'name="caption"', b'name="parse_mode"', b'name="reply_markup"'):
            self.assertNotIn(absent, body)
        self.assertIn(b'name="disable_notification"\r\n\r\nfalse', body)

    def test_reply_markup_is_sent_as_json_text(self) -> None:
        """Markup travels as a JSON string in its own form part."""
        client, fake = make_client(ok_body({}))
        client.send_document(1, "a.md", b"x", reply_markup={"k": 1})
        self.assertIn(b'name="reply_markup"\r\n\r\n{"k": 1}\r\n', fake.requests[0].data)

    def test_filename_cannot_break_out_of_the_header(self) -> None:
        """Quotes and line breaks in a file name are neutralised."""
        client, fake = make_client(ok_body({}))
        client.send_document(1, 'a"\r\nX-Evil: 1.md', b"x")
        body = fake.requests[0].data
        self.assertIn(b'filename="a%22X-Evil: 1.md"', body)
        self.assertNotIn(b'\r\nX-Evil', body)

    def test_each_upload_uses_a_fresh_boundary(self) -> None:
        """Two uploads never share a boundary."""
        client, fake = make_client(ok_body({}), ok_body({}))
        client.send_document(1, "a.md", b"x")
        client.send_document(1, "a.md", b"x")
        boundaries = {r.get_header("Content-type") for r in fake.requests}
        self.assertEqual(len(boundaries), 2)


class ErrorTests(unittest.TestCase):
    """Every failure becomes a TelegramError with a useful, token-free message."""

    def test_http_error_with_json_body(self) -> None:
        """Description, status and retry_after come from the error body."""
        body = json.dumps(
            {
                "ok": False,
                "error_code": 429,
                "description": "Too Many Requests: retry after 12",
                "parameters": {"retry_after": 12},
            }
        ).encode()
        client, _ = make_client(http_error(429, body, "Too Many Requests"))
        with self.assertRaises(TelegramError) as caught:
            client.send_message(1, "x")
        err = caught.exception
        self.assertEqual(str(err), "Telegram sendMessage failed: Too Many Requests: retry after 12")
        self.assertEqual(err.status, 429)
        self.assertEqual(err.retry_after, 12)

    def test_http_error_without_retry_after(self) -> None:
        """A plain 400 has a status and no retry_after."""
        body = json.dumps({"ok": False, "description": "Bad Request: chat not found"}).encode()
        client, _ = make_client(http_error(400, body))
        with self.assertRaises(TelegramError) as caught:
            client.get_me()
        self.assertEqual(caught.exception.status, 400)
        self.assertIsNone(caught.exception.retry_after)
        self.assertIn("chat not found", str(caught.exception))

    def test_http_error_with_non_json_body_uses_the_reason(self) -> None:
        """An HTML error page still yields a status and the HTTP reason."""
        client, _ = make_client(http_error(502, b"<html>Bad gateway</html>", "Bad Gateway"))
        with self.assertRaises(TelegramError) as caught:
            client.get_me()
        self.assertEqual(caught.exception.status, 502)
        self.assertEqual(str(caught.exception), "Telegram getMe failed: HTTP 502 Bad Gateway")

    def test_http_error_with_json_that_is_not_an_object(self) -> None:
        """A JSON list body is treated like an unusable body."""
        client, _ = make_client(http_error(500, b"[1]", "Server Error"))
        with self.assertRaises(TelegramError) as caught:
            client.get_me()
        self.assertEqual(caught.exception.status, 500)

    def test_url_error_is_a_network_error(self) -> None:
        """A DNS failure reads as a network error with the reason."""
        client, _ = make_client(urllib.error.URLError(socket.gaierror("Name or service not known")))
        with self.assertRaises(TelegramError) as caught:
            client.get_me()
        self.assertIsNone(caught.exception.status)
        self.assertIn("Telegram getMe failed: network error (", str(caught.exception))
        self.assertIn("Name or service not known", str(caught.exception))

    def test_timeout_is_a_network_error(self) -> None:
        """A socket timeout during the read becomes a network error."""
        client, _ = make_client(TimeoutError("timed out"))
        with self.assertRaises(TelegramError) as caught:
            client.get_updates()
        self.assertEqual(
            str(caught.exception), "Telegram getUpdates failed: network error (timed out)"
        )

    def test_connection_reset_is_a_network_error(self) -> None:
        """Other OSError subclasses are handled the same way."""
        client, _ = make_client(ConnectionResetError("reset by peer"))
        with self.assertRaises(TelegramError):
            client.get_me()

    def test_invalid_json_is_a_network_error(self) -> None:
        """A 200 response that is not JSON is reported, not crashed on."""
        client, _ = make_client(b"<html>captive portal</html>")
        with self.assertRaises(TelegramError) as caught:
            client.get_me()
        self.assertEqual(
            str(caught.exception), "Telegram getMe failed: network error (invalid JSON in response)"
        )

    def test_undecodable_bytes_are_a_network_error(self) -> None:
        """Bytes that are not UTF-8 are reported like invalid JSON."""
        client, _ = make_client(b"\xff\xfe")
        with self.assertRaises(TelegramError):
            client.get_me()

    def test_json_that_is_not_an_object_is_a_network_error(self) -> None:
        """A JSON array response is rejected."""
        client, _ = make_client(b"[]")
        with self.assertRaises(TelegramError):
            client.get_me()

    def test_ok_false_with_http_200_still_raises(self) -> None:
        """ok=false is an error even if the HTTP status was 200."""
        body = json.dumps({"ok": False, "error_code": 403, "description": "Forbidden: blocked"})
        client, _ = make_client(body.encode())
        with self.assertRaises(TelegramError) as caught:
            client.get_me()
        self.assertEqual(caught.exception.status, 403)
        self.assertIn("Forbidden: blocked", str(caught.exception))


class TokenSecrecyTests(unittest.TestCase):
    """The bot token must never reach an error message, a log line or a traceback chain."""

    def assert_token_free(self, err: BaseException) -> None:
        """Check the message and every exception a traceback would print for the token."""
        link: BaseException | None = err
        while link is not None:
            self.assertNotIn(TOKEN, str(link))
            self.assertNotIn(TOKEN, repr(link))
            # `raise ... from None` keeps __context__ but tracebacks do not print it.
            hidden = link.__suppress_context__ and link.__cause__ is None
            link = link.__cause__ or (None if hidden else link.__context__)

    def test_no_error_path_leaks_the_token(self) -> None:
        """HTTP errors (whose URL holds the token), URL errors and bad bodies stay clean."""
        leaky_body = json.dumps({"ok": False, "description": f"bad token {TOKEN}"}).encode()
        outcomes: list[bytes | BaseException] = [
            http_error(401, b'{"ok": false, "description": "Unauthorized"}', "Unauthorized"),
            http_error(400, leaky_body),
            http_error(404, b"not json", "Not Found"),
            urllib.error.URLError(f"cannot reach {API_BASE}/bot{TOKEN}/getMe"),
            b"garbage",
        ]
        for outcome in outcomes:
            client, _ = make_client(outcome)
            with self.subTest(outcome=repr(outcome)[:40]), self.assertRaises(TelegramError) as caught:
                client.get_me()
            self.assert_token_free(caught.exception)

    def test_leaked_token_in_description_is_masked(self) -> None:
        """If Telegram ever echoed the token, it is replaced by asterisks."""
        body = json.dumps({"ok": False, "description": f"bad {TOKEN}"}).encode()
        client, _ = make_client(http_error(400, body))
        with self.assertRaises(TelegramError) as caught:
            client.get_me()
        self.assertEqual(str(caught.exception), "Telegram getMe failed: bad ***")

    def test_no_log_line_contains_the_token(self) -> None:
        """Successful and failing calls log nothing that includes the token or the URL."""
        client, _ = make_client(
            ok_body({}), http_error(400, b'{"description": "Bad Request"}'), urllib.error.URLError("x")
        )
        with self.assertLogs("autopilot.telegram_api", level="DEBUG") as logs:
            client.get_me()
            for _ in range(2):
                with self.assertRaises(TelegramError):
                    client.get_me()
        for line in logs.output:
            self.assertNotIn(TOKEN, line)
            self.assertNotIn("/bot", line)


class ParseErrorTests(unittest.TestCase):
    """is_parse_error recognises Telegram's HTML rejection and nothing else."""

    def test_matches_status_400_with_the_phrase_in_any_case(self) -> None:
        """The check is case-insensitive."""
        for text in ("Bad Request: can't parse entities: x", "Bad Request: Can't Parse Entities"):
            with self.subTest(text=text):
                self.assertTrue(is_parse_error(TelegramError(text, status=400)))

    def test_rejects_other_statuses_and_messages(self) -> None:
        """A different status, a different 400, or no status is not a parse error."""
        self.assertFalse(is_parse_error(TelegramError("can't parse entities", status=500)))
        self.assertFalse(is_parse_error(TelegramError("Bad Request: chat not found", status=400)))
        self.assertFalse(is_parse_error(TelegramError("can't parse entities")))


if __name__ == "__main__":
    unittest.main()
