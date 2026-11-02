"""Thin Telegram Bot API client built on urllib.

Responsibility: turn six Bot API methods (getMe, getUpdates, sendMessage,
sendDocument, setMyCommands, sendChatAction) into plain Python calls and turn
every failure into a ``TelegramError`` whose message is safe to log and show.

Where it fits: ``bot.py`` (TelegramNotifier, TelegramBot), ``app.py``
(telegram-setup) and ``doctor.py`` create a ``TelegramClient`` and call it.
It depends only on ``errors.py`` and the standard library.

Invariants:
    * The bot token never appears in an error message, a log line or a
      chained exception: the request URL (which holds the token) is never
      logged and underlying network errors are not chained onto ours.
    * Every request has a timeout and uses TLS verification (urllib default).
    * ``urlopen`` is injected, so tests never touch the network.
"""

from __future__ import annotations

import http.client
import json
import logging
import re
import urllib.error
import urllib.request
import uuid
from collections.abc import Callable, Sequence
from typing import Any

from autopilot.errors import ConfigError, TelegramError

log = logging.getLogger(__name__)

API_BASE = "https://api.telegram.org"
DEFAULT_TIMEOUT = 30
LONG_POLL_SECONDS = 50
# Extra seconds the HTTP request may take beyond Telegram's own long-poll wait.
LONG_POLL_MARGIN_SECONDS = 15
# A bot token looks like "123456789:AA..." (bot id, colon, secret of 30+ URL-safe characters).
TOKEN_RE = re.compile(r"^\d{5,}:[A-Za-z0-9_-]{30,}$")

JSON_CONTENT_TYPE = "application/json"
REDACTED = "***"
PARSE_ERROR_FRAGMENT = "can't parse entities"
HTTP_BAD_REQUEST = 400


class TelegramClient:
    """Calls the Telegram Bot API with JSON (and multipart for documents).

    Example:
        >>> client = TelegramClient("123456789:" + "A" * 35)
        >>> client.send_message(42, "<b>Hello</b>")  # doctest: +SKIP
    """

    def __init__(
        self,
        token: str,
        *,
        timeout: float = DEFAULT_TIMEOUT,
        urlopen: Callable[..., Any] = urllib.request.urlopen,
    ) -> None:
        """Validate the token and store the client settings.

        Args:
            token: Bot token from @BotFather, in ``<id>:<secret>`` form.
            timeout: Seconds before an ordinary request is abandoned.
            urlopen: Function with the signature of ``urllib.request.urlopen``;
                replaced by a fake in tests.

        Raises:
            ConfigError: If the token does not look like a bot token. The
                message never contains the token itself.
        """
        if not TOKEN_RE.fullmatch(token):
            raise ConfigError(
                "TELEGRAM_BOT_TOKEN does not look like a bot token. "
                "Expected the form 123456789:AAExample... as given by @BotFather."
            )
        self._token = token
        self._timeout = timeout
        self._urlopen = urlopen

    def get_me(self) -> dict[str, Any]:
        """Fetch the bot's own identity (also a cheap token check).

        Returns:
            The Bot API ``User`` object (id, is_bot, username, ...).

        Raises:
            TelegramError: If the call fails, e.g. status 401 for a revoked token.
        """
        return self._call_json("getMe", {})

    def get_updates(
        self, offset: int | None = None, timeout: int = LONG_POLL_SECONDS
    ) -> list[dict[str, Any]]:
        """Long-poll for new message updates.

        Args:
            offset: First update id to return; pass ``last_update_id + 1`` to
                acknowledge earlier updates. None returns all pending updates.
            timeout: Seconds Telegram may hold the request open when nothing
                is pending. The HTTP timeout is this plus a safety margin.

        Returns:
            A list of ``Update`` objects, possibly empty.

        Raises:
            TelegramError: On any failure, including status 409 when another
                process is polling the same bot.
        """
        payload: dict[str, Any] = {"timeout": timeout, "allowed_updates": ["message"]}
        if offset is not None:
            payload["offset"] = offset
        http_timeout = timeout + LONG_POLL_MARGIN_SECONDS
        return self._call_json("getUpdates", payload, http_timeout=http_timeout)

    def send_message(
        self,
        chat_id: int,
        text: str,
        *,
        parse_mode: str | None = "HTML",
        reply_markup: dict[str, Any] | None = None,
        disable_notification: bool = False,
    ) -> dict[str, Any]:
        """Send a text message with link previews switched off.

        Args:
            chat_id: Destination chat id (negative for groups).
            text: Message text, at most 4096 characters.
            parse_mode: "HTML", or None to send plain text.
            reply_markup: Optional inline keyboard or similar markup.
            disable_notification: Deliver silently when True.

        Returns:
            The sent ``Message`` object.

        Raises:
            TelegramError: If Telegram rejects the message or the call fails.
                Use ``is_parse_error`` to detect an HTML parse failure.
        """
        payload: dict[str, Any] = {
            "chat_id": chat_id,
            "text": text,
            "link_preview_options": {"is_disabled": True},
            "disable_notification": disable_notification,
        }
        if parse_mode is not None:
            payload["parse_mode"] = parse_mode
        if reply_markup is not None:
            payload["reply_markup"] = reply_markup
        return self._call_json("sendMessage", payload)

    def send_document(
        self,
        chat_id: int,
        filename: str,
        content: bytes,
        *,
        caption: str | None = None,
        parse_mode: str | None = "HTML",
        reply_markup: dict[str, Any] | None = None,
        disable_notification: bool = False,
    ) -> dict[str, Any]:
        """Upload a file as a document using multipart/form-data.

        Args:
            chat_id: Destination chat id.
            filename: Name the recipient sees, e.g. ``news-run4-20261102-0730.md``.
            content: The file bytes.
            caption: Optional caption, at most 1024 characters.
            parse_mode: Caption parse mode; ignored when there is no caption.
            reply_markup: Optional inline keyboard or similar markup.
            disable_notification: Deliver silently when True.

        Returns:
            The sent ``Message`` object.

        Raises:
            TelegramError: If Telegram rejects the upload or the call fails.
        """
        fields: dict[str, str] = {
            "chat_id": str(chat_id),
            "disable_notification": "true" if disable_notification else "false",
        }
        if caption is not None:
            fields["caption"] = caption
            if parse_mode is not None:
                fields["parse_mode"] = parse_mode
        if reply_markup is not None:
            fields["reply_markup"] = json.dumps(reply_markup)
        boundary = uuid.uuid4().hex
        body = _encode_multipart(boundary, fields, "document", filename, content)
        content_type = f"multipart/form-data; boundary={boundary}"
        return self._call("sendDocument", body, content_type, self._timeout)

    def set_my_commands(self, commands: Sequence[tuple[str, str]]) -> None:
        """Publish the bot's command menu.

        Args:
            commands: ``(command, description)`` pairs. Commands are lower-case
                without the leading slash.

        Raises:
            TelegramError: If Telegram rejects the list or the call fails.
        """
        payload = {
            "commands": [
                {"command": command, "description": description}
                for command, description in commands
            ]
        }
        self._call_json("setMyCommands", payload)

    def send_chat_action(self, chat_id: int, action: str = "typing") -> None:
        """Show a status such as "typing" in the chat for a few seconds.

        Args:
            chat_id: The chat that should show the action.
            action: A Bot API chat action name, e.g. "typing".

        Raises:
            TelegramError: If the call fails.
        """
        self._call_json("sendChatAction", {"chat_id": chat_id, "action": action})

    def _call_json(
        self, method: str, payload: dict[str, Any], *, http_timeout: float | None = None
    ) -> Any:
        """POST a JSON payload to a Bot API method.

        Args:
            method: Bot API method name, e.g. "sendMessage".
            payload: JSON-serialisable request parameters.
            http_timeout: Overrides the client's default request timeout.

        Returns:
            The ``result`` field of the API response.

        Raises:
            TelegramError: On any failure; see ``_call``.
        """
        body = json.dumps(payload).encode("utf-8")
        timeout = self._timeout if http_timeout is None else http_timeout
        return self._call(method, body, JSON_CONTENT_TYPE, timeout)

    def _call(self, method: str, body: bytes, content_type: str, timeout: float) -> Any:
        """Send one request and return the decoded ``result``.

        Every failure mode is converted here so callers only ever see
        ``TelegramError`` and no message or log line carries the token.

        Args:
            method: Bot API method name.
            body: Encoded request body.
            content_type: Value of the Content-Type header.
            timeout: Seconds before the request is abandoned.

        Returns:
            The ``result`` field of a successful response.

        Raises:
            TelegramError: With ``status`` and ``retry_after`` set when
                Telegram answered with an error, or a "network error" message
                for connection problems, timeouts and unreadable responses.
        """
        # The URL embeds the token, so it is built here and never logged.
        request = urllib.request.Request(
            f"{API_BASE}/bot{self._token}/{method}",
            data=body,
            headers={"Content-Type": content_type},
            method="POST",
        )
        log.debug("Telegram %s request (%d bytes)", method, len(body))
        try:
            with self._urlopen(request, timeout=timeout) as response:
                raw = response.read()
        except urllib.error.HTTPError as err:
            # Telegram explains rejections in a JSON body, so read it before giving up.
            raise self._error_from_http(method, err) from None
        except (OSError, http.client.HTTPException) as err:
            # URLError, timeouts, resets and truncated responses all land here.
            raise self._network_error(method, _reason_of(err)) from None
        return self._result_of(method, raw)

    def _result_of(self, method: str, raw: bytes) -> Any:
        """Decode a response body and return its ``result``.

        Args:
            method: Bot API method name, for error messages.
            raw: The response body bytes.

        Returns:
            The ``result`` field.

        Raises:
            TelegramError: If the body is not a JSON object, or has ``ok`` false.
        """
        try:
            data = json.loads(raw.decode("utf-8"))
        except ValueError:
            # JSONDecodeError and UnicodeDecodeError are both ValueError subclasses.
            raise self._network_error(method, "invalid JSON in response") from None
        if not isinstance(data, dict):
            raise self._network_error(method, "unexpected response from Telegram")
        if not data.get("ok"):
            # Defensive: Telegram normally pairs ok=false with an HTTP error status.
            raise self._error_from_body(method, data, status=data.get("error_code"))
        return data.get("result")

    def _error_from_http(self, method: str, err: urllib.error.HTTPError) -> TelegramError:
        """Build a TelegramError from an HTTP error response.

        Args:
            method: Bot API method name.
            err: The HTTPError raised by urlopen; its body is consumed and closed.

        Returns:
            A TelegramError carrying the HTTP status and any ``retry_after``.
        """
        try:
            raw = err.read()
        except (OSError, http.client.HTTPException):
            raw = b""
        finally:
            err.close()
        try:
            data = json.loads(raw.decode("utf-8"))
        except ValueError:
            data = None
        if not isinstance(data, dict):
            # No usable body: fall back to the HTTP reason phrase.
            data = {"description": f"HTTP {err.code} {err.reason}"}
        return self._error_from_body(method, data, status=err.code)

    def _error_from_body(
        self, method: str, data: dict[str, Any], *, status: Any
    ) -> TelegramError:
        """Build a TelegramError from a decoded error response.

        Args:
            method: Bot API method name.
            data: Decoded response with ``description`` and ``parameters``.
            status: HTTP status or Bot API ``error_code``; non-integers become None.

        Returns:
            The error, with the token scrubbed from its text.
        """
        description = self._scrub(str(data.get("description") or "unknown error"))
        parameters = data.get("parameters")
        retry_after = parameters.get("retry_after") if isinstance(parameters, dict) else None
        return TelegramError(
            f"Telegram {method} failed: {description}",
            status=status if isinstance(status, int) else None,
            retry_after=retry_after if isinstance(retry_after, int) else None,
        )

    def _network_error(self, method: str, reason: str) -> TelegramError:
        """Build the error for a failure that produced no Telegram answer.

        Args:
            method: Bot API method name.
            reason: Short cause such as "timed out".

        Returns:
            A TelegramError with no status.
        """
        return TelegramError(
            f"Telegram {method} failed: network error ({self._scrub(reason)})"
        )

    def _scrub(self, text: str) -> str:
        """Replace the bot token in ``text`` with asterisks.

        A second line of defence: Telegram and urllib should never echo the
        token, but if they do it must not reach a message or log.

        Args:
            text: Text that may contain the token.

        Returns:
            The text with every token occurrence replaced.
        """
        return text.replace(self._token, REDACTED)


def _reason_of(err: Exception) -> str:
    """Return a short human-readable cause for a network exception.

    Args:
        err: The caught exception. For ``URLError`` the wrapped ``reason`` is used.

    Returns:
        A one-line description, never empty.
    """
    reason = getattr(err, "reason", None) or err
    return str(reason) or type(err).__name__


def _encode_multipart(
    boundary: str,
    fields: dict[str, str],
    file_field: str,
    filename: str,
    content: bytes,
) -> bytes:
    """Encode text fields plus one file as a multipart/form-data body.

    Args:
        boundary: Unique separator; must not occur inside ``content``.
        fields: Text form fields, written first in insertion order.
        file_field: Form field name of the file part.
        filename: File name sent with the part.
        content: File bytes.

    Returns:
        The complete request body.
    """
    # Quotes and line breaks would break out of the quoted header parameter.
    safe_name = filename.replace('"', "%22").replace("\r", "").replace("\n", "")
    parts: list[bytes] = []
    for name, value in fields.items():
        parts.append(
            (
                f"--{boundary}\r\n"
                f'Content-Disposition: form-data; name="{name}"\r\n\r\n'
                f"{value}\r\n"
            ).encode("utf-8")
        )
    parts.append(
        (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="{file_field}"; filename="{safe_name}"\r\n'
            "Content-Type: application/octet-stream\r\n\r\n"
        ).encode("utf-8")
    )
    parts.append(content)
    parts.append(f"\r\n--{boundary}--\r\n".encode("utf-8"))
    return b"".join(parts)


def is_parse_error(err: TelegramError) -> bool:
    """Tell whether Telegram rejected a message because its HTML was invalid.

    Args:
        err: An error raised by ``TelegramClient.send_message``.

    Returns:
        True for status 400 with "can't parse entities" in the message
        (case-insensitive); the caller may then resend the text as plain text.

    Example:
        >>> is_parse_error(TelegramError("Bad Request: Can't parse entities", status=400))
        True
    """
    return err.status == HTTP_BAD_REQUEST and PARSE_ERROR_FRAGMENT in str(err).lower()
