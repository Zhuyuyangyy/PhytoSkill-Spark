"""Stdlib HTTP transport for one OpenAI-compatible chat completions endpoint.

Design notes that matter for the experiment:

* ``model_returned`` is read from every response. It is the only evidence of who
  actually served the request, and it is what exposes a gateway that load
  balances across several model snapshots.
* The credential is never included in an exception message, and response bodies
  are scrubbed before being quoted in an error.
* The poster is injectable so the whole suite can run offline against a fake.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Protocol

from harness.config import HarnessConfig
from harness.errors import AuthError, TransportError

RETRYABLE_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504})
BACKOFF_CAP_SECONDS = 60.0
MAX_ERROR_BODY_CHARS = 400
USER_AGENT = "phytoagent-skills-harness/0.2"


class Poster(Protocol):
    """Sends one request. Returns ``(http_status, body_bytes)``."""

    def __call__(self, url: str, body: bytes, headers: dict[str, str],
                 timeout: float) -> tuple[int, bytes]: ...


def urllib_poster(proxy: str | None = None) -> Poster:
    """Build a poster backed by ``urllib``. Uses environment proxies when unset."""
    handlers = [urllib.request.ProxyHandler({"http": proxy, "https": proxy})] if proxy else []

    def post(url: str, body: bytes, headers: dict[str, str], timeout: float) -> tuple[int, bytes]:
        request = urllib.request.Request(url, data=body, headers=headers, method="POST")
        opener = urllib.request.build_opener(*handlers)
        try:
            with opener.open(request, timeout=timeout) as response:
                return response.status, response.read()
        except urllib.error.HTTPError as error:
            return error.code, error.read()
        except urllib.error.URLError as error:
            raise TransportError(f"{type(error).__name__}: {error.reason}") from error
        except TimeoutError as error:
            raise TransportError(f"Request timed out after {timeout}s") from error

    return post


@dataclass
class Completion:
    http_status: int
    model_returned: str | None
    message: dict
    finish_reason: str | None
    usage: dict
    latency_ms: float
    attempts: int
    response_id: str | None
    request_model: str

    @property
    def usage_flat(self) -> dict:
        """Token counters, including reasoning tokens when the provider reports them."""
        details = self.usage.get("completion_tokens_details") or {}
        return {
            "prompt_tokens": self.usage.get("prompt_tokens"),
            "completion_tokens": self.usage.get("completion_tokens"),
            "total_tokens": self.usage.get("total_tokens"),
            "reasoning_tokens": details.get("reasoning_tokens"),
        }

    @property
    def tool_calls(self) -> list[dict]:
        calls = self.message.get("tool_calls") or []
        return [call for call in calls if isinstance(call, dict)]

    @property
    def reasoning_content(self) -> str | None:
        return self.message.get("reasoning_content")


class Transport:
    def __init__(self, config: HarnessConfig, poster: Poster | None = None, *,
                 sleep=time.sleep):
        self.config = config
        self.poster = poster if poster is not None else urllib_poster(config.proxy)
        self._sleep = sleep
        self.model_history: list[str | None] = []

    def _scrub(self, text: str) -> str:
        """Remove the credential before any text can reach a log or a report."""
        key = self.config.api_key
        if key and key in text:
            return text.replace(key, "***REDACTED***")
        return text

    def _decode(self, status: int, body: bytes, *, attempts: int, latency_ms: float,
                request_model: str) -> Completion:
        try:
            payload = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise TransportError(
                f"HTTP {status} returned a non-JSON body: "
                f"{self._scrub(body[:MAX_ERROR_BODY_CHARS].decode('utf-8', 'replace'))}"
            ) from exc
        if not isinstance(payload, dict):
            raise TransportError(f"HTTP {status} returned {type(payload).__name__}, expected an object")

        choices = payload.get("choices") or []
        choice = choices[0] if choices and isinstance(choices[0], dict) else {}
        message = choice.get("message") or {}
        if not isinstance(message, dict):
            message = {}
        returned = payload.get("model")
        self.model_history.append(returned)
        return Completion(
            http_status=status,
            model_returned=returned,
            message=message,
            finish_reason=choice.get("finish_reason"),
            usage=payload.get("usage") or {},
            latency_ms=latency_ms,
            attempts=attempts,
            response_id=payload.get("id"),
            request_model=request_model,
        )

    def chat(self, payload: dict, *, max_retries: int | None = None) -> Completion:
        """POST one completion, retrying only the statuses that can succeed later."""
        config = self.config
        retries = config.max_retries if max_retries is None else max_retries
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {config.require_key()}",
            "User-Agent": USER_AGENT,
        }
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        started = time.perf_counter()
        last_error = "no attempt was made"

        for attempt in range(1, retries + 2):
            try:
                status, raw = self.poster(config.endpoint, body, headers, config.timeout_seconds)
            except TransportError as exc:
                last_error = str(exc)
                if attempt > retries:
                    break
                self._sleep(self._backoff(attempt))
                continue

            if status < 400:
                return self._decode(status, raw, attempts=attempt,
                                    latency_ms=round((time.perf_counter() - started) * 1000, 3),
                                    request_model=config.model)

            detail = self._scrub(raw[:MAX_ERROR_BODY_CHARS].decode("utf-8", "replace"))
            if status in (401, 403):
                raise AuthError(
                    f"HTTP {status} from {config.host}: the credential was rejected. Body: {detail}"
                )
            if status not in RETRYABLE_STATUS:
                raise TransportError(f"HTTP {status} from {config.host}: {detail}")
            last_error = f"HTTP {status}: {detail}"
            if attempt > retries:
                break
            self._sleep(self._backoff(attempt))

        raise TransportError(
            f"{config.host} failed after {retries + 1} attempt(s). Last error: {last_error}"
        )

    def _backoff(self, attempt: int) -> float:
        """Exponential backoff. Linear backoff is not enough against per-minute limits."""
        return min(1.5 * (2 ** (attempt - 1)), BACKOFF_CAP_SECONDS)

    @property
    def distinct_models(self) -> list[str]:
        """Every distinct ``model`` value seen, in first-seen order. Two or more
        values means the endpoint is load balancing and cannot be used as a
        fixed experimental condition."""
        seen: list[str] = []
        for name in self.model_history:
            if name is not None and name not in seen:
                seen.append(name)
        return seen
