"""Configuration for the live StepFun Harness, read only from the environment.

Rules this module enforces:

* The credential never has a default value. A missing key is a hard error, not
  a silent fallback to fixture mode.
* ``repr()`` of a config never contains the credential.
* A git-ignored ``.env`` may supply values, but real environment variables always
  take precedence, so a shared machine can override without editing files.
* ``api_key_file`` is accepted as an alternative to an environment variable
  because environment variables are readable by other processes on a shared
  remote host.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from harness.errors import ConfigError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DOTENV = PROJECT_ROOT / ".env"

DEFAULT_BASE_URL = "https://api.stepfun.com/v1"
DEFAULT_MODEL = "step-3.7-flash"

# Step Plan is a **separate channel with its own Credit pool**. The published
# distinction matters in practice: an account whose pay-as-you-go channel returns
# HTTP 402 "quota_exceeded" can still work on this one, because the two channels'
# quotas are independent. Verified on 2026-09-21 — same key, /v1 -> 402,
# /step_plan/v1 -> 200. That is why this is selectable rather than a footnote.
STEP_PLAN_BASE_URL = "https://api.stepfun.com/step_plan/v1"

# Published list prices (CNY per 1M tokens), kept here so a run can report what it
# is about to cost before it spends anything. Source: StepFun 开放平台「定价与限速」.
# step-5-preview is ~5x the input and ~2.5x the output price of step-3.7-flash, and
# step-3.5-flash is cheaper again. For tool-selection experiments the extra
# reasoning capability buys nothing measurable, so the default is the cheap one.
MODEL_PRICES_PER_MILLION = {
    "step-5-preview": {"input": 7.0, "output": 20.0},
    "step-3.7-flash": {"input": 1.35, "output": 8.1},
    "step-3.5-flash": {"input": 0.7, "output": 2.1},
    "step-3.5-flash-2603": {"input": 0.7, "output": 2.1},
}

# Only these models are known to honour `tools`. Others may accept the parameter
# and silently ignore it, which would make an A/B about tool selection meaningless.
TOOL_CALLING_MODELS = ("step-3.7-flash", "step-3.5-flash", "step-3.5-flash-2603")

ENV_BASE_URL = "PHYTO_STEPFUN_BASE_URL"
ENV_MODEL = "PHYTO_STEPFUN_MODEL"
ENV_API_KEY = "PHYTO_STEPFUN_API_KEY"
ENV_API_KEY_FILE = "PHYTO_STEPFUN_API_KEY_FILE"
ENV_TIMEOUT = "PHYTO_HTTP_TIMEOUT_SECONDS"
ENV_MAX_RETRIES = "PHYTO_HTTP_MAX_RETRIES"
ENV_MAX_TURNS = "PHYTO_MAX_TURNS"
ENV_TEMPERATURE = "PHYTO_TEMPERATURE"
ENV_MAX_TOKENS = "PHYTO_MAX_TOKENS"
ENV_REASONING_EFFORT = "PHYTO_REASONING_EFFORT"
ENV_PROXY = "PHYTO_HTTP_PROXY"
ENV_TRACE_PAYLOAD = "PHYTO_TRACE_PAYLOAD"
ENV_PREFLIGHT_SAMPLES = "PHYTO_PREFLIGHT_SAMPLES"

TRACE_PAYLOAD_MODES = ("hash", "full")
REASONING_EFFORTS = ("low", "medium", "high")


def parse_dotenv(path: Path) -> dict[str, str]:
    """Read a minimal ``KEY=value`` file.

    Supports ``export`` prefixes, blank lines, ``#`` comments, single or double
    quoted values, and trailing `` #`` comments on unquoted values. It is
    deliberately small: a credential file is not the place for clever parsing.
    """
    if not path.is_file():
        return {}
    values: dict[str, str] = {}
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        key, separator, value = line.partition("=")
        if not separator:
            raise ConfigError(f"{path.name} line {number}: expected KEY=value, not a bare token")
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        elif " #" in value:
            value = value.split(" #", 1)[0].rstrip()
        values[key] = value
    return values


def _resolve(environ: dict[str, str], dotenv: dict[str, str], key: str) -> str | None:
    value = environ.get(key)
    if value is not None and value.strip():
        return value.strip()
    fallback = dotenv.get(key)
    return fallback.strip() if fallback and fallback.strip() else None


BASE_URL_ALIASES = {
    # "step_plan" names the subscription channel without making the operator
    # remember the full path. It is a different quota pool from the default.
    "step_plan": STEP_PLAN_BASE_URL,
    "step-plan": STEP_PLAN_BASE_URL,
    "default": DEFAULT_BASE_URL,
}


def _resolve_base_url(environ: dict[str, str], dotenv: dict[str, str]) -> str:
    """Resolve the endpoint, accepting a shorthand as well as a full URL."""
    raw = _resolve(environ, dotenv, ENV_BASE_URL)
    if raw is None:
        return DEFAULT_BASE_URL
    return BASE_URL_ALIASES.get(raw.strip().lower(), raw.strip())


def _integer(environ, dotenv, key: str, default: int, minimum: int) -> int:
    raw = _resolve(environ, dotenv, key)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ConfigError(f"{key} must be an integer, got {raw!r}") from exc
    if value < minimum:
        raise ConfigError(f"{key} must be >= {minimum}, got {value}")
    return value


def _number(environ, dotenv, key: str) -> float | None:
    raw = _resolve(environ, dotenv, key)
    if raw is None:
        return None
    try:
        return float(raw)
    except ValueError as exc:
        raise ConfigError(f"{key} must be a number, got {raw!r}") from exc


def _optional_integer(environ, dotenv, key: str, minimum: int) -> int | None:
    raw = _resolve(environ, dotenv, key)
    if raw is None:
        return None
    try:
        value = int(raw)
    except ValueError as exc:
        raise ConfigError(f"{key} must be an integer, got {raw!r}") from exc
    if value < minimum:
        raise ConfigError(f"{key} must be >= {minimum}, got {value}")
    return value


@dataclass
class HarnessConfig:
    base_url: str = DEFAULT_BASE_URL
    model: str = DEFAULT_MODEL
    api_key: str = field(default="", repr=False)
    timeout_seconds: float = 90.0
    max_retries: int = 4
    max_turns: int = 6
    temperature: float | None = None
    max_tokens: int | None = None
    reasoning_effort: str | None = None
    proxy: str | None = None
    trace_payload: str = "hash"
    preflight_samples: int = 20
    dotenv_path: Path | None = None
    api_key_source: str = "unset"

    @property
    def endpoint(self) -> str:
        return f"{self.base_url.rstrip('/')}/chat/completions"

    @property
    def host(self) -> str:
        """Host only, so a report can name the endpoint without any path or query."""
        without_scheme = self.base_url.split("://", 1)[-1]
        return without_scheme.split("/", 1)[0]

    def sampling_parameters(self) -> dict:
        """The exact sampling fields that will be sent, for the report.

        ``None`` means the field is not sent at all, which is different from
        sending a default: it means the provider's own default applies equally
        to both arms and is therefore not a controlled variable.
        """
        sent: dict = {}
        if self.temperature is not None:
            sent["temperature"] = self.temperature
        if self.max_tokens is not None:
            sent["max_tokens"] = self.max_tokens
        if self.reasoning_effort is not None:
            sent["reasoning_effort"] = self.reasoning_effort
        return sent

    def redacted(self) -> dict:
        """A report-safe view. Contains no credential material of any kind."""
        return {
            "base_url": self.base_url,
            "host": self.host,
            "model_requested": self.model,
            "api_key_present": bool(self.api_key),
            "api_key_source": self.api_key_source,
            "timeout_seconds": self.timeout_seconds,
            "max_retries": self.max_retries,
            "max_turns": self.max_turns,
            "sampling_parameters_sent": self.sampling_parameters(),
            "reasoning_effort_pinned": self.reasoning_effort is not None,
            "proxy_configured": self.proxy is not None,
            "trace_payload": self.trace_payload,
            "dotenv_loaded": self.dotenv_path is not None,
        }

    def __post_init__(self) -> None:
        if self.trace_payload not in TRACE_PAYLOAD_MODES:
            raise ConfigError(f"trace_payload must be one of {TRACE_PAYLOAD_MODES}")
        if self.reasoning_effort is not None and self.reasoning_effort not in REASONING_EFFORTS:
            raise ConfigError(f"reasoning_effort must be one of {REASONING_EFFORTS}")

    def require_key(self) -> str:
        if not self.api_key:
            raise ConfigError(
                f"No credential found. Set {ENV_API_KEY} or {ENV_API_KEY_FILE}, or create a "
                f"git-ignored {DEFAULT_DOTENV.name} next to pyproject.toml. The Harness does not "
                "fall back to fixture mode: an unconfigured run must fail, not silently degrade."
            )
        return self.api_key

    @classmethod
    def from_env(cls, environ: dict[str, str] | None = None, *,
                 dotenv_path: Path | None = None, use_dotenv: bool = True) -> "HarnessConfig":
        environ = dict(os.environ if environ is None else environ)
        path = dotenv_path if dotenv_path is not None else DEFAULT_DOTENV
        dotenv = parse_dotenv(path) if use_dotenv else {}
        loaded = path if dotenv else None

        key_source = "unset"
        api_key = _resolve(environ, dotenv, ENV_API_KEY) or ""
        if api_key:
            key_source = ENV_API_KEY

        key_file = _resolve(environ, dotenv, ENV_API_KEY_FILE)
        if not api_key and key_file:
            candidate = Path(key_file).expanduser()
            if not candidate.is_file():
                raise ConfigError(f"{ENV_API_KEY_FILE} points at a missing file: {candidate.name}")
            api_key = candidate.read_text(encoding="utf-8").strip()
            key_source = ENV_API_KEY_FILE
        if not api_key:
            key_source = "unset"

        reasoning = _resolve(environ, dotenv, ENV_REASONING_EFFORT)
        return cls(
            base_url=_resolve_base_url(environ, dotenv),
            model=_resolve(environ, dotenv, ENV_MODEL) or DEFAULT_MODEL,
            api_key=api_key,
            timeout_seconds=_number(environ, dotenv, ENV_TIMEOUT) or 90.0,
            max_retries=_integer(environ, dotenv, ENV_MAX_RETRIES, 4, 0),
            max_turns=_integer(environ, dotenv, ENV_MAX_TURNS, 6, 1),
            temperature=_number(environ, dotenv, ENV_TEMPERATURE),
            max_tokens=_optional_integer(environ, dotenv, ENV_MAX_TOKENS, 1),
            reasoning_effort=reasoning,
            proxy=_resolve(environ, dotenv, ENV_PROXY),
            trace_payload=_resolve(environ, dotenv, ENV_TRACE_PAYLOAD) or "hash",
            preflight_samples=_integer(environ, dotenv, ENV_PREFLIGHT_SAMPLES, 20, 3),
            dotenv_path=loaded,
            api_key_source=key_source,
        )
