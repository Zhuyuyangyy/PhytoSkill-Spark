"""Errors raised by the live Harness.

The Harness is the *caller* of Skills, so it sits outside the Skill SDK error
hierarchy on purpose: a Harness failure must not be mistaken for a Skill
contract failure.
"""


class HarnessError(Exception):
    """Base error for the live Harness."""


class ConfigError(HarnessError):
    """Configuration is missing, malformed or unsafe."""


class TransportError(HarnessError):
    """The model endpoint could not be reached or returned a fatal status."""


class AuthError(TransportError):
    """The endpoint rejected the configured credential."""


class PreflightError(HarnessError):
    """A preflight probe did not meet the condition required to run A/B."""
