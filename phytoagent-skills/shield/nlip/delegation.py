"""Signed, attenuating delegation chains for agents that talk over NLIP.

A delegation token says: issuer ``iss`` lets subject ``sub`` use ``scope`` when
talking to audience ``aud``, for ``purpose``, until ``exp``. A chain is a list
of tokens where every link is issued by the subject of the link before it and
names that link by hash. Verification refuses a chain unless every link

* carries a valid Ed25519 signature by a key the verifier already trusts,
* continues the chain (``iss`` of link *i* is ``sub`` of link *i-1*, and
  ``parent`` is the SHA-256 of link *i-1*),
* only narrows authority (scope is a subset, expiry is not later),
* is neither expired, issued in the future, nor revoked.

The effective authority of a chain is the *intersection* of every link's scope.
Attenuation is checked anyway, so a widening link is reported as an escalation
attempt instead of being silently clipped.

Only data visible at the NLIP boundary is used: the tokens travel inside the
message as reserved ``token``/``authorization`` submessages (ECMA-430 reserved
token), and keys are looked up in a local trust store.
"""

from __future__ import annotations

import base64
import hashlib
import json
import secrets
import time
from dataclasses import dataclass, field
from typing import Iterable

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import (Ed25519PrivateKey,
                                                               Ed25519PublicKey)
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

TOKEN_VERSION = "agentshield-dlg/1"
CHAIN_SEPARATOR = "~"


class DelegationError(Exception):
    """A delegation chain was refused. ``code`` is a stable machine reason."""

    def __init__(self, code: str, message: str):
        super().__init__(f"{code}: {message}")
        self.code = code


def _b64e(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _b64d(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


def canonical(obj) -> bytes:
    """Deterministic JSON bytes: the only form that is ever signed or hashed."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


@dataclass
class AgentIdentity:
    """An agent (or a human principal) with an Ed25519 key pair."""

    agent_id: str
    private_key: Ed25519PrivateKey = field(default_factory=Ed25519PrivateKey.generate, repr=False)

    @property
    def public_key(self) -> Ed25519PublicKey:
        return self.private_key.public_key()

    def public_bytes(self) -> bytes:
        return self.public_key.public_bytes(Encoding.Raw, PublicFormat.Raw)

    def sign(self, data: bytes) -> bytes:
        return self.private_key.sign(data)


class TrustStore:
    """agent_id -> public key, provisioned out of band (the verifier's local state)."""

    def __init__(self, identities: Iterable[AgentIdentity] = ()):
        self._keys: dict[str, Ed25519PublicKey] = {}
        for identity in identities:
            self.add(identity.agent_id, identity.public_key)

    def add(self, agent_id: str, key: Ed25519PublicKey) -> None:
        self._keys[agent_id] = key

    def key_for(self, agent_id: str) -> Ed25519PublicKey | None:
        return self._keys.get(agent_id)

    def verify(self, agent_id: str, signature: bytes, data: bytes) -> bool:
        key = self.key_for(agent_id)
        if key is None:
            return False
        try:
            key.verify(signature, data)
            return True
        except InvalidSignature:
            return False


def issue(issuer: AgentIdentity, *, sub: str, aud: str, scope: Iterable[str], purpose: str,
          ttl_s: float = 600.0, parent: str | None = None, now: float | None = None,
          exp: float | None = None) -> str:
    """Mint one signed link. ``parent`` is the encoded parent token, if any."""
    now = time.time() if now is None else now
    payload = {
        "v": TOKEN_VERSION,
        "iss": issuer.agent_id,
        "sub": sub,
        "aud": aud,
        "scope": sorted(set(scope)),
        "purpose": purpose,
        "iat": round(now, 3),
        "exp": round(exp if exp is not None else now + ttl_s, 3),
        "jti": secrets.token_hex(8),
        "parent": token_hash(parent) if parent else None,
    }
    body = canonical(payload)
    return _b64e(body) + "." + _b64e(issuer.sign(body))


def decode(token: str) -> tuple[dict, bytes, bytes]:
    """Split a token into (payload, signed bytes, signature) without trusting it."""
    try:
        body_part, sig_part = token.split(".")
        body = _b64d(body_part)
        payload = json.loads(body)
        signature = _b64d(sig_part)
    except Exception as exc:  # malformed token of any kind
        raise DelegationError("malformed_token", str(exc)) from exc
    if not isinstance(payload, dict) or payload.get("v") != TOKEN_VERSION:
        raise DelegationError("malformed_token", "unknown token version")
    return payload, body, signature


def token_hash(token: str) -> str:
    return hashlib.sha256(token.encode("ascii")).hexdigest()


def encode_chain(tokens: list[str]) -> str:
    return CHAIN_SEPARATOR.join(tokens)


def decode_chain(text: str) -> list[str]:
    return [part for part in (text or "").split(CHAIN_SEPARATOR) if part]


@dataclass
class VerifiedChain:
    principal: str            # who the authority originates from (root issuer)
    holder: str               # the subject of the last link
    audience: str             # the agent the last link is addressed to
    effective_scope: frozenset[str]
    purpose: str
    expires: float
    depth: int
    link_ids: list[str]

    def to_dict(self) -> dict:
        return {"principal": self.principal, "holder": self.holder, "audience": self.audience,
                "effective_scope": sorted(self.effective_scope), "purpose": self.purpose,
                "expires": self.expires, "depth": self.depth, "link_ids": self.link_ids}


class RevocationList:
    def __init__(self):
        self._revoked: set[str] = set()

    def revoke(self, token: str) -> None:
        payload, _, _ = decode(token)
        self._revoked.add(payload["jti"])

    def revoked(self, jti: str) -> bool:
        return jti in self._revoked


def verify_chain(tokens: list[str], *, trust: TrustStore, verifier_id: str,
                 trusted_roots: set[str], now: float | None = None,
                 revocations: RevocationList | None = None, max_depth: int = 4,
                 clock_skew_s: float = 5.0, enforce_attenuation: bool = True) -> VerifiedChain:
    """Verify a delegation chain end to end, or raise :class:`DelegationError`."""
    now = time.time() if now is None else now
    if not tokens:
        raise DelegationError("missing_delegation", "no delegation chain presented")
    if len(tokens) > max_depth:
        raise DelegationError("chain_too_deep", f"{len(tokens)} links > max {max_depth}")
    previous_payload: dict | None = None
    previous_token: str | None = None
    scope: frozenset[str] | None = None
    link_ids: list[str] = []
    for index, token in enumerate(tokens):
        payload, body, signature = decode(token)
        issuer = payload.get("iss")
        if index == 0 and issuer not in trusted_roots:
            raise DelegationError("untrusted_root", f"root issuer {issuer!r} is not a trusted principal")
        if not trust.verify(issuer, signature, body):
            raise DelegationError("bad_signature", f"link {index} is not validly signed by {issuer!r}")
        if previous_payload is not None:
            if issuer != previous_payload["sub"]:
                raise DelegationError("broken_chain",
                                      f"link {index} issued by {issuer!r}, but link {index-1} "
                                      f"delegated to {previous_payload['sub']!r}")
            if payload.get("parent") != token_hash(previous_token):
                raise DelegationError("broken_chain", f"link {index} does not name its parent")
        elif payload.get("parent") is not None:
            raise DelegationError("broken_chain", "root link claims a parent")
        if payload.get("aud") != verifier_id:
            # Every link is bound to one resource agent, so authority granted
            # for agent X cannot be spent at agent Y (confused deputy).
            raise DelegationError("wrong_audience",
                                  f"link {index} is addressed to {payload.get('aud')!r}, not {verifier_id!r}")
        link_scope = frozenset(payload.get("scope") or [])
        if not enforce_attenuation:
            # Ablation only ("token passing"): the last link's scope is taken at
            # face value, as a plain bearer token would be.
            scope = link_scope
            previous_payload, previous_token = payload, token
            link_ids.append(payload["jti"])
            if payload["exp"] <= now:
                raise DelegationError("expired", f"link {index} expired")
            if revocations is not None and revocations.revoked(payload["jti"]):
                raise DelegationError("revoked", f"link {index} ({payload['jti']}) is revoked")
            continue
        if scope is not None and not link_scope <= scope:
            raise DelegationError("scope_escalation",
                                  f"link {index} widens scope by {sorted(link_scope - scope)}")
        if previous_payload is not None and payload["exp"] > previous_payload["exp"]:
            raise DelegationError("lifetime_escalation", f"link {index} outlives its parent")
        if payload["exp"] <= now:
            raise DelegationError("expired", f"link {index} expired")
        if payload["iat"] > now + clock_skew_s:
            raise DelegationError("not_yet_valid", f"link {index} issued in the future")
        if revocations is not None and revocations.revoked(payload["jti"]):
            raise DelegationError("revoked", f"link {index} ({payload['jti']}) is revoked")
        scope = link_scope if scope is None else scope & link_scope
        previous_payload, previous_token = payload, token
        link_ids.append(payload["jti"])
    assert previous_payload is not None and scope is not None
    first, _, _ = decode(tokens[0])
    return VerifiedChain(principal=first["iss"], holder=previous_payload["sub"],
                         audience=previous_payload["aud"], effective_scope=scope,
                         purpose=previous_payload["purpose"], expires=previous_payload["exp"],
                         depth=len(tokens), link_ids=link_ids)
