"""AgentShield as a pluggable NLIP Security Extension (SE).

One object, two control points:

* ``ingress(msg)`` runs before an agent sees an incoming NLIP message
  (Type-1 when the peer is a user client, Type-2 when it is another agent).
* ``egress(msg, recipient)`` runs before a message leaves the agent.

Every decision is made from data visible at the boundary: the NLIP message
itself, local state from earlier messages (replay cache, rate buckets,
revocations, the audit chain), and the operator's policy. Nothing depends on
the agent's framework or hidden reasoning, so the same SE can front any
``nlip_server`` session.

Message conventions (all inside standard NLIP fields, no new top-level keys):

=====================  ==============================================================
Purpose                NLIP submessage
=====================  ==============================================================
delegation chain       format=token, subformat=authorization, label=agentshield.delegation
proof of possession    format=token, subformat=authorization, label=agentshield.proof
tool request           format=structured, subformat=json, label=agentshield.tool_call
=====================  ==============================================================

The proof binds sender, audience, a nonce, a timestamp and the SHA-256 of the
rest of the message, signed by the sender's key. It is what makes a stolen
delegation token useless to anyone but its holder and makes replays visible.
"""

from __future__ import annotations

import hashlib
import json
import re
import secrets
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Callable, Iterable

from nlip_sdk.nlip import AllowedFormats, NLIP_Message, NLIP_SubMessage

from shield.nlip import delegation as dlg
from shield.nlip.injection import InjectionFinding, iter_strings, scan_text

LABEL_DELEGATION = "agentshield.delegation"
LABEL_PROOF = "agentshield.proof"
LABEL_TOOL_CALL = "agentshield.tool_call"
AUTH_SUBFORMAT = "authorization"

ALLOW, BLOCK, REDACT = "allow", "block", "redact"

# ECMA-434 security countermeasures and objectives each check implements.
SCO = {
    "authn": "ECMA-434 7.2.1 Client Authentication",
    "authz": "ECMA-434 7.2.2 Client Authorization",
    "deputy": "ECMA-434 7.2.3 Privilege Misuse Prevention",
    "audit": "ECMA-434 7.2.4 Observability & Audit",
    "throttle": "ECMA-434 7.2.5 Client Throttling and DoS",
    "injection": "ECMA-434 7.3.1 Prevention of Prompt Injection",
    "behaviour": "ECMA-434 7.3.2 Runtime Agent Behaviour Controls",
    "leakage": "ECMA-434 7.3.3 Information Leakage",
    "freshness": "Proposed SCO: Message Freshness and Replay Resistance",
}

SECRET_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("private_key", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
    ("aws_access_key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("api_key_sk", re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b")),
    ("github_token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}\b")),
    ("assignment_secret", re.compile(r"(?i)\b(api[_-]?key|secret|password|passwd|token)\s*[:=]\s*['\"]?[^\s'\"]{8,}")),
)
URL = re.compile(r"https?://([A-Za-z0-9.-]+)")


def _now() -> float:
    return time.time()


def body_digest(msg: NLIP_Message) -> str:
    """SHA-256 over the message with the proof submessage removed."""
    data = msg.to_dict()
    data["submessages"] = [s for s in data.get("submessages") or []
                           if s.get("label") != LABEL_PROOF]
    if not data["submessages"]:
        data.pop("submessages")
    return hashlib.sha256(dlg.canonical(data)).hexdigest()


def _submessages(msg: NLIP_Message) -> list[NLIP_SubMessage]:
    return list(msg.submessages or [])


def _labelled(msg: NLIP_Message, label: str) -> list[NLIP_SubMessage]:
    # nlip_sdk 0.1.3's extract_authentication_token ignores its label argument,
    # so labelled tokens are located directly.
    return [s for s in _submessages(msg) if (s.label or "").lower() == label]


def attach_delegation(msg: NLIP_Message, chain: list[str]) -> NLIP_Message:
    msg.add_submessage(NLIP_SubMessage(format=AllowedFormats.token, subformat=AUTH_SUBFORMAT,
                                       content=dlg.encode_chain(chain), label=LABEL_DELEGATION))
    return msg


def attach_tool_call(msg: NLIP_Message, *, tool: str, permissions: list[str], payload: dict,
                     tool_call_id: str | None = None) -> NLIP_Message:
    msg.add_submessage(NLIP_SubMessage(
        format=AllowedFormats.structured, subformat="json", label=LABEL_TOOL_CALL,
        content={"tool": tool, "permissions": list(permissions), "payload": payload,
                 "tool_call_id": tool_call_id or f"call-{secrets.token_hex(4)}"}))
    return msg


def seal(msg: NLIP_Message, sender: dlg.AgentIdentity, recipient: str, *,
         now: float | None = None, nonce: str | None = None) -> NLIP_Message:
    """Add the proof-of-possession submessage. Must be the last change to ``msg``."""
    for existing in _labelled(msg, LABEL_PROOF):
        msg.submessages.remove(existing)
    claims = {"sender": sender.agent_id, "aud": recipient,
              "nonce": nonce or secrets.token_hex(12),
              "ts": round(_now() if now is None else now, 3),
              "body_sha256": body_digest(msg)}
    body = dlg.canonical(claims)
    msg.add_submessage(NLIP_SubMessage(format=AllowedFormats.token, subformat=AUTH_SUBFORMAT,
                                       content=dlg._b64e(body) + "." + dlg._b64e(sender.sign(body)),
                                       label=LABEL_PROOF))
    return msg


@dataclass
class Decision:
    point: str                       # ingress | egress
    action: str                      # allow | block | redact
    reasons: list[str] = field(default_factory=list)
    scos: list[str] = field(default_factory=list)
    sender: str | None = None
    chain: dlg.VerifiedChain | None = None
    effective_scope: frozenset[str] = frozenset()   # chain ∩ host ceiling
    tool_calls: list[dict] = field(default_factory=list)
    findings: list[dict] = field(default_factory=list)
    latency_us: float = 0.0
    message: NLIP_Message | None = None   # the (possibly redacted) message to pass on

    @property
    def allowed(self) -> bool:
        return self.action != BLOCK

    def to_dict(self) -> dict:
        return {"point": self.point, "action": self.action, "reasons": self.reasons,
                "scos": sorted(set(self.scos)), "sender": self.sender,
                "chain": self.chain.to_dict() if self.chain else None,
                "effective_scope": sorted(self.effective_scope),
                "tool_calls": self.tool_calls, "findings": self.findings,
                "latency_us": round(self.latency_us, 1)}


class AuditChain:
    """Append-only, hash-chained decision log: editing any entry breaks verify()."""

    def __init__(self):
        self.entries: list[dict] = []
        self._head = "0" * 64

    def append(self, record: dict) -> dict:
        entry = {"seq": len(self.entries), "prev": self._head, "record": record}
        entry["hash"] = hashlib.sha256(dlg.canonical({k: entry[k] for k in ("seq", "prev", "record")})).hexdigest()
        self.entries.append(entry)
        self._head = entry["hash"]
        return entry

    def verify(self) -> bool:
        head = "0" * 64
        for index, entry in enumerate(self.entries):
            expected = hashlib.sha256(dlg.canonical({"seq": index, "prev": head,
                                                     "record": entry["record"]})).hexdigest()
            if entry["prev"] != head or entry["hash"] != expected:
                return False
            head = entry["hash"]
        return True


class ReplayCache:
    """Remembers (sender, nonce) for the freshness window; bounded in size."""

    def __init__(self, window_s: float, max_entries: int = 100_000):
        self.window_s = window_s
        self.max_entries = max_entries
        self._seen: OrderedDict[tuple[str, str], float] = OrderedDict()

    def check_and_add(self, sender: str, nonce: str, now: float) -> bool:
        while self._seen:
            key, seen_at = next(iter(self._seen.items()))
            if seen_at >= now - 2 * self.window_s and len(self._seen) < self.max_entries:
                break
            self._seen.popitem(last=False)
        if (sender, nonce) in self._seen:
            return False
        self._seen[(sender, nonce)] = now
        return True


class TokenBucket:
    def __init__(self, rate_per_s: float, burst: int):
        self.rate, self.burst = rate_per_s, burst
        self._state: dict[str, tuple[float, float]] = {}

    def take(self, key: str, now: float) -> bool:
        tokens, last = self._state.get(key, (float(self.burst), now))
        tokens = min(self.burst, tokens + (now - last) * self.rate)
        if tokens < 1:
            self._state[key] = (tokens, now)
            return False
        self._state[key] = (tokens - 1, now)
        return True


@dataclass
class SEPolicy:
    """Operator policy. ``host_ceiling`` caps any delegated scope on this agent."""

    host_ceiling: frozenset[str]
    trusted_roots: set[str]
    require_delegation: bool = True
    freshness_window_s: float = 30.0
    max_chain_depth: int = 4
    injection_action: str = BLOCK          # block | allow (log only)
    scan_egress_for_injection: bool = True  # stop an agent relaying injected text (Type-2)
    egress_allowed_hosts: frozenset[str] = frozenset()
    canaries: tuple[str, ...] = ()          # internal strings that must never leave
    rate_per_s: float = 50.0
    burst: int = 100
    # Ablation switches for the evaluation baselines. Production keeps all True.
    check_proof: bool = True          # proof of possession + freshness + replay cache
    check_attenuation: bool = True    # chain must only narrow; effective = intersection
    injection_screen: bool = True


class AgentShieldSE:
    """The Security Extension. One instance per protected agent."""

    def __init__(self, identity: dlg.AgentIdentity, trust: dlg.TrustStore, policy: SEPolicy, *,
                 clock: Callable[[], float] = _now, revocations: dlg.RevocationList | None = None):
        self.identity = identity
        self.trust = trust
        self.policy = policy
        self.clock = clock
        self.revocations = revocations or dlg.RevocationList()
        self.replay = ReplayCache(policy.freshness_window_s)
        self.buckets = TokenBucket(policy.rate_per_s, policy.burst)
        self.audit = AuditChain()

    # ── ingress ──────────────────────────────────────────────────────────

    def ingress(self, msg: NLIP_Message) -> Decision:
        start = time.perf_counter()
        decision = Decision(point="ingress", action=ALLOW, message=msg)
        try:
            self._ingress(msg, decision)
        finally:
            decision.latency_us = (time.perf_counter() - start) * 1e6
            self._record(decision, msg)
        return decision

    def _block(self, decision: Decision, reason: str, sco: str) -> None:
        decision.action = BLOCK
        decision.reasons.append(reason)
        decision.scos.append(SCO[sco])

    def _ingress(self, msg: NLIP_Message, d: Decision) -> None:
        now = self.clock()
        proofs = _labelled(msg, LABEL_PROOF)
        chains = _labelled(msg, LABEL_DELEGATION)
        sender = None
        # 1. proof of possession, freshness, replay
        if len(proofs) > 1 or len(chains) > 1:
            return self._block(d, "duplicate_auth_token", "authn")
        if not self.policy.check_proof:
            proofs = []
        if proofs:
            try:
                body_b64, sig_b64 = str(proofs[0].content).split(".")
                body = dlg._b64d(body_b64)
                claims = json.loads(body)
                signature = dlg._b64d(sig_b64)
            except Exception:
                return self._block(d, "malformed_proof", "authn")
            sender = claims.get("sender")
            d.sender = sender
            # Verify before charging the sender's bucket, so a forged proof
            # cannot drain an honest agent's quota; forgeries share one bucket.
            if not self.trust.verify(sender, signature, body):
                self.buckets.take("unauthenticated", now)
                return self._block(d, "bad_proof_signature", "authn")
            if not self.buckets.take(str(sender), now):
                return self._block(d, "rate_limited", "throttle")
            if claims.get("aud") != self.identity.agent_id:
                return self._block(d, "proof_wrong_audience", "freshness")
            if abs(now - float(claims.get("ts", 0))) > self.policy.freshness_window_s:
                return self._block(d, "stale_message", "freshness")
            if claims.get("body_sha256") != body_digest(msg):
                return self._block(d, "body_tampered", "authn")
            if not self.replay.check_and_add(sender, str(claims.get("nonce")), now):
                return self._block(d, "replayed_nonce", "freshness")
        elif self.policy.require_delegation and self.policy.check_proof:
            if not self.buckets.take("unauthenticated", now):
                return self._block(d, "rate_limited", "throttle")
            return self._block(d, "missing_proof", "authn")
        # 2. delegation chain
        effective: frozenset[str] = frozenset()
        if chains:
            try:
                chain = dlg.verify_chain(dlg.decode_chain(str(chains[0].content)), trust=self.trust,
                                         verifier_id=self.identity.agent_id,
                                         trusted_roots=self.policy.trusted_roots, now=now,
                                         revocations=self.revocations,
                                         max_depth=self.policy.max_chain_depth,
                                         enforce_attenuation=self.policy.check_attenuation)
            except dlg.DelegationError as exc:
                sco = "deputy" if exc.code in ("scope_escalation", "lifetime_escalation",
                                               "wrong_audience") else "authz"
                return self._block(d, f"delegation:{exc.code}", sco)
            if self.policy.check_proof and (sender is None or chain.holder != sender):
                return self._block(d, "delegation:holder_mismatch", "deputy")
            d.chain = chain
            effective = chain.effective_scope & self.policy.host_ceiling
            d.effective_scope = effective
        elif self.policy.require_delegation:
            return self._block(d, "missing_delegation", "authz")
        # 3. tool requests: requested ⊆ chain ∩ host ceiling
        for sub in _labelled(msg, LABEL_TOOL_CALL):
            call = sub.content if isinstance(sub.content, dict) else {}
            tool = call.get("tool")
            requested = set(call.get("permissions") or [])
            needed = requested | {f"skill:{tool}"}
            missing = sorted(needed - effective)
            d.tool_calls.append({"tool": tool, "requested": sorted(requested),
                                 "tool_call_id": call.get("tool_call_id"),
                                 "granted": sorted(effective), "denied": missing})
            if not tool or missing:
                self._block(d, f"unauthorised_tool_call:{tool}:{','.join(missing)}", "authz")
        if d.action == BLOCK:
            return
        # 4. prompt-injection screen over every non-token string in the message
        findings = self._scan_injection(msg) if self.policy.injection_screen else []
        if findings:
            d.findings = [f.to_dict() for f in findings]
            d.scos.append(SCO["injection"])
            if self.policy.injection_action == BLOCK:
                d.action = BLOCK
                d.reasons.append("prompt_injection:" + ",".join(sorted({f.rule for f in findings})))
            else:
                d.reasons.append("prompt_injection_logged")
        if d.action == ALLOW:
            d.scos.extend([SCO["authn"], SCO["authz"], SCO["freshness"]] if proofs else [])

    def _scan_injection(self, msg: NLIP_Message) -> list[InjectionFinding]:
        findings: list[InjectionFinding] = []
        units = [("content", msg.format, msg.content, msg.label)]
        units += [(f"submessages[{i}]", s.format, s.content, s.label)
                  for i, s in enumerate(_submessages(msg))]
        for where, fmt, content, label in units:
            if label:
                findings.extend(scan_text(label, where + ".label"))
            if str(fmt).lower() == "token":
                continue
            for path, text in iter_strings(content, where):
                findings.extend(scan_text(text, path))
        return findings

    # ── egress ───────────────────────────────────────────────────────────

    def egress(self, msg: NLIP_Message, recipient: str | None = None) -> Decision:
        start = time.perf_counter()
        d = Decision(point="egress", action=ALLOW, sender=self.identity.agent_id)
        out = msg.model_copy(deep=True)
        redactions: list[dict] = []

        def clean(text: str, where: str) -> str:
            for name, pattern in SECRET_PATTERNS:
                if pattern.search(text):
                    redactions.append({"rule": name, "where": where})
                    text = pattern.sub("[REDACTED]", text)
            for canary in self.policy.canaries:
                if canary and canary in text:
                    redactions.append({"rule": "canary", "where": where})
                    text = text.replace(canary, "[REDACTED]")
            for host in URL.findall(text):
                if self.policy.egress_allowed_hosts and host.lower() not in self.policy.egress_allowed_hosts:
                    redactions.append({"rule": "unlisted_url", "where": where, "host": host})
                    text = re.sub(r"https?://" + re.escape(host) + r"\S*", "[REDACTED-URL]", text)
            return text

        def walk(value, where):
            if isinstance(value, str):
                return clean(value, where)
            if isinstance(value, dict):
                return {k: walk(v, f"{where}.{k}") for k, v in value.items()}
            if isinstance(value, list):
                return [walk(v, f"{where}[{i}]") for i, v in enumerate(value)]
            return value

        if str(out.format).lower() != "token":
            out.content = walk(out.content, "content")
        for i, sub in enumerate(_submessages(out)):
            if str(sub.format).lower() != "token":
                sub.content = walk(sub.content, f"submessages[{i}]")
        if redactions:
            d.action = REDACT
            d.reasons.append("leakage_redacted:" + ",".join(sorted({r["rule"] for r in redactions})))
            d.scos.append(SCO["leakage"])
            d.findings.extend(redactions)
        if self.policy.scan_egress_for_injection and self.policy.injection_screen:
            findings = self._scan_injection(out)
            if findings:
                d.action = BLOCK
                d.reasons.append("injection_propagation:" + ",".join(sorted({f.rule for f in findings})))
                d.scos.append(SCO["injection"])
                d.findings.extend(f.to_dict() for f in findings)
        d.message = out
        d.latency_us = (time.perf_counter() - start) * 1e6
        self._record(d, msg, recipient=recipient)
        return d

    # ── delegation helper: attenuate before passing authority on ─────────

    def delegate(self, inbound: Decision, inbound_chain: list[str], *, to: str, aud: str,
                 scope: Iterable[str], purpose: str | None = None, ttl_s: float = 120.0) -> list[str]:
        """Extend a verified inbound chain to a downstream agent, never widening it."""
        if inbound.chain is None or inbound.action == BLOCK:
            raise dlg.DelegationError("no_authority", "cannot delegate without a verified inbound chain")
        scope = frozenset(scope)
        own = inbound.chain.effective_scope & self.policy.host_ceiling
        if not scope <= own:
            self.audit.append({"point": "delegate", "action": BLOCK, "to": to,
                               "reason": "scope_escalation", "extra": sorted(scope - own)})
            raise dlg.DelegationError("scope_escalation",
                                      f"cannot pass on {sorted(scope - own)}; holder has {sorted(own)}")
        now = self.clock()
        link = dlg.issue(self.identity, sub=to, aud=aud, scope=scope,
                         purpose=purpose or inbound.chain.purpose, now=now,
                         exp=min(now + ttl_s, inbound.chain.expires), parent=inbound_chain[-1])
        self.audit.append({"point": "delegate", "action": ALLOW, "to": to, "scope": sorted(scope)})
        return inbound_chain + [link]

    def _record(self, d: Decision, msg: NLIP_Message, recipient: str | None = None) -> None:
        self.audit.append({"point": d.point, "action": d.action, "reasons": d.reasons,
                           "sender": d.sender, "recipient": recipient,
                           "principal": d.chain.principal if d.chain else None,
                           "scope": sorted(d.chain.effective_scope) if d.chain else None,
                           "tool_calls": d.tool_calls,
                           "body_sha256": hashlib.sha256(msg.to_json().encode()).hexdigest()})


def inbound_chain_tokens(msg: NLIP_Message) -> list[str]:
    chains = _labelled(msg, LABEL_DELEGATION)
    return dlg.decode_chain(str(chains[0].content)) if chains else []
