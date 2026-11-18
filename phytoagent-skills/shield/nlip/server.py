"""Drop the AgentShield SE in front of any ``nlip_server`` application.

``shield_application(app, se)`` wraps every session the application creates:
ingress runs before the session's ``execute`` and egress runs on its reply.
The wrapped session's code is untouched, which is the framework-agnostic
property the hackathon's Area 03 asks for.

``ToolBridge`` is the Type-3 hook: a tool request that passed ingress is run
through the existing :class:`runtime.shield.ShieldRuntime`, with the delegated
scope passed as its ``task_grants``. The runtime then intersects it with the
Skill's sealed manifest and the host policy, so authority is

    delegation chain  ∩  host ceiling  ∩  sealed manifest

and none of the three can be enlarged by the party it constrains.
"""

from __future__ import annotations

import inspect
import json

from nlip_sdk.nlip import NLIP_Factory, NLIP_Message
from nlip_server.server import NLIP_Application, NLIP_Session

from shield.nlip.extension import (BLOCK, LABEL_TOOL_CALL, AgentShieldSE, Decision,
                                   _labelled)

SKILL_PREFIX = "skill:"


def refusal(decision: Decision) -> NLIP_Message:
    """An NLIP error message that says why, without echoing attacker content."""
    msg = NLIP_Factory.create_error_code("agentshield:" + ";".join(decision.reasons))
    msg.add_json({"agentshield": {"point": decision.point, "action": decision.action,
                                  "reasons": decision.reasons,
                                  "scos": sorted(set(decision.scos))}},
                 label="agentshield.decision")
    return msg


class ShieldedSession(NLIP_Session):
    def __init__(self, inner: NLIP_Session, se: AgentShieldSE, on_decision=None):
        self.inner = inner
        self.se = se
        self.on_decision = on_decision
        self.last_ingress: Decision | None = None
        self.last_egress: Decision | None = None

    async def start(self):
        result = self.inner.start()
        if inspect.isawaitable(result):
            await result

    async def stop(self):
        result = self.inner.stop()
        if inspect.isawaitable(result):
            await result

    async def execute(self, msg: NLIP_Message) -> NLIP_Message:
        inbound = self.se.ingress(msg)
        self.last_ingress = inbound
        if self.on_decision:
            self.on_decision(inbound)
        if inbound.action == BLOCK:
            return refusal(inbound)
        # The inner session can read what was authorised; it cannot change it.
        self.inner.shield_decision = inbound
        result = self.inner.execute(msg)
        response = await result if inspect.isawaitable(result) else result
        outbound = self.se.egress(response, recipient=inbound.sender)
        self.last_egress = outbound
        if self.on_decision:
            self.on_decision(outbound)
        if outbound.action == BLOCK:
            return refusal(outbound)
        return outbound.message


class ShieldedApplication(NLIP_Application):
    def __init__(self, inner: NLIP_Application, se: AgentShieldSE, on_decision=None):
        self.inner = inner
        self.se = se
        self.on_decision = on_decision
        self.sessions: list[ShieldedSession] = []

    async def startup(self):
        result = self.inner.startup()
        if inspect.isawaitable(result):
            await result

    async def shutdown(self):
        result = self.inner.shutdown()
        if inspect.isawaitable(result):
            await result

    def create_session(self) -> NLIP_Session:
        session = ShieldedSession(self.inner.create_session(), self.se, self.on_decision)
        # Keep only recent sessions: one is created per HTTP request.
        self.sessions = self.sessions[-32:] + [session]
        return session


def shield_application(inner: NLIP_Application, se: AgentShieldSE,
                       on_decision=None) -> ShieldedApplication:
    return ShieldedApplication(inner, se, on_decision)


class ToolBridge:
    """Run SE-authorised tool requests through AgentShield's ShieldRuntime."""

    def __init__(self, runtime):
        self.runtime = runtime

    def run(self, msg: NLIP_Message, decision: Decision, *, mode: str = "fixture") -> list[dict]:
        from runtime.shield import call_or_failure
        if decision.chain is None:
            return []
        grants = sorted(p for p in decision.effective_scope if not p.startswith(SKILL_PREFIX))
        results = []
        for sub in _labelled(msg, LABEL_TOOL_CALL):
            call = sub.content
            results.append(call_or_failure(
                self.runtime, call["tool"], call.get("payload") or {}, mode=mode,
                tool_call_id=call.get("tool_call_id") or "",
                requested_permissions=call.get("permissions") or [],
                task_grants=grants))
        return results


    def run_unshielded(self, msg: NLIP_Message, *, mode: str = "fixture") -> list[dict]:
        """Baseline only: what the agent does with no SE in front of it. The
        pre-existing runtime still checks manifest ∩ host, but no task term."""
        from runtime.shield import call_or_failure
        results = []
        for sub in _labelled(msg, LABEL_TOOL_CALL):
            call = sub.content if isinstance(sub.content, dict) else {}
            results.append(call_or_failure(
                self.runtime, call.get("tool", ""), call.get("payload") or {}, mode=mode,
                tool_call_id=call.get("tool_call_id") or "",
                requested_permissions=call.get("permissions") or []))
        return results


def summarise(results: list[dict]) -> str:
    return json.dumps([{k: r.get(k) for k in ("name", "status", "error")} for r in results],
                      ensure_ascii=False)
