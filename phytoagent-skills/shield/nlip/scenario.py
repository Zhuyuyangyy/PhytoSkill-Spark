"""The end-to-end scenario used by the tests, the evaluation and the demo.

A farmer (user client) asks a Coordinator agent to assess a Huangqi (黄芪)
field case. The Coordinator holds a delegation from the farmer and talks NLIP
to a Plant-Lab agent, which owns the medicinal-plant Skills. For literature
look-ups the Coordinator subcontracts a Herb-Scout agent and passes on only
part of its authority. Mallory is an agent the Plant-Lab knows (it is in the
trust store) but that holds no delegation for this task.

    user:farmer ──(root delegation)──> agent:coordinator ──NLIP──> agent:plant-lab
                                             │                         ▲
                                             └─(attenuated)─> agent:herb-scout ─NLIP─┘
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from nlip_sdk.nlip import NLIP_Factory, NLIP_Message
from nlip_server.server import NLIP_Application, NLIP_Session

from shield.nlip import delegation as dlg
from shield.nlip.extension import (AgentShieldSE, SEPolicy, attach_delegation, attach_tool_call,
                                   seal)

USER, COORD, SCOUT, LAB, MALLORY = ("user:farmer", "agent:coordinator", "agent:herb-scout",
                                    "agent:plant-lab", "agent:mallory")

FIXTURE_CASE = {"case_id": "demo-huangqi-001", "species": "黄芪",
                "image_path": "fixture://huangqi-leaf-01"}

# What the farmer lets the Coordinator do at the Plant-Lab.
USER_SCOPE = ["case_workspace:read", "corpus:read", "skill:plant_vision", "skill:growth_risk",
              "skill:herbal_knowledge"]
# What the Plant-Lab host will ever honour, whoever delegates it. Broker
# inference (a paid remote model call) is deliberately outside the ceiling.
LAB_CEILING = frozenset({"case_workspace:read", "corpus:read", "skill:plant_vision",
                         "skill:growth_risk", "skill:herbal_knowledge"})
# What the Coordinator passes on to the Herb-Scout.
SCOUT_SCOPE = ["corpus:read", "skill:herbal_knowledge"]


@dataclass
class World:
    ids: dict[str, dlg.AgentIdentity]
    trust: dlg.TrustStore
    lab_se: AgentShieldSE
    clock: list[float]
    root_chain: list[str] = field(default_factory=list)
    real_clock: bool = False

    def now(self) -> float:
        return time.time() if self.real_clock else self.clock[0]

    def advance(self, seconds: float) -> None:
        self.clock[0] += seconds


def _demo_identity(name: str, seed: str) -> dlg.AgentIdentity:
    """DEMO ONLY: a key derived from a public seed so two processes agree on it."""
    import hashlib
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    raw = hashlib.sha256(f"{seed}:{name}".encode()).digest()
    return dlg.AgentIdentity(name, Ed25519PrivateKey.from_private_bytes(raw))


def build_world(*, freshness_window_s: float = 30.0, canaries=("PLANT-LAB-SYSTEM-PROMPT-7f3a",),
                rate_per_s: float = 1000.0, burst: int = 1000, demo_seed: str | None = None,
                real_clock: bool = False) -> World:
    names = (USER, COORD, SCOUT, LAB, MALLORY)
    ids = {name: (_demo_identity(name, demo_seed) if demo_seed else dlg.AgentIdentity(name))
           for name in names}
    trust = dlg.TrustStore(ids.values())
    clock = [time.time()]
    policy = SEPolicy(host_ceiling=LAB_CEILING, trusted_roots={USER},
                      freshness_window_s=freshness_window_s, canaries=tuple(canaries),
                      egress_allowed_hosts=frozenset({"nlip-project.org"}),
                      rate_per_s=rate_per_s, burst=burst)
    se = AgentShieldSE(ids[LAB], trust, policy,
                       clock=time.time if real_clock else (lambda: clock[0]))
    world = World(ids=ids, trust=trust, lab_se=se, clock=clock, real_clock=real_clock)
    world.root_chain = [dlg.issue(ids[USER], sub=COORD, aud=LAB, scope=USER_SCOPE,
                                  purpose="assess field case demo-huangqi-001", ttl_s=3600,
                                  now=clock[0])]
    return world


def scout_chain(world: World, scope=SCOUT_SCOPE) -> list[str]:
    """The Coordinator's honest, attenuated sub-delegation to the Herb-Scout."""
    link = dlg.issue(world.ids[COORD], sub=SCOUT, aud=LAB, scope=scope,
                     purpose="literature lookup for demo-huangqi-001", ttl_s=600,
                     now=world.now(), parent=world.root_chain[-1])
    return world.root_chain + [link]


def request(world: World, *, sender: str = COORD, chain: list[str] | None = None,
            text: str = "Please assess case demo-huangqi-001.", tool: str | None = "plant_vision",
            permissions=("case_workspace:read",), payload: dict | None = None,
            recipient: str = LAB, sealed: bool = True, nonce: str | None = None,
            ts: float | None = None) -> NLIP_Message:
    msg = NLIP_Factory.create_text(text)
    attach_delegation(msg, world.root_chain if chain is None else chain)
    if tool:
        attach_tool_call(msg, tool=tool, permissions=list(permissions),
                         payload=payload if payload is not None else dict(FIXTURE_CASE))
    if sealed:
        seal(msg, world.ids[sender], recipient, now=world.now() if ts is None else ts, nonce=nonce)
    return msg


class PlantLabSession(NLIP_Session):
    """The protected agent's own logic. It knows nothing about AgentShield
    except that, when wrapped, an authorised decision is attached to it."""

    def __init__(self, bridge=None, secret_in_reply: str | None = None):
        self.bridge = bridge
        self.secret_in_reply = secret_in_reply

    def execute(self, msg: NLIP_Message) -> NLIP_Message:
        decision = getattr(self, "shield_decision", None)
        results = []
        if self.bridge is not None and decision is not None:
            results = self.bridge.run(msg, decision)
        elif self.bridge is not None:
            results = self.bridge.run_unshielded(msg)
        text = f"plant-lab processed {len(results)} tool call(s)"
        if self.secret_in_reply:
            text += " " + self.secret_in_reply
        reply = NLIP_Factory.create_text(text)
        reply.add_json({"results": [{"name": r.get("name"), "status": r.get("status"),
                                     "error": (r.get("error") or {}).get("code")}
                                    for r in results]}, label="plant-lab.results")
        return reply


class PlantLabApp(NLIP_Application):
    def __init__(self, bridge=None, secret_in_reply: str | None = None):
        self.bridge = bridge
        self.secret_in_reply = secret_in_reply

    async def startup(self):
        return None

    async def shutdown(self):
        return None

    def create_session(self) -> NLIP_Session:
        return PlantLabSession(self.bridge, self.secret_in_reply)
