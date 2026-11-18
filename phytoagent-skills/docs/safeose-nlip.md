# AgentShield as an NLIP Security Extension (NDSS SafeOSE '27 hackathon)

Branch `safeose-nlip`. Focus areas **03** (security function at a unified NLIP
control point) and **06** (cross-agent privilege delegation).

## What it is

`shield/nlip/` turns AgentShield into a pluggable Security Extension (SE) that
sits at the ingress and egress of any `nlip_server` agent, using only what is
visible at the NLIP boundary: the message, local state (replay cache, rate
buckets, revocations, audit chain) and the operator's policy.

| Module | Role |
|---|---|
| `shield/nlip/delegation.py` | Ed25519-signed delegation links; chain verification with attenuation, expiry, revocation, audience binding, depth limit |
| `shield/nlip/extension.py` | `AgentShieldSE.ingress()` / `.egress()`; proof of possession, freshness/replay cache, throttling, tool-call authorisation, injection screen, leakage redaction, hash-chained audit log, `delegate()` helper |
| `shield/nlip/injection.py` | Deterministic prompt-injection rules (EN/ZH/DE), NFKC + zero-width normalisation, base64 unwrapping |
| `shield/nlip/server.py` | `shield_application(app, se)` wraps any `NLIP_Application`; `ToolBridge` runs authorised tool calls through the existing `ShieldRuntime` |
| `shield/nlip/scenario.py` | Farmer → Coordinator → Herb-Scout / Plant-Lab scenario used by tests, eval and demo |
| `shield/nlip/attacks.py` | 33 attacks, 5 configurations, HTTP harness |
| `scripts/nlip_eval.py` | Regenerates every number in the report |
| `demo/nlip_demo.py` | Walkthrough, in-process or as a real uvicorn server + remote client |

## Authority = delegation chain ∩ host ceiling ∩ sealed manifest

1. **Delegation chain** (new): the intersection of every link's scope. A link that
   widens scope or outlives its parent is refused as an escalation, not clipped.
2. **Host ceiling** (new, per protected agent): what this agent will ever honour.
3. **Sealed Skill manifest** (existing AgentShield runtime): what the Skill
   package declared and the compile gate admitted.

The SE checks (1)∩(2) at ingress; the `ToolBridge` passes that set to
`ShieldRuntime.call(task_grants=...)`, which intersects it with (3).

## Wire conventions (inside standard NLIP fields)

| Purpose | NLIP submessage |
|---|---|
| delegation chain | `format=token`, `subformat=authorization`, `label=agentshield.delegation`, content = links joined by `~` |
| proof of possession | `format=token`, `subformat=authorization`, `label=agentshield.proof`, content = b64(json{sender, aud, nonce, ts, body_sha256}) `.` b64(Ed25519 sig) |
| tool request | `format=structured`, `subformat=json`, `label=agentshield.tool_call`, content = {tool, permissions, payload, tool_call_id} |

`body_sha256` covers the canonical JSON of the whole message minus the proof, so
any in-flight change is detected. A refused message gets an NLIP `error`
message back (`agentshield:<reasons>`) plus a structured decision submessage.

Note: `nlip_sdk` 0.1.3's `extract_authentication_token(label)` ignores the
label argument, so the SE locates labelled tokens itself.

## ECMA-434 mapping

| Check | SCO |
|---|---|
| proof of possession, chain signature, trusted root | 7.2.1 Client Authentication |
| tool request ⊆ chain ∩ host ceiling | 7.2.2 Client Authorization |
| holder binding, audience binding, attenuation | 7.2.3 Privilege Misuse Prevention (confused deputy) |
| hash-chained decision log with principal, scope and tool calls | 7.2.4 Observability & Audit |
| per-sender token bucket (charged only after signature check) | 7.2.5 Client Throttling and DoS |
| ingress injection screen, egress propagation screen | 7.3.1 Prevention of Prompt Injection |
| egress secret / canary / URL redaction | 7.3.3 Information Leakage |
| nonce cache + timestamp window + audience in proof | **proposed SCO: Message Freshness and Replay Resistance** (no existing SCO covers replay of signed NLIP messages) |

## Reproduce

```bash
cd phytoagent-skills
bash scripts/run_safeose.sh
```

or step by step:

```bash
pip install -e ".[test,nlip]"
python -m pytest -q tests/test_nlip_se.py      # 85 tests
python -m scripts.nlip_eval                     # artifacts/safeose/nlip-eval.{json,md}
python -m demo.nlip_demo                        # walkthrough
python -m demo.nlip_demo --serve --port 8010    # terminal 1
python -m demo.nlip_demo --remote http://127.0.0.1:8010   # terminal 2
```

## Known limitations (also in the report)

* The injection screen is rule-based. On the public deepset/prompt-injections
  test split it blocks a minority of injections (see `artifacts/safeose/nlip-eval.md`);
  paraphrase defeats it (attacks INJ-10, INJ-11 are kept in the suite as
  expected misses). The rules were written before that dataset was run and were
  not tuned on it afterwards.
* Keys are provisioned out of band in a local trust store; there is no key
  discovery, rotation or revocation of *keys* (only of delegation links).
* Revocation is a local list; distributing it between agents is not addressed.
* The replay cache is in memory and per process; a restarted or horizontally
  scaled agent needs a shared cache.
* TLS (ECMA-434 7.1.1) is assumed from the deployment, not provided here.
* The demo derives keys from a public seed so two processes agree; never do this
  in a deployment.
* All results come from a simulated scenario with fixture Skills; no language
  model is in the loop, so "attack success" means the message or tool call got
  through, not that a model would have obeyed it.
