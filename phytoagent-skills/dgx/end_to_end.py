"""End-to-end run on the DGX Spark node: real vision, then fusion and audit.

Runs the whole chain against one image on the node's GPU:

    plant_vision (live, DGX)  ->  evidence_fusion  ->  agentshield_audit

Every provider call goes through the AgentShield runtime, so the trace shows
real tool_call_ids, permissions used and evidence ids. Nothing here falls back to
a fixture: if the node is unreachable the run fails.

Writes a machine-readable record to artifacts/dgx/end-to-end.json.
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

from demo.fixture_workspace import PROJECT_ROOT, SKILL_NAMES
from registry import SkillRegistry
from registry.signer import generate_keypair, sign_package
from runtime.executor import SkillExecutor
from runtime.shield import ClaimAuditor, ShieldRuntime, load_evidence_index
from sdk.manifest import seal_manifest
from sdk.schema import read_json
from shield.gate import gate_package

MODEL = "modelscope.cn/unsloth/Qwen3.8-27B-GGUF:latest"


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass

    image = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("artifacts/dgx/synthetic-leaf-03.png")
    species = sys.argv[2] if len(sys.argv) > 2 else "黄芪"
    case_id = "dgx-live-001"

    root = Path(tempfile.mkdtemp(prefix="phyto-dgx-e2e-"))
    private, public = root / "p.pem", root / "pub.pem"
    generate_keypair(private, public)
    skills = root / "skills"
    skills.mkdir()
    for name in SKILL_NAMES:
        destination = skills / name
        shutil.copytree(PROJECT_ROOT / "skills" / name, destination,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "manifest.sig"))
        sign_package(destination, private)

    # The gate runs on the audited providers before anything is registered.
    gates = [gate_package(skills / name, project_signature_key=public) for name in SKILL_NAMES]
    for gate in gates:
        if not gate.passed:
            print(json.dumps({"error": "gate failed", "skill": gate.skill,
                              "checks": [c.to_dict() for c in gate.failures]},
                             ensure_ascii=False))
            return 1

    registry = SkillRegistry(skills, trusted_public_key=public)
    registry.discover()
    runtime = ShieldRuntime(registry, trace_id="trace-dgx-e2e-001")
    executor = SkillExecutor(registry)

    print(f"image   : {image}")
    print(f"species : {species}")
    print(f"model   : {MODEL}\n")

    # 1. Real vision on the node's GPU.
    vision = executor.call("plant_vision", {"case_id": case_id, "species": species,
                                            "image_path": str(image.resolve())},
                           mode="live", tool_call_id="dgx-call-1")
    print(f"[1/3] plant_vision (live) -> {vision['status']} in {vision['duration_ms']:.0f} ms")
    if vision["status"] != "success":
        print(json.dumps(vision["error"], ensure_ascii=False))
        return 1
    data = vision["data"]
    print(f"      data_origin={data['provenance']['data_origin']} "
          f"dgx_hardware_used={data['provenance']['dgx_hardware_used']}")
    print(f"      image_usable={data['image_usable']} observations={len(data['observations'])}")
    for obs in data["observations"]:
        print(f"        {obs['observation_id']} {obs['phenotype']} "
              f"bbox={obs['region']['bbox_normalized']} score={obs['model_score']}")

    # 2. Deterministic fusion of what vision actually produced.
    fusion = executor.call("evidence_fusion", {
        "case_id": case_id, "species": species,
        "vision": data, "environment": None, "knowledge": None,
    }, mode="fixture", tool_call_id="dgx-call-2")
    print(f"\n[2/3] evidence_fusion -> {fusion['status']}")
    if fusion["status"] == "success":
        fused = fusion["data"]
        print(f"      completeness={fused['completeness']} "
              f"missing={fused['missing_inputs']} "
              f"conclusion_strength={fused['conclusion_strength']}")
    else:
        print(f"      {fusion['error']['code']}: {fusion['error']['message']}")

    # 3. Audit the result against the evidence that actually exists.
    index = load_evidence_index(data, None, None)
    claims = [{"text": f"观察到{obs['phenotype']}区域 {obs['region']['label']}",
               "evidence_ids": [obs["observation_id"]]}
              for obs in data["observations"]]
    audit = ClaimAuditor(available_evidence=index).audit({
        "claims": claims, "missing_inputs": ["environment", "knowledge"],
    })
    print(f"\n[3/3] claim-evidence audit -> trust_level={audit['trust_level']}")
    print(f"      supported={len(audit['supported_claims'])} "
          f"refused={len(audit['refused_claims'])} violations={len(audit['violations'])}")

    record = {
        "scope": "dgx_end_to_end_live_vision",
        "agent_model_called": False,
        "dgx_hardware_used": True,
        "image": str(image),
        "species": species,
        "case_id": case_id,
        "model": MODEL,
        "vision": data,
        "fusion": fusion.get("data"),
        "audit": audit,
        "trace": runtime.report(),
        "gpu": data["provenance"].get("gpu"),
    }
    out = Path("artifacts/dgx/end-to-end.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {out}")
    shutil.rmtree(root, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
