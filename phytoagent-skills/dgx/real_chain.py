"""Full real chain over the herb image set: vision -> knowledge -> fusion -> audit.

One case per image, run end to end through the AgentShield runtime:

    plant_vision (herb, DGX)   ->  herbal_knowledge (corpus)  ->  evidence_fusion
                                                                 ->  claim audit

The vision leg is real GPU inference. The knowledge leg queries the vendored local
corpus for the species and the observed phenotypes. Fusion joins them by id, and
the auditor refuses anything without an evidence id.

Writes artifacts/dgx/real-chain.json.
"""

import json
import shutil
import sys
import tempfile
from pathlib import Path

from demo.fixture_workspace import PROJECT_ROOT, SKILL_NAMES
from registry import SkillRegistry
from registry.signer import generate_keypair, sign_package
from runtime.shield import ClaimAuditor, ShieldRuntime, load_evidence_index
from sdk.exceptions import SkillError
from sdk.schema import read_json

# The vendored corpus is TCM syndrome knowledge. It is asked for the species and
# the phenotype the vision leg reported, and it is allowed to come back empty.
KNOWLEDGE_QUERY = "黄芪 药材 断面 色泽 霉变 虫蛀 质量"


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass
    directory = Path(sys.argv[1])
    species = sys.argv[2] if len(sys.argv) > 2 else "黄芪"

    root = Path(tempfile.mkdtemp(prefix="phyto-real-chain-"))
    private, public = root / "p.pem", root / "pub.pem"
    generate_keypair(private, public)
    skills = root / "skills"
    skills.mkdir()
    for name in SKILL_NAMES:
        destination = skills / name
        shutil.copytree(PROJECT_ROOT / "skills" / name, destination,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "manifest.sig"))
        sign_package(destination, private)

    registry = SkillRegistry(skills, trusted_public_key=public)
    registry.discover()
    runtime = ShieldRuntime(registry, trace_id="trace-real-chain-001")
    from runtime.executor import SkillExecutor
    executor = SkillExecutor(registry)

    cases = []
    images = sorted(directory.glob("*.jpg"))
    for index, image in enumerate(images, start=1):
        case_id = f"chain-{index:03d}"
        print(f"\n=== [{index}/{len(images)}] {image.name} ({case_id}) ===")

        vision = executor.call("plant_vision", {
            "case_id": case_id, "species": species, "image_path": str(image.resolve()),
        }, mode="herb", tool_call_id=f"{case_id}-vision")
        if vision["status"] != "success":
            print(f"  vision FAILED: {vision['error']['code']}")
            cases.append({"image": image.name, "case_id": case_id,
                          "vision": "failed", "error": vision["error"]})
            continue
        vision_data = vision["data"]
        phenotypes = sorted({o["phenotype"] for o in vision_data["observations"]})
        print(f"  vision   : usable={vision_data['image_usable']} "
              f"obs={len(vision_data['observations'])} phenotypes={phenotypes}")

        knowledge = executor.call("herbal_knowledge", {
            "case_id": case_id, "species": species, "query": KNOWLEDGE_QUERY,
        }, mode="corpus", tool_call_id=f"{case_id}-knowledge")
        if knowledge["status"] != "success":
            print(f"  knowledge FAILED: {knowledge['error']['code']}")
            knowledge_data = None
        else:
            knowledge_data = knowledge["data"]
            print(f"  knowledge: evidence={len(knowledge_data['evidence'])} "
                  f"origin={knowledge_data['provenance']['data_origin']}")

        fusion = executor.call("evidence_fusion", {
            "case_id": case_id, "species": species,
            "vision": vision_data,
            "environment": None,
            "knowledge": knowledge_data,
        }, mode="fixture", tool_call_id=f"{case_id}-fusion")
        if fusion["status"] != "success":
            print(f"  fusion FAILED: {fusion['error']['code']}: {fusion['error']['message']}")
            cases.append({"image": image.name, "case_id": case_id, "vision": "success",
                          "fusion": "failed", "error": fusion["error"]})
            continue
        fused = fusion["data"]
        print(f"  fusion   : completeness={fused['completeness']} "
              f"missing={fused['missing_inputs']}")

        # Audit the claims the chain can actually support.
        index = load_evidence_index(vision_data, None, knowledge_data)
        claims = []
        for obs in vision_data["observations"]:
            claims.append({"text": f"{obs['region']['label']}（{obs['phenotype']}）",
                           "evidence_ids": [obs["observation_id"]]})
        for item in (knowledge_data or {}).get("evidence", [])[:2]:
            claims.append({"text": f"语料记载：{item['source_title']}",
                           "evidence_ids": [item["evidence_id"]]})
        audit = ClaimAuditor(available_evidence=index).audit({
            "claims": claims, "missing_inputs": fused["missing_inputs"],
        })
        print(f"  audit    : trust={audit['trust_level']} "
              f"supported={len(audit['supported_claims'])} "
              f"refused={len(audit['refused_claims'])}")

        cases.append({
            "image": image.name, "case_id": case_id,
            "vision": {"image_usable": vision_data["image_usable"],
                       "observations": vision_data["observations"],
                       "provenance": vision_data["provenance"]},
            "knowledge": {"evidence": (knowledge_data or {}).get("evidence", []),
                          "provenance": (knowledge_data or {}).get("provenance")},
            "fusion": {"completeness": fused["completeness"],
                       "missing_inputs": fused["missing_inputs"],
                       "evidence_chain": fused["evidence_chain"]},
            "audit": audit,
        })

    record = {"scope": "dgx_real_chain", "species": species,
              "knowledge_query": KNOWLEDGE_QUERY,
              "images": len(images), "cases": cases,
              "trace": runtime.report()}
    out = Path("artifacts/dgx/real-chain.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {out}")
    shutil.rmtree(root, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
