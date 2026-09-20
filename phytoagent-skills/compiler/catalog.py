"""The audited capability catalogue the Compiler is allowed to compose from.

The Compiler never invents behaviour. It selects entries from this catalogue,
renders declarative configuration, and emits a thin executor. Any capability not
listed here is rejected by the compile gate instead of being generated.
"""

from __future__ import annotations

# Ordered deliberately: the order value is also the workflow execution order.
CAPABILITIES: dict[str, dict] = {
    "plant_vision": {
        "provider_skill": "plant_vision",
        "order": 1,
        "title": "PlantVision",
        "inputs": ["image"],
        "input_slots": {"case_id": "case_id", "species": "species", "image_path": "image"},
        "outputs": ["observations"],
        "capabilities_field": ["plant_vision"],
        "triggers": [
            "叶片", "叶缘", "图片", "图像", "照片", "拍照", "黄化", "斑点", "萎蔫", "表型",
            "image", "leaf", "photo", "vision",
        ],
        "negative_triggers": ["知识问答", "人体疾病", "开处方"],
        "default_permissions": ["case_workspace:read"],
        "description": "药用植物图片的异常表型观察与区域定位。",
        "failure_mode": "缺少可用图片时记录 image_quality_gate FAILED，不猜测表型。",
    },
    "growth_risk": {
        "provider_skill": "growth_risk",
        "order": 2,
        "title": "GrowthRisk",
        "inputs": ["telemetry"],
        "input_slots": {"case_id": "case_id", "species": "species", "telemetry": "telemetry"},
        "outputs": ["factors"],
        "capabilities_field": ["growth_risk"],
        "triggers": [
            "温湿度", "温度", "湿度", "土壤", "ph", "npk", "氮", "磷", "钾", "生育期",
            "环境", "测量", "telemetry", "temperature", "humidity", "moisture",
        ],
        "negative_triggers": ["天气闲聊"],
        "default_permissions": ["case_workspace:read"],
        "description": "带单位环境测量的因素核对与缺失上下文提示。",
        "failure_mode": "缺少带单位测量时记录 missing_context，不用空气湿度冒充土壤水分。",
    },
    "herbal_knowledge": {
        "provider_skill": "herbal_knowledge",
        "order": 3,
        "title": "HerbalKnowledge",
        "inputs": ["knowledge_query"],
        "input_slots": {"case_id": "case_id", "species": "species", "query": "question"},
        "outputs": ["evidence"],
        "capabilities_field": ["herbal_knowledge"],
        "triggers": [
            "知识库", "文献", "资料", "本草", "语料", "检索", "来源", "证据来源",
            "knowledge", "reference", "corpus", "rag",
        ],
        "negative_triggers": ["编造论文", "开处方"],
        "default_permissions": ["corpus:read"],
        "description": "本地语料检索，返回文献片段、位置与证据 ID。",
        "failure_mode": "检索不到可核实来源时返回空 evidence 并记 limitations，不编造引用。",
    },
    "evidence_fusion": {
        "provider_skill": "evidence_fusion",
        "order": 4,
        "title": "EvidenceFusion",
        # Fusion reads no request field of its own: it consumes the provider
        # results the earlier steps produced. Those arrive under result keys, not
        # request slots, so they are deliberately not in input_slots.
        "inputs": [],
        "input_slots": {"case_id": "case_id", "species": "species"},
        "result_slots": {"vision": "vision", "environment": "environment",
                         "knowledge": "knowledge"},
        "outputs": ["evidence_chain"],
        "capabilities_field": ["evidence_fusion"],
        "triggers": ["研判", "融合", "综合", "评估", "结论", "assess", "fusion", "conclusion"],
        "negative_triggers": ["原始图片直接触发"],
        "default_permissions": ["package:read"],
        "description": "确定性连接多来源证据，保留缺失项与未连接证据。",
        "failure_mode": "来源案例或物种不一致时拒绝融合，不借用其他案例结果。",
    },
}

# Every generated package declares these; the Compiler cannot weaken them.
FIXED_POLICIES = {
    "evidence_policy": "every_claim_requires_evidence",
    "refusal_policy": "insufficient_evidence",
    "network": "deny",
}

# Known species used for deterministic intent extraction. An unknown species is a
# hard refusal: the Compiler must not guess a plant name.
SPECIES_ALIASES: dict[str, str] = {
    "黄芪": "黄芪",
    "huangqi": "黄芪",
    "astragalus": "黄芪",
    "人参": "人参",
    "renshen": "人参",
    "ginseng": "人参",
    "当归": "当归",
    "danggui": "当归",
    "甘草": "甘草",
    "gancao": "甘草",
    "licorice": "甘草",
    "白术": "白术",
    "baizhu": "白术",
    "茯苓": "茯苓",
    "fuling": "茯苓",
    "川芎": "川芎",
    "chuanxiong": "川芎",
    "金银花": "金银花",
    "jinyinhua": "金银花",
    "honeysuckle": "金银花",
    "枸杞": "枸杞",
    "gouqi": "枸杞",
    "goji": "枸杞",
}

TASK_KEYWORDS: dict[str, list[str]] = {
    "health-assessment": [
        "健康", "研判", "诊断", "异常", "评估", "assessment", "health", "diagnos",
        "黄化", "病害", "病因", "胁迫",
    ],
    "phenotype-survey": ["表型", "普查", "survey", "phenotype"],
    "environment-review": ["环境", "复核", "review", "environment"],
    "knowledge-brief": ["知识", "资料", "综述", "brief", "knowledge"],
}

# Package names must be ASCII and lowercase, so a species resolves to a stable
# latin slug. Unknown species are refused; a transliteration is never guessed.
SPECIES_SLUGS: dict[str, str] = {
    "黄芪": "huangqi",
    "人参": "renshen",
    "当归": "danggui",
    "甘草": "gancao",
    "白术": "baizhu",
    "茯苓": "fuling",
    "川芎": "chuanxiong",
    "金银花": "jinyinhua",
    "枸杞": "gouqi",
}

# Slot -> the request input the slot is fed from. This is the only place a
# natural-language request is mapped onto a workflow input.
INPUT_SLOTS = {
    "image": {"label": "image", "kind": "image", "required": True},
    "telemetry": {"label": "telemetry", "kind": "telemetry", "required": False},
    "question": {"label": "question", "kind": "text", "required": False},
    "knowledge_query": {"label": "knowledge_query", "kind": "text", "required": False},
}

NEGATIVE_EVAL_TEMPLATES: dict[str, dict] = {
    "missing_input": {
        "expect": "missing source recorded; trust_level LIMITED at most",
        "per_capability": False,
    },
    "cross_case_mismatch": {
        "expect": "ContractError; no cross-case fusion",
        "per_capability": False,
    },
    "unsupported_claim": {
        "expect": "claim without evidence id refused",
        "per_capability": False,
    },
    "permission_violation": {
        "expect": "middleware blocks the call before execution",
        "per_capability": False,
    },
    "unverifiable_source": {
        "expect": "no fabricated citation; empty evidence is reported",
        "per_capability": False,
    },
    "blurry_image": {
        "expect": "image quality gate FAILED; cause assessment REFUSED",
        "per_capability": True,
    },
}

# Permissions the Compiler is allowed to grant, in the canonical order it emits.
ALLOWED_FILESYSTEM_PERMISSIONS = ("case_workspace:read", "corpus:read", "package:read")


def capabilities_in_order(names: list[str]) -> list[str]:
    """Return the selected capabilities in execution order, de-duplicated."""
    unique = list(dict.fromkeys(names))
    return sorted(unique, key=lambda name: CAPABILITIES[name]["order"])


def is_composable(names: list[str]) -> bool:
    """A package needs at least one producer plus the fusion capability."""
    producers = [name for name in names if name != "evidence_fusion"]
    return bool(producers) and "evidence_fusion" in names


def union_permissions(names: list[str]) -> list[str]:
    """Least-privilege union of the selected capabilities' default permissions."""
    required = {permission for name in names for permission in CAPABILITIES[name]["default_permissions"]}
    required.add("package:read")
    return [permission for permission in ALLOWED_FILESYSTEM_PERMISSIONS if permission in required]
