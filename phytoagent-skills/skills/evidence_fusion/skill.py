"""Deterministic evidence linkage, with no planner or language model."""

from sdk import BaseSkill, ContractError


def unique_ids(items: list[dict], key: str) -> list[str]:
    values = [item[key] for item in items]
    if len(values) != len(set(values)):
        raise ContractError(f"Duplicate evidence identifier: {key}")
    return values


class EvidenceFusionSkill(BaseSkill):
    def run(self, payload: dict, *, mode: str) -> dict:
        sources = {name: payload[name] for name in ("vision", "environment", "knowledge")}
        missing = [name for name, source in sources.items() if source is None]
        limitations = ["输入为合成fixture，仅验证证据连接；不能用于真实植物诊断。",
                       "共同出现不等于因果关系；不合成疾病概率或药效下降比例。"]
        for name, source in sources.items():
            if source is None:
                continue
            if source["case_id"] != payload["case_id"] or source["species"] != payload["species"]:
                raise ContractError(f"Case/species mismatch in {name}; do not fuse unrelated observations")
            limitations.extend(source["limitations"])
        observations = sources["vision"]["observations"] if sources["vision"] else []
        factors = sources["environment"]["factors"] if sources["environment"] else []
        evidence = sources["knowledge"]["evidence"] if sources["knowledge"] else []
        observation_ids = unique_ids(observations, "observation_id")
        factor_ids = unique_ids(factors, "factor_id")
        evidence_ids = unique_ids(evidence, "evidence_id")
        chain, linked = [], set()
        for observation in observations:
            left, top, right, bottom = observation["region"]["bbox_normalized"]
            if left >= right or top >= bottom:
                raise ContractError("Bounding box must have positive width and height")
            matches = [item["evidence_id"] for item in evidence
                       if observation["phenotype"] in item["supports_phenotypes"]]
            linked.update(matches)
            chain.append({"observation_id": observation["observation_id"],
                          "phenotype": observation["phenotype"],
                          "environment_factor_ids": factor_ids,
                          "knowledge_evidence_ids": matches,
                          "relationship": "co_occurrence_only"})
        completeness = "insufficient" if len(missing) == 3 else "partial" if missing else "complete"
        summary = ("三个来源的合成结果已对齐，可追溯至各自ID；不构成真实诊断。" if not missing
                   else "部分来源缺失，保留可用结果与缺失列表；不能形成完整诊断。")
        if missing:
            limitations.append("缺失来源：" + ", ".join(missing))
        return {
            "status": "success", "case_id": payload["case_id"], "species": payload["species"],
            "provenance": {"data_origin": "synthetic_fixture", "skill": self.name,
                           "version": self.version, "model_called": False},
            "summary": summary, "completeness": completeness,
            "conclusion_strength": "synthetic_only", "missing_inputs": missing,
            "source_ids": {"vision": observation_ids, "environment": factor_ids, "knowledge": evidence_ids},
            "evidence_chain": chain, "unlinked_evidence_ids": [i for i in evidence_ids if i not in linked],
            "recommendations": ["用真实图像、带单位的测量和可核实文献替换fixture后再研判。",
                                "复核物种、生育期、土壤水分及采样时间；必要时由植保人员鉴定。"],
            "limitations": list(dict.fromkeys(limitations)),
        }
