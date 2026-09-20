"""Synthetic SDK example; no plant or model inference."""

from sdk import BaseSkill


class ContractProbeSkill(BaseSkill):
    def run(self, payload: dict, *, mode: str) -> dict:
        return {"status": "success", "mode": mode, "echo": payload["message"],
                "data_origin": "synthetic_fixture"}
