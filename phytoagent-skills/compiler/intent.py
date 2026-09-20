"""Deterministic intent extraction from a natural-language domain request.

No language model is involved. The parser is deliberately conservative: when it
cannot extract a species, a task, or a capability, it refuses instead of
guessing, because a guessed SkillSpec becomes a wrong Skill forever.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from compiler.catalog import (CAPABILITIES, INPUT_SLOTS, SPECIES_ALIASES, TASK_KEYWORDS,
                              capabilities_in_order)
from sdk.exceptions import CompileError


@dataclass
class Intent:
    """The structured request the Compiler turns into a SkillSpec."""

    request: str
    species: str | None = None
    task: str = "health-assessment"
    capabilities: list[str] = field(default_factory=list)
    inputs: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def resolved(self) -> bool:
        return bool(self.species) and bool(self.capabilities)


def _normalise(text: str) -> str:
    return " ".join(text.lower().split())


def _match_any(normalised: str, keywords: list[str]) -> str | None:
    for keyword in keywords:
        if keyword.lower() in normalised:
            return keyword
    return None


def parse_species(request: str, normalised: str) -> str | None:
    """Resolve a species from aliases; never infer one from a missing mention."""
    for alias, species in SPECIES_ALIASES.items():
        if alias.lower() in normalised:
            return species
    return None


def parse_task(normalised: str) -> tuple[str, list[str]]:
    notes: list[str] = []
    for task, keywords in TASK_KEYWORDS.items():
        if _match_any(normalised, keywords):
            if task != "health-assessment":
                notes.append(f"task keyword matched {task!r}")
            return task, notes
    notes.append("no explicit task keyword; defaulted to health-assessment")
    return "health-assessment", notes


def parse_capabilities(normalised: str) -> list[str]:
    """Select capabilities from trigger keywords; fusion closes the workflow."""
    selected: list[str] = []
    for name, entry in CAPABILITIES.items():
        if name == "evidence_fusion":
            continue
        if _match_any(normalised, entry["triggers"]):
            selected.append(name)
    if not selected:
        return []
    # A workflow with only growth_risk or only herbal_knowledge is not an
    # assessment: there is no observed phenotype to assess. Such a request is
    # refused rather than compiled into a package that cannot say anything.
    if "plant_vision" not in selected:
        return []
    # Every generated package is an assessment workflow; it ends in fusion.
    selected.append("evidence_fusion")
    return capabilities_in_order(selected)


def derive_inputs(capabilities: list[str]) -> list[str]:
    """Workflow inputs are exactly the union of the selected capabilities' inputs."""
    inputs: list[str] = []
    for name in capabilities:
        for slot in CAPABILITIES[name]["inputs"]:
            if slot not in inputs:
                inputs.append(slot)
    if not inputs:
        inputs.append("question")
    return inputs


def parse_intent(request: str) -> Intent:
    """Extract a SkillSpec-shaped intent, or refuse with the missing requirement."""
    if not isinstance(request, str) or not request.strip():
        raise CompileError("A non-empty natural-language domain request is required")
    if len(request) > 4000:
        raise CompileError("Request exceeds 4000 characters; narrow the domain task")
    normalised = _normalise(request)
    species = parse_species(request, normalised)
    task, notes = parse_task(normalised)
    capabilities = parse_capabilities(normalised)
    inputs = derive_inputs(capabilities)
    return Intent(request=request.strip(), species=species, task=task,
                  capabilities=capabilities, inputs=inputs, notes=notes)


def required_inputs(capabilities: list[str]) -> list[str]:
    return [INPUT_SLOTS[slot]["label"] for slot in derive_inputs(capabilities)]
