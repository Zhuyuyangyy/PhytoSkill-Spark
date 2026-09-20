"""Verified package discovery; discovery never imports a Skill's Python code."""

from registry.loader import SkillRegistry

__all__ = ["SkillRegistry"]
