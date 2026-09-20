"""Public errors callers may turn into structured tool failures."""


class SkillError(Exception):
    """Base error for the Skill SDK."""


class ContractError(SkillError):
    """A schema, input or output violates its declared contract."""


class ManifestError(SkillError):
    """A package is malformed or its file inventory changed."""


class SignatureError(ManifestError):
    """The package cannot be verified with the configured trusted key."""


class RegistryError(SkillError):
    """Discovery failed or a requested skill is unknown."""


class UnsupportedModeError(SkillError):
    """A skill does not implement the requested execution mode."""


class FixtureMismatchError(SkillError):
    """Synthetic data cannot answer a request for a different case."""


class CompileError(SkillError):
    """A request cannot be compiled into a constrained Skill, or the result is unsafe."""


class ShieldError(SkillError):
    """Base error for the AgentShield runtime."""


class PermissionViolation(ShieldError):
    """A tool call exceeded the permissions declared in the Skill package."""


class BudgetExceeded(ShieldError):
    """A session exceeded its declared call budget."""


class TraceError(ShieldError):
    """A tool call reached the runtime without the trace identifiers it requires."""
