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
