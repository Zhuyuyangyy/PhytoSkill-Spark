"""Temporary copies of the four local Skills, signed with disposable demo keys."""

from contextlib import contextmanager
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory

from registry import SkillRegistry
from registry.signer import generate_keypair, sign_package
from sdk.manifest import verify_manifest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SKILL_NAMES = ("plant_vision", "growth_risk", "herbal_knowledge", "evidence_fusion",
               "agentshield_audit")


@contextmanager
def fixture_registry():
    with TemporaryDirectory(prefix="phyto-domain-fixture-") as temporary:
        root = Path(temporary)
        private, public = root / "trust/private.pem", root / "trust/public.pem"
        generate_keypair(private, public)
        for name in SKILL_NAMES:
            source = PROJECT_ROOT / "skills" / name
            verify_manifest(source)
            destination = root / "skills" / name
            shutil.copytree(source, destination, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "manifest.sig"))
            sign_package(destination, private)
        registry = SkillRegistry(root / "skills", trusted_public_key=public)
        registry.discover()
        yield registry
