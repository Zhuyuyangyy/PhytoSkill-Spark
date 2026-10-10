import base64
import json
import shutil

import pytest

from registry import SkillRegistry
from registry.signer import (generate_keypair, key_id, load_public_key,
                            sign_package, verify_signature)
from sdk.exceptions import ManifestError, RegistryError, SignatureError
from sdk.manifest import seal_manifest, verify_manifest
from sdk.schema import read_json


def test_signature_verifies_with_external_pinned_key(package, keys):
    sign_package(package, keys[0])
    result = verify_signature(package, keys[1])
    assert result["status"] == "passed"
    assert result["algorithm"] == "Ed25519"
    assert len(result["key_id"]) == 64


def test_wrong_key_and_forged_signature_fail(package, keys, tmp_path):
    sign_package(package, keys[0])
    other_private, other_public = tmp_path / "other.private.pem", tmp_path / "other.public.pem"
    generate_keypair(other_private, other_public)
    with pytest.raises(SignatureError):
        verify_signature(package, other_public)
    path = package / "manifest.sig"
    envelope = json.loads(path.read_text(encoding="utf-8"))
    envelope["signature"] = base64.b64encode(bytes(64)).decode("ascii")
    path.write_text(json.dumps(envelope), encoding="utf-8")
    with pytest.raises(SignatureError, match="verification failed"):
        verify_signature(package, keys[1])
    with pytest.raises(RegistryError):
        SkillRegistry(package.parent, trusted_public_key=keys[1]).discover()


@pytest.mark.parametrize("change", ["metadata", "code"])
def test_rehashing_tampered_package_cannot_bypass_signature(package, keys, change):
    sign_package(package, keys[0])
    manifest = verify_manifest(package)
    if change == "metadata":
        manifest["description"] = "attacker rehashed metadata"
    else:
        (package / "skill.py").write_text("raise RuntimeError('changed code')", encoding="utf-8")
    seal_manifest(package, manifest)
    verify_manifest(package)  # A self-consistent hash alone authenticates nothing.
    with pytest.raises(SignatureError, match="verification failed"):
        verify_signature(package, keys[1])


def test_signing_and_verification_never_repair_modified_files(package, keys):
    sign_package(package, keys[0])
    before = (package / "manifest.json").read_bytes()
    (package / "extra.txt").write_text("added", encoding="utf-8")
    with pytest.raises(ManifestError):
        sign_package(package, keys[0])
    with pytest.raises(ManifestError):
        verify_signature(package, keys[1])
    assert (package / "manifest.json").read_bytes() == before


def test_package_cannot_supply_its_own_trust_root(package, keys):
    internal = package / "trust.public.pem"
    internal.write_bytes(keys[1].read_bytes())
    seal_manifest(package, json.loads((package / "manifest.json").read_text(encoding="utf-8")))
    sign_package(package, keys[0])
    with pytest.raises(SignatureError, match="outside"):
        verify_signature(package, internal)


@pytest.mark.parametrize("raw", ["{}", '"string"', '{"algorithm":"Ed25519"}', 'not-json'])
def test_invalid_signature_envelopes_fail_closed(package, keys, raw):
    (package / "manifest.sig").write_text(raw, encoding="utf-8")
    with pytest.raises(SignatureError):
        verify_signature(package, keys[1])


def test_keygen_never_overwrites_existing_identity(keys):
    before = keys[0].read_bytes()
    with pytest.raises(SignatureError, match="new files"):
        generate_keypair(*keys)
    assert keys[0].read_bytes() == before


# ── the publisher pin: a swapped anchor must not become the new trust root ───


def _write_config(tmp_path, *, key_id=None, require_signature=True):
    config = {"version": 1, "skills_dir": "skills",
              "require_signature": require_signature,
              "trusted_public_key": "trust/publisher.public.pem"}
    if key_id is not None:
        config["trusted_key_id"] = key_id
    path = tmp_path / "registry.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    return path


def test_the_pinned_publisher_id_accepts_the_real_anchor(package, keys, tmp_path):
    from registry.signer import key_id, load_public_key

    sign_package(package, keys[0])
    config = _write_config(tmp_path, key_id=key_id(load_public_key(keys[1])))
    registry = SkillRegistry.from_config(config)
    assert registry.discover()[0]["manifest"]["name"] == package.name


def test_a_swapped_trust_anchor_is_refused_by_the_pinned_publisher_id(package, keys, tmp_path):
    """A valid Ed25519 key that is not the publisher's must not quietly become
    the new trust root — the pin fails loudly instead."""
    from registry.signer import key_id, load_public_key

    sign_package(package, keys[0])
    config = _write_config(tmp_path, key_id=key_id(load_public_key(keys[1])))
    intruder_private = tmp_path / "intruder.private.pem"
    intruder_public = tmp_path / "intruder.public.pem"
    generate_keypair(intruder_private, intruder_public)
    (tmp_path / "trust" / "publisher.public.pem").write_bytes(intruder_public.read_bytes())
    with pytest.raises(RegistryError, match="pinned publisher id"):
        SkillRegistry.from_config(config)


def test_a_package_signed_by_a_valid_non_publisher_key_is_refused(package, keys, tmp_path):
    """The acceptance criterion: an untrusted package — a perfectly valid
    signature from the wrong publisher — never enters the registry."""
    other_private = tmp_path / "other.private.pem"
    other_public = tmp_path / "other.public.pem"
    generate_keypair(other_private, other_public)
    sign_package(package, other_private)
    with pytest.raises(RegistryError):
        SkillRegistry(package.parent, trusted_public_key=keys[1]).discover()


def test_the_trust_anchor_may_not_live_inside_the_skills_directory(package, keys):
    """An anchor a package could replace is not an anchor."""
    inside = package.parent / "publisher.public.pem"
    inside.write_bytes(keys[1].read_bytes())
    with pytest.raises(RegistryError, match="outside"):
        SkillRegistry(package.parent, trusted_public_key=inside).discover()


def test_the_gate_names_the_key_that_verified_the_package(package, keys):
    """"Verified" alone does not say whose signature it was; the operator needs
    the key id to compare against the publisher fingerprint held out of band."""
    from registry.signer import key_id, load_public_key
    from shield.gate import gate_package

    sign_package(package, keys[0])
    result = gate_package(package, project_signature_key=keys[1])
    check = next(item for item in result.checks if item.name == "project_signature")
    assert check.status == "passed"
    assert key_id(load_public_key(keys[1])) in check.detail


def test_a_demo_key_cannot_publish(package, keys, tmp_path):
    """A key generated for a demo run signs validly and is still refused: the
    demo key is not the publisher, and nothing in the package can say otherwise."""
    demo_private = tmp_path / "demo.private.pem"
    demo_public = tmp_path / "demo.public.pem"
    generate_keypair(demo_private, demo_public)
    sign_package(package, demo_private)
    verification = verify_signature(package, demo_public)
    assert verification["status"] == "passed"  # valid — and still not publishable
    with pytest.raises(RegistryError):
        SkillRegistry(package.parent, trusted_public_key=keys[1]).discover()



def test_a_key_swapped_after_initialisation_cannot_change_what_the_registry_trusts(
        package, keys, tmp_path):
    """The publisher pin must bind for the registry's lifetime, not just at
    construction.

    Sequence: construct with the real publisher key (the pin passes), swap the
    anchor file for another valid Ed25519 key, then discover. A package signed
    by the swapped key is refused, and the original package still verifies —
    the registry trusts the key it was constructed with, not the file's later
    contents.
    """
    sign_package(package, keys[0])
    config = _write_config(tmp_path, key_id=key_id(load_public_key(keys[1])))
    registry = SkillRegistry.from_config(config)
    assert registry.discover()[0]["manifest"]["name"] == package.name

    # Swap the anchor file underneath the live registry.
    intruder_private = tmp_path / "late.private.pem"
    intruder_public = tmp_path / "late.public.pem"
    generate_keypair(intruder_private, intruder_public)
    (tmp_path / "trust" / "publisher.public.pem").write_bytes(intruder_public.read_bytes())

    # A package signed by the swapped key is refused by the live registry.
    intruder_package = tmp_path / "skills" / "intruder_skill"
    shutil.copytree(package, intruder_package,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "manifest.sig"))
    # The package dir holds the sealed manifest; seal_manifest regenerates it
    # from the metadata fields, so the inventory and hash are dropped first.
    metadata = read_json(intruder_package / "manifest.json")
    metadata.pop("files", None)
    metadata.pop("manifest_sha256", None)
    metadata["name"] = "intruder_skill"
    instructions = intruder_package / "SKILL.md"
    instructions.write_text(instructions.read_text(encoding="utf-8").replace(
        "name: contract-probe", "name: intruder-skill"), encoding="utf-8")
    seal_manifest(intruder_package, metadata)
    sign_package(intruder_package, intruder_private)
    with pytest.raises(RegistryError):
        registry.discover()

    # The publisher's own package still verifies through the same registry:
    # verify_entry re-checks against the key held since construction.
    record = registry.verify_entry(package.name)
    assert record["verification"]["signature"]["status"] == "passed"
    assert record["verification"]["signature"]["key_id"] == registry.trusted_key_id
