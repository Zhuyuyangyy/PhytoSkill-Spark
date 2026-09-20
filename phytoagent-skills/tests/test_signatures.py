import base64
import json

import pytest

from registry import SkillRegistry
from registry.signer import generate_keypair, sign_package, verify_signature
from sdk.exceptions import ManifestError, RegistryError, SignatureError
from sdk.manifest import seal_manifest, verify_manifest


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
