"""Project-level Ed25519 signatures, verified against an external pinned key."""

import base64
import binascii
import hashlib
import json
from pathlib import Path

from cryptography.exceptions import InvalidSignature, UnsupportedAlgorithm
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey

from sdk.exceptions import ContractError, SignatureError
from sdk.manifest import canonical_json, verify_manifest
from sdk.schema import package_file, read_json


def _external_key(package_dir: Path, key_path: Path) -> None:
    if key_path.resolve().is_relative_to(package_dir.resolve()):
        raise SignatureError("The signing/trusted key must be outside the Skill package")


def _load_key(path: Path, *, private: bool):
    try:
        data = path.read_bytes()
        key = (serialization.load_pem_private_key(data, password=None) if private
               else serialization.load_pem_public_key(data))
    except (OSError, ValueError, TypeError, UnsupportedAlgorithm) as exc:
        raise SignatureError(f"Cannot load Ed25519 key: {path.name}") from exc
    expected = Ed25519PrivateKey if private else Ed25519PublicKey
    if not isinstance(key, expected):
        raise SignatureError("Only Ed25519 keys are supported")
    return key


def key_id(public_key: Ed25519PublicKey) -> str:
    raw = public_key.public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    return hashlib.sha256(raw).hexdigest()


def generate_keypair(private_path: str | Path, public_path: str | Path) -> str:
    """Create development keys without overwriting an existing identity."""
    private_path, public_path = Path(private_path), Path(public_path)
    if private_path.resolve() == public_path.resolve() or private_path.exists() or public_path.exists():
        raise SignatureError("Key destinations must be distinct new files")
    key = Ed25519PrivateKey.generate()
    private_bytes = key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                     serialization.NoEncryption())
    public_bytes = key.public_key().public_bytes(serialization.Encoding.PEM,
                                                serialization.PublicFormat.SubjectPublicKeyInfo)
    try:
        private_path.parent.mkdir(parents=True, exist_ok=True)
        public_path.parent.mkdir(parents=True, exist_ok=True)
        with private_path.open("xb") as stream:
            stream.write(private_bytes)
        try:
            with public_path.open("xb") as stream:
                stream.write(public_bytes)
        except OSError:
            private_path.unlink()
            raise
    except OSError as exc:
        raise SignatureError("Cannot create key files") from exc
    return key_id(key.public_key())


def sign_package(package_dir: str | Path, private_key_path: str | Path) -> dict:
    """Sign an already sealed package; never repair or regenerate its manifest."""
    package_dir, private_key_path = Path(package_dir), Path(private_key_path)
    _external_key(package_dir, private_key_path)
    manifest = verify_manifest(package_dir)
    key = _load_key(private_key_path, private=True)
    signature = {
        "algorithm": "Ed25519",
        "key_id": key_id(key.public_key()),
        "signature": base64.b64encode(key.sign(canonical_json(manifest))).decode("ascii"),
    }
    target = package_dir / "manifest.sig"
    # verify_manifest already rejects linked resources, including manifest.sig.
    try:
        target.write_text(json.dumps(signature, indent=2) + "\n", encoding="utf-8")
    except OSError as exc:
        raise SignatureError("Cannot write manifest.sig") from exc
    return {"status": "passed", "algorithm": "Ed25519", "key_id": signature["key_id"]}


def verify_signature(package_dir: str | Path, trusted_public_key: str | Path) -> dict:
    package_dir, trusted_public_key = Path(package_dir), Path(trusted_public_key)
    _external_key(package_dir, trusted_public_key)
    manifest = verify_manifest(package_dir)
    key = _load_key(trusted_public_key, private=False)
    try:
        envelope = read_json(package_file(package_dir, "manifest.sig"))
    except ContractError as exc:
        raise SignatureError("Missing or malformed manifest.sig") from exc
    if (not isinstance(envelope, dict)
            or set(envelope) != {"algorithm", "key_id", "signature"}
            or envelope["algorithm"] != "Ed25519"
            or envelope["key_id"] != key_id(key)
            or not isinstance(envelope["signature"], str)):
        raise SignatureError("Signature envelope does not match the trusted Ed25519 key")
    try:
        signature = base64.b64decode(envelope["signature"], validate=True)
        key.verify(signature, canonical_json(manifest))
    except (InvalidSignature, ValueError, binascii.Error) as exc:
        raise SignatureError("Ed25519 signature verification failed") from exc
    return {"status": "passed", "algorithm": "Ed25519", "key_id": key_id(key)}
