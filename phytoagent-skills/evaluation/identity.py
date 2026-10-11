"""Image identity and data independence for the holdout blind test.

The holdout is only a holdout if its images provably never touched the
development process. A role list inside one annotations file cannot show that:
it can show the roles do not overlap *in that file*, not that the images were
strangers to prompt-writing, calibration and model selection.

So identity lives outside the evaluation, in a manifest built when the data
arrives:

* every image's **SHA-256** — an exact duplicate is refused outright;
* a **perceptual hash** — a re-encode, crop or resize of a development image is
  reported, because "the same photograph under a new name" is the near-duplicate
  that splits a blind set silently;
* **provenance the collector records**: source, plant or batch, and the role
  assigned *before* anyone sees a model output.

The perceptual gate is deliberately modest: it catches the same photograph
re-encoded, not two different photographs of the same plant. That second kind
of overlap is what the source and batch fields exist for, and no hash can
substitute for the collector writing them down.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from evaluation.split import FreezeError

DEFAULT_IDENTITY = Path("artifacts/evaluations/development-identity.json")

# Hamming distance at or below which two perceptual hashes are reported as a
# near duplicate. 64-bit dHash: 5 bits is a re-encode or a mild crop; a
# different photograph of the same subject typically lands far above it.
NEAR_DUPLICATE_MAX_DISTANCE = 5


@dataclass(frozen=True)
class ImageIdentity:
    """What one image is, independent of any annotation."""

    path: str
    sha256: str
    bytes: int
    width: int | None
    height: int | None
    dhash: str

    def to_dict(self) -> dict:
        return {"path": self.path, "sha256": self.sha256, "bytes": self.bytes,
                "width": self.width, "height": self.height, "dhash": self.dhash}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dhash(path: Path) -> str:
    """A 64-bit difference hash: robust to re-encoding, resizing and mild crops.

    Implemented on a 9x8 grayscale reduction, compared row-wise. It is not a
    perceptual miracle — it answers "is this the same photograph", which is
    exactly the question a blind set needs answered before scoring.
    """
    from PIL import Image

    with Image.open(path) as image:
        # tobytes() over the "L" mode gives one byte per pixel, in row-major
        # order — stable across Pillow versions, unlike getdata().
        pixels = image.convert("L").resize((9, 8)).tobytes()
    bits = []
    for row in range(8):
        for column in range(8):
            left = pixels[row * 9 + column]
            right = pixels[row * 9 + column + 1]
            bits.append("1" if left > right else "0")
    return f"{int(''.join(bits), 2):016x}"


def image_identity(path: str | Path, *, root: Path | None = None) -> ImageIdentity:
    """Hash and measure one image. A file that cannot be read is refused."""
    path = Path(path)
    if not path.is_file():
        raise FreezeError(f"image not found: {path}")
    width = height = None
    try:
        from PIL import Image

        with Image.open(path) as image:
            width, height = image.size
    except Exception:  # noqa: BLE001 - dimensions are informative, identity is the hash
        pass
    return ImageIdentity(
        path=str(path.relative_to(root)) if root else str(path),
        sha256=_sha256(path), bytes=path.stat().st_size,
        width=width, height=height, dhash=_dhash(path))


def _hamming(left: str, right: str) -> int:
    return bin(int(left, 16) ^ int(right, 16)).count("1")


def scan_directory(directory: str | Path, *, root: Path | None = None,
                   patterns: tuple[str, ...] = ("*.jpg", "*.jpeg", "*.png")) -> list[ImageIdentity]:
    """Identity every image in a directory (non-recursive by default)."""
    directory = Path(directory)
    if not directory.is_dir():
        raise FreezeError(f"not a directory: {directory}")
    paths = sorted(path for pattern in patterns for path in directory.glob(pattern))
    if not paths:
        raise FreezeError(f"no images found in {directory}")
    return [image_identity(path, root=root) for path in paths]


def write_identity_manifest(identities: list[ImageIdentity], path: str | Path, *,
                            note: str = "") -> dict:
    manifest = {
        "schema_version": 1,
        "note": note,
        "written_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "count": len(identities),
        "images": [identity.to_dict() for identity in identities],
    }
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8")
    return manifest


def load_identity_manifest(path: str | Path) -> dict:
    path = Path(path)
    if not path.is_file():
        raise FreezeError(f"no identity manifest at {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or "images" not in manifest:
        raise FreezeError(f"malformed identity manifest: {path}")
    return manifest


def cross_check(new: list[ImageIdentity], reference: dict, *,
                max_distance: int = NEAR_DUPLICATE_MAX_DISTANCE) -> dict:
    """Compare new images against a reference identity manifest.

    Two independent gates, reported separately: an exact SHA-256 match is a
    duplicate and always a refusal; a perceptual-hash match within
    ``max_distance`` bits is a near duplicate and also refuses, naming the
    reference image and the distance so a collector can judge.
    """
    by_hash = {image["sha256"]: image for image in reference.get("images", [])}
    by_dhash = [(image["dhash"], image) for image in reference.get("images", [])]
    exact, near = [], []
    for identity in new:
        if identity.sha256 in by_hash:
            exact.append({"new": identity.path,
                          "reference": by_hash[identity.sha256]["path"]})
            continue
        for reference_hash, image in by_dhash:
            distance = _hamming(identity.dhash, reference_hash)
            if distance <= max_distance:
                near.append({"new": identity.path, "reference": image["path"],
                             "distance": distance})
                break
    return {"exact_duplicates": exact, "near_duplicates": near,
            "clean": not exact and not near,
            "checked": len(new),
            "reference_count": len(reference.get("images", []))}


def assert_independent(new: list[ImageIdentity], reference: dict, *,
                       max_distance: int = NEAR_DUPLICATE_MAX_DISTANCE) -> dict:
    """cross_check, but any overlap raises. Registration refuses on collision."""
    report = cross_check(new, reference, max_distance=max_distance)
    if not report["clean"]:
        details = [f"{item['new']} == {item['reference']}" for item in report["exact_duplicates"]]
        details += [f"{item['new']} ~ {item['reference']} (dHash {item['distance']} bits)"
                    for item in report["near_duplicates"]]
        raise FreezeError(
            "new images overlap the development set; a blind test cannot contain "
            f"them: {'; '.join(details)}")
    return report


__all__ = ["DEFAULT_IDENTITY", "ImageIdentity", "NEAR_DUPLICATE_MAX_DISTANCE",
           "assert_independent", "cross_check", "image_identity",
           "load_identity_manifest", "scan_directory", "write_identity_manifest"]
