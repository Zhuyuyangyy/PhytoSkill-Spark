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

from evaluation.split import FreezeError, ROLE_HOLDOUT

DEFAULT_IDENTITY = Path("artifacts/evaluations/development-identity.json")

# Hamming distance at or below which two perceptual hashes are reported as a
# near duplicate. 64-bit dHash: 5 bits is a re-encode or a mild crop; a
# different photograph of the same subject typically lands far above it.
NEAR_DUPLICATE_MAX_DISTANCE = 5


# Standard deviation of the 9x8 grayscale reduction below which an image is
# "flat": a blank or near-blank frame has no structure for a difference hash to
# describe, and any dHash match it produces is noise. Those hits go to manual
# review instead of an automatic refusal.
LOW_TEXTURE_STDDEV = 6.0


@dataclass(frozen=True)
class ImageIdentity:
    """What one image is, independent of any annotation."""

    path: str
    sha256: str
    bytes: int
    width: int | None
    height: int | None
    dhash: str
    texture: float = 0.0

    def to_dict(self) -> dict:
        return {"path": self.path, "sha256": self.sha256, "bytes": self.bytes,
                "width": self.width, "height": self.height, "dhash": self.dhash,
                "texture": self.texture}


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
    texture = _stddev(pixels)
    return f"{int(''.join(bits), 2):016x}", texture


def _stddev(pixels: bytes) -> float:
    """Population standard deviation of the reduced grayscale."""
    if not pixels:
        return 0.0
    mean = sum(pixels) / len(pixels)
    return round((sum((value - mean) ** 2 for value in pixels) / len(pixels)) ** 0.5, 3)


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
    dhash, texture = _dhash(path)
    return ImageIdentity(
        path=str(path.relative_to(root)) if root else str(path),
        sha256=_sha256(path), bytes=path.stat().st_size,
        width=width, height=height, dhash=dhash, texture=texture)


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
    """Compare new images against a reference manifest **and against each other**.

    Three independent gates, reported separately:

    * an exact SHA-256 match — against the reference or *within the batch* — is
      a duplicate and always a refusal. The intra-batch gate matters because a
      hundred "new" images that are re-encodes of two photographs would pass a
      reference-only check while carrying almost no independent information.
    * a perceptual-hash match within ``max_distance`` bits is a near duplicate
      and refuses, naming the counterpart and the distance.
    * a near-duplicate hit where the new image is **flat** (no texture for a
      difference hash to describe) is demoted to ``manual_review``: on a blank
      frame the hash is noise, and refusing real data on noise is its own kind
      of error. A human looks, and the flag says so.
    """
    by_hash = {image["sha256"]: image for image in reference.get("images", [])}
    by_dhash = [(image["dhash"], image) for image in reference.get("images", [])]
    exact, near, manual = [], [], []
    for identity in new:
        if identity.sha256 in by_hash:
            exact.append({"new": identity.path,
                          "reference": by_hash[identity.sha256]["path"]})
            continue
        for reference_hash, image in by_dhash:
            distance = _hamming(identity.dhash, reference_hash)
            if distance <= max_distance:
                hit = {"new": identity.path, "reference": image["path"],
                       "distance": distance, "texture": identity.texture}
                (manual if identity.texture < LOW_TEXTURE_STDDEV else near).append(hit)
                break
    # Intra-batch: the same bytes or the same photograph twice in one delivery.
    for index, identity in enumerate(new):
        for other in new[index + 1:]:
            if identity.sha256 == other.sha256:
                exact.append({"new": identity.path, "reference": other.path,
                              "scope": "within the batch"})
            elif _hamming(identity.dhash, other.dhash) <= max_distance:
                (manual if identity.texture < LOW_TEXTURE_STDDEV else near).append(
                    {"new": identity.path, "reference": other.path, "scope": "within the batch",
                     "distance": _hamming(identity.dhash, other.dhash),
                     "texture": identity.texture})
    return {"exact_duplicates": exact, "near_duplicates": near,
            "manual_review": manual,
            "clean": not exact and not near and not manual,
            "checked": len(new),
            "reference_count": len(reference.get("images", []))}


def assert_independent(new: list[ImageIdentity], reference: dict, *,
                       max_distance: int = NEAR_DUPLICATE_MAX_DISTANCE,
                       allow_manual_review: bool = False) -> dict:
    """cross_check, but overlap refuses registration.

    Flat-frame hits refuse too unless ``allow_manual_review``: the operator has
    looked at the named pairs and accepts them. That is a human decision — the
    hash cannot make it.
    """
    report = cross_check(new, reference, max_distance=max_distance)
    if report["exact_duplicates"] or report["near_duplicates"] or (
            report["manual_review"] and not allow_manual_review):
        details = [f"{item['new']} == {item['reference']}"
                   f"{' [' + item['scope'] + ']' if item.get('scope') else ''}"
                   for item in report["exact_duplicates"]]
        details += [f"{item['new']} ~ {item['reference']} (dHash {item['distance']} bits"
                    f"{', ' + item['scope'] if item.get('scope') else ''})"
                    for item in report["near_duplicates"]]
        details += [f"{item['new']} ~ {item['reference']} (flat frame, dHash "
                    f"{item['distance']} bits — needs a human look)"
                    for item in report["manual_review"]]
        raise FreezeError(
            "new images overlap the development set or each other; a blind test "
            f"cannot contain them: {'; '.join(details)}")
    return report


def image_id(name: str | Path) -> str:
    """The one image id every stage speaks.

    Windows and POSIX paths spell the same file differently, and a collector's
    manifest may record ``leaves/01.jpg`` while a prediction records ``01.jpg``.
    Everything downstream — registration, roles, predictions, labels, scoring —
    keys on this id, so a separator can never silently split an image in two.
    """
    return str(name).replace("\\", "/").strip().lstrip("./")


def registered_images(manifest: dict) -> dict[str, dict]:
    """The registered image set, keyed by the unified image id.

    Admission (``identity --new``) and scoring are separate paths; this is the
    bridge: a formal holdout can require that every image it scores was
    registered — same bytes, role ``holdout`` — instead of trusting that the
    file names happen to match.

    A basename collision is refused rather than silently overwritten: two
    registered images whose ids differ but whose basenames match would let one
    stand in for the other, and that is exactly the substitution a blind test
    cannot survive.
    """
    registered: dict[str, dict] = {}
    basenames: dict[str, str] = {}
    for image in manifest.get("images", []):
        identifier = image_id(image["path"])
        registered[identifier] = {"sha256": image["sha256"],
                                  "texture": image.get("texture")}
        base = identifier.rsplit("/", 1)[-1]
        if base in basenames and basenames[base] != identifier:
            raise FreezeError(
                f"registered images {basenames[base]!r} and {identifier!r} share "
                "the basename; a basename lookup could substitute one for the "
                "other — register them under distinct names")
        basenames[base] = identifier
    for name, role in (manifest.get("roles") or {}).items():
        identifier = image_id(name)
        if identifier in registered:
            registered[identifier]["role"] = role
        elif identifier.rsplit("/", 1)[-1] in basenames:
            # Roles recorded under a bare basename still bind, through the
            # unambiguous basename index.
            registered[basenames[identifier.rsplit("/", 1)[-1]]]["role"] = role
    return registered


def verify_registration(entries: list[dict], registered: dict[str, dict]) -> dict:
    """Check scored entries against the registered set, entry by entry.

    Three conditions per image, all required on the formal path:

    * the image is **registered** — an id nobody admitted cannot be scored;
    * its **bytes match** — the prediction carries ``image_sha256`` and it must
      equal the registered digest, so a same-name different-bytes substitution
      is refused rather than inherited;
    * its **role is holdout** — a development or calibration image in a blind
      test is the leak this whole apparatus exists to prevent.

    A prediction without ``image_sha256`` is refused on the formal path: the
    field is what makes "same bytes" checkable at all.
    """
    problems: list[str] = []
    verified: dict[str, dict] = {}
    for entry in entries:
        identifier = image_id(entry.get("image", ""))
        record = registered.get(identifier)
        if record is None:
            problems.append(f"{identifier}: not in the registered-data manifest")
            continue
        digest = entry.get("image_sha256")
        if not digest:
            problems.append(f"{identifier}: prediction carries no image_sha256")
        elif digest != record["sha256"]:
            problems.append(
                f"{identifier}: image bytes do not match the registered digest "
                f"({digest[:12]} != {record['sha256'][:12]})")
        if record.get("role") != ROLE_HOLDOUT:
            problems.append(f"{identifier}: registered role is "
                            f"{record.get('role') or 'unset'!r}, not holdout")
        verified[identifier] = record
    if problems:
        raise FreezeError("registered-data verification failed: " + "; ".join(problems))
    return verified


__all__ = ["DEFAULT_IDENTITY", "ImageIdentity", "LOW_TEXTURE_STDDEV",
           "NEAR_DUPLICATE_MAX_DISTANCE", "assert_independent", "cross_check",
           "image_id", "image_identity", "load_identity_manifest",
           "registered_images", "scan_directory", "verify_registration",
           "write_identity_manifest"]
