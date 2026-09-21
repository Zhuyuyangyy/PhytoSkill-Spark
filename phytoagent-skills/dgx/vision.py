"""Real plant-vision inference on the DGX Spark node, through ollama.

What this module does:

* uploads one image to the node,
* asks the resident vision model to describe what it sees,
* parses the free-text reply into the structured observations the
  ``plant_vision`` contract requires,
* records latency and GPU state for the same call.

What it deliberately does **not** do:

* **diagnose.** The model describes phenotypes; it never names a pathogen. That
  boundary is enforced by the prompt *and* by the parser, which drops any region
  that reads as a diagnosis.
* **invent regions.** If the model reports nothing usable, the result is an empty
  observation list, not a plausible-looking one.
* **claim certainty.** ``model_score`` is the confidence the model stated about
  its own description, not a probability of disease.
"""

from __future__ import annotations

import base64
import json

from dgx.client import DgxClient

# The prompt is the first line of defence: it asks for observation, not diagnosis.
PROMPT_TEMPLATE = """Report coloured regions in this image as JSON only.

Format: {"image_usable": true, "regions": [{"phenotype": "leaf_yellowing"|"leaf_spot"|"wilting"|"unknown", "label": "short text", "bbox": [l, t, r, b], "confidence": 0-1}]}

Rules: bbox normalised 0-1 (divide pixels by image width and height). Yellow areas -> leaf_yellowing, small dark spots -> leaf_spot, drooping -> wilting, else -> unknown. Never name a disease or cause. No visible region -> {"image_usable": false, "regions": []}. JSON only.

Species: @@SPECIES@@
"""

# A reply containing any of these is a diagnosis, and the region is dropped.
DIAGNOSIS_MARKERS = ("病原", "病因", "病害是", "感染了", "确诊", "disease is",
                     "caused by", "pathogen")

PHENOTYPES = ("leaf_yellowing", "leaf_spot", "wilting", "unknown")


def _image_size(image_bytes: bytes) -> tuple[int, int] | None:
    """Width and height of the image, read from its header.

    Returns None when the bytes are not a decodable image: the caller then refuses
    pixel-space boxes rather than guessing a canvas.
    """
    import io

    try:
        from PIL import Image
    except ModuleNotFoundError:  # pragma: no cover - Pillow is a test extra
        return None
    try:
        with Image.open(io.BytesIO(image_bytes)) as image:
            return image.size
    except Exception:  # noqa: BLE001 - any decode failure means "unknown size"
        return None


def _extract_json(text: str) -> dict | None:
    """Pull the first JSON object out of a free-form model reply."""
    if not text:
        return None
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None


def _normalise_region(raw: object, *, image_size: tuple[int, int] | None = None) -> dict | None:
    """Validate one model-reported region against the contract.

    Anything that fails validation is dropped rather than coerced: a bbox with
    out-of-range or inverted coordinates is not evidence.
    """
    if not isinstance(raw, dict):
        return None
    if raw.get("phenotype") not in PHENOTYPES:
        return None
    label = raw.get("label")
    if not isinstance(label, str) or not label.strip():
        return None
    bbox = raw.get("bbox")
    if not isinstance(bbox, list) or len(bbox) != 4:
        return None
    try:
        left, top, right, bottom = (float(value) for value in bbox)
    except (TypeError, ValueError):
        return None
    # A model that ignores the normalisation instruction returns pixel values.
    # Rescale rather than discard a real observation, but only against the real
    # image size, and only when the box is unambiguously in pixel space.
    #
    # The test is "every coordinate exceeds 1", not "any coordinate does". A
    # normalised box has at least one coordinate at or below 1 by definition, and
    # a mixed box such as [0, 0, 1.5, 0.5] is malformed rather than pixel-scaled:
    # dividing it by the canvas size would turn a nonsense box into a tiny
    # plausible-looking one, which is worse than refusing it.
    if min(left, top, right, bottom) > 1.0:
        if min(left, top, right, bottom) < 0.0:
            return None
        if image_size is None:
            return None
        width, height = image_size
        if width <= 0 or height <= 0:
            return None
        left, right = left / width, right / width
        top, bottom = top / height, bottom / height
    if not all(0.0 <= value <= 1.0 for value in (left, top, right, bottom)):
        return None
    if left >= right or top >= bottom:
        return None
    # A box that covers the whole frame is not a region observation. It is the
    # model declining to localise, and reporting it would turn "I see the image"
    # into an evidence-backed claim about a place in it.
    if (right - left) >= 0.98 and (bottom - top) >= 0.98:
        return None
    try:
        confidence = float(raw.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    return {"phenotype": raw["phenotype"], "label": label.strip()[:80],
            "bbox_normalized": [left, top, right, bottom],
            "model_score": max(0.0, min(1.0, confidence))}


def _is_diagnosis(text: str) -> bool:
    lowered = text.lower()
    return any(marker in text or marker in lowered for marker in DIAGNOSIS_MARKERS)


REMOTE_SCRIPT = '''
import json, time, urllib.request, urllib.error

MODEL = {model!r}
IMAGE_B64 = {image_b64!r}
SPECIES = {species!r}
PROMPT = {prompt!r}
BASE = "http://127.0.0.1:11434"

payload = {{
    "model": MODEL,
    "prompt": PROMPT,
    "images": [IMAGE_B64],
    "stream": False,
    "options": {{"temperature": 0, "num_predict": 400}},
}}

request = urllib.request.Request(
    BASE + "/api/generate", data=json.dumps(payload).encode("utf-8"),
    headers={{"Content-Type": "application/json"}}, method="POST")

start = time.perf_counter()
try:
    with urllib.request.urlopen(request, timeout=600) as response:
        body = json.load(response)
        status = response.status
except urllib.error.HTTPError as error:
    print(json.dumps({{"http_status": error.code,
                       "error": error.read().decode("utf-8", "replace")[:600]}},
                     ensure_ascii=False))
    raise SystemExit(0)

elapsed_ms = (time.perf_counter() - start) * 1000
with urllib.request.urlopen(BASE + "/api/ps", timeout=30) as ps:
    gpu_ps = json.load(ps)

print(json.dumps({{
    "http_status": status,
    "response": body.get("response") or "",
    "eval_count": body.get("eval_count"),
    "prompt_eval_count": body.get("prompt_eval_count"),
    "load_duration_ns": body.get("load_duration"),
    "total_duration_ns": body.get("total_duration"),
    "latency_ms": elapsed_ms,
    "gpu_ps": gpu_ps,
}}, ensure_ascii=False))
'''


def build_remote_script(*, model: str, image_b64: str, species: str,
                        prompt: str) -> str:
    """Render the script that runs on the node. Kept separate so tests can
    assert on the prompt without touching a live GPU."""
    return REMOTE_SCRIPT.format(model=model, image_b64=image_b64,
                                species=species, prompt=prompt)


def run_vision(client: DgxClient, *, image_bytes: bytes, species: str,
               model: str, timeout: int = 600) -> dict:
    """Run one real vision inference on the node and return a parsed record.

    The image's own dimensions are read here and passed to the parser, so a model
    that answers in pixels can be rescaled against the true canvas instead of a
    guessed one.
    """
    image_size = _image_size(image_bytes)
    """Run one real vision inference on the node and return a parsed record."""
    image_b64 = base64.b64encode(image_bytes).decode("ascii")
    prompt = PROMPT_TEMPLATE.replace("@@SPECIES@@", species)
    script = build_remote_script(model=model, image_b64=image_b64,
                                 species=species, prompt=prompt)
    result = client.run_script(script, timeout=timeout)
    if result["status"] != 0:
        raise RuntimeError(f"vision inference failed: {result['stderr'].strip()[:400]}")
    try:
        payload = json.loads(result["stdout"].strip())
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"unreadable vision reply: {result['stdout'][:300]}") from exc
    if int(payload.get("http_status", 200)) >= 400:
        raise RuntimeError(f"vision endpoint rejected the request: {payload.get('error')}")

    text = payload.get("response") or ""
    parsed = _extract_json(text)
    regions: list[dict] = []
    image_usable = False
    if isinstance(parsed, dict):
        image_usable = parsed.get("image_usable") is True
        raw_regions = parsed.get("regions")
        if isinstance(raw_regions, list):
            for raw in raw_regions:
                region = _normalise_region(raw, image_size=image_size)
                if region is None:
                    continue
                # Second line of defence: drop a region that reads as a diagnosis.
                if _is_diagnosis(json.dumps(raw, ensure_ascii=False)):
                    continue
                regions.append(region)

    gpu = client.gpu_state()
    return {
        "model": model,
        "image_usable": image_usable,
        "regions": regions,
        "latency_ms": round(float(payload.get("latency_ms") or 0.0), 1),
        "eval_count": payload.get("eval_count"),
        "prompt_eval_count": payload.get("prompt_eval_count"),
        "load_duration_ns": payload.get("load_duration_ns"),
        "total_duration_ns": payload.get("total_duration_ns"),
        "gpu": gpu.to_dict(),
        "gpu_ps": payload.get("gpu_ps"),
        "raw_response": text[:2000],
    }
