"""Vision prompt and parsing for real herb-material photographs.

The supplied dataset turned out not to be field photographs of growing plants:
it is 15 images of **dried, sliced Astragalus root** — the traded herb material.
That changes what a useful observation is.

A leaf-chlorosis prompt is the wrong instrument here. Asked about "leaf
yellowing", the model answers ``image_usable: true, regions: []`` or invents a
region covering the whole plate, because there is no leaf to describe. So this
module observes the properties that actually matter for sliced herb material:

* **断面性状** — the cut surface: fissures, powder, hollow, dense.
* **色泽** — colour, including sulphur-fumigated pale yellow.
* **霉变 / 虫蛀 / 硫熏** — the defects a buyer checks for.
* **片型** — slice shape and thickness.

The same two defences as the leaf prompt apply: the prompt asks for description
only, and the parser drops any region whose text names a cause or a verdict.
"""

from __future__ import annotations

import base64
import json

from dgx.client import DgxClient

# Observation vocabulary for sliced herb material. These are what the model is
# asked to name; anything else it says is still recorded verbatim under
# ``raw_response`` but is not promoted to a structured observation.
PHENOTYPES = (
    "cut_surface_fissure",   # 裂隙 / 炸裂
    "cut_surface_powder",    # 粉性足，断面呈粉末状
    "cut_surface_dense",     # 角质 / 致密
    "cut_surface_hollow",    # 空心
    "colour_pale_yellow",    # 色泽淡黄（硫熏后常见）
    "colour_amber",          # 黄棕 / 琥珀色
    "colour_dark_brown",     # 深褐 / 焦褐
    "mould_visible",         # 可见霉斑
    "insect_damage",         # 虫蛀孔道
    "slice_irregular",       # 片型不整 / 厚薄不均
    "unknown",
)

PROMPT_TEMPLATE = """Observe this photograph of dried sliced medicinal herb material. Report what is visible as JSON only.

Format: {"image_usable": true, "regions": [{"phenotype": "<one of PHENOTYPES>", "label": "short text", "bbox": [l, t, r, b], "confidence": 0-1}]}

PHENOTYPES = @@PHENOTYPES@@

Rules:
- bbox normalised 0-1 (divide pixel coordinates by image width and height).
- Describe the cut surface, colour and any visible defects only.
- Never state a quality grade, price, authenticity verdict, or cause. You are describing, not judging.
- No visible feature of interest -> {"image_usable": false, "regions": []}.
- JSON only, no explanation, no markdown fence.

Species: @@SPECIES@@
"""

# A reply containing any of these is a judgement, not an observation, and the
# region is dropped. "Grade A", "authentic", "because of mould" are all verdicts.
JUDGEMENT_MARKERS = ("等级", "为正品", "是假", "伪品", "优质", "劣质", "合格", "不合格",
                     "grade ", "authentic", "fake", "counterfeit", "because of",
                     "caused by", "due to", "should be rejected")

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
    "options": {{"temperature": 0, "num_predict": 1500}},
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


def build_prompt(*, species: str) -> str:
    return (PROMPT_TEMPLATE
            .replace("@@PHENOTYPES@@", "|".join(PHENOTYPES))
            .replace("@@SPECIES@@", species))


def build_remote_script(*, model: str, image_b64: str, species: str,
                        prompt: str) -> str:
    """Render the script that runs on the node. Kept separate so tests can
    assert on the prompt without touching a live GPU."""
    return REMOTE_SCRIPT.format(model=model, image_b64=image_b64,
                                species=species, prompt=prompt)


def _extract_json(text: str) -> dict | None:
    if not text:
        return None
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None


def _salvage_regions(text: str) -> list[dict]:
    """Recover region objects from JSON the model did not finish.

    A model asked to enumerate regions sometimes runs out of budget mid-array, so
    the document never closes and ``json.loads`` rejects the whole thing. This
    collects every ``{...}`` span that parses on its own and keeps the ones that
    name a phenotype, so the entries it did finish are not lost with the document.

    Implementation note: a stack of open-brace offsets is kept, and every closing
    brace tries to parse the span it closes. That finds an inner region inside an
    outer wrapper as its own candidate, which a single depth counter cannot do —
    the wrapper is still open when the region closes.
    """
    if not text:
        return []
    recovered: list[dict] = []
    open_spans: list[int] = []
    in_string = False
    escaped = False
    for index, character in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue
        if character == '"':
            in_string = True
        elif character == "{":
            open_spans.append(index)
        elif character == "}" and open_spans:
            span_start = open_spans.pop()
            try:
                candidate = json.loads(text[span_start:index + 1])
            except json.JSONDecodeError:
                continue
            # Only the region objects are wanted. The outer wrapper
            # ({"image_usable": ..., "regions": [...]}) has no phenotype and is
            # skipped; the regions inside it are their own candidates.
            if isinstance(candidate, dict) and "phenotype" in candidate:
                recovered.append(candidate)
    return recovered

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


def _is_judgement(text: str) -> bool:
    lowered = text.lower()
    return any(marker in text or marker in lowered for marker in JUDGEMENT_MARKERS)


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


def run_vision(client: DgxClient, *, image_bytes: bytes, species: str,
               model: str, timeout: int = 600) -> dict:
    """Run one real vision inference on the node and return a parsed record.

    The image's own dimensions are read here and passed to the parser, so a model
    that answers in pixels can be rescaled against the true canvas instead of a
    guessed one.
    """
    image_size = _image_size(image_bytes)
    image_b64 = base64.b64encode(image_bytes).decode("ascii")
    prompt = build_prompt(species=species)
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
        if not isinstance(raw_regions, list):
            raw_regions = _salvage_regions(text)
    else:
        # The document never closed. Keep whatever complete regions it managed.
        raw_regions = _salvage_regions(text)
        image_usable = bool(raw_regions)
    if isinstance(raw_regions, list):
        for raw in raw_regions:
                region = _normalise_region(raw, image_size=image_size)
                if region is None:
                    continue
                # Second line of defence: drop a region that reads as a verdict.
                if _is_judgement(json.dumps(raw, ensure_ascii=False)):
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
