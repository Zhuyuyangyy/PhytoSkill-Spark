"""Build an annotation worksheet: each real image beside what the model said.

The point of the worksheet is to make annotation cheap. For every image the model's
structured observation is already recorded, so the annotator judges agreement
rather than describing the image from scratch. That is what turns 79 model
observations into a measurable accuracy figure instead of an unverifiable claim.

Writes artifacts/dgx/annotation-worksheet.json and a printable .md table.
"""

from __future__ import annotations

import json
from pathlib import Path

# The vocabulary the model was asked to use. An annotation outside this set is a
# discovery, not an error — the record keeps whatever the annotator writes.
HERB_LABELS = ("cut_surface_fissure", "cut_surface_powder", "cut_surface_dense",
               "cut_surface_hollow", "colour_pale_yellow", "colour_amber",
               "colour_dark_brown", "mould_visible", "insect_damage",
               "slice_irregular")
LEAF_LABELS = ("leaf_yellowing", "leaf_spot", "wilting")

HERB_LABELS_ZH = {
    "cut_surface_fissure": "断面裂隙/炸裂",
    "cut_surface_powder": "断面粉性",
    "cut_surface_dense": "断面致密/角质",
    "cut_surface_hollow": "断面空心",
    "colour_pale_yellow": "色泽淡黄（硫熏后常见）",
    "colour_amber": "色泽黄棕/琥珀",
    "colour_dark_brown": "色泽深褐/焦褐",
    "mould_visible": "可见霉斑",
    "insect_damage": "虫蛀孔道",
    "slice_irregular": "片型不整/厚薄不均",
}
LEAF_LABELS_ZH = {
    "leaf_yellowing": "叶片黄化",
    "leaf_spot": "叶片斑点",
    "wilting": "萎蔫",
}


def _load(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    document = json.loads(path.read_text(encoding="utf-8"))
    return [case for case in document.get("cases", []) if "vision" in case]


def _predictions(observations: list[dict]) -> list[str]:
    """Distinct phenotypes the model reported, in first-seen order."""
    seen: list[str] = []
    for observation in observations:
        phenotype = observation.get("phenotype")
        if phenotype and phenotype not in seen:
            seen.append(phenotype)
    return seen


def entry_path(image: str) -> Path:
    """Where the annotator opens the image from."""
    return Path("case_workspace/annotate") / image


def _image_size(path: Path) -> str | None:
    """Width x height, read from the file itself."""
    try:
        from PIL import Image
    except ModuleNotFoundError:  # pragma: no cover - Pillow is a test extra
        return None
    try:
        with Image.open(path) as image:
            return f"{image.size[0]}x{image.size[1]}"
    except Exception:  # noqa: BLE001 - an unreadable image just shows no size
        return None


def build_worksheet() -> dict:
    herb_cases = _load(Path("artifacts/dgx/real-chain.json"))
    leaf_cases = _load(Path("artifacts/dgx/real-chain-live-leaves.json"))

    entries: list[dict] = []
    for cases, mode, labels, labels_zh in (
            (herb_cases, "herb", HERB_LABELS, HERB_LABELS_ZH),
            (leaf_cases, "live", LEAF_LABELS, LEAF_LABELS_ZH)):
        for case in cases:
            observations = case["vision"]["observations"]
            entries.append({
                "image_size": _image_size(entry_path(case["image"])),
                "image": case["image"],
                # One folder for everything, so an annotator opens one directory
                # instead of hunting through two. The images are byte-identical
                # copies; only the location changed.
                "image_path": f"case_workspace/annotate/{case['image']}",
                "mode": mode,
                "model_predicted": _predictions(observations),
                "model_observation_count": len(observations),
                "model_image_usable": case["vision"].get("image_usable"),
                "candidate_labels": list(labels),
                "candidate_labels_zh": {key: labels_zh[key] for key in labels},
                # Filled in by the annotator.
                "annotator_labels": [],
                "annotator_agrees": None,
                "annotator_note": "",
            })
    return {
        "schema_version": 1,
        "instructions": [
            "对每张图，先自己看，再对照 model_predicted。",
            "annotator_labels 填你认可实际存在的性状（可多选，也可填候选集之外的词）。",
            "annotator_agrees 填 true/false：模型报的性状你是否认可。",
            "模型没报但你看到了，算漏检（false negative）；模型报了你没看到，算误报。",
            "两者都记，才能算 precision 和 recall，只算一个会高估。",
        ],
        "images": len(entries),
        "entries": entries,
        "annotation_status": "not_started",
    }


def render_markdown(worksheet: dict) -> str:
    lines = [
        "# 标注表",
        "",
        "每行一张图。**先自己看图**，再对照模型预测，填最后一列。",
        "",
        "`annotator_labels` 用候选集里的词（可多选）；模型漏报的你也填进去——",
        "漏检和误报要分开记，只记一个会高估准确率。",
        "",
    ]
    for mode, title in (("herb", "药材切片"), ("live", "植株叶片")):
        entries = [entry for entry in worksheet["entries"] if entry["mode"] == mode]
        lines.append(f"## {title}（{len(entries)} 张）")
        lines.append("")
        first = entries[0] if entries else None
        if first:
            labels = "；".join(f"`{key}` {first['candidate_labels_zh'][key]}"
                               for key in first["candidate_labels"])
            lines.append(f"候选性状：{labels}")
            lines.append("")
        lines.append("| 图片（文件名） | 尺寸 | 模型预测 | 观察数 | 你认可的性状（待填） | 模型对不对（待填） |")
        lines.append("| --- | --- | --- | --- | --- | --- |")
        for entry in entries:
            predicted = "、".join(entry["model_predicted"]) or "（模型未报告任何区域）"
            lines.append(f"| `{entry['image']}` | {entry.get('image_size') or '?'} "
                         f"| {predicted} | {entry['model_observation_count']} |  |  |")
        lines.append("")
    lines += [
        "## 填完之后",
        "",
        "把结果写回 `artifacts/dgx/annotation-worksheet.json` 的 `annotator_labels` 和",
        "`annotator_agrees`，或直接告诉我，我用 `dgx/score_annotations.py` 算：",
        "",
        "- precision：模型报的性状里，有多少你也认可",
        "- recall：你认可的性状里，有多少模型也报了",
        "- 图像可用性一致率：模型说不可用 vs 你说不可用",
        "",
        "只报 precision 会高估（模型可以少报），只报 recall 也会高估（模型可以乱报），",
        "所以两个都要。",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    worksheet = build_worksheet()
    out = Path("artifacts/dgx/annotation-worksheet.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(worksheet, ensure_ascii=False, indent=2) + "\n",
                   encoding="utf-8")
    markdown = Path("artifacts/dgx/annotation-worksheet.md")
    markdown.write_text(render_markdown(worksheet), encoding="utf-8")
    herb = sum(1 for e in worksheet["entries"] if e["mode"] == "herb")
    leaf = len(worksheet["entries"]) - herb
    print(f"wrote {out}")
    print(f"wrote {markdown}")
    print(f"{worksheet['images']} images: {herb} herb + {leaf} leaf")
    with_predictions = sum(1 for e in worksheet["entries"] if e["model_predicted"])
    print(f"{with_predictions} images have at least one model prediction")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
