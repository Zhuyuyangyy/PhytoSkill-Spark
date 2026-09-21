"""Survey the real image set: ask the model what each photo actually shows.

Before building anything on top of a dataset, find out what is in it. This asks
the same vision model an open question per image and records the answer, so the
dataset's real composition is known rather than assumed.
"""

import base64
import json
import re
import sys
from pathlib import Path

from dgx.client import DgxClient, load_credentials

MODEL = "modelscope.cn/unsloth/Qwen3.8-27B-GGUF:latest"

QUESTION = ("What is in this image? Answer in one short sentence. "
            "Is it a whole plant, leaves, or sliced/processed herb material?")


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass
    directory = Path(sys.argv[1])
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else 15

    from dgx.vision import build_remote_script
    credentials = load_credentials(Path(".dgx.env"))
    records = []
    with DgxClient(credentials) as client:
        for index, image in enumerate(sorted(directory.glob("*.jpg"))[:limit], start=1):
            encoded = base64.b64encode(image.read_bytes()).decode("ascii")
            script = build_remote_script(model=MODEL, image_b64=encoded,
                                         species="黄芪", prompt=QUESTION)
            result = client.run_script(script, timeout=900)
            try:
                body = json.loads(result["stdout"].strip())
                answer = (body.get("response") or "").strip().replace("\n", " ")
                latency = round(float(body.get("latency_ms") or 0.0), 1)
            except (json.JSONDecodeError, ValueError):
                answer, latency = "(unreadable)", None
            print(f"[{index:2}] {image.name:18} {latency or '':>9} ms  {answer[:150]}")
            records.append({"image": image.name, "answer": answer,
                            "latency_ms": latency})

    # Name the file after the directory: two surveys of different subjects must
    # not overwrite each other. The first run did exactly that and silently lost
    # the herb survey when the leaf survey was written to the same path.
    stem = "dataset-survey-" + re.sub(r"[^0-9a-zA-Z]+", "-", directory.name).strip("-").lower()
    out = Path("artifacts/dgx") / f"{stem}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"model": MODEL, "question": QUESTION,
                               "directory": str(directory),
                               "records": records}, ensure_ascii=False, indent=2) + "\n",
                   encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
