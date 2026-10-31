"""
Build the static dashboard from the committed results.

    python site/build.py

Writes site/public/index.html (deploy this folder, e.g. on Vercel) and
site/build/preview.html (page body only). Per-seed records are left out; the
page needs the summary, explanations and metadata, so every number on it still
comes from results/results.json and results/unsw/results.json.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SITE = ROOT / "site"
SOURCES = {"kdd": ROOT / "results" / "results.json", "unsw": ROOT / "results" / "unsw" / "results.json"}


def main() -> None:
    data = {}
    for key, path in SOURCES.items():
        r = json.loads(path.read_text(encoding="utf-8"))
        data[key] = {"meta": r["meta"], "summary": r["summary"], "xai": r["xai"]}
    # "</" inside an inline script would end it early
    payload = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    body = (SITE / "page.html").read_text(encoding="utf-8").replace("/*DATA*/null", payload)

    (SITE / "build").mkdir(exist_ok=True)
    (SITE / "build" / "preview.html").write_text(body, encoding="utf-8")
    (SITE / "public").mkdir(exist_ok=True)
    full = (
        '<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
        '<meta name="description" content="Adversarial robustness of neural network intrusion detectors '
        'on KDD Cup 1999 and UNSW-NB15.">\n'
        "<style>*,*::before,*::after{box-sizing:border-box}body{margin:0}[hidden]{display:none!important}</style>\n"
        "</head>\n<body>\n" + body + "\n</body>\n</html>\n"
    )
    (SITE / "public" / "index.html").write_text(full, encoding="utf-8")
    print(f"site/public/index.html: {len(full) / 1024:.0f} KB")


if __name__ == "__main__":
    main()
