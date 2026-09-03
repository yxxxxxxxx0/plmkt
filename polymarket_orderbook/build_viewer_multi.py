"""
Builds viewer_full.html from viewer_full_template.html.

Only the *manifest* (data/viewer_full/index.json -- match identity,
market/token structure, tick counts, chunk boundaries) is inlined into the
HTML. The per-tick bulk lives under data/viewer_full/<event_slug>/ and is
fetched by the page as needed. That keeps the page a few hundred KB and opens
instantly, instead of the ~2.6 GB per match the full-fidelity books weigh.

Because the page fetches those payloads, it must be served over HTTP --
browsers block fetch() on file:// URLs. Use serve_viewer.py:

    python serve_viewer.py

Usage:
    python build_viewer_multi.py
    python build_viewer_multi.py --manifest data/viewer/index.json --output viewer_multi.html
"""

import argparse
import json
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=os.path.join(BASE_DIR, "data", "viewer_full", "index.json"))
    ap.add_argument("--template", default=os.path.join(BASE_DIR, "viewer_full_template.html"))
    ap.add_argument("--output", default=os.path.join(BASE_DIR, "viewer_full.html"))
    ap.add_argument("--data-url", default="data/viewer_full",
                     help="where the page should fetch per-match payloads from, "
                          "relative to the served page")
    args = ap.parse_args()

    with open(args.manifest, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    with open(args.template, "r", encoding="utf-8") as f:
        template = f.read()

    html = (template
            .replace("__MATCH_INDEX_DATA__", json.dumps(manifest))
            .replace("__DATA_DIR__", args.data_url))

    with open(args.output, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"Wrote {args.output} ({len(html)/1e6:.2f} MB, "
          f"{len(manifest['matches'])} matches in manifest)")
    print(f"Per-match data fetched on demand from {args.data_url}/<event_slug>.json")
    print("Serve it with:  python serve_viewer.py")


if __name__ == "__main__":
    main()
