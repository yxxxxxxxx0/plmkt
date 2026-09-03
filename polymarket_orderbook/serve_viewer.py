"""
Serves viewer_multi.html and opens it in a browser.

A plain HTTP server is needed because the viewer fetches per-match data
files at runtime, and browsers block fetch() on file:// URLs. Nothing here
is dynamic -- it just serves this directory.

Usage:
    python serve_viewer.py
    python serve_viewer.py --port 8800 --no-open
"""

import argparse
import functools
import http.server
import os
import socketserver
import threading
import webbrowser

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass  # keep the console readable; errors still surface via exceptions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8799)
    ap.add_argument("--page", default="viewer_full.html")
    ap.add_argument("--no-open", action="store_true", help="don't open a browser")
    args = ap.parse_args()

    page_path = os.path.join(BASE_DIR, args.page)
    if not os.path.exists(page_path):
        raise SystemExit(
            f"{args.page} not found. Build it first:\n"
            "  python fetch_game_windows.py          # after the games finish\n"
            "  python rebuild_duckdb.py --all        # data/uniform archive\n"
            "  python build_viewer_chunked.py        # data/viewer_full\n"
            "  python build_viewer_multi.py --manifest data/viewer_full/index.json \\\n"
            "      --template viewer_full_template.html --output viewer_full.html \\\n"
            "      --data-url data/viewer_full"
        )

    handler = functools.partial(QuietHandler, directory=BASE_DIR)
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("127.0.0.1", args.port), handler) as httpd:
        url = f"http://127.0.0.1:{args.port}/{args.page}"
        print(f"Serving {BASE_DIR}")
        print(f"  -> {url}")
        print("Ctrl+C to stop.")
        if not args.no_open:
            threading.Timer(0.5, lambda: webbrowser.open(url)).start()
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nstopped.")


if __name__ == "__main__":
    main()
