"""
Local web server for venue_viewer.html (Polymarket vs Kalshi, per game).

    python prep_venue_view.py        # build data/view/ first
    python serve_venue_viewer.py     # then open http://localhost:8765

Serves the page at /, data/view/ at /view/, and the vendored Plotly build
at /vendor/ (so the page needs no internet). JSON is gzipped on the fly
when the browser accepts it -- a day-long game is tens of MB uncompressed.
"""
import argparse
import gzip
import http.server
import os
import webbrowser

BASE = os.path.dirname(os.path.abspath(__file__))
PAGE = os.path.join(BASE, "venue_viewer.html")
VIEW = os.path.join(BASE, "data", "view")


class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path in ("/", "/index.html"):
            return self.send_file(PAGE, "text/html; charset=utf-8")
        if path.startswith("/vendor/") and path.endswith(".js"):
            name = os.path.basename(path)
            return self.send_file(os.path.join(BASE, "vendor", name), "text/javascript")
        if path.startswith("/view/"):
            rel = os.path.normpath(path[len("/view/"):]).lstrip("\\/")
            full = os.path.join(VIEW, rel)
            if not os.path.abspath(full).startswith(os.path.abspath(VIEW)) or not rel.endswith(".json"):
                return self.send_error(403)
            return self.send_file(full, "application/json")
        self.send_error(404)

    def send_file(self, full, ctype):
        if not os.path.isfile(full):
            return self.send_error(404)
        with open(full, "rb") as f:
            body = f.read()
        gz = "gzip" in self.headers.get("Accept-Encoding", "") and len(body) > 4096
        if gz:
            body = gzip.compress(body, compresslevel=5)
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-cache")
        if gz:
            self.send_header("Content-Encoding", "gzip")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    url = f"http://localhost:{args.port}"
    print(f"venue viewer on {url}  (Ctrl+C to stop)", flush=True)
    if not args.no_browser:
        webbrowser.open(url)
    srv.serve_forever()


if __name__ == "__main__":
    main()
