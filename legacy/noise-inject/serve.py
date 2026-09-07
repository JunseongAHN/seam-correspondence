#!/usr/bin/env python
"""Serve the viewer locally.   python serve.py   ->  http://localhost:8000

Static files only; no dependencies beyond the standard library.
"""
import argparse, functools, http.server, os, socketserver, threading, webbrowser

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, default=8000)
ap.add_argument("--no-open", action="store_true")
a = ap.parse_args()

here = os.path.dirname(os.path.abspath(__file__))
if not os.path.exists(os.path.join(here, "data", "contours.json")):
    print("data/contours.json is missing — run:  python export.py")

class Handler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cache-Control", "no-store")     # so a re-export shows on refresh
        super().end_headers()
    def log_message(self, *args):
        pass

socketserver.TCPServer.allow_reuse_address = True
with socketserver.TCPServer(("127.0.0.1", a.port),
                            functools.partial(Handler, directory=here)) as srv:
    url = f"http://localhost:{a.port}/"
    print(f"serving {here}\n  {url}\nCtrl+C to stop")
    if not a.no_open:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
