"""Launch AEGIS Contagion.

    python run.py            # http://127.0.0.1:8765
    python run.py --reload   # dev
"""
from __future__ import annotations

import argparse
import threading
import time
import webbrowser


def main() -> None:
    parser = argparse.ArgumentParser(description="AEGIS Contagion console")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--reload", action="store_true")
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    if not args.no_browser:
        def open_later():
            time.sleep(1.6)
            webbrowser.open(f"http://{args.host}:{args.port}/")
        threading.Thread(target=open_later, daemon=True).start()

    import uvicorn
    uvicorn.run("app.server:app", host=args.host, port=args.port,
                reload=args.reload, log_level="info")


if __name__ == "__main__":
    main()
