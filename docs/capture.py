"""Capture exact 1920x1080 screenshots of the AEGIS console via headless Chrome CDP.

Dev-only tool for the project guide. It drives the real UI; it does not modify app code.
Run: python docs/capture.py
"""
from __future__ import annotations

import asyncio
import base64
import json
import shutil
import subprocess
import time
from pathlib import Path

import requests
import websockets

ROOT = Path(__file__).resolve().parent
IMG = ROOT / "img"
PROFILE = ROOT / ".chrome-capture"
PORT = 9223
BASE = "http://127.0.0.1:8765"
CHROME = Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe")

_next_id = 0


async def cdp(ws, method: str, params: dict | None = None):
    global _next_id
    _next_id += 1
    mid = _next_id
    await ws.send(json.dumps({"id": mid, "method": method, "params": params or {}}))
    while True:
        msg = json.loads(await ws.recv())
        if msg.get("id") == mid:
            if "error" in msg:
                raise RuntimeError(f"{method}: {msg['error']}")
            return msg.get("result", {})


async def js(ws, expression: str):
    result = await cdp(ws, "Runtime.evaluate", {
        "expression": expression,
        "awaitPromise": True,
        "returnByValue": True,
    })
    if result.get("exceptionDetails"):
        raise RuntimeError(f"JS failed: {result['exceptionDetails']}")
    return result.get("result", {}).get("value")


async def wait_for(ws, expression: str, timeout: float = 30.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if await js(ws, expression):
            return
        await asyncio.sleep(0.25)
    raise TimeoutError(expression)


async def shot(ws, name: str):
    result = await cdp(ws, "Page.captureScreenshot", {
        "format": "png",
        "captureBeyondViewport": False,
    })
    path = IMG / name
    path.write_bytes(base64.b64decode(result["data"]))
    print(f"captured {name}")


async def click(ws, selector: str):
    await js(ws, f"(() => {{ const el = document.querySelector({json.dumps(selector)});"
               " if (!el) throw new Error('missing target'); el.click(); return true; })()")


async def main():
    IMG.mkdir(exist_ok=True)
    for old in IMG.glob("*.png"):
        old.unlink()
    shutil.rmtree(PROFILE, ignore_errors=True)

    if not CHROME.exists():
        raise SystemExit(f"Chrome not found: {CHROME}")

    proc = subprocess.Popen([
        str(CHROME),
        "--headless=new",
        "--disable-gpu",
        "--hide-scrollbars",
        "--no-first-run",
        "--no-default-browser-check",
        "--remote-allow-origins=*",
        f"--remote-debugging-port={PORT}",
        f"--user-data-dir={PROFILE}",
        "--window-size=1920,1080",
        "--force-device-scale-factor=1",
        BASE,
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    try:
        tabs = None
        for _ in range(80):
            try:
                tabs = requests.get(f"http://127.0.0.1:{PORT}/json", timeout=1).json()
                page = next((t for t in tabs if t.get("type") == "page"), None)
                if page:
                    break
            except Exception:
                pass
            await asyncio.sleep(0.25)
        if not page:
            raise SystemExit("Chrome DevTools did not expose a page")

        async with websockets.connect(page["webSocketDebuggerUrl"], max_size=None) as ws:
            await cdp(ws, "Page.enable")
            await cdp(ws, "Runtime.enable")
            await cdp(ws, "Emulation.setDeviceMetricsOverride", {
                "width": 1920, "height": 1080,
                "deviceScaleFactor": 1, "mobile": False,
            })
            await cdp(ws, "Page.navigate", {"url": BASE})
            await wait_for(ws, "document.readyState === 'complete'", 30)
            await wait_for(ws, "document.fonts.ready.then(() => true)", 30)
            await asyncio.sleep(1.4)

            await shot(ws, "01-boot.png")
            await click(ws, "#bootEnter")
            await wait_for(ws, "!document.querySelector('#loading').classList.contains('on')", 40)
            await wait_for(ws, "document.querySelector('#kpiR0').textContent.trim() !== '—'", 20)
            await asyncio.sleep(1.6)
            await shot(ws, "02-outbreak.png")

            await click(ws, "button[data-tab='plan']")
            await asyncio.sleep(0.9)
            await shot(ws, "03-plan.png")

            await click(ws, "button[data-tab='memory']")
            await asyncio.sleep(1.4)
            await shot(ws, "04-memory.png")

            await click(ws, "button[data-tab='model']")
            await asyncio.sleep(1.1)
            await shot(ws, "05-model.png")

            await click(ws, "button[data-tab='eval']")
            await asyncio.sleep(1.1)
            await shot(ws, "06-eval.png")

            await click(ws, "#paletteBtn")
            await wait_for(ws, "document.querySelector('#palette').classList.contains('on')", 5)
            await asyncio.sleep(0.35)
            await shot(ws, "07-palette.png")
            # Clicking the backdrop is the app's own close path (the input owns focus).
            await click(ws, "#palette")
            await wait_for(ws, "!document.querySelector('#palette').classList.contains('on')", 5)
            await asyncio.sleep(0.35)

            await click(ws, "button[data-tab='outbreak']")
            await asyncio.sleep(0.4)
            await click(ws, "#executeBtn")
            await wait_for(ws, "document.querySelector('#modal').classList.contains('on')", 5)
            await asyncio.sleep(0.35)
            await shot(ws, "08-verdict.png")
            await click(ws, "#modalClose")
            await wait_for(ws, "!document.querySelector('#modal').classList.contains('on')", 5)

            await click(ws, "#toggleDetailBtn")
            await wait_for(ws, "document.querySelector('#modal').classList.contains('on')", 5)
            await asyncio.sleep(0.35)
            await shot(ws, "09-evidence.png")
            await click(ws, "#modalClose")
            await wait_for(ws, "!document.querySelector('#modal').classList.contains('on')", 5)

            await click(ws, ".path")
            await wait_for(ws, "document.querySelector('#modal').classList.contains('on')", 5)
            await asyncio.sleep(0.35)
            await shot(ws, "10-path.png")
            await click(ws, "#modalClose")
            await wait_for(ws, "!document.querySelector('#modal').classList.contains('on')", 5)

            await click(ws, ".action-card")
            await wait_for(ws, "document.querySelector('#modal').classList.contains('on')", 5)
            await asyncio.sleep(0.35)
            await shot(ws, "11-action.png")
            await click(ws, "#modalClose")
            await wait_for(ws, "!document.querySelector('#modal').classList.contains('on')", 5)

            # Leave the live preview on the control room.
            await click(ws, "button[data-tab='outbreak']")
            print("capture complete")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        shutil.rmtree(PROFILE, ignore_errors=True)


if __name__ == "__main__":
    asyncio.run(main())
