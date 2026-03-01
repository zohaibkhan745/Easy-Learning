"""
calibration_test.py — Test coordinate accuracy without the full agent.

Starts only the WebSocket server and sends test circles at known screen
positions so you can visually verify the overlay is drawing in the right
place.  Run alongside overlay.py.

Usage:
    python calibration_test.py
"""

import asyncio
import json
import ctypes
import logging
import websockets

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger("calibration")

OVERLAY_WS_PORT = 8765
connected_clients: set = set()


def _get_screen_resolution() -> tuple[int, int]:
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
        user32 = ctypes.windll.user32
        return (user32.GetSystemMetrics(0), user32.GetSystemMetrics(1))
    except Exception:
        return (1920, 1080)


async def _ws_handler(ws, path="/"):
    connected_clients.add(ws)
    logger.info(f"Overlay connected: {ws.remote_address}")
    try:
        async for _ in ws:
            pass
    finally:
        connected_clients.discard(ws)


async def broadcast(payload: dict):
    msg = json.dumps(payload)
    if connected_clients:
        await asyncio.gather(*(c.send(msg) for c in connected_clients), return_exceptions=True)


async def main():
    W, H = _get_screen_resolution()
    logger.info(f"Screen: {W}x{H}")

    server = await websockets.serve(_ws_handler, "localhost", OVERLAY_WS_PORT)
    logger.info(f"WebSocket server on ws://localhost:{OVERLAY_WS_PORT}")
    logger.info("Start overlay.py, then press Enter here to begin calibration.")

    # Wait for at least one client
    while not connected_clients:
        await asyncio.sleep(0.5)

    logger.info("Overlay connected! Sending calibration circles…")

    # ── Calibration points ──
    # Corners + center + midpoints of each edge
    test_points = [
        ("Top-Left",      100,       100),
        ("Top-Center",    W // 2,    100),
        ("Top-Right",     W - 100,   100),
        ("Center-Left",   100,       H // 2),
        ("Dead Center",   W // 2,    H // 2),
        ("Center-Right",  W - 100,   H // 2),
        ("Bottom-Left",   100,       H - 100),
        ("Bottom-Center", W // 2,    H - 100),
        ("Bottom-Right",  W - 100,   H - 100),
    ]

    for label, x, y in test_points:
        logger.info(f"  → {label:16s}  ({x:5d}, {y:5d})")
        await broadcast({"action": "draw", "x": x, "y": y, "label": label})
        await asyncio.sleep(1.5)  # stagger so you can see each one

    logger.info("✅ Calibration complete! Check that each circle appeared in the correct position.")
    logger.info("   Press Ctrl+C to exit.")
    await asyncio.Event().wait()


if __name__ == "__main__":
    asyncio.run(main())
