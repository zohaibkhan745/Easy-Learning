"""
overlay.py — Transparent, click-through, always-on-top overlay.

Connects to ws://localhost:8765 and draws a glowing red circle wherever the
AI tutor's highlight_ui tool tells it to.

Run with:  python overlay.py           (normal mode, waits for WS)
           python overlay.py --test     (self-test: draws 4 circles immediately)
"""

import sys
import json
import ctypes
import threading
import time

from PyQt5.QtCore import (
    Qt, QTimer, pyqtSignal, QObject, QPoint, QRectF,
)
from PyQt5.QtGui import QPainter, QColor, QPen, QFont, QBrush
from PyQt5.QtWidgets import QApplication, QWidget

# ---------------------------------------------------------------------------
# DPI awareness (must be called BEFORE QApplication is created)
# ---------------------------------------------------------------------------
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(2)   # Per-monitor DPI aware
except Exception:
    pass  # Not on Windows or older OS — ignore


# ---------------------------------------------------------------------------
# WebSocket listener (runs in a background thread, emits Qt signals)
# ---------------------------------------------------------------------------

class JsonSignal(QObject):
    """Bridge between the websocket thread and the Qt main thread."""
    received = pyqtSignal(dict)
    status_changed = pyqtSignal(bool)  # connected/disconnected


def ws_listener(signal: JsonSignal):
    """
    Blocking listener that runs in a daemon thread.
    Uses the synchronous `websockets.sync.client` API so we don't need
    a second asyncio loop.
    """
    import websockets.sync.client as ws_sync

    url = "ws://localhost:8765"
    retry_delay = 1

    while True:
        try:
            print(f"[overlay] Connecting to {url} ...")
            with ws_sync.connect(url, close_timeout=4) as sock:
                print("[overlay] Connected to agent WebSocket.")
                signal.status_changed.emit(True)
                retry_delay = 1  # Reset on successful connect

                for msg in sock:
                    try:
                        data = json.loads(msg)
                        if data.get("action") == "draw":
                            print(f"[overlay] Draw: {data.get('label', '')} "
                                  f"at ({data.get('x')}, {data.get('y')})")
                            signal.received.emit(data)
                    except json.JSONDecodeError:
                        pass
        except Exception as exc:
            signal.status_changed.emit(False)
            print(f"[overlay] Disconnected ({type(exc).__name__}). "
                  f"Retrying in {retry_delay}s...")
            time.sleep(retry_delay)
            retry_delay = min(retry_delay * 1.5, 10)  # Backoff up to 10s


# ---------------------------------------------------------------------------
# Circle object drawn on the overlay
# ---------------------------------------------------------------------------

class Circle:
    """One animated circle with a label, fading out over its lifetime."""

    def __init__(self, x: int, y: int, label: str = "", lifetime_ms: int = 4000):
        self.x = x
        self.y = y
        self.label = label
        self.opacity = 1.0
        self.radius = 48
        self.lifetime_ms = lifetime_ms
        self.elapsed = 0

    @property
    def alive(self) -> bool:
        return self.elapsed < self.lifetime_ms

    def tick(self, dt_ms: int):
        self.elapsed += dt_ms
        # Start fading after 60 % of lifetime
        fade_start = self.lifetime_ms * 0.6
        if self.elapsed > fade_start:
            remaining = self.lifetime_ms - fade_start
            self.opacity = max(0.0, 1.0 - (self.elapsed - fade_start) / remaining)

    def paint(self, painter: QPainter):
        if self.opacity <= 0:
            return

        alpha = self.opacity

        # Outer glow (thick, soft)
        glow = QColor(255, 40, 40, int(50 * alpha))
        painter.setPen(QPen(glow, 10))
        painter.setBrush(Qt.NoBrush)
        painter.drawEllipse(
            QPoint(self.x, self.y), self.radius + 12, self.radius + 12
        )

        # Middle ring
        mid = QColor(255, 50, 50, int(120 * alpha))
        painter.setPen(QPen(mid, 5))
        painter.drawEllipse(
            QPoint(self.x, self.y), self.radius + 4, self.radius + 4
        )

        # Main circle
        color = QColor(255, 20, 20, int(230 * alpha))
        painter.setPen(QPen(color, 4))
        painter.drawEllipse(QPoint(self.x, self.y), self.radius, self.radius)

        # Label with dark background for readability
        if self.label:
            font = QFont("Segoe UI", 11, QFont.Bold)
            painter.setFont(font)

            text_w, text_h = 260, 26
            text_x = self.x - text_w // 2
            text_y = self.y + self.radius + 14

            # Semi-transparent dark background behind text
            bg = QColor(0, 0, 0, int(170 * alpha))
            painter.setPen(Qt.NoPen)
            painter.setBrush(QBrush(bg))
            painter.drawRoundedRect(
                QRectF(text_x, text_y, text_w, text_h), 6, 6
            )

            # White text
            text_color = QColor(255, 255, 255, int(240 * alpha))
            painter.setPen(QPen(text_color))
            painter.drawText(
                text_x, text_y, text_w, text_h,
                Qt.AlignCenter, self.label,
            )


# ---------------------------------------------------------------------------
# Overlay window
# ---------------------------------------------------------------------------

class OverlayWindow(QWidget):
    def __init__(self):
        super().__init__()

        # Full-screen, frameless, transparent, click-through, always on top
        self.setWindowFlags(
            Qt.FramelessWindowHint
            | Qt.WindowStaysOnTopHint
            | Qt.Tool                     # Hides from taskbar
            | Qt.WindowTransparentForInput # Click-through on Windows
        )
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WA_ShowWithoutActivating)

        # Cover the FULL primary screen (using physical pixels thanks to DPI awareness)
        screen = QApplication.primaryScreen().geometry()
        self.setGeometry(screen)
        self._screen_w = screen.width()
        self._screen_h = screen.height()

        print(f"[overlay] Screen geometry: {self._screen_w}x{self._screen_h} "
              f"at ({screen.x()}, {screen.y()})")

        self.circles: list[Circle] = []
        self._connected = False

        # 30 fps repaint timer
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(33)  # ~30 fps

    # -- slots ---------------------------------------------------------------

    def set_connected(self, connected: bool):
        self._connected = connected
        self.update()

    def add_circle(self, data: dict):
        x = data.get("x", 0)
        y = data.get("y", 0)
        label = data.get("label", "")
        print(f"[overlay] 🔴 Drawing circle at ({x}, {y})  label='{label}'")
        self.circles.append(Circle(x, y, label))

    def run_self_test(self):
        """Draw 4 test circles at known positions to verify rendering works."""
        w, h = self._screen_w, self._screen_h
        test_points = [
            {"x": 200,      "y": 200,      "label": "Top-Left test"},
            {"x": w - 200,  "y": 200,      "label": "Top-Right test"},
            {"x": w // 2,   "y": h // 2,   "label": "Center test"},
            {"x": 200,      "y": h - 200,  "label": "Bottom-Left test"},
        ]
        for pt in test_points:
            self.add_circle(pt)

    # -- internal ------------------------------------------------------------

    def _tick(self):
        dt = 33  # ms per frame
        for c in self.circles:
            c.tick(dt)
        # Remove dead circles
        self.circles = [c for c in self.circles if c.alive]
        self.update()

    # -- painting ------------------------------------------------------------

    def paintEvent(self, _event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Tiny connection indicator in bottom-right corner
        dot_color = QColor(59, 185, 80) if self._connected else QColor(248, 81, 73)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(dot_color))
        painter.drawEllipse(
            QPoint(self._screen_w - 12, self._screen_h - 12), 5, 5
        )

        for circle in self.circles:
            circle.paint(painter)
        painter.end()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    self_test = "--test" in sys.argv

    app = QApplication(sys.argv)

    overlay = OverlayWindow()
    overlay.show()

    if self_test:
        # ── Self-test mode: draw circles immediately, no WS needed ──
        print("[overlay] *** SELF-TEST MODE — drawing 4 test circles ***")
        # Delay slightly so the window is fully visible
        QTimer.singleShot(500, overlay.run_self_test)
    else:
        # ── Normal mode: connect to agent's WebSocket ──
        signal = JsonSignal()
        signal.received.connect(overlay.add_circle)
        signal.status_changed.connect(overlay.set_connected)

        ws_thread = threading.Thread(
            target=ws_listener, args=(signal,), daemon=True
        )
        ws_thread.start()

    print("[overlay] Overlay is running. Waiting for draw commands …")
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
