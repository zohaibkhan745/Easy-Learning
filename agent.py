import asyncio
import io
import json
import logging
import ctypes
import os
import time

import av
import httpx
import websockets
from PIL import Image
from dotenv import load_dotenv
from vision_agents.core import User, Agent, AgentLauncher, Runner
from vision_agents.plugins import getstream, gemini

# ---------------------------------------------------------------------------
# Logging — keep it clean
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("ai-tutor")

# Suppress noisy loggers that flood the console
logging.getLogger("vision_agents.core.utils.audio_queue").setLevel(logging.ERROR)
logging.getLogger("asyncio").setLevel(logging.CRITICAL)

# Enable debug logging for Gemini realtime plugin to diagnose audio issues
logging.getLogger("vision_agents.plugins.gemini.gemini_realtime").setLevel(logging.DEBUG)
logging.getLogger("vision_agents.core.agents.agents").setLevel(logging.DEBUG)

load_dotenv()

# ---------------------------------------------------------------------------
# Screen resolution (Windows physical pixels)
# ---------------------------------------------------------------------------
def _get_screen_resolution() -> tuple[int, int]:
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
        user32 = ctypes.windll.user32
        return (user32.GetSystemMetrics(0), user32.GetSystemMetrics(1))
    except Exception:
        return (1920, 1080)

SCREEN_W, SCREEN_H = _get_screen_resolution()
logger.info(f"Screen resolution: {SCREEN_W}x{SCREEN_H}")

# ---------------------------------------------------------------------------
# Frame optimization — the KEY fix for 1006 keepalive timeout
#
# The original code drew a heavy grid overlay on every full-res frame
# (1920x1080 PNG with PIL grid ≈ 2-5 MB per frame) which overwhelmed
# the upload bandwidth to Gemini, causing WebSocket keepalive pings to
# time out → 1006 abnormal closure flood.
#
# Fix: resize frames to a much smaller resolution.  The LLM only needs
# enough detail to recognise UI elements; 1024x576 is plenty and yields
# ~200-400 KB PNGs instead of multi-MB ones.
# ---------------------------------------------------------------------------
FRAME_MAX_DIM = 1024  # max width or height (maintains aspect ratio)

_frame_count = 0
# Track the actual frame dimensions Gemini sees (updated each frame)
FRAME_W = 1024
FRAME_H = 576
# Cache the latest captured frame for vision-based element detection
_latest_frame_png: bytes | None = None

def _optimized_frame_to_png(frame: av.VideoFrame) -> bytes:
    """Resize frame to reduce bandwidth, then encode as PNG."""
    global _frame_count, FRAME_W, FRAME_H, _latest_frame_png
    _frame_count += 1

    if hasattr(frame, "to_image"):
        img = frame.to_image()
    else:
        import numpy as np
        arr = frame.to_ndarray(format="rgb24")
        img = Image.fromarray(arr)

    w, h = img.size

    # Downscale if larger than target
    if w > FRAME_MAX_DIM or h > FRAME_MAX_DIM:
        ratio = min(FRAME_MAX_DIM / w, FRAME_MAX_DIM / h)
        new_w, new_h = int(w * ratio), int(h * ratio)
        img = img.resize((new_w, new_h), Image.LANCZOS)

    # Track the dimensions Gemini actually sees
    FRAME_W, FRAME_H = img.size

    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    png_bytes = buf.getvalue()

    # Cache for vision-based element detection
    _latest_frame_png = png_bytes

    # Occasional diagnostic (every 30 frames)
    if _frame_count == 1 or _frame_count % 30 == 0:
        sz_kb = len(png_bytes) / 1024
        logger.info(
            f"Frame #{_frame_count}: {w}x{h} -> {img.size[0]}x{img.size[1]}, "
            f"{sz_kb:.0f} KB"
        )

    return png_bytes


def _patch_frame_encoder():
    """Replace the default full-res PNG encoder with our optimized version."""
    from vision_agents.plugins.gemini.gemini_realtime import realtime as _rt_mod
    from vision_agents.core.utils import video_utils
    from vision_agents.plugins.gemini import gemini_realtime

    video_utils.frame_to_png_bytes = _optimized_frame_to_png
    _rt_mod.frame_to_png_bytes = _optimized_frame_to_png
    if hasattr(gemini_realtime, "frame_to_png_bytes"):
        gemini_realtime.frame_to_png_bytes = _optimized_frame_to_png

    logger.info(f"Frame encoder patched: max dimension {FRAME_MAX_DIM}px")


# ---------------------------------------------------------------------------
# Reconnect patch — handle 1006 (keepalive timeout / abnormal closure)
#
# The library only reconnects on codes 1011-1014.  Code 1006 means the
# connection dropped unexpectedly (network glitch, transient overload).
# We patch _should_reconnect to also retry on 1006.
# ---------------------------------------------------------------------------
def _patch_reconnect():
    try:
        import vision_agents.plugins.gemini.gemini_realtime as _gmod

        _original = _gmod._should_reconnect

        def _patched_should_reconnect(exc):
            if _original(exc):
                return True
            # Also reconnect on 1006 (abnormal closure / keepalive timeout)
            if (
                isinstance(exc, websockets.ConnectionClosedError)
                and exc.rcvd
                and exc.rcvd.code == 1006
            ):
                logger.warning("Got 1006 (abnormal closure) — will reconnect")
                return True
            return False

        _gmod._should_reconnect = _patched_should_reconnect
        logger.info("Reconnect patch applied: 1006 now triggers reconnection")
    except Exception as e:
        logger.warning(f"Could not patch reconnect logic: {e}")


# ---------------------------------------------------------------------------
# GetStream timeout patch — prevent ConnectTimeout on slow networks
#
# The default httpx timeout is too aggressive (5s connect).  We increase
# it to 30s so the TLS handshake has time to complete even on congested
# connections.
# ---------------------------------------------------------------------------
def _patch_getstream_timeout():
    """Increase httpx timeouts on the GetStream AsyncStream client."""
    try:
        from vision_agents.plugins.getstream.stream_edge_transport import StreamEdge
        _orig_init = StreamEdge.__init__

        def _patched_init(self, **kwargs):
            _orig_init(self, **kwargs)
            # Replace the default httpx timeout with a much more generous one
            # The original client is at self.client (AsyncStream) → .client (httpx.AsyncClient)
            self.client.client.timeout = httpx.Timeout(60.0, connect=30.0)
            logger.info("GetStream timeout patched: 60s total, 30s connect")

        StreamEdge.__init__ = _patched_init
        logger.info("GetStream timeout patch ready")
    except Exception as e:
        logger.warning(f"Could not patch GetStream timeout: {e}")


# ---------------------------------------------------------------------------
# WebSocket broadcast server (port 8765)
# ---------------------------------------------------------------------------
OVERLAY_WS_PORT = 8765
connected_clients: set = set()
highlight_log: list = []


async def _ws_handler(ws, *args):
    """Accept overlay / dashboard connections."""
    connected_clients.add(ws)
    remote = ws.remote_address
    logger.info(f"WS client connected: {remote}")

    # Send recent highlight history to new clients
    for entry in highlight_log[-10:]:
        try:
            await ws.send(json.dumps(entry))
        except Exception:
            pass

    try:
        async for msg in ws:
            try:
                data = json.loads(msg)
                if data.get("action") == "status":
                    await ws.send(json.dumps({
                        "action": "status",
                        "clients": len(connected_clients),
                        "screen": f"{SCREEN_W}x{SCREEN_H}",
                        "highlights": len(highlight_log),
                    }))
            except Exception:
                pass
    finally:
        connected_clients.discard(ws)
        logger.info(f"WS client disconnected: {remote}")


async def broadcast_draw(x: int, y: int, label: str = ""):
    """Send a draw command to every connected client (overlay + dashboard)."""
    payload = {"action": "draw", "x": x, "y": y, "label": label}
    highlight_log.append(payload)
    if len(highlight_log) > 50:
        highlight_log.pop(0)

    msg = json.dumps(payload)
    if connected_clients:
        await asyncio.gather(
            *(c.send(msg) for c in connected_clients),
            return_exceptions=True,
        )
        logger.info(f"Broadcast draw to {len(connected_clients)} client(s): {msg}")
    else:
        logger.warning("No overlay clients connected — draw not delivered")


async def start_ws_server():
    server = await websockets.serve(_ws_handler, "localhost", OVERLAY_WS_PORT)
    logger.info(f"WebSocket server on ws://localhost:{OVERLAY_WS_PORT}")
    return server


# ---------------------------------------------------------------------------
# System Instructions
# ---------------------------------------------------------------------------
SYSTEM_INSTRUCTIONS = """\
You are a friendly real-time VS Code tutor. You can see the student's screen.
You CAN highlight and circle UI elements on the student's screen automatically
when you mention them — the system handles it for you behind the scenes.

When the student asks a question:
1. SPEAK your answer naturally — explain what to do step by step.
2. Be specific about UI elements: name them clearly (e.g. "the Extensions
   icon on the left sidebar", "the Search box at the top of the sidebar").
   A red circle will automatically appear on those elements for the student.
3. When describing a click target, name the exact icon or button so the
   student can find it easily.

Keep responses concise, helpful, and conversational. If you cannot see the
screen clearly, ask the student to share their screen.
Never say "I can't highlight" — you CAN and DO highlight elements automatically.
"""


# ---------------------------------------------------------------------------
# Transcript-based auto-highlighting
#
# Since the Gemini Live API's native function calling emits <ctrl46>
# silence tokens instead of proper tool calls, we bypass tool registration
# entirely and detect UI element mentions from the AI's speech transcript.
# When a known VS Code element is mentioned, we auto-draw an overlay
# circle at its approximate screen position.
# ---------------------------------------------------------------------------
import re as _re

# Keyword → (screen_x, screen_y) for common VS Code UI elements.
# Coordinates are in real screen pixels (SCREEN_W x SCREEN_H).
# Longer / more-specific keys are checked first (see _match_element).
VSCODE_ELEMENTS: dict[str, tuple[int, int]] = {
    # Activity bar icons (left column, ~x=24)
    "extensions icon":    (24, 280),
    "extensions sidebar": (24, 280),
    "extension icon":     (24, 280),
    "extensions":         (24, 280),
    "extension":          (24, 280),
    "marketplace":        (24, 280),
    "file explorer":      (24, 55),
    "explorer icon":      (24, 55),
    "explorer":           (24, 55),
    "search icon":        (24, 100),
    "source control":     (24, 145),
    "git icon":           (24, 145),
    "debug icon":         (24, 190),
    "run and debug":      (24, 190),
    "debug":              (24, 190),
    "accounts":           (24, 1010),
    "gear icon":          (24, 1048),
    "manage icon":        (24, 1048),
    "settings gear":      (24, 1048),
    "settings icon":      (24, 1048),
    # Sidebar content area (~x=200)
    "sidebar":            (200, 400),
    "side bar":           (200, 400),
    "side panel":         (200, 400),
    "search box":         (200, 120),
    "search bar":         (200, 120),
    "search field":       (200, 120),
    # Editor area (centre)
    "code editor":        (960, 400),
    "editor area":        (960, 400),
    "editor":             (960, 400),
    # Tab bar
    "tab bar":            (500, 55),
    "tab":                (500, 55),
    # Terminal / bottom panel
    "integrated terminal": (960, 900),
    "terminal panel":     (960, 900),
    "terminal":           (960, 900),
    "bottom panel":       (960, 850),
    "output panel":       (960, 850),
    "problems panel":     (960, 850),
    # Top area
    "command palette":    (960, 80),
    "quick open":         (960, 80),
    "menu bar":           (100, 10),
    "title bar":          (960, 10),
    # Status bar
    "status bar":         (960, 1065),
    # Settings page
    "settings":           (960, 400),
    # Install button in Extensions view
    "install button":     (350, 300),
}

# Pre-sort by key length (longest first) so more specific keys match before
# shorter generic ones  (e.g. "extensions icon" before "extension").
_SORTED_ELEMENTS = sorted(VSCODE_ELEMENTS.items(), key=lambda kv: -len(kv[0]))

_HIGHLIGHT_COOLDOWN: dict[str, float] = {}  # keyword → last-highlight timestamp
HIGHLIGHT_MIN_INTERVAL = 8.0  # seconds before re-highlighting the same element


# ---------------------------------------------------------------------------
# Gemini Vision–based element locator
#
# Instead of using hardcoded coordinates, we send the current screen
# frame to the Gemini Vision API and ask it to locate the UI element.
# This gives accurate, real-time coordinates that adapt to the actual
# VS Code layout, window position, sidebar width, etc.
# ---------------------------------------------------------------------------

_genai_client = None
# Cache: keyword → (timestamp, screen_x, screen_y)
_vision_cache: dict[str, tuple[float, int, int]] = {}
VISION_CACHE_TTL = 15.0  # seconds to cache vision results


def _get_genai_client():
    """Lazy-initialize the Google GenAI client for vision API calls."""
    global _genai_client
    if _genai_client is None:
        from google import genai
        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not api_key:
            logger.error("No GOOGLE_API_KEY or GEMINI_API_KEY found for vision detection")
            return None
        _genai_client = genai.Client(api_key=api_key)
        logger.info("Gemini Vision client initialized for element detection")
    return _genai_client


async def _locate_element_vision(element_name: str) -> tuple[int, int] | None:
    """
    Use Gemini Vision API to find the exact screen position of a UI element
    in the latest captured frame.

    Returns (screen_x, screen_y) in real screen pixels, or None if not found.
    """
    # Check cache first
    now = time.time()
    cache_key = element_name.lower().strip()
    if cache_key in _vision_cache:
        ts, cx, cy = _vision_cache[cache_key]
        if now - ts < VISION_CACHE_TTL:
            logger.info(f"Vision cache hit for '{element_name}': ({cx}, {cy})")
            return (cx, cy)

    if not _latest_frame_png:
        logger.warning("No frame available for vision-based element detection")
        return None

    client = _get_genai_client()
    if not client:
        return None

    try:
        from google.genai import types as genai_types

        prompt = (
            f"You are a precise UI element locator. Look at this VS Code screenshot "
            f"(image size: {FRAME_W}x{FRAME_H} pixels).\n\n"
            f"Find the EXACT center pixel coordinates of: \"{element_name}\"\n\n"
            f"Rules:\n"
            f"- Return ONLY a JSON object: {{\"x\": <number>, \"y\": <number>}}\n"
            f"- x and y must be integers, pixel positions within the {FRAME_W}x{FRAME_H} image\n"
            f"- Point to the CENTER of the element (icon, button, panel, etc.)\n"
            f"- If the element is not visible, return {{\"x\": -1, \"y\": -1}}\n"
            f"- No explanation, ONLY the JSON object"
        )

        response = await client.aio.models.generate_content(
            model="gemini-2.0-flash",
            contents=[
                genai_types.Part.from_bytes(
                    data=_latest_frame_png, mime_type="image/png"
                ),
                prompt,
            ],
        )

        text = response.text.strip()
        logger.debug(f"Vision API response for '{element_name}': {text[:200]}")

        # Extract JSON coordinates from response
        match = _re.search(
            r'\{\s*"x"\s*:\s*(-?\d+(?:\.\d+)?)\s*,\s*"y"\s*:\s*(-?\d+(?:\.\d+)?)\s*\}',
            text,
        )
        if not match:
            # Try alternate format: {"y": ..., "x": ...}
            match = _re.search(
                r'\{\s*"y"\s*:\s*(-?\d+(?:\.\d+)?)\s*,\s*"x"\s*:\s*(-?\d+(?:\.\d+)?)\s*\}',
                text,
            )
            if match:
                fy = float(match.group(1))
                fx = float(match.group(2))
            else:
                logger.warning(f"Vision: Could not parse coords from: {text[:200]}")
                return None
        else:
            fx = float(match.group(1))
            fy = float(match.group(2))

        if fx < 0 or fy < 0:
            logger.info(f"Vision: '{element_name}' not visible in current frame")
            return None

        # Clamp to frame bounds
        fx = max(0, min(fx, FRAME_W))
        fy = max(0, min(fy, FRAME_H))

        # Scale from frame coordinates to real screen coordinates
        sx = int(fx * SCREEN_W / FRAME_W)
        sy = int(fy * SCREEN_H / FRAME_H)

        # Cache the result
        _vision_cache[cache_key] = (now, sx, sy)
        logger.info(
            f"✅ Vision located '{element_name}': "
            f"frame({fx:.0f},{fy:.0f}) → screen({sx},{sy})"
        )
        return (sx, sy)

    except asyncio.TimeoutError:
        logger.warning(f"Vision detection timed out for '{element_name}'")
        return None
    except Exception as e:
        logger.error(f"Vision detection failed for '{element_name}': {e}")
        return None


async def _check_transcript_highlights(phrase: str):
    """Scan a phrase of AI speech for VS Code UI element keywords.
    Use Gemini Vision to locate each element accurately, with hardcoded
    coordinates as a fallback."""
    phrase_lower = phrase.lower()
    now = time.time()
    matched: set[str] = set()
    for keyword, fallback_coords in _SORTED_ELEMENTS:
        if keyword in phrase_lower and keyword not in matched:
            # Cooldown: don't re-highlight the same element too often
            if now - _HIGHLIGHT_COOLDOWN.get(keyword, 0) < HIGHLIGHT_MIN_INTERVAL:
                continue
            _HIGHLIGHT_COOLDOWN[keyword] = now
            matched.add(keyword)
            label = keyword.title()

            # Try vision-based detection first (with timeout)
            try:
                coords = await asyncio.wait_for(
                    _locate_element_vision(keyword), timeout=5.0
                )
            except asyncio.TimeoutError:
                coords = None
                logger.warning(f"Vision timeout for '{label}', using fallback")

            if coords:
                sx, sy = coords
            else:
                # Fallback to hardcoded coordinates
                sx, sy = fallback_coords
                logger.info(f"Using fallback coords for '{label}': ({sx},{sy})")

            logger.info(
                f"🔴 Auto-highlight: '{label}' at screen ({sx},{sy}) "
                f"[from transcript: ...{phrase[-60:].strip()}]"
            )
            await broadcast_draw(sx, sy, label=label)
            # Skip sub-matches (e.g. after matching "extensions icon",
            # don't also match "extension")
            phrase_lower = phrase_lower.replace(keyword, "", 1)


# ---------------------------------------------------------------------------
# Agent factory
# ---------------------------------------------------------------------------
async def create_agent(**kwargs) -> Agent:
    _patch_frame_encoder()
    _patch_reconnect()
    _patch_getstream_timeout()

    # NO tools registered with Gemini.  The Live API native-audio models
    # emit <ctrl46> silence tokens when function_declarations are present
    # in the config, so we rely on transcript-based auto-highlighting
    # instead (see _check_transcript_highlights).
    llm = gemini.Realtime()
    logger.info("Gemini Realtime created (no tools — using transcript highlights)")

    return Agent(
        edge=getstream.Edge(),
        agent_user=User(name="VS Code Tutor", id="agent"),
        instructions=SYSTEM_INSTRUCTIONS,
        llm=llm,
    )


# ---------------------------------------------------------------------------
# Call handler
# ---------------------------------------------------------------------------

# Audio monitoring state
_audio_output_count = 0
_last_audio_output_time = 0.0
_audio_input_count = 0
_last_audio_input_time = 0.0


async def join_call(agent: Agent, call_type: str, call_id: str, **kwargs):
    global _audio_output_count, _last_audio_output_time
    global _audio_input_count, _last_audio_input_time

    # Retry user/call creation with exponential backoff
    for attempt in range(1, 4):
        try:
            await agent.create_user()
            call = await agent.create_call(call_type, call_id)
            break
        except (httpx.ConnectTimeout, httpx.ReadTimeout, httpx.ConnectError) as e:
            if attempt == 3:
                logger.error(f"Failed to create call after 3 attempts: {e}")
                raise
            wait = 2 ** attempt
            logger.warning(f"Attempt {attempt}/3 failed ({type(e).__name__}), retrying in {wait}s…")
            await asyncio.sleep(wait)
    ws_server = await start_ws_server()

    logger.info(
        "\n"
        "═══════════════════════════════════════════════════════════\n"
        "  AI Tutor Agent starting...\n"
        "\n"
        "  STEPS TO USE:\n"
        "  1. Open the demo URL (printed above) in your browser\n"
        "  2. Click 'Share Screen' -> share your ENTIRE SCREEN\n"
        "  3. IMPORTANT: Enable your MICROPHONE in the browser!\n"
        "  4. Run:  python overlay.py   (in another terminal)\n"
        "  5. Open dashboard.html in browser for live status\n"
        "  6. Start talking to the AI tutor!\n"
        "═══════════════════════════════════════════════════════════"
    )

    try:
        logger.info("Joining call... (connecting to Gemini + GetStream)")
        async with agent.join(call):
            logger.info("✅ Agent joined call successfully!")

            # ---------------------------------------------------------------
            # Subscribe to audio INPUT (mic from browser) for diagnostics
            # ---------------------------------------------------------------
            try:
                from vision_agents.core.events import AudioReceivedEvent

                @agent.edge.events.subscribe
                async def _on_audio_input(event: AudioReceivedEvent):
                    global _audio_input_count, _last_audio_input_time
                    _audio_input_count += 1
                    _last_audio_input_time = time.time()
                    if _audio_input_count == 1:
                        logger.info(
                            "🎤 First mic audio received from browser! "
                            "Microphone pipeline is working."
                        )
                    elif _audio_input_count % 500 == 0:
                        logger.info(f"🎤 Mic audio chunks received: {_audio_input_count}")

                logger.info("Audio INPUT monitor subscribed")
            except Exception as e:
                logger.warning(f"Could not subscribe audio input monitor: {e}")

            # ---------------------------------------------------------------
            # Subscribe to audio OUTPUT (Gemini speaking) for diagnostics
            # ---------------------------------------------------------------
            try:
                from vision_agents.core.llm.events import RealtimeAudioOutputEvent

                @agent.events.subscribe
                async def _on_audio_output(event: RealtimeAudioOutputEvent):
                    global _audio_output_count, _last_audio_output_time
                    _audio_output_count += 1
                    _last_audio_output_time = time.time()
                    if _audio_output_count <= 3 or _audio_output_count % 100 == 0:
                        logger.info(
                            f"🔊 Audio output #{_audio_output_count} from Gemini"
                        )

                logger.info("Audio OUTPUT monitor subscribed")
            except Exception as e:
                logger.warning(f"Could not subscribe audio output monitor: {e}")

            # ---------------------------------------------------------------
            # Subscribe to transcription events + <ctrl46> recovery
            #
            # Gemini's native-audio models often emit <ctrl46> control
            # tokens instead of proper speech / tool-calls when handling
            # task-oriented questions.  When we detect 2+ consecutive
            # <ctrl46> tokens we re-send the user's last utterance as a
            # TEXT prompt via send_client_content.  Text-mode prompts
            # bypass the audio↔tool-calling conflict and usually succeed.
            # ---------------------------------------------------------------
            _utterance_buf: list[str] = []          # accumulated user speech
            _ai_phrase_buf: list[str] = []           # accumulated AI speech for highlight matching
            _ctrl46 = {"n": 0, "busy": False, "ts": 0.0}
            CTRL46_COOLDOWN = 12  # seconds between recovery attempts

            async def _ctrl46_recovery():
                """Re-send user question as text when <ctrl46> detected."""
                if _ctrl46["busy"]:
                    return
                now = time.time()
                if now - _ctrl46["ts"] < CTRL46_COOLDOWN:
                    return
                question = "".join(_utterance_buf).strip()
                if not question:
                    return

                _ctrl46["busy"] = True
                _ctrl46["ts"] = now
                try:
                    # Brief pause — if real speech follows, abort
                    await asyncio.sleep(2)
                    if _ctrl46["n"] < 2:
                        return  # model recovered on its own

                    logger.warning(
                        f"⚠️  <ctrl46> recovery: re-sending as text: "
                        f"\"{question[-120:]}\""
                    )
                    from google.genai import types as genai_types
                    recovery = (
                        f'The student just asked: "{question}"\n\n'
                        "Respond verbally with a clear, step-by-step answer. "
                        "For every UI element you mention, call highlight_ui "
                        "with your best x,y estimate from the video frame."
                    )
                    await agent.llm._session.send_client_content(
                        turns=genai_types.Content(
                            role="user",
                            parts=[genai_types.Part(text=recovery)],
                        ),
                        turn_complete=True,
                    )
                    logger.info("✅ Recovery text sent via send_client_content")
                    _utterance_buf.clear()
                except Exception as e:
                    logger.error(f"<ctrl46> recovery failed: {e}")
                finally:
                    _ctrl46["busy"] = False
                    _ctrl46["n"] = 0

            try:
                from vision_agents.core.llm.events import (
                    RealtimeUserSpeechTranscriptionEvent,
                    RealtimeAgentSpeechTranscriptionEvent,
                )

                @agent.events.subscribe
                async def _on_user_transcript(event: RealtimeUserSpeechTranscriptionEvent):
                    logger.info(f"🎤 User said: \"{event.text}\"")
                    _utterance_buf.append(event.text)
                    # Keep buffer from growing unbounded
                    if len("".join(_utterance_buf)) > 500:
                        merged = "".join(_utterance_buf)[-300:]
                        _utterance_buf.clear()
                        _utterance_buf.append(merged)

                @agent.events.subscribe
                async def _on_agent_transcript(event: RealtimeAgentSpeechTranscriptionEvent):
                    logger.info(f"🤖 AI said: \"{event.text}\"")
                    if "<ctrl46>" in event.text:
                        _ctrl46["n"] += 1
                        if _ctrl46["n"] >= 2:
                            asyncio.create_task(_ctrl46_recovery())
                    else:
                        # Real speech → model is responding, reset counter
                        _ctrl46["n"] = 0
                        _utterance_buf.clear()

                        # Accumulate AI speech for highlight matching
                        _ai_phrase_buf.append(event.text)
                        phrase = "".join(_ai_phrase_buf)
                        # Check for highlights on sentence boundaries
                        if any(phrase.rstrip().endswith(c) for c in ".!?,;:"):
                            asyncio.create_task(
                                _check_transcript_highlights(phrase)
                            )
                            _ai_phrase_buf.clear()
                        # Safety: flush long buffers
                        elif len(phrase) > 300:
                            asyncio.create_task(
                                _check_transcript_highlights(phrase)
                            )
                            _ai_phrase_buf.clear()

                logger.info("Transcription monitors + <ctrl46> recovery subscribed")
            except Exception as e:
                logger.warning(f"Could not subscribe transcription monitors: {e}")

            # (Tool call monitors removed — no tools registered with Gemini.
            #  Highlights are driven by _check_transcript_highlights instead.)

            # ---------------------------------------------------------------
            # Verify Gemini session is connected
            # ---------------------------------------------------------------
            llm = agent.llm
            if hasattr(llm, "connected"):
                logger.info(f"Gemini connected: {llm.connected}")
            if hasattr(llm, "_real_session") and llm._real_session:
                logger.info("Gemini session: active ✅")
            else:
                logger.warning("Gemini session: NOT active ⚠️")

            # ---------------------------------------------------------------
            # Send the initial greeting using send_client_content which is
            # more reliable for triggering the first response than
            # send_realtime_input(text=...), because it guarantees turn
            # completion semantics.
            # ---------------------------------------------------------------
            logger.info("Waiting 3s for pipeline to stabilize before greeting...")
            await asyncio.sleep(3)

            GREETING_TEXT = (
                "Greet the student warmly. Tell them you can see their "
                "screen and you're ready to help them learn VS Code. "
                "Ask what they'd like to learn today. Speak in a "
                "friendly conversational tone."
            )

            for greeting_attempt in range(1, 4):
                logger.info(f"Sending greeting (attempt {greeting_attempt}/3)...")
                try:
                    from google.genai import types as genai_types
                    await llm._session.send_client_content(
                        turns=genai_types.Content(
                            role="user",
                            parts=[genai_types.Part(text=GREETING_TEXT)],
                        ),
                        turn_complete=True,
                    )
                    logger.info("Greeting sent via send_client_content ✅")
                except Exception as e:
                    logger.error(f"Greeting failed: {e}")
                    if greeting_attempt < 3:
                        await asyncio.sleep(2)
                    continue

                # Wait up to 10 seconds for audio output
                for _ in range(20):
                    await asyncio.sleep(0.5)
                    if _audio_output_count > 0:
                        logger.info(
                            "🎉 Gemini is speaking! Audio output pipeline working."
                        )
                        break
                else:
                    if greeting_attempt < 3:
                        logger.warning("No audio output after greeting — retrying...")
                        continue
                    logger.warning(
                        "⚠️ No audio output from Gemini after 3 attempts."
                    )
                break

            # ---------------------------------------------------------------
            # After greeting, check mic audio health and warn if needed
            # ---------------------------------------------------------------
            await asyncio.sleep(5)
            if _audio_input_count == 0:
                logger.warning(
                    "\n"
                    "══════════════════════════════════════════════════════\n"
                    "  ⚠️  NO MICROPHONE AUDIO RECEIVED FROM BROWSER!\n"
                    "\n"
                    "  The AI can't hear you. Please check:\n"
                    "  1. Browser mic permission is ALLOWED\n"
                    "  2. Correct microphone selected in browser\n"
                    "  3. Mic is not muted in OS sound settings\n"
                    "  4. Try refreshing the demo page\n"
                    "══════════════════════════════════════════════════════"
                )
            else:
                logger.info(
                    f"✅ Mic audio flowing: {_audio_input_count} chunks received. "
                    f"Speak to the AI — it should respond!"
                )

            # ---------------------------------------------------------------
            # Background monitor — log pipeline health periodically
            # ---------------------------------------------------------------
            async def _audio_monitor():
                while True:
                    await asyncio.sleep(30)
                    out_elapsed = (
                        f"{time.time() - _last_audio_output_time:.0f}s ago"
                        if _last_audio_output_time > 0
                        else "never"
                    )
                    in_elapsed = (
                        f"{time.time() - _last_audio_input_time:.0f}s ago"
                        if _last_audio_input_time > 0
                        else "never"
                    )
                    logger.info(
                        f"📊 Pipeline: "
                        f"mic_in={_audio_input_count} (last: {in_elapsed}), "
                        f"ai_out={_audio_output_count} (last: {out_elapsed})"
                    )

            monitor_task = asyncio.create_task(_audio_monitor())

            try:
                await asyncio.Event().wait()
            finally:
                monitor_task.cancel()
    finally:
        ws_server.close()
        await ws_server.wait_closed()
        logger.info("WebSocket server shut down.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    Runner(
        AgentLauncher(create_agent=create_agent, join_call=join_call)
    ).cli()