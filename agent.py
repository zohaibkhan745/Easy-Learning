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

def _optimized_frame_to_png(frame: av.VideoFrame) -> bytes:
    """Resize frame to reduce bandwidth, then encode as PNG."""
    global _frame_count
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

    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    png_bytes = buf.getvalue()

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
SYSTEM_INSTRUCTIONS = f"""\
You are a real-time VS Code tutor. You watch the student's screen and guide them.

CRITICAL RULE — YOU MUST CALL highlight_ui FOR EVERY UI ELEMENT YOU MENTION:
You have a tool called highlight_ui. Every single time you mention, describe,
or refer to ANY visible element on screen, you MUST call highlight_ui to draw
a red circle on it. This is your #1 priority. The student cannot learn without
seeing exactly where to look.

NEVER just say "click the Extensions icon" without ALSO calling highlight_ui.
ALWAYS call the tool FIRST, then explain. If you mention 3 elements, call
highlight_ui 3 separate times.

SCREEN COORDINATES:
- Resolution: {SCREEN_W}x{SCREEN_H} pixels
- (0,0) = top-left. X goes right, Y goes down.
- You see a scaled-down video — estimate coordinates on the FULL
  {SCREEN_W}x{SCREEN_H} screen based on proportional position.

VS CODE LAYOUT REFERENCE (approximate pixel positions):
- Activity Bar (left icon strip): x=25, y varies per icon
  - Explorer icon: (25, 50)
  - Search icon: (25, 100)
  - Source Control icon: (25, 150)
  - Run/Debug icon: (25, 200)
  - Extensions icon: (25, 250)
- Side Bar / Explorer panel: x=150, y=300 (middle of panel)
- Editor tabs row: y=35, first tab x=350
- Editor text area: center = ({SCREEN_W // 2}, {SCREEN_H // 2})
- Terminal panel: y={int(SCREEN_H * 0.75)}, x={SCREEN_W // 2}
- Status Bar: y={SCREEN_H - 12}, x={SCREEN_W // 2}
- Menu Bar: y=10

EXAMPLES OF CORRECT BEHAVIOR:
- User asks "how do I install an extension?":
  -> Call highlight_ui("Extensions icon", 25, 250)
  -> Say "Click this Extensions icon on the left sidebar"
  -> Call highlight_ui("Search box", 150, 80)
  -> Say "Then type the extension name in this search box"
  -> Call highlight_ui("Install button", 200, 300)
  -> Say "And click Install"

- User asks "where is the terminal?":
  -> Call highlight_ui("Terminal menu", 350, 10)
  -> Say "Go to Terminal in the menu bar"
  -> Call highlight_ui("Terminal panel", {SCREEN_W // 2}, {int(SCREEN_H * 0.78)})
  -> Say "Or look at the bottom panel here"

TEACHING STYLE:
- Be concise and friendly
- ALWAYS point first (call highlight_ui), then explain
- For multi-step tasks, highlight each step's target element
- If you can't see the screen clearly, ask to share screen
"""


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

# Explicit JSON schemas so Gemini gets full parameter descriptions
# (the default registry only extracts types, not descriptions from docstrings)
HIGHLIGHT_UI_SCHEMA = {
    "type": "object",
    "properties": {
        "element_name": {
            "type": "string",
            "description": (
                "Short label for the UI element being highlighted, "
                "e.g. 'Extensions icon', 'Run button', 'Terminal panel', 'Search box'"
            ),
        },
        "x": {
            "type": "integer",
            "description": (
                f"Horizontal pixel coordinate on the {SCREEN_W}x{SCREEN_H} screen. "
                f"0 = left edge, {SCREEN_W} = right edge. "
                "Estimate the CENTER of the element."
            ),
        },
        "y": {
            "type": "integer",
            "description": (
                f"Vertical pixel coordinate on the {SCREEN_W}x{SCREEN_H} screen. "
                f"0 = top edge, {SCREEN_H} = bottom edge. "
                "Estimate the CENTER of the element."
            ),
        },
    },
    "required": ["element_name", "x", "y"],
}

GET_SCREEN_INFO_SCHEMA = {
    "type": "object",
    "properties": {},
}


async def highlight_ui(element_name: str, x: int, y: int):
    """Draw a glowing red circle on the student's screen to point at the UI element you are explaining. You MUST call this every time you mention any visible element."""
    x = max(0, min(int(x), SCREEN_W))
    y = max(0, min(int(y), SCREEN_H))
    logger.info(f"highlight_ui: '{element_name}' at ({x}, {y})")
    await broadcast_draw(x, y, label=element_name)
    return f"Drew a red circle on '{element_name}' at ({x}, {y}). The student can now see it."


async def get_screen_info():
    """Get the student's screen resolution and coordinate system. Call this if you need to recalibrate."""
    logger.info("get_screen_info called")
    return (
        f"Screen: {SCREEN_W}x{SCREEN_H} pixels. "
        f"Origin (0,0) = top-left. X increases right, Y increases down."
    )


# ---------------------------------------------------------------------------
# Agent factory
# ---------------------------------------------------------------------------
async def create_agent(**kwargs) -> Agent:
    _patch_frame_encoder()
    _patch_reconnect()
    _patch_getstream_timeout()

    llm = gemini.Realtime()

    # Register tools WITH explicit parameter schemas so Gemini gets
    # full descriptions for each parameter (the default registry
    # extracts types but not descriptions from docstrings)
    llm.function_registry.register(
        name="highlight_ui",
        description=(
            "Draw a glowing red circle on the student's screen to highlight "
            "the UI element you are explaining. You MUST call this every time "
            "you mention, refer to, or point at ANY visible element on screen. "
            "Call it BEFORE you explain. If you mention 3 elements, call it 3 times."
        ),
        parameters_schema=HIGHLIGHT_UI_SCHEMA,
    )(highlight_ui)

    llm.function_registry.register(
        name="get_screen_info",
        description="Get the student's screen resolution and coordinate system for calibration.",
        parameters_schema=GET_SCREEN_INFO_SCHEMA,
    )(get_screen_info)

    # Verify tools are registered
    tools = llm.function_registry.get_tool_schemas()
    for t in tools:
        param_count = len(t.get("parameters_schema", {}).get("properties", {}))
        logger.info(f"Registered tool: {t['name']} ({param_count} params)")

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
            # Subscribe to transcription events (what Gemini hears/says)
            # ---------------------------------------------------------------
            try:
                from vision_agents.core.llm.events import (
                    RealtimeUserSpeechTranscriptionEvent,
                    RealtimeAgentSpeechTranscriptionEvent,
                )

                @agent.events.subscribe
                async def _on_user_transcript(event: RealtimeUserSpeechTranscriptionEvent):
                    logger.info(f"🎤 User said: \"{event.text}\"")

                @agent.events.subscribe
                async def _on_agent_transcript(event: RealtimeAgentSpeechTranscriptionEvent):
                    logger.info(f"🤖 AI said: \"{event.text}\"")

                logger.info("Transcription monitors subscribed")
            except Exception as e:
                logger.warning(f"Could not subscribe transcription monitors: {e}")

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