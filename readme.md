# Easy Learning — Real-Time AI Tutor

An AI-powered VS Code tutor that watches your screen, talks to you in real-time,
and draws red circles on UI elements it's explaining.

Built with **vision-agents** (Gemini Multimodal Live API) + **GetStream** for
real-time video/audio + **PyQt5** for the transparent overlay.

---

## Setup

```powershell
cd "C:\Users\ucs\Desktop\Easy Learning"
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in your API keys:

- `STREAM_API_KEY` / `STREAM_API_SECRET` — from https://getstream.io/dashboard/
- `GOOGLE_API_KEY` — from https://aistudio.google.com/apikey

---

## How to Run

### Terminal 1 — Start the Agent

```powershell
cd "C:\Users\ucs\Desktop\Easy Learning"
.\venv\Scripts\Activate.ps1
python agent.py run
```

This prints a **demo URL** — open it in your browser.

### Terminal 2 — Start the Overlay

```powershell
cd "C:\Users\ucs\Desktop\Easy Learning"
.\venv\Scripts\Activate.ps1
python overlay.py
```

### Browser — Demo Page + Dashboard

1. Open the **demo URL** from the agent output
2. Click **Share Screen** → share your **Entire Screen**
3. Open **dashboard.html** in another tab for live status

### Start talking to the AI tutor!

---

## Files

| File               | Purpose                                                 |
| ------------------ | ------------------------------------------------------- |
| `agent.py`         | Main agent — Gemini Realtime + WebSocket server + tools |
| `overlay.py`       | Transparent click-through overlay (draws red circles)   |
| `dashboard.html`   | Web-based status dashboard (open in any browser)        |
| `.env`             | API keys (not committed)                                |
| `.env.example`     | Template for `.env`                                     |
| `requirements.txt` | Python dependencies                                     |

---

## Architecture

```
┌─────────────┐    GetStream     ┌─────────────────┐
│  Browser     │ ◄──── video ───►│  agent.py        │
│  (demo page) │    audio/screen │  (Gemini AI)     │
└─────────────┘                  └──────┬──────────┘
                                        │ WebSocket
                              ┌─────────┴──────────┐
                              │   ws://localhost:8765│
                              ├──────────┬──────────┤
                              ▼          ▼          ▼
                         overlay.py  dashboard.html
                         (red circles) (status log)
```

---

## Troubleshooting

- **AI sees black screen**: Make sure you clicked "Share Screen" → "Entire Screen"
  in the browser demo, not just a tab or window.
- **Overlay not drawing**: Check that overlay.py shows "Connected to agent WebSocket"
  and the green dot appears in the bottom-right corner of your screen.
- **Test the overlay**: Run `python overlay.py --test` to verify circles render.
