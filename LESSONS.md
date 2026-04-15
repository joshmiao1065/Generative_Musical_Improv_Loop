# LESSONS.md — Improv Loop Project

> **Purpose**: Document every mistake, surprise, and hard-won lesson during this project so future LLM sessions do not repeat the same errors.
> **Rule**: Append an entry after every implementation session or significant discovery.
> **Format**:
> ```
> ## YYYY-MM-DD — [Topic]
> **Attempted**: What was tried.
> **Result**: What happened.
> **Root cause**: Why it happened.
> **Correct approach**: What actually works.
> **Watch for**: What to avoid going forward.
> ```

---

## 2026-04-15 — Magenta RT is a Library, Not a Cloud API

**Attempted**: N/A (documented pre-emptively from research)
**Result**: N/A
**Root cause**: Magenta RT (github.com/magenta/magenta-realtime) is an open-source Python library designed to run on TPU or high-VRAM GPU. It has no hosted endpoint.
**Correct approach**: Wrap Magenta RT in a FastAPI server deployed on Google Colab free TPU, exposed via pyngrok tunnel. The ThinkPad Python script calls this HTTP endpoint.
**Watch for**: Never write code that makes HTTP requests to a "Magenta RT endpoint" without first confirming the FastAPI server is running and the ngrok URL is set in config.

---

## 2026-04-15 — Temperature and Top-K Are Not Magenta RT / Lyria Parameters

**Attempted**: N/A (documented pre-emptively)
**Result**: N/A
**Root cause**: Temperature and top-k are LLM token-sampling parameters. Magenta RT uses a different architecture (autoregressive audio transformer) with different exposed controls.
**Correct approach**: Map PBF4 knobs to the REAL parameters: `guidance` [0–6], `density` [0–1], `brightness` [0–1], `bpm` [60–200].
**Watch for**: Do not include temperature or top-k anywhere in API calls, UI, or documentation.

---

## 2026-04-15 — MicroLab mk3 Has No Knobs or Faders

**Attempted**: N/A (documented pre-emptively)
**Result**: N/A
**Root cause**: The Arturia MicroLab mk3 is a minimal 25-key MIDI controller with only 2 touch strips and 4 buttons. It has NO rotary encoders, NO faders.
**Correct approach**: MicroLab mk3 = musical performance input ONLY (plays notes into Analog Lab / Surge XT). ALL system parameter control (guidance, density, genre weights, record/stop) is handled by the intech PBF4, which is a separate device.
**Watch for**: Never attempt to map system parameters to MicroLab mk3 CC numbers. The PBF4 is the only parameter controller.

---

## 2026-04-15 — Analog Lab Intro (Bundled) is Plugin-Only

**Attempted**: N/A (documented pre-emptively)
**Result**: N/A
**Root cause**: Analog Lab Intro (bundled with MicroLab mk3) is a VST/AU/AAX plugin format only. It requires a DAW host.
**Correct approach**: Two options:
  1. Use Ableton Live Lite (also bundled) as DAW host for Analog Lab Intro
  2. Use Surge XT standalone synthesizer (free, open source) — simpler for prototyping
**Watch for**: Do not assume Analog Lab is a standalone app. Test which is actually installed before proceeding.

---

## 2026-04-15 — VB-Cable: Use WASAPI Shared Mode, Not ASIO

**Attempted**: N/A (documented from GitHub issue research)
**Result**: ASIO driver loading fails with VB-Cable in Python sounddevice (GitHub issue #520, closed "not planned").
**Root cause**: VB-Cable's ASIO driver is not compatible with how python-sounddevice loads ASIO.
**Correct approach**: Use WASAPI shared mode (default in sounddevice). Do NOT pass `extra_settings=WasapiSettings(exclusive=True)` when recording from VB-Cable — exclusive mode conflicts with other apps using the virtual device simultaneously.
**Watch for**: When opening a sounddevice InputStream on "CABLE Output", use default settings. If latency is an issue, reduce blocksize but keep WASAPI shared mode.

---

## 2026-04-15 — ThinkPad Has No Discrete GPU: Cloud-Only Deployment

**Attempted**: N/A (confirmed from user)
**Result**: N/A
**Root cause**: The ThinkPad running the final script has no discrete GPU, making local Magenta RT inference impossible (~20–40GB VRAM required).
**Correct approach**: All AI inference must use cloud:
  - **Phase 1 (prototype)**: Lyria RealTime API via Google AI Studio free trial (WebSocket, parallel, simple)
  - **Phase 2**: Magenta RT on Colab free TPU + FastAPI + ngrok (for audio injection capability)
**Watch for**: Network latency to cloud APIs adds variable delay. The buffer-pass architecture (one loop pass of "dead time" per instrument) is specifically designed to absorb this latency.

---

## 2026-04-15 — 4 Parallel Magenta RT Instances Not Feasible on One Colab Free Session

**Attempted**: N/A (assessed from architecture constraints)
**Result**: N/A
**Root cause**: Colab free tier provides one v2-8 TPU. Loading Magenta RT (800M params) consumes most of the memory. Running 4 truly parallel inference threads is likely OOM.
**Correct approach**: Choose one:
  A. 4 separate Colab notebooks (4 browser tabs), each with own ngrok URL — clunky but free
  B. Lyria API (4 WebSocket connections, genuinely parallel) — use free trial credits
  C. Sequential inference (1 Colab, 4 sequential calls per loop pass) — slower but simpler
**Watch for**: Always test single-instance performance before assuming multi-instance works. Profile inference time on Colab before designing the multi-instance architecture.

---

*Add new entries below this line after each session.*
