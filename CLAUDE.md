# CLAUDE.md — Improv Loop: Real-Time AI Music Improvisation System

> **Project for**: Cooper Union — Generative Machine Learning, Final Project
> **Developer**: Josh Miao
> **Goal**: A Python terminal script that orchestrates up to 4 parallel AI music instruments and a live human player, synchronized in real-time, controlled by the intech PBF4 MIDI controller.
> **Last Updated**: 2026-04-15 (Session 4 — Surge XT prototype locked in, monitoring strategy decided, GitHub noted, software swappability architecture designed)
> **GitHub**: https://github.com/joshmiao1065/Generative_Musical_Improv_Loop (`git@github.com:joshmiao1065/Generative_Musical_Improv_Loop.git`)

---

## CRITICAL NOTICE FOR THE IMPLEMENTING LLM

**Before writing a single line of code, read this entire document.** Every design decision, known pitfall, and architectural constraint is documented here. The most common mistakes are:
1. Ignoring the audio synchronization problem (Section 9) — read it first
2. Writing backend-specific code instead of using the abstract interface (Section 3A) — the whole codebase must be swappable
3. Assuming Magenta RT is a cloud REST API (it is not — Section 3)
4. Touching MIDI CC numbers before testing the physical PBF4 hardware (Section 2)

**Autonomous documentation rule**: After every implementation session or significant discovery, the LLM MUST:
- Update this file (`CLAUDE.md`) with any changes to architecture, confirmed/rejected decisions, or new constraints
- Append to `LESSONS.md` (create if missing) with lessons learned, mistakes made, and correct approaches
- Update Section 17 (Open Questions) to mark resolved items and add new ones

Failing to do this will cause repeated mistakes across sessions.

---

## TABLE OF CONTENTS

1. [Project Overview](#1-project-overview)
2. [Hardware: intech PBF4 Controller](#2-hardware-intech-pbf4-controller)
3. [AI Backend: Magenta RT and Lyria](#3-ai-backend-magenta-rt-and-lyria)
   - 3A. [Software Abstraction: AIInstrumentBackend Interface](#3a-software-abstraction-aiinstrumentbackend-interface)
4. [User Input Hardware: Arturia MicroLab mk3 + Surge XT](#4-user-input-hardware-arturia-microlab-mk3--surge-xt)
5. [Audio Routing Chain](#5-audio-routing-chain)
6. [Software Architecture for Swappability](#6-software-architecture-for-swappability)
7. [Full System Data Flow Diagram](#7-full-system-data-flow-diagram)
8. [Loop Architecture and Playback Logic](#8-loop-architecture-and-playback-logic)
9. [Audio Synchronization Challenges](#9-audio-synchronization-challenges-detailed)
10. [Genre Blending via Weighted Prompts](#10-genre-blending-via-weighted-prompts)
11. [Instrument Assignment and AI Instance Management](#11-instrument-assignment-and-ai-instance-management)
12. [Metronome and Timing Engine](#12-metronome-and-timing-engine)
13. [Remote Access: SSH + VNC to ThinkPad](#13-remote-access-ssh--vnc-to-thinkpad)
14. [Python Implementation Plan](#14-python-implementation-plan)
15. [Dependency Stack](#15-dependency-stack)
16. [Free Tier / Cost Strategy](#16-free-tier--cost-strategy)
17. [Runtime Arguments and Configuration](#17-runtime-arguments-and-configuration)
18. [Open Questions and Decisions Needed](#18-open-questions-and-decisions-needed)
19. [Documentation Links](#19-documentation-links)

---

## 1. Project Overview

### What This Is

A Python terminal script (running on a ThinkPad laptop, accessed remotely via SSH/VNC) that turns a keyboard synthesizer + MIDI controller into an AI improv band. The human player performs a looped phrase using the Arturia MicroLab mk3 keyboard running Surge XT (prototype) or Analog Lab (final). Up to 4 AI instrument voices improvise in response, each building on the previous voice's combined output in a cascading loop architecture.

**GitHub**: https://github.com/joshmiao1065/Generative_Musical_Improv_Loop

### Physical Setup (Confirmed)

```
[Arturia MicroLab mk3] ──USB-C MIDI──→ [ThinkPad]
[intech PBF4]          ──USB MIDI──→  [ThinkPad]
[ThinkPad]             ──Audio────→   [Speakers/Headphones]
[Remote machine]       ──SSH/VNC──→   [ThinkPad] (script control)

[ThinkPad] ──HTTP/WebSocket──→ [AI Backend]
                                  ├── Phase 1: Lyria RealTime API (Google cloud, WebSocket)
                                  └── Phase 2: Magenta RT (Colab TPU + FastAPI + ngrok)

Software on ThinkPad:
  Surge XT standalone ──audio──→ VB-Cable ──capture──→ Python script
  (prototype; swap to Analog Lab + Ableton in later iteration)
```

### Session Flow (High Level)

```
[Runtime: python improv_loop.py --bpm 120 --measures 4 --genres "jazz" "blues" ...]
    ↓
[PBF4 initializes: knobs set guidance/density, faders set genre weights]
    ↓
[Button 1: 2-measure metronome countdown → user records N-measure loop on MiniLab]
    ↓
[Loop plays back → Magenta RT Instance 1 generates over user loop audio]
    ↓
[Buffer pass → AI 1 output arrives and is aligned]
    ↓
[User loop + AI 1 combined → input to Magenta RT Instance 2]
    ↓
[Cascade continues until up to 4 AI instruments are playing simultaneously]
    ↓
[User may re-record at any time; AI instruments continue uninterrupted]
    ↓
[Button 2: graceful shutdown]
```

### Deliverable

`improv_loop.py` — a single entry-point Python script with modular helper files, no GUI required, runs in terminal, interfaces with PBF4 hardware for all real-time parameter control.

---

## 2. Hardware: intech PBF4 Controller

### Physical Controls

The PBF4 has 12 controls in 4 columns: 4 buttons (top), 4 faders (middle), 4 knobs (bottom).

> **VERIFY BEFORE CODING**: Plug in the PBF4 and run `python -c "import mido; print(mido.get_input_names())"` to confirm the device name. Then run a CC monitor to discover actual default CC numbers: `python -c "import mido; port = mido.open_input(); [print(m) for m in port]"` and move each control.

### Control Assignment

| Column | Button (CC?) | Fader (CC?) | Knob (CC?) |
|--------|-------------|-------------|------------|
| 1 | Record start | Genre 1 weight | `guidance` [0.0–6.0] |
| 2 | Stop session | Genre 2 weight | `density` [0.0–1.0] |
| 3 | Enable AI 3 | Genre 3 weight | `bpm` override |
| 4 | Enable AI 4 | Genre 4 weight | (reserved) |

> **NOTE**: CC numbers are configurable via intech Grid Editor (LUA-based). Assign them in Grid Editor first, then update `config/pbf4_cc_map.json`. Buttons likely send CC on press AND release; distinguish by value (127=press, 0=release).

### Python MIDI Interface

```python
import mido

# Discovery
print(mido.get_input_names())  # e.g. ['Intech Studio PBF4']

port = mido.open_input('Intech Studio PBF4', callback=handle_midi)

def handle_midi(msg):
    if msg.type == 'control_change':
        cc, val = msg.control, msg.value / 127.0
        dispatch_control(cc, val)
    elif msg.type == 'note_on' and msg.velocity > 0:
        dispatch_button(msg.note)
```

### USB Setup

The PBF4 is USB class-compliant. On Windows, it appears immediately as a MIDI device with no driver installation. On WSL2, requires `usbipd` passthrough (see Section 12).

### Documentation
- Product: https://intech.studio/shop/pbf4
- Docs hub: https://docs.intech.studio/
- MIDI functions: https://docs.intech.studio/reference-manual/grid-functions/midi/
- Grid Editor: https://docs.intech.studio/guides/grid/grid-adv/editor-201/

---

## 3. AI Backend: Magenta RT — Architecture and Deployment

### What Magenta RT Is

Magenta RT is an **800-million parameter open-source autoregressive transformer** for continuous real-time music generation, developed by Google Research. It generates 2 seconds of audio per inference step, conditioned on:
- A 10-second rolling audio context window
- A style embedding (from text or audio prompts via MusicCoCa)

**It is NOT a cloud REST API**. It is a Python library designed to run on TPU (Google Colab) or a high-VRAM GPU (locally).

### ✓ CONFIRMED Parameters (verified from actual notebook source code)

| Parameter | kwarg name | Type | Range | Notes |
|-----------|-----------|------|-------|-------|
| Guidance | `guidance_weight` | float | 0.0–10.0 | Controls prompt adherence (CFG weight) |
| Temperature | `temperature` | float | 0.0–4.0 | **YES — a real parameter** (token sampling) |
| Top-K | `topk` | int | 0–1024 | **YES — a real parameter** (token sampling) |
| Model feedback | `model_feedback` | float | 0.0–1.0 | How much AI output feeds back into its own context |
| Model volume | `model_volume` | float | 0.0–1.0 | Output gain of AI audio |
| Input volume | `input_volume` | float | 0.0–1.0 | Gain of user audio in output mix |
| BPM | `bpm` | int | 60–200 | Beats per minute — used for loop/metronome alignment |
| Beats per loop | `beats_per_loop` | int | 1–16 | Loop length in beats |
| Input gap | `input_gap` | int (ms) | 0–2000 | Silences end of input window (creates "response space") |
| Metronome | `metronome` | bool | — | Whether to overlay click track |
| Text prompt | style embedding | str | — | Passed to `system.embed_style(text)` |
| Audio prompt | style embedding | audio | 16kHz mono | Passed to `system.embed_style(waveform)` |
| Audio injection | `context_tokens_orig` | ndarray | (N, 16) int32 | User audio encoded as SpectroStream tokens |

> **Correction from earlier sessions**: Temperature and top-k ARE real model parameters in Magenta RT. They are passed via `generate_chunk(**kwargs)`. Update PBF4 knob mappings accordingly — the 4 knobs can control `guidance_weight`, `temperature`, `topk`, and `model_feedback`.

### Audio Injection (Key Feature — Fully Documented from Source)

Magenta RT audio injection works as follows (verified from notebook source):

1. Each call to `streamer.generate()` receives a chunk of input audio (`inputs`, numpy array, 48kHz stereo)
2. The input chunk is accumulated into `injection_state.all_inputs`
3. A window of recent input audio is mixed with a window of recent model output audio: `mix_audio = input_window + output_window * model_feedback`
4. The mixed audio is encoded to SpectroStream tokens: `mix_tokens = spectrostream_model.encode(mix_audio)`
5. These tokens replace the model's context window: `state.context_tokens[-N:] = mix_tokens`
6. The original (pre-mix) context tokens are saved as `context_tokens_orig`
7. `generate_chunk()` receives both `state.context_tokens` (mixed) and `context_tokens_orig` (clean) — the tied CFG uses both for guidance
8. Output is 2 seconds of 48kHz stereo audio, crossfaded with previous chunk

**For the cascade**: AI Instance 2's `all_inputs` is the sum of `user_loop + AI_1_output`. This is the "listening" mechanism — each AI literally hears all prior voices via audio injection.

**Audio Injection Colab**: https://colab.research.google.com/github/magenta/magenta-realtime/blob/main/notebooks/Magenta_RT_Audio_Injection.ipynb

**IMPORTANT: Colab UI is deeply coupled**: The `AudioInjectionStreamer` uses `colab_utils.AudioStreamer` for real-time I/O — this is Colab-specific. For our non-Colab client, we extract only the core logic: `spectrostream_model.encode()`, `injection_state` management, and `generate_chunk()`. The Colab UI widgets are not needed.

### Deployment Options

#### Option A: Google Colab Free TPU + FastAPI + ngrok (RECOMMENDED FOR PROTOTYPING)

User has access to CoLab Pro using student account
Architecture:
```
[Colab Notebook on TPU]
  → Load Magenta RT
  → Wrap with FastAPI endpoint
  → pyngrok creates public HTTPS tunnel
  → Your ThinkPad Python script calls http://ngrok-url/generate
```

Setup in Colab cell:
```python
!pip install fastapi uvicorn pyngrok magenta-realtime

from fastapi import FastAPI
from pyngrok import ngrok
import uvicorn, threading

app = FastAPI()

@app.post("/generate")
async def generate(payload: dict):
    audio_bytes = payload["audio_bytes"]   # base64 encoded
    bpm = payload["bpm"]
    guidance = payload["guidance"]
    # ... call Magenta RT model ...
    return {"audio": generated_base64}

# Start tunnel
public_url = ngrok.connect(8000)
print(f"Public URL: {public_url}")

# Run server in background thread
threading.Thread(target=uvicorn.run, args=(app,), kwargs={"port": 8000}).start()
```

**Cost**: Free (Colab free tier provides v2-8 TPU; ngrok free tier provides one tunnel)
**Latency**: ~1.25s per 2s of audio (real-time factor ~1.6x on Colab TPU)
**Limitation**: Colab sessions disconnect after ~12 hours; not suitable for production

#### Option B: Local on ThinkPad GPU — ✗ NOT AVAILABLE

> **CONFIRMED**: The ThinkPad has no discrete GPU. Local Magenta RT is not feasible. Colab (Option A) or Lyria API (Option C) are the only options.

#### Option C: Lyria RealTime API (Cloud Alternative, Lower Setup Friction)

If Magenta RT deployment proves too complex, fall back to Google's **Lyria RealTime API**:
- WebSocket cloud API: `wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage.v1alpha.GenerativeService.BidiGenerateMusic`
- Parameters: `guidance`, `density`, `bpm`, `WeightedPrompts`
- Accessible via Google AI Studio API key (free trial available)
- Does NOT support audio injection natively (text + weighted prompts only)
- Documentation: https://ai.google.dev/gemini-api/docs/realtime-music-generation

**Recommendation**: Start with Colab Magenta RT (Option A) for the free prototyping. Transition to local Option B if ThinkPad GPU is sufficient, or fall back to Lyria (Option C) if Colab proves too unreliable.

### ✓ CONFIRMED: Lyria RealTime API Does NOT Support Audio Injection

Verified from official Google AI docs and developer guide: Lyria RealTime accepts only text (`WeightedPrompts`) and numerical config parameters. There is no audio input of any kind. **Lyria is removed from this project entirely.** Magenta RT is the sole AI backend.

### POC Architecture: Colab Notebook + Pre-Recorded Audio

For the proof of concept (no hardware yet):

```
[test_loop.wav]  ← any short audio clip, simulates user loop
      │
      ▼
[Colab Notebook — Magenta RT on TPU]
      │
      ├── AI Instance 1: inject(user_loop) → ai1_audio
      │
      ├── AI Instance 2: inject(user_loop + ai1_audio) → ai2_audio
      │
      ├── (optional) AI Instance 3: inject(user_loop + ai1 + ai2) → ai3_audio
      │
      └── Final mix = user_loop + ai1_audio + ai2_audio + ...
                          → saved as output.wav, played in Colab
```

**POC does not need the FastAPI server** — it all runs within one Colab notebook. The cascade is implemented as a Python loop that calls `AudioInjectionStreamer.generate()` multiple times in sequence. The FastAPI wrapper is added in the next phase.

**Two-phase POC approach**:
1. **POC Phase A**: Single Colab notebook, sequential generation, proves cascade concept. Use pre-recorded audio as user loop.
2. **POC Phase B**: Add FastAPI + ngrok to the same notebook, write local Python client, prove over-the-network audio roundtrip.

### CRITICAL: 4 Parallel Instances on Colab Free Tier

Running 4 simultaneous Magenta RT instances on a single Colab free TPU session is likely not feasible — the model is 800M parameters and each inference call is compute-heavy. Options:

| Strategy | How | Tradeoff |
|----------|-----|----------|
| **A — 4 Colab notebooks** | Open 4 separate browser tabs, each running a Magenta RT server | Free, works, but requires 4 ngrok URLs in config |
| **B — Sequential inference** | One Colab, one model, loop through all 4 instruments per pass | Free, slower: 4× inference latency per loop pass |
| **C — Lyria API x4** | 4 WebSocket connections to Lyria (cloud, paid) | Clean, truly parallel, has free trial |
| **D — Single model, voice multiplexing** | One model generates one long prompt per pass, split into 4 | Experimental, loss of instrument separation |

**Updated recommendation (Lyria eliminated — does not support audio injection)**:
1. **POC**: Single Colab notebook, 2 sequential AI instances. Proves cascade concept.
2. **v1**: Same Colab + FastAPI + ngrok + local Python client. Proves network roundtrip.
3. **v2**: 4 separate Colab notebooks (4 TPU sessions) for genuine parallelism.
4. **Final**: Colab Pro or sponsored GPU for a consolidated 4-instance session.

> The key insight: Lyria API and Magenta RT both accept audio conditioning and text prompts, so the loop architecture code is nearly identical. Build to the abstract `AIInstrumentBackend` interface (Section 6) — swapping backends is a config line change.

---

### 3A. Software Abstraction: AIInstrumentBackend Interface

See **Section 6** for the full swappability architecture. The critical point for AI backends:

- `LyriaBackend` connects via WebSocket, sends `WeightedPrompts` + `guidance` + `density` + `bpm`, receives streaming PCM audio
- `MagentaRTBackend` connects via HTTP to the Colab FastAPI server, sends base64-encoded audio + params, receives base64-encoded audio chunks
- Both implement the same `AIInstrumentBackend` abstract interface
- The `SessionManager` only calls `backend.generate(audio, params)` — it never imports `LyriaBackend` or `MagentaRTBackend` directly

**Phase 1 (current)**: Implement `LyriaBackend` only. Stub out `MagentaRTBackend` with a `NotImplementedError`.
**Phase 2**: Implement `MagentaRTBackend`. Flip `config.yaml` to use it.

### Magenta RT Python API (from source inspection)

The actual Python API must be verified by reading the Colab notebooks. Key entry points expected (verify against source):
```python
from magenta_rt import MagentaRT

model = MagentaRT()

# Style conditioning
style_embedding = model.encode_text_prompt("jazz piano trio")
# OR
style_embedding = model.encode_audio_prompt(audio_array)  # 16kHz mono numpy array

# Generation with audio injection
output_audio = model.generate(
    context_audio=prior_output_audio,  # numpy array, 10-second window
    style_embedding=style_embedding,
    bpm=120,
    guidance=4.0,
    density=0.5,
    inject_audio=user_loop_audio,      # numpy array mixed in
)
# Returns: numpy array, 48kHz stereo, 2 seconds
```

> **Verify this API by reading**: https://github.com/magenta/magenta-realtime/blob/main/notebooks/Magenta_RT_Audio_Injection.ipynb

---

## 4. User Input Hardware: Arturia MicroLab mk3 + Analog Lab

### ✓ CONFIRMED: Device is Arturia MicroLab mk3

Product page: https://www.arturia.com/products/hybrid-synths/microlab-mk3/overview

### MicroLab mk3 Specifications (Confirmed)

- **Interface**: USB-C, bus-powered, **MIDI-only** (no audio over USB)
- **Keys**: 25 velocity-sensitive slim keys (2-octave range)
- **Touch strips**: 2 — pitch bend and modulation (these send MIDI pitch bend / CC1)
- **Buttons**: 4 — octave shift up/down, preset navigation (send MIDI CC or program change)
- **NO knobs, NO faders, NO RGB pads** — this is a minimal keyboard controller
- **Bundled software**: Analog Lab Intro (~500 presets), Ableton Live Lite

### CRITICAL ARCHITECTURAL NOTE: Dual-Device Role Split

The MicroLab mk3 is the **musical performance input only**. It plays notes. It does NOT control AI parameters.

ALL parameter control (guidance, density, genre weights, record/stop) is handled exclusively by the **intech PBF4**.

```
MicroLab mk3  →  plays notes  →  Analog Lab  →  audio  →  user loop
PBF4          →  controls AI parameters, session state, genre blend
```

The MicroLab's 4 buttons (octave shift, preset nav) are musical utilities for the performer, not system controls. Ignore them in the Python state machine.

### Analog Lab Intro: Plugin — Requires DAW Host

**Confirmed**: Analog Lab Intro is a VST/AU plugin. It requires a DAW host. Ableton Live Lite (also bundled) is the correct host.

**Audio chain with Analog Lab:**
```
MicroLab mk3 (USB-C MIDI) → Ableton Live Lite → Analog Lab Intro (VST plugin)
                                     ↓
                           Windows Audio Output
                                     ↓
                           VB-Cable Virtual Input  (set as Ableton output device)
                                     ↓
                           Python sounddevice records from VB-Cable Output
```

**Ableton Live Lite setup (one-time):**
1. Open Ableton Live Lite
2. Preferences → Audio → Output Device: set to "VB-Cable Input"
3. Add MIDI track → load Analog Lab Intro as instrument
4. Arm the track for MIDI input from MicroLab mk3
5. Leave running in background during Python script sessions

**Alternative standalone synthesizers** (simpler setup, no DAW needed):
- **Surge XT** (free, open source, excellent, has standalone mode): https://surge-synthesizer.github.io/
- **VITAL** (free tier, has standalone mode): https://vital.audio/
- Either can be set to output to VB-Cable directly without Ableton

> **✓ CONFIRMED for prototype**: Use **Surge XT standalone**. Configure it to output to VB-Cable. The Python script handles monitoring (PassthroughMonitor). Transition to Analog Lab + Ableton in v3 once the loop logic is proven — this is a routing change only, no code changes required.

### MIDI CC from MicroLab mk3

The MicroLab mk3 has no configurable CC knobs. The touch strips send:
- Strip 1 (left): MIDI Pitch Bend
- Strip 2 (right): CC1 (Modulation)

These are musical expression controls, not system parameter controls. No MIDI Control Center needed for this device.

---

## 5. Audio Routing Chain

### The Core Problem

The MicroLab mk3 is MIDI-only — no audio flows over USB. Audio synthesis happens in Analog Lab (or Surge XT) on the ThinkPad. To capture that audio into the Python script, virtual audio routing is required.

### ✓ CONFIRMED: VB-Cable is Compatible with This Project

**VB-Cable technical specs (verified):**
- Supports 48kHz sample rate ✓ (matches Magenta RT output)
- Supports 24-bit depth ✓
- **Latency at 48kHz, 2048-sample buffer**: ~14–21ms (acceptable — AI generation latency is 1000x larger)
- **Python sounddevice compatibility**: YES via WASAPI shared mode ✓
- **ASIO with VB-Cable**: Does NOT work reliably with Python sounddevice (GitHub issue #520, closed as "not planned"). Use WASAPI instead.
- Free (donationware): https://shop.vb-audio.com/en/win-apps/11-vb-cable.html

### VB-Cable Architecture

After installing VB-Cable, two new audio devices appear in Windows:
- **"CABLE Input"** — a virtual playback (output) device. Set the synth to output here.
- **"CABLE Output"** — a virtual recording (input) device. Python reads from here.

```
[MicroLab mk3] ──USB-C MIDI──→ [Ableton Live Lite / Surge XT standalone]
                                              │
                                  [Audio output → "CABLE Input"]
                                              │
                                   [VB-Cable internal buffer]
                                              │
                           [Python records from "CABLE Output"]
                                              │
                                    [user_loop buffer]
                                              │
                           [Magenta RT audio injection input]
```

### VB-Cable Setup Steps

1. Download and install: https://shop.vb-audio.com/en/win-apps/11-vb-cable.html
2. Set VB-Cable sample rate to 48kHz:
   - Windows Sound Settings → Playback → "CABLE Input" → Properties → Advanced
   - Set: "2 channel, 24 bit, 48000 Hz (Studio Quality)"
   - Repeat for "CABLE Output" under Recording devices
3. In synth app (Surge XT or Ableton), set audio output device to **"CABLE Input"**
4. In Python, record from **"CABLE Output"**

### ✓ CONFIRMED: Monitoring via Python Script (PassthroughMonitor)

The Python script captures audio from VB-Cable and immediately plays it back through the output device in the same audio callback. This is the **lowest-friction, zero-extra-software** approach for prototyping.

**Latency added by passthrough**: ~20–40ms roundtrip (VB-Cable buffer + sounddevice buffer). This is imperceptible in a live music context.

**How it works**: The output callback in `sounddevice` mixes captured user audio (from VB-Cable) + all AI output buffers and writes the result to speakers. No separate monitoring app needed.

```
VB-Cable Output (Surge XT audio) → Python input callback → audio_queue
audio_queue + AI output buffers → Python output callback → speakers
```

**Future upgrade** (v3+): Swap `PassthroughMonitor` for Voicemeeter Banana (https://voicemeeter.com/) to get zero-latency hardware monitoring with per-source volume control. This is a config line change, not a code change, due to the `MonitorPlayback` abstraction.

### Python Code: Recording from VB-Cable

```python
import sounddevice as sd
import numpy as np

# List all devices — find VB-Cable Output index
for i, dev in enumerate(sd.query_devices()):
    if 'CABLE Output' in dev['name']:
        vb_cable_input_idx = i
        print(f"VB-Cable at index {i}: {dev}")

SAMPLE_RATE = 48000
CHANNELS = 2  # VB-Cable is stereo by default

def start_capture(callback_fn):
    """callback_fn(indata: np.ndarray, frames, time, status)"""
    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype='float32',
        device=vb_cable_input_idx,
        latency='low',  # Request low latency mode
        callback=callback_fn
    )
    stream.start()
    return stream
```

> **Important**: Do NOT use WASAPI exclusive mode with VB-Cable — it causes the connection to fail. WASAPI shared mode (default in sounddevice) works correctly.

### Audio Format Requirements for Magenta RT

| Use | Format | Sample Rate | Channels |
|-----|--------|-------------|----------|
| VB-Cable capture | float32 numpy | 48kHz | stereo |
| Magenta RT audio injection | float32 numpy | 16kHz | mono |
| Magenta RT output | float32 numpy | 48kHz | stereo |
| Playback mixer | float32 numpy | 48kHz | stereo |

Resampling from capture format to injection format:
```python
import librosa

def prepare_for_injection(audio_48k_stereo: np.ndarray) -> np.ndarray:
    mono = np.mean(audio_48k_stereo, axis=1)              # stereo → mono
    mono_16k = librosa.resample(mono, orig_sr=48000, target_sr=16000)
    return mono_16k.astype(np.float32)
```

---

## 6. Software Architecture for Swappability

This project will go through multiple iterations swapping out components (synth, AI backend, monitoring). The architecture enforces this through **interface-based abstraction at every boundary**. No module should depend on a concrete implementation of another module — only on its interface.

### Design Principles

1. **Program to interfaces, not implementations**: Every swappable component has an abstract base class. Concrete classes are selected at startup via config, never hardcoded.
2. **Config-driven construction**: A single `config.yaml` file drives which implementation class is instantiated for each component. Change one line to swap backends.
3. **No globals**: All shared state (current params, loop buffer, session state) lives in explicitly passed objects, not module-level globals.
4. **Async-first**: All AI calls, audio streaming, and MIDI callbacks are async-compatible. Use `asyncio` as the runtime.
5. **Fail loudly, not silently**: If a backend connection drops, raise an exception and log it — do not silently produce silence.

### Abstraction Layers

| Layer | Abstract Base Class | Prototype Implementation | Final Implementation |
|-------|--------------------|--------------------------|--------------------|
| AI Backend | `AIInstrumentBackend` | `LyriaBackend` | `MagentaRTBackend` |
| Audio Capture | `AudioCaptureSource` | `VBCableCapture` | `DirectLineInCapture` |
| Synthesizer | (external app, not Python-controlled) | Surge XT standalone | Analog Lab + Ableton |
| Monitoring | `MonitorPlayback` | `PassthroughMonitor` | `VoicemeeterMonitor` |
| MIDI Controller | `ParameterController` | `PBF4Controller` | (extensible) |

### Abstract Interface Definitions (Pseudocode — for planning, not final code)

```python
# Every AI backend implements this exact interface
class AIInstrumentBackend(ABC):
    @abstractmethod
    async def connect(self) -> None: ...

    @abstractmethod
    async def generate(
        self,
        audio_input: np.ndarray,   # 16kHz mono float32
        params: GenerationParams,
    ) -> np.ndarray: ...            # 48kHz stereo float32

    @abstractmethod
    async def disconnect(self) -> None: ...

    @abstractmethod
    def is_connected(self) -> bool: ...


# Shared parameter object — passed everywhere, mutated by MIDI callbacks
@dataclass
class GenerationParams:
    bpm: int = 120
    guidance: float = 4.0
    density: float = 0.5
    brightness: float = 0.5
    instrument: str = "piano"
    genre_weights: dict[str, float] = field(default_factory=dict)


# Audio capture — VB-Cable or line-in
class AudioCaptureSource(ABC):
    @abstractmethod
    def start(self, callback: Callable[[np.ndarray], None]) -> None: ...
    @abstractmethod
    def stop(self) -> None: ...


# Monitoring — how the user hears their own playing
class MonitorPlayback(ABC):
    @abstractmethod
    def play(self, audio: np.ndarray) -> None: ...
    @abstractmethod
    def stop(self) -> None: ...
```

### Config-Driven Backend Selection (`config.yaml`)

```yaml
# Change these single lines to swap components — no code changes needed

ai_backend: "lyria"          # options: "lyria" | "magenta_rt"
audio_capture: "vb_cable"    # options: "vb_cable" | "line_in" | "wasapi_loopback"
monitoring: "passthrough"    # options: "passthrough" | "voicemeeter" | "none"

bpm: 120
measures: 4
sample_rate: 48000

genres:
  - "jazz"
  - "blues"
  - "electronic"
  - "classical"

instruments:
  - "piano"
  - "bass"
  - "drums"
  - "violin"

lyria:
  api_key_env: "GOOGLE_API_KEY"   # reads from environment variable
  guidance: 4.0
  density: 0.5

magenta_rt:
  server_url: "https://your-ngrok-url.ngrok-free.app"
  guidance: 4.0
  density: 0.5
  brightness: 0.5

pbf4:
  cc_map_file: "config/pbf4_cc_map.json"  # update after hardware test
```

### Module Dependency Map

```
improv_loop.py
    └── loads config.yaml
    └── constructs concrete classes via factory functions
    └── passes objects to SessionManager

SessionManager
    ├── TimingEngine        (no external deps)
    ├── LoopBuffer          (no external deps)
    ├── StateMachine        (depends on TimingEngine, LoopBuffer)
    ├── PBF4Controller      (depends on GenerationParams — mutates it)
    ├── AudioCaptureSource  (concrete: VBCableCapture)
    ├── MonitorPlayback     (concrete: PassthroughMonitor)
    ├── AudioMixer          (reads from all AI instances + loop buffer)
    └── AIInstanceManager
            └── AIInstrumentBackend × 4  (concrete: LyriaBackend or MagentaRTBackend)
```

### Iteration Roadmap

| Iteration | Swap | Effort |
|-----------|------|--------|
| v1 (prototype) | Lyria API + Surge XT + PassthroughMonitor | baseline |
| v2 | Swap Lyria → Magenta RT (change 1 config line + ngrok URL) | low |
| v3 | Swap Surge XT → Analog Lab + Ableton (change audio routing, not code) | medium |
| v4 | Swap PassthroughMonitor → Voicemeeter (change 1 config line) | low |

---

## 7. Full System Data Flow Diagram

```
═══════════════════════════════════════════════════════════════════════════
                          PHYSICAL LAYER
═══════════════════════════════════════════════════════════════════════════

[MiniLab 3]──MIDI USB──→[ThinkPad]──→[Analog Lab (synth engine)]
                                              │
                                      [VB-Cable / WASAPI loopback]
                                              │
[PBF4]──────MIDI USB──→[ThinkPad]    [Python audio capture thread]
                │
         [mido callbacks]

═══════════════════════════════════════════════════════════════════════════
                          CONTROL PLANE (mido callbacks)
═══════════════════════════════════════════════════════════════════════════

PBF4 Knob 1 ──→ guidance_value (0.0–6.0)   ──→ all Magenta RT instances
PBF4 Knob 2 ──→ density_value (0.0–1.0)    ──→ all Magenta RT instances
PBF4 Knob 3 ──→ bpm_override               ──→ TimingEngine + Magenta RT
PBF4 Knob 4 ──→ brightness_value (0.0–1.0) ──→ all Magenta RT instances
PBF4 Fader 1 ──→ genre_weights[0]          ──→ PromptBuilder
PBF4 Fader 2 ──→ genre_weights[1]          ──→ PromptBuilder
PBF4 Fader 3 ──→ genre_weights[2]          ──→ PromptBuilder
PBF4 Fader 4 ──→ genre_weights[3]          ──→ PromptBuilder
PBF4 Btn 1  ──→ RECORD trigger             ──→ StateMachine
PBF4 Btn 2  ──→ STOP session               ──→ StateMachine
PBF4 Btn 3  ──→ Enable AI Instance 3       ──→ InstanceManager
PBF4 Btn 4  ──→ Enable AI Instance 4       ──→ InstanceManager

═══════════════════════════════════════════════════════════════════════════
                          AUDIO PLANE
═══════════════════════════════════════════════════════════════════════════

[Audio Capture] → raw PCM (48kHz stereo)
                        │
              [Record when in RECORDING state]
                        │
              [user_loop.wav — N×4 beats at BPM]
                        │
     ┌──────────────────┼──────────────────────────┐
     │                  │                          │
[Pass 1]           [Pass 2]                  [Pass 3+]
User loop heard    User loop heard          All active voices heard
→ sent to AI 1     (buffer: AI 1 generates)  User loop + AI outputs
                   AI 1 output arrives        → input to next AI

═══════════════════════════════════════════════════════════════════════════
                    MAGENTA RT INFERENCE PLANE
═══════════════════════════════════════════════════════════════════════════

For each active Magenta RT instance (1–4):
┌────────────────────────────────────────────────────┐
│  HTTP POST to Magenta RT server (Colab+ngrok)       │
│                                                     │
│  Input:                                             │
│    - inject_audio: resample_16k(cumulative_mix)     │
│    - context_audio: last 10s of own output          │
│    - style: build_prompt(instrument, genres[])      │
│    - bpm, guidance, density, brightness             │
│                                                     │
│  Output:                                            │
│    - 2s @ 48kHz stereo numpy array                  │
│    - accumulated in a queue                         │
│    - trimmed to loop_samples on pass boundary       │
└────────────────────────────────────────────────────┘

═══════════════════════════════════════════════════════════════════════════
                          OUTPUT MIXER
═══════════════════════════════════════════════════════════════════════════

[User Loop] + [AI 1 output] + [AI 2 output] + [AI 3?] + [AI 4?]
                     │
              [NumPy addition + normalize]
                     │
              [sounddevice output callback]
                     │
              [ThinkPad speakers/headphones]
```

---

## 7. Loop Architecture and Playback Logic

### Loop Duration Calculation

```python
SAMPLE_RATE = 48000
bpm = 120          # runtime argument
measures = 4       # runtime argument
beats_per_measure = 4
seconds_per_beat = 60.0 / bpm
loop_duration_sec = measures * beats_per_measure * seconds_per_beat
loop_samples = int(loop_duration_sec * SAMPLE_RATE)
# At 120 BPM, 4 measures = 8 seconds = 384,000 samples at 48kHz
```

### Playback Pass System

Each "pass" is exactly one loop duration. A global loop clock governs all pass transitions.

| Pass | What the user hears | What's happening in the background |
|------|--------------------|------------------------------------|
| -2, -1 | 2-measure metronome countdown | — |
| 0 | Metronome + user recording live | — |
| 1 | user_loop playback | AI 1 generating (injecting user_loop) |
| 2 | user_loop playback (buffer) | AI 1 completes; AI 2 starts generating |
| 3 | user_loop + AI 1 output | AI 2 generating (injecting user+AI1) |
| 4 | user_loop + AI 1 (buffer) | AI 2 completes; AI 3 starts (if enabled) |
| 5 | user_loop + AI 1 + AI 2 | AI 3 generating (if enabled) |
| … | all voices converge | cascade continues |

### Double-Buffer Loop Swap

When user re-records mid-session:
```python
class LoopBuffer:
    """Thread-safe double buffer for loop audio"""
    def __init__(self):
        self._lock = threading.Lock()
        self._active: np.ndarray | None = None
        self._pending: np.ndarray | None = None

    def set_pending(self, audio: np.ndarray):
        with self._lock:
            self._pending = audio

    def swap_on_boundary(self):
        """Call at the START of each pass"""
        with self._lock:
            if self._pending is not None:
                self._active = self._pending
                self._pending = None
                return True  # Signal: new loop
        return False

    def get_active(self) -> np.ndarray | None:
        with self._lock:
            return self._active
```

### Magenta RT Continuous Generation Strategy

Magenta RT generates 2-second chunks continuously. To align output to our N-second loop:

1. **Start generating** at the beginning of a buffer pass (e.g., Pass 1 start)
2. **Accumulate chunks** in a queue: `output_queue.put(chunk)`
3. **At the next pass boundary** (Pass 2 → Pass 3 start), dequeue and concatenate exactly `loop_samples` worth of audio
4. If not enough audio has arrived yet (generation was slow), pad with silence and log to `LESSONS.md`
5. Trim or crossfade the endpoint to avoid clicks

```python
async def accumulate_loop(queue: asyncio.Queue, loop_samples: int) -> np.ndarray:
    """Collect exactly loop_samples from the generation queue"""
    buffer = []
    collected = 0
    while collected < loop_samples:
        chunk = await asyncio.wait_for(queue.get(), timeout=5.0)
        buffer.append(chunk)
        collected += len(chunk)
    combined = np.concatenate(buffer)
    return combined[:loop_samples]  # Trim to exact length
```

---

## 8. Audio Synchronization Challenges (Detailed)

### Challenge 1: Magenta RT Generates Continuously, Not Loop-Aligned

Magenta RT produces a streaming output — 2-second chunks that have no inherent alignment to our loop length. The model does not know or care about our loop boundary.

**Consequence**: A 8-second loop at 120 BPM would require 4 chunks (4×2s). But generation is overlapping and crossfaded internally. The actual audio produced in "8 seconds of generation time" may not be exactly 8 seconds of usable content.

**Solution**: Use the buffer pass architecture. Collect generated audio during one full pass (the "buffer pass"), then trim to exactly `loop_samples`. The Magenta RT BPM parameter should help align the generated audio's internal rhythm to ours, but do not rely on perfect beat alignment in v1.

### Challenge 2: Magenta RT Generation Latency is ~1.25s per 2s chunk

At ~1.6x real-time speed on Colab TPU, generating one 8-second loop requires ~5 seconds of inference time. Our buffer pass (also 8 seconds) should be sufficient, but network round-trip to Colab adds variable latency.

**Mitigation**: Pipeline the requests. Start sending the next generation request before the current one completes. Use Python `asyncio` with concurrent tasks per AI instance.

### Challenge 3: Loop Boundary Timing Must Not Drift

Over many passes (e.g., 30-minute session = ~225 passes at 8s each), any per-pass timing error accumulates. `time.sleep()` on Windows has ~10–15ms jitter.

**Solution**: Absolute time anchoring:
```python
import time, math

session_start = time.perf_counter()
loop_duration = 8.0  # seconds

def schedule_next_pass(pass_num: int) -> float:
    """Returns seconds until pass_num should start"""
    target = session_start + pass_num * loop_duration
    return target - time.perf_counter()

# In loop thread:
while running:
    wait = schedule_next_pass(next_pass)
    if wait > 0:
        time.sleep(wait)
    trigger_pass_start(next_pass)
    next_pass += 1
```

### Challenge 4: Audio Output Latency Must Be Compensated

`sounddevice` has output latency (~20–100ms depending on driver settings). Schedule audio playback `output_latency` seconds early to compensate.

```python
device_info = sd.query_devices(output_device_index, 'output')
output_latency_sec = device_info['default_low_output_latency']
# Schedule audio playback: start_time - output_latency_sec
```

### Challenge 5: Multi-Stream Mixing Without Buffer Underruns

Four AI audio streams + user loop must be mixed in a single output callback without glitches. Each stream may have slightly different lengths or arrival times.

**Solution**: Each AI instance maintains a `CircularAudioBuffer`. The output callback reads from all active buffers simultaneously:
```python
def output_callback(outdata, frames, time_info, status):
    out = np.zeros((frames, 2), dtype=np.float32)
    for voice in active_voices:
        chunk = voice.read(frames)
        if chunk is not None:
            out += chunk
    peak = np.max(np.abs(out))
    if peak > 1.0:
        out /= peak
    outdata[:] = out
```

### Challenge 6: Audio Injection Format Conversion

User loop is recorded at 48kHz stereo. Magenta RT's audio injection expects 16kHz mono.

```python
import librosa
import numpy as np

def prepare_injection_audio(audio_48k_stereo: np.ndarray) -> np.ndarray:
    # Convert to mono
    mono = np.mean(audio_48k_stereo, axis=1) if audio_48k_stereo.ndim == 2 else audio_48k_stereo
    # Resample to 16kHz
    mono_16k = librosa.resample(mono, orig_sr=48000, target_sr=16000)
    return mono_16k.astype(np.float32)
```

### Challenge 7: ngrok Tunnel Reliability (Colab Deployment)

ngrok free tunnels can be interrupted, have rate limits, and Colab sessions can disconnect.

**Mitigations**:
- Implement HTTP retry logic with exponential backoff
- Keep Colab session alive using browser interactions or `keep_alive` pings
- For production (demo/final): run Magenta RT locally if ThinkPad GPU is sufficient

### Challenge 8: Cascade Delay Accumulation

With 4 AI instruments each requiring 2 buffer passes to come online:
- AI 1 ready at pass 3 (~24s at 120 BPM, 4 measures)
- AI 2 ready at pass 5 (~40s)
- AI 3 ready at pass 7 (~56s, if enabled)
- AI 4 ready at pass 9 (~72s, if enabled)

**Optimization**: Pre-generate for AI 2 during passes 1–2 (before AI 1 is even heard). This overlaps generation timelines, reducing total time-to-full-ensemble.

---

## 9. Genre Blending via Weighted Prompts

### How It Works in Magenta RT

Magenta RT uses **MusicCoCa** to produce style embeddings. A weighted average of multiple text embeddings is fed to the decoder:

```python
# Pseudocode — verify actual API from Colab notebooks
embeddings = []
for genre, weight in zip(genres, genre_weights):
    text = f"{genre} {instrument}"
    emb = model.encode_text_prompt(text)
    embeddings.append((emb, weight))

# Weighted combination
style_embedding = sum(emb * w for emb, w in embeddings)
style_embedding /= sum(w for _, w in embeddings)
```

If the Magenta RT API doesn't natively support weighted prompt averaging, implement it manually by combining embeddings before passing to the model.

### Fader → Genre Weight Mapping

```python
genres = ["jazz", "blues", "electronic", "classical"]  # set at runtime
genre_weights = [0.25, 0.25, 0.25, 0.25]

def on_fader_change(fader_index: int, value_0_to_1: float):
    genre_weights[fader_index] = value_0_to_1
    # Do NOT auto-normalize — let user have partial application
    # Values 0.0–1.0 map directly to relative weights

def build_style_prompt(instrument: str) -> str:
    """Build a single text prompt for cases where weighted embeddings aren't available"""
    parts = []
    for genre, weight in zip(genres, genre_weights):
        if weight > 0.05:
            intensity = "heavy" if weight > 0.7 else ("moderate" if weight > 0.3 else "light")
            parts.append(f"{intensity} {genre}")
    return f"{instrument}: {', '.join(parts)}"
```

---

## 10. Instrument Assignment and AI Instance Management

### Default Configuration

| Instance | Default Instrument | Enabled By Default | Enabled By |
|----------|-------------------|-------------------|------------|
| AI 1 | Piano / Keys | YES | Always |
| AI 2 | Bass | YES | Always |
| AI 3 | Drums / Percussion | NO | PBF4 Button 3 |
| AI 4 | Violin / Lead Melody | NO | PBF4 Button 4 |

All four instruments configurable at runtime:
```bash
python improv_loop.py --instruments "piano" "bass" "drums" "violin"
```

### Cascading Input Logic

```
AI 1 input: user_loop (resampled 16kHz mono)
AI 2 input: user_loop + AI_1_output (mixed, resampled)
AI 3 input: user_loop + AI_1_output + AI_2_output
AI 4 input: user_loop + AI_1_output + AI_2_output + AI_3_output
```

Each instance hears the full cumulative mix below it. This creates an emergent "conversation" where later instruments respond to the richer texture above them.

### System Prompt Construction

```python
def build_instrument_prompt(
    instrument: str,
    active_instruments: list[str],
    genres: list[str],
    genre_weights: list[float]
) -> str:
    others = [i for i in active_instruments if i != instrument]
    genre_parts = [
        f"{g} ({w:.0%})" for g, w in zip(genres, genre_weights) if w > 0.05
    ]
    return (
        f"Solo {instrument}. "
        f"Improvise over the audio input. "
        f"Style: {', '.join(genre_parts)}. "
        f"Playing alongside: {', '.join(others) if others else 'the human player'}. "
        f"Stay rhythmically locked to the loop."
    )
```

---

## 11. Metronome and Timing Engine

### Click Track Generation

```python
import numpy as np

SAMPLE_RATE = 48000

def generate_click_tone(freq_hz: float = 1000.0, duration_ms: float = 30.0,
                         sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    t = np.linspace(0, duration_ms / 1000, int(sample_rate * duration_ms / 1000))
    tone = np.sin(2 * np.pi * freq_hz * t)
    envelope = np.exp(-t * 80)  # Fast decay
    return (tone * envelope).astype(np.float32)

DOWNBEAT_CLICK = generate_click_tone(1500.0)  # High pitch for beat 1
BEAT_CLICK = generate_click_tone(1000.0)       # Lower pitch for beats 2-4

def build_metronome_track(bpm: int, measures: int,
                           sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    beats_per_measure = 4
    seconds_per_beat = 60.0 / bpm
    total_beats = measures * beats_per_measure
    total_samples = int(total_beats * seconds_per_beat * sample_rate)
    track = np.zeros(total_samples, dtype=np.float32)

    for beat in range(total_beats):
        start = int(beat * seconds_per_beat * sample_rate)
        click = DOWNBEAT_CLICK if beat % beats_per_measure == 0 else BEAT_CLICK
        end = min(start + len(click), total_samples)
        track[start:end] += click[:end - start]

    return track
```

### Master Clock

```python
import time, math, threading

class TimingEngine:
    def __init__(self, bpm: int, measures: int, sample_rate: int = 48000):
        self.bpm = bpm
        self.measures = measures
        self.sample_rate = sample_rate
        self.loop_duration = measures * 4 * (60.0 / bpm)
        self.loop_samples = int(self.loop_duration * sample_rate)
        self._start_time: float | None = None
        self._callbacks: list[callable] = []

    def start(self):
        self._start_time = time.perf_counter()
        threading.Thread(target=self._loop_thread, daemon=True).start()

    def _loop_thread(self):
        pass_num = 0
        while True:
            wait = self._time_until_pass(pass_num)
            if wait > 0:
                time.sleep(wait)
            for cb in self._callbacks:
                cb(pass_num)
            pass_num += 1

    def _time_until_pass(self, pass_num: int) -> float:
        target = self._start_time + pass_num * self.loop_duration
        return target - time.perf_counter()

    def on_pass(self, callback: callable):
        self._callbacks.append(callback)
```

---

## 12. Remote Access: SSH + VNC to ThinkPad

### SSH (Terminal Access — Primary Interface)

Windows 10/11 has a built-in OpenSSH Server:

```powershell
# Install (run as Admin in PowerShell)
Add-WindowsCapability -Online -Name OpenSSH.Server~~~~0.0.1.0

# Start and enable auto-start
Start-Service sshd
Set-Service -Name sshd -StartupType 'Automatic'

# Allow firewall
New-NetFirewallRule -Name sshd -DisplayName 'OpenSSH Server (sshd)' `
    -Enabled True -Direction Inbound -Protocol TCP -Action Allow -LocalPort 22
```

Connect from any machine on same network:
```bash
ssh username@thinkpad-ip
python improv_loop.py --bpm 120 --measures 4
```

**VS Code Remote SSH** (recommended for development):
- Install "Remote - SSH" extension: https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-ssh
- Connect to ThinkPad: Ctrl+Shift+P → "Remote-SSH: Connect to Host" → `username@thinkpad-ip`
- Full IDE experience over SSH

### VNC (GUI Access — For Analog Lab / Visual Monitoring)

Needed to interact with Analog Lab (synthesizer software):

**TightVNC** (free, open source):
- Download: https://www.tightvnc.com/download.php
- Install TightVNC Server on ThinkPad
- Connect with TightVNC Viewer from remote machine
- Port: 5900 (default)

**Security**: Tunnel VNC over SSH for encryption:
```bash
# From remote machine: create SSH tunnel
ssh -L 5900:localhost:5900 username@thinkpad-ip

# Then connect VNC viewer to localhost:5900
```

### Network Requirements

- Both machines on same LAN (or VPN)
- Find ThinkPad IP: `ipconfig` in Windows cmd → "Wireless LAN adapter" or "Ethernet adapter"
- For static IP: Windows Settings → Network → IP settings → Manual

---

## 13. Python Implementation Plan

### Module Structure

```
improv_loop/
├── improv_loop.py            # Entry point, CLI args, session orchestrator
├── config/
│   ├── pbf4_cc_map.json      # PBF4 MIDI CC assignments (update after testing hardware)
│   └── defaults.json         # Default BPM, measures, instruments, genres
├── modules/
│   ├── midi_controller.py    # PBF4 MIDI input, mido callbacks, CC dispatch
│   ├── audio_capture.py      # sounddevice/PyAudioWPatch capture, device selection
│   ├── audio_playback.py     # sounddevice output stream, mixing callback
│   ├── timing_engine.py      # Master clock, loop boundary, pass management
│   ├── loop_buffer.py        # Thread-safe double buffer for user loop audio
│   ├── metronome.py          # Click track generation and scheduling
│   ├── magenta_client.py     # HTTP client for Magenta RT FastAPI server
│   ├── ai_instance.py        # Per-instrument AI voice (generation pipeline)
│   ├── mixer.py              # Multi-stream mixing (numpy)
│   ├── prompt_builder.py     # Style prompts from genres + instruments
│   └── state_machine.py      # Session state: IDLE→COUNTDOWN→RECORDING→PLAYING
├── server/
│   └── magenta_server.py     # FastAPI wrapper for Magenta RT (runs on Colab/GPU machine)
├── requirements.txt
├── requirements_server.txt   # Server-side dependencies (Magenta RT, FastAPI)
├── LESSONS.md                # MUST BE UPDATED after every session
└── CLAUDE.md                 # This file — MUST BE UPDATED when architecture changes
```

### Session State Machine

```
IDLE
  [Button 1] ──→ COUNTDOWN

COUNTDOWN (2-measure metronome)
  [2 measures elapsed] ──→ RECORDING
  [Button 2] ──→ IDLE (cancel)

RECORDING (N-measure user input capture)
  [N measures elapsed] ──→ PLAYING (loop_buffer.set_pending(recorded_audio))
  [Button 1] ──→ COUNTDOWN (restart recording)
  [Button 2] ──→ IDLE (cancel)

PLAYING
  [Button 1] ──→ COUNTDOWN_RERECORD (AI instruments continue, user re-records)
  [Button 2] ──→ STOPPING
  [Button 3] ──→ enable AI Instance 3 (stays PLAYING)
  [Button 4] ──→ enable AI Instance 4 (stays PLAYING)
  [loop boundary] ──→ loop_buffer.swap_on_boundary()

COUNTDOWN_RERECORD
  [2 measures elapsed] ──→ RERECORDING
  (AI instruments continue uninterrupted during this state)

RERECORDING
  [N measures elapsed] ──→ PLAYING (loop_buffer.set_pending(new_audio))

STOPPING
  [all connections closed, audio faded out] ──→ IDLE
```

### Implementation Order

Each step must be fully working before moving to the next. Every step ends with a test that can be run independently.

#### Pre-Coding Checklist (MUST complete before any code)
- [ ] Plug in PBF4 → run `python -c "import mido; [print(m) for m in mido.open_input()]"` → log all CC numbers to `config/pbf4_cc_map.json`
- [ ] Install Surge XT → set output to VB-Cable → verify audio appears in Windows sound settings
- [ ] Verify Lyria RealTime API is accessible at https://aistudio.google.com/ (check for "Music Generation" in the model list)
- [ ] Clone repo, create virtualenv, confirm Python 3.11+ is active

#### Step 1 — Config + Project Skeleton
- Load `config.yaml` via `pydantic-settings` or `PyYAML`
- Factory functions that return the correct concrete class based on config
- Abstract base classes for all 4 layers
- **Test**: `python improv_loop.py --dry-run` prints loaded config, exits cleanly

#### Step 2 — PBF4 MIDI Monitor
- `PBF4Controller` reads CC/button events, dispatches to `GenerationParams`
- **Test**: Move every knob/fader/button, confirm terminal prints correct parameter name + value

#### Step 3 — Audio Capture + Passthrough Monitor
- `VBCableCapture` records from "CABLE Output" device at 48kHz
- `PassthroughMonitor` immediately plays captured audio to output device
- **Test**: Play Surge XT → hear it through speakers with ~30ms delay; record 5 seconds to WAV

#### Step 4 — Metronome + Timing Engine
- `TimingEngine` drives absolute-time loop boundaries
- Metronome click track plays on countdown
- **Test**: `--bpm 120 --measures 4` → metronome plays 8 seconds exactly, repeats, no drift over 10 loops

#### Step 5 — Loop Record + Playback (No AI)
- State machine: IDLE → COUNTDOWN → RECORDING → PLAYING
- PBF4 Button 1 triggers countdown; loop captured and plays back continuously
- **Test**: Record 4-bar phrase → loops cleanly with no click at loop boundary

#### Step 6 — Single Lyria WebSocket Connection
- `LyriaBackend` connects, sends text prompt + params, streams back audio
- **Test**: Send a static prompt "solo jazz piano at 120 BPM" → receive and play 8 seconds of audio

#### Step 7 — First AI Instrument in the Loop
- Send user loop audio to Lyria as audio prompt
- After buffer pass, mix AI output with user loop in output callback
- **Test**: Record loop → hear AI piano improvising over it after 2 passes

#### Step 8 — Multi-Instance Cascade (4 AI voices)
- Instantiate 2 `LyriaBackend` instances (default) + 2 optional (PBF4 Buttons 3/4)
- Each instance receives cumulative mix of all prior voices
- **Test**: Enable all 4 → confirm cascade timing (AI 1 at pass 3, AI 2 at pass 5, etc.)

#### Step 9 — Genre Faders + Knob Parameters
- Fader CC → `genre_weights` dict → `WeightedPrompts` rebuilt on every Lyria call
- Knob CC → `guidance` / `density` → passed to Lyria params
- **Test**: Move faders → hear style change after ~2 passes; confirm no crashes on rapid movement

#### Step 10 — Re-Recording Mid-Session
- PBF4 Button 1 during PLAYING → new countdown → new loop → atomic double-buffer swap
- **Test**: Record loop → AI plays → record new loop → AI adapts → seamless

#### Step 11 — Magenta RT Backend (Phase 2, swap only)
- Implement `MagentaRTBackend` (HTTP + audio injection)
- Change `config.yaml: ai_backend: "magenta_rt"` → everything else unchanged
- **Test**: Identical session behavior as Lyria; different sound character

#### Step 12 — Error Handling + Remote Access
- Reconnection logic for Lyria WebSocket drops
- SSH server + TightVNC configured on ThinkPad
- Graceful `CTRL+C` shutdown: fades audio, closes connections, saves session log

---

## 14. Dependency Stack

### Client (ThinkPad) — `requirements.txt`

```txt
mido>=1.3.0
python-rtmidi>=1.5.8
sounddevice>=0.4.6
PyAudioWPatch>=0.2.12.5
numpy>=1.26.0
scipy>=1.12.0
librosa>=0.10.0
httpx>=0.27.0          # Async HTTP client for Magenta RT server
requests>=2.31.0        # Sync fallback
```

### Server (Colab or GPU Machine) — `requirements_server.txt`

```txt
fastapi>=0.110.0
uvicorn>=0.29.0
pyngrok>=7.0.0
# Magenta RT (install from source per instructions in GitHub repo)
# tf2jax==0.3.8
# Hugging Face Hub for model download
huggingface_hub>=0.22.0
numpy>=1.26.0
librosa>=0.10.0
scipy>=1.12.0
```

### Audio Routing (Windows, Install Separately)

- **VB-Cable**: https://shop.vb-audio.com/en/win-apps/11-vb-cable.html (free)
- OR **Voicemeeter Banana**: https://voicemeeter.com/ (free, more control)

### Optional (MIDI Synthesis Fallback)

```txt
pyfluidsynth>=1.3.3
```

---

## 15. Free Tier / Cost Strategy

### Magenta RT on Google Colab (Free)

1. Open Colab: https://colab.research.google.com/
2. Enable TPU runtime: Runtime → Change runtime type → TPU
3. Run the audio injection notebook: https://colab.research.google.com/github/magenta/magenta-realtime/blob/main/notebooks/Magenta_RT_Audio_Injection.ipynb
4. Add FastAPI + pyngrok to expose as HTTP endpoint (see Section 3 server code)
5. Copy ngrok URL into `config/defaults.json` as `magenta_server_url`

**Limitations of free Colab**:
- Session disconnects ~every 12 hours (or sooner if idle)
- One TPU instance at a time per account
- Running 4 simultaneous Magenta RT instances on one Colab may be too slow — test first

**Workaround for 4 instances**: Run a single Magenta RT instance and call it sequentially for each instrument voice (add ~5s latency per instrument), or get 4 separate Colab notebooks open in 4 browser tabs.

### Lyria RealTime API (Free Trial Fallback)

If Magenta RT Colab proves too unreliable, use Lyria:
1. Create Google account
2. Go to https://aistudio.google.com/
3. Generate API key at https://aistudio.google.com/apikey
4. Free daily quota available (check current limits at https://ai.google.dev/pricing)
5. Parameters: `guidance`, `density`, `bpm`, `WeightedPrompts` (no audio injection)

---

## 16. Runtime Arguments and Configuration

```bash
python improv_loop.py \
    --bpm 120 \
    --measures 4 \
    --genres "jazz" "blues" "electronic" "classical" \
    --instruments "piano" "bass" "drums" "violin" \
    --input-device 2 \
    --output-device 3 \
    --midi-device "Intech Studio PBF4" \
    --magenta-server "https://abc123.ngrok-free.app" \
    --api-key "YOUR_KEY"  # or set GOOGLE_API_KEY env var
```

### `config/defaults.json`

```json
{
    "bpm": 120,
    "measures": 4,
    "sample_rate": 48000,
    "genres": ["jazz", "blues", "electronic", "classical"],
    "instruments": ["piano", "bass", "drums", "violin"],
    "ai_instances_default": 2,
    "guidance_default": 4.0,
    "density_default": 0.5,
    "brightness_default": 0.5,
    "magenta_server_url": "http://localhost:8000",
    "pbf4_cc_map": {
        "knob_guidance": 0,
        "knob_density": 1,
        "knob_bpm": 2,
        "knob_brightness": 3,
        "fader_genre_0": 4,
        "fader_genre_1": 5,
        "fader_genre_2": 6,
        "fader_genre_3": 7,
        "button_record": 8,
        "button_stop": 9,
        "button_enable_3": 10,
        "button_enable_4": 11
    }
}
```

---

## 17. Open Questions and Decisions Needed

Items marked ✓ are resolved. Items marked ✗ are still open.

| # | Question | Status | Answer |
|---|----------|--------|--------|
| 1 | MiniLab 3 vs. MicroLab mk3? | ✓ | **MicroLab mk3** confirmed — 25 keys, 2 touch strips, 4 buttons, NO knobs/faders |
| 2 | Analog Lab standalone? | ✓ | Plugin-only. **Surge XT standalone** for prototype; Analog Lab + Ableton in v3 |
| 3 | ThinkPad GPU? | ✓ | **No discrete GPU** — cloud-only inference |
| 4 | Audio routing method? | ✓ | **VB-Cable** + WASAPI shared mode, 48kHz |
| 5 | 4 parallel Magenta RT on free Colab? | ✓ | Not feasible on 1 session; use Lyria (Phase 1) or 4 separate Colabs (Phase 2) |
| 6 | AI instruments continue during re-record? | ✓ | **Yes** — AI keeps playing, user records new loop over them |
| 7 | Loop length minimum? | ✓ | CLI `--measures` flag, any int ≥ 1 |
| 8 | WSL2 or Windows Python? | ✓ | **Windows native Python** |
| 9 | Primary AI backend? | ✓ | **Phase 1: Lyria RealTime API** (free trial, parallel); **Phase 2: Magenta RT** (audio injection) |
| 10 | Monitoring strategy? | ✓ | **Python PassthroughMonitor** — script plays back captured audio through speakers |
| 11 | Prototype synth? | ✓ | **Surge XT standalone** — output to VB-Cable |
| 12 | Google AI Studio account? | ✓ | joshuamiao03@gmail.com — verify Lyria RealTime API is enabled at https://aistudio.google.com/ |
| 13 | GitHub repo? | ✓ | https://github.com/joshmiao1065/Generative_Musical_Improv_Loop |
| 14 | **PBF4 actual CC numbers?** | ✗ OPEN | **Must test hardware before writing MIDI code** — plug in PBF4, run mido monitor, move every control, log CC numbers to `config/pbf4_cc_map.json` |
| 15 | **Is Lyria RealTime Music API available on this Google account?** | ✗ OPEN | Lyria was in limited preview as of 2025; verify at https://aistudio.google.com/ — if not available, Lyria WebSocket approach may need adjustment |
| 16 | Beat-alignment preference for AI output? | ✗ OPEN | Free improv (accept drift, simpler) vs. beat-locked (trimmed to loop boundary, harder)? |
| 17 | How many genres must be specified? | ✗ OPEN | Can user provide 2 genres with 2 faders unused? Or always 4? |

---

## 18. Documentation Links

### Magenta RT
- GitHub: https://github.com/magenta/magenta-realtime
- Demo Colab: https://colab.research.google.com/github/magenta/magenta-realtime/blob/main/notebooks/Magenta_RT_Demo.ipynb
- **Audio Injection Colab**: https://colab.research.google.com/github/magenta/magenta-realtime/blob/main/notebooks/Magenta_RT_Audio_Injection.ipynb
- Hugging Face: https://huggingface.co/google/magenta-realtime
- arXiv paper: https://arxiv.org/abs/2508.04651
- Official site: https://magenta.withgoogle.com/magenta-realtime

### Lyria RealTime API (Fallback)
- Overview: https://ai.google.dev/gemini-api/docs/realtime-music-generation
- API Key (Google AI Studio): https://aistudio.google.com/apikey
- Pricing: https://ai.google.dev/pricing

### intech PBF4
- Product: https://intech.studio/shop/pbf4
- Docs: https://docs.intech.studio/
- MIDI reference: https://docs.intech.studio/reference-manual/grid-functions/midi/
- Grid Editor: https://docs.intech.studio/guides/grid/grid-adv/editor-201/

### Arturia MiniLab 3
- Product: https://www.arturia.com/products/hybrid-synths/minilab-3/overview
- MIDI Control Center Manual: https://dl.arturia.net/products/minilab-3/manual/minilab-3-mcc_Manual_1_14_1_EN.pdf
- Analog Lab V Manual: https://downloads.arturia.net/products/analoglab-v/manual/analog-lab-v_Manual_1_0_EN.pdf
- Arturia Software Center (installs/activates products): https://www.arturia.com/support/downloads

### Arturia MicroLab mk3 (if this is the correct device)
- Product: https://www.arturia.com/products/hybrid-synths/microlab-mk3/overview

### Audio Libraries
- sounddevice: https://python-sounddevice.readthedocs.io/
- PyAudioWPatch (WASAPI loopback): https://pypi.org/project/PyAudioWPatch/
- PyAudioWPatch WASAPI example: https://github.com/s0d3s/PyAudioWPatch/blob/master/examples/pawp_record_wasapi_loopback.py
- mido: https://mido.readthedocs.io/
- python-rtmidi: https://pypi.org/project/python-rtmidi/
- librosa: https://librosa.org/doc/latest/index.html

### Virtual Audio Routing (Windows)
- VB-Cable: https://shop.vb-audio.com/en/win-apps/11-vb-cable.html
- Voicemeeter Banana: https://voicemeeter.com/

### Standalone Synth Alternatives
- Surge XT (free, open source): https://surge-synthesizer.github.io/
- VITAL (free tier): https://vital.audio/

### FastAPI + ngrok (for Colab server)
- FastAPI: https://fastapi.tiangolo.com/
- pyngrok: https://pyngrok.readthedocs.io/
- ngrok with Colab: https://ngrok.com/docs/using-ngrok-with/googleColab

### Remote Access
- Windows OpenSSH: https://learn.microsoft.com/en-us/windows-server/administration/openssh/openssh_install_firstuse
- VS Code Remote SSH: https://code.visualstudio.com/docs/remote/ssh
- TightVNC: https://www.tightvnc.com/

### WSL2 USB Passthrough (if needed)
- usbipd-win: https://github.com/dorssel/usbipd-win

---

## LESSONS.md INSTRUCTIONS

Maintain `LESSONS.md` in this directory. After every session, append:

```markdown
## YYYY-MM-DD — [Topic]

**Attempted**: What was tried.
**Result**: What happened.
**Root cause**: Why it happened.
**Correct approach**: What works.
**Watch for**: What to avoid in future sessions.
```

### Pre-Populated Lessons (Known from Research)

```markdown
## 2026-04-15 — Magenta RT is not a REST API

**Attempted**: N/A (documented pre-emptively)
**Result**: N/A
**Root cause**: Magenta RT is an open-source Python library, not a hosted service.
**Correct approach**: Wrap with FastAPI + pyngrok in Colab, or run locally with sufficient GPU.
**Watch for**: Do not write code that makes HTTP requests to a "Magenta RT endpoint" without
first setting up the FastAPI server in the Colab notebook.

## 2026-04-15 — Temperature and TopK are not valid parameters

**Attempted**: N/A (documented pre-emptively)
**Result**: N/A
**Root cause**: These are LLM token-sampling parameters. Magenta RT uses guidance, density,
brightness, bpm, and stem_balance instead.
**Correct approach**: Map PBF4 knobs to guidance, density, brightness, bpm.
**Watch for**: Do not expose temperature/topk in the UI or API calls.

## 2026-04-15 — Analog Lab Intro is a plugin, may need a DAW host

**Attempted**: N/A (documented pre-emptively)
**Result**: N/A
**Root cause**: The MiniLab 3 bundle includes Analog Lab Intro (plugin format).
**Correct approach**: Either use Ableton Live Lite (also bundled) as host, verify if Analog Lab
has a standalone executable, or switch to Surge XT (free standalone synth).
**Watch for**: Do not assume Analog Lab runs standalone without verifying first.

## 2026-04-15 — MiniLab 3 ≠ MicroLab mk3

**Attempted**: N/A (documented pre-emptively)
**Result**: User said "MicroLab mk3" but likely owns MiniLab 3 (different products).
**Root cause**: Similar product names.
**Correct approach**: Confirm by reading physical device label before writing any MIDI code.
**Watch for**: CC numbers and control layout differ between models.
```

---

*Document version: 2 | Last updated: 2026-04-15*
*Update this file and LESSONS.md after every implementation session.*
