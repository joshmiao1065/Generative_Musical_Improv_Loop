# CLAUDE.md — Improv Loop: Real-Time AI Music Improvisation System

> **Project for**: Cooper Union — Generative Machine Learning, Final Project
> **Developer**: Josh Miao
> **Goal**: A Python terminal script that orchestrates up to 4 parallel AI music instruments and a live human player, synchronized in real-time, controlled by the intech PBF4 MIDI controller.
> **Last Updated**: 2026-04-15 (Session 3 — MicroLab mk3 confirmed, VB-Cable verified, cloud-only deployment locked in, Analog Lab routing chain clarified)

---

## CRITICAL NOTICE FOR THE IMPLEMENTING LLM

**Before writing a single line of code, read this entire document.** Every design decision, known pitfall, and architectural constraint is documented here. The most common mistake is beginning implementation without fully understanding:
1. The audio synchronization problem (Section 8)
2. The Magenta RT deployment model — it is NOT a cloud REST API out of the box (Section 3)
3. The audio routing chain for the user's synthesizer input (Section 5)

**Autonomous documentation rule**: After every implementation session or significant discovery, the LLM MUST:
- Update this file (`CLAUDE.md`) with any changes to architecture, confirmed/rejected decisions, or new constraints
- Append to `LESSONS.md` (create if missing) with lessons learned, mistakes made, and correct approaches
- Update Section 17 (Open Questions) to mark resolved items and add new ones

Failing to do this will cause repeated mistakes across sessions.

---

## TABLE OF CONTENTS

1. [Project Overview](#1-project-overview)
2. [Hardware: intech PBF4 Controller](#2-hardware-intech-pbf4-controller)
3. [AI Backend: Magenta RT — Architecture and Deployment](#3-ai-backend-magenta-rt--architecture-and-deployment)
4. [User Input Hardware: Arturia MiniLab 3 + Analog Lab](#4-user-input-hardware-arturia-minilab-3--analog-lab)
5. [Audio Routing Chain](#5-audio-routing-chain)
6. [Full System Data Flow Diagram](#6-full-system-data-flow-diagram)
7. [Loop Architecture and Playback Logic](#7-loop-architecture-and-playback-logic)
8. [Audio Synchronization Challenges](#8-audio-synchronization-challenges-detailed)
9. [Genre Blending via Weighted Prompts](#9-genre-blending-via-weighted-prompts)
10. [Instrument Assignment and AI Instance Management](#10-instrument-assignment-and-ai-instance-management)
11. [Metronome and Timing Engine](#11-metronome-and-timing-engine)
12. [Remote Access: SSH + VNC to ThinkPad](#12-remote-access-ssh--vnc-to-thinkpad)
13. [Python Implementation Plan](#13-python-implementation-plan)
14. [Dependency Stack](#14-dependency-stack)
15. [Free Tier / Cost Strategy](#15-free-tier--cost-strategy)
16. [Runtime Arguments and Configuration](#16-runtime-arguments-and-configuration)
17. [Open Questions and Decisions Needed](#17-open-questions-and-decisions-needed)
18. [Documentation Links](#18-documentation-links)

---

## 1. Project Overview

### What This Is

A Python terminal script (running on a ThinkPad laptop, accessed remotely via SSH/VNC) that turns a keyboard synthesizer + MIDI controller into an AI improv band. The human player performs a looped phrase using the Arturia MiniLab 3 + Analog Lab synthesizer. Up to 4 AI instrument voices powered by Magenta RT improvise in response, each building on the previous voice's combined output in a cascading loop architecture.

### Physical Setup

```
[Arturia MiniLab 3] ──USB-C──→ [ThinkPad]
[intech PBF4]       ──USB──→  [ThinkPad]
[ThinkPad]          ──Audio──→ [Speakers/Headphones]
[Remote machine]    ──SSH/VNC──→ [ThinkPad] (for script control)

[ThinkPad] ──HTTP──→ [Magenta RT Server] (Colab TPU or local GPU)
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

### Confirmed Parameters (REAL model parameters, not prompt tricks)

| Parameter | Type | Range | Description |
|-----------|------|-------|-------------|
| `bpm` | int | 60–200 | Beats per minute — cross-attention conditioned |
| `guidance` | float | 0.0–6.0 | Prompt adherence strength |
| `density` | float | 0.0–1.0 | Note density (sparse ↔ busy) |
| `brightness` | float | 0.0–1.0 | Tonal brightness |
| `stem_balance` | dict | per-stem floats | Relative instrument loudness in mix |
| `chromas` | list[float] | 12 values | Harmonic content (pitch class emphasis) |
| Text prompts | str | — | Style descriptions: "jazz piano", "ambient synth" |
| Audio prompts | audio | WAV/MP3 | Style conditioning from reference audio |
| Audio injection | audio | live/recorded | Mixed into context window each step |

> **"Temperature" and "TopK"**: These are standard LLM token-sampling parameters. The Magenta RT paper and GitHub do not document exposing these directly to end users. Do NOT plan to control these via the PBF4. Use `guidance`, `density`, `brightness` instead.

### Audio Injection (Key Feature for This Project)

Magenta RT has a dedicated **audio injection** mode (separate Colab notebook):
- At each inference step, user audio is mixed with model output audio
- The mixed signal is encoded as SpectroStream coarse tokens
- These tokens feed into the next generation step as context
- The model improvises while "hearing" both its own output and the user's input

This is the exact behavior needed: the user loop is continuously injected, and each AI instance hears the previous cumulative mix.

**Audio Injection Colab**: https://colab.research.google.com/github/magenta/magenta-realtime/blob/main/notebooks/Magenta_RT_Audio_Injection.ipynb

### Deployment Options

#### Option A: Google Colab Free TPU + FastAPI + ngrok (RECOMMENDED FOR PROTOTYPING)

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

### CRITICAL: 4 Parallel Instances on Colab Free Tier

Running 4 simultaneous Magenta RT instances on a single Colab free TPU session is likely not feasible — the model is 800M parameters and each inference call is compute-heavy. Options:

| Strategy | How | Tradeoff |
|----------|-----|----------|
| **A — 4 Colab notebooks** | Open 4 separate browser tabs, each running a Magenta RT server | Free, works, but requires 4 ngrok URLs in config |
| **B — Sequential inference** | One Colab, one model, loop through all 4 instruments per pass | Free, slower: 4× inference latency per loop pass |
| **C — Lyria API x4** | 4 WebSocket connections to Lyria (cloud, paid) | Clean, truly parallel, has free trial |
| **D — Single model, voice multiplexing** | One model generates one long prompt per pass, split into 4 | Experimental, loss of instrument separation |

**Recommended path**:
1. **Prototype with Lyria API (Option C)** — free trial is sufficient to verify the loop architecture. Simpler, genuinely parallel, no server setup.
2. **Transition to Magenta RT (Option A)** once loop logic is proven — 4 Colab notebooks is clunky but free.
3. **For final demo**: consider Colab Pro ($10/month) for a single high-memory session that may support 4 instances.

> The key insight: Lyria API and Magenta RT both accept audio conditioning and text prompts, so the loop architecture code is nearly identical. Build to an abstract `AIInstrumentClient` interface that can be swapped between backends.

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

> **Recommendation for prototype**: Use Surge XT standalone for simplicity. Configure it to output to VB-Cable. Reserve Analog Lab for when the system is stable and you want specific Arturia sounds.

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

### Monitoring: Hearing Your Own Playing

If synth output goes to VB-Cable, you won't hear it through speakers by default.

**Solution A — Voicemeeter Banana (recommended):**
- Free: https://voicemeeter.com/
- Receives from synth, routes simultaneously to VB-Cable AND physical speakers
- Also lets you control per-source volume

**Solution B — Windows Stereo Mix / Listen to this device:**
- Windows Sound Settings → Recording → "CABLE Output" → Properties → Listen tab
- Check "Listen to this device" → set playback through speakers
- Zero-cost but adds latency and is harder to configure

**Solution C — Script handles monitoring:**
- Python captures from VB-Cable and immediately plays back through output device
- Adds ~20–40ms of roundtrip latency to monitoring (likely imperceptible)

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

## 6. Full System Data Flow Diagram

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

1. **Phase 1**: PBF4 MIDI reading + CC print-out (verify hardware, map CC numbers)
2. **Phase 2**: Audio capture from VB-Cable/loopback + basic WAV recording
3. **Phase 3**: Metronome generation + playback via sounddevice
4. **Phase 4**: Timing engine + loop playback (no AI yet)
5. **Phase 5**: Magenta RT server (FastAPI on Colab) — single instance, single prompt
6. **Phase 6**: HTTP client → send user loop → receive AI audio → play back mixed
7. **Phase 7**: Multi-instance management (cascade)
8. **Phase 8**: Genre blend fader → prompt weights
9. **Phase 9**: PBF4 knobs → live Magenta RT parameter updates
10. **Phase 10**: Re-recording mid-session (double-buffer swap)
11. **Phase 11**: Error handling, reconnection logic, graceful shutdown
12. **Phase 12**: Remote access setup (SSH + VNC on ThinkPad)

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
| 1 | MiniLab 3 vs. MicroLab mk3? | ✓ RESOLVED | **MicroLab mk3** — 25 keys, 2 touch strips, 4 buttons, NO knobs/faders |
| 2 | Does Analog Lab run standalone? | ✗ OPEN | Plugin-only confirmed. Use Ableton Live Lite as host, OR use Surge XT standalone |
| 3 | ThinkPad GPU? | ✓ RESOLVED | **No discrete GPU** — cloud-only deployment required |
| 4 | Audio routing method? | ✓ RESOLVED | **VB-Cable** (WASAPI shared mode, 48kHz, ~14–21ms latency) |
| 5 | 4 parallel Magenta RT on free Colab? | ✓ RESOLVED | **Not feasible on one session** — see parallel instances strategy in Section 3 |
| 6 | AI instruments continue during re-record? | → | **Yes** — AI instruments keep playing; user records new loop over them |
| 7 | Minimum loop length? | → | CLI `--measures` arg; any integer ≥ 1 supported |
| 8 | WSL2 or Windows native Python? | ✓ RESOLVED | **Windows native Python** (audio/MIDI simpler, no USB passthrough needed) |
| 9 | Primary AI backend: Lyria or Magenta RT? | → | **Start with Lyria API** (simpler, free trial, truly parallel); **transition to Magenta RT** for audio injection in v2 |
| 10 | PBF4 actual CC numbers? | ✗ OPEN | Must test hardware; update `config/pbf4_cc_map.json` |
| 11 | Surge XT vs Analog Lab as synth? | ✗ OPEN | Surge XT recommended for prototype; Analog Lab for final |
| 12 | Monitoring solution (hear own playing)? | ✗ OPEN | Voicemeeter Banana recommended; verify during audio setup phase |

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
