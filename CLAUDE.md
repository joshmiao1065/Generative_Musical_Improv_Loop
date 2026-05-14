# CLAUDE.md — Improv Loop: Real-Time AI Music Improvisation System

> **Project**: Cooper Union — Generative Machine Learning, Final Project
> **Developer**: Josh Miao
> **GitHub**: https://github.com/joshmiao1065/Generative_Musical_Improv_Loop
> **Last Updated**: 2026-05-03 (Session 11 — redesign: per-voice genres, stem-volume faders, 3-band EQ + reverb knobs)

---

## CRITICAL NOTICE FOR THE IMPLEMENTING LLM

Read this file AND `LESSONS.md` before writing any code. After every session, update both files. Common mistakes:
1. Assuming Magenta RT is a REST API — it is a Python library deployed on Modal (Section 3)
2. Using `modal.parameter()` with `min_containers` — they are incompatible. Use 3 separate named classes: `Voice0Server`, `Voice1Server`, `Voice2Server` (see Section 3)
3. Writing MIDI CC numbers before testing physical PBF4 hardware (Section 2)
4. Starting a session before containers are fully warm — cold boot takes **5–8 min** (model load + XLA compile). Wait for `*** READY ***` in Modal dashboard logs AND `ALL VOICES READY` printed by improv_loop.py. Button 1 is gated until all pings return.
5. `AudioMixer._voice_enabled` starts `[False, False, False]` — voices must be explicitly enabled via Button 2/3/4
6. Voice audio is trim/pad-corrected at each loop boundary — the strict shape check was a bug (now fixed)
7. `pbf4_cc_map.json` has empty `cc_controls` — CC numbers were manually entered, must be verified with hardware
8. Live monitoring passthrough uses `_MonitorFIFO` in `loop_capture.py` — do NOT revert to reading the last N frames from the ring buffer. That approach plays the same frames 4× when input blocksize (2048) > output blocksize (512), distorting pitch and timbre.
9. `_generation_in_flight` flag prevents queue buildup — only one Modal generation per voice set is dispatched at a time. If generation takes longer than one loop, subsequent passes are **intentionally skipped** (logged as WARNING). Do not remove this gate.
10. **Effects chain — Freeverb allpass buffer sizes**: Allpass filters 2–4 have delay buffers of 480, 371, and 245 samples — ALL smaller than the standard 512-sample audio block. The vectorised "read-all, write-all" block approach is WRONG here because the buffer wraps within one block, and a later position should read the value written earlier in the same block. **Correct fix**: sub-block at buffer-wrap boundaries (process `min(avail, n)` samples per iteration). See `src/effects_chain.py` and `scripts/test_effects_algorithms.py`. Comb filter buffers (all ≥ 1214 samples) are safe with the simple approach.
11. **Per-voice genres (NOT blended)**: Each of the 3 AI voices generates a dedicated genre. The server receives a `genre: str` per call and computes `embed_style(genre)` at call time. There is no blending. `VOICE_STYLES` in `magenta_server.py` is obsolete — do not use it.
12. **Faders are now stem volume controls** (NOT genre weights): CC 36 = user loop volume, CC 37/38/39 = AI voice 0/1/2 volume. There is no crossfader. `set_crossfade()` and `_crossfade_ai` have been removed from `AudioMixer`.
13. **EQ knobs modify filter coefficients mid-stream**: When a gain parameter changes, `sosfilt_zi` state from the old filter is reused with new coefficients. This causes a brief transient (< 5 samples). A one-pole smoother on the gain target eliminates audible clicks — do NOT skip it.

---

## 1. Physical Setup (All Confirmed)

```
[MicroLab mk3]  ──USB-C MIDI──▶ [ThinkPad]  (musical input only — 25 keys, no knobs)
[intech PBF4]   ──USB MIDI───▶ [ThinkPad]  (ALL parameter control)
[Surge XT]      ──audio──▶ VB-Cable "CABLE Input" ──▶ Python captures "CABLE Output"
[ThinkPad]      ──asyncio HTTP──▶ Modal.com (3× A100-80GB, one per AI voice)
[ThinkPad]      ──audio──▶ speakers (Python mixes live monitoring + user loop + AI voices)
```

**Session flow:**
1. `modal deploy server/magenta_server.py` — deploy (containers begin warming immediately, ~5–8 min first boot)
2. Watch improv_loop terminal for `ALL VOICES READY` (printed when all ping calls return)
3. `python improv_loop.py --bpm 120 --beats 16 --genres "jazz" "bossa nova" "electronic"` — starts script
4. PBF4 Button 1 → 2-bar countdown → record user loop
5. Loop plays; AI voices generate in parallel (buffer pass architecture)
6. AI voices join one by one (Voice 1 at pass 2, Voice 2 at pass 3, Voice 3 at pass 4)
7. PBF4 faders adjust stem volumes; knobs shape EQ/reverb on full mix
8. PBF4 Button 1 again → re-record; AI continues uninterrupted
9. QWERTY keys `4`/`5`/`6` cycle genre for Voice 1/2/3 (takes effect next generation pass)

---

## 2. Hardware: intech PBF4

**12 controls: 4 buttons (top) × 4 faders (middle) × 4 knobs (bottom)**

| Column | Button | Fader (CC 36–39) | Knob (CC 32–35) |
|--------|--------|------------------|-----------------|
| 1 | Record / Re-record | User track volume [0–1] | EQ Bass ±12 dB (shelf @ 250 Hz) |
| 2 | Toggle AI Voice 1 on/off | AI Voice 1 volume [0–1] | EQ Mid ±12 dB (peak @ 1 kHz) |
| 3 | Toggle AI Voice 2 on/off | AI Voice 2 volume [0–1] | EQ Treble ±12 dB (shelf @ 4 kHz) |
| 4 | Toggle AI Voice 3 on/off | AI Voice 3 volume [0–1] | Reverb wet [0–1] |

**Knob semantics:**
- Knob center (CC value 64) = 0 dB / flat for all EQ bands. Full left (CC 0) = −12 dB. Full right (CC 127) = +12 dB.
- Reverb knob: full left = dry, full right = large wet room (room_size ≈ 0.98, wet mix ≈ 0.8). Applied to entire output mix.
- EQ and reverb are processed locally in the audio callback (Python/numpy). Never route these to Modal.

**Genre cycling (QWERTY, always active regardless of --qwerty flag):**
- `4` → cycle Voice 1 to next genre in `--genres` list
- `5` → cycle Voice 2
- `6` → cycle Voice 3
- `?` → print current genres and volumes

**⚠ CC NUMBERS ENTERED MANUALLY — MUST VERIFY WITH HARDWARE:**
`config/pbf4_cc_map.json` has empty `cc_controls` — only buttons were captured during
the initial discovery run. Knobs and faders were not moved. The CC numbers in
`pbf4_layout.json` (CC 32-35 for knobs, CC 36-39 for faders) are manually entered
guesses. Verify them with hardware before relying on knob/fader control.
```bat
REM Run from project root in a Windows terminal (NOT WSL):
.venv\Scripts\python scripts\discover_cc.py
REM Move EVERY knob and fader through their full range, press every button
REM → config/pbf4_cc_map.json (check that cc_controls is now non-empty)
REM Then verify the CC numbers match pbf4_layout.json
.venv\Scripts\python scripts\validate_pbf4.py   REM live param readout
```
Buttons send CC 127 (press) and 0 (release). CC numbers configurable in intech Grid Editor (LUA). USB class-compliant — no driver needed on Windows.

**CONFIRMED:** MIDI port name is `'Intech Grid MIDI device 1'` (detected 2026-04-21).
**IMPORTANT:** Must run all MIDI/audio scripts with Windows Python (`.venv\Scripts\python`), NOT WSL — USB devices are not visible from WSL2.

Docs: https://docs.intech.studio/ | Grid Editor: https://docs.intech.studio/guides/grid/grid-adv/editor-201/

---

## 3. AI Backend: Magenta RT on Modal

**What it is**: 800M-param autoregressive transformer (JAX/T5X). Generates 2s of 48kHz stereo per `step()` call. Audio injection: feeds user loop + prior AI outputs into model's SpectroStream token context. Implemented in `src/magenta_backend.py`.

**Deployment (confirmed working):**
```
modal deploy server/magenta_server.py      # deploy (one-time)
python src/modal_client.py                # verify containers alive
modal app stop magenta-rt-server          # stop billing when not playing
```
App URL: https://modal.com/apps/joshuamiao03/main/deployed/magenta-rt-server

**Confirmed benchmark (A100-80GB):**
- Warm chunk: 1.40s per 2.0s audio = **RTF 1.431×** ✓
- 16-beat pass @ 120 BPM: 5.68s gen / 8.0s loop = **0.71×** ✓ (2.32s headroom)
- Cost: ~$7.50/hr for 3 voices active.

**Current GPU: A100-80GB** (A100-40GB attempted in Session 9 but OOMs — model's LLM computation graph requires >40GB VRAM, confirmed by XLA rematerialization failure on 2026-04-29. A10G is NOT viable — RTF 0.873× means it always takes longer to generate than the loop duration regardless of settings.)

**Architecture (locked):** Three separate named Modal classes — `Voice0Server`, `Voice1Server`, `Voice2Server` — each with `min_containers=1`. Containers warm on `modal deploy`, no prime step required. One A100-80GB container per voice. All 3 called in parallel via `asyncio.gather`. One-pass lag between voices is intentional. See `src/modal_client.py` and `server/magenta_server.py`.

**Cold boot sequence per container (~5–8 min total on first deploy):**
1. GPU allocated, Python process starts → container shows "active" on dashboard
2. SpectroStream model loads (~10–20s)
3. MagentaRT large model weights load (~30–60s, or instant if HF cache volume is warm)
4. XLA JIT compile on dummy input (~2–4 min first time per container)
5. Container logs `*** READY ***` → ping returns → `improv_loop.py` unblocks Button 1

**"Active" container ≠ ready.** A container showing as "active" on the dashboard means the GPU is allocated and Python has started. The `@modal.enter()` method (model load + JIT compile) may still be running. Ping calls will be pending/queued until `@modal.enter()` completes. Watch container logs for `*** READY ***`.

To stream container logs:
```
modal logs magenta-rt-server
```

**Generation parameters** (all sent from client on each pass):

| kwarg | Type | Range | Control |
|-------|------|-------|---------|
| `guidance_weight` | float | 0–10 | fixed at 3.0 (was Knob 1 — now knobs are EQ/reverb) |
| `temperature` | float | 0–2 | fixed at 1.0 |
| `topk` | int | 0–1024 | fixed at 40 |
| `model_feedback` | float | 0–1 | fixed at 0.7 |
| `genre` | str | any text | per-voice, cycled via QWERTY 4/5/6 |
| `bpm` | int | 60–200 | from CLI |
| `beats_per_loop` | int | 1–64 | from CLI |

**Note on genre**: `embed_style(genre)` is called at the start of each `generate_pass` on the Modal server. This runs the MusicCoca text encoder (< 100ms on GPU). The result is a raw numpy array — do NOT access `.embedding` on it. Pass it directly to `generate_chunk(style=...)`.

**Lyria RealTime API: ELIMINATED.** Text-only input, no audio injection. Do not revisit.
**Colab: ELIMINATED.** Session disconnects, no persistent URL. Modal is the sole deployment target.

---

## 4. Audio Routing

```
Surge XT → "CABLE Input" (VB-Cable) → "CABLE Output" → Python InputStream (48kHz stereo float32)
                                                              ↓
                                              ┌──────────────┴──────────────┐
                                              │                             │
                                        loop capture buffer         _MonitorFIFO (streaming)
                                              │                             │
                                    WAV bytes → Modal                       │
                                     VoiceServer × 3                        │
                                              │                             │
                                     numpy decode ◀──────────────────────┐  │
                                              │                           │  │
                                   Python OutputStream: live monitoring + user_loop + AI mix
                                              │
                                        EffectsChain (3-band EQ → Freeverb)
                                              │
                                          → speakers
```

**VB-Cable setup** (Windows, one-time, CONFIRMED WORKING 2026-04-26):
1. Install: https://shop.vb-audio.com/en/win-apps/11-vb-cable.html
2. Sound Settings → CABLE Input + CABLE Output → both set to 48kHz, 24-bit
3. Surge XT → Preferences → Audio → Output: "CABLE Input"
   **OR** Analog Lab → Settings → Audio → Output device: "CABLE Input"
4. Never use WASAPI exclusive mode with VB-Cable (sounddevice GH#520 — fails)
5. After installation, launch with: `python improv_loop.py --capture-device "CABLE Output"`
   (auto-detection prefers VB-Cable if present — this flag makes it explicit)

**Synth switching** (no code changes needed — only hardware routing changes):
- **Surge XT**: Preferences → Audio → Output: "CABLE Input"
- **Analog Lab Intro (standalone)**: Settings → Audio → Output: "CABLE Input"; MIDI Input: "MicroLab mk3"
- **Any DAW**: Master output → "CABLE Input" in DAW audio settings
- Python always captures from "CABLE Output" (auto-detected) or specify: `--capture-device "CABLE Output"`

**Live monitoring (CONFIRMED WORKING 2026-04-28):** The `_MonitorFIFO` in `loop_capture.py`
streams captured audio directly to `AudioMixer`'s output callback. Player hears themselves
through speakers with ~10–20ms latency at all times (idle, recording, and playback). Disable
with `--no-monitor` if needed.

**Audio format**: All internal audio is `(N, 2) float32 @ 48000 Hz`. WAV bytes over Modal are 24-bit PCM.

**Without VB-Cable (Stereo Mix fallback)**: Works for initial testing, but captures ALL speaker
audio. AI voice output bleeds back into the capture stream → fed to Modal → model responds to
its own output. Quality degrades over time. For testing only — do not use in production sessions.

---

## 5. File Structure

```
improv_loop/
├── improv_loop.py              # ✓ main orchestrator — all session logic, debug flags
├── config/
│   ├── pbf4_layout_template.json  # ✓ layout template with all 12 control labels
│   ├── pbf4_layout.json           # ✓ CC numbers filled in (manually — verify with hardware!)
│   └── pbf4_cc_map.json           # ← auto-generated by scripts/discover_cc.py
│                                  #   WARNING: cc_controls is empty — knobs/faders not yet discovered
├── src/
│   ├── magenta_backend.py      # ✓ AIVoice, MagentaRTCFGTied, GenerationParams
│   ├── modal_client.py         # ✓ MagentaRTClient — async parallel voice dispatch
│   ├── midi_controller.py      # ✓ PBF4Controller — CC→volume/EQ/reverb, button callbacks
│   ├── audio_mixer.py          # ✓ AudioMixer — voice enable/disable + shape trim/pad
│   ├── effects_chain.py        # ✓ EffectsChain — 3-band EQ (RBJ biquad) + Freeverb reverb
│   ├── audio_devices.py        # ✓ device auto-detect + list_devices() + VB-Cable helper
│   ├── loop_capture.py         # ✓ ring buffer capture with beat-aligned snapshot
│   └── timing_engine.py        # ✓ perf_counter pass-boundary clock + metronome
├── server/
│   └── magenta_server.py       # ✓ VoiceServer — deployed on Modal A100
├── scripts/
│   ├── discover_cc.py          # ✓ interactive CC discovery → pbf4_cc_map.json
│   ├── prime_server.py         # ✓ warm up deployed Modal containers
│   ├── test_audio_logic.py     # ✓ unit tests for AudioMixer (runs in WSL, no hardware)
│   ├── test_effects_algorithms.py  # ✓ EQ + Reverb algorithm validation (22 tests, all pass)
│   └── validate_pbf4.py        # ✓ live param readout — run after filling pbf4_layout.json
├── CLAUDE.md                   # this file
└── LESSONS.md                  # bugs and lessons — READ BEFORE CODING
```

**To run a session:**
```bat
REM Test audio pipeline first (no Modal credits spent):
.venv\Scripts\python improv_loop.py --dry-run

REM List devices to confirm VB-Cable and PBF4 are visible:
.venv\Scripts\python improv_loop.py --list-devices

REM Real session with Modal:
modal deploy server/magenta_server.py   # one-time deploy (already done)
.venv\Scripts\python improv_loop.py --bpm 120 --beats 16 --genres "jazz" "bossa nova" "electronic"
```
Then press Button 1 on the PBF4 to start. Press Buttons 2/3/4 to enable AI voices.

---

## 6. Loop / Buffer Pass Architecture

**Loop duration**: `beats × (60 / bpm) × 48000` samples. At 120 BPM, 16 beats = 8.0s = 384,000 samples.

**Buffer pass system** (eliminates real-time generation constraint):

| Pass | User hears | Background |
|------|-----------|------------|
| 0 | countdown + recording | — |
| 1 | user_loop | Voice 0 generating |
| 2 | user_loop + Voice 0 | Voice 0 done; Voice 1 generating |
| 3 | user_loop + V0 + V1 | Voice 1 done; Voice 2 generating |
| 4+ | all 3 voices live | next pass generating |

**Parallel dispatch**: All active voices called simultaneously via `asyncio.gather`. Wall time = max(V0, V1, V2) ≈ 5.68s, not their sum.

**Timing**: Use `time.perf_counter()` absolute time anchoring for loop boundaries — `time.sleep()` on Windows has ~15ms jitter that accumulates over many passes.

---

## 7. State Machine

```
IDLE → [Button 1] → COUNTDOWN (2-bar metronome)
COUNTDOWN → [2 bars elapsed] → RECORDING
COUNTDOWN → [Button 1] → IDLE
RECORDING → [N bars elapsed] → PLAYING (loop captured, generation starts)
PLAYING → [Button 1] → COUNTDOWN  (AI voices continue uninterrupted)
PLAYING → [Button 2/3/4] → toggle Voice 1/2/3 on/off (stays PLAYING)
PLAYING → [Key 4/5/6] → cycle genre for Voice 1/2/3 (takes effect next pass)
PLAYING → [pass boundary] → swap double buffer if pending loop exists
PLAYING → [Ctrl-C / q] → STOPPING → IDLE
```

---

## 8. Per-Voice Genres

Each of the 3 AI voices is assigned a dedicated genre (text string). Genres do NOT blend.

**Initial assignment:** From `--genres` CLI arg (3+ strings). Voice 0 starts with `genres[0]`, Voice 1 with `genres[1]`, Voice 2 with `genres[2]`.

**Genre cycling:** QWERTY keys `4`/`5`/`6` cycle the genre for Voice 1/2/3 through the full `--genres` list (wrapping). Change takes effect on the next generation pass (not the current one in flight).

**Server-side:** Each VoiceServer's `generate_pass` receives `genre: str` and computes `embed_style(genre)` at call time. The style embedding is a raw numpy/JAX array — set directly as `self.voice.style_embedding`. Do NOT access `.embedding` on it.

**Example genres:** "jazz", "bossa nova", "electronic", "ambient", "blues", "classical", "lo-fi hip hop", "drum and bass". Magenta RT understands natural-language style descriptors. Longer descriptors (e.g. "smooth jazz with upright bass") work too.

---

## 9. Effects Chain (`src/effects_chain.py`)

All effects are applied to the **full output mix** (user loop + all AI voices + monitoring) inside the audio callback, after mixing and before the peak limiter. Processing budget: ~512 samples / 10ms per callback.

### 3-Band EQ (CC 32, 33, 34)

Implemented using RBJ Audio EQ Cookbook biquad filters (second-order sections, scipy.signal.sosfilt):

| Band | Filter type | Center freq | Knob center | Knob range |
|------|------------|-------------|-------------|------------|
| Bass (CC 32) | Low shelf | 250 Hz | 0 dB | −12 to +12 dB |
| Mid (CC 33) | Peaking EQ | 1000 Hz, Q=0.707 | 0 dB | −12 to +12 dB |
| Treble (CC 34) | High shelf | 4000 Hz | 0 dB | −12 to +12 dB |

**Parameter smoothing**: Each band uses a one-pole smoother on the gain parameter (τ ≈ 5ms, `coeff = exp(-2π × 200/48000)`). This prevents audible clicks when the knob is swept quickly. Filter coefficients are only recomputed when the smoothed gain differs from the current filter's gain by > 0.01 dB.

**State continuity**: The `zi` (filter state) from the previous buffer is reused with new coefficients. The resulting transient is < 5 samples and inaudible in practice.

### Reverb (CC 35) — Freeverb algorithm

Freeverb (Jezar at Dreampoint, public domain) — 8 parallel feedback comb filters + 4 series all-pass filters per channel.

**Delay buffer sizes at 48 kHz (scaled from original 44.1 kHz):**
- Comb filters L: 1214, 1293, 1390, 1475, 1547, 1622, 1694, 1759 (all > block size ✓)
- Comb filters R: +25 samples offset from L (stereo width)
- Allpass filters: 605, 480, 371, 245 (last three < 512-sample block — requires sub-block processing)

**Allpass implementation**: Must sub-block at buffer-wrap boundaries because buffers 2–4 are smaller than one audio block. See CRITICAL NOTICE item 10 and `scripts/test_effects_algorithms.py`.

**Knob mapping (CC 35, range 0.0–1.0):**
- 0.0: fully dry (reverb bypassed)
- 0.5: medium room, ~20% wet
- 1.0: large hall, ~80% wet, room_size = 0.98

---

## 10. Open Questions

| # | Question | Status |
|---|----------|--------|
| 1 | PBF4 actual CC numbers for knobs/faders? | **⚠ OPEN** — `cc_controls` in pbf4_cc_map.json is empty; re-run discover_cc.py and move all knobs/faders |
| 2 | Beat-alignment preference? | **⚠ OPEN** — free improv (accept drift) vs beat-locked (trim to boundary)? |
| 3 | Analog Lab routing confirmed? | **⚠ OPEN** — documented but not yet tested; Surge XT is confirmed working |
| 4 | VB-Cable installed? | **✓ RESOLVED** — installed 2026-04-26, device [2], 48kHz stereo, WASAPI shared mode |
| 5 | A100-80GB real-time benchmark post-soundfile-fix? | **⚠ OPEN** — librosa→soundfile fix reduces overhead. Expected RTF ~0.7× (well under 1.0×). Measure with a real session. |
| 6 | Per-voice genre style embedding correctness? | **⚠ OPEN** — `embed_style(genre)` called per pass. Does passing a short genre string ("jazz") produce a musically coherent embedding vs. the previous instrument description ("jazz piano solo")? Only testable on Modal. |
| 7 | EQ/reverb overhead in audio callback? | **⚠ OPEN** — estimated < 2ms for 512-sample block. Verify with `time.perf_counter()` profiling in first session. |
| 8 | Genre cycling key pickup at startup? | **⚠ OPEN** — QWERTY genre-cycle thread always runs; confirm it doesn't interfere with PBF4 mode or generate spurious input on some terminals. |

---

## 11. Key Docs

| Resource | URL |
|----------|-----|
| Magenta RT GitHub | https://github.com/magenta/magenta-realtime |
| Audio Injection notebook | https://colab.research.google.com/github/magenta/magenta-realtime/blob/main/notebooks/Magenta_RT_Audio_Injection.ipynb |
| RBJ Audio EQ Cookbook | https://www.w3.org/2011/audio/audio-eq-cookbook.html |
| Freeverb source (Jezar) | https://ccrma.stanford.edu/~jos/pasp/Freeverb.html |
| Modal docs | https://modal.com/docs |
| VB-Cable | https://shop.vb-audio.com/en/win-apps/11-vb-cable.html |
| Surge XT | https://surge-synthesizer.github.io/ |
| intech PBF4 docs | https://docs.intech.studio/ |
| sounddevice | https://python-sounddevice.readthedocs.io/ |
| mido | https://mido.readthedocs.io/ |

---

## 12. Dependencies

**Client (ThinkPad) — `requirements.txt`**:
```
mido>=1.3.0
python-rtmidi>=1.5.8
sounddevice>=0.4.6
numpy>=1.26.0
scipy>=1.10.0
soundfile>=0.12.0
modal>=0.73.0
```

**Modal containers**: All server dependencies baked into the image at `modal deploy` time. See `server/magenta_server.py` for the full image definition.
