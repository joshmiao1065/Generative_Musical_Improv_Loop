# CLAUDE.md — Improv Loop: Real-Time AI Music Improvisation System

> **Project**: Cooper Union — Generative Machine Learning, Final Project
> **Developer**: Josh Miao
> **GitHub**: https://github.com/joshmiao1065/Generative_Musical_Improv_Loop
> **Last Updated**: 2026-04-26 (Session 8 — voice enable/disable inversion fixed, shape mismatch fixed, debug flags added, Analog Lab routing documented)

---

## CRITICAL NOTICE FOR THE IMPLEMENTING LLM

Read this file AND `LESSONS.md` before writing any code. After every session, update both files. Common mistakes:
1. Assuming Magenta RT is a REST API — it is a Python library deployed on Modal (Section 3)
2. Using `min_containers` on parameterized Modal classes — use `scaledown_window` instead (LESSONS.md)
3. Writing MIDI CC numbers before testing physical PBF4 hardware (Section 2)
4. Not calling `ping_all()` before a session — containers need ~3 min to warm on first use
5. `AudioMixer._voice_enabled` starts `[False, False, False]` — voices must be explicitly enabled via Button 2/3/4
6. Voice audio is trim/pad-corrected at each loop boundary — the strict shape check was a bug (now fixed)
7. `pbf4_cc_map.json` has empty `cc_controls` — CC numbers were manually entered, must be verified with hardware

---

## 1. Physical Setup (All Confirmed)

```
[MicroLab mk3]  ──USB-C MIDI──▶ [ThinkPad]  (musical input only — 25 keys, no knobs)
[intech PBF4]   ──USB MIDI───▶ [ThinkPad]  (ALL parameter control)
[Surge XT]      ──audio──▶ VB-Cable "CABLE Input" ──▶ Python captures "CABLE Output"
[ThinkPad]      ──asyncio HTTP──▶ Modal.com (3× A100-80GB, one per AI voice)
[ThinkPad]      ──audio──▶ speakers (Python output callback mixes user+AI)
```

**Session flow:**
1. `python improv_loop.py --bpm 120 --beats 16` — starts script, warms containers
2. PBF4 Button 1 → 2-bar countdown → record user loop
3. Loop plays; AI voices generate in parallel (buffer pass architecture)
4. AI voices join one by one (Voice 1 at pass 3, Voice 2 at pass 5, Voice 3 at pass 7)
5. PBF4 knobs/faders adjust live generation params
6. PBF4 Button 1 again → re-record; AI continues uninterrupted
7. PBF4 Button 2 → graceful stop

---

## 2. Hardware: intech PBF4

**12 controls: 4 buttons (top) × 4 faders (middle) × 4 knobs (bottom)**

| Column | Button | Fader | Knob |
|--------|--------|-------|------|
| 1 | Record start/restart | Genre 1 weight | `guidance_weight` [0–10] |
| 2 | Stop session | Genre 2 weight | `temperature` [0–4] |
| 3 | Enable AI Voice 3 | Genre 3 weight | `topk` [0–1024] |
| 4 | Enable AI Voice 4 | Genre 4 weight | `model_feedback` [0–1] |

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
- Cost: ~$7.50/hr for 3 voices active. Free tier = ~4 hrs/month.

**Architecture (locked):** One A100-80GB container per voice. All 3 called in parallel via `asyncio.gather`. One-pass lag between voices is intentional — each voice hears previous pass's outputs. See `src/modal_client.py`.

**CONFIRMED Parameters** (all map to PBF4 knobs):

| kwarg | Type | Range | PBF4 |
|-------|------|-------|------|
| `guidance_weight` | float | 0–10 | Knob 1 |
| `temperature` | float | 0–4 | Knob 2 |
| `topk` | int | 0–1024 | Knob 3 |
| `model_feedback` | float | 0–1 | Knob 4 |
| `model_volume` | float | 0–1 | fixed |
| `bpm` | int | 60–200 | from CLI |
| `beats_per_loop` | int | 1–64 | from CLI |

**Lyria RealTime API: ELIMINATED.** Text-only input, no audio injection. Do not revisit.
**Colab: ELIMINATED.** Session disconnects, no persistent URL. Modal is the sole deployment target.

---

## 4. Audio Routing

```
Surge XT → "CABLE Input" (VB-Cable) → "CABLE Output" → Python InputStream (48kHz stereo float32)
                                                              ↓
                                                    loop capture buffer
                                                              ↓
                                               WAV bytes → Modal VoiceServer × 3
                                                              ↓
                                               WAV bytes → numpy decode
                                                              ↓
                                          Python OutputStream: user_loop + AI mix → speakers
```

**VB-Cable setup** (Windows, one-time):
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

**Monitoring**: Python output callback mixes VB-Cable capture + AI outputs → speakers. ~20–40ms passthrough latency. No Voicemeeter needed.

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
│   ├── midi_controller.py      # ✓ PBF4Controller — CC→GenerationParams, button callbacks
│   ├── audio_mixer.py          # ✓ AudioMixer — voice enable/disable + shape trim/pad fixed
│   ├── audio_devices.py        # ✓ device auto-detect + list_devices() + VB-Cable helper
│   ├── loop_capture.py         # ✓ ring buffer capture with beat-aligned snapshot
│   └── timing_engine.py        # ✓ perf_counter pass-boundary clock + metronome
├── server/
│   └── magenta_server.py       # ✓ VoiceServer — deployed on Modal A100
├── scripts/
│   ├── discover_cc.py          # ✓ interactive CC discovery → pbf4_cc_map.json
│   ├── prime_server.py         # ✓ warm up deployed Modal containers
│   ├── test_audio_logic.py     # ✓ unit tests (runs in WSL, no hardware required)
│   └── validate_pbf4.py        # ✓ live param readout — run after filling pbf4_layout.json
├── CLAUDE.md                   # this file
└── LESSONS.md                  # bugs and lessons — READ BEFORE CODING
```

**All core components built and tested.** Ready for end-to-end hardware testing.

**To run a session:**
```bat
REM Test audio pipeline first (no Modal credits spent):
.venv\Scripts\python improv_loop.py --dry-run

REM List devices to confirm VB-Cable and PBF4 are visible:
.venv\Scripts\python improv_loop.py --list-devices

REM Real session with Modal:
modal deploy server/magenta_server.py   # one-time deploy (already done)
.venv\Scripts\python improv_loop.py --bpm 120 --beats 16
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
| 2 | user_loop (buffer) | Voice 0 done; Voice 1 generating |
| 3 | user_loop + Voice 0 | Voice 1 generating |
| 4 | user_loop + V0 (buffer) | Voice 1 done; Voice 2 generating |
| 5 | user_loop + V0 + V1 | Voice 2 generating |
| 6+ | all 3 voices live | next pass generating |

**Parallel dispatch**: All active voices called simultaneously via `asyncio.gather`. Wall time = max(V0, V1, V2) ≈ 5.68s, not their sum.

**Timing**: Use `time.perf_counter()` absolute time anchoring for loop boundaries — `time.sleep()` on Windows has ~15ms jitter that accumulates over many passes.

---

## 7. State Machine

```
IDLE → [Button 1] → COUNTDOWN (2-bar metronome)
COUNTDOWN → [2 bars elapsed] → RECORDING
COUNTDOWN → [Button 2] → IDLE
RECORDING → [N bars elapsed] → PLAYING (loop captured, generation starts)
PLAYING → [Button 1] → COUNTDOWN_RERECORD (AI voices continue uninterrupted)
PLAYING → [Button 2] → STOPPING
PLAYING → [Button 3] → enable Voice 3 (stays PLAYING)
PLAYING → [Button 4] → enable Voice 4 (stays PLAYING)
PLAYING → [pass boundary] → swap double buffer if pending loop exists
COUNTDOWN_RERECORD → [2 bars] → RERECORDING
RERECORDING → [N bars] → PLAYING (new loop swapped in atomically)
STOPPING → [fade + close] → IDLE
```

---

## 8. Genre Blending

PBF4 faders control `genre_weights[0..3]` (0.0–1.0 each). Style prompt is built each pass:
- Weighted text embeddings: `style = Σ(weight_i × embed(f"{genre_i} {instrument}"))` normalized
- Fallback single prompt: `"jazz piano: heavy jazz, light blues"` from weights

Genre list and instrument list configurable at CLI: `--genres "jazz" "blues" --instruments "piano" "bass" "drums"`

---

## 9. Open Questions

| # | Question | Status |
|---|----------|--------|
| 1 | PBF4 actual CC numbers for knobs/faders? | **⚠ OPEN** — `cc_controls` in pbf4_cc_map.json is empty; re-run discover_cc.py and move all knobs/faders |
| 2 | Beat-alignment preference? | **⚠ OPEN** — free improv (accept drift) vs beat-locked (trim to boundary)? |
| 3 | How many genres must be specified? | **⚠ OPEN** — 2 genres with 2 faders unused OK? |
| 4 | Analog Lab routing confirmed? | **⚠ OPEN** — documented (standalone → CABLE Input) but not yet tested; Surge XT is confirmed working |
| 5 | VB-Cable installed? | **⚠ OPEN** — user purchasing; code is ready (`--capture-device "CABLE Output"`) |

---

## 10. Key Docs

| Resource | URL |
|----------|-----|
| Magenta RT GitHub | https://github.com/magenta/magenta-realtime |
| Audio Injection notebook | https://colab.research.google.com/github/magenta/magenta-realtime/blob/main/notebooks/Magenta_RT_Audio_Injection.ipynb |
| Modal docs | https://modal.com/docs |
| VB-Cable | https://shop.vb-audio.com/en/win-apps/11-vb-cable.html |
| Surge XT | https://surge-synthesizer.github.io/ |
| intech PBF4 docs | https://docs.intech.studio/ |
| sounddevice | https://python-sounddevice.readthedocs.io/ |
| mido | https://mido.readthedocs.io/ |

---

## 11. Dependencies

**Client (ThinkPad) — `requirements.txt`**:
```
mido>=1.3.0
python-rtmidi>=1.5.8
sounddevice>=0.4.6
numpy>=1.26.0
librosa>=0.10.0
soundfile>=0.12.0
modal>=0.73.0
```

**Modal containers**: All server dependencies baked into the image at `modal deploy` time. See `server/magenta_server.py` for the full image definition.
