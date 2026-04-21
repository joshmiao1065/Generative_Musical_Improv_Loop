# CLAUDE.md — Improv Loop: Real-Time AI Music Improvisation System

> **Project**: Cooper Union — Generative Machine Learning, Final Project
> **Developer**: Josh Miao
> **GitHub**: https://github.com/joshmiao1065/Generative_Musical_Improv_Loop
> **Last Updated**: 2026-04-20 (Session 6 — Modal deployed, client written, docs compacted)

---

## CRITICAL NOTICE FOR THE IMPLEMENTING LLM

Read this file AND `LESSONS.md` before writing any code. After every session, update both files. Common mistakes:
1. Assuming Magenta RT is a REST API — it is a Python library deployed on Modal (Section 3)
2. Using `min_containers` on parameterized Modal classes — use `scaledown_window` instead (LESSONS.md)
3. Writing MIDI CC numbers before testing physical PBF4 hardware (Section 2)
4. Not calling `ping_all()` before a session — containers need ~3 min to warm on first use

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

**⚠ CC NUMBERS UNKNOWN — MUST TEST HARDWARE BEFORE CODING MIDI:**
```bash
python -c "import mido; print(mido.get_input_names())"
python -c "import mido; port = mido.open_input(); [print(m) for m in port]"
# Move every knob/fader/button, log CC numbers to config/pbf4_cc_map.json
```
Buttons send CC 127 (press) and 0 (release). CC numbers configurable in intech Grid Editor (LUA). USB class-compliant — no driver needed on Windows.

Docs: https://docs.intech.studio/ | Grid Editor: https://docs.intech.studio/guides/grid/grid-adv/editor-201/

---

## 3. AI Backend: Magenta RT on Modal

**What it is**: 800M-param autoregressive transformer (JAX/T5X). Generates 2s of 48kHz stereo per `step()` call. Audio injection: feeds user loop + prior AI outputs into model's SpectroStream token context. Implemented in `src/magenta_backend.py`.

**Deployment (confirmed working):**
```
modal deploy modal/magenta_server.py      # deploy (one-time)
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
4. Never use WASAPI exclusive mode with VB-Cable (sounddevice GH#520 — fails)

**Monitoring**: Python output callback mixes VB-Cable capture + AI outputs → speakers. ~20–40ms passthrough latency. No Voicemeeter needed.

**Audio format**: All internal audio is `(N, 2) float32 @ 48000 Hz`. WAV bytes over Modal are 24-bit PCM.

---

## 5. File Structure

```
improv_loop/
├── improv_loop.py              # ← NOT YET BUILT — main orchestrator
├── config/
│   └── pbf4_cc_map.json        # ← NOT YET BUILT — discover CC numbers first
├── src/
│   ├── magenta_backend.py      # ✓ AIVoice, MagentaRTCFGTied, GenerationParams
│   └── modal_client.py         # ✓ MagentaRTClient — async parallel voice dispatch
├── modal/
│   ├── magenta_server.py       # ✓ VoiceServer — deployed on Modal A100
│   ├── generate_cascade.py     # ✓ offline batch generation (tested, works)
│   ├── benchmark_magenta.py    # ✓ A100 benchmark (RTF 1.431× confirmed)
│   └── benchmark_results.json  # ✓ benchmark data
├── colab/
│   └── poc_cascade.py          # ✓ proof of concept (Colab sequential cascade)
├── CLAUDE.md                   # this file
└── LESSONS.md                  # bugs and lessons — READ BEFORE CODING
```

**Remaining to build** (in order):
1. `src/audio_devices.py` — detect CABLE Output and output device indices
2. `src/loop_capture.py` — sounddevice InputStream, ring buffer, snapshot to numpy on command
3. `src/audio_mixer.py` — sounddevice OutputStream callback, mixes loop + voice queues
4. `config/pbf4_cc_map.json` — discover with hardware, do NOT guess
5. `src/midi_controller.py` — mido callback, CC→GenerationParams dispatch
6. `src/timing_engine.py` — absolute-time loop clock (time.perf_counter), pass boundary callbacks
7. `improv_loop.py` — wires everything: state machine + client + mixer + MIDI + timing

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
| 1 | PBF4 actual CC numbers? | **⚠ OPEN** — test hardware first |
| 2 | Beat-alignment preference? | **⚠ OPEN** — free improv (accept drift) vs beat-locked (trim to boundary)? |
| 3 | How many genres must be specified? | **⚠ OPEN** — 2 genres with 2 faders unused OK? |
| 4 | Analog Lab / Ableton for final version? | Open — Surge XT confirmed for prototype |

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

**Modal containers**: All server dependencies baked into the image at `modal deploy` time. See `modal/magenta_server.py` for the full image definition.
