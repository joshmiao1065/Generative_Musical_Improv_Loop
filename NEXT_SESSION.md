# NEXT_SESSION.md — Session 10 Summary & Session 11 Checklist

> **Last updated**: 2026-04-29 (Session 10)
> **Developer**: Josh Miao
> Read `CLAUDE.md` and `LESSONS.md` before coding anything.

---

## What Was Accomplished This Session (Session 10)

### Performance Fixes (Critical)
| Fix | Root Cause | Impact |
|-----|-----------|--------|
| `librosa.load` → `sf.read` (server + client) | librosa 1–2s per WAV decode vs soundfile ~10ms | 24–38s wall time → expected ~5–7s |
| `_generation_in_flight` flag | Passes queued up 7–11 deep when generation > loop duration | Prevents unbounded lag |

### New Features
| Feature | Files | Details |
|---------|-------|---------|
| Crossfader knob (Knob 3) | `audio_mixer.py`, `midi_controller.py`, `keyboard_controller.py`, `config/pbf4_layout.json`, `improv_loop.py` | Replaces topk. DJ-style: 0=you, 0.5=both full, 1=AI. `_crossfade_ai` multiplier is independent of per-voice `_voice_volume` |
| Memory leak fix | `src/magenta_backend.py` | `all_inputs`/`all_outputs` now capped to needed window size; constant memory after warmup |

### Bug Fixes
| Bug | Fix |
|-----|-----|
| Genre blending crash (`e.embedding` on numpy array) | `float(w) * e` directly; no StyleEmbedding wrapper |
| Modal generation exceptions silently swallowed | `logger.exception` now prints full traceback |
| `poc_cascade.py` referenced in deleted file | Docstring updated |

### Tests
22 tests passing (`python scripts/test_audio_logic.py`), including 5 new crossfader tests.

### Server Status
`modal deploy server/magenta_server.py` run at end of session — server is deployed with all fixes.
App: https://modal.com/apps/joshuamiao03/main/deployed/magenta-rt-server

---

## Session 11 — Test Checklist

Run these in strict order. Earlier steps are prerequisites for later ones.

### Step 0 — Pre-flight (no Modal cost, no hardware required)
```bat
REM Confirm unit tests still pass:
python scripts/test_audio_logic.py

REM List devices — confirm VB-Cable at [2] and PBF4 visible as MIDI:
.venv\Scripts\python improv_loop.py --list-devices

REM Dry-run with QWERTY (no PBF4, no Modal):
.venv\Scripts\python improv_loop.py --dry-run --qwerty
REM → play a few notes in Surge XT, press Space to record, hear loop + sine tones
REM → press k several times, confirm crossfade log output: "crossfade = 0.60"
REM → press K to bring it back toward center
```

### Step 1 — PBF4 CC Discovery (CRITICAL — do once on Windows, not WSL)
```bat
.venv\Scripts\python scripts\discover_cc.py
REM Move EVERY knob slowly through full range (0→127→0)
REM Move EVERY fader slowly through full range
REM Press every button
REM Ctrl+C → check config\pbf4_cc_map.json → cc_controls must be non-empty
```
Compare discovered CC numbers to `config/pbf4_layout.json` (currently guesses: knobs CC 32-35, faders CC 36-39).
Update `pbf4_layout.json` if they differ.

Validate live readout:
```bat
.venv\Scripts\python scripts\validate_pbf4.py
REM Move each knob, confirm log shows: guidance_weight/temperature/crossfade/model_feedback updating
REM Move each fader, confirm genre_0/1/2/3 updating
```

### Step 2 — Dry-run with PBF4 Hardware
```bat
.venv\Scripts\python improv_loop.py --dry-run --capture-device "CABLE Output"
```
- [ ] Press Button 1 → hear 1-bar metronome click count-in → recording begins
- [ ] Play Surge XT → loop is captured (check log: `RMS=0.xxxx` > 0.001)
- [ ] Loop plays back through speakers (sine tones + your loop)
- [ ] Press Buttons 2/3/4 → logs show `Voice 1/2/3: ON`; sine tones become audible
- [ ] Turn Knob 3 clockwise → crossfade increases (more AI) in logs
- [ ] Turn Knob 3 counterclockwise → crossfade decreases (more you)
- [ ] Turn Knob 1 → `guidance_weight` updates in logs
- [ ] Turn Knob 2 → `temperature` updates
- [ ] Turn Knob 4 → `model_feedback` updates
- [ ] Move Faders 1-4 → `genre_0/1/2/3` weights update in logs
- [ ] Press Button 1 while playing → re-records without silence gap
- [ ] Press Ctrl+C → clean shutdown

### Step 3 — Modal Integration (costs ~$0.05-0.10 per test)
```bat
REM Server already deployed — skip deploy unless you changed server code
REM Ping containers to confirm warm (takes up to 10 min cold):
modal logs magenta-rt-server    REM watch in separate terminal
.venv\Scripts\python scripts\prime_server.py --voices 1
REM Wait for: [Voice 0] READY in Xs
```

**1-voice test first:**
```bat
.venv\Scripts\python improv_loop.py --voices 1 --bpm 120 --beats 8 --capture-device "CABLE Output"
```
- [ ] Wait for `ALL VOICES READY — Press Button 1 to record`
- [ ] Record a loop (~4s at 120 BPM / 8 beats)
- [ ] Watch generation log: `Pass 1 started — dispatching generation`
- [ ] **Measure wall time**: log shows `[MagentaRTClient] Pass complete: X.XXs wall time (X.XXx loop duration)` — should be < 1.0× now with soundfile fix
- [ ] At pass 2, Voice 1 audio queued: log shows `Pass 2: Voice 1 queued (4.00s, enabled=False)`
- [ ] Press Button 2 → Voice 1 enabled → AI joins the mix at next loop boundary
- [ ] Confirm AI audio is musically coherent (not noise/silence)
- [ ] Press Button 1 → re-records while AI voice continues uninterrupted
- [ ] `modal app stop magenta-rt-server` when done

### Step 4 — Genre Blending Test (only after Step 3 passes)
```bat
.venv\Scripts\python improv_loop.py --voices 1 --genres "jazz" "electronic" --instrument piano --bpm 120 --beats 8
```
- [ ] Fader 1 full up, Fader 2 down → log: `style = jazz(100%) piano`
- [ ] Fader 1 half, Fader 2 half → log: `style = jazz(50%) + electronic(50%) piano`
- [ ] Fader 2 full up, Fader 1 down → log: `style = electronic(100%) piano`
- [ ] **Critical**: confirm no crash at the blending step. Previous crash was `'numpy.ndarray' object has no attribute 'embedding'` — fixed, but untested on Modal.
- [ ] Listen for timbral difference between jazz and electronic settings

### Step 5 — Full 3-Voice Session
```bat
.venv\Scripts\python improv_loop.py --bpm 120 --beats 16 --capture-device "CABLE Output"
```
- [ ] All 3 containers warm before recording
- [ ] Record 16-beat loop
- [ ] Voice 0 audio ready at pass 2, Voice 1 at pass 3, Voice 2 at pass 4
- [ ] Enable each voice with Buttons 2/3/4 as they arrive
- [ ] All 3 playing simultaneously — check for audio coherence and no clipping
- [ ] Crossfade knob (Knob 3) brings AI mix in/out smoothly
- [ ] `modal app stop magenta-rt-server` after session

---

## Expected Generation Timing (Post-Soundfile Fix)

At 120 BPM / 16 beats (8.0s loop):
- Server decode: ~20ms (was 1–2s with librosa × 2 = 2–4s saved)
- Model inference: ~5.6s on A100-40GB (extrapolated from A100-80GB benchmark)
- Total expected wall time: **~5.7–6.0s** = **0.71–0.75× loop duration**
- Headroom before loop ends: **~2.0–2.3s** ✓

If wall time is still > 1.0×:
1. Check Modal logs for XLA recompilation (should not happen after first pass)
2. Check if containers are on A100-40GB (not A10G) — `modal logs magenta-rt-server`
3. Reduce beats to 8 (4.0s loop) — gives containers 4s for a 5.7s job but more passes

---

## Known Issues / Watch For

### Crossfader startup jump
PBF4 does not send CC until a knob is physically moved. On startup, `_crossfade_ai = 1.0`
(AI at full). When the user first touches Knob 3, the mixer jumps to the physical position.
**This is expected behavior** — just start the knob at center position before launching.

### Genre blending `tokenize` compatibility (untested)
`embed_style()` returns a raw numpy array. The blended result is also a raw array. It's
assigned to `self.voice.style_embedding` and passed to `generate_chunk(style=...)`. Inside
`generate_chunk`, `self.style_model.tokenize(style)` is called. If this method requires a
`StyleEmbedding` object, blending will fail at this line. The non-blending path works (passes
raw array from `embed_style()` directly). If blending crashes, the fix is:
```python
# In server/magenta_server.py _impl_generate_pass, after computing blended:
from magenta_rt import musiccoca
blended = musiccoca.StyleEmbedding(embedding=blended)
```

### First pass latency (expected)
The very first generation pass per container will be slightly slower because:
1. `embed_style()` is called fresh (not yet cached) — ~200ms
2. Python warmup after `@modal.enter()` — minor

Subsequent passes use cached embeddings and will be at full speed.

---

## Open Questions

| # | Question | Status |
|---|----------|--------|
| 1 | PBF4 CC numbers for knobs/faders confirmed? | **⚠ OPEN** — run discover_cc.py (Step 1) |
| 2 | Genre blending `tokenize` with raw array? | **⚠ OPEN** — needs real Modal test (Step 4) |
| 3 | A100-40GB wall time post soundfile fix? | **⚠ OPEN** — measure in Step 3 |
| 4 | Beat-alignment: free improv vs. beat-locked? | **⚠ OPEN** — current: `align_to_beat=True` in `_finish_recording` |
| 5 | Analog Lab routing? | **⚠ OPEN** — confirmed in docs, not yet tested live |

---

## Quick Reference

```bat
REM Unit tests (WSL or Windows — no hardware needed):
python scripts/test_audio_logic.py

REM Dry-run with QWERTY (no hardware):
.venv\Scripts\python improv_loop.py --dry-run --qwerty

REM Dry-run with PBF4:
.venv\Scripts\python improv_loop.py --dry-run --capture-device "CABLE Output"

REM 1-voice Modal test (cheapest — ~$0.05):
.venv\Scripts\python improv_loop.py --voices 1 --bpm 120 --beats 8 --capture-device "CABLE Output"

REM Full session:
.venv\Scripts\python improv_loop.py --bpm 120 --beats 16 --capture-device "CABLE Output"

REM Stop billing:
modal app stop magenta-rt-server
```

**IMPORTANT**: Always use `.venv\Scripts\python` (Windows Python). WSL2 cannot see USB devices.
