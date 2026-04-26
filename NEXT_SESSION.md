# NEXT_SESSION.md — Session 8 Summary & Session 9 Checklist

> **Last updated**: 2026-04-26 (Session 8)
> **Developer**: Josh Miao
> **For**: The next Claude Code agent AND Josh as a reminder of what was accomplished

Read this file AND `CLAUDE.md` AND `LESSONS.md` before writing any code.

---

## What Was Accomplished This Session (Session 8)

### Bugs Fixed
| Bug | Root Cause | Fix Location |
|-----|-----------|--------------|
| AI voices played even when toggled OFF | `AudioMixer._voice_enabled` defaulted to `[True,True,True]` | `src/audio_mixer.py` |
| Toggling voice ON was actually toggling it OFF | Same as above — buttons were inverting | `src/audio_mixer.py` |
| Shape mismatch crash (silent drop) | Strict `==` check, no trim/pad | `src/audio_mixer.py` |
| Ctrl+C / Ctrl+Z did not stop `improv_loop.py` | `_stop_event.wait()` (no timeout) blocks inside OS `WaitForSingleObject`, never yields to Python signal handler | `improv_loop.py` — replaced with `while not _stop_event.wait(timeout=0.1)` in `try/except KeyboardInterrupt` |
| `_stop_loop()` crashed in dry-run mode | `self.client.reset()` called when `self.client is None` | `improv_loop.py` — added `if self.client is not None` guard |

### Features Added
| Feature | Files Changed |
|---------|--------------|
| **QWERTY keyboard fallback** (`--qwerty`) | `src/keyboard_controller.py` (new), `improv_loop.py` |
| **Genre blending end-to-end** | `improv_loop.py`, `src/modal_client.py`, `server/magenta_server.py` |
| **`--genres` and `--instrument` CLI args** | `improv_loop.py` |
| **Debug flags**: `--dry-run`, `--dry-run-latency`, `--list-devices`, `--capture-device`, `--playback-device`, `--capture-idx`, `--playback-idx`, `--save-loops`, `--no-click`, `--log-level` | `improv_loop.py` |
| **Modular synth routing** | Documented in `CLAUDE.md` — no code changes needed to switch synths, only hardware routing |
| **VB-Cable integration** | `src/audio_devices.py`, `improv_loop.py` (`--capture-device "CABLE Output"`) |
| **`quit` event** in QwertyController | `src/keyboard_controller.py` — `q` key → fires quit callback → `_stop_event.set()` |
| Unit test suite (17 tests) | `scripts/test_audio_logic.py` |

### VB-Cable Status
**CONFIRMED WORKING.** Device index [2], name "CABLE Output", 48kHz.
Josh purchased and installed VB-Cable during Session 8.
```
python improv_loop.py --dry-run --capture-device "CABLE Output"
```
Surge XT confirmed routed to CABLE Input. Analog Lab routing documented but not yet tested (standalone mode → Audio → Output: "CABLE Input").

---

## Current Architecture State

```
[MicroLab mk3] ─MIDI─▶ [ThinkPad]          (NOT tested this session — not available)
[intech PBF4]  ─MIDI─▶ [ThinkPad]          (NOT tested this session — not available)
[Surge XT]     ─audio─▶ CABLE Input ─▶ CABLE Output ─▶ Python capture   ✓ CONFIRMED
[ThinkPad]     ─HTTP──▶ Modal A100 × 3     (deployed, NOT tested this session)
[ThinkPad]     ─audio─▶ speakers           ✓ CONFIRMED (dry-run sine tones audible)
```

---

## Session 9 Hardware Test Checklist

Run these in order when PBF4 and MicroLab are available again.

### Pre-Flight (run first, no Modal credits spent)
- [ ] `python improv_loop.py --list-devices` — confirm PBF4 visible as MIDI device
- [ ] `python improv_loop.py --list-devices` — confirm VB-Cable at index [2]
- [ ] `python improv_loop.py --dry-run --capture-device "CABLE Output"` — play Surge XT and confirm loop capture (check RMS > 0.001)
- [ ] Press PBF4 Button 1 → confirm 2-bar metronome count-in
- [ ] Press PBF4 Button 2/3/4 → confirm voices toggle ON/OFF in logs
- [ ] Press PBF4 Button 1 again while playing → confirm seamless re-record
- [ ] Press Ctrl+C → confirm clean shutdown (the bug was fixed this session)

### CC Map Verification (CRITICAL — do this before knob/fader testing)
- [ ] Run CC discovery (Windows cmd, NOT WSL):
  ```bat
  .venv\Scripts\python scripts\discover_cc.py
  ```
  Move EVERY knob and fader through full range. Press every button.
  → Check `config/pbf4_cc_map.json` — `cc_controls` must now be non-empty.
- [ ] Verify CC numbers match `config/pbf4_layout.json` (knobs CC 32-35, faders CC 36-39 are guesses — update if wrong)
- [ ] `python scripts/validate_pbf4.py` — confirm live param readout as knobs/faders moved

### Modal Integration Test (costs credits — ~$0.10-0.30 per test run)
- [ ] `modal deploy server/magenta_server.py` (already deployed — only if server changed)
- [ ] `python src/modal_client.py` — confirm all 3 containers respond to ping
- [ ] `python improv_loop.py --voices 1 --bpm 120 --beats 8 --capture-device "CABLE Output"` — 1-voice test
  - Wait for Voice 1 container warm (~5 min cold start, instant if warm)
  - Press Button 1, play something, let loop run
  - At pass 2, AI Voice 1 should be audible in the output mix
  - Press Button 2 to toggle Voice 1 off — confirm it goes silent
  - Press Button 2 again to toggle ON — confirm it comes back
- [ ] `modal app stop magenta-rt-server` — stop billing after test

### Genre Blending Test (after Modal test passes)
- [ ] `python improv_loop.py --genres "jazz" "blues" --instrument piano --voices 1`
  - Move PBF4 Fader 1 fully up, Fader 2 fully down → confirm logs say `style = jazz(100%) piano`
  - Move Fader 1 to 50%, Fader 2 to 50% → confirm logs say `style = jazz(50%) + blues(50%) piano`
  - Listen for timbral change between the two settings
- [ ] Test `--qwerty` mode with genre blend: `python improv_loop.py --dry-run --qwerty --genres "jazz" "blues"`
  - Press `]` key to increase genre 0 weight → confirm log update
  - Press `'` key to increase genre 1 weight

### Full Session Flow Test
- [ ] All 3 voices enabled, 16 beats, 120 BPM
- [ ] Verify staggered voice entry: Voice 1 audible at pass 2, Voice 2 at pass 3, Voice 3 at pass 4
- [ ] Verify knobs affect generation params in real time (move Knob 1, check logs)
- [ ] Re-record while AI voices continue: press Button 1 while PLAYING → confirm seamless handoff

---

## Open Issues / Unresolved Questions

| # | Question | Status |
|---|----------|--------|
| 1 | PBF4 actual CC numbers for knobs/faders? | **⚠ OPEN** — `cc_controls` in `pbf4_cc_map.json` is empty; run `discover_cc.py` and move all knobs/faders |
| 2 | Analog Lab (standalone) routing confirmed? | **⚠ OPEN** — documented (Settings → Audio → Output: "CABLE Input") but not yet tested |
| 3 | Modal containers warm and working? | **⚠ OPEN** — deployed but no real generation test this session (PBF4 not available) |
| 4 | Genre blending audibly correct? | **⚠ OPEN** — code is complete end-to-end; needs a real Modal call to verify |
| 5 | Beat-alignment preference? | **⚠ OPEN** — free improv (accept drift) vs beat-locked (trim to boundary)? |
| 6 | Genre 3 weight key (M/m) ergonomic? | **⚠ OPEN** — may want to remap in `src/keyboard_controller.py` `_GENRE_ADJUSTMENTS` |

---

## Key File Locations (for next agent)

| File | Purpose |
|------|---------|
| `improv_loop.py` | Main orchestrator — start here |
| `src/keyboard_controller.py` | QWERTY fallback (NEW this session) |
| `src/midi_controller.py` | PBF4 MIDI controller |
| `src/audio_mixer.py` | Voice mixing, enable/disable, shape correction |
| `src/modal_client.py` | Async client for Modal — sends genre args |
| `server/magenta_server.py` | Deployed Modal server — does the actual blending |
| `config/pbf4_layout.json` | CC map (knob/fader CCs are GUESSES — verify!) |
| `config/pbf4_cc_map.json` | Auto-generated by `discover_cc.py` — `cc_controls` currently EMPTY |
| `scripts/discover_cc.py` | Run on Windows to discover actual CC numbers |
| `scripts/test_audio_logic.py` | 17 unit tests (run in WSL, no hardware needed) |
| `LESSONS.md` | Bugs and lessons — ALWAYS READ before coding |
| `CLAUDE.md` | Full project context — ALWAYS READ before coding |

---

## How to Run Next Session (quick reference)

```bat
REM 1. Discover PBF4 CC numbers (Windows cmd, FIRST TIME only):
.venv\Scripts\python scripts\discover_cc.py

REM 2. Dry-run to confirm audio pipeline (no Modal cost):
.venv\Scripts\python improv_loop.py --dry-run --capture-device "CABLE Output"

REM 3. Real session (1 voice to save credits during testing):
modal deploy server/magenta_server.py      REM only if server code changed
.venv\Scripts\python improv_loop.py --voices 1 --bpm 120 --beats 8 --capture-device "CABLE Output"

REM 4. Stop Modal billing after session:
modal app stop magenta-rt-server

REM 5. Full 3-voice session when everything is confirmed:
.venv\Scripts\python improv_loop.py --bpm 120 --beats 16 --capture-device "CABLE Output"
```

**IMPORTANT**: Always use `.venv\Scripts\python` (Windows Python), NOT `python` or WSL Python.
USB devices (PBF4, VB-Cable) are not visible from WSL2.

---

## Architecture Notes for Next Agent

### Genre Blending (new this session)
- CLI: `--genres "jazz" "blues" "electronic" "ambient" --instrument piano`
- PBF4 Faders 1-4 → `genre_weights[0..3]` (0.0–1.0 each, independent)
- QWERTY: `]/[` → genre 0 ±0.1, `'/ ;` → genre 1, `./,` → genre 2, `M/m` → genre 3
- Flow: `improv_loop._generate_pass()` → `modal_client.generate_pass(genres, instrument, genre_weights)` → `VoiceServer.generate_pass()` on Modal
- Server blends: `voice.style_embedding = blend(embed("{genre} {instrument}") for each active genre)` using linear interpolation in embedding space
- Embeddings cached in `VoiceServer._style_cache` (dict keyed by "{genre} {instrument}") — computed once per container lifetime

### Ctrl+C Fix
- Old code: `signal.signal(SIGINT, handler) + _stop_event.wait()` — the bare `wait()` blocks inside `WaitForSingleObject`, never yields
- New code: `while not _stop_event.wait(timeout=0.1)` inside `try/except KeyboardInterrupt` — releases GIL every 100ms
- `q` key in QWERTY mode fires `quit` callback → `_stop_event.set()` → same shutdown path

### QWERTY Controller
- `src/keyboard_controller.py` — `QwertyController` class
- Same interface as `PBF4Controller`: `on()`, `get_toggle()`, `get_genre_weights()`, `start()`, `stop()`
- Extra event: `"quit"` (fired by q key; not in PBF4Controller)
- Windows: `msvcrt.kbhit()` + `msvcrt.getwch()` — no Enter needed
- Unix/WSL: `tty.setraw()` + `select.select()` — terminal restored on `stop()`
