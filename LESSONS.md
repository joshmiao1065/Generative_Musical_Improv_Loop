# LESSONS.md — Improv Loop Project

> Append an entry after every session. Format: Attempted / Result / Root cause / Correct approach / Watch for.

---

## Hardware

**MicroLab mk3 has NO knobs or faders.**
It is musical input only (25 keys, 2 touch strips, 4 octave/nav buttons). ALL system parameter control (guidance, temperature, topk, model_feedback, genre weights, record/stop) is handled exclusively by the **intech PBF4**. Never map system params to MicroLab CC numbers.

**Analog Lab Intro is plugin-only; use Surge XT standalone for prototype.**
Surge XT → output to "CABLE Input" (VB-Cable) → Python captures from "CABLE Output". No DAW needed. Set Surge XT output device before every session (it may reset to system default).

**VB-Cable: WASAPI shared mode only.**
ASIO with VB-Cable fails in python-sounddevice (GitHub issue #520, closed "not planned"). Use default sounddevice settings on "CABLE Output" InputStream. Never pass `extra_settings=WasapiSettings(exclusive=True)` — exclusive mode conflicts with other apps on the virtual device.

---

## Magenta RT Model

**Temperature and top-k ARE real Magenta RT parameters.**
Early research (README/paper) suggested they didn't exist. Source code (`generate_chunk()` kwargs) confirms: `guidance_weight`, `temperature`, `topk`, `model_feedback` are all real. Map all four to PBF4 knobs.

**AIVoice output is shorter than CHUNK_SAMPLES.**
`_AudioFade.__call__()` removes `fade_samples` from the right end for crossfading. Output shape is `(CHUNK_SAMPLES - fade_samples, 2)`. Always zero-pad back to CHUNK_SAMPLES with `pad_to_chunk()` before adding to the cascade cumulative mix.

**Model weights are shared across voices; only state is per-voice.**
`MagentaRTCFGTied` and `SpectroStreamJAX` each load once per container. `_InjectionState` and `MagentaRTState` are per-voice. On Modal: one container per voice IS the correct design because of GPU parallelism, not because each needs its own model copy.

**Audio injection mechanism (implemented in `src/magenta_backend.py`):**
Input audio accumulates in `all_inputs`. A window of recent input + recent output × `model_feedback` is encoded to SpectroStream tokens and injected into the model's context. Cascade input for Voice N = `user_loop + voice_0_out + ... + voice_{N-1}_out`. Colab UI (`colab_utils.AudioStreamer`) is not needed — core logic is `encode → inject → generate_chunk`.

---

## GPU / Modal Deployment

**Magenta RT requires three mandatory patches for GPU.**
Without `t5x_partitioning.py.patch`, the model crashes on GPU (calls TPU-only `bounds_from_last_device()`). All three patches are in `magenta-realtime/patch/`. T5X must be pinned to commit `7781d167` — patches fail on HEAD. Install order: patch `t5x/setup.py` → `pip install -e t5x` → `pip install jax[cuda12]` → `pip install magenta-realtime` → patch `t5x/partitioning.py` and `seqio/vocabularies.py`.

**A10G is 13% too slow for real-time; A100-80GB is confirmed viable.**
- A10G: RTF 0.873× (chunk takes 2.29s to generate 2.0s audio) — use for offline only
- A100-80GB: RTF 1.431× (1.40s per 2.0s chunk). Single voice = 5.68s to generate 8s loop = 0.71× ✓
- Threading on one GPU: 0.99× speedup — JAX serializes GPU calls. Useless.
- **Architecture locked**: 1 A100 per voice, 3 containers in parallel. Wall time ≈ 5.68s vs 8s loop. 2.32s headroom.

**`modal.parameter()` + `min_containers > 0` fails at deploy.**
Error: "Parameterized Functions cannot have `min_containers > 0`". Correct: omit `min_containers` entirely (defaults to 0). Use `scaledown_window=600` to keep containers alive 10 min after last call. Before a session, call `ping_all()` to warm all 3 containers (triggers model load, ~3 min first time, instant after).

**`buffer_containers=3` on a parameterized class burns the 10-GPU quota.**
`buffer_containers=N` keeps N warm containers per parameterized class regardless of demand. With 3 voices × `buffer_containers=3` = 9 idle GPUs + overhead = quota exhausted. Solution: use no `buffer_containers`, rely on `scaledown_window` to keep containers warm during a session.

**`modal.Cls.from_name()` — how to call deployed parameterized classes from the client:**
```python
VoiceServer = modal.Cls.from_name("magenta-rt-server", "VoiceServer")
voice = VoiceServer(voice_index=0)
result = await voice.generate_pass.remote.aio(...)  # async
result = voice.generate_pass.remote(...)            # sync
```
Only works after `modal deploy`. Raises `NotFoundError` if app is not deployed. Run `python src/modal_client.py` to verify before a session.

**Modal image build: do NOT use `t5x[gpu]` extra — it pulls fasttext which fails.**
`t5x[gpu]` triggers a C++17 fasttext build that breaks the image. Install `t5x` base (no extras) + `jax[cuda12]` separately. Magenta RT never uses fasttext.

**`t5x.__file__` is None with editable installs — use absolute path for patches.**
`pip install -e` sets `__file__ = None`. Don't try `python -c "import t5x; print(t5x.__file__)"` to find patch targets. Use the known clone path directly: `patch /t5x/t5x/partitioning.py`.

---

## Architecture Decisions (locked)

**Lyria RealTime API eliminated.** Text-only input — no audio injection. Magenta RT is the sole AI backend.

**Colab eliminated.** Session disconnects, no persistent URL. Replaced by Modal.com (persistent, callable, GPU-backed).

**One-pass lag is intentional.** In parallel dispatch, Voice N hears Voice N-1's *previous pass* output, not the current one. This is required by the buffer-pass architecture. Do not try to eliminate it.

**Monitoring via Python passthrough.** sounddevice output callback mixes VB-Cable capture + AI outputs → speakers. ~20–40ms latency. No Voicemeeter needed for prototype.

---

## Audio Devices (Session 7 — 2026-04-21)

**MIDI devices detected on Windows:** `['MicroLab mk3 0', 'Intech Grid MIDI device 1']`
Both are present and accessible via `mido` on Windows Python. WSL2 cannot see USB
MIDI/audio devices — always run these scripts with the Windows `.venv` Python.

**VB-Cable is NOT installed.** Not present in `sd.query_devices()` on 2026-04-21.

**Stereo Mix (device 13) IS present and capturing.** At 48kHz stereo, it matches the
pipeline format. RMS=0.0018, Peak=0.022 in a quick background test while Surge was
presumably idle/quiet. Works as a capture device BUT has the feedback problem (see
`docs/vb_cable_analysis.md`): Python AI output → speakers → Stereo Mix → Modal →
creates uncontrolled audio feedback loop. For initial testing only.

**Stereo Mix must be enabled in Windows Sound Settings** before use:
Sound Settings → Recording → Stereo Mix → right-click → Enable.

**VB-Cable recommendation: buy it ($5).** Stereo Mix works for testing (no feedback
if AI output is muted), but VB-Cable is required for production sessions. CABLE Output
(input side) will appear at 48kHz stereo, same format — minimal code change required.

**pyaudiowpatch is a free WASAPI loopback alternative.** Can open a specific output
device (Surge's output) as a loopback input, avoiding the feedback problem without
VB-Cable. More complex setup. Not tested yet.

**CC discovery: use `scripts/discover_cc.py` on Windows Python.** Run in a terminal,
move every PBF4 control, Ctrl+C. Results → `config/pbf4_cc_map.json`. WSL Python
cannot open ALSA sequencer — must use Windows venv.

---

## Audio Output Bugs Fixed (Session 8 — 2026-04-26)

**Voice enable/disable was inverted — the most likely cause of "no AI audio."**
`AudioMixer._voice_enabled` was initialised `[True, True, True]`, but the PBF4
toggle state starts `False`. The first button press was a no-op (set True, already
True). The second press disabled the voice. Voices actually auto-played without any
button press (after pass 2), which is why it "worked without knowing how."
Fix: initialise `_voice_enabled = [False, False, False]`. First button press now
correctly enables; second disables.

**Voice audio was silently dropped on length mismatch (non-120 BPM / 16-beat sessions).**
`_audio_callback` had `if va.shape[0] != loop_len: continue` with no log. This
silently suppressed AI voices at any BPM/beat count where the generated audio length
(= `chunks_per_pass × CHUNK_SAMPLES`) didn't exactly equal the user loop length.
At 120 BPM / 16 beats both are 384,000 samples so it worked; at 90 BPM / 8 beats
the loop is 256,000 but generated audio is 288,000 — silently dropped.
Fix: trim or zero-pad incoming audio to loop_len inside `_on_boundary()` before
assigning to `_voice_audio[i]`. The length check in `_audio_callback` now logs a
warning instead of silently continuing.

**Analog Lab Intro has a standalone application mode.** Previous LESSONS entry
said "plugin-only." This is incorrect. Analog Lab Intro ships as a standalone app
and as a VST/AU plugin. Standalone mode: open Analog Lab, set Audio Output → CABLE
Input, set MIDI Input → MicroLab mk3. No DAW needed. Python code is unchanged —
it captures from whatever is routed to CABLE Output.

**`config/pbf4_cc_map.json` has empty `cc_controls`.** The knobs and faders were
never moved during `discover_cc.py` — only buttons were pressed. CC numbers in
`pbf4_layout.json` (32–39) were entered manually. Run `scripts/discover_cc.py` again
on Windows Python, move every knob and fader slowly through full range, then verify
the discovered CC numbers match the layout file.

**Debugging flags added to `improv_loop.py`.** Key flags for testing without hardware:
- `--dry-run`: skip Modal entirely, AI voices = sine tones (440/554/659 Hz)
- `--dry-run-latency 5.7`: simulate A100 timing to test buffer headroom
- `--list-devices`: list all audio and MIDI devices then exit
- `--capture-device "CABLE Output"`: force VB-Cable after installation
- `--capture-device "Stereo Mix"`: force Stereo Mix explicitly
- `--save-loops ./debug_loops`: save each captured loop to WAV
- `--no-click`: silent count-in
- `--log-level debug`: full trace logging
