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

**Analog Lab Intro has a standalone application mode** (corrects earlier entry saying "plugin-only").
Standalone mode: open Analog Lab → hamburger menu → Settings → Audio/MIDI → set Audio Device Type: WASAPI, Output Device: "CABLE Input (VB-Audio Virtual Cable)", Sample Rate: 48000, Buffer Size: 512. Set MIDI Input: "MicroLab mk3". No DAW needed. If Analog Lab only shows ASIO devices, install ASIO4ALL or use the plugin version in a DAW. Python code is unchanged — it always captures from CABLE Output regardless of which synth is upstream.

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

## Audio Devices

**MIDI devices detected on Windows (confirmed 2026-04-21):** `['MicroLab mk3 0', 'Intech Grid MIDI device 1']`
Both are accessible via `mido` on Windows Python. WSL2 cannot see USB MIDI/audio
devices — always run MIDI/audio scripts with the Windows `.venv` Python.

**VB-Cable installed (confirmed 2026-04-26).** Detected as device [2] "CABLE Output
(VB-Audio Virtual Cable)" at 48kHz stereo. Auto-detection in `audio_devices.py`
prefers it over Stereo Mix. Use `--capture-device "CABLE Output"` to be explicit.
Stereo Mix remains present (device 13) as a fallback.

**VB-Cable signal chain:**
```
Surge XT / Analog Lab → audio out → "CABLE Input" (set in synth audio settings)
    → VB-Cable kernel driver
    → "CABLE Output" → Python InputStream (device [2], 48kHz stereo float32)
    → LoopCapture ring buffer → Modal → AI audio
    → Python OutputStream → Speakers (device [15], Realtek)
```
Python's speaker output is completely isolated from CABLE Output. There is no
feedback path. This is why VB-Cable is required for production sessions.

**VB-Cable sample rate must be 48kHz in Windows Sound Settings — one-time setup.**
Right-click speaker icon → Sound Settings → More sound settings:
- Recording tab → CABLE Output → Properties → Advanced → 2ch 24-bit 48000 Hz
- Playback tab → CABLE Input → Properties → Advanced → 2ch 24-bit 48000 Hz
If left at 44.1kHz (Windows default), sounddevice silently resamples → pitch shift
in the captured audio. The loop will sound correct but AI generation will be misaligned.

**Surge XT forgets its audio output device between sessions.** Check Preferences →
Audio → Output Device = "CABLE Input" at the start of every session. It often resets
to the system default speakers on close.

**Stereo Mix feedback loop (without VB-Cable):** Python AI output → Speakers →
Stereo Mix → LoopCapture ring buffer → user_loop sent to Modal → model generates
against its own previous output. Quality degrades each pass. For testing without
VB-Cable: mute Python output or use `--dry-run` (which doesn't send to Modal anyway).

**CC discovery: use `scripts/discover_cc.py` on Windows Python.** Run in a terminal,
move every PBF4 control, Ctrl+C. Results → `config/pbf4_cc_map.json`. WSL Python
cannot open ALSA sequencer — must use Windows venv.

**`config/pbf4_cc_map.json` has empty `cc_controls` as of 2026-04-26.** Knobs and
faders were never moved during discovery — only buttons were pressed. CC numbers
in `pbf4_layout.json` (CC 32-35 for knobs, CC 36-39 for faders) are unverified
guesses. If knobs/faders appear to do nothing, this is why. Re-run `discover_cc.py`
with all controls moved before relying on them.

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

---

## VB-Cable Debugging Reference (Session 8 — 2026-04-26)

These are the definitive failure modes and fixes for the VB-Cable capture chain.
Check these before assuming a bug in Python code.

**Symptom: loop WAV saved via `--save-loops` is silent (flat line)**
Cause checklist:
1. Surge XT / Analog Lab audio output is still pointing at system speakers, not "CABLE Input"
2. VB-Cable sample rate mismatch — CABLE Output set to 44.1kHz in Windows Sound Settings
3. No synth is running / no MIDI notes playing — silence is correct behaviour
4. CABLE Output is set as Default Communication Device — some apps reroute it
Fix: confirm Surge XT → Preferences → Audio → Output = "CABLE Input". Run
`src/loop_capture.py` standalone to capture 4s and check RMS independently.

**Symptom: captured audio has wrong pitch (chipmunk = too fast, slow-mo = too slow)**
Cause: VB-Cable device sample rate in Windows Sound Settings doesn't match the 48kHz
Python opens the InputStream at. The driver resamples transparently but wrongly.
Fix: Sound Settings → Recording → CABLE Output → Properties → Advanced → set to
"2 channel, 24 bit, 48000 Hz (Studio Quality)". Do the same for CABLE Input.

**Symptom: you hear your synth directly but Python loop playback is silent**
Cause: Surge XT is outputting to speakers AND CABLE Input simultaneously, or Python's
OutputStream opened on the wrong device.
Check: `--list-devices` output → confirm playback device is Speakers [15], not CABLE.
Note: `_find_playback()` deliberately avoids selecting CABLE Input as playback.

**Symptom: you hear the metronome click but not your own playing in the loop**
Cause: capture is returning silence (see above) but the mixer is working. The click
fires through `play_oneshot()` independent of loop audio.
Fix: same as "loop WAV is silent" above.

**Symptom: AI sine tones (dry-run) audible on pass 2+, but no user audio in the mix**
Cause: VB-Cable capture is working but the synth isn't routed to it. The ring buffer
fills with silence; `mixer.set_loop(silence_array)` plays nothing. AI tones play
because they're generated independently from the (silent) user_loop.
Fix: route synth to CABLE Input.

**Symptom: echo / doubling of the synth**
Cause: synth is outputting to both CABLE Input (captured by Python) AND speakers
(direct monitoring). Python then plays the loop back through speakers. Two copies.
Fix: in Surge XT, set output to CABLE Input only. Use Python as the sole monitoring
path. There will be ~10–20ms latency from Python's output blocksize (512/48000).

**Symptom: AI voices don't play even after pressing Button 2**
When VB-Cable and PBF4 are both present: use `--log-level debug`. Confirm you see:
```
[Session] Pass 2: Voice 1 queued (8.00s, enabled=True)
[AudioMixer] Voice 0 swapped in at boundary
```
If `enabled=False` appears: the button press is not reaching `set_voice_enabled`.
Check PBF4 is connected and note_number 41 matches Button 2 in `pbf4_layout.json`.
If no "queued" log appears: generation is failing silently — check Modal deploy status.

**Symptom: `--list-devices` shows VB-Cable at wrong sample rate**
The `Rate` column in `--list-devices` output shows the device's default_samplerate
as reported by the driver. If it shows 44100, fix it in Windows Sound Settings before
starting a session. Python will try to open at 48000 and may fail or resample.

**Confirmed working configuration (2026-04-26):**
- VB-Cable: device [2], 48kHz stereo, WASAPI shared mode
- Speakers (Realtek): device [15], 48kHz stereo, WASAPI
- No MIDI devices present during this test (PBF4 / MicroLab not connected)
- `--dry-run --capture-device "CABLE Output"` launched successfully
- Session flow confirmed: devices detected → layout loaded → streams started → ready

---

## Live Monitoring Passthrough (Session 9 — 2026-04-28)

**"Read last N frames" approach produces wrong pitch and timbre.**
First monitoring implementation called `_read_last(n)` from LoopCapture's ring buffer in
AudioMixer's output callback. This worked when input and output blocksizes matched, but
LoopCapture defaults to blocksize=2048 and AudioMixer uses blocksize=512. With a 4:1 ratio,
`get_monitor_frames(512)` returned the same 512 frames 4× per input block before new audio
arrived. The output played each 512-sample chunk 4 times at normal sample rate, which is
equivalent to slowing the audio to 1/4 speed then looping it — every keyboard note sounded
the same pitch and had no synth timbre.
Fix: replaced with `_MonitorFIFO` (in `src/loop_capture.py`). Input callback writes all
captured audio into the FIFO with `write(indata)`. Output callback drains exactly `frames`
from the FIFO with `read(frames)` every tick. Each sample is consumed exactly once, in order,
regardless of blocksize mismatch. The FIFO's capacity is 4096 frames (~85ms) so it absorbs
any scheduling jitter between input and output PortAudio threads.
**Do not revert to ring-buffer reads for monitoring.** The FIFO is the correct architecture.
Confirmed working: Surge XT timbre and pitch pass through correctly at ~10–20ms latency.

---

## Modal Architecture — Separate Classes Required (Session 9 — 2026-04-28)

**`modal.parameter()` + `min_containers > 0` is unsupported. Use 3 separate named classes.**
Previous sessions used `VoiceServer` with `voice_index: int = modal.parameter()` which
requires lazy loading (min_containers=0). Containers only spun up on first call, causing
perpetually-pending tasks if the GPU wasn't immediately available. Attempts to add
`min_containers=1` to the parameterized class failed with Modal's own validation error.
Fix: replaced with `Voice0Server`, `Voice1Server`, `Voice2Server` — three explicit classes,
each with `VOICE_INDEX` as a class attribute and `min_containers=1`. The server still has
only one copy of the logic (in `_impl_*` module-level functions). Each class is a thin
wrapper that delegates to these. The client (`src/modal_client.py`) looks up each class by
name: `modal.Cls.from_name(APP_NAME, "Voice0Server")()` etc.
**Do not re-introduce `modal.parameter()`.** The three-class approach is the locked design.

**A100-40GB confirmed available (Session 9).** A100-80GB was also available but burned
6-10 GPU slots due to `buffer_containers=3` bug in an earlier session (see GPU section).
After switching to `GPU_TYPE = "A100-40GB"`, 3 containers appeared on the Modal dashboard
within ~1 minute of `modal deploy`. GPU type string in Modal is case-insensitive; `"A100-40GB"`
and `"a100-40gb"` both work.

**"Active" container ≠ model ready. Cold boot takes 5–8 min.**
When Modal shows a container as "active," the GPU is allocated and Python has started, but
`@modal.enter()` (which loads SpectroStream + MagentaRT weights + JIT-compiles XLA kernels)
is still running. Any `.remote()` method calls during this window are queued/pending — they
will NOT appear as "running" on the Modal dashboard until `@modal.enter()` completes.
Observed symptom: `prime_server.py` calls `.ping.remote.aio()`, but the call doesn't appear
on the dashboard and never returns (pending). This is NORMAL for the first 5–8 min after
deploy. Watch the container logs for `[Voice N] *** READY ***` — only after that will pings
return. To stream logs: `modal logs magenta-rt-server`.
