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
