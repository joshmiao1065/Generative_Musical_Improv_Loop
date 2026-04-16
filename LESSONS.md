# LESSONS.md — Improv Loop Project

> **Purpose**: Document every mistake, surprise, and hard-won lesson during this project so future LLM sessions do not repeat the same errors.
> **Rule**: Append an entry after every implementation session or significant discovery.
> **Format**:
> ```
> ## YYYY-MM-DD — [Topic]
> **Attempted**: What was tried.
> **Result**: What happened.
> **Root cause**: Why it happened.
> **Correct approach**: What actually works.
> **Watch for**: What to avoid going forward.
> ```

---

## 2026-04-15 — Magenta RT is a Library, Not a Cloud API

**Attempted**: N/A (documented pre-emptively from research)
**Result**: N/A
**Root cause**: Magenta RT (github.com/magenta/magenta-realtime) is an open-source Python library designed to run on TPU or high-VRAM GPU. It has no hosted endpoint.
**Correct approach**: Wrap Magenta RT in a FastAPI server deployed on Google Colab free TPU, exposed via pyngrok tunnel. The ThinkPad Python script calls this HTTP endpoint.
**Watch for**: Never write code that makes HTTP requests to a "Magenta RT endpoint" without first confirming the FastAPI server is running and the ngrok URL is set in config.

---

## 2026-04-15 — Temperature and Top-K Are Not Magenta RT / Lyria Parameters

**Attempted**: N/A (documented pre-emptively)
**Result**: N/A
**Root cause**: Temperature and top-k are LLM token-sampling parameters. Magenta RT uses a different architecture (autoregressive audio transformer) with different exposed controls.
**Correct approach**: Map PBF4 knobs to the REAL parameters: `guidance` [0–6], `density` [0–1], `brightness` [0–1], `bpm` [60–200].
**Watch for**: Do not include temperature or top-k anywhere in API calls, UI, or documentation.

---

## 2026-04-15 — MicroLab mk3 Has No Knobs or Faders

**Attempted**: N/A (documented pre-emptively)
**Result**: N/A
**Root cause**: The Arturia MicroLab mk3 is a minimal 25-key MIDI controller with only 2 touch strips and 4 buttons. It has NO rotary encoders, NO faders.
**Correct approach**: MicroLab mk3 = musical performance input ONLY (plays notes into Analog Lab / Surge XT). ALL system parameter control (guidance, density, genre weights, record/stop) is handled by the intech PBF4, which is a separate device.
**Watch for**: Never attempt to map system parameters to MicroLab mk3 CC numbers. The PBF4 is the only parameter controller.

---

## 2026-04-15 — Analog Lab Intro (Bundled) is Plugin-Only

**Attempted**: N/A (documented pre-emptively)
**Result**: N/A
**Root cause**: Analog Lab Intro (bundled with MicroLab mk3) is a VST/AU/AAX plugin format only. It requires a DAW host.
**Correct approach**: Two options:
  1. Use Ableton Live Lite (also bundled) as DAW host for Analog Lab Intro
  2. Use Surge XT standalone synthesizer (free, open source) — simpler for prototyping
**Watch for**: Do not assume Analog Lab is a standalone app. Test which is actually installed before proceeding.

---

## 2026-04-15 — VB-Cable: Use WASAPI Shared Mode, Not ASIO

**Attempted**: N/A (documented from GitHub issue research)
**Result**: ASIO driver loading fails with VB-Cable in Python sounddevice (GitHub issue #520, closed "not planned").
**Root cause**: VB-Cable's ASIO driver is not compatible with how python-sounddevice loads ASIO.
**Correct approach**: Use WASAPI shared mode (default in sounddevice). Do NOT pass `extra_settings=WasapiSettings(exclusive=True)` when recording from VB-Cable — exclusive mode conflicts with other apps using the virtual device simultaneously.
**Watch for**: When opening a sounddevice InputStream on "CABLE Output", use default settings. If latency is an issue, reduce blocksize but keep WASAPI shared mode.

---

## 2026-04-15 — ThinkPad Has No Discrete GPU: Cloud-Only Deployment

**Attempted**: N/A (confirmed from user)
**Result**: N/A
**Root cause**: The ThinkPad running the final script has no discrete GPU, making local Magenta RT inference impossible (~20–40GB VRAM required).
**Correct approach**: All AI inference must use cloud:
  - **Phase 1 (prototype)**: Lyria RealTime API via Google AI Studio free trial (WebSocket, parallel, simple)
  - **Phase 2**: Magenta RT on Colab free TPU + FastAPI + ngrok (for audio injection capability)
**Watch for**: Network latency to cloud APIs adds variable delay. The buffer-pass architecture (one loop pass of "dead time" per instrument) is specifically designed to absorb this latency.

---

## 2026-04-15 — 4 Parallel Magenta RT Instances Not Feasible on One Colab Free Session

**Attempted**: N/A (assessed from architecture constraints)
**Result**: N/A
**Root cause**: Colab free tier provides one v2-8 TPU. Loading Magenta RT (800M params) consumes most of the memory. Running 4 truly parallel inference threads is likely OOM.
**Correct approach**: Choose one:
  A. 4 separate Colab notebooks (4 browser tabs), each with own ngrok URL — clunky but free
  B. Lyria API (4 WebSocket connections, genuinely parallel) — use free trial credits
  C. Sequential inference (1 Colab, 4 sequential calls per loop pass) — slower but simpler
**Watch for**: Always test single-instance performance before assuming multi-instance works. Profile inference time on Colab before designing the multi-instance architecture.

---

*Add new entries below this line after each session.*

---

## 2026-04-15 — Prototype Synth: Surge XT, Not Analog Lab

**Attempted**: N/A (decision documented)
**Root cause**: Analog Lab Intro (bundled with MicroLab mk3) is a VST plugin, requires DAW host. Ableton Live Lite adds setup friction during prototyping.
**Correct approach**: Use Surge XT standalone (https://surge-synthesizer.github.io/) for the prototype. Set its audio output to "CABLE Input" (VB-Cable). Python script captures from "CABLE Output". No DAW needed.
**Watch for**: Surge XT configuration must be done manually before each session — ensure output device is set to VB-Cable, not system default speakers.

---

## 2026-04-15 — Monitoring: Python Passthrough, Not Voicemeeter

**Attempted**: N/A (decision documented)
**Root cause**: Voicemeeter adds setup overhead. Monitoring can be handled inside the Python output callback by mixing captured VB-Cable audio with AI output.
**Correct approach**: `PassthroughMonitor` class reads from audio capture queue and writes to output device in the same sounddevice output callback as the AI mix. ~20–40ms roundtrip is acceptable.
**Watch for**: The passthrough audio must be mixed at the same gain level as AI voices. If user loop is too loud vs. AI, add a configurable `monitor_gain` parameter.

---

## 2026-04-15 — Phase 1 Backend: Lyria API (Not Magenta RT)

**Attempted**: N/A (decision documented)
**Root cause**: Running 4 parallel Magenta RT instances on Colab free tier is infeasible. Lyria RealTime API is a genuine cloud service supporting 4 parallel WebSocket connections without any server infrastructure.
**Correct approach**: Implement `LyriaBackend` first. Stub `MagentaRTBackend` with NotImplementedError. The `AIInstrumentBackend` abstraction means swapping them is a config change, not a rewrite.
**Watch for**: Verify Lyria RealTime Music API (not just Gemini text) is available on the account at https://aistudio.google.com/ before writing any Lyria connection code.

---

## 2026-04-15 — Lyria RealTime API Cannot Accept Audio Input — Eliminated

**Attempted**: Planned to use Lyria RealTime API as Phase 1 backend.
**Result**: Lyria confirmed text-only. Official docs and developer guide both state: input types are `WeightedPrompt` (text) and `MusicGenerationConfig` (numerical params). No audio input of any kind.
**Root cause**: Lyria is a generative model that produces music from style descriptions. It does not have an "audio continuation" or "audio injection" mode.
**Correct approach**: Use **Magenta RT exclusively**. It is the only option with audio injection. Lyria is removed from this project entirely.
**Watch for**: Do not revisit Lyria for any backend role in this project.

---

## 2026-04-15 — Temperature and Top-K ARE Real Magenta RT Parameters

**Attempted**: Documented that temperature/topk were "not available" based on initial research.
**Result**: Actual notebook source code shows both are real `generate_chunk()` kwargs.
**Root cause**: Early research was based on the project README/paper, not the actual code.
**Correct approach**: Map PBF4 knobs to: `guidance_weight`, `temperature`, `topk`, `model_feedback`. All four are real model parameters verified from source.
**Watch for**: Always verify parameters by reading actual source code, not just docs.

---

## 2026-04-15 — Magenta RT Audio Injection: Exact Mechanism Documented

**Attempted**: N/A (research session)
**Root cause**: Needed to understand audio injection before designing the cascade.
**Correct approach (verified from notebook source)**:
1. Input audio (numpy, 48kHz stereo) is accumulated in `injection_state.all_inputs`
2. A window of recent input is mixed with recent model output: `mix = input_window + output_window * model_feedback`
3. Mix is encoded to SpectroStream tokens via `spectrostream_model.encode(mix_audio)`
4. Tokens replace model context: `state.context_tokens[-N:] = mix_tokens`
5. Both original and mixed contexts are passed to `generate_chunk()` (tied CFG)
6. Output is 2s of 48kHz stereo audio
**For cascade**: AI Instance N receives `sum(user_loop, ai_1_output, ..., ai_n-1_output)` as its injection input.
**Watch for**: The `colab_utils.AudioStreamer` used in the notebook is Colab-specific. The core generation logic (encode → inject → generate_chunk) can be extracted and run independently without Colab UI.

---

## 2026-04-15 — AIVoice Output is Shorter than CHUNK_SAMPLES

**Attempted**: N/A (design-time discovery)
**Root cause**: `_AudioFade.__call__()` removes `fade_samples` from the right end of each chunk to enable crossfading. The output is `(CHUNK_SAMPLES - fade_samples, 2)`, not `(CHUNK_SAMPLES, 2)`.
**Correct approach**: Use `pad_to_chunk()` helper in `poc_cascade.py` to zero-pad voice outputs back to CHUNK_SAMPLES before adding them to the cascade cumulative mix. This ensures each voice receives a consistent CHUNK_SAMPLES input.
**Watch for**: Never assume voice output length equals input length. Always use `pad_to_chunk()` when building cumulative_input.

---

## 2026-04-15 — Model is Shared Across All Voices; SpectroStream Too

**Attempted**: N/A (design-time decision)
**Root cause**: Loading MagentaRTCFGTied takes 3–5 minutes and significant memory. Each voice does NOT need its own model instance.
**Correct approach**: Load one `MagentaRTCFGTied` and one `SpectroStreamJAX`. Pass both to all `AIVoice()` constructors. State is isolated per-voice in `_InjectionState` and `MagentaRTState` — the model weights are stateless.
**Watch for**: Do not instantiate a separate model per voice. One model, multiple stateful voice wrappers.
