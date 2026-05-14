# Improv Loop

**Cooper Union — Generative Machine Learning, Final Project**
**Josh Miao, Spring 2026**

---

A real-time AI music improvisation system. You play a loop on a synthesizer, and three parallel AI voices running on cloud GPUs generate responses in the same style. The system mixes everything together with per-stem faders, a 3-band EQ, and a reverb.

It mostly works. It doesn't work *well*.

---

## What it does

You plug in a synthesizer (Surge XT or Analog Lab) routed through VB-Cable, connect an intech PBF4 MIDI controller, and run the script. Press Button 1 to record a loop. The loop plays back continuously while three Modal containers — each running an 800M-parameter Magenta RT autoregressive model on an A100 GPU — generate audio responses in parallel. The AI voices trickle in one by one starting on the second pass. You can toggle voices on/off, adjust stem volumes, dial in EQ and reverb, and cycle each voice through different genre prompts.

The architecture is genuinely interesting. The generation pipeline fits inside the loop window: at 120 BPM / 16 beats (8 seconds), each generation pass takes ~5.7s, leaving 2.3 seconds of headroom before the next loop boundary. Audio injection feeds the user loop and prior AI outputs into the model's SpectroStream token context. Three A100 containers run in parallel so total wall time is the max of the three, not their sum.

---

## What doesn't work

**The AI doesn't react to what you're playing.** This is the central limitation, and it's not a bug — it's structural. Magenta RT is an autoregressive sequence model with a fixed context window. It generates plausible continuations of its own prior output, informed by the audio injection mechanism, but "informed by" is a long way from "reacting to." If you switch from a jazz riff to a chromatic run, the model won't notice for several passes. If it does adapt, it's because the injected audio nudged the token probabilities slightly, not because the model understood a harmonic change and responded idiomatically.

The latency makes this worse. The minimum lag between what you play and when an AI voice starts responding is one full loop (8 seconds at the default settings), because each generation pass covers one complete loop. Reducing the loop length helps but pushes the RTF budget.

**Genre prompts are cosmetic.** The text encoder converts "jazz" or "drum and bass" into a style embedding that gets mixed into the transformer's context. In practice the difference between genre prompts is audible but subtle — the model tends toward its own learned distribution and genre steering is a soft nudge. "Jazz" doesn't produce swing; "drum and bass" doesn't produce breakbeats. It produces something vaguely genre-adjacent.

**The outputs are coherent but not musical.** Magenta RT generates audio that sounds like music in the sense that it has texture, rhythm, and pitch. But it doesn't construct phrases, respond to harmonic context, or build toward anything. After several passes it tends toward a kind of ambient wash. Extended sessions with multiple voices devolve into increasingly indistinct layered textures.

These are not implementation failures. They reflect the state of the art for models of this class: generative transformers trained on audio can produce convincing local continuations but lack the global musical reasoning that would make real-time improvisation genuinely responsive. A human improviser hears you land on a IV chord and knows to answer it; this model hears tokens and predicts the next token.

---

## What was built over 11 sessions

The architecture evolved considerably:

- **Sessions 1–3**: Basic loop capture → Modal dispatch → playback pipeline. Colab was the original deployment target; eliminated after the first session disconnect.
- **Sessions 4–5**: Hardware integration — VB-Cable routing, PBF4 MIDI mapping, WASAPI audio I/O on Windows.
- **Sessions 6–7**: Fixed the real-time constraint. Early version took 24–38 seconds per generation pass due to `librosa.load` overhead on the server. Replacing it with `soundfile.read` dropped this to ~5.7 seconds.
- **Sessions 8–9**: A10G attempted and abandoned (13% too slow for the target RTF). A100-40GB attempted and abandoned (XLA OOM on the 800M model). Settled on A100-80GB.
- **Sessions 10–11**: Redesigned parameter model — removed genre blending (per-voice genres instead), removed crossfader (per-stem faders instead), added 3-band EQ and Freeverb reverb processed locally. Fixed Freeverb allpass buffer wrap bug (buffers 2–4 are smaller than a 512-sample audio block, causing pitch artifacts with a naive vectorized implementation).

The codebase has a `CLAUDE.md` and `LESSONS.md` that document this evolution in more detail, including specific bugs, root causes, and architectural decisions.

---

## Setup

**Requirements:**
- Windows (USB MIDI and WASAPI audio don't work from WSL2)
- Python 3.11+ with `.venv`
- [VB-Cable](https://vb-audio.com/Cable/) (virtual audio cable for clean synth routing)
- [intech PBF4](https://intech.studio/products/pbf4) (MIDI controller)
- [Surge XT](https://surge-synthesizer.github.io/) or any WASAPI-capable synthesizer
- Modal account with A100-80GB access (~$7.50/hr for 3 active voices)

**Install:**
```bat
pip install -r requirements.txt
modal setup
```

**Deploy the Modal server (one-time):**
```bat
modal deploy server/magenta_server.py
```

**Run:**
```bat
.venv\Scripts\python improv_loop.py --bpm 120 --beats 16 --genres "jazz" "bossa nova" "electronic"
```

**Or test without Modal (sine tones as placeholders):**
```bat
.venv\Scripts\python improv_loop.py --dry-run --qwerty
```

**List available audio/MIDI devices:**
```bat
.venv\Scripts\python improv_loop.py --list-devices
```

---

## Controls

| Hardware | Action |
|----------|--------|
| PBF4 Button 1 | Count-in → record → play → re-record |
| PBF4 Buttons 2/3/4 | Toggle AI Voice 1/2/3 on/off |
| PBF4 Faders (CC 36–39) | User loop / AI Voice 1/2/3 volume |
| PBF4 Knobs (CC 32–34) | EQ Bass / Mid / Treble (±12 dB) |
| PBF4 Knob 4 (CC 35) | Reverb wet mix |
| Keys `4` / `5` / `6` | Cycle genre for Voice 1/2/3 |
| Key `?` | Print current genres and volumes |

**Note:** PBF4 CC numbers were entered manually and need verification. Run `scripts/discover_cc.py` and move all knobs/faders to confirm the mapping.

---

## Architecture

```
Surge XT → VB-Cable Input → CABLE Output → Python InputStream (48kHz stereo)
                                               ↓
                                   Ring buffer + Monitor FIFO
                                               ↓
                                   WAV bytes → Modal (3× A100-80GB)
                                               ↓
                              Voice0Server  Voice1Server  Voice2Server
                              (parallel, ~5.7s wall time)
                                               ↓
                              AudioMixer: loop + voices + EQ + reverb
                                               ↓
                                          Speakers
```

**Buffer pass architecture:** Each AI generation covers one full loop. Voice 0 joins at pass 2, Voice 1 at pass 3, Voice 2 at pass 4. The 5.7s generation time fits inside the 8s loop window with 2.3s headroom.

---

## File structure

```
improv_loop.py           Main session orchestrator
config/
  pbf4_layout.json       CC assignments for PBF4 controls (verify with hardware)
  pbf4_cc_map.json       Auto-generated by discover_cc.py
src/
  magenta_backend.py     AIVoice, MagentaRTCFGTied — audio injection logic
  modal_client.py        Async client for Modal voice servers
  audio_mixer.py         Real-time output mixer with loop + voice management
  effects_chain.py       3-band RBJ EQ + Freeverb reverb
  loop_capture.py        Ring buffer audio capture with MonitorFIFO
  midi_controller.py     PBF4 MIDI handler
  keyboard_controller.py QWERTY fallback (--qwerty flag)
  timing_engine.py       Pass-boundary clock using perf_counter
  audio_devices.py       Device auto-detection
server/
  magenta_server.py      Modal deployment — 3 VoiceServer classes
scripts/
  discover_cc.py         Interactive CC discovery for PBF4
  prime_server.py        Check Modal container warm status
  test_audio_logic.py    AudioMixer unit tests (no hardware)
  test_effects_algorithms.py  EQ + reverb algorithm tests
```

---

## Dependencies

**Client (Windows):**
```
mido>=1.3.0
python-rtmidi>=1.5.8
sounddevice>=0.4.6
numpy>=1.26.0
scipy>=1.10.0
soundfile>=0.12.0
modal>=0.73.0
```

**Modal containers:** Built from CUDA 12.6 + Magenta RT + T5X. See `server/magenta_server.py` for the full image definition.

---

## Reflections

The core problem this project attempted to solve — real-time reactionary AI improvisation — is harder than it looks from the outside. The word "real-time" sets expectations that current generative models can't meet without significant compromises. You can make the generation fast enough to fit inside a loop window, but fast-enough-to-fit is not the same as fast-enough-to-feel-responsive. One loop of lag (8 seconds) is an eternity in musical time.

The deeper issue is that autoregressive audio models optimize for local coherence, not musical structure. Human improvisation involves tracking harmonic motion, predicting phrase endings, deciding when to hold back and when to push — all of which require a kind of real-time formal reasoning that these models don't do. What they do instead is generate audio that sounds like something that might follow from what came before, which produces output that is plausible but not reactive.

This is interesting territory. The gap between "generates music" and "responds to music" is large and not primarily a compute problem.
