"""
poc_cascade.py — Proof of Concept: AI Musician Cascade Loop
============================================================
Demonstrates that multiple Magenta RT voices can listen and respond to
each other through audio injection. No hardware required.

HOW TO RUN (Google Colab — TPU required):
─────────────────────────────────────────
1. Open https://colab.research.google.com
2. Runtime → Change runtime type → TPU
3. In a new code cell, paste and run the INSTALL block below.
4. Upload this file and src/magenta_backend.py (or clone the repo).
5. Upload a test audio loop (any WAV/MP3, 4–30 seconds of music).
6. Set USER_LOOP_PATH to your file's name, then run this script.

Expected runtime (Colab free TPU v2-8):
  ~1.25s per 2-second chunk generated
  N_PASSES=6 × 4 chunks/loop × 2 voices = 48 chunks ≈ 60s generation time
  → produces ~48 seconds of output audio

Output: cascade_output.wav written to current directory.

INSTALL (run in a separate Colab cell before this script):
──────────────────────────────────────────────────────────
    !git clone https://github.com/magenta/magenta-realtime.git
    !git clone https://github.com/google-research/t5x.git
    !sed -i '/optax/d' t5x/setup.py
    !pip install -q -e t5x/[tpu] && pip install -q -e magenta-realtime/[tpu] && pip install -q tf2jax==0.3.8 resampy soundfile
    !sed -i '/import tensorflow_text as tf_text/d' /usr/local/lib/python3.12/dist-packages/seqio/vocabularies.py
    !sed -i "s|device_kind == 'TPU v4 lite'|device_kind == 'TPU v4 lite' or device_kind == 'TPU v5 lite' or device_kind == 'TPU v6 lite'|g" t5x/t5x/partitioning.py
"""

import os
import sys
import time

import librosa
import numpy as np
import soundfile as sf

# ── path setup: works whether run from repo root or from colab/ ───────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from magenta_rt import spectrostream
from src.magenta_backend import (
    AIVoice,
    GenerationParams,
    MagentaRTCFGTied,
    CHUNK_SAMPLES,
    SAMPLE_RATE,
)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG — edit these before running
# ─────────────────────────────────────────────────────────────────────────────

USER_LOOP_PATH = "user_loop.wav"    # path to test audio file; upload to Colab first
OUTPUT_PATH    = "cascade_output.wav"

BPM            = 120
BEATS_PER_LOOP = 8                  # 8 beats = 2 bars at 4/4
N_PASSES       = 6                  # how many full loops to generate
N_AI_VOICES    = 2                  # 2–4; each additional voice adds ~N_PASSES chunks of generation

# Style prompts — one per voice. Each describes the instrument's musical role.
VOICE_STYLES = [
    "jazz piano solo, expressive, melodic improvisation",
    "upright bass, walking bass line, jazz",
    "jazz drum kit, brushed snare, ride cymbal",   # used when N_AI_VOICES >= 3
    "electric guitar, jazz chord comping",          # used when N_AI_VOICES == 4
]

# Shared generation parameters for all voices.
# These will be wired to PBF4 knobs in a future iteration.
PARAMS = GenerationParams(
    guidance_weight=1.5,
    temperature=1.2,
    topk=30,
    model_feedback=0.95,
    model_volume=0.85,
    beats_per_loop=BEATS_PER_LOOP,
    bpm=BPM,
)

# ─────────────────────────────────────────────────────────────────────────────
# AUDIO UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def load_loop(path: str) -> np.ndarray:
    """Load audio, resample to SAMPLE_RATE, return (N, 2) float32 stereo."""
    audio, sr = librosa.load(path, sr=SAMPLE_RATE, mono=False)
    if audio.ndim == 1:
        audio = np.stack([audio, audio], axis=0)  # mono → stereo
    return audio.T.astype(np.float32)  # (N, 2)


def get_loop_chunk(loop: np.ndarray, step: int) -> np.ndarray:
    """Return one CHUNK_SAMPLES slice of the looping audio, wrapping at end."""
    n = loop.shape[0]
    start = (step * CHUNK_SAMPLES) % n
    end   = start + CHUNK_SAMPLES
    if end <= n:
        return loop[start:end].copy()
    # wrap around
    return np.concatenate([loop[start:], loop[: end - n]], axis=0)


def pad_to_chunk(audio: np.ndarray) -> np.ndarray:
    """Zero-pad (or trim) audio to exactly CHUNK_SAMPLES along axis 0."""
    n = audio.shape[0]
    if n >= CHUNK_SAMPLES:
        return audio[:CHUNK_SAMPLES]
    pad = np.zeros((CHUNK_SAMPLES - n, 2), dtype=np.float32)
    return np.concatenate([audio, pad], axis=0)


def normalize(audio: np.ndarray, headroom: float = 0.95) -> np.ndarray:
    """Peak normalize to avoid clipping."""
    peak = np.max(np.abs(audio))
    if peak > 1e-6:
        return audio * (headroom / peak)
    return audio

# ─────────────────────────────────────────────────────────────────────────────
# CASCADE LOOP
# ─────────────────────────────────────────────────────────────────────────────

def run_cascade(
    voices: list[AIVoice],
    user_loop: np.ndarray,
    n_passes: int,
) -> np.ndarray:
    """
    Run the AI musician cascade for n_passes full loop iterations.

    At each step:
      - Voice 1 receives: user_loop_chunk
      - Voice 2 receives: user_loop_chunk + voice_1_output (padded)
      - Voice N receives: cumulative mix of all prior voices + user_loop_chunk

    All voices run sequentially per chunk (single Colab TPU, no true parallelism).
    In the full system, voices run on separate Colab sessions in parallel.

    Returns:
        Mixed output array, shape (total_samples, 2), float32.
    """
    loop_duration_sec = BEATS_PER_LOOP * (60.0 / BPM)
    chunks_per_loop   = max(1, int(loop_duration_sec * SAMPLE_RATE / CHUNK_SAMPLES))
    total_chunks      = n_passes * chunks_per_loop

    print(f"\n{'─'*60}")
    print(f"CASCADE CONFIGURATION")
    print(f"  {len(voices)} AI voices, {n_passes} passes, {chunks_per_loop} chunks/pass")
    print(f"  Total chunks: {total_chunks}  (~{total_chunks * 1.25:.0f}s generation time)")
    print(f"  Output duration: ~{total_chunks * CHUNK_SAMPLES / SAMPLE_RATE:.0f}s")
    print(f"{'─'*60}\n")

    output_chunks: list[np.ndarray] = []
    t_start = time.time()

    for step in range(total_chunks):
        user_chunk = get_loop_chunk(user_loop, step)  # (CHUNK_SAMPLES, 2)

        # Input to voice N = user_loop + padded outputs of voices 1..N-1
        cumulative_input = user_chunk.copy()
        voice_outputs: list[np.ndarray] = []

        for i, voice in enumerate(voices):
            out = voice.step(cumulative_input)          # (CHUNK_SAMPLES - fade, 2)
            out_padded = pad_to_chunk(out)              # (CHUNK_SAMPLES, 2)
            voice_outputs.append(out_padded)
            cumulative_input = cumulative_input + out_padded  # next voice hears more

        # Final mix: user + all voice outputs
        mix = user_chunk.copy()
        for vo in voice_outputs:
            mix = mix + vo

        output_chunks.append(normalize(mix))

        # Progress logging
        if (step + 1) % chunks_per_loop == 0:
            elapsed = time.time() - t_start
            done    = (step + 1) / total_chunks
            eta     = (elapsed / done) * (1 - done) if done > 0 else 0
            print(f"  Pass {(step + 1) // chunks_per_loop}/{n_passes} complete "
                  f"[{elapsed:.0f}s elapsed, ~{eta:.0f}s remaining]")

    print(f"\nCascade complete. Total time: {time.time() - t_start:.0f}s")
    return np.concatenate(output_chunks, axis=0)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # 1. Load user loop
    print(f"Loading user loop: {USER_LOOP_PATH}")
    if not os.path.exists(USER_LOOP_PATH):
        raise FileNotFoundError(
            f"No audio file found at '{USER_LOOP_PATH}'.\n"
            "Upload a WAV or MP3 file to Colab and update USER_LOOP_PATH."
        )
    user_loop = load_loop(USER_LOOP_PATH)
    print(f"  Loaded: {user_loop.shape[0] / SAMPLE_RATE:.2f}s at {SAMPLE_RATE} Hz stereo")

    # 2. Load SpectroStream encoder (shared across all voices)
    print("\nLoading SpectroStream encoder...")
    ss_model = spectrostream.SpectroStreamJAX(lazy=False)
    print("  Done.")

    # 3. Load Magenta RT model (single instance shared across voices)
    print("\nLoading Magenta RT model (large) — first run downloads weights, ~3–5 min...")
    model = MagentaRTCFGTied(tag="large", lazy=False)
    print("  Model loaded.")

    # 4. Create AI voices
    n_voices = min(N_AI_VOICES, len(VOICE_STYLES))
    print(f"\nCreating {n_voices} AI voices:")
    voices: list[AIVoice] = []
    for i in range(n_voices):
        style = VOICE_STYLES[i]
        print(f"  Voice {i+1}: '{style}'")
        voices.append(AIVoice(model, ss_model, style, PARAMS))

    # 5. Run cascade
    output = run_cascade(voices, user_loop, N_PASSES)

    # 6. Save output
    os.makedirs(os.path.dirname(os.path.abspath(OUTPUT_PATH)), exist_ok=True)
    sf.write(OUTPUT_PATH, output, SAMPLE_RATE)
    print(f"\nSaved: {OUTPUT_PATH}  ({output.shape[0]/SAMPLE_RATE:.1f}s)")

    # 7. Play in Colab if available
    try:
        from IPython.display import Audio, display
        print("Playing output in notebook...")
        display(Audio(output.T, rate=SAMPLE_RATE))
    except ImportError:
        print("Not running in Colab — play the output WAV file directly.")


if __name__ == "__main__":
    main()
