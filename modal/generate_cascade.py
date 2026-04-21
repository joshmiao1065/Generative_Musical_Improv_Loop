"""
generate_cascade.py — Offline cascade audio generation on Modal GPU
====================================================================
Takes a user loop WAV, runs the 3-voice AI cascade, returns a mixed
output WAV. Not real-time (generation is ~1.1× loop duration per voice
on A100), but produces actual musical output you can listen to.

USAGE
─────
  # Upload your loop and generate:
  modal run modal/generate_cascade.py --loop path/to/your_loop.wav

  # Adjust style, passes, BPM:
  modal run modal/generate_cascade.py --loop your_loop.wav --bpm 110 --passes 4

OUTPUT
──────
  cascade_output_<timestamp>.wav written to the current directory.

GENERATION TIME (estimated)
────────────────────────────
  A100-80GB (~1.0s/chunk):  4-pass, 3-voice, 16-beat @ 120 BPM ≈ 3 min
  A10G      (~2.3s/chunk):  same config ≈ 7 min
"""

import io
import sys
import time
from pathlib import Path

import modal

# ─────────────────────────────────────────────────────────────────────────────
# Reuse the same image from benchmark_magenta.py
# ─────────────────────────────────────────────────────────────────────────────

GPU_TYPE    = "A100-80GB"
HF_CACHE_DIR = "/hf-cache"

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.6.1-cudnn-runtime-ubuntu24.04",
        add_python="3.12",
    )
    .apt_install("git", "build-essential", "patch", "libsndfile1")
    .run_commands(
        "git clone --depth 1 https://github.com/magenta/magenta-realtime.git /magenta-realtime",
        "git clone        https://github.com/google-research/t5x.git          /t5x",
        "cd /t5x && git checkout 7781d167ab421dae96281860c09d5bd785983853",
    )
    .run_commands(
        "cd /t5x && patch setup.py < /magenta-realtime/patch/t5x_setup.py.patch",
        "cd /t5x && pip install -e '.'",
        "pip install 'jax[cuda12]'",
        "pip install -e '/magenta-realtime/'",
        "pip install tf2jax==0.3.8 soundfile librosa resampy huggingface_hub",
    )
    .run_commands(
        "patch /t5x/t5x/partitioning.py < /magenta-realtime/patch/t5x_partitioning.py.patch",
        'SEQIO=$(pip show seqio | grep Location | cut -d" " -f2) && '
        'patch "$SEQIO/seqio/vocabularies.py" < /magenta-realtime/patch/seqio_vocabularies.py.patch',
    )
    .add_local_dir(
        str(Path(__file__).parent.parent / "src"),
        remote_path="/root/src",
        copy=True,
    )
    .env({
        "HF_HOME":                       HF_CACHE_DIR,
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
    })
)

hf_cache_vol = modal.Volume.from_name("magenta-rt-hf-cache", create_if_missing=True)

app = modal.App(
    "magenta-rt-generate",
    image=image,
    volumes={HF_CACHE_DIR: hf_cache_vol},
)

# ─────────────────────────────────────────────────────────────────────────────
# VOICE STYLES — edit these to change the AI musicians
# ─────────────────────────────────────────────────────────────────────────────

VOICE_STYLES = [
    "jazz piano solo, expressive, melodic improvisation",
    "upright bass, walking bass line, jazz",
    "jazz drum kit, brushed snare, ride cymbal",
]


# ─────────────────────────────────────────────────────────────────────────────
# GENERATION CLASS
# ─────────────────────────────────────────────────────────────────────────────

@app.cls(
    gpu=GPU_TYPE,
    timeout=3600,         # 1 hour max
    min_containers=0,
    scaledown_window=120,
)
class CascadeGenerator:

    @modal.enter()
    def load(self) -> None:
        import jax
        import numpy as np

        sys.path.insert(0, "/root/src")
        from magenta_rt import spectrostream
        from magenta_backend import (
            AIVoice, GenerationParams, MagentaRTCFGTied,
            CHUNK_SAMPLES, SAMPLE_RATE,
        )

        self.AIVoice         = AIVoice
        self.GenerationParams = GenerationParams
        self.CHUNK_SAMPLES   = CHUNK_SAMPLES
        self.SAMPLE_RATE     = SAMPLE_RATE

        print(f"GPU: {jax.devices()}")
        print("Loading SpectroStream...")
        self.ss_model = spectrostream.SpectroStreamJAX(lazy=False)
        print("Loading Magenta RT (large)...")
        self.model = MagentaRTCFGTied(tag="large", lazy=False)

        # One JIT warm-up so the first real chunk doesn't pay compilation cost
        print("Warming up JAX JIT...")
        params = GenerationParams()
        v = AIVoice(self.model, self.ss_model, "piano", params)
        v.step(np.zeros((CHUNK_SAMPLES, 2), dtype=np.float32))
        print("Ready.")

    @modal.method()
    def generate(
        self,
        loop_bytes: bytes,
        bpm: int = 120,
        beats_per_loop: int = 16,
        n_passes: int = 4,
        guidance_weight: float = 1.5,
        temperature: float = 1.2,
        topk: int = 30,
        model_feedback: float = 0.95,
    ) -> bytes:
        """
        Run the cascade and return the mixed output as WAV bytes.

        Args:
            loop_bytes:     WAV/MP3 file content as bytes.
            bpm:            Tempo of the input loop.
            beats_per_loop: Loop length in beats (sets chunk count per pass).
            n_passes:       How many full loop iterations to generate.
            guidance_weight / temperature / topk / model_feedback: model params.

        Returns:
            WAV bytes of the full mixed output (user loop + all AI voices).
        """
        import io
        import numpy as np
        import soundfile as sf
        import librosa

        SAMPLE_RATE   = self.SAMPLE_RATE
        CHUNK_SAMPLES = self.CHUNK_SAMPLES
        CHUNK_SECS    = CHUNK_SAMPLES / SAMPLE_RATE

        # ── Load user loop ────────────────────────────────────────────────────
        audio, sr = librosa.load(io.BytesIO(loop_bytes), sr=SAMPLE_RATE, mono=False)
        if audio.ndim == 1:
            audio = np.stack([audio, audio], axis=0)
        user_loop = audio.T.astype(np.float32)  # (N, 2)
        print(f"User loop: {user_loop.shape[0]/SAMPLE_RATE:.2f}s  @{SAMPLE_RATE}Hz stereo")

        # ── Build voices ──────────────────────────────────────────────────────
        params = self.GenerationParams(
            guidance_weight=guidance_weight,
            temperature=temperature,
            topk=topk,
            model_feedback=model_feedback,
            model_volume=0.85,
            beats_per_loop=beats_per_loop,
            bpm=bpm,
        )
        voices = [
            self.AIVoice(self.model, self.ss_model, style, params)
            for style in VOICE_STYLES
        ]

        # ── Cascade loop ──────────────────────────────────────────────────────
        loop_sec        = beats_per_loop * (60.0 / bpm)
        chunks_per_pass = max(1, round(loop_sec * SAMPLE_RATE / CHUNK_SAMPLES))
        total_chunks    = n_passes * chunks_per_pass

        print(f"\n{'─'*56}")
        print(f"CASCADE: {len(voices)} voices × {n_passes} passes × {chunks_per_pass} chunks/pass")
        print(f"         = {total_chunks} total chunks  (~{total_chunks * CHUNK_SECS:.0f}s audio out)")
        print(f"{'─'*56}")

        def get_chunk(step: int) -> np.ndarray:
            n = user_loop.shape[0]
            start = (step * CHUNK_SAMPLES) % n
            end   = start + CHUNK_SAMPLES
            if end <= n:
                return user_loop[start:end].copy()
            return np.concatenate([user_loop[start:], user_loop[:end - n]])

        def pad(arr: np.ndarray) -> np.ndarray:
            out = np.zeros((CHUNK_SAMPLES, 2), dtype=np.float32)
            out[:len(arr)] = arr
            return out

        output_chunks: list[np.ndarray] = []
        t_start = time.perf_counter()

        for step in range(total_chunks):
            user_chunk = get_chunk(step)
            cumulative = user_chunk.copy()
            voice_outputs = []

            for voice in voices:
                out = voice.step(cumulative)
                out_padded = pad(out)
                voice_outputs.append(out_padded)
                cumulative = cumulative + out_padded

            # Mix: user loop + all voices
            mix = user_chunk.copy()
            for vo in voice_outputs:
                mix = mix + vo

            # Normalize to prevent clipping
            peak = np.max(np.abs(mix))
            if peak > 1e-6:
                mix = mix * (0.95 / peak)
            output_chunks.append(mix)

            if (step + 1) % chunks_per_pass == 0:
                elapsed = time.perf_counter() - t_start
                done    = (step + 1) / total_chunks
                eta     = (elapsed / done) * (1 - done)
                print(f"  Pass {(step+1)//chunks_per_pass}/{n_passes} "
                      f"[{elapsed:.0f}s elapsed, ~{eta:.0f}s remaining]")

        print(f"\nGeneration complete: {time.perf_counter() - t_start:.0f}s total")

        # ── Encode to WAV bytes ───────────────────────────────────────────────
        output = np.concatenate(output_chunks, axis=0)
        buf = io.BytesIO()
        sf.write(buf, output, SAMPLE_RATE, format="WAV", subtype="PCM_24")
        print(f"Output: {output.shape[0]/SAMPLE_RATE:.1f}s of audio ({len(buf.getvalue())//1024}KB)")
        return buf.getvalue()


# ─────────────────────────────────────────────────────────────────────────────
# LOCAL ENTRYPOINT
# ─────────────────────────────────────────────────────────────────────────────

@app.local_entrypoint()
def main(
    loop: str = "",
    bpm: int = 120,
    beats: int = 16,
    passes: int = 4,
) -> None:
    """
    Usage:
      modal run modal/generate_cascade.py --loop your_loop.wav
      modal run modal/generate_cascade.py --loop your_loop.wav --bpm 110 --beats 8 --passes 6
    """
    import datetime

    if not loop:
        print("ERROR: --loop <path> is required.")
        print("  Example: modal run modal/generate_cascade.py --loop my_loop.wav")
        return

    loop_path = Path(loop)
    if not loop_path.exists():
        print(f"ERROR: File not found: {loop_path}")
        return

    print(f"Loop file  : {loop_path}  ({loop_path.stat().st_size // 1024}KB)")
    print(f"Config     : {beats} beats @ {bpm} BPM  ×  {passes} passes  ×  {len(VOICE_STYLES)} voices")
    loop_sec = beats * (60.0 / bpm)
    from math import ceil
    chunks_per_pass = max(1, round(loop_sec * 48000 / 96000))
    total_chunks = passes * chunks_per_pass
    print(f"Est. output: ~{total_chunks * 2:.0f}s of audio")
    print(f"Est. time  : ~{total_chunks * 2.5 / 60:.1f} min on A100 / ~{total_chunks * 2.5 * 2.3 / 60:.1f} min on A10G\n")

    wav_bytes = loop_path.read_bytes()

    gen = CascadeGenerator()
    result_bytes = gen.generate.remote(
        loop_bytes=wav_bytes,
        bpm=bpm,
        beats_per_loop=beats,
        n_passes=passes,
    )

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path  = Path(f"cascade_output_{timestamp}.wav")
    out_path.write_bytes(result_bytes)
    print(f"\nSaved: {out_path}  ({out_path.stat().st_size // 1024}KB)")
