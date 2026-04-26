"""
magenta_server.py — Persistent parallel Modal server for real-time improv
=========================================================================
Deploys one A100-80GB container per AI voice. Each container keeps its
model and voice state warm between calls, enabling sub-loop-duration
generation per pass.

DEPLOY (run once — survives terminal close):
  modal deploy modal/magenta_server.py

STOP (halts billing):
  modal app stop magenta-rt-server

CALL from Python (see src/modal_client.py):
  from modal_client import MagentaRTClient
  client = MagentaRTClient()
  outputs = await client.generate_pass(user_loop_np)

ARCHITECTURE
────────────
  ThinkPad client                 Modal (3 × A100, parallel)
  ───────────────                 ──────────────────────────
  Pass N playing (8s)   ───────▶  Voice 0  hears user_loop
                         ───────▶  Voice 1  hears user_loop + V0_prev_pass
                         ───────▶  Voice 2  hears user_loop + V0_prev + V1_prev
                                  ↓ all generate simultaneously (~5.6s)
  Pass N+1 ready        ◀───────  outputs arrive with ~2.4s headroom

One-pass lag between voices is intentional — each hears the previous
pass's outputs. This creates a natural "call and response" feel.
"""

import io
import sys
import time
from pathlib import Path

import modal
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

GPU_TYPE     = "A100-80GB"
HF_CACHE_DIR = "/hf-cache"
APP_NAME     = "magenta-rt-server"

VOICE_STYLES = [
    "Electric guitar, expressive, melodic",
    "bass guitar",
    "EDM synthesized drum machine",
]

# ─────────────────────────────────────────────────────────────────────────────
# IMAGE (same as benchmark — cached, builds in ~6s)
# ─────────────────────────────────────────────────────────────────────────────

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
    .env({
        "HF_HOME":                       HF_CACHE_DIR,
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
    })
    .add_local_file(
        str(Path(__file__).parent.parent / "src" / "magenta_backend.py"),
        remote_path="/root/src/magenta_backend.py",
    )
)

hf_cache_vol = modal.Volume.from_name("magenta-rt-hf-cache", create_if_missing=True)

app = modal.App(
    APP_NAME,
    image=image,
    volumes={HF_CACHE_DIR: hf_cache_vol},
)


# ─────────────────────────────────────────────────────────────────────────────
# VOICE SERVER
# Each voice_index gets its own container pool on its own A100.
# ─────────────────────────────────────────────────────────────────────────────

@app.cls(
    gpu=GPU_TYPE,
    scaledown_window=600,    # stay alive 10 min after last call (covers inter-pass gaps)
    timeout=600,             # 10 min: cold start + XLA compile can take >5 min on first boot
    env={
        "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.85",
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        "HF_HOME": HF_CACHE_DIR,
    }
)
class VoiceServer:
    voice_index: int = modal.parameter()

    @modal.method()
    def prime(self) -> str:
        """Run a dummy inference pass to force JIT compilation."""
        import numpy as np
        print(f"Voice {self.voice_index} priming...")
        dummy = np.zeros((self.CHUNK_SAMPLES, 2), dtype=np.float32)
        self.voice.step(dummy)
        return "Model primed and JIT compiled."

    @modal.enter()
    def load(self) -> None:
        """Load model and voice state once per container lifetime."""
        import jax

        sys.path.insert(0, "/root/src")
        from magenta_rt import spectrostream
        from magenta_backend import (
            AIVoice, GenerationParams, MagentaRTCFGTied,
            CHUNK_SAMPLES, SAMPLE_RATE,
        )

        self.CHUNK_SAMPLES = CHUNK_SAMPLES
        self.SAMPLE_RATE   = SAMPLE_RATE

        style = VOICE_STYLES[self.voice_index]
        print(f"Voice {self.voice_index} container starting on {jax.devices()}")

        self.ss_model = spectrostream.SpectroStreamJAX(lazy=False)
        self.model    = MagentaRTCFGTied(tag="large", lazy=False)

        self.params = GenerationParams(
            guidance_weight=1.5,
            temperature=1.2,
            topk=30,
            model_feedback=0.95,
            model_volume=0.85,
            beats_per_loop=16,
            bpm=120,
        )

        self.voice = AIVoice(self.model, self.ss_model, style, self.params)

        # Cache for genre style embeddings — populated lazily on first call per genre string.
        # Keys are "{genre} {instrument}" prompts; values are StyleEmbedding objects.
        # Avoids recomputing MusicCoCa forward pass every generation pass.
        self._style_cache: dict = {}

        # JIT warm-up
        dummy = np.zeros((CHUNK_SAMPLES, 2), dtype=np.float32)
        self.voice.step(dummy)
        print(f"Voice {self.voice_index} ({style}) ready.")

    # ── Generation ────────────────────────────────────────────────────────────

    @modal.method()
    def generate_pass(
        self,
        user_loop_bytes: bytes,
        prior_mix_bytes: bytes,
        beats_per_loop: int = 16,
        bpm: int = 120,
        guidance_weight: float = 1.5,
        temperature: float = 1.2,
        topk: int = 30,
        model_feedback: float = 0.95,
        genres: list = None,
        instrument: str = "piano",
        genre_weights: list = None,
    ) -> bytes:
        """
        Generate one full loop pass for this voice.

        Args:
            user_loop_bytes:  WAV bytes of the user's loop (full loop duration).
            prior_mix_bytes:  WAV bytes of all prior voices' combined output
                              from the last pass. Pass silence (zeros) for pass 0
                              or for Voice 0 (which hears only the user loop).
            beats_per_loop:   Loop length in beats — sets how many chunks to generate.
            bpm:              Tempo of the loop.
            guidance_weight / temperature / topk / model_feedback: generation params,
                              updated live from PBF4 knobs.
            genres:           List of genre strings (e.g. ["jazz", "blues"]).
            instrument:       Instrument name (e.g. "piano"). Combined with each genre.
            genre_weights:    Per-genre blend weights [0.0–1.0]. Normalized internally.
                              Embeddings are cached after first computation per genre.

        Returns:
            WAV bytes of this voice's generated audio for the full pass.
        """
        import soundfile as sf
        import librosa
        from magenta_rt import musiccoca

        SR = self.SAMPLE_RATE
        CS = self.CHUNK_SAMPLES

        def _load(wav_bytes: bytes) -> np.ndarray:
            audio, sr = librosa.load(io.BytesIO(wav_bytes), sr=SR, mono=False)
            if audio.ndim == 1:
                audio = np.stack([audio, audio], axis=0)
            return audio.T.astype(np.float32)

        def _chunk(loop: np.ndarray, idx: int) -> np.ndarray:
            n = loop.shape[0]
            start = (idx * CS) % n
            end   = start + CS
            if end <= n:
                return loop[start:end].copy()
            return np.concatenate([loop[start:], loop[:end - n]])

        def _pad(arr: np.ndarray) -> np.ndarray:
            out = np.zeros((CS, 2), dtype=np.float32)
            out[:len(arr)] = arr
            return out

        # Update params from knob values (enables live PBF4 / QWERTY control)
        self.params.guidance_weight = guidance_weight
        self.params.temperature     = temperature
        self.params.topk            = topk
        self.params.model_feedback  = model_feedback
        self.params.beats_per_loop  = beats_per_loop
        self.params.bpm             = bpm

        # ── Genre blending ────────────────────────────────────────────────────
        # Build a blended StyleEmbedding from the live per-genre weights.
        # Embeddings are computed once per unique "{genre} {instrument}" prompt
        # and cached in self._style_cache for the container's lifetime.
        if genres and genre_weights:
            # Filter to genres that have non-trivial weight and non-empty name
            active = [
                (g, w) for g, w in zip(genres, genre_weights)
                if g and w > 1e-4
            ]
            if active:
                active_genres, active_weights = zip(*active)
                ws = np.array(active_weights, dtype=np.float64)
                ws /= ws.sum()   # normalize so weights always sum to 1.0

                embeddings = []
                for g in active_genres:
                    prompt = f"{g} {instrument}"
                    if prompt not in self._style_cache:
                        print(f"Voice {self.voice_index}: computing style embedding for '{prompt}'")
                        self._style_cache[prompt] = self.model.embed_style(prompt)
                    embeddings.append(self._style_cache[prompt])

                if len(embeddings) == 1:
                    blended = embeddings[0]
                else:
                    # Linear interpolation in embedding space, then wrap back into StyleEmbedding
                    blended_vec = sum(
                        float(w) * e.embedding for w, e in zip(ws, embeddings)
                    )
                    blended = musiccoca.StyleEmbedding(embedding=blended_vec)

                self.voice.style_embedding = blended
                blend_desc = " + ".join(
                    f"{g}({w:.0%})" for g, w in zip(active_genres, ws)
                )
                print(f"Voice {self.voice_index}: style = {blend_desc} {instrument}")

        user_loop  = _load(user_loop_bytes)
        prior_mix  = _load(prior_mix_bytes)

        loop_sec        = beats_per_loop * (60.0 / bpm)
        chunks_per_pass = max(1, round(loop_sec * SR / CS))

        t0 = time.perf_counter()
        chunk_outputs = []
        for i in range(chunks_per_pass):
            combined = _chunk(user_loop, i) + _chunk(prior_mix, i)
            out      = self.voice.step(combined)
            chunk_outputs.append(_pad(out))

        elapsed = time.perf_counter() - t0
        audio_generated = chunks_per_pass * CS / SR
        print(f"Voice {self.voice_index}: {elapsed:.2f}s to generate "
              f"{audio_generated:.1f}s audio (RTF {audio_generated/elapsed:.2f}×)")

        full_output = np.concatenate(chunk_outputs, axis=0)

        buf = io.BytesIO()
        sf.write(buf, full_output, SR, format="WAV", subtype="PCM_24")
        return buf.getvalue()

    # ── Session management ────────────────────────────────────────────────────

    @modal.method()
    def reset(self) -> None:
        """Clear voice generation state. Call between jam sessions."""
        self.voice.reset()
        print(f"Voice {self.voice_index} state reset.")

    @modal.method()
    def ping(self) -> str:
        """Health check — confirms container is warm and ready."""
        import jax
        return f"Voice {self.voice_index} alive on {jax.devices()[0]}"


# ─────────────────────────────────────────────────────────────────────────────
# LOCAL ENTRYPOINT — use this for warm-up instead of the Python client:
#   modal run server/magenta_server.py
#   modal run server/magenta_server.py --n-voices 1
# ─────────────────────────────────────────────────────────────────────────────

@app.local_entrypoint()
async def warmup(n_voices: int = 3):
    """Ping all voice containers in parallel to force cold-start + XLA compile."""
    import asyncio
    import time

    async def ping_one(i: int):
        t0 = time.perf_counter()
        print(f"[warmup] Voice {i}: pinging...")
        result = await VoiceServer(voice_index=i).ping.remote.aio()
        print(f"[warmup] Voice {i}: ready in {time.perf_counter() - t0:.0f}s — {result}")

    print(f"[warmup] Firing {n_voices} container(s) in parallel "
          f"(cold start + XLA compile takes ~5-8 min on first run)...")
    await asyncio.gather(*[ping_one(i) for i in range(n_voices)])
    print("[warmup] All containers warm. Run improv_loop.py within 10 min.")

