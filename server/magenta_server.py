"""
magenta_server.py — Persistent parallel Modal server for real-time improv
=========================================================================
Three separate named classes (Voice0Server, Voice1Server, Voice2Server), each
with min_containers=1.  Containers start warming on `modal deploy` — no separate
prime step required.  By the time you launch improv_loop.py the model is already
loaded and JIT-compiled.

DEPLOY (containers warm immediately — takes ~5-8 min on first cold boot):
  modal deploy server/magenta_server.py

VERIFY warm (optional):
  modal run server/magenta_server.py          # pings all 3
  modal run server/magenta_server.py --n-voices 1

STOP billing:
  modal app stop magenta-rt-server

ARCHITECTURE
────────────
  ThinkPad client                 Modal (3 × A100, parallel)
  ───────────────                 ──────────────────────────
  Pass N playing (8s)   ───────▶  Voice0Server  hears user_loop
                         ───────▶  Voice1Server  hears user_loop + V0_prev_pass
                         ───────▶  Voice2Server  hears user_loop + V0_prev + V1_prev
                                  ↓ all generate simultaneously (~5.6s)
  Pass N+1 ready        ◀───────  outputs arrive with ~2.4s headroom

One-pass lag between voices is intentional.
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

GPU_TYPE     = "A100-40GB"   # 40GB has same compute as 80GB, wider availability on Modal
HF_CACHE_DIR = "/hf-cache"
APP_NAME     = "magenta-rt-server"

VOICE_STYLES = [
    "Electric guitar, expressive, melodic",
    "bass guitar",
    "EDM synthesized drum machine",
]

# ─────────────────────────────────────────────────────────────────────────────
# IMAGE
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

# Shared @app.cls kwargs — avoids repeating for each of the 3 voice classes.
# min_containers=1: Modal keeps one container allocated at all times after deploy.
# This eliminates cold starts during sessions entirely.
_CLS_KWARGS = dict(
    gpu=GPU_TYPE,
    min_containers=1,
    scaledown_window=600,
    timeout=600,
    env={
        "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.85",
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        "HF_HOME": HF_CACHE_DIR,
    },
)


# ─────────────────────────────────────────────────────────────────────────────
# SHARED IMPLEMENTATION
# Module-level functions that contain the real logic. Each voice class is a
# thin wrapper that sets VOICE_INDEX and delegates here. This keeps the three
# classes in sync without code duplication.
# ─────────────────────────────────────────────────────────────────────────────

def _impl_load(self) -> None:
    import jax
    import time as _time

    t_start = _time.perf_counter()

    sys.path.insert(0, "/root/src")
    from magenta_rt import spectrostream
    from magenta_backend import (
        AIVoice, GenerationParams, MagentaRTCFGTied,
        CHUNK_SAMPLES, SAMPLE_RATE,
    )

    self.CHUNK_SAMPLES = CHUNK_SAMPLES
    self.SAMPLE_RATE   = SAMPLE_RATE

    style = VOICE_STYLES[self.VOICE_INDEX]
    devices = jax.devices()
    print(f"[Voice {self.VOICE_INDEX}] Container started. GPU: {devices}. Style: '{style}'")

    print(f"[Voice {self.VOICE_INDEX}] Loading SpectroStream model...")
    t1 = _time.perf_counter()
    self.ss_model = spectrostream.SpectroStreamJAX(lazy=False)
    print(f"[Voice {self.VOICE_INDEX}] SpectroStream loaded in {_time.perf_counter()-t1:.1f}s")

    print(f"[Voice {self.VOICE_INDEX}] Loading MagentaRT large model weights...")
    t2 = _time.perf_counter()
    self.model = MagentaRTCFGTied(tag="large", lazy=False)
    print(f"[Voice {self.VOICE_INDEX}] MagentaRT loaded in {_time.perf_counter()-t2:.1f}s")

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
    self._style_cache: dict = {}

    print(f"[Voice {self.VOICE_INDEX}] Running JIT warm-up (XLA compile, first time ~2-4 min)...")
    t3 = _time.perf_counter()
    dummy = np.zeros((CHUNK_SAMPLES, 2), dtype=np.float32)
    self.voice.step(dummy)
    print(f"[Voice {self.VOICE_INDEX}] JIT compile done in {_time.perf_counter()-t3:.1f}s")

    total = _time.perf_counter() - t_start
    print(f"[Voice {self.VOICE_INDEX}] *** READY *** ({style}) — total startup {total:.1f}s")


def _impl_generate_pass(
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
    import soundfile as sf

    SR = self.SAMPLE_RATE
    CS = self.CHUNK_SAMPLES

    def _load(wav_bytes: bytes) -> np.ndarray:
        audio, _sr = sf.read(io.BytesIO(wav_bytes), dtype="float32", always_2d=True)
        if audio.shape[1] == 1:
            audio = np.concatenate([audio, audio], axis=1)
        return audio

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

    self.params.guidance_weight = guidance_weight
    self.params.temperature     = temperature
    self.params.topk            = topk
    self.params.model_feedback  = model_feedback
    self.params.beats_per_loop  = beats_per_loop
    self.params.bpm             = bpm

    if genres and genre_weights:
        active = [
            (g, w) for g, w in zip(genres, genre_weights)
            if g and w > 1e-4
        ]
        if active:
            active_genres, active_weights = zip(*active)
            ws = np.array(active_weights, dtype=np.float64)
            ws /= ws.sum()

            embeddings = []
            for g in active_genres:
                prompt = f"{g} {instrument}"
                if prompt not in self._style_cache:
                    print(f"Voice {self.VOICE_INDEX}: computing style embedding for '{prompt}'")
                    self._style_cache[prompt] = self.model.embed_style(prompt)
                embeddings.append(self._style_cache[prompt])

            if len(embeddings) == 1:
                blended = embeddings[0]
            else:
                blended = sum(float(w) * e for w, e in zip(ws, embeddings))

            self.voice.style_embedding = blended
            blend_desc = " + ".join(
                f"{g}({w:.0%})" for g, w in zip(active_genres, ws)
            )
            print(f"Voice {self.VOICE_INDEX}: style = {blend_desc} {instrument}")

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
    print(f"Voice {self.VOICE_INDEX}: {elapsed:.2f}s to generate "
          f"{audio_generated:.1f}s audio (RTF {audio_generated/elapsed:.2f}×)")

    full_output = np.concatenate(chunk_outputs, axis=0)

    buf = io.BytesIO()
    sf.write(buf, full_output, SR, format="WAV", subtype="PCM_24")
    return buf.getvalue()


def _impl_ping(self) -> str:
    import jax
    return f"Voice {self.VOICE_INDEX} alive on {jax.devices()[0]}"


def _impl_reset(self) -> None:
    self.voice.reset()
    print(f"Voice {self.VOICE_INDEX} state reset.")


def _impl_prime(self) -> str:
    dummy = np.zeros((self.CHUNK_SAMPLES, 2), dtype=np.float32)
    self.voice.step(dummy)
    return f"Voice {self.VOICE_INDEX} primed."


# ─────────────────────────────────────────────────────────────────────────────
# VOICE CLASSES
# Three separate named classes so each can have min_containers=1.
# Using modal.parameter() is incompatible with min_containers > 0 (Modal
# limitation) — hence three explicit classes instead of one parameterized one.
# ─────────────────────────────────────────────────────────────────────────────

@app.cls(**_CLS_KWARGS)
class Voice0Server:
    VOICE_INDEX = 0

    @modal.enter()
    def load(self) -> None:
        _impl_load(self)

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
        return _impl_generate_pass(
            self, user_loop_bytes, prior_mix_bytes,
            beats_per_loop, bpm, guidance_weight, temperature, topk,
            model_feedback, genres, instrument, genre_weights,
        )

    @modal.method()
    def ping(self) -> str:
        return _impl_ping(self)

    @modal.method()
    def reset(self) -> None:
        _impl_reset(self)

    @modal.method()
    def prime(self) -> str:
        return _impl_prime(self)


@app.cls(**_CLS_KWARGS)
class Voice1Server:
    VOICE_INDEX = 1

    @modal.enter()
    def load(self) -> None:
        _impl_load(self)

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
        return _impl_generate_pass(
            self, user_loop_bytes, prior_mix_bytes,
            beats_per_loop, bpm, guidance_weight, temperature, topk,
            model_feedback, genres, instrument, genre_weights,
        )

    @modal.method()
    def ping(self) -> str:
        return _impl_ping(self)

    @modal.method()
    def reset(self) -> None:
        _impl_reset(self)

    @modal.method()
    def prime(self) -> str:
        return _impl_prime(self)


@app.cls(**_CLS_KWARGS)
class Voice2Server:
    VOICE_INDEX = 2

    @modal.enter()
    def load(self) -> None:
        _impl_load(self)

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
        return _impl_generate_pass(
            self, user_loop_bytes, prior_mix_bytes,
            beats_per_loop, bpm, guidance_weight, temperature, topk,
            model_feedback, genres, instrument, genre_weights,
        )

    @modal.method()
    def ping(self) -> str:
        return _impl_ping(self)

    @modal.method()
    def reset(self) -> None:
        _impl_reset(self)

    @modal.method()
    def prime(self) -> str:
        return _impl_prime(self)


# ─────────────────────────────────────────────────────────────────────────────
# LOCAL ENTRYPOINT — verify all containers are warm after deploy:
#   modal run server/magenta_server.py
#   modal run server/magenta_server.py --n-voices 1
# ─────────────────────────────────────────────────────────────────────────────

_SERVERS = [Voice0Server, Voice1Server, Voice2Server]

@app.local_entrypoint()
async def warmup(n_voices: int = 3):
    """Ping all voice containers to verify they're warm after deploy."""
    import asyncio
    import time

    async def ping_one(i: int):
        t0 = time.perf_counter()
        print(f"[warmup] Voice {i}: pinging...")
        result = await _SERVERS[i]().ping.remote.aio()
        print(f"[warmup] Voice {i}: ready in {time.perf_counter() - t0:.0f}s — {result}")

    print(f"[warmup] Pinging {n_voices} container(s) in parallel "
          f"(with min_containers=1 they should respond in seconds if already warm)...")
    await asyncio.gather(*[ping_one(i) for i in range(n_voices)])
    print("[warmup] All containers confirmed warm.")
