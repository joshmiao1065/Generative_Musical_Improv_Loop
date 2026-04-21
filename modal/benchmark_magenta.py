"""
benchmark_magenta.py — Magenta RT GPU Benchmark on Modal
=========================================================
Measures whether Modal A100 GPU inference is fast enough for the
real-time improv loop. Tests the production AIVoice/cascade path,
not a toy proxy.

Key threshold: can N chunks be generated within one loop duration
(the buffer pass window)? If yes, Modal is viable. If no, Colab TPU
remains the only option.

USAGE
─────
  # First time: builds image, downloads model (~3.2 GB). Allow 15–20 min.
  modal run modal/benchmark_magenta.py

  # Subsequent runs use cached image + HF weights. Allow 5–10 min.
  modal run modal/benchmark_magenta.py

  # Test a cheaper GPU first:
  modal run modal/benchmark_magenta.py --gpu A10G

OUTPUT
──────
  Prints results live. Saves modal/benchmark_results.json locally.

INSTALL (once, in your virtualenv)
───────────────────────────────────
  pip install modal
  modal setup   # authenticates your account
"""

import json
import sys
import time
from pathlib import Path
from typing import Any

import modal

# ─────────────────────────────────────────────────────────────────────────────
# BENCHMARK CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

# Primary GPU to benchmark. Override via CLI: modal run ... --gpu A10G or A100-80GB
GPU_TYPE = "A100-80GB"

N_WARMUP_CHUNKS = 2   # Discarded: allow JAX JIT to compile before timing
N_TIMED_CHUNKS  = 5   # Timed single-chunk calls for mean/std

# Loop configs matching the user's improv session scenarios.
# The benchmark answers: does generation fit inside the buffer pass window?
LOOP_CONFIGS = [
    {"label": "8 beats  @ 120 BPM (4.0s)",  "beats": 8,  "bpm": 120},
    {"label": "16 beats @ 120 BPM (8.0s)",  "beats": 16, "bpm": 120},  # current POC config
    {"label": "8 beats  @  90 BPM (5.3s)",  "beats": 8,  "bpm": 90},
]

# Voice style prompts replicating poc_cascade.py
VOICE_STYLES = [
    "jazz piano solo, expressive, melodic improvisation",
    "upright bass, walking bass line, jazz",
    "jazz drum kit, brushed snare, ride cymbal",
]

HF_CACHE_DIR = "/hf-cache"

# ─────────────────────────────────────────────────────────────────────────────
# MODAL IMAGE
# Follows the official magenta-realtime Dockerfile exactly:
#   - Ubuntu 24.04 + CUDA 12.6
#   - T5X at pinned commit 7781d167 (patches apply cleanly against this snapshot)
#   - Three mandatory patches from magenta-realtime/patch/
# ─────────────────────────────────────────────────────────────────────────────

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.6.1-cudnn-runtime-ubuntu24.04",
        add_python="3.12",
    )
    .apt_install("git", "build-essential", "patch", "libsndfile1")
    .run_commands(
        # Clone both repos — magenta-realtime provides the patch files
        "git clone --depth 1 https://github.com/magenta/magenta-realtime.git /magenta-realtime",
        "git clone        https://github.com/google-research/t5x.git          /t5x",
        "cd /t5x && git checkout 7781d167ab421dae96281860c09d5bd785983853",
    )
    .run_commands(
        # Patch 1: t5x/setup.py — remove pinned fasttext/pysimdjson that fail on Python 3.12
        "cd /t5x && patch setup.py < /magenta-realtime/patch/t5x_setup.py.patch",
        # Install t5x BASE only — no [gpu]/[tpu] extras.
        # Those extras pull fasttext (NLP eval tool) which fails to compile on this image.
        # Magenta RT never uses fasttext; it only needs core t5x inference code.
        "cd /t5x && pip install -e '.'",
        # Install JAX with CUDA 12 explicitly (what [gpu] extra would have done).
        "pip install 'jax[cuda12]'",
        # Install magenta-realtime BASE only — jax + t5x are already present above.
        "pip install -e '/magenta-realtime/'",
        # Pin tf2jax per official Dockerfile; add audio deps.
        "pip install tf2jax==0.3.8 soundfile librosa resampy huggingface_hub",
    )
    .run_commands(
        # Patch 2: t5x/partitioning.py — GPU mesh routing (REQUIRED: without this
        #          the partitioner calls TPU-only bounds_from_last_device() and crashes).
        #          Use the known clone path — editable installs leave __file__ as None.
        "patch /t5x/t5x/partitioning.py < /magenta-realtime/patch/t5x_partitioning.py.patch",
        # Patch 3: seqio/vocabularies.py — removes unused tensorflow_text import.
        #          seqio is a regular (non-editable) install; use pip show for its location.
        'SEQIO=$(pip show seqio | grep Location | cut -d" " -f2) && '
        'patch "$SEQIO/seqio/vocabularies.py" < /magenta-realtime/patch/seqio_vocabularies.py.patch',
    )
    # Bake src/magenta_backend.py into the image so AIVoice / MagentaRTCFGTied
    # are available without duplicating code. copy_local_dir runs at build time.
    .add_local_dir(
        str(Path(__file__).parent.parent / "src"),
        remote_path="/root/src",
        copy=True,
    )
    .env({
        "HF_HOME":                       HF_CACHE_DIR,
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",  # JAX allocates on demand; reveals true peak VRAM
    })
)

# Persist HuggingFace model weights (~11 GB total) between runs.
# First run downloads; all subsequent runs load from this volume (~30s vs ~10 min).
hf_cache_vol = modal.Volume.from_name("magenta-rt-hf-cache", create_if_missing=True)

app = modal.App(
    "magenta-rt-benchmark",
    image=image,
    volumes={HF_CACHE_DIR: hf_cache_vol},
)


# ─────────────────────────────────────────────────────────────────────────────
# BENCHMARK CLASS
# ─────────────────────────────────────────────────────────────────────────────

@app.cls(
    gpu=GPU_TYPE,
    timeout=2400,       # 40 min: covers image-warm first run + full benchmark
    min_containers=0,   # benchmark is one-shot; no reason to keep warm after
    scaledown_window=120,
)
class MagentaRTBenchmark:
    """
    Loads Magenta RT once via @modal.enter(), then runs four timed tests.
    The model stays in GPU VRAM for the lifetime of this container.
    """

    @modal.enter()
    def setup(self) -> None:
        """
        Model load + JAX JIT warm-up. Runs once when the container starts.
        Load time is reported as 'cold start overhead'; everything after is
        amortized across calls for a persistent deployment.
        """
        import jax
        import numpy as np

        sys.path.insert(0, "/root/src")
        from magenta_rt import spectrostream
        from magenta_backend import (
            AIVoice,
            GenerationParams,
            MagentaRTCFGTied,
            CHUNK_SAMPLES,
            SAMPLE_RATE,
        )

        self.AIVoice        = AIVoice
        self.GenerationParams = GenerationParams
        self.CHUNK_SAMPLES  = CHUNK_SAMPLES
        self.SAMPLE_RATE    = SAMPLE_RATE
        self.CHUNK_SECS     = CHUNK_SAMPLES / SAMPLE_RATE  # 2.0

        print(f"JAX devices : {jax.devices()}")
        print(f"JAX version : {jax.__version__}")

        print("Loading SpectroStream encoder...")
        t0 = time.perf_counter()
        self.ss_model = spectrostream.SpectroStreamJAX(lazy=False)
        print(f"  SpectroStream: {time.perf_counter() - t0:.1f}s")

        print("Loading Magenta RT (large)...")
        t0 = time.perf_counter()
        self.model = MagentaRTCFGTied(tag="large", lazy=False)
        self.model_load_sec = time.perf_counter() - t0
        print(f"  Model loaded: {self.model_load_sec:.1f}s")

        self.default_params = GenerationParams(
            guidance_weight=1.5,
            temperature=1.2,
            topk=30,
            model_feedback=0.95,
            model_volume=0.85,
            beats_per_loop=16,
            bpm=120,
        )

        # Warm up JAX JIT with N_WARMUP_CHUNKS before any timed measurement.
        # The first call triggers XLA compilation; subsequent calls are stable.
        print(f"JAX JIT warm-up ({N_WARMUP_CHUNKS} chunks, results discarded)...")
        warmup_voice = AIVoice(self.model, self.ss_model, "jazz piano", self.default_params)
        dummy = np.zeros((CHUNK_SAMPLES, 2), dtype=np.float32)
        self.warmup_times = []
        for i in range(N_WARMUP_CHUNKS):
            t0 = time.perf_counter()
            warmup_voice.step(dummy)
            elapsed = time.perf_counter() - t0
            self.warmup_times.append(elapsed)
            print(f"  Warmup {i+1}/{N_WARMUP_CHUNKS}: {elapsed:.2f}s")

        print("Container ready.\n")

    # ─────────────────────────────────────────────────────────────────────────
    # Test 1: Single-chunk latency
    # ─────────────────────────────────────────────────────────────────────────

    def test_1_single_chunk(self) -> dict[str, Any]:
        """
        Baseline: one AIVoice, one chunk at a time, N_TIMED_CHUNKS iterations.
        Audio input is a sine tone (non-silent) to exercise the full injection path.
        """
        import numpy as np

        voice = self.AIVoice(self.model, self.ss_model, VOICE_STYLES[0], self.default_params)
        t_arr = np.linspace(0, self.CHUNK_SECS, self.CHUNK_SAMPLES, dtype=np.float32)
        audio_in = (0.1 * np.sin(2 * np.pi * 220 * t_arr))[:, None]
        audio_in = np.concatenate([audio_in, audio_in], axis=1)  # (N, 2)

        times = []
        for _ in range(N_TIMED_CHUNKS):
            t0 = time.perf_counter()
            voice.step(audio_in)
            times.append(time.perf_counter() - t0)

        mean_t = float(np.mean(times))
        return {
            "chunk_secs":          self.CHUNK_SECS,
            "times_sec":           times,
            "mean_sec":            mean_t,
            "std_sec":             float(np.std(times)),
            "min_sec":             float(np.min(times)),
            "max_sec":             float(np.max(times)),
            "real_time_factor":    self.CHUNK_SECS / mean_t,
            "faster_than_rt":      mean_t < self.CHUNK_SECS,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Test 2: Sequential pass — single voice, full loop
    # ─────────────────────────────────────────────────────────────────────────

    def test_2_sequential_pass(self) -> dict[str, Any]:
        """
        For each loop config: generate exactly chunks_per_pass chunks sequentially.
        Answers: does generation fit within the buffer pass window for voice 1 alone?
        """
        import numpy as np

        results = {}
        for cfg in LOOP_CONFIGS:
            loop_sec        = cfg["beats"] * (60.0 / cfg["bpm"])
            chunks_per_pass = max(1, round(loop_sec * self.SAMPLE_RATE / self.CHUNK_SAMPLES))

            params = self.GenerationParams(
                guidance_weight=self.default_params.guidance_weight,
                temperature=self.default_params.temperature,
                topk=self.default_params.topk,
                model_feedback=self.default_params.model_feedback,
                model_volume=self.default_params.model_volume,
                beats_per_loop=cfg["beats"],
                bpm=cfg["bpm"],
            )
            voice = self.AIVoice(self.model, self.ss_model, VOICE_STYLES[0], params)

            t_arr = np.linspace(0, self.CHUNK_SECS, self.CHUNK_SAMPLES, dtype=np.float32)
            audio_in = (0.1 * np.sin(2 * np.pi * 220 * t_arr))[:, None]
            audio_in = np.concatenate([audio_in, audio_in], axis=1)

            chunk_times = []
            t_pass = time.perf_counter()
            for _ in range(chunks_per_pass):
                t0 = time.perf_counter()
                voice.step(audio_in)
                chunk_times.append(time.perf_counter() - t0)
            total = time.perf_counter() - t_pass

            results[cfg["label"]] = {
                "loop_sec":          loop_sec,
                "chunks_per_pass":   chunks_per_pass,
                "chunk_times_sec":   chunk_times,
                "total_sec":         total,
                "gen_loop_ratio":    total / loop_sec,
                "feasible":          total <= loop_sec,
            }
        return results

    # ─────────────────────────────────────────────────────────────────────────
    # Test 3: Cascade simulation — 3 voices, sequential and threaded
    # ─────────────────────────────────────────────────────────────────────────

    def test_3_cascade(self) -> dict[str, Any]:
        """
        Simulates the actual poc_cascade.py loop for the user's primary config
        (16 beats @ 120 BPM).

        Sequential: voice1 → voice2 → voice3 per chunk step (exact POC behavior).
        Threaded:   all 3 voices generate the same step in parallel threads.
                    JAX on a single GPU will likely serialize internally, but this
                    measures whether thread-level dispatch reduces wall time.

        The cascade uses real cumulative audio mixing (voice N hears voices 1..N-1),
        replicating the actual audio injection path.
        """
        import concurrent.futures
        import numpy as np

        TARGET_BEATS = 16
        TARGET_BPM   = 120
        loop_sec        = TARGET_BEATS * (60.0 / TARGET_BPM)
        chunks_per_pass = max(1, round(loop_sec * self.SAMPLE_RATE / self.CHUNK_SAMPLES))

        t_arr    = np.linspace(0, self.CHUNK_SECS, self.CHUNK_SAMPLES, dtype=np.float32)
        user_buf = (0.1 * np.sin(2 * np.pi * 220 * t_arr))[:, None]
        user_buf = np.concatenate([user_buf, user_buf], axis=1)

        def _make_voices():
            return [
                self.AIVoice(self.model, self.ss_model, style, self.default_params)
                for style in VOICE_STYLES
            ]

        def _pad(arr: np.ndarray) -> np.ndarray:
            """Zero-pad voice output back to CHUNK_SAMPLES (crossfade shortens it)."""
            out = np.zeros((self.CHUNK_SAMPLES, 2), dtype=np.float32)
            out[: len(arr)] = arr
            return out

        # ── Sequential cascade ────────────────────────────────────────────────
        voices_seq = _make_voices()
        seq_chunk_times: list[list[float]] = [[] for _ in range(len(VOICE_STYLES))]
        t_seq = time.perf_counter()
        for _step in range(chunks_per_pass):
            cum = user_buf.copy()
            for vi, voice in enumerate(voices_seq):
                t0 = time.perf_counter()
                out = voice.step(cum)
                seq_chunk_times[vi].append(time.perf_counter() - t0)
                cum = cum + _pad(out)
        total_seq = time.perf_counter() - t_seq

        # ── Threaded cascade (voices share the GPU, each in its own thread) ───
        # Each voice still processes the *same* chunk step in parallel; the
        # cumulative mix fed to each voice is approximated (can't use sequential
        # output of earlier voices when running in parallel — represents a true
        # parallel deployment where each voice runs independently).
        voices_thr = _make_voices()
        t_thr = time.perf_counter()
        for _step in range(chunks_per_pass):
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(VOICE_STYLES)) as ex:
                futs = [ex.submit(v.step, user_buf.copy()) for v in voices_thr]
                [f.result() for f in futs]
        total_thr = time.perf_counter() - t_thr

        n_voices     = len(VOICE_STYLES)
        total_gen    = n_voices * chunks_per_pass * self.CHUNK_SECS

        return {
            "target":               "16 beats @ 120 BPM (8.0s)",
            "n_voices":             n_voices,
            "chunks_per_pass":      chunks_per_pass,
            "loop_sec":             loop_sec,
            "total_audio_sec":      total_gen,
            "sequential": {
                "total_sec":        total_seq,
                "gen_loop_ratio":   total_seq / loop_sec,
                "feasible":         total_seq <= loop_sec,
                "per_voice_mean":   [float(sum(t)/len(t)) for t in seq_chunk_times],
            },
            "threaded": {
                "total_sec":        total_thr,
                "gen_loop_ratio":   total_thr / loop_sec,
                "feasible":         total_thr <= loop_sec,
                "speedup_vs_seq":   total_seq / total_thr,
            },
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Test 4: N-chunk parallel lookahead — overlapping generation strategy
    # ─────────────────────────────────────────────────────────────────────────

    def test_4_lookahead(self) -> dict[str, Any]:
        """
        In the live system, generation can overlap with playback: while the user
        hears pass N, pass N+1 is being generated. The only hard constraint is
        that pass N+1 must be FULLY generated before the start of pass N+1 playback.

        This test runs 3 back-to-back passes and checks the steady-state rhythm:
        if generation time < loop duration, the pipeline stays ahead of playback.
        If generation time > loop duration, it falls behind (glitches/silence).
        """
        import numpy as np

        TARGET_BEATS = 16
        TARGET_BPM   = 120
        loop_sec        = TARGET_BEATS * (60.0 / TARGET_BPM)
        chunks_per_pass = max(1, round(loop_sec * self.SAMPLE_RATE / self.CHUNK_SAMPLES))
        N_PASSES        = 3

        t_arr    = np.linspace(0, self.CHUNK_SECS, self.CHUNK_SAMPLES, dtype=np.float32)
        audio_in = (0.1 * np.sin(2 * np.pi * 220 * t_arr))[:, None]
        audio_in = np.concatenate([audio_in, audio_in], axis=1)
        voice = self.AIVoice(self.model, self.ss_model, VOICE_STYLES[0], self.default_params)

        pass_times = []
        for _p in range(N_PASSES):
            t0 = time.perf_counter()
            for _c in range(chunks_per_pass):
                voice.step(audio_in)
            pass_times.append(time.perf_counter() - t0)

        return {
            "loop_sec":        loop_sec,
            "chunks_per_pass": chunks_per_pass,
            "n_passes":        N_PASSES,
            "pass_times_sec":  pass_times,
            "mean_pass_sec":   float(sum(pass_times) / len(pass_times)),
            "max_pass_sec":    max(pass_times),
            "gen_loop_ratio":  max(pass_times) / loop_sec,
            "sustainable_rt":  max(pass_times) <= loop_sec,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Master runner
    # ─────────────────────────────────────────────────────────────────────────

    @modal.method()
    def run(self) -> dict[str, Any]:
        import jax

        print("═" * 64)
        print("MAGENTA RT — MODAL GPU BENCHMARK")
        print("═" * 64)

        results: dict[str, Any] = {
            "gpu":              GPU_TYPE,
            "jax_version":      jax.__version__,
            "jax_devices":      str(jax.devices()),
            "model_load_sec":   self.model_load_sec,
            "jit_warmup_times": self.warmup_times,
        }

        _header("TEST 1  Single-chunk latency (1 voice, N=%d timed calls)" % N_TIMED_CHUNKS)
        r1 = self.test_1_single_chunk()
        results["test1_single_chunk"] = r1
        _print_t1(r1)

        _header("TEST 2  Sequential pass — single voice, full loop")
        r2 = self.test_2_sequential_pass()
        results["test2_sequential_pass"] = r2
        _print_t2(r2)

        _header("TEST 3  Cascade simulation — %d voices, sequential vs threaded" % len(VOICE_STYLES))
        r3 = self.test_3_cascade()
        results["test3_cascade"] = r3
        _print_t3(r3)

        _header("TEST 4  Lookahead feasibility — steady-state pass timing")
        r4 = self.test_4_lookahead()
        results["test4_lookahead"] = r4
        _print_t4(r4)

        return results


# ─────────────────────────────────────────────────────────────────────────────
# PRINT HELPERS
# (called inside the Modal container — output is streamed to the terminal)
# ─────────────────────────────────────────────────────────────────────────────

def _header(title: str) -> None:
    print(f"\n{'─'*64}")
    print(title)
    print("─" * 64)


def _print_t1(r: dict) -> None:
    rtf = r["real_time_factor"]
    sym = "✓" if r["faster_than_rt"] else "✗"
    print(f"  Generated / call :  {r['chunk_secs']:.1f}s audio")
    print(f"  Wall time (mean) :  {r['mean_sec']:.3f}s  ±{r['std_sec']:.3f}s")
    print(f"  Range            :  {r['min_sec']:.3f}s – {r['max_sec']:.3f}s")
    print(f"  Real-time factor :  {rtf:.3f}x  [{sym} {'faster' if rtf > 1 else 'SLOWER'} than real-time]")


def _print_t2(r: dict) -> None:
    for label, d in r.items():
        sym = "✓" if d["feasible"] else "✗"
        print(f"\n  {label}")
        print(f"    Chunks / pass    : {d['chunks_per_pass']}")
        print(f"    Generation time  : {d['total_sec']:.2f}s")
        print(f"    Loop duration    : {d['loop_sec']:.2f}s")
        print(f"    Ratio            : {d['gen_loop_ratio']:.2f}×  [{sym} {'fits' if d['feasible'] else 'EXCEEDS'} buffer pass]")


def _print_t3(r: dict) -> None:
    seq = r["sequential"]
    thr = r["threaded"]
    sym_s = "✓" if seq["feasible"] else "✗"
    sym_t = "✓" if thr["feasible"] else "✗"
    print(f"\n  Target : {r['target']}")
    print(f"  Voices : {r['n_voices']}  |  Chunks/pass : {r['chunks_per_pass']}")
    print(f"\n  Sequential (current POC behavior):")
    print(f"    Total time       : {seq['total_sec']:.2f}s")
    print(f"    Ratio            : {seq['gen_loop_ratio']:.2f}×  [{sym_s} {'fits' if seq['feasible'] else 'EXCEEDS'}]")
    print(f"    Per-voice means  : {[f'{t:.2f}s' for t in seq['per_voice_mean']]}")
    print(f"\n  Threaded (all voices generate simultaneously on same GPU):")
    print(f"    Total time       : {thr['total_sec']:.2f}s")
    print(f"    Ratio            : {thr['gen_loop_ratio']:.2f}×  [{sym_t} {'fits' if thr['feasible'] else 'EXCEEDS'}]")
    print(f"    Speedup vs seq   : {thr['speedup_vs_seq']:.2f}×")


def _print_t4(r: dict) -> None:
    sym = "✓" if r["sustainable_rt"] else "✗"
    print(f"\n  Loop duration    : {r['loop_sec']:.2f}s")
    print(f"  Pass times       : {[f'{t:.2f}s' for t in r['pass_times_sec']]}")
    print(f"  Mean / worst     : {r['mean_pass_sec']:.2f}s / {r['max_pass_sec']:.2f}s")
    print(f"  Sustainable RT   : [{sym} {'YES' if r['sustainable_rt'] else 'NO'}]  (worst-case ratio: {r['gen_loop_ratio']:.2f}×)")


def _print_verdict(results: dict) -> None:
    """Actionable go/no-go printed locally after receiving results."""
    t1  = results.get("test1_single_chunk", {})
    t2  = results.get("test2_sequential_pass", {})
    t3  = results.get("test3_cascade", {})
    t4  = results.get("test4_lookahead", {})

    rtf         = t1.get("real_time_factor", 0.0)
    target_key  = "16 beats @ 120 BPM (8.0s)"
    sp          = t2.get(target_key, {})
    cascade_seq = t3.get("sequential", {})
    cascade_thr = t3.get("threaded", {})

    print("\n" + "═" * 64)
    print("PROJECT VERDICT — 16 beats @ 120 BPM (current POC config)")
    print("═" * 64)

    print(f"\n  Single-chunk RTF     : {rtf:.3f}×   (>1.0 = faster than real-time)")
    if sp:
        print(f"  Single-voice pass    : {sp.get('gen_loop_ratio', 0):.2f}× loop  ({'✓ fits' if sp.get('feasible') else '✗ exceeds'} buffer pass)")
    if cascade_seq:
        print(f"  3-voice sequential   : {cascade_seq.get('gen_loop_ratio', 0):.2f}× loop  ({'✓' if cascade_seq.get('feasible') else '✗'})")
    if cascade_thr:
        print(f"  3-voice threaded     : {cascade_thr.get('gen_loop_ratio', 0):.2f}× loop  ({'✓' if cascade_thr.get('feasible') else '✗'})")
    if t4:
        print(f"  Sustained RT (1 v)   : {'✓ YES' if t4.get('sustainable_rt') else '✗ NO'}  (worst pass: {t4.get('max_pass_sec', 0):.2f}s)")

    print()
    sp_feasible      = sp.get("feasible", False) if sp else False
    thr_feasible     = cascade_thr.get("feasible", False) if cascade_thr else False
    sustained        = t4.get("sustainable_rt", False) if t4 else False

    if sp_feasible and sustained:
        print("  RECOMMENDATION: Modal A100 is VIABLE for this project.")
        print("  Even single-voice generation fits the buffer pass.")
        if thr_feasible:
            print("  3-voice threaded generation also fits — full cascade is feasible.")
        else:
            seq_ratio = cascade_seq.get("gen_loop_ratio", 0)
            print(f"  3-voice sequential does NOT fit ({seq_ratio:.1f}× loop).")
            print("  Use threaded or per-container parallelism for multi-voice.")
        print("\n  Next step: deploy modal/magenta_server.py (Phase B).")
    elif not sp_feasible and rtf > 0.3:
        print("  RECOMMENDATION: Modal is MARGINAL.")
        print("  Single-voice generation exceeds the buffer pass.")
        print("  Options:")
        print("    A) Lengthen loop (more beats) to widen the buffer window.")
        print("    B) Use base model (tag='base') — 1.3 GB vs 2.8 GB, likely 2× faster.")
        print("    C) Keep Colab TPU for inference; use Modal only for the FastAPI server.")
    else:
        print("  RECOMMENDATION: Modal is NOT viable at this performance level.")
        print("  Keep Colab TPU (free v2-8 TPU, ~1.25s/chunk) for inference.")
        print("  Modal can still be used as a persistent server wrapper around Colab.")

    print("═" * 64)


# ─────────────────────────────────────────────────────────────────────────────
# ENTRYPOINT
# ─────────────────────────────────────────────────────────────────────────────

@app.local_entrypoint()
def main() -> None:
    """
    Called when you run: modal run modal/benchmark_magenta.py
    Executes the full benchmark remotely and writes results locally.
    """
    print(f"Launching Magenta RT benchmark on Modal ({GPU_TYPE})...")
    print("First run: image build + model download adds ~15 min to total time.\n")

    benchmark = MagentaRTBenchmark()
    results   = benchmark.run.remote()

    out_path = Path(__file__).parent / "benchmark_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nFull results saved to: {out_path}")

    _print_verdict(results)
