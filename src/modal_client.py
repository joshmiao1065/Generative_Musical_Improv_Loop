"""
modal_client.py — ThinkPad-side async client for the Magenta RT Modal server
=============================================================================
Calls the 3 VoiceServer containers in parallel (one per AI voice).
Each voice generates independently on its own A100. Wall time ≈ max of all
three, not their sum — fitting within the 8-second buffer pass with ~2.4s
headroom at 120 BPM, 16 beats.

PREREQUISITE: Deploy the server first (one-time, survives terminal close):
  modal deploy modal/magenta_server.py

USAGE (from improv_loop.py or standalone):
  import asyncio
  from src.modal_client import MagentaRTClient

  client = MagentaRTClient()
  asyncio.run(client.ping_all())               # confirm containers are warm

  prev = None
  while True:
      user_loop_np = capture_user_loop()        # (N, 2) float32 48kHz stereo
      voice_outputs = asyncio.run(
          client.generate_pass(user_loop_np, prev_voice_outputs=prev)
      )
      prev = voice_outputs
      play_mix(user_loop_np, voice_outputs)

PARALLEL ARCHITECTURE
─────────────────────
  Pass N playing (8s)  ──▶  Voice 0: hears user_loop
                        ──▶  Voice 1: hears user_loop + V0_prev_pass
                        ──▶  Voice 2: hears user_loop + V0_prev + V1_prev
                             ↓ all generate in parallel (~5.6s on A100)
  Pass N+1 ready       ◀──  outputs arrive with ~2.4s headroom

One-pass lag is intentional: each voice hears the *previous pass's* outputs.
"""

import asyncio
import io
import time
from typing import Optional

import numpy as np
import soundfile as sf

# ── Optional import guard for environments without modal installed ─────────────
try:
    import modal
    _MODAL_AVAILABLE = True
except ImportError:
    _MODAL_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

APP_NAME    = "magenta-rt-server"
CLASS_NAME  = "VoiceServer"
SAMPLE_RATE = 48000
N_VOICES    = 3


# ─────────────────────────────────────────────────────────────────────────────
# AUDIO UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def _np_to_wav_bytes(audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> bytes:
    """Encode (N, 2) float32 numpy array to 24-bit WAV bytes."""
    buf = io.BytesIO()
    sf.write(buf, audio, sample_rate, format="WAV", subtype="PCM_24")
    return buf.getvalue()


def _wav_bytes_to_np(wav_bytes: bytes, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Decode WAV bytes to (N, 2) float32 numpy array at SAMPLE_RATE."""
    import librosa
    audio, sr = librosa.load(io.BytesIO(wav_bytes), sr=sample_rate, mono=False)
    if audio.ndim == 1:
        audio = np.stack([audio, audio], axis=0)
    return audio.T.astype(np.float32)


def _silence_like(reference: np.ndarray) -> np.ndarray:
    """Return a zero array with the same shape and dtype as reference."""
    return np.zeros_like(reference)


# ─────────────────────────────────────────────────────────────────────────────
# CLIENT
# ─────────────────────────────────────────────────────────────────────────────

class MagentaRTClient:
    """
    Async client for the deployed Modal VoiceServer.

    Parameters
    ----------
    n_voices : int
        Number of AI voices to use (1–3). Defaults to 3.
    beats_per_loop : int
        Loop length in beats. Synced with live PBF4 value each pass.
    bpm : int
        Tempo. Synced with live PBF4 value each pass.
    guidance_weight : float
        CFG guidance strength [0–10]. Updated before each pass.
    temperature : float
        Sampling temperature [0–4]. Updated before each pass.
    topk : int
        Top-K sampling. Updated before each pass.
    model_feedback : float
        Model self-feedback [0–1]. Updated before each pass.
    """

    def __init__(
        self,
        n_voices: int = N_VOICES,
        beats_per_loop: int = 16,
        bpm: int = 120,
        guidance_weight: float = 1.5,
        temperature: float = 1.2,
        topk: int = 30,
        model_feedback: float = 0.95,
    ):
        if not _MODAL_AVAILABLE:
            raise ImportError(
                "modal package not installed. Run: pip install modal"
            )

        self.n_voices = n_voices
        self.beats_per_loop = beats_per_loop
        self.bpm = bpm
        self.guidance_weight = guidance_weight
        self.temperature = temperature
        self.topk = topk
        self.model_feedback = model_feedback

        # Look up the deployed VoiceServer class
        # This does NOT start containers — it just creates callable handles
        self._VoiceServer = modal.Cls.from_name(APP_NAME, CLASS_NAME)

        # One handle per voice. Voice containers are identified by voice_index.
        self._voices = [
            self._VoiceServer(voice_index=i) for i in range(n_voices)
        ]

        # Previous pass outputs, indexed by voice. None = first pass (use silence).
        self._prev_outputs: list[Optional[np.ndarray]] = [None] * n_voices

    # ── Health check ──────────────────────────────────────────────────────────

    async def ping_all(self) -> list[str]:
        """
        Confirm all voice containers are warm and responsive.
        Returns list of status strings from each container.
        Tip: call this ~10s before starting a session to pre-warm containers.
        """
        tasks = [voice.ping.remote.aio() for voice in self._voices]
        results = await asyncio.gather(*tasks)
        for i, result in enumerate(results):
            print(f"  Voice {i}: {result}")
        return list(results)

    # ── Main generation ───────────────────────────────────────────────────────

    async def generate_pass(
        self,
        user_loop_np: np.ndarray,
        *,
        # Live PBF4 knob values — pass current values each call
        beats_per_loop: Optional[int] = None,
        bpm: Optional[int] = None,
        guidance_weight: Optional[float] = None,
        temperature: Optional[float] = None,
        topk: Optional[int] = None,
        model_feedback: Optional[float] = None,
    ) -> list[np.ndarray]:
        """
        Generate one full loop pass for all voices simultaneously.

        Each voice runs on its own A100 container. All three fire at the same
        time and their results arrive together after ~5.6s. Call this once per
        loop pass while the *current* pass is playing back.

        Args:
            user_loop_np:   User loop audio as (N, 2) float32 numpy array at
                            48kHz. Should be exactly beats_per_loop beats long.

            beats_per_loop, bpm, guidance_weight, temperature, topk,
            model_feedback: Optional overrides for live PBF4 parameter updates.
                            If None, uses the values from the previous call or
                            the constructor defaults.

        Returns:
            List of N voice output arrays, each (M, 2) float32 at 48kHz where
            M ≈ user_loop_np.shape[0] (one full loop worth of generated audio).
            Voice 0 is at index 0, Voice 2 at index 2.

        Side effects:
            Saves returned outputs as self._prev_outputs for the next call's
            prior_mix computation.
        """
        # Update params if caller passed new values (from PBF4 MIDI thread)
        if beats_per_loop  is not None: self.beats_per_loop  = beats_per_loop
        if bpm             is not None: self.bpm             = bpm
        if guidance_weight is not None: self.guidance_weight = guidance_weight
        if temperature     is not None: self.temperature     = temperature
        if topk            is not None: self.topk            = topk
        if model_feedback  is not None: self.model_feedback  = model_feedback

        # Encode user loop once (same bytes sent to all voices)
        user_loop_bytes = _np_to_wav_bytes(user_loop_np)

        # Build prior_mix for each voice from the *previous pass's* outputs.
        # Voice 0: silence (hears only user loop — first in cascade)
        # Voice 1: Voice 0 prev output
        # Voice 2: Voice 0 prev + Voice 1 prev
        prior_mixes: list[np.ndarray] = []
        for i in range(self.n_voices):
            if i == 0 or self._prev_outputs[0] is None:
                # Voice 0 always hears only user loop; silence for prior_mix
                prior_mixes.append(_silence_like(user_loop_np))
            else:
                # Sum all previous voices' prior outputs
                mix = np.zeros_like(user_loop_np)
                for j in range(i):
                    prev = self._prev_outputs[j]
                    if prev is not None:
                        # Trim or pad prev to match user loop length
                        n = user_loop_np.shape[0]
                        if prev.shape[0] >= n:
                            mix += prev[:n]
                        else:
                            mix[:prev.shape[0]] += prev
                prior_mixes.append(mix)

        prior_mix_bytes = [_np_to_wav_bytes(m) for m in prior_mixes]

        # Dispatch all voices in parallel
        t0 = time.perf_counter()
        tasks = [
            self._voices[i].generate_pass.remote.aio(
                user_loop_bytes=user_loop_bytes,
                prior_mix_bytes=prior_mix_bytes[i],
                beats_per_loop=self.beats_per_loop,
                bpm=self.bpm,
                guidance_weight=self.guidance_weight,
                temperature=self.temperature,
                topk=self.topk,
                model_feedback=self.model_feedback,
            )
            for i in range(self.n_voices)
        ]
        raw_results = await asyncio.gather(*tasks)
        elapsed = time.perf_counter() - t0

        loop_sec = self.beats_per_loop * (60.0 / self.bpm)
        print(
            f"[MagentaRTClient] Pass complete: {elapsed:.2f}s wall time "
            f"({elapsed / loop_sec:.2f}× loop duration)"
        )

        # Decode WAV bytes → numpy arrays
        voice_outputs = [_wav_bytes_to_np(b) for b in raw_results]

        # Save for next call's prior_mix
        self._prev_outputs = voice_outputs

        return voice_outputs

    # ── Session management ────────────────────────────────────────────────────

    async def reset(self) -> None:
        """
        Reset all voice generation states and clear the prev_outputs cache.
        Call between jam sessions (new user, new key, new genre).
        """
        tasks = [voice.reset.remote.aio() for voice in self._voices]
        await asyncio.gather(*tasks)
        self._prev_outputs = [None] * self.n_voices
        print("[MagentaRTClient] All voices reset.")

    def reset_sync(self) -> None:
        """Synchronous wrapper for reset(), for use in non-async contexts."""
        asyncio.run(self.reset())

    # ── Convenience: sync generate for simple scripts ─────────────────────────

    def generate_pass_sync(
        self,
        user_loop_np: np.ndarray,
        **kwargs,
    ) -> list[np.ndarray]:
        """
        Synchronous wrapper for generate_pass().
        For use in simple scripts or interactive testing.
        Do NOT use inside an existing asyncio event loop.
        """
        return asyncio.run(self.generate_pass(user_loop_np, **kwargs))

    # ── Mix utility ───────────────────────────────────────────────────────────

    @staticmethod
    def mix(
        user_loop: np.ndarray,
        voice_outputs: list[np.ndarray],
        headroom: float = 0.95,
    ) -> np.ndarray:
        """
        Mix user loop + all voice outputs into a normalized stereo output.

        Args:
            user_loop:     (N, 2) float32 user loop audio.
            voice_outputs: list of (M, 2) float32 voice output arrays.
            headroom:      Peak normalization target [0–1].

        Returns:
            (N, 2) float32 mixed audio, peak-normalized to headroom.
        """
        n = user_loop.shape[0]
        mix = user_loop.copy()
        for vo in voice_outputs:
            # Trim or pad to user_loop length before mixing
            if vo.shape[0] >= n:
                mix += vo[:n]
            else:
                chunk = np.zeros_like(user_loop)
                chunk[:vo.shape[0]] = vo
                mix += chunk
        peak = np.max(np.abs(mix))
        if peak > 1e-6:
            mix = mix * (headroom / peak)
        return mix


# ─────────────────────────────────────────────────────────────────────────────
# STANDALONE TEST ENTRYPOINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    """
    Quick connectivity test — run from the repo root:
      python src/modal_client.py

    Expected output:
      Pinging 3 voice containers...
        Voice 0: Voice 0 alive on CudaDevice(id=0)
        Voice 1: Voice 1 alive on CudaDevice(id=0)
        Voice 2: Voice 2 alive on CudaDevice(id=0)
      All containers alive.

    If you see timeout errors, the containers are cold — wait 30s and retry,
    or check that you deployed with 'modal deploy modal/magenta_server.py'.
    """
    import sys

    async def _test():
        print("Pinging 3 voice containers...")
        client = MagentaRTClient(n_voices=3)
        try:
            results = await client.ping_all()
            if all(results):
                print("All containers alive.")
            else:
                print("WARNING: some containers did not respond.")
                sys.exit(1)
        except Exception as e:
            print(f"ERROR: {e}")
            print(
                "\nMake sure the server is deployed:\n"
                "  modal deploy modal/magenta_server.py\n"
            )
            sys.exit(1)

    asyncio.run(_test())
