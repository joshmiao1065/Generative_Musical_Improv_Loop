"""
magenta_backend.py — Clean Magenta RT backend for the Improv Loop project.

Implements:
  - GenerationParams: typed dataclass of all controllable model parameters
  - MagentaRTCFGTied: MagentaRT subclass with tied classifier-free guidance
  - AIVoice: single AI instrument voice with audio injection (Option B clean design)

Design principles:
  - No Colab UI dependencies (no ipywidgets, colab_utils, etc.)
  - No globals — all state is encapsulated in AIVoice instances
  - Swappable: AIVoice implements the AIInstrumentBackend contract (see CLAUDE.md Section 6)
  - The cascade logic lives in the caller (poc_cascade.py), not here

Source reference: Official Magenta RT Audio Injection notebook
  https://colab.research.google.com/github/magenta/magenta-realtime/blob/main/notebooks/Magenta_RT_Audio_Injection.ipynb
"""

import dataclasses
import warnings
from typing import Optional, Tuple

import jax
import numpy as np

from magenta_rt import audio as audio_lib
from magenta_rt import musiccoca
from magenta_rt import spectrostream
from magenta_rt import system
from magenta_rt import utils

SAMPLE_RATE: int = 48000
CHUNK_SECONDS: float = 2.0
CHUNK_SAMPLES: int = int(CHUNK_SECONDS * SAMPLE_RATE)

# Injection tuning constants (from official notebook)
_MIX_PREFIX_FRAMES: int = 16       # extra context frames prepended to mix window
_LEFT_EDGE_DROP: int = 8           # SpectroStream encoder left-edge artifact frames to discard


# ─────────────────────────────────────────────────────────────────────────────
# Public parameter dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclasses.dataclass
class GenerationParams:
    """
    All runtime-controllable generation parameters.
    Intended to be mutated by the MIDI controller thread and read by AIVoice.step().
    All values are validated on assignment via __post_init__.
    """
    guidance_weight: float = 1.5    # CFG strength [0.0–10.0]; higher = more prompt-adherent
    temperature: float = 1.2        # Sampling temperature [0.0–4.0]; higher = more random
    topk: int = 30                  # Top-K sampling [0–1024]; 0 = disabled
    model_feedback: float = 0.95    # How much AI output feeds back into its own context [0.0–1.0]
    model_volume: float = 0.85      # Output gain of this voice [0.0–1.0]
    beats_per_loop: int = 8         # Loop length in beats (used for metronome alignment)
    bpm: int = 120                  # Beats per minute

    def __post_init__(self):
        assert 0.0 <= self.guidance_weight <= 10.0, "guidance_weight out of range"
        assert 0.0 <= self.temperature <= 4.0, "temperature out of range"
        assert 0 <= self.topk <= 1024, "topk out of range"
        assert 0.0 <= self.model_feedback <= 1.0, "model_feedback out of range"
        assert 0.0 <= self.model_volume <= 1.0, "model_volume out of range"


# ─────────────────────────────────────────────────────────────────────────────
# Magenta RT model subclass with tied CFG
# ─────────────────────────────────────────────────────────────────────────────

class MagentaRTCFGTied(system.MagentaRTT5X):
    """
    Magenta RT with Tied Classifier-Free Guidance.

    Difference from base class: generate_chunk() receives both the original
    (clean) context tokens and the injection-mixed context tokens. The encoder
    sees both via CFG, allowing the model to be steered by the injected audio
    while retaining coherence with the clean signal.

    Adapted verbatim from the official audio injection notebook (no logic changes).
    See: https://github.com/magenta/magenta-realtime/blob/main/notebooks/Magenta_RT_Audio_Injection.ipynb
    """

    def generate_chunk(
        self,
        state: Optional[system.MagentaRTState] = None,
        style: Optional[musiccoca.StyleEmbedding] = None,
        seed: Optional[int] = None,
        **kwargs,
    ) -> Tuple[audio_lib.Waveform, system.MagentaRTState]:
        if state is None:
            state = self.init_state()
        if seed is None:
            seed = np.random.randint(0, 2**31)

        context_tokens = {
            "orig": kwargs.get("context_tokens_orig", state.context_tokens),
            "mix":  state.context_tokens,
        }
        codec_tokens_lm = {}
        for key, tokens in context_tokens.items():
            codec_tokens_lm[key] = np.where(
                tokens >= 0,
                utils.rvq_to_llm(
                    np.maximum(tokens, 0),
                    self.config.codec_rvq_codebook_size,
                    self.config.vocab_codec_offset,
                ),
                np.full_like(tokens, self.config.vocab_mask_token),
            )

        if style is None:
            style_tokens_lm = np.full(
                (self.config.encoder_style_rvq_depth,),
                self.config.vocab_mask_token,
                dtype=np.int32,
            )
        else:
            style_tokens = self.style_model.tokenize(style)[: self.config.encoder_style_rvq_depth]
            style_tokens_lm = utils.rvq_to_llm(
                style_tokens,
                self.config.style_rvq_codebook_size,
                self.config.vocab_style_offset,
            )

        batch_size, _, _ = self._device_params

        enc_pos = np.concatenate([
            codec_tokens_lm["mix"][:, : self.config.encoder_codec_rvq_depth].reshape(-1),
            style_tokens_lm,
        ])
        enc_neg = np.concatenate([
            codec_tokens_lm["orig"][:, : self.config.encoder_codec_rvq_depth].reshape(-1),
            style_tokens_lm,
        ])
        enc_neg[-self.config.encoder_style_rvq_depth :] = self.config.vocab_mask_token
        encoder_inputs = np.stack([enc_pos, enc_neg], axis=0)

        max_decode_frames = kwargs.get("max_decode_frames", self.config.chunk_length_frames)
        generated_tokens, _ = self._llm(
            {
                "encoder_input_tokens": encoder_inputs,
                "decoder_input_tokens": np.zeros(
                    (batch_size,
                     self.config.chunk_length_frames * self.config.decoder_codec_rvq_depth),
                    dtype=np.int32,
                ),
            },
            {
                "max_decode_steps": np.array(
                    max_decode_frames * self.config.decoder_codec_rvq_depth, dtype=np.int32
                ),
                "guidance_weight": kwargs.get("guidance_weight", self._guidance_weight),
                "temperature":     kwargs.get("temperature",     self._temperature),
                "topk":            kwargs.get("topk",            self._topk),
            },
            jax.random.PRNGKey(seed + state.chunk_index),
        )

        generated_tokens = np.array(generated_tokens)[:1].reshape(
            self.config.chunk_length_frames, self.config.decoder_codec_rvq_depth
        )[:max_decode_frames]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            generated_rvq = utils.llm_to_rvq(
                generated_tokens,
                self.config.codec_rvq_codebook_size,
                self.config.vocab_codec_offset,
                safe=False,
            )

        xfade_frames = state.context_tokens[-self.config.crossfade_length_frames :]
        if state.chunk_index == 0:
            xfade_frames = np.zeros_like(xfade_frames)
        waveform = self.codec.decode(np.concatenate([xfade_frames, generated_rvq], axis=0))
        state.update(generated_rvq, waveform[-self.config.crossfade_length_samples :])
        return waveform, state


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

class _AudioFade:
    """Crossfades successive audio chunks to eliminate boundary clicks."""

    def __init__(self, fade_samples: int, stereo: bool = True):
        self.size = fade_samples
        ramp = np.sin(np.linspace(0, np.pi / 2, fade_samples, dtype=np.float32)) ** 2
        self._ramp = ramp[:, None] if stereo else ramp
        shape = (fade_samples, 2) if stereo else (fade_samples,)
        self._prev = np.zeros(shape, dtype=np.float32)

    def __call__(self, chunk: np.ndarray) -> np.ndarray:
        """Apply crossfade. Returns chunk shortened by fade_samples on the right."""
        chunk[: self.size] = chunk[: self.size] * self._ramp + self._prev
        self._prev = chunk[-self.size :] * np.flip(self._ramp, axis=0)
        return chunk[: -self.size]

    def reset(self):
        self._prev[:] = 0.0


@dataclasses.dataclass
class _InjectionState:
    """Per-voice audio history buffers for audio injection."""
    context_tokens_orig: np.ndarray   # saved clean context tokens (for CFG)
    all_inputs: np.ndarray            # accumulated injection input audio (N, 2)
    all_outputs: np.ndarray           # accumulated model output audio (N, 2)
    step: int = -1


# ─────────────────────────────────────────────────────────────────────────────
# AIVoice — the core abstraction
# ─────────────────────────────────────────────────────────────────────────────

class AIVoice:
    """
    A single AI instrument voice using Magenta RT audio injection.

    Each voice maintains its own generation state and audio history.
    The caller (cascade loop) is responsible for:
      - Providing the correct cumulative mix as `input_chunk`
      - Collecting and mixing outputs from all voices

    The cascade pattern:
        voice_1_out = voice_1.step(user_loop_chunk)
        voice_2_out = voice_2.step(user_loop_chunk + voice_1_out_padded)
        ...

    Audio contract:
        input:  np.ndarray, shape (CHUNK_SAMPLES, 2), float32, 48 kHz stereo
        output: np.ndarray, shape (CHUNK_SAMPLES - fade_samples, 2), float32, 48 kHz stereo
                (slightly shorter due to crossfade; pad with zeros if needed)
    """

    def __init__(
        self,
        model: MagentaRTCFGTied,
        ss_model: spectrostream.SpectroStreamJAX,
        style: str,
        params: GenerationParams,
    ):
        """
        Args:
            model:    Loaded MagentaRTCFGTied instance (shared across all voices).
            ss_model: Loaded SpectroStreamJAX encoder (shared across all voices).
            style:    Text description of this instrument, e.g. "jazz piano solo".
            params:   GenerationParams instance (may be shared/mutated by MIDI thread).
        """
        self.model = model
        self.ss_model = ss_model
        self.params = params
        self.style_embedding = model.embed_style(style)

        cfg = model.config
        context_frames   = int(cfg.context_length * cfg.codec_frame_rate)
        context_samples  = int(cfg.context_length * SAMPLE_RATE)
        fade_samples     = int(cfg.codec_sample_rate * cfg.crossfade_length)
        frame_samples    = cfg.frame_length_samples
        crossfade_samples = int(cfg.crossfade_length * SAMPLE_RATE)

        # Window sizes for injection mix (derived from notebook constants)
        self._mix_samples = CHUNK_SAMPLES + _MIX_PREFIX_FRAMES * frame_samples
        self._io_offset   = CHUNK_SAMPLES - crossfade_samples
        self._max_frames  = round(CHUNK_SECONDS * cfg.codec_frame_rate)
        self._rvq_depth   = cfg.decoder_codec_rvq_depth

        self._state: Optional[system.MagentaRTState] = None
        self._inj = _InjectionState(
            context_tokens_orig=np.zeros((context_frames, self._rvq_depth), dtype=np.int32),
            all_inputs=np.zeros((context_samples, 2), dtype=np.float32),
            all_outputs=np.zeros((context_samples, 2), dtype=np.float32),
        )
        self._fade = _AudioFade(fade_samples, stereo=True)

    # ── public API ────────────────────────────────────────────────────────────

    def step(self, input_chunk: np.ndarray) -> np.ndarray:
        """
        Process one chunk of audio and return the model's generated response.

        The input_chunk should be the cumulative mix of all audio sources this
        voice is supposed to "hear" (user loop + any prior AI voices).

        Args:
            input_chunk: (CHUNK_SAMPLES, 2) float32 array at 48 kHz.
        Returns:
            output chunk: (CHUNK_SAMPLES - fade_samples, 2) float32 array at 48 kHz.
        """
        chunk = self._ensure_stereo(input_chunk)

        # 1. Accumulate input history
        self._inj.all_inputs = np.concatenate([self._inj.all_inputs, chunk], axis=0)

        # 2. Build mix: recent input + recent model output × feedback
        in_window  = self._inj.all_inputs[-(self._io_offset + self._mix_samples):-self._io_offset]
        out_window = self._inj.all_outputs[-self._mix_samples:]
        mix = in_window + out_window * self.params.model_feedback

        # 3. Encode mix to SpectroStream tokens, drop left-edge artifacts
        mix_waveform = audio_lib.Waveform(mix, sample_rate=SAMPLE_RATE)
        mix_tokens   = self.ss_model.encode(mix_waveform)[_LEFT_EDGE_DROP:]

        # 4. Inject tokens into model context
        if self._state is not None:
            self._inj.context_tokens_orig = self._state.context_tokens.copy()
            n = len(mix_tokens)
            self._state.context_tokens[-n:] = mix_tokens[:, : self._rvq_depth]

        # 5. Generate next 2-second chunk
        waveform, self._state = self.model.generate_chunk(
            state=self._state,
            style=self.style_embedding,
            seed=None,
            max_decode_frames=self._max_frames,
            context_tokens_orig=self._inj.context_tokens_orig,
            guidance_weight=self.params.guidance_weight,
            temperature=self.params.temperature,
            topk=self.params.topk,
        )

        # 6. Accumulate output history (pre-fade portion for accurate future injection)
        pre_fade = waveform.samples[self._fade.size :]
        self._inj.all_outputs = np.concatenate([self._inj.all_outputs, pre_fade], axis=0)

        # 7. Crossfade + volume scale
        output = self._fade(waveform.samples) * self.params.model_volume
        self._inj.step += 1
        return output

    def reset(self):
        """Reset generation state (call between sessions)."""
        self._state = None
        self._fade.reset()
        context_samples = self._inj.all_inputs.shape[0]
        context_frames  = self._inj.context_tokens_orig.shape[0]
        self._inj = _InjectionState(
            context_tokens_orig=np.zeros((context_frames, self._rvq_depth), dtype=np.int32),
            all_inputs=np.zeros((context_samples, 2), dtype=np.float32),
            all_outputs=np.zeros((context_samples, 2), dtype=np.float32),
        )

    # ── internal ──────────────────────────────────────────────────────────────

    @staticmethod
    def _ensure_stereo(audio: np.ndarray) -> np.ndarray:
        """Convert mono (N,) to stereo (N, 2) if needed."""
        if audio.ndim == 1:
            return np.stack([audio, audio], axis=-1).astype(np.float32)
        return audio.astype(np.float32)
