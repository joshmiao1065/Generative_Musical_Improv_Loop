"""
effects_chain.py — 3-band EQ + Freeverb reverb for the full output mix.

Applied by AudioMixer after summing all sources, before the peak limiter.
All parameter setters are called from MIDI/keyboard threads; process() runs
on the audio thread. Float assignments are GIL-atomic, so no extra locking
is needed.

EQ: RBJ Audio EQ Cookbook biquad filters (scipy.signal.sosfilt).
    One-pole gain smoother (τ=5ms) prevents clicks on fast knob sweeps.
    Filter state (zi) is reused across blocks for continuous phase response.
    Coefficients only recomputed when smoothed gain changes by > 0.01 dB.

Reverb: Freeverb algorithm (Jezar at Dreampoint, public domain).
    8 parallel feedback comb filters + 4 series allpass filters, stereo.
    Allpass buffers 2-4 are smaller than a 512-sample block — sub-blocked
    at wrap boundaries (see CLAUDE.md critical notice #10).

Knob mapping (PBF4):
    CC 32  EQ Bass   [0-127] → -12 to +12 dB low shelf  @ 250 Hz
    CC 33  EQ Mid    [0-127] → -12 to +12 dB peaking EQ @ 1 kHz Q=0.707
    CC 34  EQ Treble [0-127] → -12 to +12 dB high shelf @ 4 kHz
    CC 35  Reverb    [0-127] → wet mix 0.0-1.0 (0=dry, 1=large hall)
"""

import math
import numpy as np
from scipy.signal import sosfilt, sosfilt_zi, lfilter

FS = 48000
CHANNELS = 2

_SMOOTH_COEFF = math.exp(-1.0 / (0.005 * FS))   # τ = 5ms, per-sample coefficient
_GAIN_THRESHOLD_DB = 0.01


# ─────────────────────────────────────────────────────────────────────────────
# RBJ biquad filter coefficient builders
# ─────────────────────────────────────────────────────────────────────────────

def _low_shelf_sos(db: float, f0: float = 250.0, fs: int = FS) -> np.ndarray:
    A = 10 ** (db / 40.0)
    w0 = 2 * math.pi * f0 / fs
    alpha = math.sin(w0) / math.sqrt(2)
    cosw0, sqrtA = math.cos(w0), math.sqrt(A)
    b0 = A * ((A+1) - (A-1)*cosw0 + 2*sqrtA*alpha)
    b1 = 2*A * ((A-1) - (A+1)*cosw0)
    b2 = A * ((A+1) - (A-1)*cosw0 - 2*sqrtA*alpha)
    a0 = (A+1) + (A-1)*cosw0 + 2*sqrtA*alpha
    a1 = -2 * ((A-1) + (A+1)*cosw0)
    a2 = (A+1) + (A-1)*cosw0 - 2*sqrtA*alpha
    return np.array([[b0/a0, b1/a0, b2/a0, 1.0, a1/a0, a2/a0]])


def _high_shelf_sos(db: float, f0: float = 4000.0, fs: int = FS) -> np.ndarray:
    A = 10 ** (db / 40.0)
    w0 = 2 * math.pi * f0 / fs
    alpha = math.sin(w0) / math.sqrt(2)
    cosw0, sqrtA = math.cos(w0), math.sqrt(A)
    b0 = A * ((A+1) + (A-1)*cosw0 + 2*sqrtA*alpha)
    b1 = -2*A * ((A-1) + (A+1)*cosw0)
    b2 = A * ((A+1) + (A-1)*cosw0 - 2*sqrtA*alpha)
    a0 = (A+1) - (A-1)*cosw0 + 2*sqrtA*alpha
    a1 = 2 * ((A-1) - (A+1)*cosw0)
    a2 = (A+1) - (A-1)*cosw0 - 2*sqrtA*alpha
    return np.array([[b0/a0, b1/a0, b2/a0, 1.0, a1/a0, a2/a0]])


def _peak_eq_sos(db: float, f0: float = 1000.0, Q: float = 0.707, fs: int = FS) -> np.ndarray:
    A = 10 ** (db / 40.0)
    w0 = 2 * math.pi * f0 / fs
    alpha = math.sin(w0) / (2 * Q)
    cosw0 = math.cos(w0)
    b0 = 1 + alpha * A;  b1 = -2 * cosw0;  b2 = 1 - alpha * A
    a0 = 1 + alpha / A;  a1 = -2 * cosw0;  a2 = 1 - alpha / A
    return np.array([[b0/a0, b1/a0, b2/a0, 1.0, a1/a0, a2/a0]])


# ─────────────────────────────────────────────────────────────────────────────
# EQ band
# ─────────────────────────────────────────────────────────────────────────────

class _EQBand:
    """One RBJ biquad EQ band with one-pole gain smoothing and persistent zi state."""

    def __init__(self, kind: str, f0: float):
        self._kind = kind
        self._f0   = f0
        self._gain_target:   float = 0.0
        self._gain_smoothed: float = 0.0
        self._gain_applied:  float = 0.0
        self._sos = self._make_sos(0.0)
        zi = sosfilt_zi(self._sos)         # shape (1, 2) for a single-section filter
        self._zi_L = zi.copy()
        self._zi_R = zi.copy()

    def _make_sos(self, db: float) -> np.ndarray:
        if self._kind == "low_shelf":
            return _low_shelf_sos(db, self._f0)
        elif self._kind == "high_shelf":
            return _high_shelf_sos(db, self._f0)
        else:
            return _peak_eq_sos(db, self._f0)

    def set_gain_db(self, db: float) -> None:
        """Set gain target in dB (clamped to ±12). Called from MIDI thread."""
        self._gain_target = max(-12.0, min(12.0, float(db)))

    def process(self, audio: np.ndarray) -> np.ndarray:
        """
        Process stereo block (N, 2) float32 in-place and return it.

        Advances the one-pole smoother over the block length using the exact
        N-step formula: y_new = alpha^N * y_old + (1-alpha^N) * target.
        Recomputes filter coefficients only when smoothed gain changes by
        more than _GAIN_THRESHOLD_DB from the last-applied gain.
        """
        n = audio.shape[0]
        alpha_n = _SMOOTH_COEFF ** n
        self._gain_smoothed = alpha_n * self._gain_smoothed + (1.0 - alpha_n) * self._gain_target

        # Bypass if gain is essentially zero
        if abs(self._gain_smoothed) < _GAIN_THRESHOLD_DB and abs(self._gain_applied) < _GAIN_THRESHOLD_DB:
            return audio

        if abs(self._gain_smoothed - self._gain_applied) > _GAIN_THRESHOLD_DB:
            self._sos = self._make_sos(self._gain_smoothed)
            self._gain_applied = self._gain_smoothed

        out_L, self._zi_L = sosfilt(self._sos, audio[:, 0].astype(np.float64), zi=self._zi_L)
        out_R, self._zi_R = sosfilt(self._sos, audio[:, 1].astype(np.float64), zi=self._zi_R)
        audio[:, 0] = out_L.astype(np.float32)
        audio[:, 1] = out_R.astype(np.float32)
        return audio


# ─────────────────────────────────────────────────────────────────────────────
# Freeverb components
# ─────────────────────────────────────────────────────────────────────────────

_SCALE       = FS / 44100
_COMB_L      = [round(t * _SCALE) for t in [1116, 1188, 1277, 1356, 1422, 1491, 1557, 1617]]
_COMB_R      = [t + round(23 * _SCALE) for t in _COMB_L]
_AP_SIZES    = [round(t * _SCALE) for t in [556, 441, 341, 225]]
_AP_FEEDBACK = 0.5
_FIXED_GAIN  = 0.015


class _CombFilter:
    def __init__(self, size: int):
        self.buf      = np.zeros(size, dtype=np.float32)
        self.pos      = 0
        self.feedback = 0.84
        self.damp1    = 0.2
        self.damp2    = 0.8
        self._zi      = np.zeros(1, dtype=np.float64)

    def set_damp(self, val: float) -> None:
        self.damp1 = val
        self.damp2 = 1.0 - val

    def process(self, inp: np.ndarray) -> np.ndarray:
        n       = len(inp)
        buf_len = len(self.buf)
        end     = self.pos + n

        if end <= buf_len:
            buf_read = self.buf[self.pos:end].copy()
        else:
            split    = buf_len - self.pos
            buf_read = np.concatenate([self.buf[self.pos:], self.buf[:n - split]])

        filtered, self._zi = lfilter(
            [self.damp2], [1.0, -self.damp1],
            buf_read.astype(np.float64), zi=self._zi,
        )
        filtered = filtered.astype(np.float32)
        new_vals = inp + filtered * self.feedback

        if end <= buf_len:
            self.buf[self.pos:end] = new_vals
        else:
            split = buf_len - self.pos
            self.buf[self.pos:]    = new_vals[:split]
            self.buf[:n - split]   = new_vals[split:]

        self.pos = end % buf_len
        return buf_read


class _AllpassFilter:
    """Freeverb allpass filter. Sub-blocks at wrap boundaries for small buffers."""

    def __init__(self, size: int):
        self.buf = np.zeros(size, dtype=np.float32)
        self.pos = 0

    def process(self, inp: np.ndarray) -> np.ndarray:
        n       = len(inp)
        buf_len = len(self.buf)
        output  = np.empty(n, dtype=np.float32)
        pos_inp = 0

        while pos_inp < n:
            avail = buf_len - self.pos
            chunk = min(avail, n - pos_inp)

            buf_slice = self.buf[self.pos:self.pos + chunk].copy()
            inp_slice = inp[pos_inp:pos_inp + chunk]

            output[pos_inp:pos_inp + chunk]      = buf_slice - inp_slice
            self.buf[self.pos:self.pos + chunk]  = inp_slice + buf_slice * _AP_FEEDBACK

            self.pos  = (self.pos + chunk) % buf_len
            pos_inp  += chunk

        return output


class _FreeverbMono:
    """Mono Freeverb (one L or R instance). Processes flat float32 arrays."""

    def __init__(self, comb_sizes):
        self._combs = [_CombFilter(s) for s in comb_sizes]
        self._aps   = [_AllpassFilter(s) for s in _AP_SIZES]

    def set_room_size(self, val: float) -> None:
        feedback = val * 0.28 + 0.7
        for c in self._combs:
            c.feedback = feedback

    def set_damp(self, val: float) -> None:
        d = val * 0.4
        for c in self._combs:
            c.set_damp(d)

    def process(self, inp: np.ndarray) -> np.ndarray:
        inp = inp.astype(np.float32) * _FIXED_GAIN
        out = np.zeros(len(inp), dtype=np.float32)
        for c in self._combs:
            out += c.process(inp)
        for ap in self._aps:
            out = ap.process(out)
        return out


# ─────────────────────────────────────────────────────────────────────────────
# Public interface
# ─────────────────────────────────────────────────────────────────────────────

class EffectsChain:
    """
    3-band EQ + stereo Freeverb reverb.

    Intended to be called from AudioMixer._audio_callback after all sources
    are summed and before the peak limiter:

        mix = self.effects.process(mix)

    All setters are safe to call from the MIDI thread at any time.
    """

    def __init__(self):
        self._bass   = _EQBand("low_shelf",  f0=250.0)
        self._mid    = _EQBand("peak_eq",    f0=1000.0)
        self._treble = _EQBand("high_shelf", f0=4000.0)

        self._verb_L = _FreeverbMono(_COMB_L)
        self._verb_R = _FreeverbMono(_COMB_R)
        self._room_size:  float = 0.5
        self._damp:       float = 0.5
        self._reverb_wet: float = 0.0   # 0.0 = bypass

    # ── Parameter setters ─────────────────────────────────────────────────────

    def set_eq_bass(self, db: float) -> None:
        """Set bass shelf gain in dB. Called from MIDI thread."""
        self._bass.set_gain_db(db)

    def set_eq_mid(self, db: float) -> None:
        """Set mid peaking EQ gain in dB. Called from MIDI thread."""
        self._mid.set_gain_db(db)

    def set_eq_treble(self, db: float) -> None:
        """Set treble shelf gain in dB. Called from MIDI thread."""
        self._treble.set_gain_db(db)

    def set_reverb_wet(self, pos: float) -> None:
        """
        Set reverb amount from knob position [0.0-1.0].
            0.0 → dry (bypassed)
            0.5 → medium room, ~20% wet
            1.0 → large hall, ~80% wet, room_size=0.98
        """
        pos = max(0.0, min(1.0, float(pos)))
        # Quadratic wet mix: 0→0, 0.5→0.2, 1.0→0.8
        self._reverb_wet = pos * pos * 0.8
        room_size = 0.5 + pos * 0.48     # 0.5 → 0.98
        self._room_size = room_size
        self._verb_L.set_room_size(room_size)
        self._verb_R.set_room_size(room_size)
        self._verb_L.set_damp(self._damp)
        self._verb_R.set_damp(self._damp)

    # ── Processing ────────────────────────────────────────────────────────────

    def process(self, audio: np.ndarray) -> np.ndarray:
        """
        Apply EQ then reverb to a (N, 2) float32 block.
        Modifies audio in-place and returns it.
        """
        # 3-band EQ (each band is a no-op when gain ≈ 0)
        audio = self._bass.process(audio)
        audio = self._mid.process(audio)
        audio = self._treble.process(audio)

        # Freeverb reverb
        if self._reverb_wet > 1e-6:
            wet_L = self._verb_L.process(audio[:, 0])
            wet_R = self._verb_R.process(audio[:, 1])
            w = self._reverb_wet
            audio[:, 0] = audio[:, 0] * (1.0 - w) + wet_L * w
            audio[:, 1] = audio[:, 1] * (1.0 - w) + wet_R * w

        return audio
