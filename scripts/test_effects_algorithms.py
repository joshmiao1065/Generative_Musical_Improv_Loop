"""
test_effects_algorithms.py — Algorithmic validation for EQ + Reverb.

Verifies the RBJ biquad EQ formulas and the Freeverb algorithm are
mathematically correct before they go into effects_chain.py.

No audio hardware required — numpy + scipy only.

Usage:
    python scripts/test_effects_algorithms.py
"""

import sys
import unittest
from pathlib import Path

import numpy as np
from scipy.signal import sosfilt, sosfilt_zi, lfilter

sys.path.insert(0, str(Path(__file__).parent.parent))

FS = 48000   # sample rate throughout


# ─────────────────────────────────────────────────────────────────────────────
# Reference implementations of the three RBJ biquad filter types
# ─────────────────────────────────────────────────────────────────────────────

def _low_shelf_sos(fs, f0, db_gain):
    """RBJ Audio EQ Cookbook lowShelf, S=1 (maximally flat shelf slope)."""
    A = 10 ** (db_gain / 40.0)
    w0 = 2 * np.pi * f0 / fs
    alpha = np.sin(w0) / np.sqrt(2)   # S=1 → alpha = sin(w0)/sqrt(2)
    cosw0, sqrtA = np.cos(w0), np.sqrt(A)

    b0 = A * ((A+1) - (A-1)*cosw0 + 2*sqrtA*alpha)
    b1 = 2*A * ((A-1) - (A+1)*cosw0)
    b2 = A * ((A+1) - (A-1)*cosw0 - 2*sqrtA*alpha)
    a0 = (A+1) + (A-1)*cosw0 + 2*sqrtA*alpha
    a1 = -2 * ((A-1) + (A+1)*cosw0)
    a2 = (A+1) + (A-1)*cosw0 - 2*sqrtA*alpha

    return np.array([[b0/a0, b1/a0, b2/a0, 1.0, a1/a0, a2/a0]])


def _high_shelf_sos(fs, f0, db_gain):
    """RBJ Audio EQ Cookbook highShelf, S=1."""
    A = 10 ** (db_gain / 40.0)
    w0 = 2 * np.pi * f0 / fs
    alpha = np.sin(w0) / np.sqrt(2)
    cosw0, sqrtA = np.cos(w0), np.sqrt(A)

    b0 = A * ((A+1) + (A-1)*cosw0 + 2*sqrtA*alpha)
    b1 = -2*A * ((A-1) + (A+1)*cosw0)
    b2 = A * ((A+1) + (A-1)*cosw0 - 2*sqrtA*alpha)
    a0 = (A+1) - (A-1)*cosw0 + 2*sqrtA*alpha
    a1 = 2 * ((A-1) - (A+1)*cosw0)
    a2 = (A+1) - (A-1)*cosw0 - 2*sqrtA*alpha

    return np.array([[b0/a0, b1/a0, b2/a0, 1.0, a1/a0, a2/a0]])


def _peak_eq_sos(fs, f0, db_gain, Q=0.707):
    """RBJ Audio EQ Cookbook peakingEQ."""
    A = 10 ** (db_gain / 40.0)
    w0 = 2 * np.pi * f0 / fs
    alpha = np.sin(w0) / (2 * Q)
    cosw0 = np.cos(w0)

    b0 = 1 + alpha * A
    b1 = -2 * cosw0
    b2 = 1 - alpha * A
    a0 = 1 + alpha / A
    a1 = -2 * cosw0
    a2 = 1 - alpha / A

    return np.array([[b0/a0, b1/a0, b2/a0, 1.0, a1/a0, a2/a0]])


def _gain_db_at_freq(sos, freq, fs=FS, duration=0.5):
    """
    Empirically measure steady-state gain of a filter at a given frequency.
    Returns the gain in dB by comparing RMS of input and (settled) output.
    """
    n = int(duration * fs)
    t = np.linspace(0, duration, n, endpoint=False)
    signal = np.sin(2 * np.pi * freq * t).astype(np.float64)
    output = sosfilt(sos, signal)
    # Discard first 20% to let transient settle
    skip = n // 5
    in_rms  = np.sqrt(np.mean(signal[skip:] ** 2))
    out_rms = np.sqrt(np.mean(output[skip:] ** 2))
    return 20.0 * np.log10(out_rms / in_rms)


# ─────────────────────────────────────────────────────────────────────────────
# Reference Freeverb implementation (vectorised, numpy only)
# ─────────────────────────────────────────────────────────────────────────────

_SCALE = 48000 / 44100

_COMB_L  = [round(t * _SCALE) for t in [1116, 1188, 1277, 1356, 1422, 1491, 1557, 1617]]
_COMB_R  = [t + round(23 * _SCALE) for t in _COMB_L]
_AP_SIZES = [round(t * _SCALE) for t in [556, 441, 341, 225]]

_AP_FEEDBACK   = 0.5
_FIXED_GAIN    = 0.015


class _CombFilter:
    """One Freeverb comb filter (vectorised block processing)."""

    def __init__(self, size):
        self.buf = np.zeros(size, dtype=np.float32)
        self.pos = 0
        self.feedback = 0.84   # set via set_room_size
        self.damp1 = 0.2       # set via set_damp
        self.damp2 = 0.8
        self._zi = np.zeros(1, dtype=np.float64)  # state for the 1-pole damping IIR

    def set_damp(self, val):
        self.damp1 = val
        self.damp2 = 1.0 - val

    def process(self, inp: np.ndarray) -> np.ndarray:
        """Process a block of N samples. Returns N output samples."""
        n = len(inp)
        buf_len = len(self.buf)

        # Circular read from delay buffer
        end = self.pos + n
        if end <= buf_len:
            buf_read = self.buf[self.pos:end].copy()
        else:
            split = buf_len - self.pos
            buf_read = np.concatenate([self.buf[self.pos:], self.buf[:n - split]])

        # Apply 1-pole damping IIR to the read values (models the comb filter's
        # low-pass in the feedback path — the core of the Freeverb "tone" control).
        filtered, self._zi = lfilter(
            [self.damp2], [1.0, -self.damp1],
            buf_read.astype(np.float64), zi=self._zi
        )
        filtered = filtered.astype(np.float32)

        # Write back: input + filtered × feedback
        new_vals = inp + filtered * self.feedback
        if end <= buf_len:
            self.buf[self.pos:end] = new_vals
        else:
            split = buf_len - self.pos
            self.buf[self.pos:] = new_vals[:split]
            self.buf[:n - split] = new_vals[split:]

        self.pos = end % buf_len
        return buf_read   # output is the raw read (before damping)


class _AllpassFilter:
    """One Freeverb all-pass filter.

    Allpass buffers 2–4 are 371, 480, 245 samples — all smaller than the typical
    512-sample audio block. The implementation sub-blocks at buffer-wrap boundaries
    so that each write is seen by the correct subsequent read within the same block.
    This is the only correct approach when block_size >= buf_len.
    """

    def __init__(self, size):
        self.buf = np.zeros(size, dtype=np.float32)
        self.pos = 0

    def process(self, inp: np.ndarray) -> np.ndarray:
        n = len(inp)
        buf_len = len(self.buf)
        output = np.empty(n, dtype=np.float32)
        pos_inp = 0

        while pos_inp < n:
            # Number of contiguous samples available before the buffer pointer wraps
            avail = buf_len - self.pos
            chunk = min(avail, n - pos_inp)

            buf_slice  = self.buf[self.pos:self.pos + chunk].copy()
            inp_slice  = inp[pos_inp:pos_inp + chunk]

            output[pos_inp:pos_inp + chunk] = buf_slice - inp_slice
            self.buf[self.pos:self.pos + chunk] = inp_slice + buf_slice * _AP_FEEDBACK

            self.pos  = (self.pos + chunk) % buf_len
            pos_inp  += chunk

        return output


class Freeverb:
    """
    Simplified Freeverb for algorithm validation.
    Processes mono signals; stereo support via separate L/R instances.
    """

    def __init__(self, room_size=0.5, damp=0.5):
        self._combs = [_CombFilter(s) for s in _COMB_L]
        self._aps   = [_AllpassFilter(s) for s in _AP_SIZES]
        self.set_room_size(room_size)
        self.set_damp(damp)

    def set_room_size(self, val):
        feedback = val * 0.28 + 0.7
        for c in self._combs:
            c.feedback = feedback

    def set_damp(self, val):
        d = val * 0.4
        for c in self._combs:
            c.set_damp(d)

    def process_block(self, inp: np.ndarray) -> np.ndarray:
        """Process one block of mono float32 samples."""
        inp = inp.astype(np.float32) * _FIXED_GAIN
        out = np.zeros(len(inp), dtype=np.float32)
        for c in self._combs:
            out += c.process(inp)
        for ap in self._aps:
            out = ap.process(out)
        return out


# ─────────────────────────────────────────────────────────────────────────────
# Tests: 3-band EQ
# ─────────────────────────────────────────────────────────────────────────────

class TestEQIdentity(unittest.TestCase):
    """At 0 dB, every filter must be transparent (gain ≈ 0 dB everywhere)."""

    FREQS = [50, 100, 250, 500, 1000, 2000, 4000, 8000, 15000]
    TOL   = 0.1   # ±0.1 dB tolerance for numerical precision

    def _check_passthrough(self, sos, label):
        for freq in self.FREQS:
            if freq >= FS / 2:
                continue
            gain = _gain_db_at_freq(sos, freq)
            self.assertAlmostEqual(
                gain, 0.0, delta=self.TOL,
                msg=f"{label} at {freq} Hz: expected 0 dB, got {gain:.3f} dB"
            )

    def test_low_shelf_zero_gain(self):
        self._check_passthrough(_low_shelf_sos(FS, 250, 0.0), "LowShelf(0dB)")

    def test_high_shelf_zero_gain(self):
        self._check_passthrough(_high_shelf_sos(FS, 4000, 0.0), "HighShelf(0dB)")

    def test_peak_eq_zero_gain(self):
        self._check_passthrough(_peak_eq_sos(FS, 1000, 0.0), "PeakEQ(0dB)")


class TestLowShelf(unittest.TestCase):
    """Low shelf boosts/cuts bass frequencies and leaves treble alone."""

    def test_boost_well_below_shelf(self):
        """A +12 dB shelf centred at 250 Hz must boost 50 Hz by ~12 dB."""
        sos  = _low_shelf_sos(FS, 250, 12.0)
        gain = _gain_db_at_freq(sos, 50)
        self.assertAlmostEqual(gain, 12.0, delta=1.5,
                               msg=f"Low shelf +12dB at 50 Hz: got {gain:.2f} dB")

    def test_cut_well_below_shelf(self):
        """A -12 dB shelf must cut 50 Hz by ~12 dB."""
        sos  = _low_shelf_sos(FS, 250, -12.0)
        gain = _gain_db_at_freq(sos, 50)
        self.assertAlmostEqual(gain, -12.0, delta=1.5,
                               msg=f"Low shelf -12dB at 50 Hz: got {gain:.2f} dB")

    def test_treble_unaffected_by_bass_boost(self):
        """8 kHz is well above 250 Hz shelf — gain there should be near 0 dB."""
        sos  = _low_shelf_sos(FS, 250, 12.0)
        gain = _gain_db_at_freq(sos, 8000)
        self.assertAlmostEqual(gain, 0.0, delta=0.5,
                               msg=f"Low shelf +12dB: treble (8 kHz) gain {gain:.2f} dB (should be ~0)")

    def test_gain_at_shelf_freq_is_half_boost(self):
        """At the shelf frequency, gain should be exactly half the boost (6 dB for +12 dB shelf)."""
        sos  = _low_shelf_sos(FS, 250, 12.0)
        gain = _gain_db_at_freq(sos, 250)
        self.assertAlmostEqual(gain, 6.0, delta=1.5,
                               msg=f"Low shelf +12dB at shelf freq 250 Hz: got {gain:.2f} dB (expected ~6)")


class TestHighShelf(unittest.TestCase):
    """High shelf boosts/cuts treble frequencies."""

    def test_boost_well_above_shelf(self):
        """A +12 dB shelf at 4 kHz must boost 15 kHz by ~12 dB."""
        sos  = _high_shelf_sos(FS, 4000, 12.0)
        gain = _gain_db_at_freq(sos, 15000)
        self.assertAlmostEqual(gain, 12.0, delta=1.5,
                               msg=f"High shelf +12dB at 15 kHz: got {gain:.2f} dB")

    def test_cut_well_above_shelf(self):
        """A -12 dB shelf must cut 15 kHz by ~12 dB."""
        sos  = _high_shelf_sos(FS, 4000, -12.0)
        gain = _gain_db_at_freq(sos, 15000)
        self.assertAlmostEqual(gain, -12.0, delta=1.5,
                               msg=f"High shelf -12dB at 15 kHz: got {gain:.2f} dB")

    def test_bass_unaffected_by_treble_boost(self):
        """100 Hz is far below 4 kHz shelf — should be ~0 dB."""
        sos  = _high_shelf_sos(FS, 4000, 12.0)
        gain = _gain_db_at_freq(sos, 100)
        self.assertAlmostEqual(gain, 0.0, delta=0.5,
                               msg=f"High shelf +12dB: bass (100 Hz) gain {gain:.2f} dB (should be ~0)")


class TestPeakEQ(unittest.TestCase):
    """Peaking EQ boosts/cuts near the center frequency."""

    def test_boost_at_center(self):
        """Peak EQ +12 dB at 1 kHz must measure ~+12 dB at exactly 1 kHz."""
        sos  = _peak_eq_sos(FS, 1000, 12.0, Q=0.707)
        gain = _gain_db_at_freq(sos, 1000)
        self.assertAlmostEqual(gain, 12.0, delta=1.5,
                               msg=f"PeakEQ +12dB at 1 kHz: got {gain:.2f} dB")

    def test_cut_at_center(self):
        """Peak EQ -12 dB at 1 kHz must measure ~-12 dB at exactly 1 kHz."""
        sos  = _peak_eq_sos(FS, 1000, -12.0, Q=0.707)
        gain = _gain_db_at_freq(sos, 1000)
        self.assertAlmostEqual(gain, -12.0, delta=1.5,
                               msg=f"PeakEQ -12dB at 1 kHz: got {gain:.2f} dB")

    def test_extremes_unaffected(self):
        """Boost at 1 kHz should leave 50 Hz and 18 kHz nearly flat (Q=0.707)."""
        sos = _peak_eq_sos(FS, 1000, 12.0, Q=0.707)
        for freq in [50, 18000]:
            gain = _gain_db_at_freq(sos, freq)
            self.assertAlmostEqual(gain, 0.0, delta=1.0,
                                   msg=f"PeakEQ: {freq} Hz gain {gain:.2f} dB (should be ~0)")


class TestEQBlockProcessing(unittest.TestCase):
    """
    Block-by-block processing must produce the same result as processing
    the whole signal at once, provided state (zi) is correctly threaded.
    """

    BLOCKSIZE = 512

    def _process_in_blocks(self, sos, signal):
        n_sections = sos.shape[0]
        # Initial state: sosfilt_zi scaled by first sample (steady-state assumption)
        zi = sosfilt_zi(sos) * signal[0]
        output = np.zeros_like(signal)
        pos = 0
        while pos < len(signal):
            block = signal[pos:pos + self.BLOCKSIZE]
            out_block, zi = sosfilt(sos, block, zi=zi)
            output[pos:pos + len(block)] = out_block
            pos += self.BLOCKSIZE
        return output

    def _process_whole(self, sos, signal):
        zi = sosfilt_zi(sos) * signal[0]
        out, _ = sosfilt(sos, signal, zi=zi)
        return out

    def _make_signal(self, n=48000):
        t = np.linspace(0, n / FS, n, endpoint=False)
        return (
            0.3 * np.sin(2 * np.pi * 100 * t) +
            0.3 * np.sin(2 * np.pi * 1000 * t) +
            0.3 * np.sin(2 * np.pi * 10000 * t)
        ).astype(np.float64)

    def test_low_shelf_block_matches_whole(self):
        sig = self._make_signal()
        sos = _low_shelf_sos(FS, 250, 6.0)
        self.assertTrue(
            np.allclose(
                self._process_in_blocks(sos, sig),
                self._process_whole(sos, sig),
                atol=1e-5,
            ),
            "Low shelf: block-by-block output differs from single-pass output",
        )

    def test_high_shelf_block_matches_whole(self):
        sig = self._make_signal()
        sos = _high_shelf_sos(FS, 4000, -6.0)
        self.assertTrue(
            np.allclose(
                self._process_in_blocks(sos, sig),
                self._process_whole(sos, sig),
                atol=1e-5,
            ),
            "High shelf: block-by-block output differs from single-pass output",
        )

    def test_peak_eq_block_matches_whole(self):
        sig = self._make_signal()
        sos = _peak_eq_sos(FS, 1000, 9.0)
        self.assertTrue(
            np.allclose(
                self._process_in_blocks(sos, sig),
                self._process_whole(sos, sig),
                atol=1e-5,
            ),
            "Peak EQ: block-by-block output differs from single-pass output",
        )


# ─────────────────────────────────────────────────────────────────────────────
# Tests: Freeverb
# ─────────────────────────────────────────────────────────────────────────────

class TestFreeverbBasic(unittest.TestCase):

    BLOCKSIZE = 512

    def _process(self, verb, signal):
        """Feed signal block-by-block, return full output."""
        out = np.zeros(len(signal), dtype=np.float32)
        for pos in range(0, len(signal), self.BLOCKSIZE):
            block = signal[pos:pos + self.BLOCKSIZE]
            out[pos:pos + len(block)] = verb.process_block(block)
        return out

    def test_impulse_produces_reverb_tail(self):
        """An impulse at t=0 should produce energy spread over many seconds."""
        fs = FS
        duration = 3.0
        n = int(duration * fs)
        signal = np.zeros(n, dtype=np.float32)
        signal[0] = 0.5   # impulse

        verb = Freeverb(room_size=0.7, damp=0.5)
        output = self._process(verb, signal)

        # Energy in 0–0.5s window
        e_early = np.sum(output[:int(0.5 * fs)] ** 2)
        # Energy in 0.5–1.0s window
        e_mid   = np.sum(output[int(0.5 * fs):int(1.0 * fs)] ** 2)
        # Energy in 1.0–2.0s window
        e_late  = np.sum(output[int(1.0 * fs):int(2.0 * fs)] ** 2)

        self.assertGreater(e_early, 1e-10, "No energy in early reverb tail")
        self.assertGreater(e_mid,   1e-10, "No energy in mid reverb tail (should still be ringing)")
        self.assertGreater(e_early, e_late,
                           "Reverb tail should decay: early energy must exceed late energy")

    def test_silence_in_silence_out(self):
        """Zero input should produce zero output (no self-oscillation without input)."""
        n = int(1.0 * FS)
        signal = np.zeros(n, dtype=np.float32)
        verb = Freeverb(room_size=0.8, damp=0.5)
        output = self._process(verb, signal)
        self.assertAlmostEqual(float(np.max(np.abs(output))), 0.0, delta=1e-10,
                               msg="Silent input should produce silent output")

    def test_output_bounded(self):
        """A loud sustained signal must not cause the reverb to blow up."""
        fs = FS
        n = int(2.0 * fs)
        t = np.linspace(0, 2.0, n, endpoint=False)
        signal = (0.9 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

        verb = Freeverb(room_size=0.9, damp=0.5)
        output = self._process(verb, signal)

        max_amp = float(np.max(np.abs(output)))
        self.assertLess(max_amp, 5.0,
                        f"Reverb output blew up: peak amplitude {max_amp:.2f}")

    def test_high_room_size_is_stable(self):
        """Room size near 0.98 (near-maximum) must not produce instability."""
        n = int(3.0 * FS)
        t = np.linspace(0, 3.0, n, endpoint=False)
        signal = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

        verb = Freeverb(room_size=0.98, damp=0.5)
        output = self._process(verb, signal)

        self.assertFalse(np.any(np.isnan(output)), "NaN in reverb output at high room size")
        self.assertFalse(np.any(np.isinf(output)), "Inf in reverb output at high room size")
        max_amp = float(np.max(np.abs(output)))
        self.assertLess(max_amp, 10.0,
                        f"Reverb unstable at room_size=0.98: peak {max_amp:.2f}")

    def test_block_size_consistency(self):
        """
        Processing in 512-sample blocks must produce the same output as 128-sample
        blocks (within float32 precision). Block size must not change output.
        """
        n = int(1.0 * FS)
        t = np.linspace(0, 1.0, n, endpoint=False)
        signal = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

        verb_512 = Freeverb(room_size=0.6, damp=0.5)
        verb_128 = Freeverb(room_size=0.6, damp=0.5)

        out_512 = self._process(verb_512, signal)

        out_128 = np.zeros(n, dtype=np.float32)
        for pos in range(0, n, 128):
            block = signal[pos:pos + 128]
            out_128[pos:pos + len(block)] = verb_128.process_block(block)

        self.assertTrue(
            np.allclose(out_512, out_128, atol=1e-5),
            "Reverb output differs between 512-sample and 128-sample block sizes"
        )


class TestFreeverbDecay(unittest.TestCase):
    """Reverb decay characteristics under different room sizes."""

    BLOCKSIZE = 512

    def _decay_ratio(self, room_size):
        """
        Feed a 0.5s burst, then silence. Measure energy ratio
        (first 0.5s of tail) vs (second 0.5s of tail).
        Higher room_size should produce slower decay (higher ratio).
        """
        n = int(2.5 * FS)
        burst_n = int(0.5 * FS)
        signal = np.zeros(n, dtype=np.float32)
        t = np.linspace(0, 0.5, burst_n, endpoint=False)
        signal[:burst_n] = 0.3 * np.sin(2 * np.pi * 440 * t)

        verb = Freeverb(room_size=room_size, damp=0.5)
        out = np.zeros(n, dtype=np.float32)
        for pos in range(0, n, self.BLOCKSIZE):
            block = signal[pos:pos + self.BLOCKSIZE]
            out[pos:pos + len(block)] = verb.process_block(block)

        e_first_half  = np.sum(out[burst_n:burst_n + int(0.5 * FS)] ** 2)
        e_second_half = np.sum(out[burst_n + int(0.5 * FS):burst_n + int(1.0 * FS)] ** 2)
        if e_second_half < 1e-20:
            return float('inf')  # instant decay
        return e_first_half / e_second_half

    def test_larger_room_decays_slower(self):
        """room_size=0.8 should produce a longer reverb tail than room_size=0.5."""
        ratio_08 = self._decay_ratio(0.8)
        ratio_05 = self._decay_ratio(0.5)
        self.assertLess(
            ratio_08, ratio_05,
            f"Expected room_size=0.8 to decay more slowly than 0.5, "
            f"but decay ratios were {ratio_08:.2f} vs {ratio_05:.2f} "
            f"(lower ratio = slower decay = longer tail)"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Run
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 64)
    print("  Effects Algorithm Tests (EQ + Reverb)")
    print("  No audio hardware required")
    print("=" * 64)
    print()

    loader = unittest.TestLoader()
    suite  = unittest.TestSuite()

    eq_classes    = [TestEQIdentity, TestLowShelf, TestHighShelf,
                     TestPeakEQ, TestEQBlockProcessing]
    verb_classes  = [TestFreeverbBasic, TestFreeverbDecay]

    for cls in eq_classes + verb_classes:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    print()
    if result.wasSuccessful():
        print("All tests passed — algorithms are correct.")
    else:
        print(f"FAILURES: {len(result.failures)}  ERRORS: {len(result.errors)}")
    sys.exit(0 if result.wasSuccessful() else 1)
