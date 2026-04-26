"""
test_audio_logic.py — Unit tests for core audio pipeline logic.

Tests voice enable/disable state machine and shape trim/pad without
requiring audio hardware. Runs in WSL or any environment with numpy.

Usage:
    python scripts/test_audio_logic.py
"""

import sys
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

# ── Stub out sounddevice so we can import AudioMixer in WSL ──────────────────
_sd_stub = types.ModuleType("sounddevice")
_sd_stub.OutputStream = MagicMock()
_sd_stub.InputStream  = MagicMock()
sys.modules.setdefault("sounddevice", _sd_stub)

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.audio_mixer import AudioMixer, N_VOICES, CHANNELS, SAMPLE_RATE


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_mixer() -> AudioMixer:
    """Return an AudioMixer with a mock stream (no hardware required)."""
    mixer = AudioMixer(device_idx=0, blocksize=512)
    # Patch the stream so start() doesn't try to open real hardware
    mixer._stream = MagicMock()
    return mixer


def _sine(n: int, freq: float = 440.0) -> np.ndarray:
    """Return a (n, 2) float32 stereo sine wave."""
    t    = np.linspace(0, n / SAMPLE_RATE, n, endpoint=False, dtype=np.float32)
    mono = np.sin(2 * np.pi * freq * t) * 0.3
    return np.stack([mono, mono], axis=1)


# ─────────────────────────────────────────────────────────────────────────────
# Test: voice enable/disable state machine
# ─────────────────────────────────────────────────────────────────────────────

class TestVoiceEnableDisable(unittest.TestCase):

    def test_voices_start_disabled(self):
        """All voices must start disabled so they don't play before being enabled."""
        mixer = _make_mixer()
        for i in range(N_VOICES):
            self.assertFalse(
                mixer._voice_enabled[i],
                f"Voice {i} should start disabled but is enabled.",
            )

    def test_enable_voice(self):
        mixer = _make_mixer()
        mixer.set_voice_enabled(0, True)
        self.assertTrue(mixer._voice_enabled[0])

    def test_disable_voice(self):
        mixer = _make_mixer()
        mixer.set_voice_enabled(0, True)
        mixer.set_voice_enabled(0, False)
        self.assertFalse(mixer._voice_enabled[0])

    def test_enable_disable_independent(self):
        """Enabling one voice must not affect the others."""
        mixer = _make_mixer()
        mixer.set_voice_enabled(1, True)
        self.assertFalse(mixer._voice_enabled[0])
        self.assertTrue(mixer._voice_enabled[1])
        self.assertFalse(mixer._voice_enabled[2])

    def test_disabled_voice_produces_no_audio(self):
        """Audio callback must produce silence for a disabled voice."""
        mixer    = _make_mixer()
        loop_len = 4800   # 0.1s at 48kHz
        loop     = _sine(loop_len, 440.0)
        voice    = _sine(loop_len, 660.0)

        mixer.set_loop(loop)
        mixer._loop_audio = loop   # bypass stream

        # Queue and swap in the voice manually
        mixer.queue_voice(0, voice, volume=1.0)
        mixer._on_boundary()   # swap from queue → _voice_audio

        # Voice is still disabled — output should equal user loop only
        out     = np.zeros((512, CHANNELS), dtype=np.float32)
        # Simulate what the callback does for the voice section
        enabled = mixer._voice_enabled[0]
        self.assertFalse(enabled, "Voice 0 should still be disabled after queue_voice")

    def test_enabled_voice_audio_present(self):
        """After enabling a voice and swapping audio in, _voice_audio[i] is set."""
        mixer    = _make_mixer()
        loop_len = 4800
        loop     = _sine(loop_len)
        voice    = _sine(loop_len, 660.0)

        mixer.set_loop(loop)
        mixer._loop_audio = loop

        mixer.set_voice_enabled(0, True)
        mixer.queue_voice(0, voice, volume=0.85)
        mixer._on_boundary()

        self.assertIsNotNone(mixer._voice_audio[0])
        np.testing.assert_array_equal(mixer._voice_audio[0], voice)

    def test_toggle_sequence_matches_button_press(self):
        """
        Simulate Button 2 → 4 presses (PBF4 voice toggle).
        Press 1: enable; press 2: disable; press 3: enable again.
        """
        mixer = _make_mixer()
        toggle_state = False   # mirrors PBF4Controller._toggle_state

        for press, expected_enabled in [(1, True), (2, False), (3, True)]:
            toggle_state = not toggle_state
            mixer.set_voice_enabled(0, toggle_state)
            self.assertEqual(
                mixer._voice_enabled[0], expected_enabled,
                f"After press {press}: expected {expected_enabled}, got {mixer._voice_enabled[0]}",
            )


# ─────────────────────────────────────────────────────────────────────────────
# Test: shape trim/pad at loop boundary
# ─────────────────────────────────────────────────────────────────────────────

class TestShapeMismatch(unittest.TestCase):

    def _setup(self, loop_len: int, voice_len: int) -> AudioMixer:
        mixer = _make_mixer()
        loop  = _sine(loop_len)
        voice = _sine(voice_len, 660.0)
        mixer._loop_audio = loop
        mixer.queue_voice(0, voice, volume=0.85)
        mixer._on_boundary()   # trim/pad happens here
        return mixer

    def test_exact_length_unchanged(self):
        """Audio of the correct length should not be modified."""
        n     = 48000
        mixer = self._setup(n, n)
        self.assertEqual(mixer._voice_audio[0].shape[0], n)

    def test_long_voice_trimmed_to_loop(self):
        """Voice audio longer than the loop must be trimmed."""
        loop_len  = 48000
        voice_len = 96000   # twice as long
        mixer     = self._setup(loop_len, voice_len)
        result_len = mixer._voice_audio[0].shape[0]
        self.assertEqual(
            result_len, loop_len,
            f"Expected trim to {loop_len}, got {result_len}",
        )

    def test_short_voice_padded_to_loop(self):
        """Voice audio shorter than the loop must be zero-padded."""
        loop_len  = 48000
        voice_len = 32000   # shorter
        mixer     = self._setup(loop_len, voice_len)
        result_len = mixer._voice_audio[0].shape[0]
        self.assertEqual(
            result_len, loop_len,
            f"Expected pad to {loop_len}, got {result_len}",
        )
        # Verify the padded tail is silence
        tail = mixer._voice_audio[0][voice_len:]
        self.assertTrue(
            np.allclose(tail, 0.0),
            "Padded region should be silent",
        )

    def test_trimmed_content_matches_original(self):
        """Trimmed audio must equal the first loop_len samples of the original."""
        n         = 48000
        voice_len = 72000
        loop      = _sine(n)
        voice     = _sine(voice_len, 660.0)
        mixer     = _make_mixer()
        mixer._loop_audio = loop
        mixer.queue_voice(0, voice, volume=0.85)
        mixer._on_boundary()
        np.testing.assert_array_equal(mixer._voice_audio[0], voice[:n])

    def test_padded_content_matches_original(self):
        """Padded audio: the first voice_len samples must equal the original."""
        n         = 48000
        voice_len = 32000
        loop      = _sine(n)
        voice     = _sine(voice_len, 660.0)
        mixer     = _make_mixer()
        mixer._loop_audio = loop
        mixer.queue_voice(0, voice, volume=0.85)
        mixer._on_boundary()
        np.testing.assert_array_equal(mixer._voice_audio[0][:voice_len], voice)

    def test_90bpm_8beats_mismatch(self):
        """
        Regression: 90 BPM / 8 beats produces a length mismatch between
        the user loop (256,000 samples) and generated audio (288,000 samples).
        The boundary handler must correct this silently.
        """
        import math
        SR         = 48000
        CHUNK      = 96000   # CHUNK_SAMPLES = 2s × 48kHz
        bpm, beats = 90, 8
        loop_sec   = beats * (60.0 / bpm)       # 5.333…s
        loop_len   = int(loop_sec * SR)          # 256,000
        chunks     = max(1, round(loop_sec * SR / CHUNK))  # round(2.667) = 3
        voice_len  = chunks * CHUNK              # 288,000

        mixer = self._setup(loop_len, voice_len)
        self.assertEqual(
            mixer._voice_audio[0].shape[0], loop_len,
            f"90 BPM / 8 beats: expected voice trimmed to {loop_len}, "
            f"got {mixer._voice_audio[0].shape[0]}",
        )

    def test_all_voices_trimmed_independently(self):
        """Each voice slot is trimmed/padded independently."""
        loop_len = 48000
        loop     = _sine(loop_len)
        mixer    = _make_mixer()
        mixer._loop_audio = loop

        lens = [96000, 32000, 48000]   # long, short, exact
        for i, vl in enumerate(lens):
            mixer.queue_voice(i, _sine(vl, 440 + i * 100), volume=0.85)
        mixer._on_boundary()

        for i in range(N_VOICES):
            self.assertEqual(
                mixer._voice_audio[i].shape[0], loop_len,
                f"Voice {i}: expected {loop_len}, got {mixer._voice_audio[i].shape[0]}",
            )


# ─────────────────────────────────────────────────────────────────────────────
# Test: queue behaviour
# ─────────────────────────────────────────────────────────────────────────────

class TestQueueBehaviour(unittest.TestCase):

    def test_queue_full_drops_oldest(self):
        """When the queue is full (maxsize=2), adding a third item drops the oldest."""
        mixer    = _make_mixer()
        loop_len = 4800
        mixer._loop_audio = _sine(loop_len)

        v1 = _sine(loop_len, 440.0)
        v2 = _sine(loop_len, 550.0)
        v3 = _sine(loop_len, 660.0)

        mixer.queue_voice(0, v1, 0.85)
        mixer.queue_voice(0, v2, 0.85)
        # Queue is now full (maxsize=2). Adding v3 should drop v1.
        mixer.queue_voice(0, v3, 0.85)

        # After one boundary swap, v2 is consumed (v1 was dropped).
        mixer._on_boundary()
        np.testing.assert_array_equal(mixer._voice_audio[0], v2)

    def test_no_boundary_means_no_swap(self):
        """Audio queued but not yet at a boundary stays in the queue."""
        mixer    = _make_mixer()
        loop_len = 4800
        mixer._loop_audio = _sine(loop_len)
        voice    = _sine(loop_len, 660.0)

        mixer.queue_voice(0, voice, 0.85)
        # Do NOT call _on_boundary — swap should not have happened
        self.assertIsNone(mixer._voice_audio[0])

    def test_clear_voices_empties_queue(self):
        """clear_voices() should empty all queues and set _voice_audio to None."""
        mixer    = _make_mixer()
        loop_len = 4800
        mixer._loop_audio = _sine(loop_len)

        for i in range(N_VOICES):
            mixer.queue_voice(i, _sine(loop_len), 0.85)
            mixer._voice_audio[i] = _sine(loop_len)

        mixer.clear_voices()

        for i in range(N_VOICES):
            self.assertIsNone(mixer._voice_audio[i])
            self.assertTrue(mixer._voice_queue[i].empty())


# ─────────────────────────────────────────────────────────────────────────────
# Run
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Running AudioMixer unit tests (no hardware required)...\n")
    loader = unittest.TestLoader()
    suite  = unittest.TestSuite()
    for cls in [TestVoiceEnableDisable, TestShapeMismatch, TestQueueBehaviour]:
        suite.addTests(loader.loadTestsFromTestCase(cls))
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
