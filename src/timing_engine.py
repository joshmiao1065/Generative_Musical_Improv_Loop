"""
timing_engine.py — Absolute-time loop clock and metronome.

Responsibilities:
  - Track the current pass number and loop position using time.perf_counter()
  - Fire the on_pass_end callback at each loop boundary (triggers Modal generation)
  - Play a metronome click track during the 2-bar count-in before recording
  - Provide beat_phase() for any UI that wants to know position within the loop

Why perf_counter and not sleep():
    time.sleep() on Windows has ~15ms jitter. Over many passes this accumulates.
    Instead, the engine computes the absolute time of each future beat/boundary
    from a fixed anchor and spins with a short sleep until that time arrives.
    Drift is bounded to one perf_counter resolution (~100ns) per boundary.

Integration with AudioMixer:
    The mixer's on_loop_boundary callback (fired from the audio thread) is the
    ground truth for when the loop actually restarted in audio time. The timing
    engine listens to this via notify_boundary() and uses it to re-anchor its
    clock, preventing the timer from drifting away from the actual audio output.

Count-in:
    start_countdown(beats) plays `beats` click tones through the mixer, then
    fires on_countdown_done. The session calls snapshot() on LoopCapture when
    it receives on_countdown_done, marks anchor_time, and the engine begins
    tracking pass boundaries from there.
"""

import logging
import math
import threading
import time
from typing import Callable, Optional

import numpy as np

logger = logging.getLogger(__name__)

SAMPLE_RATE = 48000


def _make_click(freq: float = 1000.0, duration: float = 0.02,
                volume: float = 0.4, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Short sine-burst click for the metronome."""
    n   = int(duration * sample_rate)
    t   = np.linspace(0, duration, n, endpoint=False, dtype=np.float32)
    env = np.sin(np.linspace(0, np.pi, n, dtype=np.float32))   # half-sine envelope
    mono = np.sin(2 * np.pi * freq * t) * env * volume
    return np.stack([mono, mono], axis=1)   # (N, 2)


# Pre-render two click tones
_CLICK_ACCENT  = _make_click(freq=1400.0, volume=0.5)   # beat 1
_CLICK_NORMAL  = _make_click(freq=900.0,  volume=0.3)   # beats 2-4


class TimingEngine:
    """
    Loop clock, metronome, and pass-boundary coordinator.

    Parameters
    ----------
    bpm : float
        Tempo in beats per minute.
    beats_per_loop : int
        Number of beats in one user loop (also one generation pass).
    on_pass_end : callable
        Called at each loop boundary. Receives pass_number (int).
        This is where improv_loop.py triggers the next Modal generation.
        Runs in the timing thread — hand off work via queue or Event.
    on_countdown_done : callable
        Called when the count-in finishes and recording should start.
    """

    def __init__(
        self,
        bpm: float,
        beats_per_loop: int,
        on_pass_end: Callable[[int], None],
        on_countdown_done: Callable[[], None],
    ):
        self.bpm             = bpm
        self.beats_per_loop  = beats_per_loop
        self.on_pass_end     = on_pass_end
        self.on_countdown_done = on_countdown_done

        self._beat_dur       = 60.0 / bpm
        self._loop_dur       = beats_per_loop * self._beat_dur

        self._anchor: Optional[float] = None   # perf_counter time of loop start
        self._pass_number: int = 0

        self._running        = threading.Event()
        self._thread: Optional[threading.Thread] = None

        # Metronome click injection — set by caller before start_countdown
        self._mixer_queue_fn: Optional[Callable] = None  # mixer.queue_voice or direct inject

    # ── Configuration ──────────────────────────────────────────────────────────

    def set_tempo(self, bpm: float, beats_per_loop: int):
        """Update tempo and loop length (takes effect on next pass)."""
        self.bpm            = bpm
        self.beats_per_loop = beats_per_loop
        self._beat_dur      = 60.0 / bpm
        self._loop_dur      = beats_per_loop * self._beat_dur
        logger.info("[Timing] Tempo updated: %.1f BPM, %d beats (%.2fs loop)",
                    bpm, beats_per_loop, self._loop_dur)

    def set_click_output(self, fn: Callable[[np.ndarray], None]):
        """
        Register a function that receives click audio chunks (N, 2) float32.
        Typically: lambda audio: mixer.queue_click(audio)
        The click is injected directly into the output stream, not queued as
        a voice (so it doesn't get swapped out at loop boundaries).
        """
        self._mixer_queue_fn = fn

    # ── Count-in + recording trigger ───────────────────────────────────────────

    def start_countdown(self, count_beats: Optional[int] = None):
        """
        Play a metronome count-in of `count_beats` beats, then fire
        on_countdown_done. Defaults to 2 full bars (2 × beats_per_loop)
        but capped at a reasonable maximum (2 bars).

        Runs in a new thread so it doesn't block the caller.
        """
        if count_beats is None:
            # 1 bar (= beats_per_loop), capped at 8 beats max
            count_beats = min(self.beats_per_loop, 8)

        t = threading.Thread(target=self._countdown_loop, args=(count_beats,),
                             daemon=True, name="Timing-Countdown")
        t.start()

    def _countdown_loop(self, count_beats: int):
        beats_per_bar = self.beats_per_loop  # treat loop length as one bar

        logger.info("[Timing] Count-in: %d beats (%.1f BPM)", count_beats, self.bpm)
        t0      = time.perf_counter()
        beat_num = 0

        while beat_num < count_beats:
            target = t0 + beat_num * self._beat_dur
            _wait_until(target)

            pos_in_bar = beat_num % beats_per_bar
            click = _CLICK_ACCENT if pos_in_bar == 0 else _CLICK_NORMAL
            if self._mixer_queue_fn:
                try:
                    self._mixer_queue_fn(click)
                except Exception as e:
                    logger.error("[Timing] Click inject error: %s", e)

            beat_num += 1

        # Fire at the exact moment recording should begin
        record_start = t0 + count_beats * self._beat_dur
        _wait_until(record_start)
        self._anchor = record_start
        self._pass_number = 0
        logger.info("[Timing] Count-in done. Recording starts now.")
        try:
            self.on_countdown_done()
        except Exception as e:
            logger.error("[Timing] on_countdown_done error: %s", e)

    # ── Pass boundary tracking ─────────────────────────────────────────────────

    def start(self):
        """
        Start the pass-boundary tracking thread.
        Call after on_countdown_done has set self._anchor.
        """
        if self._thread and self._thread.is_alive():
            return
        self._running.set()
        self._thread = threading.Thread(target=self._boundary_loop,
                                        daemon=True, name="Timing-Boundary")
        self._thread.start()
        logger.info("[Timing] Boundary tracker started.")

    def stop(self):
        self._running.clear()
        if self._thread:
            self._thread.join(timeout=2.0)
        logger.info("[Timing] Stopped.")

    def _boundary_loop(self):
        """Wait for each successive loop end, then fire on_pass_end."""
        while self._running.is_set():
            if self._anchor is None:
                time.sleep(0.005)
                continue

            next_boundary = self._anchor + (self._pass_number + 1) * self._loop_dur
            _wait_until(next_boundary)

            if not self._running.is_set():
                break

            self._pass_number += 1
            logger.debug("[Timing] Pass %d ended (t=%.3f)",
                         self._pass_number, time.perf_counter() - self._anchor)
            try:
                self.on_pass_end(self._pass_number)
            except Exception as e:
                logger.error("[Timing] on_pass_end error: %s", e)

    def notify_boundary(self):
        """
        Called by AudioMixer.on_loop_boundary to re-anchor the clock to
        the actual audio output boundary. Prevents timer drift from
        accumulating over many passes.
        """
        if self._anchor is not None:
            now = time.perf_counter()
            expected = self._anchor + self._pass_number * self._loop_dur
            drift = now - expected
            if abs(drift) > 0.010:  # >10ms drift — re-anchor
                self._anchor += drift
                logger.debug("[Timing] Re-anchored: drift was %.1f ms", drift * 1000)

    # ── Position query ─────────────────────────────────────────────────────────

    def beat_phase(self) -> float:
        """
        Current position within the loop as a fraction [0.0, 1.0).
        0.0 = start of loop, 0.5 = halfway through.
        Returns 0.0 if not yet started.
        """
        if self._anchor is None:
            return 0.0
        elapsed = (time.perf_counter() - self._anchor) % self._loop_dur
        return elapsed / self._loop_dur

    def current_beat(self) -> int:
        """Current beat number within the loop (0-indexed)."""
        if self._anchor is None:
            return 0
        elapsed = (time.perf_counter() - self._anchor) % self._loop_dur
        return int(elapsed / self._beat_dur)

    @property
    def loop_duration(self) -> float:
        return self._loop_dur


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _wait_until(target: float, spin_threshold: float = 0.002):
    """
    Sleep until perf_counter() >= target.
    Uses sleep() for coarse waiting, then spins for the last 2ms
    to avoid OS scheduler jitter on the final approach.
    """
    remaining = target - time.perf_counter()
    if remaining > spin_threshold:
        time.sleep(remaining - spin_threshold)
    while time.perf_counter() < target:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# Standalone test: count-in + 4 pass boundaries at 120 BPM, 4 beats/loop
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s.%(msecs)03d  %(message)s",
                        datefmt="%H:%M:%S")

    from src.audio_devices import detect
    from src.audio_mixer   import AudioMixer

    devs  = detect()
    mixer = AudioMixer(device_idx=devs.playback_idx, blocksize=512)
    mixer.start()

    # Silence loop so mixer has something to play
    silence = np.zeros((int(2.0 * SAMPLE_RATE), 2), dtype=np.float32)
    mixer.set_loop(silence)

    passes = []

    def on_pass_end(n):
        t = time.perf_counter()
        passes.append(t)
        logger.info("[test] Pass %d ended", n)

    done_evt = threading.Event()
    def on_countdown_done():
        logger.info("[test] Recording! (count-in complete)")
        engine.start()
        done_evt.set()

    engine = TimingEngine(
        bpm=120, beats_per_loop=4,
        on_pass_end=on_pass_end,
        on_countdown_done=on_countdown_done,
    )

    # Wire click output to a direct play (inject into mixer's loop slot for test)
    click_buf = []
    engine.set_click_output(lambda c: click_buf.append(len(c)))

    # Wire mixer boundary → timing re-anchor
    mixer.on_loop_boundary = engine.notify_boundary

    print("Starting 4-beat count-in at 120 BPM...")
    engine.start_countdown(count_beats=4)
    done_evt.wait()

    # Run for 4 passes (4 beats each = 2s per pass at 120 BPM)
    time.sleep(8.5)
    engine.stop()
    mixer.stop()

    print(f"\n{len(passes)} pass boundaries fired.")
    if len(passes) >= 2:
        intervals = [passes[i+1]-passes[i] for i in range(len(passes)-1)]
        print(f"Inter-pass intervals: {[f'{x:.3f}s' for x in intervals]}")
        print(f"Expected: 2.000s each (120 BPM, 4 beats)")
        max_err = max(abs(x - 2.0) for x in intervals)
        print(f"Max timing error: {max_err*1000:.1f} ms")
