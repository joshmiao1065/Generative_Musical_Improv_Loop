"""
loop_capture.py — Continuous audio capture with on-demand loop snapshot.

Runs a callback-based InputStream (required for WDM-KS / Stereo Mix) that
feeds into a fixed-size ring buffer. When the session state machine calls
snapshot(), the last `loop_samples` samples are returned as a numpy array
— that becomes the user loop sent to Modal.

Design rationale:
    The ring buffer rolls continuously from the moment start() is called.
    Recording is not "armed" — audio is always captured. When Button 1 is
    pressed, the state machine waits exactly `loop_duration` seconds, then
    calls snapshot(). The snapshot grabs the most recent chunk from the
    buffer. This avoids any startup gap between button-press and capture-start.

Beat alignment:
    snapshot() accepts an optional `align_to_beat` flag. When True, the
    captured length is rounded to the nearest complete beat boundary based
    on bpm, minimising the drift between the loop length and a whole number
    of beats. The session decides whether to use this.

Thread safety:
    The sounddevice callback writes to the ring buffer from the audio thread.
    snapshot() reads from the main/session thread. A threading.Lock protects
    the write pointer and the buffer slice. The lock is held only for the
    numpy copy — kept as short as possible to avoid blocking the audio thread.
"""

import logging
import threading
import time
from typing import Optional

import numpy as np
import sounddevice as sd

logger = logging.getLogger(__name__)

SAMPLE_RATE = 48000
CHANNELS    = 2


class _MonitorFIFO:
    """
    Streaming FIFO for monitoring passthrough between input and output callbacks.

    The input callback (e.g. LoopCapture at blocksize=2048) writes large chunks;
    the output callback (AudioMixer at blocksize=512) reads small chunks.
    Without a FIFO, the output would read the same frames repeatedly, causing
    pitch and timbre distortion.  This FIFO properly serialises the stream.

    Capacity: 4096 frames (~85ms at 48kHz).  If the FIFO fills, the oldest
    frames are discarded to prevent unbounded latency build-up.
    """

    _SIZE = 4096

    def __init__(self):
        self._buf  = np.zeros((self._SIZE, CHANNELS), dtype=np.float32)
        self._wp   = 0   # write position
        self._rp   = 0   # read position
        self._n    = 0   # frames currently available
        self._lock = threading.Lock()

    def write(self, frames: np.ndarray) -> None:
        n = frames.shape[0]
        with self._lock:
            overflow = self._n + n - self._SIZE
            if overflow > 0:
                self._rp = (self._rp + overflow) % self._SIZE
                self._n -= overflow
            end = self._wp + n
            if end <= self._SIZE:
                self._buf[self._wp:end] = frames
            else:
                f = self._SIZE - self._wp
                self._buf[self._wp:] = frames[:f]
                self._buf[:end - self._SIZE] = frames[f:]
            self._wp = end % self._SIZE
            self._n += n

    def read(self, n: int) -> np.ndarray:
        out = np.zeros((n, CHANNELS), dtype=np.float32)
        with self._lock:
            n_read = min(n, self._n)
            if n_read == 0:
                return out
            end = self._rp + n_read
            if end <= self._SIZE:
                out[:n_read] = self._buf[self._rp:end]
            else:
                f = self._SIZE - self._rp
                out[:f] = self._buf[self._rp:]
                out[f:n_read] = self._buf[:end - self._SIZE]
            self._rp = end % self._SIZE
            self._n -= n_read
        return out


class LoopCapture:
    """
    Continuous ring-buffer audio capture with on-demand snapshot.

    Parameters
    ----------
    device_idx : int
        sounddevice input device index (from audio_devices.detect()).
    max_loop_seconds : float
        Maximum loop duration supported. Ring buffer is sized to hold this
        much audio. Should be >= the longest loop you'll ever record.
    blocksize : int
        Frames per audio callback. 2048 = ~42ms at 48kHz (low-latency default).
    sample_rate : int
        Must match the device's native rate (48000).
    """

    def __init__(
        self,
        device_idx: int,
        max_loop_seconds: float = 32.0,
        blocksize: int = 2048,
        sample_rate: int = SAMPLE_RATE,
    ):
        self.device_idx       = device_idx
        self.sample_rate      = sample_rate
        self.blocksize        = blocksize
        self._max_samples     = int(max_loop_seconds * sample_rate)

        # Ring buffer: pre-allocated, written circularly
        self._buf             = np.zeros((self._max_samples, CHANNELS), dtype=np.float32)
        self._write_pos       = 0          # next frame index to write into
        self._frames_captured = 0          # total frames written (monotonic)
        self._lock            = threading.Lock()

        self._stream: Optional[sd.InputStream] = None
        self._capture_start_time: Optional[float] = None

        # Streaming FIFO for live monitoring passthrough to AudioMixer output callback
        self._monitor_fifo = _MonitorFIFO()

    # ── Stream control ────────────────────────────────────────────────────────

    def start(self):
        """Open the input stream and begin filling the ring buffer."""
        if self._stream is not None:
            logger.warning("[LoopCapture] Already running.")
            return

        self._stream = sd.InputStream(
            device=self.device_idx,
            samplerate=self.sample_rate,
            channels=CHANNELS,
            dtype="float32",
            blocksize=self.blocksize,
            callback=self._audio_callback,
        )
        self._stream.start()
        self._capture_start_time = time.perf_counter()
        logger.info("[LoopCapture] Started on device %d (max %.1fs buffer)",
                    self.device_idx, self._max_samples / self.sample_rate)

    def stop(self):
        """Stop and close the input stream."""
        if self._stream is None:
            return
        self._stream.stop()
        self._stream.close()
        self._stream = None
        logger.info("[LoopCapture] Stopped.")

    # ── Audio callback (runs on audio thread) ─────────────────────────────────

    def _audio_callback(self, indata: np.ndarray, frames: int, time_info, status):
        if status:
            logger.warning("[LoopCapture] Stream status: %s", status)

        with self._lock:
            end = self._write_pos + frames
            if end <= self._max_samples:
                self._buf[self._write_pos:end] = indata
            else:
                # Wrap around the ring buffer
                first = self._max_samples - self._write_pos
                self._buf[self._write_pos:] = indata[:first]
                self._buf[:end - self._max_samples] = indata[first:]
            self._write_pos    = end % self._max_samples
            self._frames_captured += frames

        # Write to monitoring FIFO outside the ring buffer lock so the two
        # locks are never nested (prevents potential deadlock with AudioMixer).
        self._monitor_fifo.write(indata)

    # ── Snapshot ──────────────────────────────────────────────────────────────

    def snapshot(
        self,
        loop_seconds: float,
        align_to_beat: bool = False,
        bpm: float = 120.0,
    ) -> np.ndarray:
        """
        Return the most recent `loop_seconds` of captured audio as a numpy array.

        Parameters
        ----------
        loop_seconds : float
            Desired loop duration in seconds.
        align_to_beat : bool
            If True, trim or extend loop_seconds to the nearest whole beat.
            Useful to keep the loop length a round number of beats even if
            the button was pressed slightly early or late.
        bpm : float
            Tempo in BPM — used only when align_to_beat=True.

        Returns
        -------
        np.ndarray
            Shape (N, 2), float32, 48 kHz stereo.
            N = loop_seconds * sample_rate (possibly adjusted for beat alignment).

        Raises
        ------
        RuntimeError
            If the stream hasn't been started yet, or if the requested
            duration exceeds the ring buffer capacity.
        """
        if self._stream is None:
            raise RuntimeError("LoopCapture.start() must be called before snapshot().")

        if align_to_beat:
            beat_seconds  = 60.0 / bpm
            n_beats       = max(1, round(loop_seconds / beat_seconds))
            loop_seconds  = n_beats * beat_seconds
            logger.debug("[LoopCapture] Beat-aligned to %d beats = %.3fs", n_beats, loop_seconds)

        n_samples = int(loop_seconds * self.sample_rate)
        if n_samples > self._max_samples:
            raise RuntimeError(
                f"Requested {loop_seconds:.1f}s exceeds ring buffer "
                f"({self._max_samples / self.sample_rate:.1f}s max)."
            )

        with self._lock:
            available = min(self._frames_captured, self._max_samples)
            if available < n_samples:
                # Not enough audio captured yet — pad with silence at the front
                logger.warning(
                    "[LoopCapture] Only %.2fs available, requested %.2fs — "
                    "padding front with silence.",
                    available / self.sample_rate, loop_seconds,
                )
                chunk = np.zeros((n_samples, CHANNELS), dtype=np.float32)
                src = self._read_last(available)
                chunk[n_samples - available:] = src
                return chunk

            return self._read_last(n_samples).copy()

    def _read_last(self, n: int) -> np.ndarray:
        """
        Read the last `n` frames from the ring buffer.
        Caller must hold self._lock.
        """
        end = self._write_pos
        start = end - n
        if start >= 0:
            return self._buf[start:end]
        # Wrap: two slices
        tail = self._buf[start:]   # start is negative, so this is buf[max+start:]
        head = self._buf[:end]
        return np.concatenate([tail, head], axis=0)

    def read_monitor_frames(self, n: int) -> np.ndarray:
        """
        Drain n frames from the monitor FIFO for real-time output passthrough.
        Called from AudioMixer's output callback thread.  Returns silence if
        the FIFO hasn't filled yet (e.g. on first callback after start).
        """
        return self._monitor_fifo.read(n)

    # ── Diagnostics ───────────────────────────────────────────────────────────

    @property
    def seconds_captured(self) -> float:
        """Total seconds of audio captured since start() (capped at buffer size)."""
        return min(self._frames_captured, self._max_samples) / self.sample_rate

    @property
    def rms(self) -> float:
        """RMS level of the entire current ring buffer (rough input level check)."""
        with self._lock:
            return float(np.sqrt(np.mean(self._buf ** 2)))


# ─────────────────────────────────────────────────────────────────────────────
# Standalone test: capture 4 seconds from Stereo Mix, print RMS, save wav
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))

    import soundfile as sf
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    from src.audio_devices import detect
    devs = detect()
    print(devs)
    print("\nPlay something in Surge XT now...")
    print("Capturing 4 seconds, then snapshotting a 2-second loop.\n")

    cap = LoopCapture(device_idx=devs.capture_idx)
    cap.start()
    time.sleep(4.0)

    print(f"Buffer RMS: {cap.rms:.4f}  ({cap.seconds_captured:.2f}s captured)")

    loop = cap.snapshot(loop_seconds=2.0)
    loop_aligned = cap.snapshot(loop_seconds=2.0, align_to_beat=True, bpm=120.0)

    cap.stop()

    rms = float(np.sqrt(np.mean(loop ** 2)))
    print(f"Snapshot shape: {loop.shape}  RMS: {rms:.4f}")
    if rms < 0.001:
        print("WARNING: Very quiet — is Surge XT playing and Stereo Mix enabled?")
    else:
        print("Audio captured OK.")

    out = Path(__file__).parent.parent / "scripts" / "loop_capture_test.wav"
    sf.write(str(out), loop, SAMPLE_RATE)
    print(f"Saved: {out}")
