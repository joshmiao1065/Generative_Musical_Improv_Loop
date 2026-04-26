"""
audio_mixer.py — Real-time output mixer for user loop + AI voices.

Runs a WASAPI OutputStream callback that mixes:
  - The current user loop (played continuously, looping)
  - Up to 3 AI voice outputs (each queued as a full-loop numpy array)

Each AI voice has its own queue. When a new generation arrives from Modal,
it is put() into the voice's queue. On the next loop boundary, the mixer
swaps it in atomically. Voices that have no queued audio play silence.

Loop playback:
    The user loop plays on repeat. A `_loop_pos` counter tracks the current
    frame within the loop. At the end of each loop, the mixer checks each
    voice queue for a pending array and swaps it in — this is the "buffer
    pass" boundary. The timing_engine detects this boundary via the
    on_loop_boundary callback.

Thread safety:
    - _loop_audio and each _voice_audio[i] are replaced atomically (single
      assignment of a numpy array reference under the GIL — safe).
    - Voice queues use queue.Queue (thread-safe by design).
    - _loop_pos is read/written only by the audio callback thread.

Volume:
    Each voice has an independent gain (0.0–1.0) set by model_volume from
    GenerationParams. The user loop plays at user_volume (default 1.0).
    A simple peak-limiter prevents clipping on the final mix.
"""

import logging
import queue
import threading
from typing import Callable, List, Optional

import numpy as np
import sounddevice as sd

logger = logging.getLogger(__name__)

SAMPLE_RATE = 48000
CHANNELS    = 2
N_VOICES    = 3


class AudioMixer:
    """
    Real-time loop + AI voice mixer.

    Parameters
    ----------
    device_idx : int
        WASAPI output device index (from audio_devices.detect()).
    blocksize : int
        Frames per output callback. 512 = ~10ms at 48kHz.
        Lower = less output latency; higher = safer on slow machines.
    on_loop_boundary : callable, optional
        Called (from the audio thread) each time the loop restarts.
        The timing_engine hooks this to trigger the next generation pass.
        Keep it fast — it runs inside the audio callback.
    """

    def __init__(
        self,
        device_idx: int,
        blocksize: int = 512,
        on_loop_boundary: Optional[Callable] = None,
    ):
        self.device_idx       = device_idx
        self.blocksize        = blocksize
        self.on_loop_boundary = on_loop_boundary

        # User loop — replaced atomically when a new loop is recorded
        self._loop_audio: Optional[np.ndarray] = None   # (N, 2) float32
        self._loop_pos: int = 0
        self.user_volume: float = 1.0

        # Per-voice current audio and pending queue
        # _voice_audio[i]: currently playing array (None = silence)
        # _voice_queue[i]: queue of (audio_array, volume) waiting to swap in
        self._voice_audio: List[Optional[np.ndarray]] = [None] * N_VOICES
        self._voice_queue: List[queue.Queue] = [queue.Queue(maxsize=2) for _ in range(N_VOICES)]
        self._voice_volume: List[float] = [0.85] * N_VOICES
        # Voices start DISABLED — must be explicitly enabled via set_voice_enabled().
        # Buttons 2/3/4 on the PBF4 toggle each voice on/off.
        self._voice_enabled: List[bool] = [False, False, False]

        self._stream: Optional[sd.OutputStream] = None
        self._boundary_lock = threading.Lock()  # guards on_loop_boundary swap

        # Oneshot queue for metronome clicks (played once, not looped)
        self._oneshot_queue: queue.Queue = queue.Queue(maxsize=32)
        self._oneshot_pos: int = 0
        self._oneshot_cur: Optional[np.ndarray] = None

    # ── Stream control ────────────────────────────────────────────────────────

    def start(self):
        if self._stream is not None:
            return
        self._stream = sd.OutputStream(
            device=self.device_idx,
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype="float32",
            blocksize=self.blocksize,
            callback=self._audio_callback,
        )
        self._stream.start()
        logger.info("[AudioMixer] Started on device %d (blocksize=%d)", self.device_idx, self.blocksize)

    def stop(self):
        if self._stream is None:
            return
        self._stream.stop()
        self._stream.close()
        self._stream = None
        logger.info("[AudioMixer] Stopped.")

    # ── Loop control ──────────────────────────────────────────────────────────

    def set_loop(self, audio: np.ndarray):
        """
        Set the user loop. Swapped in at the next callback frame — effectively
        immediate (< one blocksize = ~10ms delay).

        Args:
            audio: (N, 2) float32 array at 48 kHz.
        """
        assert audio.ndim == 2 and audio.shape[1] == CHANNELS
        self._loop_audio = audio.astype(np.float32)
        self._loop_pos   = 0
        logger.info("[AudioMixer] Loop set: %.2fs", audio.shape[0] / SAMPLE_RATE)

    def clear_loop(self):
        """Stop loop playback (plays silence until set_loop is called again)."""
        self._loop_audio = None
        self._loop_pos   = 0

    # ── Voice control ─────────────────────────────────────────────────────────

    def queue_voice(self, voice_idx: int, audio: np.ndarray, volume: float = 0.85):
        """
        Queue a new AI voice output to be swapped in at the next loop boundary.

        If the queue is full (previous generation not yet consumed), the oldest
        entry is dropped to make room — the newest generation always wins.

        Args:
            voice_idx: 0, 1, or 2.
            audio:     (N, 2) float32 array, same length as the current loop.
            volume:    Output gain for this voice [0.0–1.0].
        """
        q = self._voice_queue[voice_idx]
        if q.full():
            try:
                q.get_nowait()
                logger.debug("[AudioMixer] Voice %d: dropped stale queued audio", voice_idx)
            except queue.Empty:
                pass
        q.put_nowait((audio.astype(np.float32), float(volume)))

    def set_voice_enabled(self, voice_idx: int, enabled: bool):
        """Enable or disable a voice (Button 2/3/4 toggle). Thread-safe."""
        self._voice_enabled[voice_idx] = enabled
        logger.info("[AudioMixer] Voice %d: %s", voice_idx, "ON" if enabled else "OFF")

    def set_voice_volume(self, voice_idx: int, volume: float):
        """Set per-voice output gain [0.0–1.0]. Thread-safe (float write is atomic)."""
        self._voice_volume[voice_idx] = max(0.0, min(1.0, volume))

    def play_oneshot(self, audio: np.ndarray):
        """
        Queue a short audio clip to be played once (not looped).
        Used for metronome clicks. Drops silently if queue is full.
        """
        try:
            self._oneshot_queue.put_nowait(audio.astype(np.float32))
        except queue.Full:
            pass

    def clear_voices(self):
        """Silence all voices and empty their queues (call on session reset)."""
        for i in range(N_VOICES):
            self._voice_audio[i] = None
            while not self._voice_queue[i].empty():
                try:
                    self._voice_queue[i].get_nowait()
                except queue.Empty:
                    break

    # ── Audio callback (runs on audio thread) ─────────────────────────────────

    def _audio_callback(self, outdata: np.ndarray, frames: int, time_info, status):
        if status:
            logger.warning("[AudioMixer] Stream status: %s", status)

        out = np.zeros((frames, CHANNELS), dtype=np.float32)
        loop = self._loop_audio

        # Oneshot (metronome clicks) — always processed, even before loop is set
        remaining_out = frames
        click_written = 0
        while remaining_out > 0:
            if self._oneshot_cur is None:
                try:
                    self._oneshot_cur = self._oneshot_queue.get_nowait()
                    self._oneshot_pos = 0
                except queue.Empty:
                    break
            clip      = self._oneshot_cur
            clip_left = clip.shape[0] - self._oneshot_pos
            n         = min(clip_left, remaining_out)
            out[click_written:click_written + n] += clip[self._oneshot_pos:self._oneshot_pos + n]
            self._oneshot_pos += n
            click_written     += n
            remaining_out     -= n
            if self._oneshot_pos >= clip.shape[0]:
                self._oneshot_cur = None

        if loop is None:
            outdata[:] = out
            return

        loop_len = loop.shape[0]
        written  = 0

        while written < frames:
            remaining_in_loop  = loop_len - self._loop_pos
            remaining_in_block = frames - written
            n = min(remaining_in_loop, remaining_in_block)

            # User loop
            out[written:written + n] += (
                loop[self._loop_pos:self._loop_pos + n] * self.user_volume
            )

            # AI voices
            for i in range(N_VOICES):
                if not self._voice_enabled[i]:
                    continue
                va = self._voice_audio[i]
                if va is None:
                    continue
                if va.shape[0] != loop_len:
                    # Stale audio from a previous loop length — will be corrected at
                    # the next boundary when a freshly-trimmed array is swapped in.
                    logger.warning("[AudioMixer] Voice %d length mismatch (%d vs %d) — "
                                   "skipping until next boundary", i, va.shape[0], loop_len)
                    continue
                out[written:written + n] += (
                    va[self._loop_pos:self._loop_pos + n] * self._voice_volume[i]
                )

            self._loop_pos += n
            written        += n

            # Loop boundary reached — swap in any pending voice audio
            if self._loop_pos >= loop_len:
                self._loop_pos = 0
                self._on_boundary()

        # Soft peak limiter — prevents clipping without hard distortion
        peak = np.max(np.abs(out))
        if peak > 0.95:
            out *= 0.95 / peak

        outdata[:] = out

    def _on_boundary(self):
        """Called at every loop boundary (from audio thread). Fast path only."""
        loop = self._loop_audio
        loop_len = loop.shape[0] if loop is not None else None

        # Swap in pending voice audio, trimming/padding to match the current loop length.
        # This corrects any length mismatch that arises when BPM or beat count is not
        # an exact multiple of CHUNK_SAMPLES (e.g. 90 BPM / 8 beats → 256k vs 288k).
        for i in range(N_VOICES):
            try:
                audio, vol = self._voice_queue[i].get_nowait()
                if loop_len is not None and audio.shape[0] != loop_len:
                    orig_len = audio.shape[0]
                    if audio.shape[0] > loop_len:
                        audio = audio[:loop_len]
                    else:
                        padded = np.zeros((loop_len, CHANNELS), dtype=np.float32)
                        padded[:audio.shape[0]] = audio
                        audio = padded
                    logger.debug("[AudioMixer] Voice %d: resized %d→%d samples at boundary",
                                 i, orig_len, loop_len)
                self._voice_audio[i] = audio
                self._voice_volume[i] = vol
                logger.debug("[AudioMixer] Voice %d swapped in at boundary", i)
            except queue.Empty:
                pass

        # Notify timing engine
        if self.on_loop_boundary is not None:
            try:
                self.on_loop_boundary()
            except Exception as e:
                logger.error("[AudioMixer] on_loop_boundary error: %s", e)


# ─────────────────────────────────────────────────────────────────────────────
# Standalone test: play a 440 Hz sine loop through the output device
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import time
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    from src.audio_devices import detect
    devs = detect()

    # 2-second 440 Hz sine wave as the "user loop"
    t       = np.linspace(0, 2.0, int(2.0 * SAMPLE_RATE), endpoint=False, dtype=np.float32)
    sine    = (np.sin(2 * np.pi * 440 * t) * 0.3).reshape(-1, 1)
    loop    = np.concatenate([sine, sine], axis=1)  # stereo

    # 2-second 660 Hz sine as a fake "AI voice"
    sine2   = (np.sin(2 * np.pi * 660 * t) * 0.2).reshape(-1, 1)
    voice   = np.concatenate([sine2, sine2], axis=1)

    boundary_count = [0]
    def on_boundary():
        boundary_count[0] += 1
        logger.info("[test] Loop boundary #%d", boundary_count[0])

    mixer = AudioMixer(device_idx=devs.playback_idx, on_loop_boundary=on_boundary)
    mixer.start()
    mixer.set_loop(loop)

    print("Playing 440 Hz loop for 4 seconds...")
    time.sleep(4.0)

    print("Queuing 660 Hz AI voice...")
    mixer.queue_voice(0, voice, volume=0.8)
    time.sleep(4.0)

    print("Disabling voice 0...")
    mixer.set_voice_enabled(0, False)
    time.sleep(2.0)

    mixer.stop()
    print(f"Done. {boundary_count[0]} loop boundaries crossed.")
