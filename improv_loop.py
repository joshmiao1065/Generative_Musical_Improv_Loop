"""
improv_loop.py — Main session orchestrator for the Improv Loop system.

Usage:
    python improv_loop.py --bpm 120 --beats 16
    python improv_loop.py --bpm 90 --beats 8 --voices 2

    # Debug without Modal (uses sine tones as AI placeholder):
    python improv_loop.py --dry-run

    # Simulate Modal latency during dry-run (~5.7s matches real A100 timing):
    python improv_loop.py --dry-run --dry-run-latency 5.7

    # List audio and MIDI devices then exit:
    python improv_loop.py --list-devices

    # Force a specific capture/playback device by name substring:
    python improv_loop.py --capture-device "CABLE Output"   # VB-Cable
    python improv_loop.py --capture-device "Stereo Mix"     # fallback
    python improv_loop.py --playback-device "Headphones"

    # Override by device index (from --list-devices output):
    python improv_loop.py --capture-idx 13 --playback-idx 4

    # Save every captured loop to WAV for post-session analysis:
    python improv_loop.py --save-loops ./debug_loops

    # Verbose debug logging:
    python improv_loop.py --log-level debug

    # Disable metronome click (silent count-in):
    python improv_loop.py --no-click

Controls (PBF4):
    Button 1  — record_toggle: first press starts count-in then records;
                press while recording ends it early; press while playing restarts.
    Button 2  — voice_1_toggle: enable/disable AI Voice 1  (starts DISABLED)
    Button 3  — voice_2_toggle: enable/disable AI Voice 2  (starts DISABLED)
    Button 4  — voice_3_toggle: enable/disable AI Voice 3  (starts DISABLED)
    Faders    — genre blend weights (passed to Modal as style context)
    Knob 1    — guidance_weight  [0–10]
    Knob 2    — temperature      [0–2]
    Knob 3    — topk             [0–100]
    Knob 4    — model_feedback   [0–1]

State machine:
    IDLE → [Button 1] → COUNTDOWN
    COUNTDOWN → [done] → RECORDING
    COUNTDOWN → [Button 1] → IDLE
    RECORDING → [loop elapsed OR Button 1] → PLAYING
    PLAYING → [Button 1] → COUNTDOWN  (AI voices continue uninterrupted)
    PLAYING → [Ctrl-C] → STOPPING → IDLE

Synth routing (hardware — no code change needed when switching synths):
    Surge XT:    Preferences → Audio → Output: "CABLE Input"
    Analog Lab:  Settings → Audio → Output device: "CABLE Input"
    Any DAW:     Master output → routed to "CABLE Input" (audio device in DAW prefs)
    Python always captures from "CABLE Output" (VB-Cable) or Stereo Mix (fallback).
"""

import argparse
import asyncio
import dataclasses
import logging
import signal
import sys
import threading
import time
import types
from enum import Enum, auto
from pathlib import Path
from typing import List, Optional

import numpy as np

# ── Project imports ───────────────────────────────────────────────────────────
from src.audio_devices   import detect as detect_devices, list_devices, list_midi_ports
from src.audio_mixer     import AudioMixer
from src.loop_capture    import LoopCapture
from src.midi_controller import PBF4Controller
from src.timing_engine   import TimingEngine

# GenerationParams lives in magenta_backend but improv_loop doesn't need JAX.
# Import it directly; if JAX isn't available on this machine, use the stub.
try:
    from src.magenta_backend import GenerationParams
except ImportError:
    @dataclasses.dataclass
    class GenerationParams:
        guidance_weight: float = 3.0
        temperature:     float = 1.0
        topk:            int   = 40
        model_feedback:  float = 0.7
        model_volume:    float = 0.85
        beats_per_loop:  int   = 16
        bpm:             int   = 120

from src.modal_client import MagentaRTClient

logger = logging.getLogger(__name__)

_DRY_RUN_FREQS = [440.0, 554.37, 659.25]   # A4, C#5, E5 — A-major chord, one per voice


# ─────────────────────────────────────────────────────────────────────────────
# State machine
# ─────────────────────────────────────────────────────────────────────────────

class State(Enum):
    IDLE       = auto()
    COUNTDOWN  = auto()
    RECORDING  = auto()
    PLAYING    = auto()
    STOPPING   = auto()


# ─────────────────────────────────────────────────────────────────────────────
# Session
# ─────────────────────────────────────────────────────────────────────────────

class ImprovSession:

    def __init__(self, args):
        self.args    = args
        self.state   = State.IDLE
        self._lock   = threading.Lock()   # guards state transitions

        # Shared generation params — written by MIDI thread, read by main loop
        self.params = GenerationParams(
            bpm=args.bpm,
            beats_per_loop=args.beats,
            guidance_weight=3.0,
            temperature=1.0,
            topk=40,
            model_feedback=0.7,
        )

        # Current user loop audio (numpy array, set after first recording)
        self._user_loop: Optional[np.ndarray] = None

        # asyncio event loop running in a background thread for Modal calls
        self._aio_loop = asyncio.new_event_loop()
        self._aio_thread = threading.Thread(
            target=self._aio_loop.run_forever, daemon=True, name="AsyncIO"
        )

        # Pass tracking for cascade voice entry schedule
        self._pass_number = 0

        # Signal main thread to exit
        self._stop_event = threading.Event()

    # ── Setup ─────────────────────────────────────────────────────────────────

    def setup(self):
        logger.info("[Session] Initialising hardware and audio devices...")

        self.devs = detect_devices(
            capture_idx=getattr(self.args, "capture_idx",  None),
            playback_idx=getattr(self.args, "playback_idx", None),
            capture_name=getattr(self.args, "capture_device",  None),
            playback_name=getattr(self.args, "playback_device", None),
        )
        self.capture = LoopCapture(
            device_idx=self.devs.capture_idx,
            max_loop_seconds=64.0,
        )
        self.mixer = AudioMixer(
            device_idx=self.devs.playback_idx,
            blocksize=512,
            on_loop_boundary=self._on_loop_boundary,
        )
        self.engine = TimingEngine(
            bpm=self.args.bpm,
            beats_per_loop=self.args.beats,
            on_pass_end=self._on_pass_end,
            on_countdown_done=self._on_countdown_done,
        )
        if not self.args.no_click:
            self.engine.set_click_output(self.mixer.play_oneshot)

        self.ctrl = PBF4Controller(
            self.params,
            layout_path="config/pbf4_layout.json",
        )
        self.ctrl.on("record_toggle",  self._handle_record_toggle)
        self.ctrl.on("voice_1_toggle", lambda: self._handle_voice_toggle(0))
        self.ctrl.on("voice_2_toggle", lambda: self._handle_voice_toggle(1))
        self.ctrl.on("voice_3_toggle", lambda: self._handle_voice_toggle(2))

        if not self.args.dry_run:
            self.client = MagentaRTClient(
                n_voices=self.args.voices,
                bpm=self.args.bpm,
                beats_per_loop=self.args.beats,
            )
        else:
            self.client = None
            logger.info("[Session] DRY-RUN mode — Modal calls skipped; AI voices replaced "
                        "with %.0f Hz / %.0f Hz / %.0f Hz sine tones.",
                        *_DRY_RUN_FREQS)

        # Start asyncio thread (used for Modal calls and warmup ping)
        self._aio_thread.start()

        # Start hardware immediately — don't block on container warm-up
        self.capture.start()
        self.mixer.start()
        self.ctrl.start()

        logger.info("[Session] Setup complete. BPM=%d  beats=%d  voices=%d  dry_run=%s",
                    self.args.bpm, self.args.beats, self.args.voices, self.args.dry_run)

        if not self.args.dry_run:
            # Ping containers in background — logs status as each responds
            asyncio.run_coroutine_threadsafe(self._warm_containers(), self._aio_loop)
        else:
            logger.info("[Session] Skipping Modal warmup (dry-run).")

    # ── Main loop ─────────────────────────────────────────────────────────────

    def run(self):
        """Block until Ctrl-C or stop is called."""
        # Install signal handler so Ctrl-C works even during asyncio or blocking calls
        signal.signal(signal.SIGINT, self._handle_sigint)

        self._print_banner()
        self._stop_event.wait()
        self._shutdown()

    def _handle_sigint(self, sig, frame):
        print("\n[Ctrl-C] Stopping session...")
        self._stop_event.set()

    def _shutdown(self):
        logger.info("[Session] Shutting down...")
        self.engine.stop()
        self.mixer.stop()
        self.capture.stop()
        self.ctrl.stop()
        self._aio_loop.call_soon_threadsafe(self._aio_loop.stop)
        logger.info("[Session] Goodbye.")

    # ── Button callbacks (called from MIDI thread) ────────────────────────────

    def _handle_record_toggle(self):
        with self._lock:
            s = self.state
        if s == State.IDLE:
            self._transition(State.COUNTDOWN)
            self.engine.start_countdown()
        elif s == State.COUNTDOWN:
            # Cancel count-in — back to idle
            self._transition(State.IDLE)
            logger.info("[Session] Count-in cancelled.")
        elif s == State.RECORDING:
            # Stop recording early — commit whatever is in the buffer
            self._finish_recording()
        elif s == State.PLAYING:
            # Stop the loop and all voices — return to IDLE for a fresh start
            self._stop_loop()

    def _stop_loop(self):
        """Stop loop playback, silence all voices, reset to IDLE."""
        self._transition(State.STOPPING)
        self.engine.stop()
        self.mixer.clear_loop()
        self.mixer.clear_voices()
        self._user_loop = None
        self._pass_number = 0
        asyncio.run_coroutine_threadsafe(self.client.reset(), self._aio_loop)
        self._transition(State.IDLE)
        logger.info("[Session] Stopped. Press Button 1 to record a new loop.")

    def _handle_voice_toggle(self, idx: int):
        label   = f"voice_{idx+1}_toggle"
        enabled = self.ctrl.get_toggle(label)
        self.mixer.set_voice_enabled(idx, enabled)
        logger.info("[Session] Voice %d: %s", idx + 1, "ON" if enabled else "OFF")

    # ── Timing callbacks ──────────────────────────────────────────────────────

    def _on_countdown_done(self):
        """Called by timing engine when count-in completes. Start recording."""
        self._transition(State.RECORDING)
        logger.info("[Session] Recording... (%.1fs loop at %d BPM)",
                    self.args.beats * 60.0 / self.args.bpm, self.args.bpm)
        # Schedule end of recording
        loop_dur = self.args.beats * 60.0 / self.args.bpm
        t = threading.Timer(loop_dur, self._finish_recording)
        t.daemon = True
        t.start()

    def _finish_recording(self):
        with self._lock:
            if self.state != State.RECORDING:
                return
        loop = self.capture.snapshot(
            loop_seconds=self.args.beats * 60.0 / self.args.bpm,
            align_to_beat=True,
            bpm=self.args.bpm,
        )
        self._user_loop = loop
        self.mixer.set_loop(loop)
        self._pass_number = 0

        rms = float(np.sqrt(np.mean(loop ** 2)))
        logger.info("[Session] Loop captured: %.2fs  %d samples  RMS=%.4f",
                    loop.shape[0] / 48000, loop.shape[0], rms)
        if rms < 0.001:
            logger.warning("[Session] RMS is very low (%.4f) — is the synth playing and "
                           "routed to the capture device?  Check --capture-device.", rms)

        if self.args.save_loops:
            self._save_loop_wav(loop)

        self._transition(State.PLAYING)
        self.engine.start()

    def _on_loop_boundary(self):
        """Called from audio thread at every loop restart. Re-anchor timer."""
        self.engine.notify_boundary()

    def _on_pass_end(self, pass_num: int):
        """Called by timing engine at each loop boundary. Dispatch Modal generation."""
        with self._lock:
            if self.state != State.PLAYING:
                return
            loop = self._user_loop

        if loop is None:
            return

        self._pass_number = pass_num
        logger.info("[Session] Pass %d started — dispatching generation", pass_num)
        asyncio.run_coroutine_threadsafe(
            self._generate_pass(loop, pass_num),
            self._aio_loop,
        )

    # ── Modal warm-up + generation (runs in asyncio thread) ───────────────────

    async def _warm_containers(self):
        """Ping all Modal containers in parallel and log each response as it arrives."""
        logger.info("[Modal] Pinging %d container(s) in parallel — "
                    "cold start (XLA compile) takes ~5-8 min on first boot...",
                    self.args.voices)

        async def _ping_one(i: int):
            logger.info("[Modal] Waiting for Voice %d container...", i + 1)
            try:
                result = await asyncio.wait_for(
                    self.client._voices[i].ping.remote.aio(),
                    timeout=600,   # 10 min: covers cold start + XLA compile
                )
                logger.info("[Modal] Voice %d ready: %s", i + 1, result)
            except asyncio.TimeoutError:
                logger.error("[Modal] Voice %d timed out (>10 min). "
                             "Check modal.com/apps for errors.", i + 1)
            except Exception as e:
                logger.error("[Modal] Voice %d error: %s", i + 1, e)

        try:
            # All voices pinged simultaneously — warm-up time = max(V0,V1,V2), not sum
            await asyncio.gather(*[_ping_one(i) for i in range(self.args.voices)])
            logger.info("[Modal] All containers checked. Press Button 1 to begin.")
        except Exception as e:
            logger.error("[Modal] Warm-up error: %s", e)

    async def _generate_pass(self, user_loop: np.ndarray, pass_num: int):
        """Dispatch one generation pass (Modal or dry-run) and queue results into mixer."""
        p = self.params

        if self.args.dry_run:
            voice_outputs = self._dry_run_voices(user_loop)
            if self.args.dry_run_latency > 0:
                logger.debug("[DryRun] Simulating %.1fs Modal latency...",
                             self.args.dry_run_latency)
                await asyncio.sleep(self.args.dry_run_latency)
        else:
            try:
                voice_outputs = await self.client.generate_pass(
                    user_loop,
                    guidance_weight=p.guidance_weight,
                    temperature=p.temperature,
                    topk=p.topk,
                    model_feedback=p.model_feedback,
                    beats_per_loop=self.args.beats,
                    bpm=self.args.bpm,
                )
            except Exception as e:
                logger.error("[Session] Modal generation error (pass %d): %s", pass_num, e)
                return

        for i, audio in enumerate(voice_outputs):
            # Voice entry schedule: V0 joins pass 2, V1 pass 3, V2 pass 4.
            # Audio is always queued here — the mixer's _voice_enabled flag
            # determines whether it's mixed into output at the next boundary.
            # This means toggling a voice ON is instant at the next loop wrap,
            # without waiting for another full generation round-trip.
            if pass_num < i + 2:
                logger.debug("[Session] Pass %d: Voice %d not yet scheduled (need pass %d)",
                             pass_num, i + 1, i + 2)
                continue
            self.mixer.queue_voice(i, audio, volume=p.model_volume)
            logger.info("[Session] Pass %d: Voice %d queued (%.2fs, enabled=%s)",
                        pass_num, i + 1, audio.shape[0] / 48000,
                        self.mixer._voice_enabled[i])

    # ── Dry-run + debug helpers ────────────────────────────────────────────────

    def _dry_run_voices(self, user_loop: np.ndarray) -> list:
        """
        Generate simple sine tones as placeholder AI voice outputs.
        Used in --dry-run mode to test the full pipeline without Modal.
        Each voice gets a different pitch so they're distinguishable by ear.
        """
        n  = user_loop.shape[0]
        sr = 48000
        t  = np.linspace(0, n / sr, n, endpoint=False, dtype=np.float32)
        outputs = []
        for i in range(self.args.voices):
            freq = _DRY_RUN_FREQS[i % len(_DRY_RUN_FREQS)]
            mono = np.sin(2 * np.pi * freq * t) * 0.15
            outputs.append(np.stack([mono, mono], axis=1))
        return outputs

    def _save_loop_wav(self, loop: np.ndarray) -> None:
        """Save a captured loop to a timestamped WAV file for debugging."""
        try:
            import soundfile as sf
            import datetime
            ts = datetime.datetime.now().strftime("%H%M%S_%f")[:10]
            out_dir = Path(self.args.save_loops)
            out_dir.mkdir(parents=True, exist_ok=True)
            path = out_dir / f"loop_{ts}.wav"
            sf.write(str(path), loop, 48000, subtype="PCM_24")
            logger.info("[Session] Loop saved to: %s  (%.2fs, RMS=%.4f)",
                        path, loop.shape[0] / 48000,
                        float(np.sqrt(np.mean(loop ** 2))))
        except Exception as e:
            logger.error("[Session] Failed to save loop WAV: %s", e)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _transition(self, new_state: State):
        with self._lock:
            old = self.state
            self.state = new_state
        logger.info("[Session] %s → %s", old.name, new_state.name)

    def _print_banner(self):
        dry = "  *** DRY-RUN: Modal disabled, sine tones used ***" if self.args.dry_run else ""
        print()
        print("=" * 62)
        print("  Improv Loop — Ready")
        print(f"  BPM: {self.args.bpm}  Beats: {self.args.beats}  Voices: {self.args.voices}")
        print(f"  Capture:  [{self.devs.capture_idx}] {self.devs.capture_name}")
        print(f"  Playback: [{self.devs.playback_idx}] {self.devs.playback_name}")
        if dry:
            print(dry)
        print()
        print("  [Button 1] Count-in → Record → Play → Re-record")
        print("  [Button 2] Toggle AI Voice 1  (press to ENABLE — starts off)")
        print("  [Button 3] Toggle AI Voice 2  (press to ENABLE — starts off)")
        print("  [Button 4] Toggle AI Voice 3  (press to ENABLE — starts off)")
        print("  [Ctrl-C]   Quit")
        print("=" * 62)
        print()
        print("  Waiting for Button 1 (PBF4 col-1 top)...")
        print()


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Improv Loop — Real-time AI music improvisation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Normal session
  python improv_loop.py --bpm 120 --beats 16

  # Test audio pipeline without spending Modal credits
  python improv_loop.py --dry-run

  # Simulate real Modal timing during dry-run
  python improv_loop.py --dry-run --dry-run-latency 5.7

  # List audio & MIDI devices then exit
  python improv_loop.py --list-devices

  # Force VB-Cable capture (after installation)
  python improv_loop.py --capture-device "CABLE Output"

  # Use Stereo Mix explicitly (testing without VB-Cable)
  python improv_loop.py --capture-device "Stereo Mix"

  # Save captured loops for analysis
  python improv_loop.py --save-loops ./debug_loops --dry-run
        """,
    )

    # ── Session parameters ─────────────────────────────────────────────────
    p.add_argument("--bpm",    type=int, default=120, help="Tempo in BPM (default: 120)")
    p.add_argument("--beats",  type=int, default=16,  help="Beats per loop (default: 16)")
    p.add_argument("--voices", type=int, default=3,   choices=[1, 2, 3],
                   help="Number of AI voices to use (default: 3)")

    # ── Audio device routing ───────────────────────────────────────────────
    dev = p.add_argument_group("audio device routing")
    dev.add_argument("--capture-device",  metavar="NAME",
                     help="Capture device name substring (e.g. 'CABLE Output', 'Stereo Mix'). "
                          "Overrides auto-detection. Use after VB-Cable install.")
    dev.add_argument("--playback-device", metavar="NAME",
                     help="Playback device name substring (e.g. 'Speakers', 'Headphones'). "
                          "Overrides auto-detection.")
    dev.add_argument("--capture-idx",  type=int, metavar="IDX",
                     help="Force capture device by index (from --list-devices).")
    dev.add_argument("--playback-idx", type=int, metavar="IDX",
                     help="Force playback device by index (from --list-devices).")
    dev.add_argument("--list-devices", action="store_true",
                     help="List all audio and MIDI devices then exit.")

    # ── Debug / testing ───────────────────────────────────────────────────
    dbg = p.add_argument_group("debug / testing")
    dbg.add_argument("--dry-run", action="store_true",
                     help="Skip Modal calls entirely. AI voices are replaced with sine tones "
                          "(440/554/659 Hz). Tests the full audio pipeline without GPU cost.")
    dbg.add_argument("--dry-run-latency", type=float, default=0.0, metavar="SECONDS",
                     help="Artificial delay per dry-run generation pass (default: 0). "
                          "Set to ~5.7 to simulate real A100 timing and test buffer headroom.")
    dbg.add_argument("--save-loops", metavar="DIR",
                     help="Directory to save each captured loop as a timestamped WAV file. "
                          "Useful for diagnosing capture or routing issues.")
    dbg.add_argument("--no-click", action="store_true",
                     help="Disable metronome click during count-in (silent count-in).")
    dbg.add_argument("--log-level", default="info",
                     choices=["debug", "info", "warning", "error"],
                     help="Logging verbosity (default: info). Use 'debug' for full trace.")

    return p.parse_args()


def main():
    args = parse_args()

    # Configure logging before anything else
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s.%(msecs)03d  %(message)s",
        datefmt="%H:%M:%S",
    )

    # --list-devices: print device tables and exit (no session started)
    if args.list_devices:
        list_devices()
        list_midi_ports()
        return

    session = ImprovSession(args)
    session.setup()
    session.run()


if __name__ == "__main__":
    main()
