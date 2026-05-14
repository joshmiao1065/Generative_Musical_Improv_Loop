"""
keyboard_controller.py — QWERTY keyboard fallback for the intech PBF4.

Named "qwerty" (not "keyboard") to avoid confusion with the MIDI keyboard
(Arturia MicroLab mk3). Use with --qwerty when the PBF4 is not available.

Key bindings:
    Space / Enter    record_toggle  (PBF4 Button 1)
    1                voice_1_toggle (PBF4 Button 2)
    2                voice_2_toggle (PBF4 Button 3)
    3                voice_3_toggle (PBF4 Button 4)
    q / Q            quit session
    4                cycle genre for AI Voice 1 (next in --genres list)
    5                cycle genre for AI Voice 2
    6                cycle genre for AI Voice 3
    + or =           guidance_weight  +0.5
    -                guidance_weight  -0.5
    t                temperature      +0.1
    T                temperature      -0.1
    f                model_feedback   +0.05
    F                model_feedback   -0.05
    ?                print current param status

Platform notes:
    Windows  — uses msvcrt (built-in, no deps, instant key read without Enter)
    Unix/WSL — uses tty+termios raw mode (same instant-read behaviour)
    Terminal is always restored on stop() regardless of exit path.

GenreCycleThread:
    A lightweight always-on thread that listens for 4/5/6/? keys only.
    Used when PBF4 is active (--qwerty not set) so genre cycling still works.
    On Windows it uses msvcrt; on Unix it shares stdin with a select() check.
    Do NOT start both QwertyController and GenreCycleThread simultaneously —
    they will compete for the same key events.
"""

import logging
import sys
import threading
import time
from typing import Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

# ── Platform detection ─────────────────────────────────────────────────────────

try:
    import msvcrt
    _PLATFORM = "windows"
except ImportError:
    import select
    import termios
    import tty
    _PLATFORM = "unix"

# ── Constants ──────────────────────────────────────────────────────────────────

_BUTTON_LABELS = {
    "record_toggle",
    "voice_1_toggle", "voice_2_toggle", "voice_3_toggle",
    "quit",
}
_TOGGLE_LABELS = {"voice_1_toggle", "voice_2_toggle", "voice_3_toggle"}

_KEY_MAP: Dict[str, str] = {
    " ":  "record_toggle",
    "\r": "record_toggle",   # Enter (Windows)
    "\n": "record_toggle",   # Enter (Unix)
    "1":  "voice_1_toggle",
    "2":  "voice_2_toggle",
    "3":  "voice_3_toggle",
    "q":  "quit",
    "Q":  "quit",
    "\x03": "quit",          # Ctrl+C in Unix raw mode
}

_PARAM_ADJUSTMENTS: Dict[str, tuple] = {
    "+": ("guidance_weight", +0.5),
    "=": ("guidance_weight", +0.5),
    "-": ("guidance_weight", -0.5),
    "t": ("temperature",     +0.1),
    "T": ("temperature",     -0.1),
    "f": ("model_feedback",  +0.05),
    "F": ("model_feedback",  -0.05),
}

_PARAM_RANGES = {
    "guidance_weight": (0.0,  10.0, False),
    "temperature":     (0.0,   4.0, False),
    "model_feedback":  (0.0,   1.0, False),
}

# Keys that cycle the genre for Voice 1/2/3 — always-active regardless of --qwerty
_GENRE_CYCLE_KEYS: Dict[str, int] = {
    "4": 0,   # cycle Voice 1 genre
    "5": 1,   # cycle Voice 2 genre
    "6": 2,   # cycle Voice 3 genre
}


# ─────────────────────────────────────────────────────────────────────────────
# Shared key-read helpers
# ─────────────────────────────────────────────────────────────────────────────

def _read_key_nonblocking() -> Optional[str]:
    """Return one character if a key is waiting, else None (non-blocking)."""
    if _PLATFORM == "windows":
        if msvcrt.kbhit():
            ch = msvcrt.getwch()
            if ch in ("\x00", "\xe0"):
                msvcrt.getwch()  # consume second byte of special key
                return None
            return ch
        return None
    else:
        r, _, _ = select.select([sys.stdin], [], [], 0.0)
        if r:
            return sys.stdin.read(1)
        return None


# ─────────────────────────────────────────────────────────────────────────────
# GenreCycleThread — always-on genre cycling, used when PBF4 is active
# ─────────────────────────────────────────────────────────────────────────────

class GenreCycleThread:
    """
    Minimal always-on thread that listens for keys 4/5/6/? only.

    Used when the PBF4 is active (--qwerty not set) so genre cycling still
    works from the keyboard. On Windows, msvcrt.kbhit() is called from this
    thread; the first thread to consume a key gets it.

    Do NOT start this alongside QwertyController — they share the same
    keyboard input stream and will race.

    Parameters
    ----------
    on_genre_cycle : callable(voice_idx: int)
        Called when 4/5/6 is pressed. voice_idx is 0/1/2.
    on_status : callable, optional
        Called when ? is pressed to print status.
    poll_interval : float
        Seconds between key polls (default 20 ms).
    """

    def __init__(
        self,
        on_genre_cycle: Callable,
        on_status: Optional[Callable] = None,
        poll_interval: float = 0.02,
    ):
        self._on_genre_cycle = on_genre_cycle
        self._on_status      = on_status
        self.poll_interval   = poll_interval

        self._thread:    Optional[threading.Thread] = None
        self._running    = threading.Event()
        self._saved_term = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._running.set()
        self._thread = threading.Thread(
            target=self._poll_loop, daemon=True, name="GenreCycle-KB"
        )
        self._thread.start()
        logger.info("[GenreCycle] Genre cycling thread started (keys 4/5/6). "
                    "Press 4/5/6 to cycle Voice 1/2/3 genre.")

    def stop(self) -> None:
        self._running.clear()
        if _PLATFORM == "unix":
            self._restore_term()
        if self._thread:
            self._thread.join(timeout=1.0)

    def _poll_loop(self) -> None:
        if _PLATFORM == "unix":
            self._set_raw_mode()
        try:
            while self._running.is_set():
                ch = _read_key_nonblocking()
                if ch is None:
                    time.sleep(self.poll_interval)
                    continue
                if ch in _GENRE_CYCLE_KEYS:
                    voice_idx = _GENRE_CYCLE_KEYS[ch]
                    try:
                        self._on_genre_cycle(voice_idx)
                    except Exception as e:
                        logger.error("[GenreCycle] Callback error: %s", e)
                elif ch == "?":
                    if self._on_status is not None:
                        try:
                            self._on_status()
                        except Exception as e:
                            logger.error("[GenreCycle] Status callback error: %s", e)
        finally:
            if _PLATFORM == "unix":
                self._restore_term()

    def _set_raw_mode(self) -> None:
        try:
            fd = sys.stdin.fileno()
            self._saved_term = termios.tcgetattr(fd)
            tty.setraw(fd)
        except Exception as e:
            logger.warning("[GenreCycle] Could not set terminal raw mode: %s", e)

    def _restore_term(self) -> None:
        if self._saved_term is not None:
            try:
                termios.tcsetattr(
                    sys.stdin.fileno(), termios.TCSADRAIN, self._saved_term
                )
            except Exception:
                pass
            self._saved_term = None


# ─────────────────────────────────────────────────────────────────────────────
# QwertyController — full keyboard fallback when PBF4 is not connected
# ─────────────────────────────────────────────────────────────────────────────

class QwertyController:
    """
    QWERTY keyboard controller — drop-in replacement for PBF4Controller when
    the intech hardware is not available.

    Same interface as PBF4Controller:
        ctrl.on("record_toggle", cb)
        ctrl.get_toggle("voice_1_toggle")   → bool
        ctrl.params                         → GenerationParams (shared, writable)
        ctrl.start() / ctrl.stop()
        ctrl.on_genre_cycle(cb)             → cb(voice_idx: int) on 4/5/6

    Extra event: "quit" — fired when q/Q/Ctrl-C is pressed.

    Parameters
    ----------
    params : GenerationParams
        Shared params object — written by this thread, read by main loop.
    poll_interval : float
        Seconds between key polls (default 20 ms). Keeps CPU near zero.
    """

    def __init__(self, params, poll_interval: float = 0.02):
        self.params        = params
        self.poll_interval = poll_interval

        self._genre_cycle_cb: Optional[Callable] = None
        self._status_cb:      Optional[Callable] = None

        self._callbacks:    Dict[str, List[Callable]] = {lbl: [] for lbl in _BUTTON_LABELS}
        self._toggle_state: Dict[str, bool]           = {lbl: False for lbl in _TOGGLE_LABELS}

        self._thread:     Optional[threading.Thread] = None
        self._running     = threading.Event()
        self._saved_term  = None

    # ── Public callbacks ───────────────────────────────────────────────────────

    def on(self, event: str, callback: Callable) -> None:
        """Register a callback for a key event.

        Valid events: 'record_toggle', 'voice_1_toggle', 'voice_2_toggle',
                      'voice_3_toggle', 'quit'
        Callbacks run on the keyboard thread — keep them fast or hand off.
        """
        if event not in self._callbacks:
            raise ValueError(
                f"Unknown event '{event}'. Valid: {sorted(self._callbacks)}"
            )
        self._callbacks[event].append(callback)

    def on_genre_cycle(self, cb: Callable) -> None:
        """Register callback for 4/5/6 genre cycle keys. Called with voice_idx (0/1/2)."""
        self._genre_cycle_cb = cb

    def on_status(self, cb: Callable) -> None:
        """Register callback for ? key. Called with no arguments."""
        self._status_cb = cb

    # ── Thread control ─────────────────────────────────────────────────────────

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._running.set()
        self._thread = threading.Thread(
            target=self._poll_loop, daemon=True, name="Qwerty-KB"
        )
        self._thread.start()
        logger.info("[QWERTY] Keyboard controller started (%s mode).", _PLATFORM)
        self._print_help()

    def stop(self) -> None:
        self._running.clear()
        if _PLATFORM == "unix":
            self._restore_term()
        if self._thread:
            self._thread.join(timeout=1.0)
        logger.info("[QWERTY] Keyboard controller stopped.")

    # ── Key polling loop ───────────────────────────────────────────────────────

    def _poll_loop(self) -> None:
        if _PLATFORM == "unix":
            self._set_raw_mode()
        try:
            while self._running.is_set():
                ch = _read_key_nonblocking()
                if ch is None:
                    time.sleep(self.poll_interval)
                    continue
                self._dispatch(ch)
        finally:
            if _PLATFORM == "unix":
                self._restore_term()

    # ── Terminal setup (Unix only) ─────────────────────────────────────────────

    def _set_raw_mode(self) -> None:
        try:
            fd = sys.stdin.fileno()
            self._saved_term = termios.tcgetattr(fd)
            tty.setraw(fd)
        except Exception as e:
            logger.warning("[QWERTY] Could not set terminal raw mode: %s", e)

    def _restore_term(self) -> None:
        if self._saved_term is not None:
            try:
                termios.tcsetattr(
                    sys.stdin.fileno(), termios.TCSADRAIN, self._saved_term
                )
            except Exception:
                pass
            self._saved_term = None

    # ── Dispatch ──────────────────────────────────────────────────────────────

    def _dispatch(self, ch: str) -> None:
        if ch in _KEY_MAP:
            self._fire_button(_KEY_MAP[ch])

        elif ch in _GENRE_CYCLE_KEYS:
            voice_idx = _GENRE_CYCLE_KEYS[ch]
            if self._genre_cycle_cb is not None:
                try:
                    self._genre_cycle_cb(voice_idx)
                except Exception as e:
                    logger.error("[QWERTY] Genre cycle callback error: %s", e)

        elif ch in _PARAM_ADJUSTMENTS:
            attr, delta = _PARAM_ADJUSTMENTS[ch]
            lo, hi, as_int = _PARAM_RANGES[attr]
            cur = getattr(self.params, attr)
            new = max(lo, min(hi, cur + delta))
            if as_int:
                new = int(round(new))
            setattr(self.params, attr, new)
            logger.info("[QWERTY] %s = %s", attr, new)

        elif ch == "?":
            if self._status_cb is not None:
                try:
                    self._status_cb()
                except Exception as e:
                    logger.error("[QWERTY] Status callback error: %s", e)
            else:
                self.print_status()

    def _fire_button(self, label: str) -> None:
        if label in _TOGGLE_LABELS:
            self._toggle_state[label] = not self._toggle_state[label]
            state_str = "ON" if self._toggle_state[label] else "OFF"
            logger.info("[QWERTY] %s → %s", label, state_str)
        else:
            logger.info("[QWERTY] %s", label)

        for cb in self._callbacks.get(label, []):
            try:
                cb()
            except Exception as e:
                logger.error("[QWERTY] Callback error '%s': %s", label, e)

    # ── Accessors (same interface as PBF4Controller) ───────────────────────────

    def get_toggle(self, label: str) -> bool:
        """Current on/off state for a toggle button label."""
        return self._toggle_state.get(label, False)

    def print_status(self) -> None:
        p = self.params
        print("\n[QWERTY Status]")
        print(f"  guidance_weight = {p.guidance_weight:.2f}   (+ / -)")
        print(f"  temperature     = {p.temperature:.2f}   (t=up  T=down)")
        print(f"  model_feedback  = {p.model_feedback:.3f}   (f=up  F=down)")
        for lbl in sorted(_TOGGLE_LABELS):
            state = "ON" if self.get_toggle(lbl) else "OFF"
            print(f"  {lbl:18s} = {state}")
        print()

    # ── Banner ─────────────────────────────────────────────────────────────────

    def _print_help(self) -> None:
        print()
        print("  [QWERTY Controls]  (PBF4 not connected)")
        print("  ─────────────────────────────────────────")
        print("    Space / Enter    record_toggle (count-in → record → loop)")
        print("    1 / 2 / 3        toggle AI Voice 1 / 2 / 3")
        print("    q                quit session")
        print("    4 / 5 / 6        cycle genre for Voice 1 / 2 / 3")
        print("    + / -            guidance_weight  ±0.5")
        print("    t / T            temperature      ±0.1")
        print("    f / F            model_feedback   ±0.05")
        print("    ?                print current params + genres")
        print()
