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
    + or =           guidance_weight  +0.5
    -                guidance_weight  -0.5
    t                temperature      +0.1
    T                temperature      -0.1
    k                topk             +5
    K                topk             -5
    f                model_feedback   +0.05
    F                model_feedback   -0.05
    ] / [            genre_0 weight   ±0.1
    ' / ;            genre_1 weight   ±0.1
    . / ,            genre_2 weight   ±0.1
    M / m            genre_3 weight   ±0.1
    ?                print current param status

Platform notes:
    Windows  — uses msvcrt (built-in, no deps, instant key read without Enter)
    Unix/WSL — uses tty+termios raw mode (same instant-read behaviour)
    Terminal is always restored on stop() regardless of exit path.
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
    "=": ("guidance_weight", +0.5),   # = is + without Shift
    "-": ("guidance_weight", -0.5),
    "t": ("temperature",     +0.1),
    "T": ("temperature",     -0.1),
    "k": ("topk",            +5),
    "K": ("topk",            -5),
    "f": ("model_feedback",  +0.05),
    "F": ("model_feedback",  -0.05),
}

# (genre_index, delta)
_GENRE_ADJUSTMENTS: Dict[str, tuple] = {
    "]": (0, +0.1),
    "[": (0, -0.1),
    "'": (1, +0.1),
    ";": (1, -0.1),
    ".": (2, +0.1),
    ",": (2, -0.1),
    "M": (3, +0.1),
    "m": (3, -0.1),
}

_PARAM_RANGES = {
    "guidance_weight": (0.0,  10.0, False),
    "temperature":     (0.0,   4.0, False),
    "topk":            (0,    1024, True),
    "model_feedback":  (0.0,   1.0, False),
}


# ─────────────────────────────────────────────────────────────────────────────
# QwertyController
# ─────────────────────────────────────────────────────────────────────────────

class QwertyController:
    """
    QWERTY keyboard controller — drop-in replacement for PBF4Controller when
    the intech hardware is not available.

    Same interface as PBF4Controller:
        ctrl.on("record_toggle", cb)
        ctrl.get_toggle("voice_1_toggle")   → bool
        ctrl.get_genre_weights()            → List[float]
        ctrl.params                         → GenerationParams (shared, writable)
        ctrl.start() / ctrl.stop()

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

        self.genre_weights: List[float] = [1.0, 0.0, 0.0, 0.0]
        self._genre_lock   = threading.Lock()

        self._callbacks:    Dict[str, List[Callable]] = {lbl: [] for lbl in _BUTTON_LABELS}
        self._toggle_state: Dict[str, bool]           = {lbl: False for lbl in _TOGGLE_LABELS}

        self._thread:     Optional[threading.Thread] = None
        self._running     = threading.Event()
        self._saved_term  = None   # Unix: saved terminal attributes

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
                ch = self._read_key()
                if ch is None:
                    time.sleep(self.poll_interval)
                    continue
                self._dispatch(ch)
        finally:
            if _PLATFORM == "unix":
                self._restore_term()

    def _read_key(self) -> Optional[str]:
        """Return one character if a key is waiting, else None (non-blocking)."""
        if _PLATFORM == "windows":
            if msvcrt.kbhit():
                ch = msvcrt.getwch()
                # Special keys (arrows, F-keys) emit 0x00 or 0xe0 + a second byte
                if ch in ("\x00", "\xe0"):
                    msvcrt.getwch()   # consume the second byte, ignore
                    return None
                return ch
            return None
        else:
            # Unix: select with 0-second timeout → non-blocking
            r, _, _ = select.select([sys.stdin], [], [], 0.0)
            if r:
                return sys.stdin.read(1)
            return None

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

        elif ch in _PARAM_ADJUSTMENTS:
            attr, delta = _PARAM_ADJUSTMENTS[ch]
            lo, hi, as_int = _PARAM_RANGES[attr]
            cur = getattr(self.params, attr)
            new = max(lo, min(hi, cur + delta))
            if as_int:
                new = int(round(new))
            setattr(self.params, attr, new)
            logger.info("[QWERTY] %s = %s", attr, new)

        elif ch in _GENRE_ADJUSTMENTS:
            idx, delta = _GENRE_ADJUSTMENTS[ch]
            with self._genre_lock:
                self.genre_weights[idx] = round(
                    max(0.0, min(1.0, self.genre_weights[idx] + delta)), 3
                )
                val = self.genre_weights[idx]
            logger.info("[QWERTY] genre_%d = %.2f", idx, val)

        elif ch == "?":
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

    def get_genre_weights(self) -> List[float]:
        """Thread-safe snapshot of current genre weights [g0, g1, g2, g3]."""
        with self._genre_lock:
            return list(self.genre_weights)

    def get_toggle(self, label: str) -> bool:
        """Current on/off state for a toggle button label."""
        return self._toggle_state.get(label, False)

    def print_status(self) -> None:
        p  = self.params
        gw = self.get_genre_weights()
        print("\n[QWERTY Status]")
        print(f"  guidance_weight = {p.guidance_weight:.2f}   (+ / -)")
        print(f"  temperature     = {p.temperature:.2f}   (t=up  T=down)")
        print(f"  topk            = {p.topk}   (k=up  K=down)")
        print(f"  model_feedback  = {p.model_feedback:.3f}   (f=up  F=down)")
        print(f"  genre_weights   = {[f'{w:.2f}' for w in gw]}")
        print(f"                    genre0=]/[  genre1=';/;  genre2=./,  genre3=M/m")
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
        print("    + / -            guidance_weight  ±0.5")
        print("    t / T            temperature      ±0.1")
        print("    k / K            topk             ±5")
        print("    f / F            model_feedback   ±0.05")
        print("    ] / [            genre 0 weight   ±0.1")
        print("    ' / ;            genre 1 weight   ±0.1")
        print("    . / ,            genre 2 weight   ±0.1")
        print("    M / m            genre 3 weight   ±0.1")
        print("    ?                print current params")
        print()
