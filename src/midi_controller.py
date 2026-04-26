"""
midi_controller.py — PBF4 live MIDI input handler.

Runs a background daemon thread that:
  - Reads CC and note_on messages from the intech PBF4
  - CC messages (knobs/faders) → GenerationParams fields, scaled to physical ranges
  - Note_on messages (buttons) → fires registered callbacks + manages toggle state
  - Writes genre_weights[0..3] for style prompt blending each pass

Hardware facts (confirmed 2026-04-21):
  - Port name: "Intech Grid MIDI device 0"
  - Knobs (col 1–4 bottom):  CC 32, 33, 34, 35  on ch 0
  - Faders (col 1–4 middle): CC 36, 37, 38, 39  on ch 0
  - Buttons (col 1–4 top):   note_on messages on ch 0, note numbers TBD

Button behavior:
  - record_toggle: fires on every press; session state machine decides what to do
      (first press → 2-bar count-in → record; press while recording → stop+loop;
       press while playing → restart)
  - voice_1_toggle, voice_2_toggle, voice_3_toggle: flip on/off each press

Thread safety:
  float/int attribute assignments are atomic under the GIL — GenerationParams fields
  can be written from this thread and read from the main loop safely.
  genre_weights uses a threading.Lock (list mutation is not atomic).

Usage:
    params = GenerationParams()
    ctrl = PBF4Controller(params, "config/pbf4_layout.json")
    ctrl.on("record_toggle",  session.handle_record)
    ctrl.on("voice_1_toggle", lambda: session.toggle_voice(0))
    ctrl.on("voice_2_toggle", lambda: session.toggle_voice(1))
    ctrl.on("voice_3_toggle", lambda: session.toggle_voice(2))
    ctrl.start()
    # ... run session ...
    ctrl.stop()
"""

import json
import logging
import threading
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

import mido

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Parameter scaling
# ─────────────────────────────────────────────────────────────────────────────

# label → (min, max, return_int)
_PARAM_RANGES: Dict[str, tuple] = {
    "guidance_weight": (0.0,  10.0, False),
    "temperature":     (0.0,   2.0, False),
    "topk":            (0,     100, True),
    "model_feedback":  (0.0,   1.0, False),
    "genre_0":         (0.0,   1.0, False),
    "genre_1":         (0.0,   1.0, False),
    "genre_2":         (0.0,   1.0, False),
    "genre_3":         (0.0,   1.0, False),
}

_BUTTON_LABELS  = {"record_toggle", "voice_1_toggle", "voice_2_toggle", "voice_3_toggle"}
_TOGGLE_LABELS  = {"voice_1_toggle", "voice_2_toggle", "voice_3_toggle"}
_GENRE_LABELS   = {"genre_0", "genre_1", "genre_2", "genre_3"}


def _scale(raw: int, lo, hi, as_int: bool):
    v = lo + (raw / 127.0) * (hi - lo)
    return int(round(v)) if as_int else float(v)


# ─────────────────────────────────────────────────────────────────────────────
# Controller
# ─────────────────────────────────────────────────────────────────────────────

class PBF4Controller:
    """
    Background MIDI listener for the intech PBF4.

    Parameters
    ----------
    params : GenerationParams
        Shared params object written by this thread, read by the main loop.
    layout_path : str | Path
        Path to pbf4_layout.json.
    poll_interval : float
        Seconds between mido iter_pending() calls. 1ms gives <1ms latency.
    """

    def __init__(self, params, layout_path: str = "config/pbf4_layout.json",
                 poll_interval: float = 0.001):
        self.params = params
        self.layout_path = Path(layout_path)
        self.poll_interval = poll_interval

        self.genre_weights: List[float] = [1.0, 0.0, 0.0, 0.0]
        self._genre_lock = threading.Lock()

        self._callbacks: Dict[str, List[Callable]] = {lbl: [] for lbl in _BUTTON_LABELS}
        self._toggle_state: Dict[str, bool] = {lbl: False for lbl in _TOGGLE_LABELS}

        # Lookup tables built from layout file
        self._cc_map:   Dict[tuple, dict] = {}   # (channel, cc)   → control dict
        self._note_map: Dict[tuple, dict] = {}   # (channel, note) → control dict
        self._port_name_substr = "Intech Grid"
        self._load_layout()

        self._thread:  Optional[threading.Thread] = None
        self._running  = threading.Event()
        self._port:    Optional[mido.ports.BaseInput] = None

    # ── Layout ────────────────────────────────────────────────────────────────

    def _load_layout(self):
        self._cc_map.clear()
        self._note_map.clear()

        if not self.layout_path.exists():
            logger.warning(
                "[PBF4] Layout not found: %s — running in log-only mode.\n"
                "  Run scripts/discover_cc.py (press all buttons!), fill\n"
                "  config/pbf4_layout.json with note_number values.",
                self.layout_path,
            )
            return

        with open(self.layout_path) as f:
            layout = json.load(f)

        self._port_name_substr = layout.get("port_name_substring", "Intech Grid")
        loaded_cc = loaded_note = skipped = 0

        for ctrl in layout.get("controls", []):
            label     = ctrl.get("label")
            ch        = ctrl.get("channel")
            midi_type = ctrl.get("midi_type", "cc")

            if label is None or ch is None:
                skipped += 1
                continue

            if midi_type == "note":
                note_num = ctrl.get("note_number")
                if note_num is None:
                    logger.warning("[PBF4] Button '%s' has no note_number — skipped. "
                                   "Re-run discover_cc.py and press all buttons.", label)
                    skipped += 1
                    continue
                self._note_map[(int(ch), int(note_num))] = {
                    "label": label, "type": ctrl.get("type", "button"),
                }
                loaded_note += 1

            else:  # "cc"
                cc = ctrl.get("cc")
                if cc is None:
                    skipped += 1
                    continue
                self._cc_map[(int(ch), int(cc))] = {
                    "label":  label,
                    "type":   ctrl.get("type", "knob"),
                    "range":  ctrl.get("range"),
                }
                loaded_cc += 1

        logger.info("[PBF4] Layout loaded — %d CC controls, %d note buttons, %d skipped",
                    loaded_cc, loaded_note, skipped)

    def reload_layout(self):
        self._load_layout()

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def on(self, event: str, callback: Callable):
        """
        Register a callback for a button event.
        Valid events: 'record_toggle', 'voice_1_toggle', 'voice_2_toggle', 'voice_3_toggle'
        Callbacks run on the MIDI thread — keep them fast or hand off via queue/Event.
        """
        if event not in self._callbacks:
            raise ValueError(f"Unknown event '{event}'. Valid: {sorted(self._callbacks)}")
        self._callbacks[event].append(callback)

    # ── Thread control ─────────────────────────────────────────────────────────

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._running.set()
        self._thread = threading.Thread(target=self._listen_loop, daemon=True, name="PBF4-MIDI")
        self._thread.start()
        logger.info("[PBF4] Listener started.")

    def stop(self):
        self._running.clear()
        if self._thread:
            self._thread.join(timeout=2.0)
        if self._port:
            try:
                self._port.close()
            except Exception:
                pass
        logger.info("[PBF4] Listener stopped.")

    # ── Port selection ─────────────────────────────────────────────────────────

    def _find_port(self) -> Optional[str]:
        names = mido.get_input_names()
        for name in names:
            if self._port_name_substr.lower() in name.lower():
                return name
        logger.error("[PBF4] No port matching '%s'. Available: %s",
                     self._port_name_substr, names)
        return None

    # ── Listen loop ───────────────────────────────────────────────────────────

    def _listen_loop(self):
        port_name = self._find_port()
        if port_name is None:
            return

        logger.info("[PBF4] Listening on: %s", port_name)
        try:
            self._port = mido.open_input(port_name)
        except Exception as e:
            logger.error("[PBF4] Failed to open port: %s", e)
            return

        while self._running.is_set():
            for msg in self._port.iter_pending():
                if msg.type == "control_change":
                    self._handle_cc(msg)
                elif msg.type == "note_on":
                    self._handle_note(msg)
                # note_off / pitchwheel / etc. ignored
            time.sleep(self.poll_interval)

        self._port.close()

    # ── Message handlers ──────────────────────────────────────────────────────

    def _handle_cc(self, msg):
        key  = (msg.channel, msg.control)
        ctrl = self._cc_map.get(key)
        if ctrl is None:
            logger.debug("[PBF4] Unmapped CC ch=%d cc=%d val=%d",
                         msg.channel, msg.control, msg.value)
            return
        self._apply_continuous(ctrl["label"], msg.value, ctrl.get("range"))

    def _handle_note(self, msg):
        # Buttons: note_on vel=127 is press, vel=0 is release (some firmware sends
        # note_on vel=0 instead of note_off). Only act on press.
        if msg.velocity == 0:
            return

        key  = (msg.channel, msg.note)
        ctrl = self._note_map.get(key)
        if ctrl is None:
            logger.debug("[PBF4] Unmapped note ch=%d note=%d vel=%d",
                         msg.channel, msg.note, msg.velocity)
            return

        label = ctrl["label"]
        if label in _TOGGLE_LABELS:
            self._toggle_state[label] = not self._toggle_state[label]
            logger.info("[PBF4] %s → %s",
                        label, "ON" if self._toggle_state[label] else "OFF")
        else:
            logger.info("[PBF4] %s pressed", label)

        for cb in self._callbacks.get(label, []):
            try:
                cb()
            except Exception as e:
                logger.error("[PBF4] Callback error '%s': %s", label, e)

    def _apply_continuous(self, label: str, raw: int, range_override):
        if label in _GENRE_LABELS:
            idx    = int(label[-1])
            scaled = raw / 127.0
            with self._genre_lock:
                self.genre_weights[idx] = scaled
            logger.debug("[PBF4] genre_%d = %.3f", idx, scaled)
            return

        if label not in _PARAM_RANGES:
            logger.warning("[PBF4] No range for label '%s'", label)
            return

        lo, hi, as_int = _PARAM_RANGES[label]
        if range_override is not None:
            lo, hi  = range_override[0], range_override[1]
            as_int  = isinstance(lo, int) and isinstance(hi, int)

        value = _scale(raw, lo, hi, as_int)
        setattr(self.params, label, value)
        logger.debug("[PBF4] %s = %s (raw=%d)", label, value, raw)

    # ── Accessors ─────────────────────────────────────────────────────────────

    def get_genre_weights(self) -> List[float]:
        """Thread-safe snapshot of current genre weights [g0, g1, g2, g3]."""
        with self._genre_lock:
            return list(self.genre_weights)

    def get_toggle(self, label: str) -> bool:
        """Current on/off state for a toggle button label."""
        return self._toggle_state.get(label, False)

    def print_status(self):
        p  = self.params
        gw = self.get_genre_weights()
        print("[PBF4 Status]")
        print(f"  guidance_weight = {p.guidance_weight:.2f}")
        print(f"  temperature     = {p.temperature:.2f}")
        print(f"  topk            = {p.topk}")
        print(f"  model_feedback  = {p.model_feedback:.3f}")
        print(f"  genre_weights   = {[f'{w:.2f}' for w in gw]}")
        for lbl in _TOGGLE_LABELS:
            state = "ON" if self.get_toggle(lbl) else "OFF"
            print(f"  {lbl:18s} = {state}")


# ─────────────────────────────────────────────────────────────────────────────
# Standalone: log-only validation (run from project root on Windows)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys, dataclasses, types

    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s.%(msecs)03d  %(message)s",
                        datefmt="%H:%M:%S")

    try:
        from src.magenta_backend import GenerationParams
    except ImportError:
        @dataclasses.dataclass
        class GenerationParams:
            guidance_weight: float = 1.5
            temperature: float = 1.2
            topk: int = 30
            model_feedback: float = 0.95
            model_volume: float = 0.85
            beats_per_loop: int = 8
            bpm: int = 120

    layout = sys.argv[1] if len(sys.argv) > 1 else "config/pbf4_layout.json"
    params = GenerationParams()
    ctrl   = PBF4Controller(params, layout_path=layout)

    ctrl.on("record_toggle",  lambda: print(">>> RECORD toggled"))
    ctrl.on("voice_1_toggle", lambda: print(f">>> Voice 1: {'ON' if ctrl.get_toggle('voice_1_toggle') else 'OFF'}"))
    ctrl.on("voice_2_toggle", lambda: print(f">>> Voice 2: {'ON' if ctrl.get_toggle('voice_2_toggle') else 'OFF'}"))
    ctrl.on("voice_3_toggle", lambda: print(f">>> Voice 3: {'ON' if ctrl.get_toggle('voice_3_toggle') else 'OFF'}"))
    ctrl.start()

    print("Listening. Move all controls. Ctrl+C to quit.\n")
    try:
        while True:
            time.sleep(2.0)
            ctrl.print_status()
    except KeyboardInterrupt:
        ctrl.stop()
