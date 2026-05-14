"""
midi_controller.py — PBF4 live MIDI input handler.

Runs a background daemon thread that:
  - Reads CC and note_on messages from the intech PBF4
  - CC messages (knobs/faders) → registered callbacks, scaled to physical ranges
  - Note_on messages (buttons) → fires registered callbacks + manages toggle state

Hardware facts (confirmed 2026-04-21):
  - Port name: "Intech Grid MIDI device 0"
  - Knobs (col 1–4 bottom):  CC 32, 33, 34, 35  on ch 0
  - Faders (col 1–4 middle): CC 36, 37, 38, 39  on ch 0
  - Buttons (col 1–4 top):   note_on messages on ch 0, note numbers TBD

Fader mapping (CC 36–39):
  CC 36 → user_volume       [0.0–1.0]  user loop stem volume
  CC 37 → voice_0_volume    [0.0–1.0]  AI Voice 1 stem volume
  CC 38 → voice_1_volume    [0.0–1.0]  AI Voice 2 stem volume
  CC 39 → voice_2_volume    [0.0–1.0]  AI Voice 3 stem volume

Knob mapping (CC 32–35):
  CC 32 → eq_bass    [-12 to +12 dB]  low shelf @ 250 Hz
  CC 33 → eq_mid     [-12 to +12 dB]  peaking EQ @ 1 kHz
  CC 34 → eq_treble  [-12 to +12 dB]  high shelf @ 4 kHz
  CC 35 → reverb_wet [0.0–1.0]        Freeverb wet mix

Button behavior:
  - record_toggle: fires on every press
  - voice_1_toggle, voice_2_toggle, voice_3_toggle: flip on/off each press

Thread safety:
  All callbacks are called from the MIDI thread — keep them fast or hand off
  via queue/Event. Float attribute writes are atomic under the GIL.

Usage:
    params = GenerationParams()
    ctrl = PBF4Controller(params, "config/pbf4_layout.json")
    ctrl.on("record_toggle",  session.handle_record)
    ctrl.on("voice_1_toggle", lambda: session.toggle_voice(0))
    ctrl.on_user_volume(mixer.set_user_volume)
    ctrl.on_voice_volume(0, lambda v: mixer.set_voice_volume(0, v))
    ctrl.on_eq_bass(effects.set_eq_bass)
    ctrl.on_reverb_wet(effects.set_reverb_wet)
    ctrl.start()
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
# Constants
# ─────────────────────────────────────────────────────────────────────────────

_BUTTON_LABELS = {"record_toggle", "voice_1_toggle", "voice_2_toggle", "voice_3_toggle"}
_TOGGLE_LABELS = {"voice_1_toggle", "voice_2_toggle", "voice_3_toggle"}

# Volume fader labels → scaled to [0.0, 1.0]
_VOLUME_LABELS = {"user_volume", "voice_0_volume", "voice_1_volume", "voice_2_volume"}

# EQ knob labels → scaled to [-12.0, +12.0] dB (CC 64 = 0 dB)
_EQ_LABELS = {"eq_bass", "eq_mid", "eq_treble"}

# Reverb knob label → scaled to [0.0, 1.0]
_REVERB_LABEL = "reverb_wet"

# Fallback for any layout entries that map to GenerationParams fields
_PARAM_RANGES: Dict[str, tuple] = {
    "guidance_weight": (0.0, 10.0, False),
    "temperature":     (0.0,  2.0, False),
    "model_feedback":  (0.0,  1.0, False),
}


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
        Shared params object (written for any layout entries that match
        GenerationParams field names — currently unused with new knob layout).
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

        # Button callbacks
        self._callbacks: Dict[str, List[Callable]] = {lbl: [] for lbl in _BUTTON_LABELS}
        self._toggle_state: Dict[str, bool] = {lbl: False for lbl in _TOGGLE_LABELS}

        # CC callbacks — one callable per label, called with the scaled float value
        self._cc_callbacks: Dict[str, Optional[Callable]] = {
            "user_volume":   None,
            "voice_0_volume": None,
            "voice_1_volume": None,
            "voice_2_volume": None,
            "eq_bass":       None,
            "eq_mid":        None,
            "eq_treble":     None,
            "reverb_wet":    None,
        }

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

    # ── Button callbacks ──────────────────────────────────────────────────────

    def on(self, event: str, callback: Callable):
        """
        Register a callback for a button event.
        Valid events: 'record_toggle', 'voice_1_toggle', 'voice_2_toggle', 'voice_3_toggle'
        Callbacks run on the MIDI thread — keep them fast or hand off via queue/Event.
        """
        if event not in self._callbacks:
            raise ValueError(f"Unknown event '{event}'. Valid: {sorted(self._callbacks)}")
        self._callbacks[event].append(callback)

    # ── CC callbacks (volume / EQ / reverb) ───────────────────────────────────

    def on_user_volume(self, cb: Callable) -> None:
        """Register callback for user loop fader. Called with float [0.0–1.0]."""
        self._cc_callbacks["user_volume"] = cb

    def on_voice_volume(self, voice_idx: int, cb: Callable) -> None:
        """Register callback for AI voice fader (idx=0/1/2). Called with float [0.0–1.0]."""
        self._cc_callbacks[f"voice_{voice_idx}_volume"] = cb

    def on_eq_bass(self, cb: Callable) -> None:
        """Register callback for bass EQ knob. Called with dB float [-12.0–+12.0]."""
        self._cc_callbacks["eq_bass"] = cb

    def on_eq_mid(self, cb: Callable) -> None:
        """Register callback for mid EQ knob. Called with dB float [-12.0–+12.0]."""
        self._cc_callbacks["eq_mid"] = cb

    def on_eq_treble(self, cb: Callable) -> None:
        """Register callback for treble EQ knob. Called with dB float [-12.0–+12.0]."""
        self._cc_callbacks["eq_treble"] = cb

    def on_reverb_wet(self, cb: Callable) -> None:
        """Register callback for reverb knob. Called with float [0.0–1.0]."""
        self._cc_callbacks["reverb_wet"] = cb

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

    def _find_ports(self) -> list:
        """Return all input port names containing the configured substring."""
        names = mido.get_input_names()
        matches = [n for n in names if self._port_name_substr.lower() in n.lower()]
        if not matches:
            logger.error("[PBF4] No port matching '%s'. Available: %s",
                         self._port_name_substr, names)
        return matches

    # ── Listen loop ───────────────────────────────────────────────────────────

    def _listen_loop(self):
        candidates = self._find_ports()
        if not candidates:
            return

        # The intech Grid exposes multiple ports (e.g. device 0 = SysEx/config,
        # device 1 = MIDI I/O). Try each candidate in order; use the first that
        # opens successfully.
        for port_name in candidates:
            logger.info("[PBF4] Trying port: %s", port_name)
            try:
                self._port = mido.open_input(port_name)
                logger.info("[PBF4] Opened: %s", port_name)
                break
            except Exception as e:
                logger.warning("[PBF4] Cannot open '%s': %s — trying next", port_name, e)
        else:
            logger.error("[PBF4] No usable port found among: %s", candidates)
            return

        while self._running.is_set():
            for msg in self._port.iter_pending():
                if msg.type == "control_change":
                    self._handle_cc(msg)
                elif msg.type == "note_on":
                    self._handle_note(msg)
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
        if label in _VOLUME_LABELS or label == _REVERB_LABEL:
            val = raw / 127.0
        elif label in _EQ_LABELS:
            # CC 0 → -12 dB, CC 64 ≈ 0 dB, CC 127 → +12 dB
            val = (raw / 127.0) * 24.0 - 12.0
        elif label in _PARAM_RANGES:
            lo, hi, as_int = _PARAM_RANGES[label]
            if range_override is not None:
                lo, hi  = range_override[0], range_override[1]
                as_int  = isinstance(lo, int) and isinstance(hi, int)
            val = _scale(raw, lo, hi, as_int)
            setattr(self.params, label, val)
            logger.info("[PBF4] %s = %s (raw=%d)", label, val, raw)
            return
        else:
            logger.warning("[PBF4] No handler for CC label '%s'", label)
            return

        cb = self._cc_callbacks.get(label)
        if cb is not None:
            try:
                cb(val)
            except Exception as e:
                logger.error("[PBF4] %s callback error: %s", label, e)
        logger.info("[PBF4] %s = %.3f (raw=%d)", label, val, raw)

    # ── Accessors ─────────────────────────────────────────────────────────────

    def get_toggle(self, label: str) -> bool:
        """Current on/off state for a toggle button label."""
        return self._toggle_state.get(label, False)

    def print_status(self):
        print("[PBF4 Status]")
        for lbl in _TOGGLE_LABELS:
            state = "ON" if self.get_toggle(lbl) else "OFF"
            print(f"  {lbl:18s} = {state}")


# ─────────────────────────────────────────────────────────────────────────────
# Standalone: log-only validation (run from project root on Windows)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys, dataclasses

    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s.%(msecs)03d  %(message)s",
                        datefmt="%H:%M:%S")

    try:
        from src.magenta_backend import GenerationParams
    except ImportError:
        @dataclasses.dataclass
        class GenerationParams:
            guidance_weight: float = 3.0
            temperature: float = 1.0
            topk: int = 40
            model_feedback: float = 0.7
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
    ctrl.on_user_volume(lambda v: print(f">>> user_volume = {v:.3f}"))
    ctrl.on_voice_volume(0, lambda v: print(f">>> voice_0_volume = {v:.3f}"))
    ctrl.on_voice_volume(1, lambda v: print(f">>> voice_1_volume = {v:.3f}"))
    ctrl.on_voice_volume(2, lambda v: print(f">>> voice_2_volume = {v:.3f}"))
    ctrl.on_eq_bass(lambda db: print(f">>> eq_bass = {db:.1f} dB"))
    ctrl.on_eq_mid(lambda db: print(f">>> eq_mid = {db:.1f} dB"))
    ctrl.on_eq_treble(lambda db: print(f">>> eq_treble = {db:.1f} dB"))
    ctrl.on_reverb_wet(lambda v: print(f">>> reverb_wet = {v:.3f}"))
    ctrl.start()

    print("Listening. Move all controls. Ctrl+C to quit.\n")
    try:
        while True:
            time.sleep(2.0)
            ctrl.print_status()
    except KeyboardInterrupt:
        ctrl.stop()
