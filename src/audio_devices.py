"""
audio_devices.py — Auto-detect capture and playback device indices.

Priority order for capture:
  1. VB-Cable "CABLE Output" (if installed — clean Surge-only feed)
  2. Stereo Mix at 48 kHz (built-in loopback — works but captures all speaker output)

Priority order for playback:
  1. WASAPI Speakers at 48 kHz (lowest latency, no resampling)
  2. MME Speakers (fallback)

Usage:
    from src.audio_devices import detect, AudioDevices

    devs = detect()          # auto-detect
    print(devs)

    # or override one or both indices:
    devs = detect(capture_idx=13, playback_idx=8)
"""

import dataclasses
import logging
from typing import Optional

import sounddevice as sd

logger = logging.getLogger(__name__)

SAMPLE_RATE = 48000
CHANNELS    = 2


@dataclasses.dataclass
class AudioDevices:
    capture_idx:   int
    capture_name:  str
    playback_idx:  int
    playback_name: str
    sample_rate:   int = SAMPLE_RATE
    channels:      int = CHANNELS

    def __str__(self):
        return (
            f"AudioDevices(\n"
            f"  capture  [{self.capture_idx:2d}]: {self.capture_name}\n"
            f"  playback [{self.playback_idx:2d}]: {self.playback_name}\n"
            f"  format: {self.channels}ch @ {self.sample_rate} Hz\n"
            f")"
        )


def _all_devices():
    return [(i, sd.query_devices(i)) for i in range(len(sd.query_devices()))]


def list_devices() -> None:
    """Print a formatted table of all audio devices. Useful for picking capture/playback indices."""
    devs = sd.query_devices()
    print(f"\n{'Idx':>4}  {'In':>3}  {'Out':>4}  {'Rate':>6}  {'API':<14}  Name")
    print("-" * 78)
    for i, d in enumerate(devs):
        api = sd.query_hostapis(d["hostapi"])["name"][:14]
        print(
            f"{i:>4}  {d['max_input_channels']:>3}  {d['max_output_channels']:>4}  "
            f"{int(d['default_samplerate']):>6}  {api:<14}  {d['name']}"
        )
    print()

    # Quick VB-Cable status
    names = [d["name"].lower() for d in devs]
    if any("cable output" in n for n in names):
        print("  VB-Cable: FOUND (cable output detected) -- preferred capture device")
    else:
        print("  VB-Cable: NOT FOUND -- install from https://shop.vb-audio.com/en/win-apps/11-vb-cable.html")

    if any("stereo mix" in n for n in names):
        print("  Stereo Mix: FOUND (fallback capture -- causes AI feedback loop without VB-Cable)")
    print()


def list_midi_ports() -> None:
    """Print all available MIDI input ports."""
    try:
        import mido
        ports = mido.get_input_names()
        print(f"MIDI Input Ports ({len(ports)}):")
        for i, name in enumerate(ports):
            print(f"  [{i}] {name}")
        print()
    except ImportError:
        print("  mido not installed — run: pip install mido python-rtmidi")


def vb_cable_installed() -> bool:
    """Return True if VB-Cable CABLE Output is visible as a capture device."""
    return any(
        "cable output" in d["name"].lower()
        for _, d in _all_devices()
        if d["max_input_channels"] >= CHANNELS
    )


def _find_capture(
    override_idx: Optional[int],
    name_substr: Optional[str],
) -> tuple[int, str]:
    """Return (index, name) for the best available capture device."""
    if override_idx is not None:
        d = sd.query_devices(override_idx)
        return override_idx, d["name"]

    devs = _all_devices()

    # Explicit name override (e.g. --capture-device "Stereo Mix" or "CABLE Output")
    if name_substr is not None:
        sub = name_substr.lower()
        for idx, d in devs:
            if sub in d["name"].lower() and d["max_input_channels"] >= CHANNELS:
                logger.info("[AudioDevices] Capture (--capture-device match): [%d] %s", idx, d["name"])
                return idx, d["name"]
        raise RuntimeError(
            f"No capture device matching '{name_substr}' found.\n"
            f"Run with --list-devices to see all available devices."
        )

    # 1. VB-Cable (preferred — clean Surge/Analog Lab-only feed, no AI feedback)
    for idx, d in devs:
        if "cable output" in d["name"].lower() and d["max_input_channels"] >= CHANNELS:
            logger.info("[AudioDevices] VB-Cable capture: [%d] %s", idx, d["name"])
            return idx, d["name"]

    # 2. Stereo Mix at native 48 kHz
    for idx, d in devs:
        if (
            "stereo mix" in d["name"].lower()
            and d["max_input_channels"] >= CHANNELS
            and int(d["default_samplerate"]) == SAMPLE_RATE
        ):
            logger.warning(
                "[AudioDevices] Stereo Mix selected [%d] %s\n"
                "  WARNING: captures ALL speaker audio — AI voices will feed back into the model.\n"
                "  This degrades AI output quality over time. Install VB-Cable ($5) to fix.",
                idx, d["name"],
            )
            return idx, d["name"]

    # 3. Any Stereo Mix (wrong sample rate — sounddevice will resample)
    for idx, d in devs:
        if "stereo mix" in d["name"].lower() and d["max_input_channels"] >= CHANNELS:
            logger.warning(
                "[AudioDevices] Stereo Mix found but not at 48 kHz: [%d] %s — "
                "sounddevice will resample (may cause pitch shift).",
                idx, d["name"],
            )
            return idx, d["name"]

    raise RuntimeError(
        "No usable capture device found.\n"
        "Options:\n"
        "  1. Install VB-Cable ($5): https://shop.vb-audio.com/en/win-apps/11-vb-cable.html\n"
        "     Route Surge XT or Analog Lab → 'CABLE Input' in audio settings.\n"
        "  2. Enable Stereo Mix: Windows Sound Settings → Recording tab → Stereo Mix → Enable\n"
        "  3. Override explicitly: --capture-device 'device name substring'"
    )


def _find_playback(
    override_idx: Optional[int],
    name_substr: Optional[str],
) -> tuple[int, str]:
    """Return (index, name) for the best available playback device."""
    if override_idx is not None:
        d = sd.query_devices(override_idx)
        return override_idx, d["name"]

    devs = _all_devices()

    # Explicit name override (e.g. --playback-device "Headphones")
    if name_substr is not None:
        sub = name_substr.lower()
        for idx, d in devs:
            if sub in d["name"].lower() and d["max_output_channels"] >= CHANNELS:
                logger.info("[AudioDevices] Playback (--playback-device match): [%d] %s", idx, d["name"])
                return idx, d["name"]
        raise RuntimeError(
            f"No playback device matching '{name_substr}' found.\n"
            f"Run with --list-devices to see all available devices."
        )

    # 1. WASAPI Speakers at 48 kHz (lowest latency, no resampling)
    for idx, d in devs:
        api_name = sd.query_hostapis(d["hostapi"])["name"]
        if (
            "wasapi" in api_name.lower()
            and "speaker" in d["name"].lower()
            and d["max_output_channels"] >= CHANNELS
            and int(d["default_samplerate"]) == SAMPLE_RATE
        ):
            logger.info("[AudioDevices] WASAPI Speakers: [%d] %s", idx, d["name"])
            return idx, d["name"]

    # 2. Any WASAPI output at 48 kHz
    for idx, d in devs:
        api_name = sd.query_hostapis(d["hostapi"])["name"]
        if (
            "wasapi" in api_name.lower()
            and d["max_output_channels"] >= CHANNELS
            and int(d["default_samplerate"]) == SAMPLE_RATE
        ):
            logger.info("[AudioDevices] WASAPI output: [%d] %s", idx, d["name"])
            return idx, d["name"]

    # 3. MME Speakers (fallback — higher latency, may resample)
    for idx, d in devs:
        if "speaker" in d["name"].lower() and d["max_output_channels"] >= CHANNELS:
            logger.warning("[AudioDevices] Falling back to MME: [%d] %s", idx, d["name"])
            return idx, d["name"]

    raise RuntimeError("No usable playback device found.")


def detect(
    capture_idx:   Optional[int] = None,
    playback_idx:  Optional[int] = None,
    capture_name:  Optional[str] = None,
    playback_name: Optional[str] = None,
) -> AudioDevices:
    """
    Auto-detect audio devices and return an AudioDevices instance.

    Parameters
    ----------
    capture_idx : int, optional
        Force a specific capture device index (skips auto-detection).
    playback_idx : int, optional
        Force a specific playback device index (skips auto-detection).
    capture_name : str, optional
        Substring to match against device names for capture (e.g. "CABLE Output",
        "Stereo Mix"). Overrides auto-detection priority order. Ignored if
        capture_idx is set.
    playback_name : str, optional
        Substring to match against device names for playback (e.g. "Speakers",
        "Headphones"). Ignored if playback_idx is set.

    Raises
    ------
    RuntimeError
        If no suitable capture or playback device is found.
    """
    cap_idx,  cap_name  = _find_capture(capture_idx, capture_name)
    play_idx, play_name = _find_playback(playback_idx, playback_name)

    devs = AudioDevices(
        capture_idx=cap_idx,
        capture_name=cap_name,
        playback_idx=play_idx,
        playback_name=play_name,
    )
    logger.info("[AudioDevices] Selected:\n%s", devs)
    return devs


def validate(devs: AudioDevices) -> bool:
    """
    Open both devices briefly to confirm they accept 48 kHz stereo.
    Uses callback mode — required for WDM-KS devices (e.g. Stereo Mix).
    Returns True if both work, False otherwise.
    """
    import threading
    ok = True

    # Capture: open with a no-op callback (WDM-KS requires callback mode)
    try:
        opened = threading.Event()
        def _noop_in(indata, frames, time_info, status):
            opened.set()
        stream = sd.InputStream(
            device=devs.capture_idx,
            samplerate=devs.sample_rate,
            channels=devs.channels,
            dtype="float32",
            blocksize=2048,
            callback=_noop_in,
        )
        with stream:
            opened.wait(timeout=1.0)
        logger.info("[AudioDevices] Capture device OK: [%d] %s",
                    devs.capture_idx, devs.capture_name)
    except Exception as e:
        logger.error("[AudioDevices] Capture device FAILED [%d] %s: %s",
                     devs.capture_idx, devs.capture_name, e)
        ok = False

    # Playback: WASAPI supports both blocking and callback; callback is safer
    try:
        def _noop_out(outdata, frames, time_info, status):
            outdata[:] = 0
        stream = sd.OutputStream(
            device=devs.playback_idx,
            samplerate=devs.sample_rate,
            channels=devs.channels,
            dtype="float32",
            blocksize=2048,
            callback=_noop_out,
        )
        with stream:
            pass
        logger.info("[AudioDevices] Playback device OK: [%d] %s",
                    devs.playback_idx, devs.playback_name)
    except Exception as e:
        logger.error("[AudioDevices] Playback device FAILED [%d] %s: %s",
                     devs.playback_idx, devs.playback_name, e)
        ok = False

    return ok


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    list_devices()
    list_midi_ports()

    print("VB-Cable installed:", vb_cable_installed())
    print()

    devs = detect()
    print(devs)
    print("Validating devices...")
    ok = validate(devs)
    print("PASS" if ok else "FAIL — check errors above")
    sys.exit(0 if ok else 1)
