"""
CC + Note Discovery Script for intech PBF4 and MicroLab mk3.
Run this, move every knob/fader AND press every button on both controllers,
then press Ctrl+C. Results saved to config/pbf4_cc_map.json.

Buttons on the PBF4 send note_on messages (not CC).
This script now saves both CC controls and note-based buttons to the JSON.
"""
import mido
import json
import time
from pathlib import Path
from collections import defaultdict

MIDI_PORTS = mido.get_input_names()
print("Found MIDI ports:", MIDI_PORTS)

# CC: port -> channel -> cc -> list of values
cc_seen = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
# Notes (buttons): port -> channel -> note -> list of velocities
note_seen = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

print("\nListening for MIDI messages on ALL ports.")
print("Move every knob and fader, then press every button on the PBF4.")
print("Press Ctrl+C when done.\n")

# Some controllers expose multiple virtual ports (e.g. intech Grid device 0 =
# SysEx/config, device 1 = MIDI I/O). Skip ports that refuse to open.
open_ports = []
open_names = []
for p in MIDI_PORTS:
    try:
        port_obj = mido.open_input(p)
        open_ports.append(port_obj)
        open_names.append(p)
        print(f"  Opened: {p}")
    except Exception as e:
        print(f"  Skipped (cannot open): {p} — {e}")

if not open_ports:
    print("ERROR: no MIDI ports could be opened.")
    raise SystemExit(1)

MIDI_PORTS = open_names  # only the ones that opened
ports = open_ports

try:
    while True:
        for port, p in zip(MIDI_PORTS, ports):
            for msg in p.iter_pending():
                ts = time.strftime("%H:%M:%S")
                if msg.type == "control_change":
                    cc_seen[port][msg.channel][msg.control].append(msg.value)
                    print(f"[{ts}] {port} | CC  ch={msg.channel} cc={msg.control} val={msg.value}")
                elif msg.type in ("note_on", "note_off"):
                    note_seen[port][msg.channel][msg.note].append(msg.velocity)
                    print(f"[{ts}] {port} | {msg.type} ch={msg.channel} note={msg.note} vel={msg.velocity}")
        time.sleep(0.001)
except KeyboardInterrupt:
    pass

# Build output — two sections per port: "cc_controls" and "note_buttons"
result = {}
for port in set(list(cc_seen.keys()) + list(note_seen.keys())):
    result[port] = {"cc_controls": {}, "note_buttons": {}}

    for ch in cc_seen.get(port, {}):
        for cc, vals in cc_seen[port][ch].items():
            key = f"ch{ch}_cc{cc}"
            result[port]["cc_controls"][key] = {
                "channel": ch,
                "cc": cc,
                "min_val": min(vals),
                "max_val": max(vals),
                "samples": len(vals),
                "label": "UNKNOWN",
            }

    for ch in note_seen.get(port, {}):
        for note, vels in note_seen[port][ch].items():
            key = f"ch{ch}_note{note}"
            result[port]["note_buttons"][key] = {
                "channel": ch,
                "note": note,
                "velocities_seen": sorted(set(vels)),
                "label": "UNKNOWN",
            }

out_path = Path(__file__).parent.parent / "config" / "pbf4_cc_map.json"
out_path.parent.mkdir(exist_ok=True)
with open(out_path, "w") as f:
    json.dump(result, f, indent=2)

print(f"\nSaved to {out_path}")
print(f"CC controls:   {sum(len(v['cc_controls']) for v in result.values())}")
print(f"Note buttons:  {sum(len(v['note_buttons']) for v in result.values())}")
for port, data in result.items():
    print(f"\n  {port}")
    print(f"    CC:    {list(data['cc_controls'].keys())}")
    print(f"    Notes: {list(data['note_buttons'].keys())}")
