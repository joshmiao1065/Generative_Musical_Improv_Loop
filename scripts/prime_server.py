"""
prime_server.py — Verify that all 3 Modal voice containers are warm and ready.

With min_containers=1, containers begin warming the moment 'modal deploy' runs.
Cold boot takes ~5-8 min (model load + XLA compile). This script polls each
container with a visible elapsed-time display until they all respond.

What to expect:
  - Pings are QUEUED (pending) while @modal.enter() is still running (model loading)
  - They will NOT appear on the Modal dashboard until the container is fully ready
  - This is NORMAL — do not assume something is broken if pings hang for 5-8 min

Usage:
  .venv\\Scripts\\python scripts\\prime_server.py          # check all 3 voices
  .venv\\Scripts\\python scripts\\prime_server.py --voices 1

To watch container startup logs in real time (run in a separate terminal):
  modal logs magenta-rt-server

Expected log sequence per container:
  [Voice N] Container started. GPU: ...
  [Voice N] Loading SpectroStream model...   (~10-20s)
  [Voice N] Loading MagentaRT large model... (~30-60s, instant if HF cache warm)
  [Voice N] Running JIT warm-up...           (~2-4 min first time)
  [Voice N] JIT compile done in ...s
  [Voice N] *** READY ***                    ← ping will return after this line
"""

import asyncio
import argparse
import sys
import time
import traceback

import modal

APP_NAME          = "magenta-rt-server"
VOICE_CLASS_NAMES = ["Voice0Server", "Voice1Server", "Voice2Server"]
POLL_INTERVAL_S   = 10   # how often to print elapsed-time updates while waiting
TIMEOUT_S         = 600  # 10 min — covers worst-case cold boot


async def ping_voice(voice_index: int) -> bool:
    name = VOICE_CLASS_NAMES[voice_index]
    print(f"[Voice {voice_index}] Pinging {name}...")
    print(f"[Voice {voice_index}] Note: ping is QUEUED until @modal.enter() finishes "
          f"(model load + XLA compile, ~5-8 min first boot).")

    try:
        cls      = modal.Cls.from_name(APP_NAME, name)
        instance = cls()
    except modal.exception.NotFoundError:
        print(f"[Voice {voice_index}] ERROR: App '{APP_NAME}' not found. "
              f"Run: modal deploy server/magenta_server.py")
        return False
    except Exception as e:
        print(f"[Voice {voice_index}] ERROR connecting: {e}")
        traceback.print_exc()
        return False

    t0 = time.perf_counter()

    # Fire the ping as a background task so we can print progress while waiting
    ping_task = asyncio.create_task(instance.ping.remote.aio())

    while not ping_task.done():
        elapsed = time.perf_counter() - t0
        print(f"[Voice {voice_index}] Waiting... {elapsed:.0f}s  "
              f"(run 'modal logs magenta-rt-server' in another terminal to watch)",
              end="\r", flush=True)
        try:
            await asyncio.wait_for(asyncio.shield(ping_task), timeout=POLL_INTERVAL_S)
        except asyncio.TimeoutError:
            pass  # expected — just update the progress line

    print()  # end the \r line

    elapsed = time.perf_counter() - t0
    try:
        result = ping_task.result()
        print(f"[Voice {voice_index}] READY in {elapsed:.0f}s — {result}")
        return True
    except asyncio.TimeoutError:
        print(f"[Voice {voice_index}] TIMED OUT after {elapsed:.0f}s. "
              f"Check 'modal logs magenta-rt-server' for errors.")
        return False
    except Exception as e:
        print(f"[Voice {voice_index}] FAILED after {elapsed:.0f}s: {type(e).__name__}: {e}")
        traceback.print_exc()
        return False


async def main(n_voices: int):
    print(f"Checking {n_voices} voice container(s) for '{APP_NAME}'...")
    print(f"Expected wait: <10s if containers already warm, ~5-8 min if cold boot.\n")

    # Run all pings in parallel — total wait = max(V0, V1, V2), not their sum
    results = await asyncio.gather(*[ping_voice(i) for i in range(n_voices)])

    n_ready  = sum(results)
    n_failed = n_voices - n_ready

    print()
    if all(results):
        print("=" * 50)
        print(f"  ALL {n_voices} VOICE(S) READY.")
        print("=" * 50)
    else:
        failed = [i for i, ok in enumerate(results) if not ok]
        print(f"WARNING: {n_failed} voice(s) did not respond: {failed}")
        print("Steps to diagnose:")
        print("  1. modal logs magenta-rt-server   (stream container logs)")
        print("  2. Check modal.com/apps → magenta-rt-server → container status")
        print("  3. If containers show 'active' but no *** READY *** in logs yet, wait longer")
        print("  4. If containers never show as active, check billing: modal.com/settings")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--voices", type=int, default=3, choices=[1, 2, 3],
                        help="Number of voice containers to check (default: 3)")
    args = parser.parse_args()
    asyncio.run(main(args.voices))
