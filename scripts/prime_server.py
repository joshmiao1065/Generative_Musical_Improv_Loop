import modal
import asyncio
import traceback

async def prime_voice(voice_index: int):
    """
    Connects to an existing deployment and triggers a dummy inference pass.
    Includes robust error logging to diagnose OOM or connection issues.
    """
    try:
        # Connect to the persistent deployed class
        VoiceServer = modal.Cls.from_name("magenta-rt-server", "VoiceServer")
        instance = VoiceServer(voice_index=voice_index)
        
        print(f"[Voice {voice_index}] Starting prime sequence...")
        
        # Trigger the remote prime method
        # We use .remote.aio() because this is an async client
        result = await instance.prime.remote.aio()
        
        print(f"[Voice {voice_index}] Success: {result}")
        return True

    except Exception as e:
        print(f"\n[Voice {voice_index}] FAILED to prime.")
        print(f"Error Type: {type(e).__name__}")
        print(f"Error Details: {str(e)}")
        # Print a concise traceback so you know if it's an OOM or a Network error
        traceback.print_exc()
        return False

async def main():
    print("Connecting to deployed 'magenta-rt-server'...")
    n_voices = 3
    
    # Run in parallel
    results = await asyncio.gather(*[prime_voice(i) for i in range(n_voices)])
    
    if all(results):
        print("\n✅ All voices successfully primed and warm.")
    else:
        print("\n❌ Some voices failed to prime. Check the Modal Dashboard logs.")

if __name__ == "__main__":
    asyncio.run(main())