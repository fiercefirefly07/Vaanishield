
import asyncio
import json

import sounddevice as sd
from websockets.asyncio.client import connect


SERVER_URL = "ws://127.0.0.1:8000/ws/live-audio"

SAMPLE_RATE = 16000
CHUNK_SAMPLES = 8000
DURATION = 12


async def receive_results(websocket):
    async for message in websocket:
        result = json.loads(message)

        if result["type"] == "prediction":
            print("\n--- LIVE AI ANALYSIS ---")
            print(
                "Audio window:",
                result["window_start_seconds"],
                "to",
                result["window_end_seconds"],
                "seconds",
            )
            print(
                "Spoof score:",
                result["spoof_score"],
            )
            print(
                "Genuine score:",
                result["genuine_score"],
            )
            print(
                "Inference time:",
                result["inference_seconds"],
                "seconds",
            )
            print("Mode: MONITOR_ONLY")

        elif result["type"] == "audio_quality":
            print(
                "Audio quality warning:",
                result["status"],
            )

        elif result["type"] == "error":
            print("Server error:", result["message"])

        elif result["type"] == "finished":
            print(
                "\nStream finished:",
                result["duration_seconds"],
                "seconds",
            )


async def main():
    print("VAANISHIELD — LIVE MICROPHONE TEST")
    print("Connecting to streaming server...")

    async with connect(SERVER_URL) as websocket:
        ready = json.loads(await websocket.recv())

        print("Connected:", ready["type"])
        print("Model:", ready["model"])

        receiver = asyncio.create_task(
            receive_results(websocket)
        )

        print(
            f"\nRecording and streaming for {DURATION} seconds."
        )
        print("Speak continuously and clearly.\n")

        with sd.RawInputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="int16",
            blocksize=CHUNK_SAMPLES,
        ) as stream:

            for _ in range(
                DURATION * SAMPLE_RATE // CHUNK_SAMPLES
            ):
                data, overflowed = await asyncio.to_thread(
                    stream.read,
                    CHUNK_SAMPLES,
                )

                if overflowed:
                    print("Warning: Microphone input overflow")

                await websocket.send(bytes(data))

        await websocket.send("stop")
        await receiver

    print("\nLive microphone test completed!")


if __name__ == "__main__":
    asyncio.run(main())
