
from pathlib import Path
import time

import numpy as np
import sounddevice as sd
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent

OUTPUT_DIR = ROOT / "data" / "evaluation" / "genuine"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_RATE = 16000
DURATION = 6
TOTAL_RECORDINGS = 5

PHRASES = [
    "Hello, I would like to check my account balance.",
    "I need assistance with my registered mobile number.",
    "Please tell me the status of my recent transaction.",
    "I am calling to ask about my banking services.",
    "I would like to speak with a customer service agent."
]


def next_filename():
    index = 2

    while True:
        path = OUTPUT_DIR / f"real_{index:02d}.wav"

        if not path.exists():
            return path

        index += 1


def record_audio():
    print("\nVAANISHIELD - GENUINE VOICE COLLECTION")
    print("=" * 45)

    for i, phrase in enumerate(PHRASES, start=1):
        print(f"\nRecording {i}/{TOTAL_RECORDINGS}")
        print(f"Suggested sentence: {phrase}")
        print("Speak naturally for about 6 seconds.")
        input("Press Enter when ready...")

        for second in (3, 2, 1):
            print(second)
            time.sleep(1)

        print("RECORDING NOW!")

        audio = sd.rec(
            int(DURATION * SAMPLE_RATE),
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32"
        )

        sd.wait()

        peak = float(np.max(np.abs(audio)))
        rms = float(np.sqrt(np.mean(audio ** 2)))

        print(f"Peak amplitude: {peak:.4f}")
        print(f"RMS: {rms:.4f}")

        if rms < 0.01:
            print("Warning: Recording may be too quiet.")

        if peak >= 0.999:
            print("Warning: Audio may be clipped.")

        output_path = next_filename()

        sf.write(
            str(output_path),
            audio,
            SAMPLE_RATE
        )

        print(f"Saved: {output_path.name}")

    print("\nAll five recordings completed!")
    print(f"Location: {OUTPUT_DIR}")


if __name__ == "__main__":
    record_audio()
