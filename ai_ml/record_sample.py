
import time
from pathlib import Path

import sounddevice as sd
import soundfile as sf
import numpy as np

# Recording settings
SAMPLE_RATE = 16000
DURATION = 6
CHANNELS = 1

# Create output directory
project_root = Path(__file__).resolve().parent.parent
output_dir = project_root / "data" / "sample_audio"
output_dir.mkdir(parents=True, exist_ok=True)

output_file = output_dir / "real_voice.wav"

print("VAANISHIELD - VOICE RECORDER")
print("Preparing to record your voice...")

for i in [3, 2, 1]:
    print(i)
    time.sleep(1)

print("\nRecording! Speak clearly...")

recording = sd.rec(
    int(DURATION * SAMPLE_RATE),
    samplerate=SAMPLE_RATE,
    channels=CHANNELS,
    dtype="float32"
)

sd.wait()

sf.write(
    str(output_file),
    recording,
    SAMPLE_RATE
)

print("\nRecording completed!")
print(f"Saved to: {output_file}")
print(f"Duration: {DURATION} seconds")
print(f"Sample Rate: {SAMPLE_RATE} Hz")
print(f"Peak amplitude: {np.max(np.abs(recording)):.4f}")
