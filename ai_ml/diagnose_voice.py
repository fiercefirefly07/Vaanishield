
from pathlib import Path

import numpy as np

from audio_preprocessing import (
    MODEL_SAMPLES,
    SAMPLE_RATE,
    load_audio,
)
from voice_detector import VoiceDetector


PROJECT_ROOT = Path(__file__).resolve().parent.parent
AUDIO_PATH = (
    PROJECT_ROOT / "data" / "sample_audio" / "real_voice.wav"
)


def rms(audio):
    audio = np.asarray(audio, dtype=np.float64)

    if audio.size == 0:
        return 0.0

    return float(np.sqrt(np.mean(audio ** 2)))


def main():
    audio, sr = load_audio(AUDIO_PATH)

    # Create mono audio for diagnostics
    if audio.ndim == 2:
        mono = np.mean(audio, axis=1)
    else:
        mono = audio

    duration = len(mono) / sr
    window_duration = MODEL_SAMPLES / SAMPLE_RATE

    print("\nVAANISHIELD - AUDIO DIAGNOSTICS")
    print("--------------------------------")
    print(f"Sample rate: {sr} Hz")
    print(f"Duration: {duration:.2f}s")
    print(f"Peak: {np.max(np.abs(mono)):.4f}")
    print(f"Overall RMS: {rms(mono):.4f}")
    print(f"First-second RMS: {rms(mono[:sr]):.4f}")

    clipped = np.sum(np.abs(mono) >= 0.999)
    print(f"Clipped samples: {clipped}")

    print("\nANALYSING DIFFERENT AUDIO WINDOWS")

    detector = VoiceDetector()

    last_start = max(0.0, duration - window_duration)

    # First, middle and final full-size windows
    start_times = [
        0.0,
        min(1.0, last_start),
        last_start,
    ]

    seen = set()

    for start in start_times:
        start_sample = int(start * sr)

        if start_sample in seen:
            continue

        seen.add(start_sample)

        end_sample = start_sample + int(
            np.ceil(window_duration * sr)
        )

        segment = audio[start_sample:end_sample]

        result = detector.predict(segment, sr)

        print(f"\nWindow starting at {start:.2f}s")
        print(f"Spoof score: {result['spoof_score']:.4f}")
        print(f"Genuine score: {result['genuine_score']:.4f}")
        print(f"Prediction: {result['prediction']}")

    print("\nDiagnostics completed!")


if __name__ == "__main__":
    main()
