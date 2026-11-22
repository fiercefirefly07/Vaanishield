
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf


SAMPLE_RATE = 16000
MODEL_SAMPLES = 64600


def load_audio(file_path):
    """Load a WAV or FLAC recording."""
    path = Path(file_path)

    if not path.is_file():
        raise FileNotFoundError(f"Audio not found: {path}")

    audio, sample_rate = sf.read(
        str(path),
        dtype="float32",
        always_2d=True
    )

    return audio, sample_rate


def preprocess_audio(audio, sample_rate):
    """Prepare a waveform for the AASIST model."""

    if sample_rate <= 0:
        raise ValueError("Invalid sample rate")

    audio = np.asarray(audio, dtype=np.float32)

    if audio.size == 0:
        raise ValueError("Audio is empty")

    # Convert multichannel audio to mono
    if audio.ndim == 2:
        audio = np.mean(audio, axis=1)
    elif audio.ndim != 1:
        raise ValueError("Unsupported audio shape")

    if not np.all(np.isfinite(audio)):
        raise ValueError("Audio contains invalid values")

    # Convert to 16 kHz
    if sample_rate != SAMPLE_RATE:
        audio = librosa.resample(
            audio,
            orig_sr=sample_rate,
            target_sr=SAMPLE_RATE
        )

    audio = np.clip(audio, -1.0, 1.0)

    # Follow AASIST evaluation preprocessing:
    # crop long audio, repeat short audio
    if len(audio) >= MODEL_SAMPLES:
        audio = audio[:MODEL_SAMPLES]
    else:
        repeats = int(np.ceil(
            MODEL_SAMPLES / len(audio)
        ))
        audio = np.tile(audio, repeats)[:MODEL_SAMPLES]

    return audio.astype(np.float32)


if __name__ == "__main__":
    # Test preprocessing using a generated signal
    original_sr = 44100
    t = np.arange(original_sr * 2) / original_sr
    tone = 0.2 * np.sin(2 * np.pi * 440 * t)

    processed = preprocess_audio(tone, original_sr)

    print("VAANISHIELD AUDIO PREPROCESSING")
    print(f"Sample rate: {SAMPLE_RATE} Hz")
    print(f"Samples: {len(processed)}")
    print(f"Shape: {processed.shape}")
    print(f"Duration: {len(processed)/SAMPLE_RATE:.4f}s")
    print("Preprocessing successful!")
