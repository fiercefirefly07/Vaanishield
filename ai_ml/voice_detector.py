
import argparse
from pathlib import Path

import numpy as np
import onnxruntime as ort

from audio_preprocessing import (
    SAMPLE_RATE,
    MODEL_SAMPLES,
    load_audio,
    preprocess_audio,
)


MODEL_PATH = (
    Path(__file__).resolve().parent
    / "models"
    / "aasist.onnx"
)


class VoiceDetector:
    def __init__(self):
        if not MODEL_PATH.is_file():
            raise FileNotFoundError(
                f"AASIST model not found: {MODEL_PATH}"
            )

        self.session = ort.InferenceSession(
            str(MODEL_PATH),
            providers=["CPUExecutionProvider"]
        )

        print("AASIST model loaded successfully!")

    def predict(self, audio, sample_rate):
        # Validate the source recording
        if np.asarray(audio).size == 0:
            raise ValueError("Empty audio")

        # Prepare a model-compatible waveform
        processed = preprocess_audio(audio, sample_rate)

        if processed.shape != (MODEL_SAMPLES,):
            raise ValueError("Incorrect input shape")

        rms = float(np.sqrt(
            np.mean(processed.astype(np.float64) ** 2)
        ))

        if rms < 0.001:
            raise ValueError(
                "Audio is silent or too quiet for analysis"
            )

        batch = processed[np.newaxis, :]

        # Run inference using the actual ONNX names
        outputs = self.session.run(
            ["logits"],
            {"wav": batch}
        )

        logits = np.asarray(
            outputs[0][0], dtype=np.float64
        )

        if logits.shape != (2,):
            raise ValueError("Unexpected model output")

        # Stable softmax
        shifted = logits - np.max(logits)
        exp_values = np.exp(shifted)
        scores = exp_values / np.sum(exp_values)

        # AASIST class convention:
        # 0 = spoof, 1 = bona fide
        spoof_score = float(scores[0])
        genuine_score = float(scores[1])

        predicted_class = (
            "SPOOF INDICATORS"
            if spoof_score > genuine_score
            else "BONA FIDE INDICATORS"
        )

        return {
            "spoof_score": spoof_score,
            "genuine_score": genuine_score,
            "prediction": predicted_class,
            "sample_rate": SAMPLE_RATE,
        }


def main():
    parser = argparse.ArgumentParser(
        description="VaaniShield AASIST detector"
    )

    group = parser.add_mutually_exclusive_group(
        required=True
    )
    group.add_argument(
        "--file",
        type=str,
        help="WAV or FLAC recording to analyse"
    )
    group.add_argument(
        "--self-test",
        action="store_true",
        help="Run an inference smoke test"
    )

    args = parser.parse_args()

    if args.self_test:
        # Artificial test signal: NOT human speech
        sr = SAMPLE_RATE
        t = np.arange(sr * 2) / sr
        audio = (
            0.2 * np.sin(2 * np.pi * 440 * t)
        ).astype(np.float32)

        print("Running synthetic-tone smoke test")
    else:
        audio, sr = load_audio(args.file)
        print(f"Analysing: {args.file}")

    detector = VoiceDetector()
    result = detector.predict(audio, sr)

    print("\n========== VAANISHIELD ==========")
    print(f"Spoof score:   {result['spoof_score']:.4f}")
    print(f"Genuine score: {result['genuine_score']:.4f}")
    print(f"Prediction:    {result['prediction']}")
    print("=================================")

    if args.self_test:
        print(
            "\nModel execution passed. "
            "The test tone is not speech, so "
            "its classification has no meaning."
        )
    else:
        print(
            "\nThese are uncalibrated model scores, "
            "not verified fraud probabilities."
        )


if __name__ == "__main__":
    main()
