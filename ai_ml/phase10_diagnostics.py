
import argparse
import sys
import time

from pathlib import Path

import numpy as np
import soundfile as sf


# ==========================================
# PROJECT IMPORTS
# ==========================================

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ai_ml.audio_preprocessing import (
    preprocess_audio,
    SAMPLE_RATE,
    MODEL_SAMPLES,
)

from backend.app.main import get_model


# ==========================================
# PATHS
# ==========================================

REAL_VOICE = (
    ROOT
    / "data"
    / "sample_audio"
    / "real_voice.wav"
)

GENUINE_DIR = (
    ROOT
    / "data"
    / "evaluation"
    / "genuine"
)

SPOOF_DIR = (
    ROOT
    / "data"
    / "evaluation"
    / "spoof"
)


# ==========================================
# AUDIO QUALITY
# ==========================================

def inspect_audio(audio, sample_rate):
    mono = np.mean(audio, axis=1)

    duration = len(mono) / sample_rate

    peak = float(
        np.max(np.abs(mono))
    )

    rms = float(
        np.sqrt(
            np.mean(
                mono.astype(np.float64) ** 2
            )
        )
    )

    clipping_ratio = float(
        np.mean(np.abs(mono) >= 0.99)
    )

    quiet_ratio = float(
        np.mean(np.abs(mono) < 0.005)
    )

    return {
        "duration": duration,
        "sample_rate": sample_rate,
        "channels": audio.shape[1],
        "peak": peak,
        "rms": rms,
        "clipping_ratio": clipping_ratio,
        "quiet_ratio": quiet_ratio,
    }


# ==========================================
# MODEL INFERENCE
# ==========================================

def predict(session, audio, sample_rate):
    processed = preprocess_audio(
        audio,
        sample_rate,
    )

    processed = np.asarray(
        processed,
        dtype=np.float32,
    )

    if processed.shape != (MODEL_SAMPLES,):
        raise ValueError(
            f"Unexpected model input shape: "
            f"{processed.shape}"
        )

    if not np.all(np.isfinite(processed)):
        raise ValueError(
            "Non-finite audio samples"
        )

    rms = float(
        np.sqrt(
            np.mean(
                processed.astype(np.float64) ** 2
            )
        )
    )

    if rms < 0.001:
        raise ValueError(
            "Audio too quiet for analysis"
        )

    batch = processed[np.newaxis, :]

    start = time.perf_counter()

    output = session.run(
        ["logits"],
        {"wav": batch},
    )

    inference_time = (
        time.perf_counter() - start
    )

    logits = np.asarray(
        output[0][0],
        dtype=np.float64,
    )

    if (
        logits.shape != (2,)
        or not np.all(np.isfinite(logits))
    ):
        raise ValueError(
            "Invalid model logits"
        )

    # Assumed model convention:
    # Class 0 = spoof
    # Class 1 = genuine
    shifted = logits - np.max(logits)

    values = np.exp(shifted)

    scores = values / values.sum()

    spoof_score = float(scores[0])
    genuine_score = float(scores[1])

    prediction = (
        "SPOOF_INDICATORS"
        if spoof_score > genuine_score
        else "GENUINE_INDICATORS"
    )

    return {
        "spoof_score": spoof_score,
        "genuine_score": genuine_score,
        "prediction": prediction,
        "inference_time": inference_time,
        "logits": logits.tolist(),
    }


# ==========================================
# TEST ONE GENUINE RECORDING
# ==========================================

def test_single_recording(session, path):
    print("\n" + "=" * 60)
    print("PHASE 10 - SINGLE RECORDING DIAGNOSTICS")
    print("=" * 60)

    if not path.is_file():
        print(f"Audio file not found: {path}")
        return

    audio, sample_rate = sf.read(
        path,
        dtype="float32",
        always_2d=True,
    )

    quality = inspect_audio(
        audio,
        sample_rate,
    )

    print("\nAUDIO INFORMATION")
    print("File:", path.name)

    for key, value in quality.items():
        if isinstance(value, float):
            print(f"{key}: {value:.6f}")
        else:
            print(f"{key}: {value}")

    # --------------------------------------
    # ORIGINAL BACKEND PREPROCESSING
    # --------------------------------------

    print("\nTEST A - FULL FILE PREPROCESSING")

    full_result = predict(
        session,
        audio,
        sample_rate,
    )

    display_result(full_result)

    # --------------------------------------
    # COMPARE DIFFERENT AUDIO REGIONS
    # --------------------------------------

    required_duration = (
        MODEL_SAMPLES / SAMPLE_RATE
    )

    if quality["duration"] > required_duration:

        samples_needed = max(
            1,
            round(
                required_duration * sample_rate
            ),
        )

        first_audio = audio[:samples_needed]

        last_audio = audio[-samples_needed:]

        print(
            "\nTEST B - FIRST 4.04 SECONDS"
        )

        first_result = predict(
            session,
            first_audio,
            sample_rate,
        )

        display_result(first_result)

        print(
            "\nTEST C - LAST 4.04 SECONDS"
        )

        last_result = predict(
            session,
            last_audio,
            sample_rate,
        )

        display_result(last_result)

        scores = [
            full_result["spoof_score"],
            first_result["spoof_score"],
            last_result["spoof_score"],
        ]

        variation = max(scores) - min(scores)

        print("\nSCORE VARIATION")
        print(
            "Maximum difference:",
            f"{variation * 100:.2f} percentage points"
        )

        if variation > 0.30:
            print(
                "Observation: Large score changes "
                "across segments of the same recording."
            )
        else:
            print(
                "Observation: Scores are relatively "
                "similar across these audio segments."
            )

    print("\nNote:")
    print(
        "This test checks model sensitivity to "
        "recording segments, not fraud probability."
    )


def display_result(result):
    print(
        "Spoof score:",
        f"{result['spoof_score'] * 100:.4f}%"
    )

    print(
        "Genuine score:",
        f"{result['genuine_score'] * 100:.4f}%"
    )

    print(
        "Prediction:",
        result["prediction"]
    )

    print(
        "Inference time:",
        f"{result['inference_time']:.4f}s"
    )

    print("Raw logits:", result["logits"])


# ==========================================
# DATASET EVALUATION
# ==========================================

def list_audio_files(folder):
    if not folder.exists():
        return []

    return sorted(
        path
        for path in folder.rglob("*")
        if (
            path.is_file()
            and path.suffix.lower()
            in {".wav", ".flac"}
        )
    )


def test_dataset(session):
    print("\n" + "=" * 60)
    print("PHASE 10 - DATASET EVALUATION")
    print("=" * 60)

    genuine_files = list_audio_files(
        GENUINE_DIR
    )

    spoof_files = list_audio_files(
        SPOOF_DIR
    )

    print(
        "\nGenuine samples:",
        len(genuine_files)
    )

    print(
        "Spoof samples:",
        len(spoof_files)
    )

    if not genuine_files or not spoof_files:
        print(
            "\nMissing evaluation files. "
            "Check data/evaluation folders."
        )
        return

    results = []

    for label, files in [
        ("GENUINE", genuine_files),
        ("SPOOF", spoof_files),
    ]:
        print(
            f"\n--- {label} AUDIO ---"
        )

        for path in files:
            try:
                audio, sr = sf.read(
                    path,
                    dtype="float32",
                    always_2d=True,
                )

                result = predict(
                    session,
                    audio,
                    sr,
                )

                predicted_spoof = (
                    result["spoof_score"] > 0.5
                )

                expected_spoof = (
                    label == "SPOOF"
                )

                correct = (
                    predicted_spoof == expected_spoof
                )

                results.append({
                    "label": label,
                    "file": path.name,
                    "score": result["spoof_score"],
                    "correct": correct,
                })

                print(
                    f"{path.name[:35]:35s} "
                    f"spoof={result['spoof_score']:.4f} "
                    f"{'OK' if correct else 'WRONG'}"
                )

            except Exception as exc:
                print(
                    f"{path.name}: "
                    f"SKIPPED ({exc})"
                )

    genuine = [
        item for item in results
        if item["label"] == "GENUINE"
    ]

    spoof = [
        item for item in results
        if item["label"] == "SPOOF"
    ]

    false_positives = [
        item for item in genuine
        if not item["correct"]
    ]

    false_negatives = [
        item for item in spoof
        if not item["correct"]
    ]

    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)

    print(
        "Evaluated genuine samples:",
        len(genuine)
    )

    print(
        "Evaluated spoof samples:",
        len(spoof)
    )

    print(
        "False positives:",
        len(false_positives)
    )

    print(
        "False negatives:",
        len(false_negatives)
    )

    if genuine:
        fpr = (
            len(false_positives)
            / len(genuine)
        )

        print(
            "Genuine false-positive rate:",
            f"{fpr * 100:.2f}%"
        )

    if spoof:
        fnr = (
            len(false_negatives)
            / len(spoof)
        )

        print(
            "Spoof false-negative rate:",
            f"{fnr * 100:.2f}%"
        )

    print(
        "\nMisclassified genuine recordings:"
    )

    if not false_positives:
        print("None in this evaluation set.")

    for item in false_positives:
        print(
            f"- {item['file']}: "
            f"{item['score'] * 100:.2f}% "
            "spoof score"
        )

    print("\nImportant:")
    print(
        "These are exploratory results on a "
        "small dataset, not validated banking "
        "performance estimates."
    )


# ==========================================
# MAIN
# ==========================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        choices=["single", "dataset"],
        default="single",
    )

    parser.add_argument(
        "--file",
        type=Path,
        default=REAL_VOICE,
    )

    args = parser.parse_args()

    print("\nVAANISHIELD - PHASE 10")
    print("AI MODEL DIAGNOSTICS")
    print("Model: W2V2-AASIST")

    session = get_model()

    print(
        "ONNX providers:",
        session.get_providers()
    )

    print(
        "CUDA provider enabled:",
        "CUDAExecutionProvider"
        in session.get_providers()
    )

    if args.mode == "single":
        test_single_recording(
            session,
            args.file,
        )
    else:
        test_dataset(session)


if __name__ == "__main__":
    main()
