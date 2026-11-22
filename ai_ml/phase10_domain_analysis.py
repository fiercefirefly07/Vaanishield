
import csv
import statistics
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ai_ml.audio_preprocessing import (
    preprocess_audio,
    SAMPLE_RATE,
    MODEL_SAMPLES,
)
from backend.app.main import get_model


# ============================================
# CONFIGURATION
# ============================================

GENUINE_DIR = ROOT / "data" / "evaluation" / "genuine"
SPOOF_DIR = ROOT / "data" / "evaluation" / "spoof"

REPORT_PATH = (
    ROOT
    / "data"
    / "diagnostics"
    / "phase10_domain_report.csv"
)

SUPPORTED_EXTENSIONS = {".wav", ".flac"}

MODEL_DURATION = MODEL_SAMPLES / SAMPLE_RATE


# ============================================
# DISCOVER AUDIO
# ============================================

def audio_files(folder):
    if not folder.exists():
        return []

    return sorted(
        path
        for path in folder.rglob("*")
        if (
            path.is_file()
            and path.suffix.lower()
            in SUPPORTED_EXTENSIONS
        )
    )


def build_test_set():
    samples = []

    for path in audio_files(GENUINE_DIR):
        if path.name.lower().startswith("real_"):
            group = "MIC_GENUINE"
        else:
            group = "OTHER_GENUINE"

        samples.append((group, path))

    for path in audio_files(SPOOF_DIR):
        samples.append(("SPOOF", path))

    return samples


# ============================================
# RECORDING QUALITY
# ============================================

def inspect_recording(audio, sample_rate):
    mono = audio.mean(axis=1).astype(np.float64)

    if mono.size == 0:
        raise ValueError("Empty recording")

    if not np.all(np.isfinite(mono)):
        raise ValueError("Invalid recording samples")

    rms = float(np.sqrt(np.mean(mono ** 2)))
    peak = float(np.max(np.abs(mono)))

    near_silence = float(
        np.mean(np.abs(mono) < 0.005) * 100
    )

    clipping = float(
        np.mean(np.abs(mono) >= 0.99) * 100
    )

    return {
        "sample_rate": sample_rate,
        "channels": audio.shape[1],
        "duration_sec": len(audio) / sample_rate,
        "rms": rms,
        "peak": peak,
        "near_silence_pct": near_silence,
        "clipping_pct": clipping,
    }


# ============================================
# SELECT AUDIO SEGMENTS
# ============================================

def select_segments(audio, sample_rate):
    """
    Choose up to three segments from each recording:

    - Beginning
    - Middle
    - End

    Very short recordings use one segment.
    This keeps the evaluation at file level.
    """

    samples_per_window = round(
        MODEL_DURATION * sample_rate
    )

    max_start = max(
        0,
        len(audio) - samples_per_window
    )

    # Avoid comparing almost identical windows
    # when the recording is only slightly longer
    # than the model's input duration.
    if max_start < sample_rate * 0.5:
        starts = [0]
    else:
        starts = [
            0,
            max_start // 2,
            max_start,
        ]

    starts = sorted(set(starts))

    segments = []

    for start in starts:
        segment = audio[
            start:start + samples_per_window
        ]

        segments.append({
            "start_seconds": start / sample_rate,
            "audio": segment,
        })

    return segments


# ============================================
# MODEL INFERENCE
# ============================================

def predict_spoof_score(session, audio, sr):
    processed = preprocess_audio(audio, sr)

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
            "Preprocessed audio contains invalid values"
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
            "Selected segment is too quiet"
        )

    batch = np.ascontiguousarray(
        processed[np.newaxis, :]
    )

    output = session.run(
        ["logits"],
        {"wav": batch},
    )

    logits = np.asarray(
        output[0],
        dtype=np.float64,
    ).reshape(-1)

    if (
        logits.shape != (2,)
        or not np.all(np.isfinite(logits))
    ):
        raise ValueError(
            "Invalid model output"
        )

    # Existing VaaniShield class convention:
    # 0 = spoof, 1 = genuine.
    # This must also be verified against
    # the original model documentation.
    shifted = logits - logits.max()

    values = np.exp(shifted)
    probabilities = values / values.sum()

    return float(probabilities[0])


# ============================================
# ANALYSE ONE RECORDING
# ============================================

def analyse_file(session, group, path):
    audio, sample_rate = sf.read(
        path,
        dtype="float32",
        always_2d=True,
    )

    quality = inspect_recording(
        audio,
        sample_rate,
    )

    segments = select_segments(
        audio,
        sample_rate,
    )

    scores = []

    for segment in segments:
        score = predict_spoof_score(
            session,
            segment["audio"],
            sample_rate,
        )

        scores.append(score)

    first_score = scores[0]

    # Only the first-window prediction is used
    # for the file-level exploratory classification.
    predicted_spoof = first_score >= 0.5

    expected_spoof = group == "SPOOF"

    correct = (
        predicted_spoof == expected_spoof
    )

    return {
        "group": group,
        "file": path.name,
        **quality,
        "windows": len(scores),
        "first_spoof_pct": first_score * 100,
        "min_spoof_pct": min(scores) * 100,
        "max_spoof_pct": max(scores) * 100,
        "spread_pp": (
            max(scores) - min(scores)
        ) * 100,
        "correct": correct,
    }


# ============================================
# GROUP SUMMARY
# ============================================

def print_group_summary(group, rows):
    print("\n" + "-" * 65)
    print("GROUP:", group)
    print("-" * 65)

    if not rows:
        print("No evaluated recordings.")
        return

    print("Recordings:", len(rows))

    print(
        "Sample rates:",
        sorted(set(
            row["sample_rate"]
            for row in rows
        ))
    )

    print(
        "Median duration:",
        f"{statistics.median(row['duration_sec'] for row in rows):.2f}s"
    )

    print(
        "Median RMS:",
        f"{statistics.median(row['rms'] for row in rows):.5f}"
    )

    print(
        "Median near-silence fraction:",
        f"{statistics.median(row['near_silence_pct'] for row in rows):.2f}%"
    )

    print(
        "Median clipping fraction:",
        f"{statistics.median(row['clipping_pct'] for row in rows):.2f}%"
    )

    print(
        "Median spoof score:",
        f"{statistics.median(row['first_spoof_pct'] for row in rows):.2f}%"
    )

    print(
        "Median within-file score spread:",
        f"{statistics.median(row['spread_pp'] for row in rows):.2f} percentage points"
    )

    errors = sum(
        not row["correct"]
        for row in rows
    )

    print("Misclassified recordings:", errors)

    print(
        "Observed file-level error rate:",
        f"{100 * errors / len(rows):.2f}%"
    )


# ============================================
# SAVE METADATA REPORT
# ============================================

def save_report(rows):
    if not rows:
        return

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = list(rows[0].keys())

    with REPORT_PATH.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)

    print(
        "\nMetadata report saved:",
        REPORT_PATH
    )

    print(
        "The report contains recording "
        "metadata and model scores, not audio."
    )


# ============================================
# MAIN
# ============================================

def main():
    print("=" * 65)
    print("VAANISHIELD - PHASE 10.2")
    print("RECORDING DOMAIN ANALYSIS")
    print("=" * 65)

    samples = build_test_set()

    print(
        "\nAudio files discovered:",
        len(samples)
    )

    if not samples:
        print(
            "No evaluation files found. "
            "Check data/evaluation/genuine "
            "and data/evaluation/spoof."
        )
        return

    session = get_model()

    print(
        "ONNX providers:",
        session.get_providers()
    )

    print(
        "Model input duration:",
        f"{MODEL_DURATION:.4f}s"
    )

    print(
        "\nNOTE: Scores are uncalibrated. "
        "The 50% cutoff is only the existing "
        "experimental classification rule."
    )

    results = []

    print("\nANALYSING RECORDINGS")
    print("=" * 65)

    for index, (group, path) in enumerate(
        samples,
        start=1,
    ):
        try:
            row = analyse_file(
                session,
                group,
                path,
            )

            results.append(row)

            status = (
                "OK"
                if row["correct"]
                else "MISCLASSIFIED"
            )

            print(
                f"[{index}/{len(samples)}] "
                f"{path.name} | "
                f"{group} | "
                f"first={row['first_spoof_pct']:.2f}% | "
                f"spread={row['spread_pp']:.2f}pp | "
                f"{status}"
            )

        except Exception as exc:
            print(
                f"[{index}/{len(samples)}] "
                f"SKIPPED {path.name}: {exc}"
            )

    print("\n" + "=" * 65)
    print("SOURCE-BY-SOURCE COMPARISON")
    print("=" * 65)

    for group in (
        "MIC_GENUINE",
        "OTHER_GENUINE",
        "SPOOF",
    ):
        group_rows = [
            row
            for row in results
            if row["group"] == group
        ]

        print_group_summary(
            group,
            group_rows,
        )

    print("\n" + "=" * 65)
    print("GENUINE MICROPHONE RECORDING DETAILS")
    print("=" * 65)

    microphone_rows = [
        row
        for row in results
        if row["group"] == "MIC_GENUINE"
    ]

    for row in microphone_rows:
        print(
            f"\n{row['file']}"
        )

        print(
            f"  Sample rate: {row['sample_rate']} Hz"
        )

        print(
            f"  RMS: {row['rms']:.5f}"
        )

        print(
            f"  Near silence: "
            f"{row['near_silence_pct']:.2f}%"
        )

        print(
            f"  First-window spoof: "
            f"{row['first_spoof_pct']:.2f}%"
        )

        print(
            f"  Minimum spoof: "
            f"{row['min_spoof_pct']:.2f}%"
        )

        print(
            f"  Maximum spoof: "
            f"{row['max_spoof_pct']:.2f}%"
        )

        print(
            f"  Score spread: "
            f"{row['spread_pp']:.2f}pp"
        )

    save_report(results)

    print("\n" + "=" * 65)
    print("NEXT INVESTIGATION")
    print("=" * 65)

    print(
        "1. Compare microphone and benchmark "
        "recording characteristics."
    )

    print(
        "2. Identify which genuine recordings "
        "show inconsistent window predictions."
    )

    print(
        "3. Collect independent, consented "
        "genuine voice samples for testing."
    )

    print(
        "4. Do not calibrate thresholds or "
        "claim banking accuracy using these "
        "exploratory samples."
    )


if __name__ == "__main__":
    main()
