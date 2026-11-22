
import statistics
import sys
import time

from pathlib import Path

import numpy as np
import soundfile as sf

# Load CUDA runtime libraries before ONNX Runtime.
import torch
import onnxruntime as ort


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ai_ml.audio_preprocessing import (
    preprocess_audio,
    MODEL_SAMPLES,
)

MODEL_PATH = (
    ROOT / "ai_ml" / "models" / "w2v2-aasist.onnx"
)

SAMPLE_DIR = ROOT / "data" / "sample_audio"
GENUINE_DIR = ROOT / "data" / "evaluation" / "genuine"
SPOOF_DIR = ROOT / "data" / "evaluation" / "spoof"

REPEATS = 5


def create_session(providers):
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4

    return ort.InferenceSession(
        str(MODEL_PATH),
        sess_options=options,
        providers=providers,
    )


def prepare_audio(path):
    audio, sr = sf.read(
        path,
        dtype="float32",
        always_2d=True,
    )

    processed = preprocess_audio(audio, sr)

    processed = np.asarray(
        processed,
        dtype=np.float32,
    )

    if processed.shape != (MODEL_SAMPLES,):
        raise ValueError("Invalid audio shape")

    if not np.all(np.isfinite(processed)):
        raise ValueError("Invalid audio values")

    return np.ascontiguousarray(
        processed[np.newaxis, :]
    )


def run_repeated(session, batch):
    results = []
    timings = []

    # Warm up the execution provider.
    session.run(["logits"], {"wav": batch})

    for _ in range(REPEATS):
        start = time.perf_counter()

        output = session.run(
            ["logits"],
            {"wav": batch},
        )

        elapsed = time.perf_counter() - start

        logits = np.asarray(
            output[0][0],
            dtype=np.float64,
        )

        shifted = logits - logits.max()
        values = np.exp(shifted)
        probabilities = values / values.sum()

        results.append({
            "logits": logits,
            "spoof": float(probabilities[0]),
            "genuine": float(probabilities[1]),
        })

        timings.append(elapsed)

    return {
        "results": results,
        "median_ms": (
            statistics.median(timings) * 1000
        ),
    }


def find_audio_files(folder):
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


def get_test_files():
    files = []

    own_voice = SAMPLE_DIR / "real_voice.wav"

    if own_voice.exists():
        files.append(
            ("Own genuine voice", own_voice)
        )

    genuine = find_audio_files(GENUINE_DIR)

    benchmark = next(
        (
            path for path in genuine
            if not path.name.lower().startswith("real_")
        ),
        None,
    )

    if benchmark:
        files.append(
            ("Benchmark genuine", benchmark)
        )

    spoof = find_audio_files(SPOOF_DIR)

    if spoof:
        files.append(
            ("Spoof sample", spoof[0])
        )

    return files


def compare_file(label, path, cpu, gpu):
    print("\n" + "=" * 60)
    print(label)
    print("File:", path.name)
    print("=" * 60)

    batch = prepare_audio(path)

    cpu_data = run_repeated(cpu, batch)
    gpu_data = run_repeated(gpu, batch)

    cpu_results = cpu_data["results"]
    gpu_results = gpu_data["results"]

    cpu_scores = [
        item["spoof"]
        for item in cpu_results
    ]

    gpu_scores = [
        item["spoof"]
        for item in gpu_results
    ]

    cpu_avg = statistics.mean(cpu_scores)
    gpu_avg = statistics.mean(gpu_scores)

    score_difference = abs(
        cpu_avg - gpu_avg
    ) * 100

    cpu_range = (
        max(cpu_scores) - min(cpu_scores)
    ) * 100

    gpu_range = (
        max(gpu_scores) - min(gpu_scores)
    ) * 100

    logit_difference = max(
        float(
            np.max(
                np.abs(
                    a["logits"] - b["logits"]
                )
            )
        )
        for a, b in zip(
            cpu_results,
            gpu_results,
        )
    )

    print(
        f"CPU spoof score: {cpu_avg * 100:.6f}%"
    )

    print(
        f"GPU spoof score: {gpu_avg * 100:.6f}%"
    )

    print(
        "CPU/GPU score difference:",
        f"{score_difference:.6f} percentage points"
    )

    print(
        "Maximum raw-logit difference:",
        f"{logit_difference:.8f}"
    )

    print(
        "CPU repeated-run variation:",
        f"{cpu_range:.6f} percentage points"
    )

    print(
        "GPU repeated-run variation:",
        f"{gpu_range:.6f} percentage points"
    )

    print(
        "CPU median inference:",
        f"{cpu_data['median_ms']:.2f} ms"
    )

    print(
        "GPU median inference:",
        f"{gpu_data['median_ms']:.2f} ms"
    )

    if score_difference > 1:
        print(
            "CHECK: CPU and GPU spoof scores "
            "differ by more than 1 percentage point."
        )
    else:
        print(
            "CHECK: CPU and GPU spoof scores "
            "are close for this audio sample."
        )

    if gpu_range > 0.1:
        print(
            "CHECK: GPU output varies across "
            "identical repeated inputs."
        )


def main():
    print("=" * 60)
    print("VAANISHIELD - CPU VS CUDA CONSISTENCY TEST")
    print("=" * 60)

    print(
        "PyTorch CUDA available:",
        torch.cuda.is_available()
    )

    print(
        "ONNX available providers:",
        ort.get_available_providers()
    )

    if "CUDAExecutionProvider" not in (
        ort.get_available_providers()
    ):
        print("CUDA provider is unavailable.")
        return

    print("\nLoading CPU model...")
    cpu = create_session(
        ["CPUExecutionProvider"]
    )

    print("Loading CUDA model...")
    gpu = create_session([
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ])

    print(
        "CPU session providers:",
        cpu.get_providers()
    )

    print(
        "GPU session providers:",
        gpu.get_providers()
    )

    if "CUDAExecutionProvider" not in gpu.get_providers():
        print(
            "CUDA session was not activated. "
            "Comparison cancelled."
        )
        return

    files = get_test_files()

    if not files:
        print("No audio test files found.")
        return

    for label, path in files:
        try:
            compare_file(
                label,
                path,
                cpu,
                gpu,
            )
        except Exception as exc:
            print(
                f"Could not test {path.name}: {exc}"
            )

    print("\nTesting completed.")
    print(
        "Provider availability does not prove "
        "that every ONNX operation ran on GPU."
    )
    print(
        "These scores are uncalibrated "
        "research outputs."
    )


if __name__ == "__main__":
    main()
