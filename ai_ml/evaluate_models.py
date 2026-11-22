
from pathlib import Path
import time

import numpy as np
import onnxruntime as ort
from sklearn.metrics import roc_auc_score

from audio_preprocessing import (
    load_audio,
    preprocess_audio,
)


ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT / "data" / "evaluation"

MODELS = {
    "AASIST": ROOT / "ai_ml/models/aasist.onnx",
    "W2V2-AASIST": ROOT / "ai_ml/models/w2v2-aasist.onnx",
}


def collect_samples():
    samples = []

    for folder, label in [
        ("genuine", 1),
        ("spoof", 0),
    ]:
        directory = DATA_DIR / folder

        if not directory.exists():
            raise FileNotFoundError(directory)

        for path in sorted(directory.iterdir()):
            if path.suffix.lower() in (".wav", ".flac"):
                samples.append((path, label))

    return samples


def evaluate_model(name, model_path, samples):
    print(f"\n{'=' * 50}")
    print(f"MODEL: {name}")
    print("=" * 50)

    session = ort.InferenceSession(
        str(model_path),
        providers=["CPUExecutionProvider"],
    )

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    actual_labels = []
    predicted_labels = []
    genuine_logits = []
    inference_times = []

    for path, label in samples:
        audio, sample_rate = load_audio(path)
        processed = preprocess_audio(audio, sample_rate)

        batch = processed[np.newaxis, :]

        start = time.perf_counter()

        output = session.run(
            [output_name],
            {input_name: batch}
        )

        elapsed = time.perf_counter() - start

        logits = np.asarray(
            output[0][0], dtype=np.float64
        )

        if logits.shape != (2,):
            raise ValueError("Unexpected model output")

        shifted = logits - logits.max()
        probabilities = np.exp(shifted)
        probabilities /= probabilities.sum()

        # Index 1 represents genuine speech
        prediction = int(probabilities[1] > 0.5)

        actual_labels.append(label)
        predicted_labels.append(prediction)
        genuine_logits.append(float(logits[1]))
        inference_times.append(elapsed)

        actual = "GENUINE" if label == 1 else "SPOOF"
        predicted = (
            "GENUINE" if prediction == 1 else "SPOOF"
        )

        print(
            f"{path.name:25s} "
            f"Actual={actual:8s} "
            f"Predicted={predicted:8s} "
            f"SpoofScore={probabilities[0]:.4f}"
        )

    actual = np.asarray(actual_labels)
    predicted = np.asarray(predicted_labels)

    false_positives = int(
        np.sum((actual == 1) & (predicted == 0))
    )

    false_negatives = int(
        np.sum((actual == 0) & (predicted == 1))
    )

    print("\nSUMMARY")
    print(f"Total samples: {len(samples)}")
    print(f"Genuine samples: {np.sum(actual == 1)}")
    print(f"Spoof samples: {np.sum(actual == 0)}")
    print(f"Genuine incorrectly flagged: {false_positives}")
    print(f"Spoofs incorrectly accepted: {false_negatives}")

    print(
        f"Average inference time: "
        f"{np.mean(inference_times):.3f}s"
    )

    if len(np.unique(actual)) == 2:
        auc = roc_auc_score(
            actual,
            genuine_logits
        )
        print(f"ROC-AUC (exploratory): {auc:.4f}")

    if min(
        np.sum(actual == 1),
        np.sum(actual == 0)
    ) < 10:
        print(
            "\nWARNING: Too few independent samples "
            "for reliable model evaluation."
        )


def main():
    samples = collect_samples()

    if not samples:
        print("No evaluation audio files found.")
        return

    print("VAANISHIELD - MODEL EVALUATION")
    print(f"Found {len(samples)} recordings")

    for name, model_path in MODELS.items():
        evaluate_model(name, model_path, samples)

    print("\nEvaluation completed.")
    print(
        "These model scores and default decisions "
        "are not validated for banking use."
    )


if __name__ == "__main__":
    main()
