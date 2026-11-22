
from pathlib import Path
import time

import numpy as np
import onnxruntime as ort

from audio_preprocessing import (
    load_audio,
    preprocess_audio,
    MODEL_SAMPLES,
)


ROOT = Path(__file__).resolve().parent.parent

MODELS = {
    "AASIST": ROOT / "ai_ml/models/aasist.onnx",
    "W2V2-AASIST": ROOT / "ai_ml/models/w2v2-aasist.onnx",
}

TEST_FILES = {
    "GENUINE VOICE": ROOT / "data/sample_audio/real_voice.wav",
    "SYNTHETIC VOICE": ROOT / "data/sample_audio/synthetic_voice.wav",
}


def softmax(logits):
    logits = np.asarray(logits, dtype=np.float64)
    logits = logits - np.max(logits)

    exp_values = np.exp(logits)
    return exp_values / exp_values.sum()


def prepare_test_audio():
    prepared = {}

    for label, path in TEST_FILES.items():
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing test recording: {path}"
            )

        audio, sample_rate = load_audio(path)
        waveform = preprocess_audio(audio, sample_rate)

        assert waveform.shape == (MODEL_SAMPLES,)

        prepared[label] = waveform[None, :]

    return prepared


def evaluate_model(name, model_path, recordings):
    print(f"\n{'=' * 48}", flush=True)
    print(f"LOADING MODEL: {name}", flush=True)
    print(f"{'=' * 48}", flush=True)

    if not model_path.is_file():
        raise FileNotFoundError(model_path)

    options = ort.SessionOptions()
    options.intra_op_num_threads = 4

    start = time.perf_counter()

    session = ort.InferenceSession(
        str(model_path),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )

    loading_time = time.perf_counter() - start

    print(
        f"Model loaded in {loading_time:.2f}s",
        flush=True
    )

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    results = {}

    for label, waveform in recordings.items():

        print(f"\nTesting: {label}", flush=True)

        start = time.perf_counter()

        outputs = session.run(
            [output_name],
            {input_name: waveform.astype(np.float32)}
        )

        inference_time = time.perf_counter() - start

        logits = np.asarray(outputs[0]).reshape(-1)

        if logits.shape != (2,):
            raise ValueError(
                f"Unexpected output shape: {logits.shape}"
            )

        probabilities = softmax(logits)

        spoof_score = float(probabilities[0])
        genuine_score = float(probabilities[1])

        prediction = (
            "SPOOF"
            if spoof_score > genuine_score
            else "GENUINE"
        )

        print(f"Spoof score:    {spoof_score:.4f}")
        print(f"Genuine score:  {genuine_score:.4f}")
        print(f"Raw genuine logit: {logits[1]:.4f}")
        print(f"Prediction:     {prediction}")
        print(f"Inference time: {inference_time:.2f}s")

        results[label] = {
            "spoof_score": spoof_score,
            "genuine_score": genuine_score,
            "prediction": prediction,
            "inference_time": inference_time,
        }

    return results


def main():
    print("VAANISHIELD - MODEL COMPARISON")
    print("Preparing audio recordings...", flush=True)

    recordings = prepare_test_audio()

    results = {}

    for name, model_path in MODELS.items():
        results[name] = evaluate_model(
            name, model_path, recordings
        )

    print("\n" + "=" * 55)
    print("FINAL MODEL COMPARISON")
    print("=" * 55)

    for model_name, model_results in results.items():
        print(f"\n{model_name}")

        for label, data in model_results.items():
            print(
                f"{label}: "
                f"Spoof={data['spoof_score']:.4f}, "
                f"Prediction={data['prediction']}"
            )

    print("\nComparison completed!")
    print(
        "Note: Scores are uncalibrated. "
        "Two recordings cannot establish model accuracy."
    )


if __name__ == "__main__":
    main()
