
from pathlib import Path
from io import BytesIO
import json

import soundfile as sf
from datasets import load_dataset, Audio


ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "data" / "evaluation"

DATASET = "SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA"
TARGET_PER_CLASS = 15
MAX_SCANNED = 2000


def main():
    print("VAANISHIELD - BENCHMARK DATA COLLECTION")
    print("Connecting to Hugging Face...")

    dataset = load_dataset(
        DATASET,
        split="test",
        streaming=True
    )

    # Access original audio bytes without TorchCodec
    dataset = dataset.cast_column(
        "audio",
        Audio(decode=False)
    )

    counts = {"genuine": 0, "spoof": 0}
    speakers = {"genuine": set(), "spoof": set()}

    # Dataset: 0=bonafide, 1=spoof
    for index, row in enumerate(dataset):
        if index >= MAX_SCANNED:
            break

        label = "genuine" if row["label"] == 0 else "spoof"

        if counts[label] >= TARGET_PER_CLASS:
            continue

        notes = json.loads(row["notes"])
        speaker = notes.get("speaker_id")

        if speaker and speaker in speakers[label]:
            continue

        audio_bytes = row["audio"].get("bytes")

        if not audio_bytes:
            raise RuntimeError(
                "Audio bytes unavailable. "
                "Check dataset streaming format."
            )

        audio, sr = sf.read(
            BytesIO(audio_bytes),
            dtype="float32"
        )

        folder = OUTPUT / label
        folder.mkdir(parents=True, exist_ok=True)

        filename = (
            f"asv2019_{label}_"
            f"{counts[label] + 1:02d}.wav"
        )

        path = folder / filename

        sf.write(str(path), audio, sr)

        counts[label] += 1

        if speaker:
            speakers[label].add(speaker)

        print(f"Saved {filename} ({sr} Hz)")

        if all(
            count >= TARGET_PER_CLASS
            for count in counts.values()
        ):
            break

    print("\nCOLLECTION COMPLETE")
    print(f"Genuine: {counts['genuine']}")
    print(f"Spoof: {counts['spoof']}")
    print("Keep the dataset's license and provenance.")


if __name__ == "__main__":
    main()
