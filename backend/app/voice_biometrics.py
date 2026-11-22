import io
import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np
import soundfile as sf
import librosa

logger = logging.getLogger("vaanishield.biometrics")

ROOT = Path(__file__).resolve().parents[2]
VOICEPRINT_DB_PATH = ROOT / "data" / "customer_voiceprints.json"
BIOMETRIC_MATCH_THRESHOLD = 0.80


@dataclass
class BiometricVerificationResult:
    customer_id: str
    customer_name: str
    enrolled: bool
    similarity_score: float  # 0.0 to 1.0
    threshold: float
    biometric_match: bool
    tri_factor_verdict: str  # AUTHENTIC_ACCOUNT_HOLDER, SYNTHETIC_DEEPFAKE_CLONE, HUMAN_IMPOSTER, UNENROLLED_CUSTOMER
    security_explanation: str


def extract_voice_embedding(audio_bytes: bytes) -> np.ndarray:
    """
    Extracts a 128-dimensional L2-normalized acoustic biometric embedding vector
    from voice audio using multi-feature spectral moments (MFCC, Delta, Contrast, Chroma).
    """
    y, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
    if len(y.shape) > 1:
        y = y.mean(axis=1)

    if sr != 16000:
        y = librosa.resample(y, orig_sr=sr, target_sr=16000)
        sr = 16000

    # 1. MFCC (20 coefficients)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)

    # 2. Spectral Contrast (7 bands)
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_bands=6)

    # 3. Chroma STFT (12 bins)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)

    # Statistical aggregation across time frames: mean and standard deviation
    features = []
    for mat in [mfcc, mfcc_delta, mfcc_delta2, contrast, chroma]:
        features.append(np.mean(mat, axis=1))
        features.append(np.std(mat, axis=1))

    raw_vec = np.concatenate(features)  # Length: 20*2 + 20*2 + 20*2 + 7*2 + 12*2 = 158

    # Truncate or project to exact 128 dimensions
    embedding = raw_vec[:128].astype(np.float32)

    # L2-normalization
    norm = np.linalg.norm(embedding)
    if norm > 1e-6:
        embedding = embedding / norm

    return embedding


def load_voiceprint_db() -> Dict[str, Any]:
    """Loads enrolled customer voiceprints from storage."""
    if not VOICEPRINT_DB_PATH.exists():
        # Initialize default demo profile
        return {}
    try:
        with open(VOICEPRINT_DB_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        logger.exception("Failed to read voiceprint database")
        return {}


def save_voiceprint_db(data: Dict[str, Any]) -> None:
    """Saves enrolled customer voiceprints to storage."""
    VOICEPRINT_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(VOICEPRINT_DB_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def ensure_demo_enrolled_customer():
    """Enrolls default demo customer CUST-10492 using real_voice.wav if database is empty."""
    db = load_voiceprint_db()
    if "CUST-10492" not in db:
        real_voice_sample = ROOT / "data" / "sample_audio" / "real_voice.wav"
        if real_voice_sample.exists():
            with open(real_voice_sample, "rb") as f:
                emb = extract_voice_embedding(f.read())
            db["CUST-10492"] = {
                "customer_id": "CUST-10492",
                "customer_name": "Sarah Jenkins",
                "account_number": "•••• 8921",
                "enrolled_at": datetime.now(timezone.utc).isoformat(),
                "embedding": emb.tolist(),
            }
            save_voiceprint_db(db)
            logger.info("Enrolled demo customer Sarah Jenkins (CUST-10492).")


def enroll_customer(
    customer_id: str,
    customer_name: str,
    audio_bytes: bytes,
    account_number: str = "•••• 0000",
) -> Dict[str, Any]:
    """Enrolls a customer voice biometric profile."""
    embedding = extract_voice_embedding(audio_bytes)
    db = load_voiceprint_db()

    profile = {
        "customer_id": customer_id,
        "customer_name": customer_name,
        "account_number": account_number,
        "enrolled_at": datetime.now(timezone.utc).isoformat(),
        "embedding": embedding.tolist(),
    }
    db[customer_id] = profile
    save_voiceprint_db(db)

    return {
        "success": True,
        "customer_id": customer_id,
        "customer_name": customer_name,
        "enrolled_at": profile["enrolled_at"],
        "embedding_dimensions": len(embedding),
        "message": f"Customer voiceprint successfully enrolled for {customer_name}.",
    }


def verify_speaker_identity(
    customer_id: str,
    caller_audio_bytes: bytes,
    audio_spoof_score: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Verifies caller voice against enrolled customer voiceprint and evaluates the Tri-Factor Verdict:
    1. AUTHENTIC_ACCOUNT_HOLDER: Natural organic voice matching enrolled profile.
    2. SYNTHETIC_DEEPFAKE_CLONE: Spoof characteristics detected (AI cloned customer).
    3. HUMAN_IMPOSTER: Organic human voice, but similarity doesn't match account holder!
    4. UNENROLLED_CUSTOMER: No enrolled voiceprint found.
    """
    ensure_demo_enrolled_customer()
    db = load_voiceprint_db()

    if customer_id not in db:
        return asdict(
            BiometricVerificationResult(
                customer_id=customer_id,
                customer_name="Unknown / Unenrolled",
                enrolled=False,
                similarity_score=0.0,
                threshold=BIOMETRIC_MATCH_THRESHOLD,
                biometric_match=False,
                tri_factor_verdict="UNENROLLED_CUSTOMER",
                security_explanation=(
                    f"Customer {customer_id} has no enrolled voice biometric profile. "
                    "Biometric identity match unavailable."
                ),
            )
        )

    profile = db[customer_id]
    enrolled_emb = np.array(profile["embedding"], dtype=np.float32)

    caller_emb = extract_voice_embedding(caller_audio_bytes)

    # Cosine similarity between normalized vectors
    similarity = float(np.dot(enrolled_emb, caller_emb))
    similarity = max(0.0, min(1.0, (similarity + 1.0) / 2.0))  # Scale [-1, 1] to [0.0, 1.0]

    biometric_match = similarity >= BIOMETRIC_MATCH_THRESHOLD
    spoof_detected = audio_spoof_score is not None and audio_spoof_score >= 0.70

    if spoof_detected:
        verdict = "SYNTHETIC_DEEPFAKE_CLONE"
        explanation = (
            f"ALERT: Acoustic deepfake detected. Attacker is using an AI synthetic voice clone "
            f"targeting {profile['customer_name']}'s account."
        )
    elif biometric_match:
        verdict = "AUTHENTIC_ACCOUNT_HOLDER"
        explanation = (
            f"VERIFIED: Caller acoustic voiceprint matches authorized account holder "
            f"{profile['customer_name']} ({int(similarity * 100)}% biometric confidence)."
        )
    else:
        verdict = "HUMAN_IMPOSTER"
        explanation = (
            f"FRAUD WARNING: Caller voice does NOT match authorized customer "
            f"{profile['customer_name']} (Biometric similarity: {int(similarity * 100)}% vs "
            f"{int(BIOMETRIC_MATCH_THRESHOLD * 100)}% threshold). Caller is a human imposter."
        )

    return asdict(
        BiometricVerificationResult(
            customer_id=customer_id,
            customer_name=profile["customer_name"],
            enrolled=True,
            similarity_score=round(similarity, 4),
            threshold=BIOMETRIC_MATCH_THRESHOLD,
            biometric_match=biometric_match,
            tri_factor_verdict=verdict,
            security_explanation=explanation,
        )
    )


if __name__ == "__main__":
    from pprint import pprint

    ensure_demo_enrolled_customer()
    print("Enrolled customers:", list(load_voiceprint_db().keys()))

    # Test verification against enrolled Sarah Jenkins
    real_sample = ROOT / "data" / "sample_audio" / "real_voice.wav"
    synthetic_sample = ROOT / "data" / "sample_audio" / "synthetic_voice.wav"

    print("\n--- Test 1: Real Customer calling ---")
    with open(real_sample, "rb") as f:
        res1 = verify_speaker_identity("CUST-10492", f.read(), audio_spoof_score=0.08)
        pprint(res1)

    print("\n--- Test 2: AI Voice Clone calling ---")
    with open(synthetic_sample, "rb") as f:
        res2 = verify_speaker_identity("CUST-10492", f.read(), audio_spoof_score=0.98)
        pprint(res2)
