import io
import logging
from dataclasses import dataclass, asdict
from typing import Dict, Any, List

import numpy as np
import soundfile as sf
import librosa

logger = logging.getLogger("vaanishield.forensics")


@dataclass
class ForensicReport:
    spectral_centroid_hz: float
    spectral_rolloff_hz: float
    vocoder_cutoff_detected: bool
    unnatural_silence_ratio: float
    pitch_variability_hz: float
    snr_db: float
    clipping_detected: bool
    forensic_anomaly_score: int  # 0 - 100
    forensic_flags: List[str]
    forensic_summary: str


def analyze_audio_forensics(audio_bytes: bytes) -> Dict[str, Any]:
    """
    Extracts acoustic forensic diagnostics to explain synthetic audio anomalies:
    - High-frequency vocoder cutoff and unnatural bandwidth cliff
    - Unnatural silence / digital zero gaps
    - Pitch stability / unnatural lack of micro-tremors (shimmer/jitter)
    - Clipping & Signal-to-Noise Ratio (SNR)
    """
    try:
        y, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
        if len(y.shape) > 1:
            y = y.mean(axis=1)

        # Ensure minimal length
        if len(y) < sr * 0.5:
            return asdict(
                ForensicReport(
                    spectral_centroid_hz=0.0,
                    spectral_rolloff_hz=0.0,
                    vocoder_cutoff_detected=False,
                    unnatural_silence_ratio=0.0,
                    pitch_variability_hz=0.0,
                    snr_db=0.0,
                    clipping_detected=False,
                    forensic_anomaly_score=0,
                    forensic_flags=["INSUFFICIENT_DURATION"],
                    forensic_summary="Audio sample too short for spectral diagnostics.",
                )
            )

        flags: List[str] = []
        anomaly_points = 0

        # 1. Clipping Detection
        clipped = bool(np.sum(np.abs(y) >= 0.99) > (len(y) * 0.001))
        if clipped:
            flags.append("AUDIO_CLIPPING_DETECTED")
            anomaly_points += 15

        # 2. Resample to 16kHz for standardized spectral analysis if necessary
        if sr != 16000:
            y_16k = librosa.resample(y, orig_sr=sr, target_sr=16000)
            sample_rate = 16000
        else:
            y_16k = y
            sample_rate = sr

        rolloff = librosa.feature.spectral_rolloff(y=y_16k, sr=sample_rate, roll_percent=0.85)
        mean_rolloff = float(np.mean(rolloff))

        centroid = librosa.feature.spectral_centroid(y=y_16k, sr=sample_rate)
        mean_centroid = float(np.mean(centroid))

        # Check for telephony narrowband vs vocoder artifact
        vocoder_cutoff = False
        if mean_rolloff < 2400 and mean_centroid < 1400:
            vocoder_cutoff = True
            flags.append("VOCODER_SPECTRAL_BANDWIDTH_LIMIT")
            anomaly_points += 25
        elif mean_rolloff > 7600:
            flags.append("HIGH_FREQUENCY_SYNTHETIC_HISS")
            anomaly_points += 15

        # 3. Synthetic Silence & Energy Dynamics
        hop_length = 512
        rms = librosa.feature.rms(y=y_16k, hop_length=hop_length)[0]
        max_rms = np.max(rms) if len(rms) > 0 and np.max(rms) > 0 else 1.0

        # Silent frames (less than 1.5% of peak energy)
        silent_frames = rms < (0.015 * max_rms)
        silence_ratio = float(np.mean(silent_frames))

        # Check for unnaturally high silence or hard digital zeros
        if silence_ratio > 0.22:
            flags.append("UNNATURAL_INTER_WORD_SILENCE_RATIO")
            anomaly_points += 20

        if np.any(silent_frames):
            silent_energies = rms[silent_frames]
            if np.all(silent_energies < 1e-5):
                flags.append("ARTIFICIAL_DIGITAL_ZERO_SILENCE")
                anomaly_points += 25

        # 4. Pitch Variability & Natural Micro-Tremors
        try:
            pitches, magnitudes = librosa.piptrack(y=y_16k, sr=sample_rate, hop_length=hop_length)
            pitch_track = []
            for t in range(pitches.shape[1]):
                index = magnitudes[:, t].argmax()
                pitch = pitches[index, t]
                if pitch > 50 and pitch < 500:  # Valid human vocal range
                    pitch_track.append(pitch)

            if len(pitch_track) > 10:
                pitch_std = float(np.std(pitch_track))
                if pitch_std < 10.0:
                    flags.append("ROBOTIC_MONOTONE_PITCH_ANOMALY")
                    anomaly_points += 25
            else:
                pitch_std = 0.0
        except Exception:
            pitch_std = 0.0

        # 5. Signal-to-Noise Ratio (SNR) estimation
        noise_floor = np.median(rms) if len(rms) > 0 else 1e-4
        snr = float(20 * np.log10((max_rms + 1e-6) / (noise_floor + 1e-6)))

        forensic_score = min(100, anomaly_points)

        # Construct explanation summary
        if len(flags) == 0:
            summary = "Acoustic characteristics display natural room harmonics, normal pitch variability, and organic spectral bandwidth."
        else:
            summary = f"Detected {len(flags)} forensic anomalies: {', '.join(flags)}."

        return asdict(
            ForensicReport(
                spectral_centroid_hz=round(mean_centroid, 2),
                spectral_rolloff_hz=round(mean_rolloff, 2),
                vocoder_cutoff_detected=vocoder_cutoff,
                unnatural_silence_ratio=round(silence_ratio, 4),
                pitch_variability_hz=round(pitch_std, 2),
                snr_db=round(snr, 2),
                clipping_detected=clipped,
                forensic_anomaly_score=forensic_score,
                forensic_flags=flags,
                forensic_summary=summary,
            )
        )

    except Exception as e:
        logger.exception("Forensic analysis failed")
        return asdict(
            ForensicReport(
                spectral_centroid_hz=0.0,
                spectral_rolloff_hz=0.0,
                vocoder_cutoff_detected=False,
                unnatural_silence_ratio=0.0,
                pitch_variability_hz=0.0,
                snr_db=0.0,
                clipping_detected=False,
                forensic_anomaly_score=0,
                forensic_flags=["ANALYSIS_ERROR"],
                forensic_summary=f"Forensic engine could not parse audio: {str(e)}",
            )
        )
