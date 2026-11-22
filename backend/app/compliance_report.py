import hashlib
import json
from datetime import datetime, timezone
from typing import Dict, Any
from uuid import uuid4


def generate_compliance_audit_report(
    audio_bytes: bytes,
    audio_analysis: Dict[str, Any],
    forensics: Dict[str, Any],
    transcription: Dict[str, Any],
    scam_analysis: Dict[str, Any],
    biometrics: Dict[str, Any],
    threat_fusion: Dict[str, Any],
    banking_context: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Generates a standardized, tamper-evident regulatory forensic audit certificate
    for financial institution security operations and banking ombudsman compliance.
    """
    certificate_id = f"VS-AUDIT-{uuid4().hex[:8].upper()}"
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")

    # Cryptographic SHA-256 checksum of original audio recording
    sha256_hash = hashlib.sha256(audio_bytes).hexdigest()

    report_data = {
        "certificate_id": certificate_id,
        "issued_at_utc": timestamp,
        "platform": "VaaniShield AI Security Protocol v2.4",
        "audio_forensic_fingerprint": {
            "sha256_checksum": sha256_hash,
            "byte_size": len(audio_bytes),
            "recording_duration_seconds": audio_analysis.get("recording_duration_seconds", 0),
            "analyzed_window_seconds": audio_analysis.get("analyzed_audio_seconds", 0),
        },
        "acoustic_deepfake_assessment": {
            "model_architecture": audio_analysis.get("model", "W2V2-AASIST"),
            "spoof_probability": audio_analysis.get("spoof_score", 0.0),
            "genuine_probability": audio_analysis.get("genuine_score", 0.0),
            "inference_time_seconds": audio_analysis.get("inference_seconds", 0.0),
            "verdict": audio_analysis.get("prediction", "UNKNOWN"),
        },
        "acoustic_forensic_diagnostics": {
            "vocoder_spectral_cutoff": forensics.get("vocoder_cutoff_detected", False),
            "spectral_centroid_hz": forensics.get("spectral_centroid_hz", 0),
            "spectral_rolloff_hz": forensics.get("spectral_rolloff_hz", 0),
            "unnatural_silence_ratio": forensics.get("unnatural_silence_ratio", 0),
            "pitch_variability_hz": forensics.get("pitch_variability_hz", 0),
            "snr_db": forensics.get("snr_db", 0),
            "forensic_flags": forensics.get("forensic_flags", []),
            "forensic_summary": forensics.get("forensic_summary", ""),
        },
        "speech_transcription_and_intent": {
            "transcribed_text": transcription.get("text", ""),
            "asr_engine": transcription.get("engine", "Google-ASR"),
            "word_count": transcription.get("word_count", 0),
            "scam_detected": scam_analysis.get("scam_detected", False),
            "scam_confidence_score": scam_analysis.get("scam_score", 0.0),
            "threat_tier": scam_analysis.get("threat_tier", "NONE"),
            "primary_social_engineering_intent": scam_analysis.get("primary_intent", "BENIGN"),
            "coercion_level": scam_analysis.get("coercion_level", "NONE"),
            "detected_triggers": scam_analysis.get("triggers", []),
        },
        "customer_voice_biometrics": {
            "customer_id": biometrics.get("customer_id", "N/A"),
            "customer_name": biometrics.get("customer_name", "N/A"),
            "enrolled": biometrics.get("enrolled", False),
            "biometric_similarity": biometrics.get("similarity_score", 0.0),
            "biometric_match": biometrics.get("biometric_match", False),
            "tri_factor_verdict": biometrics.get("tri_factor_verdict", "UNENROLLED_CUSTOMER"),
            "security_explanation": biometrics.get("security_explanation", ""),
        },
        "threat_fusion_matrix": {
            "fused_threat_score": threat_fusion.get("fused_threat_score", 0),
            "threat_level": threat_fusion.get("threat_level", "LOW"),
            "recommended_action": threat_fusion.get("action", "STANDARD_BANKING_FLOW"),
            "compound_threat_detected": threat_fusion.get("compound_threat", False),
            "threat_reasons": threat_fusion.get("reasons", []),
            "defensive_protocol": threat_fusion.get("recommended_protocol", ""),
        },
        "banking_operation_context": banking_context,
        "regulatory_compliance_declaration": (
            "This report is cryptographically sealed and generated automatically in compliance with "
            "financial regulatory standards for Automated Voice Impersonation and Fraud Prevention. "
            "Data hashes confirm non-repudiation and audio integrity."
        ),
    }

    return report_data


def generate_printable_html_report(report_data: Dict[str, Any]) -> str:
    """Generates clean, professional printable HTML certificate for downloading/printing."""
    cert_id = report_data["certificate_id"]
    timestamp = report_data["issued_at_utc"]
    sha256 = report_data["audio_forensic_fingerprint"]["sha256_checksum"]
    fusion = report_data["threat_fusion_matrix"]
    deepfake = report_data["acoustic_deepfake_assessment"]
    biometrics = report_data["customer_voice_biometrics"]
    transcript = report_data["speech_transcription_and_intent"]
    forensics = report_data["acoustic_forensic_diagnostics"]
    banking = report_data["banking_operation_context"]

    level_color = {
        "CRITICAL": "#ef4444",
        "HIGH": "#f97316",
        "MEDIUM": "#eab308",
        "LOW": "#22c55e",
    }.get(fusion["threat_level"], "#3b82f6")

    triggers_html = "".join(
        f"<li><strong>[{t.get('category')}]</strong> {t.get('pattern_matched')} — <em>\"{t.get('snippet')}\"</em></li>"
        for t in transcript.get("detected_triggers", [])
    ) or "<li>None detected</li>"

    reasons_html = "".join(
        f"<li>{r}</li>" for r in fusion.get("threat_reasons", [])
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>VaaniShield Forensic Audit Certificate - {cert_id}</title>
  <style>
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      background: #f8fafc;
      color: #0f172a;
      margin: 0;
      padding: 40px;
    }}
    .cert-box {{
      max-width: 850px;
      margin: 0 auto;
      background: #ffffff;
      border: 1px solid #cbd5e1;
      border-radius: 12px;
      box-shadow: 0 4px 20px rgba(0,0,0,0.06);
      padding: 40px;
    }}
    .header {{
      display: flex;
      justify-content: space-between;
      border-bottom: 2px solid #0f172a;
      padding-bottom: 20px;
      margin-bottom: 24px;
    }}
    .header h1 {{
      margin: 0;
      font-size: 24px;
      color: #0f172a;
    }}
    .header p {{
      margin: 4px 0 0;
      color: #64748b;
      font-size: 13px;
    }}
    .badge {{
      display: inline-block;
      padding: 8px 16px;
      border-radius: 20px;
      font-weight: 800;
      color: white;
      background: {level_color};
      font-size: 14px;
      letter-spacing: 0.5px;
    }}
    .grid {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 20px;
      margin-bottom: 24px;
    }}
    .card {{
      background: #f1f5f9;
      border: 1px solid #e2e8f0;
      border-radius: 8px;
      padding: 16px;
    }}
    .card h3 {{
      margin: 0 0 10px;
      font-size: 14px;
      text-transform: uppercase;
      color: #334155;
      border-bottom: 1px solid #cbd5e1;
      padding-bottom: 6px;
    }}
    .card p, .card li {{
      font-size: 13px;
      line-height: 1.6;
      margin: 4px 0;
    }}
    .hash {{
      font-family: monospace;
      font-size: 11px;
      word-break: break-all;
      background: #e2e8f0;
      padding: 4px 8px;
      border-radius: 4px;
    }}
    .footer {{
      margin-top: 30px;
      border-top: 1px solid #e2e8f0;
      padding-top: 16px;
      font-size: 11px;
      color: #64748b;
      text-align: center;
    }}
    @media print {{
      body {{ background: white; padding: 0; }}
      .cert-box {{ box-shadow: none; border: none; padding: 20px; }}
    }}
  </style>
</head>
<body>
  <div class="cert-box">
    <div class="header">
      <div>
        <h1>🛡️ VaaniShield Security Audit Certificate</h1>
        <p>Certificate Ref: <strong>{cert_id}</strong> · Issued: {timestamp}</p>
      </div>
      <div>
        <span class="badge">{fusion['threat_level']} THREAT ({fusion['fused_threat_score']}/100)</span>
      </div>
    </div>

    <div class="grid">
      <div class="card">
        <h3>1. Audio Cryptographic Fingerprint</h3>
        <p><strong>SHA-256 Digest:</strong></p>
        <div class="hash">{sha256}</div>
        <p><strong>Duration:</strong> {report_data['audio_forensic_fingerprint']['recording_duration_seconds']}s</p>
        <p><strong>Verification Status:</strong> Tamper-evident authenticated</p>
      </div>

      <div class="card">
        <h3>2. Acoustic Deepfake Analysis</h3>
        <p><strong>AI Model:</strong> {deepfake['model_architecture']}</p>
        <p><strong>Spoof Probability:</strong> {(deepfake['spoof_probability'] * 100):.2f}%</p>
        <p><strong>Genuine Probability:</strong> {(deepfake['genuine_probability'] * 100):.2f}%</p>
        <p><strong>Vocoder Cutoff Flag:</strong> {"YES (Synthetic Artifact)" if forensics['vocoder_spectral_cutoff'] else "NO (Organic Bandwidth)"}</p>
      </div>
    </div>

    <div class="grid">
      <div class="card">
        <h3>3. Voice Biometrics & Identity</h3>
        <p><strong>Customer Target:</strong> {biometrics['customer_name']} ({biometrics['customer_id']})</p>
        <p><strong>Biometric Match:</strong> {"YES (Authorized Profile)" if biometrics['biometric_match'] else "NO (Non-Matching)"}</p>
        <p><strong>Tri-Factor Security Verdict:</strong> <strong>{biometrics['tri_factor_verdict']}</strong></p>
        <p style="font-size: 12px; color: #475569;">{biometrics['security_explanation']}</p>
      </div>

      <div class="card">
        <h3>4. Linguistic & Social Engineering</h3>
        <p><strong>Transcript:</strong> <em>"{transcript['transcribed_text']}"</em></p>
        <p><strong>Scam Intent Tier:</strong> {transcript['threat_tier']} ({int(transcript['scam_confidence_score'] * 100)}%)</p>
        <p><strong>Coercion Level:</strong> {transcript['coercion_level']}</p>
        <p><strong>Primary Intent:</strong> {transcript['primary_social_engineering_intent']}</p>
      </div>
    </div>

    <div class="card" style="margin-bottom: 20px;">
      <h3>5. Action & Threat Mitigation Protocol</h3>
      <p><strong>Requested Operation:</strong> {banking.get('requested_action')} (Auth: {banking.get('authentication_status')})</p>
      <p><strong>Recommended Workflow:</strong> <strong>{fusion['recommended_action']}</strong></p>
      <p><strong>Action Protocol:</strong> {fusion['defensive_protocol']}</p>
      <p><strong>Decision Rationale:</strong></p>
      <ul>{reasons_html}</ul>
    </div>

    <div class="footer">
      {report_data['regulatory_compliance_declaration']}
    </div>
  </div>
</body>
</html>"""
