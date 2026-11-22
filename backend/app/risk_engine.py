from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional


# Illustrative banking policy values.
# These represent banking operational sensitivity.
ACTION_RISK = {
    "balance_inquiry": (10, False),
    "transaction_status": (15, False),
    "password_reset": (55, True),
    "mobile_number_change": (65, True),
    "transfer_limit_increase": (70, True),
    "high_value_transfer": (75, True),
}

AUTH_RISK = {
    "verified": 0,
    "unverified": 10,
    "failed": 25,
}


@dataclass
class RiskDecision:
    risk_score: int
    risk_level: str
    action: str
    reasons: list
    audio_policy: str


@dataclass
class ThreatFusionDecision:
    fused_threat_score: int  # 0 to 100
    threat_level: str  # LOW, MEDIUM, HIGH, CRITICAL
    action: str  # STANDARD_BANKING_FLOW, STEP_UP_VERIFICATION, HUMAN_REVIEW_REQUIRED, CRITICAL_FRAUD_INTERCEPT
    acoustic_risk: int
    linguistic_risk: int
    transaction_risk: int
    auth_risk: int
    compound_threat: bool
    reasons: List[str]
    recommended_protocol: str


def assess_risk(
    banking_action: str,
    authentication_status: str,
    unusual_activity: bool = False,
    audio_spoof_score: float | None = None,
):
    """Legacy baseline policy assessment (Monitor-only audio)."""
    if banking_action not in ACTION_RISK:
        raise ValueError("Unsupported banking action")

    if authentication_status not in AUTH_RISK:
        raise ValueError("Invalid authentication status")

    if audio_spoof_score is not None:
        if not 0 <= audio_spoof_score <= 1:
            raise ValueError("Invalid audio score")

    base_risk, sensitive = ACTION_RISK[banking_action]
    risk = base_risk + AUTH_RISK[authentication_status]

    reasons = [f"Requested action: {banking_action}"]

    if sensitive:
        reasons.append("Sensitive banking action")

    if authentication_status == "failed":
        reasons.append("Authentication failure")
    elif authentication_status == "unverified":
        reasons.append("Authentication not verified")

    if unusual_activity:
        risk += 20
        reasons.append("Unusual activity reported")

    risk = min(100, risk)

    if risk >= 90 and authentication_status == "failed":
        level = "HIGH"
        decision = "HUMAN_REVIEW_REQUIRED"
    elif sensitive or authentication_status == "failed":
        level = "MEDIUM" if risk < 90 else "HIGH"
        decision = "STEP_UP_VERIFICATION"
    else:
        level = "LOW" if risk < 40 else "MEDIUM"
        decision = "STANDARD_BANKING_FLOW"

    return asdict(
        RiskDecision(
            risk_score=risk,
            risk_level=level,
            action=decision,
            reasons=reasons,
            audio_policy="MONITOR_ONLY",
        )
    )


def assess_fused_threat(
    banking_action: str,
    authentication_status: str,
    unusual_activity: bool = False,
    audio_spoof_score: Optional[float] = None,
    scam_analysis: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Multi-Modal Threat Fusion Engine:
    Fuses Acoustic Deepfake Score (AASIST), Linguistic Scam Intent (STT + NLP),
    Banking Action Sensitivity, and Authentication Behavior into a unified security verdict.
    """
    if banking_action not in ACTION_RISK:
        raise ValueError("Unsupported banking action")
    if authentication_status not in AUTH_RISK:
        raise ValueError("Invalid authentication status")

    base_action_risk, sensitive = ACTION_RISK[banking_action]
    reasons: List[str] = []

    # 1. Acoustic Deepfake Factor (0 - 100)
    acoustic_score = 0
    if audio_spoof_score is not None:
        acoustic_score = int(round(audio_spoof_score * 100))
        if audio_spoof_score >= 0.70:
            reasons.append(f"Acoustic AI: High synthetic/spoof probability detected ({acoustic_score}%)")
        elif audio_spoof_score >= 0.40:
            reasons.append(f"Acoustic AI: Ambiguous voice characteristics ({acoustic_score}% spoof confidence)")
        else:
            reasons.append(f"Acoustic AI: Voice biometric signal appears organic ({acoustic_score}% spoof probability)")

    # 2. Linguistic Scam & Social Engineering Factor (0 - 100)
    linguistic_score = 0
    scam_tier = "NONE"
    if scam_analysis:
        scam_raw = scam_analysis.get("scam_score", 0.0)
        linguistic_score = int(round(scam_raw * 100))
        scam_tier = scam_analysis.get("threat_tier", "NONE")
        triggers = scam_analysis.get("triggers", [])

        if scam_analysis.get("scam_detected", False):
            primary = scam_analysis.get("primary_intent", "SUSPICIOUS_CONVERSATION")
            reasons.append(
                f"Linguistic AI: Flagged {len(triggers)} social engineering trigger(s). Primary: {primary} ({linguistic_score}%)"
            )
        else:
            reasons.append("Linguistic AI: Call transcript contains benign conversational patterns.")

    # 3. Transaction & Action Sensitivity (0 - 100)
    transaction_risk = base_action_risk
    if sensitive:
        reasons.append(f"Banking Policy: High-impact sensitive operation ({banking_action})")
    else:
        reasons.append(f"Banking Policy: Standard operation ({banking_action})")

    # 4. Authentication & Behavioral Risk (0 - 100)
    auth_risk = AUTH_RISK[authentication_status] * 3  # Scale to ~0 - 75
    if authentication_status == "failed":
        reasons.append("Identity Context: Caller failed direct authentication challenge")
    elif authentication_status == "unverified":
        reasons.append("Identity Context: Caller identity is currently unverified")

    if unusual_activity:
        auth_risk = min(100, auth_risk + 30)
        reasons.append("Behavioral Anomaly: Unusual account velocity or geo-ip anomaly reported")

    # ==========================================
    # WEIGHTED MULTI-MODAL FUSION FORMULA
    # ==========================================
    # Weights:
    # Acoustic Deepfake: 35%
    # Linguistic Scam Intent: 35%
    # Banking Action: 15%
    # Auth & Behavior: 15%
    weighted_score = (
        (0.35 * acoustic_score)
        + (0.35 * linguistic_score)
        + (0.15 * transaction_risk)
        + (0.15 * auth_risk)
    )

    # Check Compound Threat Multiplier:
    # If acoustic spoof AND linguistic scam are simultaneously high, threat escalates critically!
    compound_threat = (acoustic_score >= 65 and linguistic_score >= 60) or (
        linguistic_score >= 80 and authentication_status == "failed"
    )

    if compound_threat:
        weighted_score = max(weighted_score, 92.0)
        reasons.insert(0, "🚨 COMPOUND THREAT DETECTED: Simultaneous synthetic voice and social engineering scam vectors identified!")

    fused_score = min(100, int(round(weighted_score)))

    # Determine Action & Security Level
    if fused_score >= 80 or (compound_threat and fused_score >= 75):
        threat_level = "CRITICAL"
        action = "CRITICAL_FRAUD_INTERCEPT"
        protocol = "Immediate automated call termination. Freeze online banking access and dispatch urgent SMS security alert to account owner."
    elif fused_score >= 55 or linguistic_score >= 75 or acoustic_score >= 80:
        threat_level = "HIGH"
        action = "HUMAN_REVIEW_REQUIRED"
        protocol = "Live call intercept: Transfer session directly to Senior Fraud Investigation Desk. Freeze high-risk funds movement."
    elif fused_score >= 30 or sensitive or authentication_status != "verified":
        threat_level = "MEDIUM"
        action = "STEP_UP_VERIFICATION"
        protocol = "Trigger mandatory Out-of-Band (OOB) push challenge on verified mobile device before proceeding."
    else:
        threat_level = "LOW"
        action = "STANDARD_BANKING_FLOW"
        protocol = "Standard banking protocol: Authorize requested operation."

    return asdict(
        ThreatFusionDecision(
            fused_threat_score=fused_score,
            threat_level=threat_level,
            action=action,
            acoustic_risk=acoustic_score,
            linguistic_risk=linguistic_score,
            transaction_risk=transaction_risk,
            auth_risk=auth_risk,
            compound_threat=compound_threat,
            reasons=reasons,
            recommended_protocol=protocol,
        )
    )


if __name__ == "__main__":
    from pprint import pprint

    # Test benign call
    benign_threat = assess_fused_threat(
        banking_action="balance_inquiry",
        authentication_status="verified",
        unusual_activity=False,
        audio_spoof_score=0.05,
        scam_analysis={"scam_detected": False, "scam_score": 0.04, "threat_tier": "NONE", "triggers": []},
    )
    print("--- BENIGN CALL ---")
    pprint(benign_threat)

    # Test full compound threat (Synthetic voice + OTP Harvesting)
    compound_test = assess_fused_threat(
        banking_action="mobile_number_change",
        authentication_status="unverified",
        unusual_activity=True,
        audio_spoof_score=0.88,
        scam_analysis={
            "scam_detected": True,
            "scam_score": 0.95,
            "threat_tier": "CRITICAL",
            "primary_intent": "CREDENTIAL_HARVESTING",
            "triggers": [{"pattern": "otp", "severity": "CRITICAL"}],
        },
    )
    print("\n--- COMPOUND FRAUD CALL ---")
    pprint(compound_test)
