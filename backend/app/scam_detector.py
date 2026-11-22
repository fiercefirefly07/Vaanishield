import re
from dataclasses import dataclass, asdict
from typing import List, Dict, Any


@dataclass
class ScamTrigger:
    category: str
    severity: str  # CRITICAL, HIGH, MEDIUM, LOW
    pattern_matched: str
    snippet: str
    weight: int


@dataclass
class ScamAnalysisResult:
    scam_detected: bool
    scam_score: float  # 0.0 to 1.0
    threat_tier: str  # NONE, LOW, MEDIUM, HIGH, CRITICAL
    primary_intent: str
    triggers: List[Dict[str, Any]]
    total_triggers_count: int
    matched_categories: List[str]
    coercion_level: str
    explanation: str


# ==========================================
# SCAM & SOCIAL ENGINEERING PATTERNS
# ==========================================

SCAM_TAXONOMY = {
    "CREDENTIAL_HARVESTING": {
        "severity": "CRITICAL",
        "weight": 95,
        "patterns": [
            r"\b(?:otp|one[\s-]time[\s-]password)\b",
            r"\b(?:6|six)[\s-]digit[\s-]code\b",
            r"\b(?:atm|debit|credit)[\s-]card[\s-]pin\b",
            r"\b(?:cvv|cvv2|security code)\b",
            r"\b(?:net[\s-]?banking|online[\s-]banking)[\s-]password\b",
            r"\b(?:read|tell|give)[\s-]me[\s-]the[\s-](?:code|otp|sms|digits?)\b",
            r"\b(?:verification|security)[\s-]code[\s-]sent[\s-]to\b",
            r"\bshare[\s-]your[\s-]credentials?\b",
        ],
        "description": "Attempt to extract secret credentials, PINs, or One-Time Passwords (OTP).",
    },
    "REMOTE_ACCESS_COERCION": {
        "severity": "CRITICAL",
        "weight": 95,
        "patterns": [
            r"\b(?:anydesk|teamviewer|quicksupport|rustdesk|ultraviewer)\b",
            r"\b(?:download|install)[\s-](?:the|this)?[\s-]app(?:lication)?\b",
            r"\b(?:screen[\s-]share|share[\s-]your[\s-]screen)\b",
            r"\b(?:remote[\s-]access|remote[\s-]support|remote[\s-]control)\b",
            r"\b(?:download|install)[\s-](?:apk|software)\b",
            r"\bopen[\s-](?:browser|play[\s-]store|app[\s-]store)[\s-]and[\s-]type\b",
            r"\bgrant[\s-]permission\b",
        ],
        "description": "Coercing the customer into installing remote desktop control tools.",
    },
    "URGENCY_AND_INTIMIDATION": {
        "severity": "HIGH",
        "weight": 80,
        "patterns": [
            r"\b(?:within|in)[\s-](?:5|10|15|30)[\s-]minutes?\b",
            r"\b(?:immediately|right[\s-]now|urgent(?:ly)?)\b",
            r"\b(?:account|card|service)[\s-]will[\s-]be[\s-](?:blocked|frozen|suspended|deactivated|cancelled)\b",
            r"\b(?:electricity|power|gas)[\s-]will[\s-]be[\s-]cut[\s-]off\b",
            r"\b(?:arrest[\s-]warrant|police[\s-]fir|court[\s-]notice|legal[\s-]action)\b",
            r"\b(?:cyber[\s-]crime|cbi|ed|narcotics)[\s-](?:cell|department|officer)\b",
            r"\bdo[\s-]not[\s-](?:hang[\s-]up|disconnect|call[\s-]anyone)\b",
            r"\bkeep[\s-]it[\s-]confidential\b",
        ],
        "description": "Fabricating extreme urgency or legal threats to bypass cognitive skepticism.",
    },
    "UNAUTHORIZED_FUND_DIVERSION": {
        "severity": "HIGH",
        "weight": 85,
        "patterns": [
            r"\b(?:safe|reserve|temporary|holding|verification)[\s-]account\b",
            r"\btransfer[\s-](?:funds?|money)[\s-]to[\s-](?:verify|secure|protect)\b",
            r"\b(?:pay|send)[\s-]security[\s-]deposit\b",
            r"\b(?:gift[\s-]card|cryptocurrency|crypto|bitcoin|usdt)\b",
            r"\brefund[\s-](?:processing|failed|claimed|approved)\b",
            r"\b(?:overpaid|accidental[\s-]transfer)[\s-]send[\s-]it[\s-]back\b",
        ],
        "description": "Directing funds into 'safe accounts', crypto wallets, or mule accounts.",
    },
    "AUTHORITY_IMPERSONATION": {
        "severity": "MEDIUM",
        "weight": 65,
        "patterns": [
            r"\bcalling[\s-]from[\s-](?:the\s+)?(?:fraud|security|investigation)[\s-]department\b",
            r"\b(?:rbi|reserve[\s-]bank|central[\s-]bank)[\s-](?:officer|guidelines?)\b",
            r"\b(?:bank|head[\s-]office)[\s-]manager\b",
            r"\bcustoms[\s-](?:officer|department|parcel)\b",
            r"\btax[\s-](?:department|officer|evasion)\b",
            r"\bkyc[\s-](?:verification|update|expired|mandatory)\b",
        ],
        "description": "Falsely claiming authority from government, banking, or regulatory bodies.",
    },
}


def extract_snippet(text: str, match_start: int, match_end: int, window: int = 40) -> str:
    """Extracts a readable context snippet around a matched phrase."""
    start = max(0, match_start - window)
    end = min(len(text), match_end + window)
    prefix = "..." if start > 0 else ""
    suffix = "..." if end < len(text) else ""
    return f"{prefix}{text[start:end].strip()}{suffix}"


def detect_scam_intent(transcript: str) -> Dict[str, Any]:
    """
    Analyzes a call transcript for social engineering tactics, credential harvesting,
    and coercion patterns.
    """
    if not transcript or not transcript.strip():
        return asdict(
            ScamAnalysisResult(
                scam_detected=False,
                scam_score=0.0,
                threat_tier="NONE",
                primary_intent="BENIGN_OR_SILENT",
                triggers=[],
                total_triggers_count=0,
                matched_categories=[],
                coercion_level="NONE",
                explanation="No audible speech or transcript provided for intent analysis.",
            )
        )

    text = transcript.lower().strip()
    triggers: List[ScamTrigger] = []
    category_scores: Dict[str, int] = {}

    for category, config in SCAM_TAXONOMY.items():
        cat_triggers = 0
        for pattern in config["patterns"]:
            for match in re.finditer(pattern, text):
                snippet = extract_snippet(transcript, match.start(), match.end())
                triggers.append(
                    ScamTrigger(
                        category=category,
                        severity=config["severity"],
                        pattern_matched=match.group(0),
                        snippet=snippet,
                        weight=config["weight"],
                    )
                )
                cat_triggers += 1

        if cat_triggers > 0:
            category_scores[category] = config["weight"] + min(20, (cat_triggers - 1) * 10)

    # Multi-category fusion calculation
    if not triggers:
        return asdict(
            ScamAnalysisResult(
                scam_detected=False,
                scam_score=0.04,
                threat_tier="NONE",
                primary_intent="BENIGN_CUSTOMER_INQUIRY",
                triggers=[],
                total_triggers_count=0,
                matched_categories=[],
                coercion_level="NONE",
                explanation="Transcript contains standard customer dialogue with no suspicious coercion or scam triggers.",
            )
        )

    # Calculate overall scam confidence (0.0 to 1.0)
    max_cat_score = max(category_scores.values())
    additional_factor = (len(category_scores) - 1) * 10 + (len(triggers) - 1) * 3
    composite_raw = min(100, max_cat_score + additional_factor)
    scam_score = round(composite_raw / 100.0, 4)

    # Identify primary intent (highest weighted category)
    primary_category = max(category_scores.items(), key=lambda x: x[1])[0]

    # Coercion level classification
    if "URGENCY_AND_INTIMIDATION" in category_scores or "REMOTE_ACCESS_COERCION" in category_scores:
        coercion_level = "CRITICAL" if composite_raw >= 85 else "HIGH"
    elif len(category_scores) >= 2:
        coercion_level = "MEDIUM"
    else:
        coercion_level = "LOW"

    # Threat tier
    if scam_score >= 0.85:
        threat_tier = "CRITICAL"
    elif scam_score >= 0.65:
        threat_tier = "HIGH"
    elif scam_score >= 0.40:
        threat_tier = "MEDIUM"
    else:
        threat_tier = "LOW"

    # Explanation construction
    categories_str = ", ".join(list(category_scores.keys()))
    explanation = (
        f"Detected {len(triggers)} suspicious indicators across {len(category_scores)} "
        f"threat categories ({categories_str}). Primary concern: {primary_category}."
    )

    return asdict(
        ScamAnalysisResult(
            scam_detected=True,
            scam_score=scam_score,
            threat_tier=threat_tier,
            primary_intent=primary_category,
            triggers=[asdict(t) for t in triggers],
            total_triggers_count=len(triggers),
            matched_categories=list(category_scores.keys()),
            coercion_level=coercion_level,
            explanation=explanation,
        )
    )


# Sample presets for testing
SAMPLE_SCENARIOS = {
    "otp_fraud": (
        "Hello sir, I am calling from the bank fraud department. We noticed an unauthorized "
        "transaction on your account. Your account will be blocked within 10 minutes if not verified. "
        "Please read me the 6-digit OTP code sent to your phone right now."
    ),
    "remote_access": (
        "Ma'am, this is senior security officer from RBI head office. Your banking app is compromised. "
        "Please open your browser, download AnyDesk right now and share your screen so I can secure your funds."
    ),
    "safe_account": (
        "Your debit card was used in Mumbai for illegal transactions. An arrest warrant has been issued. "
        "To clear your name immediately, you must transfer all funds to our RBI safe holding account for verification."
    ),
    "benign_inquiry": (
        "Hello, I am calling to check my account balance and confirm if my last monthly deposit went through successfully."
    ),
}

if __name__ == "__main__":
    for name, text in SAMPLE_SCENARIOS.items():
        print("=" * 60)
        print(f"Scenario: {name}")
        res = detect_scam_intent(text)
        print(f"Detected: {res['scam_detected']} | Score: {res['scam_score']} | Tier: {res['threat_tier']}")
        print(f"Primary Intent: {res['primary_intent']} | Coercion: {res['coercion_level']}")
        print(f"Triggers: {res['total_triggers_count']}")
        for trig in res["triggers"]:
            print(f"  - [{trig['category']} - {trig['severity']}] '{trig['pattern_matched']}' in '{trig['snippet']}'")
