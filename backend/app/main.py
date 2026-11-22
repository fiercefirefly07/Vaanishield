
import logging
import time

from functools import lru_cache
from io import BytesIO
from pathlib import Path
from typing import Literal

import numpy as np
import soundfile as sf

# Import PyTorch before ONNX Runtime so its CUDA
# and cuDNN libraries are available on Windows.
import torch
import onnxruntime as ort

from fastapi import (
    FastAPI,
    UploadFile,
    File,
    Form,
    HTTPException,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from ai_ml.audio_preprocessing import (
    preprocess_audio,
    SAMPLE_RATE,
    MODEL_SAMPLES,
)
from backend.app.risk_engine import (
    assess_risk,
    assess_fused_threat,
)
from backend.app.scam_detector import (
    detect_scam_intent,
    SAMPLE_SCENARIOS,
)
from backend.app.transcription import (
    transcribe_audio_bytes,
)
from backend.app.audio_forensics import (
    analyze_audio_forensics,
)
from backend.app.voice_biometrics import (
    verify_speaker_identity,
    enroll_customer,
    load_voiceprint_db,
    ensure_demo_enrolled_customer,
)
from backend.app.compliance_report import (
    generate_compliance_audit_report,
    generate_printable_html_report,
)
from backend.app.verification_cases import (
    router as verification_router,
)


# ==========================================
# CONFIGURATION
# ==========================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("vaanishield")

ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    ROOT / "ai_ml" / "models" / "w2v2-aasist.onnx"
)

MAX_FILE_SIZE = 10 * 1024 * 1024
MAX_DURATION = 30


BankingAction = Literal[
    "balance_inquiry",
    "transaction_status",
    "password_reset",
    "mobile_number_change",
    "transfer_limit_increase",
    "high_value_transfer",
]

AuthenticationStatus = Literal[
    "verified",
    "unverified",
    "failed",
]


class RiskRequest(BaseModel):
    banking_action: BankingAction
    authentication_status: AuthenticationStatus
    unusual_activity: bool = False


# ==========================================
# FASTAPI APPLICATION
# ==========================================

app = FastAPI(
    title="VaaniShield API",
    description=(
        "Experimental GPU-accelerated voice "
        "anti-spoofing, banking risk assessment "
        "and verification case management"
    ),
    version="0.4.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ==========================================
# CUDA / CPU MODEL LOADING
# ==========================================

def make_session(providers):
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4

    return ort.InferenceSession(
        str(MODEL_PATH),
        sess_options=options,
        providers=providers,
    )


@lru_cache(maxsize=1)
def get_model():
    """
    Prefer RTX GPU through CUDAExecutionProvider.

    If CUDA fails, use CPUExecutionProvider.

    The model session is cached so it is reused
    across uploaded-audio and live-streaming calls.
    """

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    available = ort.get_available_providers()

    logger.info(
        "Available ONNX Runtime providers: %s",
        available,
    )

    if "CUDAExecutionProvider" in available:
        try:
            # PyTorch was imported above to make its
            # CUDA/cuDNN runtime libraries available.
            # ONNX Runtime 1.21+ also supports
            # explicit DLL preloading on Windows.
            if hasattr(ort, "preload_dlls"):
                ort.preload_dlls(
                    cuda=True,
                    cudnn=True,
                    msvc=True,
                )

            session = make_session([
                (
                    "CUDAExecutionProvider",
                    {"device_id": 0},
                ),
                "CPUExecutionProvider",
            ])

            active = session.get_providers()

            if "CUDAExecutionProvider" in active:
                logger.info(
                    "VaaniShield CUDA provider enabled: %s",
                    active,
                )
            else:
                logger.warning(
                    "CUDA was requested but not "
                    "activated. Active providers: %s",
                    active,
                )

            return session

        except Exception:
            logger.exception(
                "CUDA initialization failed. "
                "Falling back to CPU."
            )

    logger.info(
        "Loading W2V2-AASIST with CPU provider"
    )

    session = make_session([
        "CPUExecutionProvider"
    ])

    logger.info(
        "Active providers: %s",
        session.get_providers(),
    )

    return session


def model_provider_info(session):
    providers = session.get_providers()

    return {
        "execution_providers": providers,
        "cuda_provider_enabled": (
            "CUDAExecutionProvider" in providers
        ),
    }


# ==========================================
# AUDIO ANALYSIS FUNCTION
# ==========================================

def run_voice_analysis(
    file: UploadFile,
    content: bytes | None = None,
):
    filename = file.filename or ""

    if Path(filename).suffix.lower() not in (
        ".wav",
        ".flac",
    ):
        raise HTTPException(
            status_code=400,
            detail=(
                "Only WAV and FLAC files are supported"
            ),
        )

    if content is None:
        content = file.file.read(
            MAX_FILE_SIZE + 1
        )

    if not content:
        raise HTTPException(
            status_code=400,
            detail="Empty audio file",
        )

    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail="Audio file exceeds 10 MB",
        )

    try:
        with sf.SoundFile(
            BytesIO(content)
        ) as audio_file:

            sr = audio_file.samplerate

            duration = (
                len(audio_file) / sr
            )

            if (
                duration < 1
                or duration > MAX_DURATION
            ):
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "Audio must be 1 to 30 seconds"
                    ),
                )

            if audio_file.channels > 2:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "Maximum two channels supported"
                    ),
                )

            audio = audio_file.read(
                dtype="float32",
                always_2d=True,
            )

    except HTTPException:
        raise

    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Unable to decode audio",
        )

    try:
        processed = preprocess_audio(
            audio,
            sr,
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )

    processed = np.asarray(
        processed,
        dtype=np.float32,
    )

    rms = float(
        np.sqrt(
            np.mean(
                processed.astype(
                    np.float64
                ) ** 2
            )
        )
    )

    if rms < 0.001:
        raise HTTPException(
            status_code=400,
            detail="Audio is silent or too quiet",
        )

    batch = processed[np.newaxis, :]

    try:
        session = get_model()

        start = time.perf_counter()

        result = session.run(
            ["logits"],
            {"wav": batch},
        )

        elapsed = (
            time.perf_counter() - start
        )

    except Exception:
        logger.exception(
            "Voice model inference failed"
        )

        raise HTTPException(
            status_code=500,
            detail="Model inference failed",
        )

    logits = np.asarray(
        result[0][0],
        dtype=np.float64,
    )

    if (
        logits.shape != (2,)
        or not np.all(np.isfinite(logits))
    ):
        raise HTTPException(
            status_code=500,
            detail="Invalid model output",
        )

    shifted = (
        logits - np.max(logits)
    )

    exp_values = np.exp(shifted)

    probabilities = (
        exp_values / exp_values.sum()
    )

    spoof_score = float(
        probabilities[0]
    )

    genuine_score = float(
        probabilities[1]
    )

    prediction = (
        "SPOOF_INDICATORS"
        if spoof_score > genuine_score
        else "GENUINE_INDICATORS"
    )

    return {
        "model": "W2V2-AASIST",

        "prediction": prediction,

        "spoof_score": round(
            spoof_score, 6
        ),

        "genuine_score": round(
            genuine_score, 6
        ),

        "inference_seconds": round(
            elapsed, 3
        ),

        "recording_duration_seconds": round(
            duration, 3
        ),

        "analyzed_audio_seconds": round(
            MODEL_SAMPLES / SAMPLE_RATE, 4
        ),

        **model_provider_info(session),

        "mode": "MONITOR_ONLY",

        "validation_status": (
            "UNVALIDATED_FOR_BANKING"
        ),

        "action": (
            "NO_AUTOMATIC_AUDIO_ACTION"
        ),

        "disclaimer": (
            "Audio scores are uncalibrated and "
            "must not be used as proof of fraud."
        ),
    }


# ==========================================
# HOME ENDPOINT
# ==========================================

@app.get("/")
def home():
    return {
        "project": "VaaniShield",
        "status": "running",
        "version": "0.4.0",
        "mode": "EXPERIMENTAL",
    }


# ==========================================
# HEALTH / CUDA STATUS
# ==========================================

@app.get("/health")
def health():
    available = ort.get_available_providers()

    if not MODEL_PATH.is_file():
        return {
            "status": "degraded",
            "model_file_available": False,
            "model": "W2V2-AASIST",
            "onnxruntime_available_providers": (
                available
            ),
        }

    try:
        session = get_model()

        provider_info = model_provider_info(
            session
        )

        return {
            "status": "healthy",
            "model_file_available": True,
            "model": "W2V2-AASIST",
            "onnxruntime_available_providers": (
                available
            ),
            **provider_info,
            "pytorch_cuda_available": (
                torch.cuda.is_available()
            ),
            "mode": "MONITOR_ONLY",
        }

    except Exception:
        logger.exception(
            "Health check model initialization failed"
        )

        return {
            "status": "degraded",
            "model_file_available": True,
            "model": "W2V2-AASIST",
            "model_loaded": False,
            "onnxruntime_available_providers": (
                available
            ),
        }


# ==========================================
# AUDIO ANALYSIS
# ==========================================

@app.post("/api/analyze")
def analyze_audio(
    file: UploadFile = File(...),
):
    return run_voice_analysis(file)


# ==========================================
# BANKING RISK ASSESSMENT
# ==========================================

@app.post("/api/risk-assessment")
def risk_assessment(
    request: RiskRequest,
):
    result = assess_risk(
        banking_action=request.banking_action,
        authentication_status=(
            request.authentication_status
        ),
        unusual_activity=(
            request.unusual_activity
        ),
    )

    return {
        "risk_assessment": result,
        "mode": "SIMULATION",
        "message": (
            "Banking context policy evaluated. "
            "No real banking action was performed."
        ),
    }


class TextIntentRequest(BaseModel):
    text: str


class HtmlReportRequest(BaseModel):
    report_data: dict


# ==========================================
# COMBINED MULTI-MODAL THREAT FUSION CALL
# ==========================================

@app.post("/api/analyze-call")
def analyze_call(
    file: UploadFile = File(...),
    banking_action: BankingAction = Form(...),
    authentication_status: AuthenticationStatus = Form(...),
    unusual_activity: bool = Form(False),
    transcript_override: str | None = Form(None),
    customer_id: str = Form("CUST-10492"),
):
    content = file.file.read(MAX_FILE_SIZE + 1)
    if not content:
        raise HTTPException(
            status_code=400,
            detail="Empty audio file",
        )
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail="Audio file exceeds 10 MB",
        )

    # 1. Acoustic voice analysis (AASIST deepfake model)
    audio_result = run_voice_analysis(file, content=content)

    # 2. Acoustic Forensics & Spectral Anomaly Diagnostics
    forensics_result = analyze_audio_forensics(content)

    # 3. Transcription (use override if supplied, else Google ASR)
    if transcript_override and transcript_override.strip():
        transcription_result = {
            "success": True,
            "text": transcript_override.strip(),
            "word_count": len(transcript_override.strip().split()),
            "engine": "User/Browser-Transcript",
            "error": None,
        }
    else:
        transcription_result = transcribe_audio_bytes(content)

    # 4. Linguistic Scam & Social Engineering Intent Detection
    scam_result = detect_scam_intent(transcription_result["text"])

    # 5. Customer Voice Biometrics & Speaker Verification (1:1)
    biometrics_result = verify_speaker_identity(
        customer_id=customer_id,
        caller_audio_bytes=content,
        audio_spoof_score=audio_result["spoof_score"],
    )

    # 6. Multi-Modal Threat Fusion Engine
    fused_threat = assess_fused_threat(
        banking_action=banking_action,
        authentication_status=authentication_status,
        unusual_activity=unusual_activity,
        audio_spoof_score=audio_result["spoof_score"],
        scam_analysis=scam_result,
    )

    # 7. Legacy policy calculation (for backward compatibility)
    risk_result = assess_risk(
        banking_action=banking_action,
        authentication_status=authentication_status,
        unusual_activity=unusual_activity,
        audio_spoof_score=audio_result["spoof_score"],
    )

    banking_ctx = {
        "requested_action": banking_action,
        "authentication_status": authentication_status,
        "unusual_activity": unusual_activity,
        "target_customer_id": customer_id,
    }

    # 8. Forensic Audit & Regulatory Compliance Report
    compliance_report = generate_compliance_audit_report(
        audio_bytes=content,
        audio_analysis=audio_result,
        forensics=forensics_result,
        transcription=transcription_result,
        scam_analysis=scam_result,
        biometrics=biometrics_result,
        threat_fusion=fused_threat,
        banking_context=banking_ctx,
    )

    return {
        "project": "VaaniShield",
        "audio_analysis": audio_result,
        "audio_forensics": forensics_result,
        "transcription": transcription_result,
        "scam_analysis": scam_result,
        "voice_biometrics": biometrics_result,
        "threat_fusion": fused_threat,
        "compliance_report": compliance_report,
        "banking_context": banking_ctx,
        "risk_assessment": risk_result,
        "mode": "MULTI_MODAL_THREAT_FUSION",
        "notice": (
            "Multi-Modal Threat Engine active. Fuses acoustic spoof probability, vocoder "
            "forensics, linguistic intent triggers, and customer voice biometrics for unified defense."
        ),
    }


# ==========================================
# VOICE BIOMETRICS ENDPOINTS
# ==========================================

@app.get("/api/voice-biometrics/customers")
def get_enrolled_customers():
    ensure_demo_enrolled_customer()
    db = load_voiceprint_db()
    return {
        "total_enrolled": len(db),
        "customers": [
            {
                "customer_id": k,
                "customer_name": v.get("customer_name"),
                "account_number": v.get("account_number"),
                "enrolled_at": v.get("enrolled_at"),
            }
            for k, v in db.items()
        ],
    }


@app.post("/api/voice-biometrics/enroll")
def enroll_customer_voice(
    file: UploadFile = File(...),
    customer_id: str = Form(...),
    customer_name: str = Form(...),
    account_number: str = Form("•••• 8921"),
):
    content = file.file.read(MAX_FILE_SIZE + 1)
    if not content or len(content) > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="Invalid audio file for enrollment")
    return enroll_customer(
        customer_id=customer_id,
        customer_name=customer_name,
        audio_bytes=content,
        account_number=account_number,
    )


# ==========================================
# COMPLIANCE REPORT PRINTABLE HTML ENDPOINT
# ==========================================

@app.post("/api/compliance/generate-html", response_class=HTMLResponse)
def generate_html_report(request: HtmlReportRequest):
    return generate_printable_html_report(request.report_data)


# ==========================================
# TEXT INTENT & SCAM PRESETS ENDPOINTS
# ==========================================

@app.post("/api/analyze-text-intent")
def analyze_text_intent(request: TextIntentRequest):
    return detect_scam_intent(request.text)


@app.get("/api/scam-presets")
def get_scam_presets():
    return {
        key: {
            "title": key.replace("_", " ").title(),
            "sample_transcript": text,
        }
        for key, text in SAMPLE_SCENARIOS.items()
    }


# ==========================================
# SQLITE VERIFICATION CASE MANAGEMENT
# ==========================================

# Preserves existing Phase 8 / Phase 9 routes:
#
# POST /api/verification-cases
# GET  /api/verification-cases
# GET  /api/verification-cases/{case_id}
# POST /api/verification-cases/{case_id}/respond
# POST /api/verification-cases/{case_id}/review

app.include_router(verification_router)


# ==========================================
# LIVE WEBSOCKET AUDIO STREAMING (PHASE 6)
# ==========================================

import backend.app.live_app  # noqa: F401 - registers @app.websocket("/ws/live-audio")
