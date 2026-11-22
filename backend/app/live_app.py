
import asyncio
import logging
import time

import numpy as np

from fastapi import (
    WebSocket,
    WebSocketDisconnect,
)

from backend.app.main import (
    app,
    get_model,
    model_provider_info,
)


# ==========================================
# STREAM CONFIGURATION
# ==========================================

SAMPLE_RATE = 16000

WINDOW_SAMPLES = 64600

HOP_SAMPLES = 32000

MAX_STREAM_SECONDS = 30

MAX_CHUNK_BYTES = 32000

ALLOWED_ORIGINS = {
    "http://localhost:5173",
    "http://127.0.0.1:5173",
}

logger = logging.getLogger(
    "vaanishield.live"
)


# ==========================================
# CUDA-ACCELERATED WINDOW ANALYSIS
# ==========================================

def analyse_window(waveform):
    """
    Perform W2V2-AASIST inference.

    Uses the cached model session from main.py,
    which prefers CUDA and falls back to CPU.
    """

    session = get_model()

    batch = np.asarray(
        waveform,
        dtype=np.float32,
    )[np.newaxis, :]

    start = time.perf_counter()

    output = session.run(
        ["logits"],
        {"wav": batch},
    )

    elapsed = (
        time.perf_counter() - start
    )

    logits = np.asarray(
        output[0][0],
        dtype=np.float64,
    )

    if (
        logits.shape != (2,)
        or not np.all(np.isfinite(logits))
    ):
        raise ValueError(
            "Invalid model output"
        )

    shifted = (
        logits - np.max(logits)
    )

    exp_values = np.exp(shifted)

    probabilities = (
        exp_values / exp_values.sum()
    )

    return {
        "spoof_score": round(
            float(probabilities[0]), 6
        ),

        "genuine_score": round(
            float(probabilities[1]), 6
        ),

        "inference_seconds": round(
            elapsed, 3
        ),

        **model_provider_info(session),
    }


# ==========================================
# LIVE WEBSOCKET ENDPOINT
# ==========================================

@app.websocket("/ws/live-audio")
async def live_audio(websocket: WebSocket):
    """
    Receive 16 kHz mono signed 16-bit PCM.

    Analyse overlapping ~4-second windows,
    advancing every 2 seconds.

    Experimental monitoring only.
    No automatic banking actions.
    """

    origin = websocket.headers.get(
        "origin"
    )

    if (
        origin
        and origin not in ALLOWED_ORIGINS
    ):
        await websocket.close(
            code=1008
        )
        return

    await websocket.accept()

    # Initialize the cached model once.
    # Run loading in a worker thread so the
    # WebSocket event loop is not blocked.
    try:
        session = await asyncio.to_thread(
            get_model
        )

    except Exception:
        logger.exception(
            "Unable to initialize AI model"
        )

        await websocket.send_json({
            "type": "error",
            "message": (
                "Unable to initialize AI model"
            ),
        })

        await websocket.close(
            code=1011
        )
        return

    provider_info = model_provider_info(
        session
    )

    logger.info(
        "Live monitoring providers: %s",
        provider_info["execution_providers"],
    )

    await websocket.send_json({
        "type": "ready",
        "model": "W2V2-AASIST",
        "sample_rate": SAMPLE_RATE,
        "window_samples": WINDOW_SAMPLES,
        "hop_samples": HOP_SAMPLES,
        **provider_info,
        "mode": "MONITOR_ONLY",
    })

    # Per-connection audio buffer.
    buffer = np.empty(
        0,
        dtype=np.float32,
    )

    total_samples = 0
    window_start = 0

    try:
        while True:
            message = await websocket.receive()

            # --------------------------------
            # DISCONNECT
            # --------------------------------

            if (
                message["type"]
                == "websocket.disconnect"
            ):
                break

            command = message.get("text")

            # --------------------------------
            # STOP COMMAND
            # --------------------------------

            if command == "stop":
                await websocket.send_json({
                    "type": "finished",
                    "duration_seconds": round(
                        total_samples / SAMPLE_RATE,
                        2,
                    ),
                })

                await websocket.close()
                break

            if command is not None:
                await websocket.send_json({
                    "type": "error",
                    "message": "Unknown command",
                })
                continue

            # --------------------------------
            # RECEIVE PCM AUDIO
            # --------------------------------

            data = message.get("bytes")

            if not data:
                continue

            if (
                len(data) % 2 != 0
                or len(data) > MAX_CHUNK_BYTES
            ):
                await websocket.close(
                    code=1009,
                    reason="Invalid PCM chunk size",
                )
                break

            incoming = (
                np.frombuffer(
                    data,
                    dtype="<i2",
                )
                .astype(np.float32)
                / 32768.0
            )

            if (
                total_samples + len(incoming)
                > MAX_STREAM_SECONDS * SAMPLE_RATE
            ):
                await websocket.close(
                    code=1009,
                    reason=(
                        "Stream duration limit exceeded"
                    ),
                )
                break

            total_samples += len(incoming)

            buffer = np.concatenate(
                (buffer, incoming)
            )

            # --------------------------------
            # ANALYSE AUDIO WINDOWS
            # --------------------------------

            while (
                len(buffer) >= WINDOW_SAMPLES
            ):
                window = buffer[
                    :WINDOW_SAMPLES
                ].copy()

                end_seconds = (
                    window_start + WINDOW_SAMPLES
                ) / SAMPLE_RATE

                rms = float(
                    np.sqrt(
                        np.mean(
                            window.astype(
                                np.float64
                            ) ** 2
                        )
                    )
                )

                if rms < 0.001:
                    await websocket.send_json({
                        "type": "audio_quality",
                        "status": "TOO_QUIET",
                        "window_end_seconds": round(
                            end_seconds,
                            2,
                        ),
                    })

                else:
                    result = await asyncio.to_thread(
                        analyse_window,
                        window,
                    )

                    await websocket.send_json({
                        "type": "prediction",

                        "model": "W2V2-AASIST",

                        "window_start_seconds": round(
                            window_start / SAMPLE_RATE,
                            2,
                        ),

                        "window_end_seconds": round(
                            end_seconds,
                            2,
                        ),

                        **result,

                        "mode": "MONITOR_ONLY",

                        "banking_action": "NONE",

                        "disclaimer": (
                            "Uncalibrated model output. "
                            "No banking action is taken."
                        ),
                    })

                # Keep overlapping samples
                # for the next prediction.
                buffer = buffer[
                    HOP_SAMPLES:
                ]

                window_start += HOP_SAMPLES

    except WebSocketDisconnect:
        logger.info(
            "Live audio client disconnected"
        )

    except Exception:
        logger.exception(
            "Live audio analysis failed"
        )

        try:
            await websocket.send_json({
                "type": "error",
                "message": (
                    "Live audio analysis failed"
                ),
            })

            await websocket.close(
                code=1011
            )

        except Exception:
            pass
