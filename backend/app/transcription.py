import io
import logging
import soundfile as sf
import speech_recognition as sr

logger = logging.getLogger("vaanishield.transcription")


def transcribe_audio_bytes(audio_bytes: bytes) -> dict:
    """
    Transcribes audio bytes to text using speech_recognition with Google ASR.
    Converts input audio to standard 16kHz mono WAV format if necessary.
    """
    try:
        data, samplerate = sf.read(io.BytesIO(audio_bytes))

        # Convert to mono if multi-channel
        if len(data.shape) > 1:
            data = data.mean(axis=1)

        # Write to WAV in-memory buffer for speech_recognition
        wav_buffer = io.BytesIO()
        sf.write(wav_buffer, data, samplerate, format="WAV", subtype="PCM_16")
        wav_buffer.seek(0)

        recognizer = sr.Recognizer()
        with sr.AudioFile(wav_buffer) as source:
            audio_data = recognizer.record(source)

        text = recognizer.recognize_google(audio_data)
        words = text.strip().split()

        return {
            "success": True,
            "text": text.strip(),
            "word_count": len(words),
            "engine": "Google-ASR",
            "error": None,
        }

    except sr.UnknownValueError:
        return {
            "success": True,
            "text": "",
            "word_count": 0,
            "engine": "Google-ASR",
            "error": "No recognizable speech detected in audio.",
        }
    except sr.RequestError as e:
        logger.warning("Speech recognition network request failed: %s", e)
        return {
            "success": False,
            "text": "",
            "word_count": 0,
            "engine": "Google-ASR",
            "error": f"Speech recognition service unavailable: {str(e)}",
        }
    except Exception as e:
        logger.exception("Audio transcription failed")
        return {
            "success": False,
            "text": "",
            "word_count": 0,
            "engine": "Google-ASR",
            "error": str(e),
        }
