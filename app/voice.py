"""Local speech-to-text and text-to-speech services for the API."""
import io
import logging
import threading
import wave
from pathlib import Path
from tempfile import NamedTemporaryFile

import pyttsx3
from faster_whisper import WhisperModel

from app import config

logger = logging.getLogger(__name__)


class VoiceEngine:
    """Lazy-loaded voice services so normal text API startup stays lightweight."""

    def __init__(self):
        logger.info(
            "Loading Whisper model: size=%s device=%s compute_type=%s",
            config.WHISPER_MODEL_SIZE,
            config.WHISPER_DEVICE,
            config.WHISPER_COMPUTE_TYPE,
        )
        config.WHISPER_MODEL_DIR.mkdir(parents=True, exist_ok=True)
        self.whisper = WhisperModel(
            config.WHISPER_MODEL_SIZE,
            device=config.WHISPER_DEVICE,
            compute_type=config.WHISPER_COMPUTE_TYPE,
            download_root=str(config.WHISPER_MODEL_DIR),
            cpu_threads=config.WHISPER_CPU_THREADS,
        )
        self._tts_lock = threading.Lock()

    def transcribe(self, audio_bytes: bytes, suffix: str) -> tuple[str, str]:
        """Transcribe an uploaded audio file and return (text, language)."""
        safe_suffix = suffix if suffix and len(suffix) <= 10 else ".wav"
        with NamedTemporaryFile(suffix=safe_suffix, delete=False) as temp_file:
            temp_file.write(audio_bytes)
            audio_path = Path(temp_file.name)

        try:
            segments, info = self.whisper.transcribe(
                str(audio_path),
                beam_size=5,
                vad_filter=True,
            )
            text = " ".join(segment.text.strip() for segment in segments).strip()
            return text, info.language or "en"
        finally:
            audio_path.unlink(missing_ok=True)

    def synthesize(self, text: str, language: str) -> bytes:
        """Create WAV speech using a local operating-system voice."""
        # pyttsx3 is backed by SAPI on Windows and espeak on Linux. It does not
        # provide a Hindi voice by itself; we select one when installed.
        with self._tts_lock:
            engine = pyttsx3.init()
            engine.setProperty("rate", config.TTS_RATE)
            self._select_voice(engine, language)

            with NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                output_path = Path(temp_file.name)
            try:
                engine.save_to_file(text, str(output_path))
                engine.runAndWait()
                audio = output_path.read_bytes()
            finally:
                engine.stop()
                output_path.unlink(missing_ok=True)

        # Ensure the API always returns a normal WAV payload.
        with wave.open(io.BytesIO(audio), "rb") as wav_file:
            if wav_file.getnframes() == 0:
                raise RuntimeError("Text-to-speech returned empty audio")
        return audio

    @staticmethod
    def _select_voice(engine, language: str) -> None:
        if language != "hi":
            return
        for voice in engine.getProperty("voices"):
            voice_text = " ".join(
                str(value).lower()
                for value in (voice.id, voice.name, getattr(voice, "languages", []))
            )
            if any(marker in voice_text for marker in ("hindi", "hi-in", "indic")):
                engine.setProperty("voice", voice.id)
                return

