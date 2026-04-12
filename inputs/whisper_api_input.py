#!/usr/bin/env python3
"""
Whisper API Input Device

Lightweight speech-to-text input using OpenAI's Whisper API.
Captures audio from the robot's microphone via ALSA (arecord), detects speech
with energy-based VAD, and sends completed utterances to the Whisper API for
transcription.  Results are published to /brain/chat_in.

Compared to micro_input.py (OpenAI Realtime), this is:
  - Cheaper (standard Whisper API vs Realtime WebSocket)
  - Simpler (no proxy dependency, no persistent WebSocket)
  - Easier to debug (audio segments written to disk if DEBUG_AUDIO=1)

Requires: OPENAI_API_KEY in the environment (or via proxy config).
"""

import base64
import io
import os
import queue
import struct
import threading
import time
import wave
from typing import Optional

from brain_client.input_types import InputDevice
from brain_client.logging_config import UniversalLogger


SAMPLE_RATE = 16_000
CHANNELS = 1
CHUNK_DURATION_SEC = 0.02
BYTES_PER_SAMPLE = 2

# VAD parameters
SILENCE_THRESHOLD = 300       # int16 RMS below this = silence
SILENCE_TIMEOUT_SEC = 1.5     # end utterance after this much silence
MIN_SPEECH_SEC = 0.4          # ignore bursts shorter than this
MAX_SPEECH_SEC = 30.0         # force-flush after this long


def _rms_int16(pcm: bytes) -> float:
    """RMS of s16le PCM buffer."""
    n = len(pcm) // 2
    if n == 0:
        return 0.0
    total = 0
    for i in range(n):
        s = struct.unpack_from("<h", pcm, i * 2)[0]
        total += s * s
    return (total / n) ** 0.5


def _pcm_to_wav(pcm: bytes, sample_rate: int = SAMPLE_RATE) -> bytes:
    """Wrap raw s16le mono PCM in a WAV container (in memory)."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(BYTES_PER_SAMPLE)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm)
    return buf.getvalue()


class WhisperApiInput(InputDevice):
    """
    Speech-to-text via OpenAI Whisper API.

    Audio capture uses ArecordStreamer (same as micro_input.py).
    VAD is energy-based: accumulate chunks while RMS > threshold, flush to
    Whisper API after a silence gap.
    """

    POST_TTS_GUARD_SEC = 0.5

    def __init__(self):
        super().__init__()
        self._stop_evt = threading.Event()
        self._audio_thread: Optional[threading.Thread] = None
        self._vad_thread: Optional[threading.Thread] = None
        self._is_robot_talking = False
        self._tts_ended_at: float = 0.0
        self._speech_queue: "queue.Queue[bytes]" = queue.Queue(maxsize=200)
        self._openai_client = None
        self.mic = None
        self.logger = UniversalLogger(enabled=False)

    @property
    def name(self) -> str:
        return "whisper_api"

    def set_logger(self, logger):
        super().set_logger(logger)
        self.logger = UniversalLogger(enabled=True, wrapped_logger=logger)

    def set_tts_playing(self, is_playing: bool):
        was_talking = self._is_robot_talking
        self._is_robot_talking = is_playing
        if was_talking and not is_playing:
            self._tts_ended_at = time.time()
            self._flush_queue()

    def _flush_queue(self):
        drained = 0
        if self.mic:
            while not self.mic.queue.empty():
                try:
                    self.mic.queue.get_nowait()
                    drained += 1
                except queue.Empty:
                    break
        if drained:
            self.logger.info(f"Flushed {drained} stale audio chunks after TTS")

    def _is_ducked(self) -> bool:
        if self._is_robot_talking:
            return True
        return (time.time() - self._tts_ended_at) < self.POST_TTS_GUARD_SEC

    def on_open(self):
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key and self.proxy and self.proxy.is_available():
            api_key = getattr(self.proxy, "_service_key", "") or ""

        if not api_key:
            self.logger.error("OPENAI_API_KEY not set — cannot start Whisper API input")
            return

        try:
            import openai
            self._openai_client = openai.OpenAI(api_key=api_key)
        except ImportError:
            self.logger.error("openai package not installed (pip install openai)")
            return

        detected_device = self._detect_audio_device()
        if not detected_device:
            detected_device = "default"

        self.logger.info(f"Using audio device: {detected_device}")

        try:
            from micro_input import ArecordStreamer
        except ImportError:
            self.logger.error("Could not import ArecordStreamer from micro_input")
            return

        self.mic = ArecordStreamer(self.logger)
        self.mic.start(
            device=detected_device,
            sample_rate=SAMPLE_RATE,
            channels=CHANNELS,
        )
        self.logger.info(f"Microphone started (rate={SAMPLE_RATE})")

        self._stop_evt.clear()
        self._vad_thread = threading.Thread(target=self._vad_loop, daemon=True)
        self._vad_thread.start()

    def _vad_loop(self):
        """Read mic chunks, do energy-based VAD, transcribe on silence."""
        self.logger.info("VAD loop started")
        speech_buf = bytearray()
        speaking = False
        silence_start: Optional[float] = None
        speech_start: Optional[float] = None

        chunk_bytes = int(SAMPLE_RATE * CHUNK_DURATION_SEC * CHANNELS * BYTES_PER_SAMPLE)

        while not self._stop_evt.is_set():
            try:
                chunk = self.mic.queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if self._is_ducked():
                speech_buf.clear()
                speaking = False
                silence_start = None
                continue

            rms = _rms_int16(chunk)
            now = time.time()

            if rms > SILENCE_THRESHOLD:
                if not speaking:
                    speaking = True
                    speech_start = now
                    self.logger.info("Speech detected")
                silence_start = None
                speech_buf.extend(chunk)

                if speech_start and (now - speech_start) >= MAX_SPEECH_SEC:
                    self.logger.info(f"Max speech duration reached ({MAX_SPEECH_SEC}s), flushing")
                    self._maybe_transcribe(bytes(speech_buf), speech_start, now)
                    speech_buf.clear()
                    speaking = False
                    silence_start = None
                    speech_start = None
            elif speaking:
                speech_buf.extend(chunk)
                if silence_start is None:
                    silence_start = now
                elif (now - silence_start) >= SILENCE_TIMEOUT_SEC:
                    duration = now - (speech_start or now)
                    self.logger.info(f"Speech ended ({duration:.1f}s)")
                    self._maybe_transcribe(bytes(speech_buf), speech_start, now)
                    speech_buf.clear()
                    speaking = False
                    silence_start = None
                    speech_start = None

    def _maybe_transcribe(self, pcm: bytes, start: Optional[float], end: float):
        duration = len(pcm) / (SAMPLE_RATE * CHANNELS * BYTES_PER_SAMPLE)
        if duration < MIN_SPEECH_SEC:
            self.logger.info(f"Speech too short ({duration:.2f}s), skipping")
            return

        if os.environ.get("DEBUG_AUDIO") == "1":
            ts = int(time.time())
            path = f"/tmp/whisper_debug_{ts}.wav"
            with open(path, "wb") as f:
                f.write(_pcm_to_wav(pcm))
            self.logger.info(f"Debug audio saved: {path}")

        threading.Thread(
            target=self._transcribe_worker,
            args=(pcm,),
            daemon=True,
        ).start()

    def _transcribe_worker(self, pcm: bytes):
        if not self._openai_client:
            return
        wav_data = _pcm_to_wav(pcm)
        wav_file = io.BytesIO(wav_data)
        wav_file.name = "speech.wav"

        try:
            t0 = time.time()
            result = self._openai_client.audio.transcriptions.create(
                model="whisper-1",
                file=wav_file,
                language="en",
                response_format="text",
            )
            elapsed = time.time() - t0
            text = str(result).strip()

            if not text:
                self.logger.info(f"Whisper returned empty ({elapsed:.1f}s)")
                return

            self.logger.info(f"Transcript ({elapsed:.1f}s): {text}")
            if self.is_active():
                self.send_data(text, data_type="chat_in")
        except Exception as e:
            self.logger.error(f"Whisper API error: {e}")

    def on_close(self):
        self._stop_evt.set()
        if self._vad_thread:
            self._vad_thread.join(timeout=2.0)
        if self.mic:
            try:
                self.mic.stop()
            except Exception:
                pass
            self.mic = None
        self._openai_client = None

    def _detect_audio_device(self) -> Optional[str]:
        """Reuse micro_input's device detection logic."""
        import subprocess
        import re
        devices = []
        try:
            result = subprocess.run(
                ["arecord", "-l"], capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                pattern = r"card (\d+):.*?\[([^\]]+)\].*?device (\d+):"
                for match in re.finditer(pattern, result.stdout):
                    card_num = match.group(1)
                    card_name = match.group(2)
                    device_num = match.group(3)
                    device_id = f"plughw:{card_num},{device_num}"
                    devices.append(
                        {"card": card_num, "device": device_num, "name": card_name, "id": device_id}
                    )
        except Exception:
            pass

        self.logger.info(f"Found {len(devices)} audio devices: {[d['name'] for d in devices]}")

        preferred = None
        for dev in devices:
            nl = dev["name"].lower()
            if "mic" in nl and "usb" in nl:
                preferred = dev
                break
        if not preferred:
            for dev in devices:
                nl = dev["name"].lower()
                if "sound" in nl and ("usb" in nl or "pnp" in nl):
                    preferred = dev
                    break
        if not preferred:
            for dev in devices:
                if "mic" in dev["name"].lower():
                    preferred = dev
                    break
        if not preferred:
            for dev in devices:
                nl = dev["name"].lower()
                if "usb" in nl and "camera" not in nl and "webcam" not in nl:
                    preferred = dev
                    break
        if not preferred and devices:
            preferred = devices[0]

        if preferred:
            self.logger.info(f"Selected audio device: {preferred['name']} ({preferred['id']})")
            return preferred["id"]
        return None
