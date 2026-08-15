import time
import logging
import numpy as np
from typing import List, Dict, Any, Optional, Union

from src.asr.vad import VoiceActivityDetector
from src.asr.asr import HinglishTranscriber
from src.config import settings

logger = logging.getLogger("ArrestShield.StreamingASR")


class StreamingASRProcessor:
    """
    Real-Time Incremental Streaming ASR Processor.
    
    Accepts chunked binary PCM audio data from client stream (e.g. WebSocket),
    runs stateful Voice Activity Detection (Silero VAD), maintains sliding audio
    context window, transcribes speech incrementally under target latency (<800ms),
    and returns structured text segments.
    """
    def __init__(
        self,
        vad_detector: Optional[VoiceActivityDetector] = None,
        transcriber: Optional[HinglishTranscriber] = None,
        sample_rate: int = 16000,
        window_duration_sec: float = 2.0,
        latency_target_ms: int = 800,
        min_speech_duration_sec: float = 0.4
    ):
        self.sample_rate = sample_rate
        self.window_duration_sec = window_duration_sec
        self.max_window_samples = int(sample_rate * window_duration_sec)
        self.min_speech_samples = int(sample_rate * min_speech_duration_sec)
        self.latency_target_ms = latency_target_ms

        self.vad = vad_detector or VoiceActivityDetector(sample_rate=sample_rate)
        self.transcriber = transcriber or HinglishTranscriber()

        # Audio sample buffers
        self.audio_samples_buffer = np.array([], dtype=np.float32)
        self.speech_buffer = np.array([], dtype=np.float32)
        self.is_in_speech = False
        self.total_samples_processed = 0
        self.last_transcribe_time = 0.0

    def reset(self):
        """Resets stream processing buffers and VAD iterator state."""
        self.audio_samples_buffer = np.array([], dtype=np.float32)
        self.speech_buffer = np.array([], dtype=np.float32)
        self.is_in_speech = False
        self.total_samples_processed = 0
        self.last_transcribe_time = 0.0
        if self.vad:
            self.vad.reset()

    def decode_chunk(
        self,
        binary_chunk: bytes,
        dtype: Union[type, str] = np.int16
    ) -> np.ndarray:
        """
        Decodes raw binary PCM bytes into float32 numpy array normalized to [-1.0, 1.0].
        
        Supports 16-bit signed PCM (np.int16) and 32-bit float PCM (np.float32).
        """
        if not binary_chunk:
            return np.array([], dtype=np.float32)

        if dtype == np.int16 or dtype == "int16":
            # Trim odd byte if chunk is not aligned to 2-byte int16 boundary
            remainder = len(binary_chunk) % 2
            if remainder != 0:
                binary_chunk = binary_chunk[:-remainder]
            if not binary_chunk:
                return np.array([], dtype=np.float32)
            int16_samples = np.frombuffer(binary_chunk, dtype=np.int16)
            return int16_samples.astype(np.float32) / 32768.0

        elif dtype == np.float32 or dtype == "float32":
            remainder = len(binary_chunk) % 4
            if remainder != 0:
                binary_chunk = binary_chunk[:-remainder]
            if not binary_chunk:
                return np.array([], dtype=np.float32)
            return np.frombuffer(binary_chunk, dtype=np.float32)

        else:
            raise ValueError(f"Unsupported PCM audio dtype: {dtype}")

    def process_audio_chunk(
        self,
        binary_chunk: bytes,
        dtype: Union[type, str] = np.int16,
        language: str = "hi"
    ) -> List[Dict[str, Any]]:
        """
        Processes an incoming binary chunk of PCM audio.
        
        Workflow:
        1. Decode bytes into normalized float32 samples.
        2. Append to audio buffer and slice into 512-sample frames for VAD.
        3. Detect speech activity transitions (speech start / speech end).
        4. Accumulate active speech audio in sliding speech buffer.
        5. Trigger incremental transcription when speech is active and audio duration reaches threshold.
        6. Measure end-to-end processing latency.
        7. Return list of transcript payloads.
        """
        start_time = time.perf_counter()
        results: List[Dict[str, Any]] = []

        new_samples = self.decode_chunk(binary_chunk, dtype=dtype)
        if len(new_samples) == 0:
            return results

        self.audio_samples_buffer = np.concatenate([self.audio_samples_buffer, new_samples])

        # VAD processes 512 samples per frame at 16000Hz (32ms)
        frame_size = 512
        while len(self.audio_samples_buffer) >= frame_size:
            frame = self.audio_samples_buffer[:frame_size]
            self.audio_samples_buffer = self.audio_samples_buffer[frame_size:]

            vad_res = self.vad.process_chunk(frame)
            
            if vad_res is not None:
                if "start" in vad_res:
                    self.is_in_speech = True
                    logger.debug(f"VAD Speech Start detected at sample {self.total_samples_processed}")
                if "end" in vad_res:
                    self.is_in_speech = False
                    logger.debug(f"VAD Speech End detected at sample {self.total_samples_processed}")

                    # End of speech segment detected: flush and transcribe current speech buffer
                    if len(self.speech_buffer) >= self.min_speech_samples:
                        text, segments = self.transcriber.transcribe(self.speech_buffer, language=language)
                        latency_ms = (time.perf_counter() - start_time) * 1000.0
                        
                        if text:
                            results.append({
                                "type": "transcript",
                                "text": text,
                                "is_final": True,
                                "speech_detected": True,
                                "latency_ms": round(latency_ms, 2),
                                "segments": segments
                            })
                        self.speech_buffer = np.array([], dtype=np.float32)

            # Accumulate frames into speech buffer during active speech or continuously for sliding window
            self.speech_buffer = np.concatenate([self.speech_buffer, frame])
            
            # Enforce max window size for sliding window to prevent unbounded memory growth
            if len(self.speech_buffer) > self.max_window_samples:
                self.speech_buffer = self.speech_buffer[-self.max_window_samples:]

            self.total_samples_processed += frame_size

        # Perform incremental transcription if we have enough speech samples and haven't transcribed on speech_end
        if len(self.speech_buffer) >= self.min_speech_samples and len(results) == 0:
            text, segments = self.transcriber.transcribe(self.speech_buffer, language=language)
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            
            if text:
                is_final_flag = not self.is_in_speech
                results.append({
                    "type": "transcript",
                    "text": text,
                    "is_final": is_final_flag,
                    "speech_detected": self.is_in_speech,
                    "latency_ms": round(latency_ms, 2),
                    "segments": segments
                })
                # If window reaches max duration, slide the buffer by retaining overlap
                if len(self.speech_buffer) >= self.max_window_samples:
                    overlap_samples = self.max_window_samples // 4
                    self.speech_buffer = self.speech_buffer[-overlap_samples:]

        return results

    def flush(self, language: str = "hi") -> List[Dict[str, Any]]:
        """
        Flushes all pending audio in speech buffer and returns final transcription.
        """
        start_time = time.perf_counter()
        results: List[Dict[str, Any]] = []

        if len(self.speech_buffer) > 0:
            text, segments = self.transcriber.transcribe(self.speech_buffer, language=language)
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            if text:
                results.append({
                    "type": "transcript",
                    "text": text,
                    "is_final": True,
                    "speech_detected": self.is_in_speech,
                    "latency_ms": round(latency_ms, 2),
                    "segments": segments
                })
        self.reset()
        return results
