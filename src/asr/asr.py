import numpy as np
import torch
from faster_whisper import WhisperModel
from typing import Dict, Any, List, Tuple, Optional

class HinglishTranscriber:
    """
    ASR Transcriber utilizing faster-whisper and shunya-labs/zero-stt-hinglish
    to transcribe code-switched Indian languages (Hinglish, Hindi, English).
    """
    def __init__(
        self,
        model_name: str = "shunya-labs/zero-stt-hinglish",
        device: str = "cpu",
        compute_type: str = "float32",
        beam_size: int = 5
    ):
        self.model_name = model_name
        self.device = device
        
        # Fallback to CPU if CUDA is requested but not available
        if self.device == "cuda" and not torch.cuda.is_available():
            self.device = "cpu"
            
        self.compute_type = compute_type
        # Fallback to float32 on CPU if float16 is requested (faster-whisper CPU doesn't support float16)
        if self.device == "cpu" and self.compute_type == "float16":
            self.compute_type = "float32"
            
        self.beam_size = beam_size
        
        # Load the Whisper Model
        try:
            self.model = WhisperModel(
                model_size_or_path=self.model_name,
                device=self.device,
                compute_type=self.compute_type
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load Whisper model '{self.model_name}': {e}")
            
    def transcribe(
        self,
        audio: np.ndarray,
        language: Optional[str] = "hi",
        beam_size: Optional[int] = None,
        vad_filter: bool = False,
        **kwargs
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Transcribes the given audio numpy array.
        
        Parameters:
            audio: 1-D float32 numpy array normalized to [-1.0, 1.0].
            language: Language code to transcribe. Defaults to "hi" (Hindi/Hinglish).
            beam_size: Beam size override.
            vad_filter: Whether to apply VAD filter.
            
        Returns:
            Tuple of (full_transcript_text, segments_info)
        """
        if beam_size is None:
            beam_size = self.beam_size
            
        # Call faster-whisper transcribe
        segments, info = self.model.transcribe(
            audio,
            beam_size=beam_size,
            language=language,
            vad_filter=vad_filter,
            **kwargs
        )
        
        # Consume the generator to get segments
        segments_list = list(segments)
        
        # Build transcripts
        text_segments = []
        detailed_segments = []
        for segment in segments_list:
            text_segments.append(segment.text)
            detailed_segments.append({
                "start": segment.start,
                "end": segment.end,
                "text": segment.text,
                "probability": getattr(segment, "avg_logprob", 0.0)
            })
            
        full_text = "".join(text_segments).strip()
        return full_text, detailed_segments


class DummyHinglishTranscriber:
    """
    Fallback Transcriber for offline or test environments.
    Simulates speech recognition without requiring model downloads.
    """
    def __init__(self, *args, **kwargs):
        pass

    def transcribe(self, audio: np.ndarray, **kwargs) -> Tuple[str, List[Dict[str, Any]]]:
        return "Simulated ASR transcription.", [{"start": 0.0, "end": 1.0, "text": "Simulated ASR transcription.", "probability": 0.99}]
