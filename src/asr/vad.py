import numpy as np
import torch
import warnings
from typing import List, Dict, Union, Optional

# Suppress warnings from PyTorch JIT serialization/load in python 3.14+
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=ResourceWarning)

from silero_vad import load_silero_vad, VADIterator

class VoiceActivityDetector:
    """
    Voice Activity Detection (VAD) wrapper using Silero VAD.
    Supports stateful chunk-by-chunk streaming VAD.
    """
    def __init__(
        self,
        sample_rate: int = 16000,
        threshold: float = 0.5,
        min_silence_duration_ms: int = 250,
        speech_pad_ms: int = 30
    ):
        self.sample_rate = sample_rate
        self.threshold = threshold
        
        # Disable gradient calculations for VAD model inference
        torch.set_grad_enabled(False)
        
        try:
            # Load the model using silero-vad's native loader (highly robust)
            self.model = load_silero_vad(onnx=False)
        except Exception as e:
            raise RuntimeError(f"Failed to load Silero VAD model: {e}")
            
        # Instantiate VADIterator for stateful chunk-by-chunk streaming VAD
        self.iterator = VADIterator(
            model=self.model,
            threshold=self.threshold,
            sampling_rate=self.sample_rate,
            min_silence_duration_ms=min_silence_duration_ms,
            speech_pad_ms=speech_pad_ms
        )
        
    def reset(self):
        """Resets the state of the VAD iterator."""
        self.iterator.reset_states()
        
    def process_chunk(self, chunk: Union[np.ndarray, torch.Tensor]) -> Optional[Dict[str, int]]:
        """
        Processes a single audio chunk.
        The chunk size for 16000 Hz must be 512, 1024, or 1536 samples.
        Returns a dict indicating speech state transition if detected:
            - {'start': sample_idx} when speech starts
            - {'end': sample_idx} when speech ends
            - None if no state change
        """
        # Convert numpy array to torch tensor if needed
        if isinstance(chunk, np.ndarray):
            if chunk.dtype != np.float32:
                chunk = chunk.astype(np.float32)
            if np.max(np.abs(chunk)) > 1.0:
                chunk = chunk / 32768.0
            tensor_chunk = torch.from_numpy(chunk)
        else:
            tensor_chunk = chunk
            
        # Call VADIterator
        res = self.iterator(tensor_chunk)
        return res
