import unittest
from unittest.mock import MagicMock, patch
import numpy as np

from src.asr.asr import HinglishTranscriber

class TestHinglishTranscriber(unittest.TestCase):
    @patch("src.asr.asr.WhisperModel")
    def test_transcriber_initialization_and_mock_transcribe(self, mock_whisper_model_class):
        # Setup mock instances
        mock_model_instance = MagicMock()
        mock_whisper_model_class.return_value = mock_model_instance
        
        # Setup mock segment
        mock_segment = MagicMock()
        mock_segment.text = "Hello, kaise ho aap?"
        mock_segment.start = 0.0
        mock_segment.end = 2.0
        mock_segment.avg_logprob = -0.15
        
        # Setup mock transcribe return value: (segments_generator/iterator, info)
        mock_model_instance.transcribe.return_value = ([mock_segment], MagicMock())
        
        # Instantiate HinglishTranscriber
        transcriber = HinglishTranscriber(
            model_name="shunya-labs/zero-stt-hinglish",
            device="cpu",
            compute_type="float32"
        )
        
        # Check call to WhisperModel constructor
        mock_whisper_model_class.assert_called_once_with(
            model_size_or_path="shunya-labs/zero-stt-hinglish",
            device="cpu",
            compute_type="float32"
        )
        
        # Simulate dummy audio array (e.g. 1 sec of silence at 16kHz)
        dummy_audio = np.zeros(16000, dtype=np.float32)
        
        # Run transcribe
        text, segments = transcriber.transcribe(dummy_audio)
        
        # Verify text and segment details
        self.assertEqual(text, "Hello, kaise ho aap?")
        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0]["text"], "Hello, kaise ho aap?")
        self.assertEqual(segments[0]["start"], 0.0)
        self.assertEqual(segments[0]["end"], 2.0)
        self.assertEqual(segments[0]["probability"], -0.15)
        
        # Check that model.transcribe was called
        mock_model_instance.transcribe.assert_called_once()

if __name__ == "__main__":
    unittest.main()
