import unittest

# pyrefly: ignore [missing-import]
import numpy as np
from src.asr.vad import VoiceActivityDetector

class TestVoiceActivityDetector(unittest.TestCase):
    def test_init_and_process_silence(self):
        print("Initializing VoiceActivityDetector...")
        detector = VoiceActivityDetector(sample_rate=16000, threshold=0.5)
        
        print("VAD loaded successfully. Simulating silence...")
        # Create silent chunks of 512 samples
        chunk_size = 512
        silent_chunk = np.zeros(chunk_size, dtype=np.float32)
        
        # Feed 10 silent chunks (approx 320ms) and verify no crash
        for i in range(10):
            res = detector.process_chunk(silent_chunk)
            # Silence shouldn't trigger speech start/end events in a zero signal
            if res:
                print(f"Triggered event on silence: {res}")
                
        print("Resetting detector...")
        detector.reset()
        print("VAD test passed successfully.")

if __name__ == "__main__":
    unittest.main()
