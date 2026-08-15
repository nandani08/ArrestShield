import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import warnings
warnings.filterwarnings("ignore")

import unittest
from src.extraction.gliner_extractor import GLiNERThreatExtractor, DummyThreatExtractor


class TestGLiNERThreatExtractor(unittest.TestCase):
    def setUp(self):
        self.extractor = GLiNERThreatExtractor(model_name="dummy")

    def test_empty_transcript_input(self):
        entities = self.extractor.extract_threat_entities("")
        self.assertEqual(entities, [])

    def test_threat_entity_extraction(self):
        transcript = (
            "Main Mumbai Police se Senior Inspector Sharma speak kar raha hu (Badge #MH-4912). "
            "Aapka case reference number #CR-2024-8842 hai. "
            "Abhi RBI Secret Clearance UPI ID `rbi.verify@okicici` par payment send karo. "
            "Call us immediately on 9876543210."
        )

        entities = self.extractor.extract_threat_entities(transcript)

        self.assertGreaterEqual(len(entities), 4)

        labels = [e["label"] for e in entities]
        texts = [e["text"] for e in entities]

        self.assertIn("UPI ID", labels)
        self.assertIn("phone number", labels)
        self.assertIn("police badge ID", labels)
        self.assertIn("claimed agency", labels)

        self.assertIn("rbi.verify@okicici", texts)
        self.assertIn("9876543210", texts)
        self.assertIn("MH-4912", texts)
        self.assertIn("Mumbai Police", texts)

    def test_dummy_threat_extractor_direct(self):
        dummy = DummyThreatExtractor()
        text = "Contact CBI Officer Badge #4912 or send to cbi.pay@paytm."
        res = dummy.predict_entities(text)
        self.assertGreaterEqual(len(res), 2)


if __name__ == "__main__":
    unittest.main()
