import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import warnings
warnings.filterwarnings("ignore")

import unittest
from src.extraction.post_processor import ThreatPostProcessor


class TestThreatPostProcessor(unittest.TestCase):
    def setUp(self):
        self.processor = ThreatPostProcessor()

    def test_validate_upi(self):
        self.assertEqual(self.processor.validate_upi("rbi.clearance@okicici"), "rbi.clearance@okicici")
        self.assertEqual(self.processor.validate_upi("`cbi.verify@paytm`."), "cbi.verify@paytm")
        self.assertIsNone(self.processor.validate_upi("invalid_upi_without_at"))
        self.assertIsNone(self.processor.validate_upi("a@b"))

    def test_validate_phone_number(self):
        self.assertEqual(self.processor.validate_phone_number("9876543210"), "+91-98765-43210")
        self.assertEqual(self.processor.validate_phone_number("+91 98765 43210"), "+91-98765-43210")
        self.assertEqual(self.processor.validate_phone_number("09876543210"), "+91-98765-43210")
        self.assertIsNone(self.processor.validate_phone_number("123456"))  # Invalid digit length
        self.assertIsNone(self.processor.validate_phone_number("1876543210"))  # Does not start with 6-9

    def test_validate_url(self):
        text = "Visit http://rbi-verify-clearance.com for verification or email support@trai.gov.in."
        urls = self.processor.validate_url(text)
        self.assertEqual(urls, ["http://rbi-verify-clearance.com"])

    def test_clean_identifier(self):
        self.assertEqual(self.processor.clean_identifier("Badge #MH-4912."), "MH-4912")
        self.assertEqual(self.processor.clean_identifier("Case #CR-2024-8842"), "CR-2024-8842")

    def test_clean_agency(self):
        self.assertEqual(self.processor.clean_agency("mumbai police"), "Mumbai Police Cyber Cell")
        self.assertEqual(self.processor.clean_agency("CBI"), "CBI Cyber Crime Division")

    def test_process_extracted_threats_end_to_end(self):
        raw_entities = [
            {"text": "rbi.clearance@okicici", "label": "UPI ID"},
            {"text": "9876543210", "label": "phone number"},
            {"text": "Badge #MH-4912", "label": "police badge ID"},
            {"text": "mumbai police", "label": "claimed agency"},
            {"text": "Case #CR-2024-8842", "label": "case ID"}
        ]

        text = "Visit http://cybercell-mumbai.in for details."
        report = self.processor.process_extracted_threats(raw_entities, transcript_text=text)

        self.assertEqual(report["upi_ids"], ["rbi.clearance@okicici"])
        self.assertEqual(report["phone_numbers"], ["+91-98765-43210"])
        self.assertEqual(report["police_badge_ids"], ["MH-4912"])
        self.assertEqual(report["case_ids"], ["CR-2024-8842"])
        self.assertEqual(report["claimed_agencies"], ["Mumbai Police Cyber Cell"])
        self.assertEqual(report["urls"], ["http://cybercell-mumbai.in"])
        self.assertEqual(report["total_valid_threat_indicators"], 6)


if __name__ == "__main__":
    unittest.main()
