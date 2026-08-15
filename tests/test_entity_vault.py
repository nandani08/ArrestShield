import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import warnings
warnings.filterwarnings("ignore")

import unittest
from src.honeypot.entity_vault import SyntheticEntityVault


class TestSyntheticEntityVault(unittest.TestCase):
    def setUp(self):
        self.vault = SyntheticEntityVault()

    def test_get_decoy_credentials(self):
        creds = self.vault.get_decoy_credentials()

        self.assertIn("name", creds)
        self.assertIn("bank_name", creds)
        self.assertIn("account_number", creds)
        self.assertIn("ifsc_code", creds)
        self.assertIn("upi_id", creds)
        self.assertIn("decoy_otp", creds)

        self.assertEqual(len(creds["decoy_otp"]), 6)
        self.assertTrue(creds["upi_id"].endswith(("@okaxis", "@upi", "@paytm", "@ybl", "@icici")))

    def test_get_decoy_context_string(self):
        context_str = self.vault.get_decoy_context_string()
        self.assertIn("Name:", context_str)
        self.assertIn("Bank:", context_str)
        self.assertIn("UPI ID:", context_str)

    def test_sanitize_and_intercept_phone_and_upi(self):
        raw_text = "Please transfer money to upi user real.scammer@okicici or call 9812345678."
        sanitized, replacements = self.vault.sanitize_and_intercept(raw_text)

        self.assertNotIn("real.scammer@okicici", sanitized)
        self.assertNotIn("9812345678", sanitized)
        self.assertGreaterEqual(len(replacements), 2)
        self.assertIn("real.scammer@okicici", replacements)
        self.assertIn("9812345678", replacements)

    def test_sanitize_and_intercept_otp_and_aadhar(self):
        raw_text = "Your OTP code is 849201 for Aadhar 1234 5678 9012."
        sanitized, replacements = self.vault.sanitize_and_intercept(raw_text)

        self.assertNotIn("849201", sanitized)
        self.assertNotIn("1234 5678 9012", sanitized)
        self.assertIn("849201", replacements)
        self.assertIn("1234 5678 9012", replacements)

    def test_reset_active_decoys(self):
        initial_upi = self.vault.active_upi
        self.vault.reset_active_decoys()
        self.assertEqual(len(self.vault.intercept_log), 0)


if __name__ == "__main__":
    unittest.main()
