import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import warnings
warnings.filterwarnings("ignore")

import unittest
from fastapi.testclient import TestClient

from app import app


class TestServerApp(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_root_endpoint(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)

    def test_system_status_endpoint(self):
        response = self.client.get("/api/status")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "healthy")
        self.assertIn("config", data)

    def test_decoy_credentials_endpoint(self):
        response = self.client.get("/api/decoy_credentials")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("bank_name", data)
        self.assertIn("upi_id", data)

    def test_analyze_text_endpoint(self):
        payload = {
            "text": "Main Mumbai Police Cyber Cell se Officer Sharma speak kar raha hu. Payment UPI rbi.verify@okicici par send karo.",
            "reset_session": True
        }
        response = self.client.post("/api/analyze_text", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()

        self.assertEqual(data["type"], "analysis_turn")
        self.assertIn("detection", data)
        self.assertIn("threat_extraction", data)
        self.assertIn("honeypot", data)

        self.assertIn(data["detection"]["state"], ["SAFE", "UNCERTAIN", "FRAUD"])
        self.assertIn("rbi.verify@okicici", data["threat_extraction"]["upi_ids"])


if __name__ == "__main__":
    unittest.main()
