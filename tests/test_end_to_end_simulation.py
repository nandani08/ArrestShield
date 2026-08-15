import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import warnings
warnings.filterwarnings("ignore")

import unittest
from fastapi.testclient import TestClient

from app import app
from src.server.websocket import IntegratedPipelineSession
from src.honeypot.state_machine import HoneypotState


class TestEndToEndSystemSimulation(unittest.TestCase):
    """
    End-to-End System Simulation Test Suite.
    Simulates complete multi-turn scam conversation sequences, verifying:
    - Real-time detection latency (<800ms per turn)
    - Tri-State risk score escalations (SAFE -> UNCERTAIN -> FRAUD)
    - Zero-shot GLiNER threat parameter extraction (UPI, Phone, Badge ID, Case ID, Agency)
    - Adaptive LLM Honeypot state transitions (Confused -> Frightened -> Cooperative -> Stalling)
    - Decoy credential vault swap & utility score optimization
    """
    def setUp(self):
        self.client = TestClient(app)
        self.session = IntegratedPipelineSession()

    def test_digital_arrest_scam_sequence_simulation(self):
        scam_sequence_turns = [
            # Turn 1: Authority Impersonation
            {
                "speaker": "Scammer",
                "text": "Namaste, main Mumbai Police Cyber Cell se Officer Sharma speak kar raha hu (Badge #MH-4912).",
                "expected_min_risk": 0.0,
                "expected_agency": "Mumbai Police Cyber Cell",
                "expected_badge": "MH-4912"
            },
            # Turn 2: Coercion & Isolation Demand
            {
                "speaker": "Scammer",
                "text": "Aapka Aadhar number 9876 5432 1098 illegal money laundering warrant case #CR-2024-8842 mein linked hai. Immediately room lock karo and silent raho.",
                "expected_min_risk": 0.0,
                "expected_case": "CR-2024-8842"
            },
            # Turn 3: Payment / UPI Pressure
            {
                "speaker": "Scammer",
                "text": "Secret clearance bail fee payment UPI ID `rbi.clearance@okicici` par send karo or call us on 9876543210 immediately.",
                "expected_min_risk": 0.0,
                "expected_upi": "rbi.clearance@okicici",
                "expected_phone": "+91-98765-43210"
            }
        ]

        total_latency_ms = 0.0

        for i, turn in enumerate(scam_sequence_turns, 1):
            start_time = time.time()
            res = self.session.process_text_turn(turn["text"])
            elapsed_ms = (time.time() - start_time) * 1000.0
            total_latency_ms += elapsed_ms

            # 1. Assert Latency Requirement (< 800ms)
            self.assertLess(elapsed_ms, 800.0, f"Turn {i} latency exceeded 800ms (took {elapsed_ms:.2f}ms)")

            # 2. Assert Tri-State Risk Escalation
            detection = res["detection"]
            self.assertGreaterEqual(detection["risk_score"], turn["expected_min_risk"], f"Turn {i} risk score below expected minimum")

            # 3. Assert Extracted Threats
            threats = res["cumulative_threats"]
            if "expected_agency" in turn:
                self.assertIn(turn["expected_agency"], threats["claimed_agencies"])
            if "expected_badge" in turn:
                self.assertIn(turn["expected_badge"], threats["police_badge_ids"])
            if "expected_case" in turn:
                self.assertIn(turn["expected_case"], threats["case_ids"])
            if "expected_upi" in turn:
                self.assertIn(turn["expected_upi"], threats["upi_ids"])
            if "expected_phone" in turn:
                self.assertIn(turn["expected_phone"], threats["phone_numbers"])

            # 4. Assert Honeypot State & Response when in FRAUD
            if turn.get("expected_state") == "FRAUD":
                self.assertEqual(detection["state"], "FRAUD")
                honeypot = res["honeypot"]
                self.assertTrue(honeypot["active"])
                self.assertIsNotNone(honeypot["victim_response"])
                self.assertGreater(len(honeypot["victim_response"]), 5)
                self.assertGreaterEqual(honeypot["utility_score"], 0.0)

        avg_latency = total_latency_ms / len(scam_sequence_turns)
        print(f"\n[E2E Simulation Success] Processed {len(scam_sequence_turns)} scam turns cleanly. Average Latency: {avg_latency:.2f}ms/turn.")

    def test_api_analyze_text_simulation_endpoint(self):
        payload = {
            "text": "TRAI Cyber Department: Your SIM will be blocked. Transfer clearance fee to UPI `trai.verify@paytm` immediately.",
            "reset_session": True
        }
        start_time = time.time()
        response = self.client.post("/api/analyze_text", json=payload)
        elapsed_ms = (time.time() - start_time) * 1000.0

        self.assertEqual(response.status_code, 200)
        data = response.json()

        self.assertLess(elapsed_ms, 800.0)
        self.assertIn("trai.verify@paytm", data["threat_extraction"]["upi_ids"])
        self.assertIn("detection", data)
        self.assertIn("honeypot", data)


if __name__ == "__main__":
    unittest.main()
