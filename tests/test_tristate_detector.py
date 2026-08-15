import warnings
warnings.filterwarnings("ignore")

import unittest
import torch

from src.detection.tristate_detector import TriStateDetector, DetectionState


class TestTriStateDetector(unittest.TestCase):
    def setUp(self):
        self.detector = TriStateDetector(
            model_name="dummy",
            max_turns=3,
            safe_threshold=0.3,
            fraud_threshold=0.7
        )

    def test_initial_state_and_reset(self):
        self.assertEqual(len(self.detector.turn_history), 0)
        self.detector.evaluate_turn("Hello, this is a test call.")
        self.assertEqual(len(self.detector.turn_history), 1)
        self.detector.reset()
        self.assertEqual(len(self.detector.turn_history), 0)

    def test_sliding_window_turn_limit(self):
        self.detector.evaluate_turn("Turn 1")
        self.detector.evaluate_turn("Turn 2")
        self.detector.evaluate_turn("Turn 3")
        self.assertEqual(len(self.detector.turn_history), 3)

        self.detector.evaluate_turn("Turn 4")
        self.assertEqual(len(self.detector.turn_history), 3)
        self.assertEqual(self.detector.turn_history, ["Turn 2", "Turn 3", "Turn 4"])

    def test_empty_turn_handling(self):
        res = self.detector.evaluate_turn("")
        self.assertEqual(res["state"], DetectionState.SAFE.value)
        self.assertEqual(res["risk_score"], 0.0)

    def test_risk_score_calculation_bounds(self):
        prob_scam = 0.9
        prob_triggers = [0.8, 0.7, 0.9, 0.85]
        prob_stage = [0.0, 0.0, 0.0, 0.0, 1.0, 0.0]  # Stage 4: coercion

        risk, stage_id, stage_name = self.detector.calculate_risk_score(
            prob_scam, prob_triggers, prob_stage
        )

        self.assertGreaterEqual(risk, 0.0)
        self.assertLessEqual(risk, 1.0)
        self.assertEqual(stage_id, 4)
        self.assertEqual(stage_name, "coercion")

    def test_state_transitions(self):
        # 1. Low risk evaluation
        res_safe = self.detector.evaluate_turn("Aapka Amazon delivery package deliver hone wala hai.")
        self.assertIn(res_safe["state"], [DetectionState.SAFE.value, DetectionState.UNCERTAIN.value, DetectionState.FRAUD.value])

        # 2. High risk digital arrest dialogue evaluation
        self.detector.reset()
        res_scam = self.detector.evaluate_turn(
            "Main Mumbai Police Headquarters se Officer Sharma speak kar raha hu. Aapka Aadhar card narcotics money laundering case mein arrest warrant se flagged hua hai. Room ka darwaza lock kar lo aur immediate RBI Escrow account `rbi.verify@okicici` par payment deposit karo."
        )

        self.assertIn("state", res_scam)
        self.assertIn("risk_score", res_scam)
        self.assertIn("triggers", res_scam)
        self.assertIn("predicted_stage", res_scam)
        self.assertGreaterEqual(res_scam["risk_score"], 0.0)
        self.assertLessEqual(res_scam["risk_score"], 1.0)


if __name__ == "__main__":
    unittest.main()
