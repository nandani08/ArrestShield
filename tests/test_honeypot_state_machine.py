import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import warnings
warnings.filterwarnings("ignore")

import unittest
from src.honeypot.state_machine import HoneypotStateMachine, HoneypotState


class TestHoneypotStateMachine(unittest.TestCase):
    def setUp(self):
        self.machine = HoneypotStateMachine(alpha_utility=1.0, beta_utility=0.5)

    def test_initial_state_and_reset(self):
        self.assertEqual(self.machine.current_state, HoneypotState.CONFUSED)
        self.assertEqual(self.machine.turn_count, 0)

        self.machine.generate_honeypot_turn("Namaste, main Mumbai Police se hu.")
        self.assertEqual(self.machine.turn_count, 1)

        self.machine.reset()
        self.assertEqual(self.machine.current_state, HoneypotState.CONFUSED)
        self.assertEqual(self.machine.turn_count, 0)
        self.assertEqual(len(self.machine.history), 0)

    def test_state_transition_progression(self):
        # Turn 1: Initial contact -> transitions to FRIGHTENED due to police authority claim
        res1 = self.machine.generate_honeypot_turn(
            "Mumbai Police Headquarters se Officer Sharma speak kar raha hu. Aapka Aadhar crime se link hai.",
            detection_result={"predicted_stage": "impersonation", "triggers": {"authority": 0.9, "urgency": 0.8}}
        )
        self.assertEqual(res1["state"], HoneypotState.FRIGHTENED.value)

        # Turn 2: Allegation & Verification request -> transitions to COOPERATIVE
        res2 = self.machine.generate_honeypot_turn(
            "Aapka account freeze hone wala hai. Abhi digital verification audit follow karo.",
            detection_result={"predicted_stage": "allegation", "triggers": {"authority": 0.9, "urgency": 0.9}}
        )
        self.assertIn(res2["state"], [HoneypotState.FRIGHTENED.value, HoneypotState.COOPERATIVE.value])

        # Turn 3: Demand for payment / UPI transfer -> transitions to STALLING
        res3 = self.machine.generate_honeypot_turn(
            "Abhi ₹50,000 RBI Escrow UPI ID `rbi.verify@okicici` par send karo.",
            detection_result={"predicted_stage": "payment", "triggers": {"payment_pressure": 0.95}}
        )
        self.assertEqual(res3["state"], HoneypotState.STALLING.value)

    def test_utility_score_calculation(self):
        self.assertEqual(self.machine.calculate_utility_score(new_extracted_count=2), 2.0)
        self.assertEqual(self.machine.calculate_utility_score(new_extracted_count=1, pii_leak_penalty=1.0), 2.5)

    def test_multi_turn_dialogue_history(self):
        turn1 = self.machine.generate_honeypot_turn("Who is this?")
        self.assertIn("victim_response", turn1)
        self.assertEqual(len(self.machine.history), 2)  # 1 user turn + 1 assistant turn

        turn2 = self.machine.generate_honeypot_turn("This is TRAI Department. Your mobile number will be blocked.")
        self.assertEqual(len(self.machine.history), 4)


if __name__ == "__main__":
    unittest.main()
