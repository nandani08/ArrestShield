import warnings
warnings.filterwarnings("ignore")

import unittest
from unittest.mock import patch, MagicMock

from src.honeypot.prompts import build_honeypot_prompt, VICTIM_SYSTEM_PROMPT, STAGE_PROMPTS
from src.honeypot.llm_client import HoneypotLLMClient


class TestHoneypotLLM(unittest.TestCase):
    def test_build_honeypot_prompt(self):
        messages = build_honeypot_prompt(
            stage="frightened",
            conversation_history=[
                {"role": "user", "content": "Aapka arrest warrant nikal raha hai!"}
            ],
            decoy_context="Decoy UPI: fake.rbi@okicici"
        )

        self.assertGreaterEqual(len(messages), 2)
        self.assertEqual(messages[0]["role"], "system")
        self.assertIn(VICTIM_SYSTEM_PROMPT, messages[0]["content"])
        self.assertIn("FRIGHTENED", messages[0]["content"])
        self.assertIn("fake.rbi@okicici", messages[0]["content"])
        self.assertEqual(messages[1]["content"], "Aapka arrest warrant nikal raha hai!")

    def test_mock_fallback_generation(self):
        client = HoneypotLLMClient(api_url="http://localhost:9999/v1")
        for stage in ["confused", "frightened", "cooperative", "stalling"]:
            response = client.generate_mock_fallback(stage=stage)
            self.assertIsInstance(response, str)
            self.assertGreater(len(response), 10)

    def test_generate_response_offline_fallback(self):
        # Point to invalid port to test graceful fallback
        client = HoneypotLLMClient(api_url="http://localhost:59999/v1", timeout_sec=0.5)
        response = client.generate_response(
            conversation_history=[{"role": "user", "content": "Police ko rishwat do!"}],
            stage="cooperative"
        )
        self.assertIsInstance(response, str)
        self.assertGreater(len(response), 5)

    @patch("urllib.request.urlopen")
    def test_generate_response_http_success(self, mock_urlopen):
        import json
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.read.return_value = json.dumps({
            "choices": [
                {"message": {"content": "Ji sir, main dukan par hu, dar lag raha hai."}}
            ]
        }).encode("utf-8")
        mock_urlopen.return_value.__enter__.return_value = mock_resp

        client = HoneypotLLMClient(api_url="http://localhost:11434/v1")
        response = client.generate_response(
            conversation_history=[{"role": "user", "content": "Aapka name kya hai?"}],
            stage="confused"
        )
        self.assertEqual(response, "Ji sir, main dukan par hu, dar lag raha hai.")


if __name__ == "__main__":
    unittest.main()
