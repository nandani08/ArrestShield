import warnings
warnings.filterwarnings("ignore")

import unittest
import torch

from src.detection.model import MultitaskMuRILDetector, DummyBackbone
from src.detection.dataset import MultitaskScamDataset, DummyTokenizer


class TestMultitaskMuRILDetector(unittest.TestCase):
    def setUp(self):
        self.model = MultitaskMuRILDetector(
            model_name="dummy",
            num_triggers=4,
            num_stages=6
        )
        self.tokenizer = DummyTokenizer(max_length=64)

    def test_forward_logits_and_probabilities_shapes(self):
        batch_size = 4
        seq_len = 64

        input_ids = torch.randint(0, 1000, (batch_size, seq_len), dtype=torch.long)
        attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long)

        output = self.model(input_ids=input_ids, attention_mask=attention_mask)

        self.assertIn("logits_scam", output)
        self.assertIn("logits_triggers", output)
        self.assertIn("logits_stage", output)
        self.assertIn("prob_scam", output)
        self.assertIn("prob_triggers", output)
        self.assertIn("prob_stage", output)

        # Verify shapes
        self.assertEqual(output["logits_scam"].shape, (batch_size, 2))
        self.assertEqual(output["logits_triggers"].shape, (batch_size, 4))
        self.assertEqual(output["logits_stage"].shape, (batch_size, 6))

        self.assertEqual(output["prob_scam"].shape, (batch_size,))
        self.assertEqual(output["prob_triggers"].shape, (batch_size, 4))
        self.assertEqual(output["prob_stage"].shape, (batch_size, 6))

        # Check probability bounds
        self.assertTrue((output["prob_scam"] >= 0.0).all() and (output["prob_scam"] <= 1.0).all())
        self.assertTrue((output["prob_triggers"] >= 0.0).all() and (output["prob_triggers"] <= 1.0).all())
        
        stage_sum = torch.sum(output["prob_stage"], dim=-1)
        self.assertTrue(torch.allclose(stage_sum, torch.ones(batch_size), atol=1e-4))

    def test_forward_with_loss_computation(self):
        batch_size = 4
        seq_len = 64

        input_ids = torch.randint(0, 1000, (batch_size, seq_len), dtype=torch.long)
        attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long)

        is_scam_labels = torch.tensor([1, 0, 1, 0], dtype=torch.long)
        triggers_labels = torch.tensor([
            [1.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0]
        ], dtype=torch.float32)
        stage_labels = torch.tensor([4, 0, 2, 0], dtype=torch.long)

        output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            is_scam_labels=is_scam_labels,
            triggers_labels=triggers_labels,
            stage_labels=stage_labels
        )

        self.assertIn("loss", output)
        self.assertIn("loss_components", output)
        
        loss = output["loss"]
        self.assertIsInstance(loss, torch.Tensor)
        self.assertGreater(loss.item(), 0.0)

    def test_integration_with_dataset_batch(self):
        sample_data = [
            {
                "id": "test_001",
                "text": "Main Mumbai Police headquarters se Senior Inspector bol raha hu.",
                "is_scam": 1,
                "triggers": {"authority": 1, "urgency": 1, "isolation": 0, "payment_pressure": 0},
                "scam_stage": 1,
                "scam_stage_name": "impersonation"
            }
        ]
        dataset = MultitaskScamDataset(data=sample_data, tokenizer=self.tokenizer, max_length=64)
        batch_item = dataset[0]

        # Unsqueeze to simulate batch dimension of 1
        input_ids = batch_item["input_ids"].unsqueeze(0)
        attention_mask = batch_item["attention_mask"].unsqueeze(0)
        is_scam_labels = batch_item["is_scam"].unsqueeze(0)
        triggers_labels = batch_item["triggers"].unsqueeze(0)
        stage_labels = batch_item["scam_stage"].unsqueeze(0)

        output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            is_scam_labels=is_scam_labels,
            triggers_labels=triggers_labels,
            stage_labels=stage_labels
        )

        self.assertIn("loss", output)
        self.assertEqual(output["prob_scam"].shape, (1,))


if __name__ == "__main__":
    unittest.main()
