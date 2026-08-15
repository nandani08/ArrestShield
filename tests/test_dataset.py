import os
import unittest
import torch
from torch.utils.data import DataLoader

from src.detection.dataset import (
    MultitaskScamDataset,
    DummyTokenizer,
    create_multitask_dataloaders,
    TRIGGER_NAMES,
    STAGE_MAP
)


class TestMultitaskScamDataset(unittest.TestCase):
    def setUp(self):
        self.sample_data = [
            {
                "id": "test_001",
                "text": "Main Mumbai Cyber Cell se Officer bol raha hu. Emergency transfer karo.",
                "is_scam": 1,
                "triggers": {"authority": 1, "urgency": 1, "isolation": 0, "payment_pressure": 1},
                "scam_stage": 4,
                "scam_stage_name": "coercion"
            },
            {
                "id": "test_002",
                "text": "Hello sir, aapka Amazon delivery agent gate par hai.",
                "is_scam": 0,
                "triggers": {"authority": 0, "urgency": 0, "isolation": 0, "payment_pressure": 0},
                "scam_stage": 0,
                "scam_stage_name": "none"
            }
        ]
        self.tokenizer = DummyTokenizer(max_length=64)

    def test_dataset_item_structure(self):
        dataset = MultitaskScamDataset(data=self.sample_data, tokenizer=self.tokenizer, max_length=64)
        self.assertEqual(len(dataset), 2)

        item0 = dataset[0]
        self.assertIn("input_ids", item0)
        self.assertIn("attention_mask", item0)
        self.assertIn("token_type_ids", item0)
        self.assertIn("is_scam", item0)
        self.assertIn("triggers", item0)
        self.assertIn("scam_stage", item0)

        # Shapes & Dtypes verification
        self.assertEqual(item0["input_ids"].shape, (64,))
        self.assertEqual(item0["attention_mask"].shape, (64,))
        self.assertEqual(item0["token_type_ids"].shape, (64,))
        self.assertEqual(item0["is_scam"].item(), 1)
        self.assertEqual(item0["is_scam"].dtype, torch.long)
        self.assertEqual(item0["triggers"].shape, (4,))
        self.assertEqual(item0["triggers"].dtype, torch.float32)
        self.assertEqual(item0["triggers"].tolist(), [1.0, 1.0, 0.0, 1.0])
        self.assertEqual(item0["scam_stage"].item(), 4)

        item1 = dataset[1]
        self.assertEqual(item1["is_scam"].item(), 0)
        self.assertEqual(item1["triggers"].tolist(), [0.0, 0.0, 0.0, 0.0])
        self.assertEqual(item1["scam_stage"].item(), 0)

    def test_dataloaders_factory_with_dataset_file(self):
        dataset_path = "data/scam_dataset.json"
        if not os.path.exists(dataset_path):
            self.skipTest("data/scam_dataset.json not found")

        train_loader, val_loader = create_multitask_dataloaders(
            data_path=dataset_path,
            tokenizer=self.tokenizer,
            batch_size=16,
            val_split=0.2,
            max_length=64
        )

        # 200 total samples -> 160 train, 40 val
        self.assertEqual(len(train_loader.dataset), 160)
        self.assertEqual(len(val_loader.dataset), 40)

        # Test iteration over train loader
        batch = next(iter(train_loader))
        self.assertEqual(batch["input_ids"].shape[0], 16)
        self.assertEqual(batch["input_ids"].shape[1], 64)
        self.assertEqual(batch["is_scam"].shape, (16,))
        self.assertEqual(batch["triggers"].shape, (16, 4))
        self.assertEqual(batch["scam_stage"].shape, (16,))


if __name__ == "__main__":
    unittest.main()
