import warnings
warnings.filterwarnings("ignore")

import os
import unittest
import torch

from src.detection.train_detector import (
    EarlyStopping,
    calculate_metrics,
    train_one_epoch,
    evaluate,
    run_training
)
from src.detection.model import MultitaskMuRILDetector
from src.detection.dataset import create_multitask_dataloaders, DummyTokenizer


class TestTrainDetector(unittest.TestCase):
    def setUp(self):
        self.checkpoint_dir = "models"
        self.checkpoint_path = os.path.join(self.checkpoint_dir, "test_detector.pt")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def tearDown(self):
        if os.path.exists(self.checkpoint_path):
            try:
                os.remove(self.checkpoint_path)
            except Exception:
                pass

    def test_early_stopping(self):
        dummy_model = torch.nn.Linear(10, 2)
        early_stopping = EarlyStopping(patience=2, delta=1e-4, save_path=self.checkpoint_path)

        # 1. First evaluation: loss 0.5 (should improve and save model, returns False for stop)
        stop = early_stopping(0.5, dummy_model)
        self.assertFalse(stop)
        self.assertEqual(early_stopping.best_loss, 0.5)
        self.assertEqual(early_stopping.counter, 0)
        self.assertTrue(os.path.exists(self.checkpoint_path))

        # 2. Second evaluation: loss 0.6 (worse, counter -> 1, returns False)
        stop = early_stopping(0.6, dummy_model)
        self.assertFalse(stop)
        self.assertEqual(early_stopping.counter, 1)
        self.assertFalse(early_stopping.early_stop)

        # 3. Third evaluation: loss 0.65 (worse, counter -> 2 == patience -> returns True)
        stop = early_stopping(0.65, dummy_model)
        self.assertTrue(stop)
        self.assertEqual(early_stopping.counter, 2)
        self.assertTrue(early_stopping.early_stop)

    def test_calculate_metrics(self):
        all_targets = {
            "scam": [1, 0, 1, 0],
            "triggers": [[1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
            "stage": [4, 0]
        }
        all_preds = {
            "scam": [1, 0, 1, 1],  # 3/4 correct = 0.75
            "triggers": [[0.9, 0.8, 0.1, 0.1], [0.1, 0.1, 0.1, 0.1]],  # 8/8 correct = 1.0
            "stage": [4, 0]  # 2/2 correct = 1.0
        }

        metrics = calculate_metrics(all_targets, all_preds)
        self.assertIn("scam_accuracy", metrics)
        self.assertIn("triggers_accuracy", metrics)
        self.assertIn("stage_accuracy", metrics)

        self.assertEqual(metrics["scam_accuracy"], 0.75)
        self.assertEqual(metrics["triggers_accuracy"], 1.0)
        self.assertEqual(metrics["stage_accuracy"], 1.0)

    def test_run_training_execution(self):
        dataset_path = "data/scam_dataset.json"
        if not os.path.exists(dataset_path):
            self.skipTest("data/scam_dataset.json not found")

        result = run_training(
            dataset_path=dataset_path,
            model_name="dummy",
            save_path=self.checkpoint_path,
            epochs=2,
            batch_size=16,
            learning_rate=1e-3,
            patience=2,
            device_str="cpu"
        )

        self.assertEqual(result["status"], "completed")
        self.assertIn("best_val_loss", result)
        self.assertIn("checkpoint_path", result)
        self.assertEqual(len(result["history"]), 2)
        self.assertTrue(os.path.exists(self.checkpoint_path))


if __name__ == "__main__":
    unittest.main()
