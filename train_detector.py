import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import logging
from src.detection import run_training

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

def main():
    parser = argparse.ArgumentParser(description="Train Multitask MuRIL Scam Classifier")
    parser.add_argument("--dataset_path", type=str, default="data/scam_dataset.json", help="Path to annotated JSON dataset")
    parser.add_argument("--model_name", type=str, default="google/muril-base-cased", help="Backbone model name or 'dummy'")
    parser.add_argument("--save_path", type=str, default="models/best_multitask_detector.pt", help="Path to save checkpoint")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--patience", type=int, default=3, help="Early stopping patience")
    args = parser.parse_args()

    result = run_training(
        dataset_path=args.dataset_path,
        model_name=args.model_name,
        save_path=args.save_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        patience=args.patience
    )
    print("\nTraining Finished!")
    print(f"Status: {result['status']}")
    print(f"Best Validation Loss: {result['best_val_loss']:.4f}")
    print(f"Model Checkpoint: {result['checkpoint_path']}")

if __name__ == "__main__":
    main()
