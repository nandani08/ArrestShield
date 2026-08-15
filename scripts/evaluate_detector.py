import os
import argparse
import logging
import torch
import numpy as np
from typing import Dict, Any

from src.detection.dataset import create_multitask_dataloaders, TRIGGER_NAMES, STAGE_MAP
from src.detection.model import MultitaskMuRILDetector
from src.detection.train_detector import evaluate

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


def print_confusion_matrix(matrix: np.ndarray, labels: list, title: str):
    print(f"\n--- Confusion Matrix: {title} ---")
    header = f"{'True \\ Pred':<15}" + "".join([f"{l:>12}" for l in labels])
    print(header)
    print("-" * len(header))
    for i, row in enumerate(matrix):
        row_str = f"{labels[i]:<15}" + "".join([f"{val:>12}" for val in row])
        print(row_str)


def evaluate_model(
    data_path: str = "data/scam_dataset.json",
    model_path: str = "models/best_multitask_detector.pt",
    model_name: str = "dummy"
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading dataset from '{data_path}' for evaluation...")

    _, val_loader = create_multitask_dataloaders(
        data_path=data_path,
        tokenizer=model_name,
        batch_size=16,
        val_split=0.3
    )

    model = MultitaskMuRILDetector(model_name=model_name).to(device)

    if os.path.exists(model_path):
        print(f"Loading fine-tuned model weights from '{model_path}'...")
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict, strict=False)
    else:
        print(f"Note: No checkpoint found at '{model_path}'. Evaluating initialized baseline model.")

    val_loss, metrics = evaluate(model, val_loader, device)

    print("\n" + "=" * 55)
    print("      MULTITASK DETECTOR MODEL EVALUATION REPORT     ")
    print("=" * 55)
    print(f"Validation Loss:             {val_loss:.4f}")
    print(f"Head A - Scam Accuracy:       {metrics.get('scam_accuracy', 0.0)*100:.2f}%")
    print(f"Head A - Scam Precision:      {metrics.get('scam_precision', 0.0):.4f}")
    print(f"Head A - Scam Recall:         {metrics.get('scam_recall', 0.0):.4f}")
    print(f"Head A - Scam F1-Score:       {metrics.get('scam_f1', 0.0):.4f}")
    print(f"Head B - Triggers Accuracy:   {metrics.get('triggers_accuracy', 0.0)*100:.2f}%")
    print(f"Head C - Scam Stage Accuracy: {metrics.get('stage_accuracy', 0.0)*100:.2f}%")

    if "scam_cm" in metrics:
        print_confusion_matrix(metrics["scam_cm"], ["Legit (0)", "Scam (1)"], "Binary Scam Detection")

    if "stage_cm" in metrics:
        stage_labels = [STAGE_MAP[i] for i in range(len(STAGE_MAP))]
        print_confusion_matrix(metrics["stage_cm"], stage_labels, "Categorical Scam Stage")

    print("=" * 55 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Multitask MuRIL Scam Classifier")
    parser.add_argument("--data_path", type=str, default="data/scam_dataset.json")
    parser.add_argument("--model_path", type=str, default="models/best_multitask_detector.pt")
    parser.add_argument("--model_name", type=str, default="dummy")
    args = parser.parse_args()

    evaluate_model(data_path=args.data_path, model_path=args.model_path, model_name=args.model_name)
