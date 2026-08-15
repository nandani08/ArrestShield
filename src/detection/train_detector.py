import os
import time
import argparse
import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional, Tuple, List
from transformers import get_linear_schedule_with_warmup

from src.detection.dataset import create_multitask_dataloaders, TRIGGER_NAMES, STAGE_MAP
from src.detection.model import MultitaskMuRILDetector
from src.config import settings

logger = logging.getLogger("ArrestShield.TrainDetector")


class EarlyStopping:
    """
    Early Stopping callback to monitor validation loss and stop training when progress plateaus.
    Saves the best performing model checkpoint to disk.
    """
    def __init__(self, patience: int = 3, delta: float = 1e-4, save_path: str = "models/best_multitask_detector.pt"):
        self.patience = patience
        self.delta = delta
        self.save_path = save_path
        self.counter = 0
        self.best_loss = float("inf")
        self.early_stop = False

        os.makedirs(os.path.dirname(save_path), exist_ok=True)

    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
            torch.save(model.state_dict(), self.save_path)
            logger.info(f"Validation loss improved to {val_loss:.4f}. Model saved to '{self.save_path}'.")
            return False
        else:
            self.counter += 1
            logger.info(f"Validation loss did not improve ({val_loss:.4f} vs best {self.best_loss:.4f}). Patience counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
            return self.early_stop


def calculate_metrics(all_targets: Dict[str, List], all_preds: Dict[str, List]) -> Dict[str, Any]:
    """
    Computes accuracy, precision, recall, F1-scores, and confusion matrices for:
    - Head A: Binary Scam Classification
    - Head B: Multi-Label Psychological Triggers
    - Head C: Categorical Scam Stage Progression
    """
    import numpy as np
    metrics: Dict[str, Any] = {}

    # Head A: Binary Scam Classification (0 = Legit, 1 = Scam)
    scam_true = np.array(all_targets["scam"])
    scam_pred = np.array(all_preds["scam"])
    if len(scam_true) > 0:
        scam_acc = np.mean(scam_true == scam_pred)
        tp = np.sum((scam_true == 1) & (scam_pred == 1))
        fp = np.sum((scam_true == 0) & (scam_pred == 1))
        fn = np.sum((scam_true == 1) & (scam_pred == 0))
        tn = np.sum((scam_true == 0) & (scam_pred == 0))

        precision = tp / max(tp + fp, 1e-9)
        recall = tp / max(tp + fn, 1e-9)
        f1 = 2 * precision * recall / max(precision + recall, 1e-9)

        metrics["scam_accuracy"] = float(scam_acc)
        metrics["scam_precision"] = float(precision)
        metrics["scam_recall"] = float(recall)
        metrics["scam_f1"] = float(f1)
        metrics["scam_cm"] = np.array([[int(tn), int(fp)], [int(fn), int(tp)]])
    else:
        metrics["scam_accuracy"] = 0.0
        metrics["scam_precision"] = 0.0
        metrics["scam_recall"] = 0.0
        metrics["scam_f1"] = 0.0
        metrics["scam_cm"] = np.zeros((2, 2), dtype=int)

    # Head B: Multi-Label Psychological Triggers
    triggers_true = np.array(all_targets["triggers"])  # [N, 4]
    triggers_pred = (np.array(all_preds["triggers"]) >= 0.5).astype(int)  # [N, 4]
    if len(triggers_true) > 0:
        triggers_acc = np.mean(triggers_true == triggers_pred)
        metrics["triggers_accuracy"] = float(triggers_acc)
    else:
        metrics["triggers_accuracy"] = 0.0

    # Head C: Categorical Scam Stage Progression (0 to 5)
    stage_true = np.array(all_targets["stage"])
    stage_pred = np.array(all_preds["stage"])
    if len(stage_true) > 0:
        stage_acc = np.mean(stage_true == stage_pred)
        metrics["stage_accuracy"] = float(stage_acc)

        num_stages = 6
        stage_cm = np.zeros((num_stages, num_stages), dtype=int)
        for t, p in zip(stage_true, stage_pred):
            if 0 <= t < num_stages and 0 <= p < num_stages:
                stage_cm[t, p] += 1
        metrics["stage_cm"] = stage_cm
    else:
        metrics["stage_accuracy"] = 0.0
        metrics["stage_cm"] = np.zeros((6, 6), dtype=int)

    return metrics


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    device: torch.device
) -> Tuple[float, Dict[str, float]]:
    model.train()
    total_loss = 0.0
    loss_components_acc = {"scam": 0.0, "triggers": 0.0, "stage": 0.0}

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch.get("token_type_ids")
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(device)

        is_scam_labels = batch["is_scam"].to(device)
        triggers_labels = batch["triggers"].to(device)
        stage_labels = batch["scam_stage"].to(device)

        optimizer.zero_grad()

        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            is_scam_labels=is_scam_labels,
            triggers_labels=triggers_labels,
            stage_labels=stage_labels
        )

        loss = output["loss"]
        if loss.requires_grad:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            if scheduler:
                scheduler.step()

        total_loss += loss.item()
        if "loss_components" in output:
            for k, v in output["loss_components"].items():
                loss_components_acc[k] += v

    num_batches = max(len(dataloader), 1)
    avg_loss = total_loss / num_batches
    avg_components = {k: v / num_batches for k, v in loss_components_acc.items()}
    return avg_loss, avg_components


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device
) -> Tuple[float, Dict[str, float]]:
    model.eval()
    total_loss = 0.0
    all_targets: Dict[str, List[Any]] = {"scam": [], "triggers": [], "stage": []}
    all_preds: Dict[str, List[Any]] = {"scam": [], "triggers": [], "stage": []}

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch.get("token_type_ids")
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)

            is_scam_labels = batch["is_scam"].to(device)
            triggers_labels = batch["triggers"].to(device)
            stage_labels = batch["scam_stage"].to(device)

            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                is_scam_labels=is_scam_labels,
                triggers_labels=triggers_labels,
                stage_labels=stage_labels
            )

            total_loss += output["loss"].item()

            pred_scam = torch.argmax(output["logits_scam"], dim=-1).cpu().tolist()
            pred_stage = torch.argmax(output["logits_stage"], dim=-1).cpu().tolist()
            pred_triggers = output["prob_triggers"].cpu().tolist()

            all_preds["scam"].extend(pred_scam)
            all_preds["stage"].extend(pred_stage)
            all_preds["triggers"].extend(pred_triggers)

            all_targets["scam"].extend(is_scam_labels.cpu().tolist())
            all_targets["stage"].extend(stage_labels.cpu().tolist())
            all_targets["triggers"].extend(triggers_labels.cpu().tolist())

    num_batches = max(len(dataloader), 1)
    avg_loss = total_loss / num_batches
    metrics = calculate_metrics(all_targets, all_preds)
    metrics["val_loss"] = avg_loss
    return avg_loss, metrics


def run_training(
    dataset_path: str = "data/scam_dataset.json",
    model_name: str = "google/muril-base-cased",
    save_path: str = "models/best_multitask_detector.pt",
    epochs: int = 5,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    patience: int = 3,
    device_str: Optional[str] = None
) -> Dict[str, Any]:
    device = torch.device(device_str if device_str else ("cuda" if torch.cuda.is_available() else "cpu"))
    logger.info(f"Starting Multitask Detector Training on device '{device}'...")

    train_loader, val_loader = create_multitask_dataloaders(
        data_path=dataset_path,
        tokenizer=model_name,
        batch_size=batch_size,
        val_split=0.2
    )

    model = MultitaskMuRILDetector(model_name=model_name).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * 0.1),
        num_training_steps=max(total_steps, 1)
    )

    early_stopping = EarlyStopping(patience=patience, save_path=save_path)
    training_history = []

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss, train_components = train_one_epoch(model, train_loader, optimizer, scheduler, device)
        val_loss, val_metrics = evaluate(model, val_loader, device)
        elapsed = time.time() - t0

        logger.info(
            f"Epoch {epoch}/{epochs} ({elapsed:.1f}s) - "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Scam Acc: {val_metrics['scam_accuracy']:.2f} | "
            f"Triggers Acc: {val_metrics['triggers_accuracy']:.2f} | "
            f"Stage Acc: {val_metrics['stage_accuracy']:.2f}"
        )

        history_entry = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "metrics": val_metrics
        }
        training_history.append(history_entry)

        stop = early_stopping(val_loss, model)
        if stop:
            logger.info("Early stopping triggered. Training stopped.")
            break

    return {
        "status": "completed",
        "best_val_loss": early_stopping.best_loss,
        "checkpoint_path": save_path,
        "history": training_history
    }
