# Detection module initialization
from .dataset import (
    MultitaskScamDataset,
    create_multitask_dataloaders,
    TRIGGER_NAMES,
    STAGE_MAP
)
from .model import MultitaskMuRILDetector, DummyBackbone
from .train_detector import run_training, train_one_epoch, evaluate, EarlyStopping, calculate_metrics
from .tristate_detector import TriStateDetector, DetectionState

__all__ = [
    "MultitaskScamDataset",
    "create_multitask_dataloaders",
    "TRIGGER_NAMES",
    "STAGE_MAP",
    "MultitaskMuRILDetector",
    "DummyBackbone",
    "run_training",
    "train_one_epoch",
    "evaluate",
    "EarlyStopping",
    "calculate_metrics",
    "TriStateDetector",
    "DetectionState"
]
