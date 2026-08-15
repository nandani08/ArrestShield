# Detection module initialization
from .dataset import (
    MultitaskScamDataset,
    create_multitask_dataloaders,
    TRIGGER_NAMES,
    STAGE_MAP
)

__all__ = [
    "MultitaskScamDataset",
    "create_multitask_dataloaders",
    "TRIGGER_NAMES",
    "STAGE_MAP"
]
