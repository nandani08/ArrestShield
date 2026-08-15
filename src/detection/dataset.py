import json
import logging
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from typing import List, Dict, Any, Union, Tuple, Optional
from transformers import AutoTokenizer, PreTrainedTokenizerBase

logger = logging.getLogger("ArrestShield.Dataset")

TRIGGER_NAMES = ["authority", "urgency", "isolation", "payment_pressure"]
STAGE_MAP = {
    0: "none",
    1: "impersonation",
    2: "allegation",
    3: "isolation",
    4: "coercion",
    5: "payment"
}


class DummyTokenizer:
    """
    Fallback Tokenizer for offline or test environments.
    Converts text to basic token ID tensors compatible with BERT models.
    """
    def __init__(self, max_length: int = 128, vocab_size: int = 30522):
        self.max_length = max_length
        self.vocab_size = vocab_size

    def __call__(
        self,
        text: Union[str, List[str]],
        max_length: Optional[int] = None,
        padding: str = "max_length",
        truncation: bool = True,
        return_tensors: str = "pt",
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        if max_length is None:
            max_length = self.max_length

        if isinstance(text, str):
            words = text.split()
        else:
            words = [w for t in text for w in t.split()]

        tokens = [101] + [hash(w) % (self.vocab_size - 2) + 1 for w in words[: max_length - 2]] + [102]
        pad_len = max(0, max_length - len(tokens))
        input_ids = tokens + [0] * pad_len
        attention_mask = [1] * len(tokens) + [0] * pad_len
        token_type_ids = [0] * max_length

        input_ids_tensor = torch.tensor([input_ids[:max_length]], dtype=torch.long)
        attention_mask_tensor = torch.tensor([attention_mask[:max_length]], dtype=torch.long)
        token_type_ids_tensor = torch.tensor([token_type_ids[:max_length]], dtype=torch.long)

        return {
            "input_ids": input_ids_tensor,
            "attention_mask": attention_mask_tensor,
            "token_type_ids": token_type_ids_tensor
        }


class MultitaskScamDataset(Dataset):
    """
    PyTorch Dataset for multi-task scam detection using MuRIL / BERT backbone.
    
    Loads Hinglish conversation samples with 3 prediction targets:
    1. Head A (is_scam): Binary classification (0: Legit, 1: Scam)
    2. Head B (triggers): Multi-label sigmoid binary vector [authority, urgency, isolation, payment_pressure]
    3. Head C (scam_stage): Categorical stage classification (0: None, 1: Impersonation, 2: Allegation, 3: Isolation, 4: Coercion, 5: Payment)
    """
    def __init__(
        self,
        data: Union[str, List[Dict[str, Any]]],
        tokenizer: Union[str, PreTrainedTokenizerBase, Any] = "google/muril-base-cased",
        max_length: int = 128
    ):
        if isinstance(data, str):
            with open(data, "r", encoding="utf-8") as f:
                self.samples = json.load(f)
        else:
            self.samples = data

        self.max_length = max_length

        # Set or load tokenizer
        if isinstance(tokenizer, str):
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
            except Exception as e:
                logger.warning(f"Could not load AutoTokenizer '{tokenizer}' online ({e}). Attempting local fallback.")
                try:
                    self.tokenizer = AutoTokenizer.from_pretrained(tokenizer, local_files_only=True)
                except Exception:
                    logger.warning(f"Local tokenizer fallback failed. Initializing DummyTokenizer for testing.")
                    self.tokenizer = DummyTokenizer(max_length=max_length)
        else:
            self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        text = sample.get("text", "")

        # Tokenize text input
        encoded = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        input_ids = encoded["input_ids"].squeeze(0)
        attention_mask = encoded["attention_mask"].squeeze(0)
        token_type_ids = encoded.get("token_type_ids", torch.zeros_like(input_ids)).squeeze(0)

        # Head A: Binary scam label (0 or 1)
        is_scam_label = torch.tensor(sample.get("is_scam", 0), dtype=torch.long)

        # Head B: Multi-label trigger binary vector [authority, urgency, isolation, payment_pressure]
        triggers_dict = sample.get("triggers", {})
        trigger_vector = [float(triggers_dict.get(k, 0)) for k in TRIGGER_NAMES]
        triggers_label = torch.tensor(trigger_vector, dtype=torch.float32)

        # Head C: Categorical scam stage (0..5)
        stage_label = torch.tensor(sample.get("scam_stage", 0), dtype=torch.long)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "is_scam": is_scam_label,
            "triggers": triggers_label,
            "scam_stage": stage_label
        }


def create_multitask_dataloaders(
    data_path: str = "data/scam_dataset.json",
    tokenizer: Union[str, PreTrainedTokenizerBase, Any] = "google/muril-base-cased",
    batch_size: int = 16,
    val_split: float = 0.2,
    max_length: int = 128,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader]:
    """
    Constructs train and validation DataLoaders from the synthetic JSON dataset.
    """
    dataset = MultitaskScamDataset(
        data=data_path,
        tokenizer=tokenizer,
        max_length=max_length
    )

    total_len = len(dataset)
    val_len = int(total_len * val_split)
    train_len = total_len - val_len

    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(dataset, [train_len, val_len], generator=generator)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader
