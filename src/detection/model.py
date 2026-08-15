import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple, Union
from transformers import AutoModel

from src.config import settings

logger = logging.getLogger("ArrestShield.DetectionModel")


class DummyBackbone(nn.Module):
    """
    Fallback embedding backbone for offline testing and environments without internet/model cache.
    """
    def __init__(self, vocab_size: int = 30522, hidden_size: int = 768):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, hidden_size)
        self.fc = nn.Linear(hidden_size, hidden_size)

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, token_type_ids: Optional[torch.Tensor] = None):
        embeds = self.embeddings(input_ids)
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
            pooled = torch.sum(embeds * mask, dim=1) / torch.clamp(mask.sum(dim=1), min=1e-9)
        else:
            pooled = torch.mean(embeds, dim=1)
        pooled = torch.tanh(self.fc(pooled))
        return type("Output", (), {"pooler_output": pooled, "last_hidden_state": embeds})()


class MultitaskMuRILDetector(nn.Module):
    """
    Multi-Task Deep Learning Model for Scam Detection built over google/muril-base-cased backbone.
    
    Head Architecture:
    - Shared Backbone: google/muril-base-cased (768 hidden dimension)
    - Head A (is_scam): 2 output logits (Binary Classification: Non-Scam vs Scam)
    - Head B (triggers): 4 output logits (Multi-Label Sigmoid: authority, urgency, isolation, payment_pressure)
    - Head C (scam_stage): 6 output logits (Categorical Softmax: none, impersonation, allegation, isolation, coercion, payment)
    """
    def __init__(
        self,
        model_name: str = "google/muril-base-cased",
        num_triggers: int = 4,
        num_stages: int = 6,
        dropout_rate: float = 0.2,
        loss_weights: Optional[Tuple[float, float, float]] = None,
        **kwargs
    ):
        super().__init__()
        self.model_name = model_name
        self.num_triggers = num_triggers
        self.num_stages = num_stages

        # Loss weights for multitask loss fusion (lambda_scam, lambda_triggers, lambda_stage)
        self.loss_weights = loss_weights or (
            settings.detection.weight_scam,
            settings.detection.weight_triggers,
            settings.detection.weight_stage
        )

        # Initialize shared backbone
        self.use_dummy_backbone = False
        if model_name == "dummy" or kwargs.get("use_dummy", False):
            self.use_dummy_backbone = True
            hidden_size = 768
            self.backbone = DummyBackbone(vocab_size=30522, hidden_size=hidden_size)
        else:
            try:
                # Try loading cached local model first to avoid network retry delays
                self.backbone = AutoModel.from_pretrained(model_name, local_files_only=True)
                hidden_size = getattr(self.backbone.config, "hidden_size", 768)
            except Exception:
                try:
                    self.backbone = AutoModel.from_pretrained(model_name)
                    hidden_size = getattr(self.backbone.config, "hidden_size", 768)
                except Exception as e:
                    logger.warning(f"Could not load AutoModel '{model_name}' ({e}). Utilizing DummyBackbone for offline execution.")
                    self.use_dummy_backbone = True
                    hidden_size = 768
                    self.backbone = DummyBackbone(vocab_size=30522, hidden_size=hidden_size)

        self.dropout = nn.Dropout(dropout_rate)

        # Head A: Binary Scam Classification (2 logits)
        self.head_scam = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size // 2, 2)
        )

        # Head B: Multi-label Psychological Triggers (4 logits)
        self.head_triggers = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size // 2, num_triggers)
        )

        # Head C: Categorical Scam Stage Progression (6 logits)
        self.head_stage = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size // 2, num_stages)
        )

        # Loss Functions
        self.criterion_scam = nn.CrossEntropyLoss()
        self.criterion_triggers = nn.BCEWithLogitsLoss()
        self.criterion_stage = nn.CrossEntropyLoss()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        is_scam_labels: Optional[torch.Tensor] = None,
        triggers_labels: Optional[torch.Tensor] = None,
        stage_labels: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        """
        Forward pass through shared MuRIL backbone and task classification heads.
        """
        if self.use_dummy_backbone:
            outputs = self.backbone(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            pooled_output = outputs.pooler_output
        else:
            kwargs = {"attention_mask": attention_mask}
            if token_type_ids is not None:
                kwargs["token_type_ids"] = token_type_ids
            outputs = self.backbone(input_ids, **kwargs)
            if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                pooled_output = outputs.pooler_output
            else:
                pooled_output = outputs.last_hidden_state[:, 0, :]

        pooled_output = self.dropout(pooled_output)

        # Task Heads Logits
        logits_scam = self.head_scam(pooled_output)          # [batch_size, 2]
        logits_triggers = self.head_triggers(pooled_output)  # [batch_size, 4]
        logits_stage = self.head_stage(pooled_output)        # [batch_size, 6]

        # Computed Output Probabilities
        prob_scam = F.softmax(logits_scam, dim=-1)[:, 1]       # P(scam = 1)
        prob_triggers = torch.sigmoid(logits_triggers)       # P(trigger_k = 1)
        prob_stage = F.softmax(logits_stage, dim=-1)           # P(stage = s)

        result = {
            "logits_scam": logits_scam,
            "logits_triggers": logits_triggers,
            "logits_stage": logits_stage,
            "prob_scam": prob_scam,
            "prob_triggers": prob_triggers,
            "prob_stage": prob_stage
        }

        # Calculate Loss if labels are passed
        if is_scam_labels is not None and triggers_labels is not None and stage_labels is not None:
            loss_scam = self.criterion_scam(logits_scam, is_scam_labels)
            loss_triggers = self.criterion_triggers(logits_triggers, triggers_labels)
            loss_stage = self.criterion_stage(logits_stage, stage_labels)

            w1, w2, w3 = self.loss_weights
            total_loss = w1 * loss_scam + w2 * loss_triggers + w3 * loss_stage

            result["loss"] = total_loss
            result["loss_components"] = {
                "scam": loss_scam.item(),
                "triggers": loss_triggers.item(),
                "stage": loss_stage.item()
            }

        return result
