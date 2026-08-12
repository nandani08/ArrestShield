import os
from typing import List
from pydantic import BaseModel, Field

class ASRConfig(BaseModel):
    """Configuration parameters for the Real-Time Streaming ASR Pipeline."""
    vad_model_path: str = Field(
        default="silero_vad",
        description="Path or model name for Silero VAD."
    )
    asr_model_name: str = Field(
        default="shunya-labs/zero-stt-hinglish",
        description="Hinglish ASR model identifier (e.g. Whisper Hinglish)."
    )
    window_duration_sec: float = Field(
        default=2.0,
        description="Sliding audio window duration in seconds."
    )
    latency_target_ms: int = Field(
        default=800,
        description="Latency target for chunk processing in milliseconds."
    )
    sample_rate: int = Field(
        default=16000,
        description="Audio sample rate required by models."
    )


class DetectionConfig(BaseModel):
    """Configuration parameters for the Tri-State ML Detection Engine."""
    model_backbone: str = Field(
        default="google/muril-base-cased",
        description="Backbone model for target classification."
    )
    weight_scam: float = Field(
        default=0.5,
        description="Weight w1 for Scam binary classification probability."
    )
    weight_triggers: float = Field(
        default=0.3,
        description="Weight w2 for Psychological Triggers probability sum."
    )
    weight_stage: float = Field(
        default=0.2,
        description="Weight w3 for Stage Progression probability."
    )
    safe_threshold: float = Field(
        default=0.3,
        description="Upper bound risk score for SAFE classification."
    )
    fraud_threshold: float = Field(
        default=0.7,
        description="Lower bound risk score for FRAUD classification."
    )


class HoneypotConfig(BaseModel):
    """Configuration parameters for the LLM Honeypot Engine."""
    llm_model: str = Field(
        default="Qwen/Qwen2.5-7B-Instruct",
        description="LLM model identifier used for generation."
    )
    llm_api_url: str = Field(
        default="http://localhost:11434/v1",
        description="API endpoint for Ollama or vLLM."
    )
    alpha_utility: float = Field(
        default=1.0,
        description="Utility weight (alpha) for extracted indicators."
    )
    beta_utility: float = Field(
        default=0.5,
        description="Utility weight (beta) penalizing PII leak risk."
    )


class ExtractionConfig(BaseModel):
    """Configuration parameters for Zero-Shot Threat Extraction."""
    gliner_model: str = Field(
        default="urchade/gliner_medium-v2.1",
        description="GLiNER model name for named entity recognition."
    )
    entity_labels: List[str] = Field(
        default_factory=lambda: [
            "UPI ID",
            "phone number",
            "police badge ID",
            "court reference number",
            "claimed agency",
            "case ID"
        ],
        description="List of entities to extract from the conversation."
    )


class ServerConfig(BaseModel):
    """Configuration parameters for the FastAPI & WebSocket Server."""
    host: str = Field(
        default="0.0.0.0",
        description="Server host IP address."
    )
    port: int = Field(
        default=8000,
        description="Server port."
    )
    ws_asr_path: str = Field(
        default="/ws/asr",
        description="WebSocket endpoint path for streaming ASR."
    )


class AppConfig(BaseModel):
    """Global Application configuration."""
    env: str = Field(
        default=os.getenv("APP_ENV", "development"),
        description="Current deployment environment."
    )
    asr: ASRConfig = Field(default_factory=ASRConfig)
    detection: DetectionConfig = Field(default_factory=DetectionConfig)
    honeypot: HoneypotConfig = Field(default_factory=HoneypotConfig)
    extraction: ExtractionConfig = Field(default_factory=ExtractionConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)


# Global settings instance
settings = AppConfig()
