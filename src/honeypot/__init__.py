# Honeypot module initialization
from .prompts import VICTIM_SYSTEM_PROMPT, STAGE_PROMPTS, build_honeypot_prompt
from .llm_client import HoneypotLLMClient

__all__ = [
    "VICTIM_SYSTEM_PROMPT",
    "STAGE_PROMPTS",
    "build_honeypot_prompt",
    "HoneypotLLMClient"
]
