from __future__ import annotations
import os
from dotenv import load_dotenv

load_dotenv()

def get_llm_config() -> dict:
    """
    CrewAI supports multiple LLM backends. Many setups route via LiteLLM semantics.
    For Ollama, the key knobs are model name and base URL.

    We keep it as a dict so you can adapt if your CrewAI version expects
    a different wrapper class/constructor.
    """
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
    model = os.getenv("OLLAMA_MODEL", "ollama/llama3.2")
    # Some stacks want "ollama/<model>", others just "<model>" with base_url.
    # We'll store both.
    return {
        "base_url": base_url,
        "model": model,
        "provider_model": f"ollama/{model}",
    }
