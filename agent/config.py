import os
from dataclasses import dataclass

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    api_key: str
    base_url: str
    llm_model: str
    embed_model: str


def load_settings() -> Settings:
    load_dotenv()

    api_key = os.getenv("NVIDIA_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "NVIDIA_API_KEY is missing. Add it to a .env file or set it in your shell."
        )

    return Settings(
        api_key=api_key,
        base_url=os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1"),
        llm_model=os.getenv("NVIDIA_LLM_MODEL", "meta/llama-3.1-8b-instruct"),
        embed_model=os.getenv("NVIDIA_EMBED_MODEL", "nvidia/nv-embedqa-e5-v5"),
    )

