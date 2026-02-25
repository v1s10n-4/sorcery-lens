"""
Centralised configuration — all values come from environment variables.
Never hardcode secrets here or anywhere else in this codebase.
"""
from pydantic_settings import BaseSettings
from pydantic import field_validator
from typing import List


class Settings(BaseSettings):
    # Auth
    api_keys: str  # comma-separated raw values; validated on startup

    # CORS
    allowed_origins: str = "https://sorcery-companion.vercel.app"

    # Rate limiting
    rate_limit_per_minute: int = 60

    # CLIP model
    clip_model: str = "ViT-B-32"
    clip_pretrained: str = "openai"

    # Index paths
    embeddings_path: str = "data/embeddings.npz"
    index_path: str = "data/index.json"

    class Config:
        env_file = ".env"
        case_sensitive = False

    @property
    def api_key_set(self) -> set[str]:
        """Return the set of valid API keys (stripped, non-empty)."""
        return {k.strip() for k in self.api_keys.split(",") if k.strip()}

    @property
    def origins_list(self) -> List[str]:
        return [o.strip() for o in self.allowed_origins.split(",") if o.strip()]


settings = Settings()
