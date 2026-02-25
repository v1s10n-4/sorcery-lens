"""
Centralised configuration — all values come from environment variables.
Never hardcode secrets here or anywhere else in this codebase.
"""
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import AliasChoices, Field
from typing import List


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=False,
        extra="ignore",
    )

    # Auth
    api_keys: str | None = Field(
        default=None,
        validation_alias=AliasChoices("API_KEYS", "API_KEY"),
    )  # comma-separated raw values; validated on startup

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

    @property
    def api_key_set(self) -> set[str]:
        """Return the set of valid API keys (stripped, non-empty)."""
        if not self.api_keys:
            return set()
        return {k.strip() for k in self.api_keys.split(",") if k.strip()}

    @property
    def origins_list(self) -> List[str]:
        return [o.strip() for o in self.allowed_origins.split(",") if o.strip()]


settings = Settings()
