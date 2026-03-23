from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    database_url: str
    redis_url: str = "redis://localhost:6379"
    upstash_redis_rest_url: str = ""
    upstash_redis_rest_token: str = ""

    # News & sentiment
    news_api_key: str = ""
    alpha_vantage_api_key: str = ""

    # Macro data
    fred_api_key: str = ""

    # Model storage — HuggingFace Hub
    hf_token: str = ""
    hf_repo_id: str = ""  # e.g. "username/stocksage-models"

    # Local filesystem fallback for model artifacts
    model_artifacts_dir: str = "./models"
    mlflow_tracking_uri: str = "http://localhost:5000"

    secret_key: str
    api_key_salt: str
    environment: str = "development"
    rate_limit_requests: int = 100
    rate_limit_window_seconds: int = 3600
    allowed_origins: list[str] = ["http://localhost:3000"]

    @property
    def using_upstash(self) -> bool:
        return bool(self.upstash_redis_rest_url and self.upstash_redis_rest_token)

    @property
    def using_hf(self) -> bool:
        return bool(self.hf_token and self.hf_repo_id)

    @property
    def using_fred(self) -> bool:
        return bool(self.fred_api_key)

    class Config:
        env_file = ".env"


@lru_cache
def get_settings() -> Settings:
    return Settings()
