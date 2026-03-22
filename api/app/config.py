from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    database_url: str
    redis_url: str = "redis://localhost:6379"
    # Upstash Redis (used in production on Koyeb — free, no CC)
    upstash_redis_rest_url: str = ""
    upstash_redis_rest_token: str = ""
    alpha_vantage_api_key: str = ""
    news_api_key: str = ""
    # Local filesystem for model artifacts — no S3 needed
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

    class Config:
        env_file = ".env"


@lru_cache
def get_settings() -> Settings:
    return Settings()
