# config.py  (lives at repo root, not inside src/)

from pydantic_settings import BaseSettings
from pydantic import Field
from pathlib import Path


class Settings(BaseSettings):

    # --- Database ---
    # PostgreSQL connection string. In local dev this points to a Docker container.
    # In prod it points to your cloud DB. The code never changes — only the env var does.
    database_url: str = Field(
        ...,  # the ... means required — no default, app won't start without it
        description="PostgreSQL connection string"
    )

    # --- Object store ---
    # MinIO locally, AWS S3 in prod. Same API, different endpoint.
    object_store_endpoint: str = Field(
        default="http://localhost:9000",
        description="S3-compatible endpoint URL"
    )
    object_store_access_key: str = Field(
        ...,
        description="Access key for object store"
    )
    object_store_secret_key: str = Field(
        ...,
        description="Secret key for object store"
    )
    object_store_bucket: str = Field(
        default="quant-filings",
        description="Bucket where cleaned filing text is stored"
    )

    # --- EDGAR ---
    # EDGAR asks that scrapers identify themselves via User-Agent.
    # They will block you if you don't. Format: "Name Email"
    edgar_user_agent: str = Field(
        ...,
        description="User-Agent header sent to EDGAR. Format: 'Your Name your@email.com'"
    )
    edgar_base_url: str = Field(
        default="https://efts.sec.gov",
        description="EDGAR full-text search base URL"
    )
    edgar_rss_url: str = Field(
        default="https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent&type=10-K&dateb=&owner=include&count=40&search_text=&output=atom",
        description="EDGAR RSS feed for latest filings"
    )

    # How long to wait between EDGAR requests.
    # EDGAR rate-limits aggressively — 10 requests/second max.
    # We stay well under that to avoid getting blocked.
    edgar_request_delay_seconds: float = Field(
        default=0.5,
        description="Seconds to wait between EDGAR HTTP requests"
    )
    edgar_max_retries: int = Field(
        default=3,
        description="How many times to retry a failed EDGAR request"
    )

    # --- Ingestion filters ---
    # We skip filings with fewer than this many words.
    # A real 10-K is 20,000+ words. Anything tiny is a cover page or amendment stub.
    min_word_count: int = Field(
        default=1000,
        description="Minimum word count to accept a filing as valid"
    )

    # Form types we care about. Everything else gets skipped at the poller stage.
    target_form_types: list[str] = Field(
        default=["10-K", "10-Q", "8-K"],
        description="SEC form types to ingest"
    )

    # --- Paths ---
    # Local cache for raw HTML before it goes to the object store.
    # Useful for debugging — you can inspect what came off the wire.
    raw_cache_dir: Path = Field(
        default=Path("data/raw_cache"),
        description="Local directory for raw HTML cache"
    )
    log_dir: Path = Field(
        default=Path("logs"),
        description="Directory for log files"
    )

    # --- Logging ---
    log_level: str = Field(
        default="INFO",
        description="Logging level: DEBUG, INFO, WARNING, ERROR"
    )

    # --- Weights & Biases (used later in training, declared here so config is complete) ---
    wandb_api_key: str = Field(
        default="",
        description="W&B API key — leave empty to disable experiment tracking"
    )
    wandb_project: str = Field(
        default="quant-intelligence",
        description="W&B project name"
    )

    model_config = {
        # Tells pydantic-settings to look for a .env file at repo root.
        # Variables in the environment take precedence over .env values.
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        # Allows extra fields without crashing — useful when you add new vars
        # to .env before updating this class
        "extra": "ignore"
    }


# This is the single instance the entire codebase imports.
# Every other file does: from config import settings
# Never instantiate Settings() yourself elsewhere — always use this.
settings = Settings()