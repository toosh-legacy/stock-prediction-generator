# src/ingestion/schemas.py

from datetime import datetime
from enum import Enum
from pydantic import BaseModel, field_validator
import hashlib


class FormType(str, Enum):
    # We only care about these three form types.
    # 10-K = annual report, 10-Q = quarterly, 8-K = material event (earnings, CEO change etc.)
    TEN_K  = "10-K"
    TEN_Q  = "10-Q"
    EIGHT_K = "8-K"


class FilingMetadata(BaseModel):
    # Every filing that enters the system gets one of these.
    # Think of it as the passport for a document.

    ticker: str                  # e.g. "AAPL"
    cik: str                     # SEC's internal company ID, e.g. "0000320193"
    form_type: FormType          # one of the three above
    filed_at: datetime           # when SEC received it — NOT when we scraped it
    accession_number: str        # SEC's unique filing ID, e.g. "0000320193-24-000123"
    filing_url: str              # the URL we downloaded from
    content_hash: str            # SHA-256 of the raw text — used for deduplication
    raw_text_path: str           # path in object store where clean text lives
    scraped_at: datetime         # when we downloaded it — for auditing
    word_count: int              # rough size signal — filter out empty/tiny filings

    @field_validator("cik")
    @classmethod
    def pad_cik(cls, v: str) -> str:
        # EDGAR CIK numbers are always 10 digits, zero-padded.
        # The API sometimes returns them without padding. We normalize here
        # so downstream code never has to think about this.
        return v.zfill(10)

    @field_validator("ticker")
    @classmethod
    def uppercase_ticker(cls, v: str) -> str:
        return v.upper().strip()


class RawFiling(BaseModel):
    # This is the intermediate object between fetcher and extractor.
    # It carries the raw bytes BEFORE cleaning.
    # We keep it separate from FilingMetadata because at this point
    # we don't yet have a content_hash or raw_text_path.

    url: str
    html_content: str            # raw HTML as downloaded — not yet cleaned
    fetched_at: datetime
    http_status: int             # we store this so we can audit failed fetches


class ExtractedText(BaseModel):
    # Output of extractor.py — clean prose ready to be tokenized.

    accession_number: str
    sections: dict[str, str]     # {"mda": "...", "risk_factors": "...", "financials": "..."}
    full_text: str               # all sections concatenated, for simple use cases
    word_count: int
    extraction_warnings: list[str]  # anything weird we noticed during cleaning