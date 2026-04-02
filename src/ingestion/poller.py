# src/ingestion/poller.py

import time
import logging
import hashlib
from datetime import datetime, timezone
from dataclasses import dataclass
from typing import Generator

import requests
import xml.etree.ElementTree as ET

from config import settings

logger = logging.getLogger(__name__)


@dataclass
class FilingReference:
    # Lightweight object — just enough info to hand to the fetcher.
    # We use a dataclass here (not pydantic) because this object never
    # touches the database — it's purely in-memory, passed between workers.
    ticker: str
    cik: str
    form_type: str
    filed_at: datetime
    accession_number: str
    primary_document_url: str


class EdgarPoller:
    # Encapsulating the poller as a class (rather than bare functions) lets us
    # hold state — specifically the HTTP session and the seen_accessions set —
    # across multiple calls without passing them around as arguments.

    def __init__(self, seen_accessions: set[str]):
        # seen_accessions is passed in from outside — the writer will populate
        # this from the database on startup so we never reprocess old filings.
        # The poller itself doesn't touch the database — clean separation.
        self.seen_accessions = seen_accessions

        # requests.Session reuses the underlying TCP connection across requests.
        # Without this, every EDGAR request opens and closes a new connection —
        # slower and more likely to trigger rate limiting.
        self.session = requests.Session()
        self.session.headers.update({
            # EDGAR requires this. Without it they block you.
            # The format must be "Name Email" — they actually check it.
            "User-Agent": settings.edgar_user_agent,
            "Accept-Encoding": "gzip, deflate",
            "Host": "www.sec.gov"
        })

    def _get_with_retry(self, url: str, params: dict = None) -> requests.Response:
        # Every EDGAR request goes through this method.
        # It handles retries with exponential backoff — meaning if the first
        # attempt fails, we wait 1s, then 2s, then 4s before giving up.
        # This is standard practice for any HTTP client hitting an external API.

        for attempt in range(settings.edgar_max_retries):
            try:
                # Sleep BEFORE the request (except on the first attempt).
                # This is the polite way to rate-limit — you never fire two
                # requests back-to-back even on retry.
                if attempt > 0:
                    wait = settings.edgar_request_delay_seconds * (2 ** attempt)
                    logger.debug(f"Retry {attempt} for {url}, waiting {wait:.1f}s")
                    time.sleep(wait)
                else:
                    time.sleep(settings.edgar_request_delay_seconds)

                response = self.session.get(url, params=params, timeout=30)

                # raise_for_status() converts HTTP error codes (404, 429, 500)
                # into Python exceptions. Without this, requests returns a
                # response object regardless of status and you'd have to check
                # response.status_code everywhere.
                response.raise_for_status()
                return response

            except requests.exceptions.HTTPError as e:
                # 429 = Too Many Requests. EDGAR is telling us to slow down.
                if e.response.status_code == 429:
                    logger.warning("Rate limited by EDGAR — backing off 60s")
                    time.sleep(60)
                else:
                    logger.error(f"HTTP error on attempt {attempt+1}: {e}")

            except requests.exceptions.RequestException as e:
                # Network errors, timeouts, DNS failures etc.
                logger.error(f"Request failed on attempt {attempt+1}: {e}")

        # If we get here, all retries failed.
        raise RuntimeError(f"Failed to fetch {url} after {settings.edgar_max_retries} attempts")

    def _parse_rss_feed(self, xml_content: str) -> list[dict]:
        # EDGAR's RSS feed is Atom XML. We parse it with the standard library —
        # no third-party XML library needed for something this simple.
        # ElementTree requires namespace-aware parsing for Atom feeds.

        # Atom namespace — every tag in the feed is prefixed with this.
        ns = {
            "atom": "http://www.w3.org/2005/Atom",
            "edgar": "https://www.sec.gov/Archives/edgar/data/"
        }

        root = ET.fromstring(xml_content)
        entries = []

        for entry in root.findall("atom:entry", ns):
            try:
                # Each <entry> in the feed represents one filing.
                # We pull out only the fields we need.
                title = entry.find("atom:title", ns).text  # e.g. "10-K for AAPL"
                updated = entry.find("atom:updated", ns).text  # ISO 8601 datetime
                filing_href = entry.find("atom:link", ns).attrib.get("href", "")

                # The category tag holds the form type.
                category = entry.find("atom:category", ns)
                form_type = category.attrib.get("term", "") if category is not None else ""

                # accession number is embedded in the URL.
                # URL format: .../Archives/edgar/data/{CIK}/{accession-no-dashes}/
                # We extract it by splitting the URL path.
                accession_number = self._extract_accession_from_url(filing_href)
                cik = self._extract_cik_from_url(filing_href)

                if not all([accession_number, cik, form_type]):
                    logger.warning(f"Incomplete entry in RSS feed, skipping: {filing_href}")
                    continue

                entries.append({
                    "cik": cik,
                    "form_type": form_type,
                    "filed_at": datetime.fromisoformat(updated.replace("Z", "+00:00")),
                    "accession_number": accession_number,
                    "filing_index_url": filing_href,
                })

            except (AttributeError, KeyError, ValueError) as e:
                # One malformed entry shouldn't kill the whole feed parse.
                # Log it and continue — this is the extraction_warnings pattern
                # from schemas.py applied at the poller level.
                logger.warning(f"Failed to parse RSS entry: {e}")
                continue

        return entries

    def _extract_accession_from_url(self, url: str) -> str:
        # EDGAR URLs look like:
        # https://www.sec.gov/Archives/edgar/data/320193/000032019324000123/
        # The accession number is the last path segment: "000032019324000123"
        # We reformat it to the canonical "0000320193-24-000123" form.
        try:
            parts = [p for p in url.rstrip("/").split("/") if p]
            raw = parts[-1]  # "000032019324000123"
            # Accession numbers are always 18 digits: 10 + 2 + 6
            return f"{raw[:10]}-{raw[10:12]}-{raw[12:]}"
        except (IndexError, ValueError):
            return ""

    def _extract_cik_from_url(self, url: str) -> str:
        # CIK is the second-to-last numeric path segment in the URL.
        try:
            parts = [p for p in url.rstrip("/").split("/") if p]
            # Walk backwards to find the CIK (all digits, appears before accession)
            for part in reversed(parts[:-1]):
                if part.isdigit():
                    return part.zfill(10)
            return ""
        except (IndexError, ValueError):
            return ""

    def _get_primary_document_url(self, filing_index_url: str) -> str:
        # Each filing has an index page listing all the documents in the submission.
        # We need to find the primary document (the actual 10-K text, not exhibits).
        # The index page is JSON — much easier to parse than the old HTML index.

        # Convert the filing index URL to the JSON index URL.
        # HTML: .../000032019324000123/
        # JSON: .../000032019324000123/index.json
        json_url = filing_index_url.rstrip("/") + "/index.json"

        try:
            response = self._get_with_retry(json_url)
            data = response.json()

            # The index JSON has a "files" array. We want the file where
            # "type" matches our form type (e.g. "10-K") — that's the primary doc.
            for file_info in data.get("files", []):
                if file_info.get("type") in settings.target_form_types:
                    doc_name = file_info.get("name", "")
                    # Reconstruct the full URL from the base index URL and filename.
                    base = filing_index_url.rstrip("/")
                    return f"https://www.sec.gov{base}/{doc_name}"

            # If we can't find an exact type match, fall back to the first .htm file.
            for file_info in data.get("files", []):
                name = file_info.get("name", "")
                if name.endswith(".htm") or name.endswith(".html"):
                    base = filing_index_url.rstrip("/")
                    return f"https://www.sec.gov{base}/{name}"

        except Exception as e:
            logger.error(f"Failed to get primary document URL for {filing_index_url}: {e}")

        return ""

    def poll_rss(self, form_type: str = "10-K") -> Generator[FilingReference, None, None]:
        # Generator function — uses `yield` instead of `return`.
        # This means the caller gets one FilingReference at a time,
        # rather than waiting for the entire feed to be processed.
        # The fetcher can start downloading the first filing while
        # the poller is still processing the rest of the feed.

        rss_url = (
            f"https://www.sec.gov/cgi-bin/browse-edgar"
            f"?action=getcurrent&type={form_type}&dateb=&owner=include"
            f"&count=40&search_text=&output=atom"
        )

        logger.info(f"Polling EDGAR RSS for {form_type} filings")

        response = self._get_with_retry(rss_url)
        entries = self._parse_rss_feed(response.text)

        for entry in entries:
            accession = entry["accession_number"]

            # Deduplication check — the entire reason seen_accessions exists.
            if accession in self.seen_accessions:
                logger.debug(f"Already seen {accession}, skipping")
                continue

            if entry["form_type"] not in settings.target_form_types:
                logger.debug(f"Skipping form type {entry['form_type']}")
                continue

            # Resolve the actual document URL from the filing index.
            doc_url = self._get_primary_document_url(entry["filing_index_url"])
            if not doc_url:
                logger.warning(f"Could not resolve document URL for {accession}")
                continue

            # We don't know the ticker yet — that comes from a separate EDGAR
            # lookup by CIK. We leave it as "UNKNOWN" and the fetcher resolves it.
            # This is a deliberate design choice: the poller is fast, the ticker
            # lookup is slow — we don't want to block the feed on it.
            yield FilingReference(
                ticker="UNKNOWN",
                cik=entry["cik"],
                form_type=entry["form_type"],
                filed_at=entry["filed_at"],
                accession_number=accession,
                primary_document_url=doc_url,
            )

            # Mark as seen immediately after yielding.
            # If the fetcher fails on this filing, we won't retry it automatically —
            # that's intentional. Failed filings should be retried explicitly,
            # not silently re-queued on the next poll cycle.
            self.seen_accessions.add(accession)

    def poll_historical(
        self,
        cik: str,
        form_type: str = "10-K",
        start_date: str = "2000-01-01",
        end_date: str = "2024-01-01",
    ) -> Generator[FilingReference, None, None]:
        # Backfill mode — queries EDGAR's full-text search API for a specific
        # company's filing history. Used once to build the training dataset,
        # not during live operation.
        #
        # EDGAR's search API paginates — returns 10 results per page by default.
        # We page through until we've seen everything in the date range.

        logger.info(f"Polling historical {form_type} filings for CIK {cik}")

        page = 0
        page_size = 10

        while True:
            params = {
                "q": f'"{form_type}"',
                "dateRange": "custom",
                "startdt": start_date,
                "enddt": end_date,
                "entity": cik,
                "forms": form_type,
                "from": page * page_size,
                "size": page_size,
            }

            response = self._get_with_retry(
                f"{settings.edgar_base_url}/efts/hits.json",
                params=params
            )
            data = response.json()
            hits = data.get("hits", {}).get("hits", [])

            # Empty page means we've exhausted the results.
            if not hits:
                logger.info(f"Historical poll complete for CIK {cik}: {page * page_size} total checked")
                break

            for hit in hits:
                source = hit.get("_source", {})
                accession = source.get("file_date", "")
                accession_raw = hit.get("_id", "").replace("/", "-")

                if accession_raw in self.seen_accessions:
                    continue

                filed_at_str = source.get("file_date", "")
                try:
                    filed_at = datetime.strptime(filed_at_str, "%Y-%m-%d").replace(
                        tzinfo=timezone.utc
                    )
                except ValueError:
                    logger.warning(f"Could not parse date: {filed_at_str}")
                    continue

                filing_index_url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession_raw.replace('-', '')}/"
                doc_url = self._get_primary_document_url(filing_index_url)

                if not doc_url:
                    continue

                yield FilingReference(
                    ticker="UNKNOWN",
                    cik=cik.zfill(10),
                    form_type=form_type,
                    filed_at=filed_at,
                    accession_number=accession_raw,
                    primary_document_url=doc_url,
                )

                self.seen_accessions.add(accession_raw)

            page += 1