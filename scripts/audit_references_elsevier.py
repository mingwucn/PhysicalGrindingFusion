#!/usr/bin/env python3
"""Validate DOI-bearing BibTeX entries against Elsevier Scopus metadata.

The API key is read from ``ELSEVIER_API_KEY`` or an untracked ``.env`` file.
The key is never written to reports, cache files, or logs.

Elsevier is the primary source. Crossref is used only as a DOI-registry
fallback when a DOI is absent from Scopus. Missing or incorrect DOIs are
repaired only after a high-confidence Scopus title and author match.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import os
import re
import time
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BIB = ROOT / "overleaf" / "ref.bib"
DEFAULT_REPORT = ROOT / "reports" / "evidence" / "tables" / "reference_doi_audit.csv"
DEFAULT_SUMMARY = ROOT / "reports" / "evidence" / "tables" / "reference_doi_audit_summary.json"
DEFAULT_CACHE = ROOT / ".cache" / "elsevier_reference_audit"
API_TEMPLATE = "https://api.elsevier.com/content/abstract/doi/{doi}"
SEARCH_ENDPOINT = "https://api.elsevier.com/content/search/scopus"
CROSSREF_TEMPLATE = "https://api.crossref.org/works/{doi}"
CROSSREF_SEARCH_ENDPOINT = "https://api.crossref.org/works"
SUPPORTED_FIELDS = ("doi", "title", "journal", "booktitle", "year", "volume", "number", "pages", "issn")


@dataclass(frozen=True)
class BibField:
    name: str
    value: str
    value_start: int
    value_end: int


@dataclass(frozen=True)
class BibEntry:
    entry_type: str
    key: str
    start: int
    close: int
    fields: dict[str, BibField]


def find_closing_brace(text: str, opening: int) -> int:
    depth = 0
    escaped = False
    for index in range(opening, len(text)):
        char = text[index]
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return index
    raise ValueError(f"Unclosed BibTeX entry beginning at byte {opening}")


def parse_braced_value(text: str, opening: int, limit: int) -> tuple[int, int, int]:
    depth = 0
    escaped = False
    for index in range(opening, limit):
        char = text[index]
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return opening + 1, index, index + 1
    raise ValueError(f"Unclosed braced BibTeX value at byte {opening}")


def parse_quoted_value(text: str, opening: int, limit: int) -> tuple[int, int, int]:
    escaped = False
    for index in range(opening + 1, limit):
        char = text[index]
        if escaped:
            escaped = False
        elif char == "\\":
            escaped = True
        elif char == '"':
            return opening + 1, index, index + 1
    raise ValueError(f"Unclosed quoted BibTeX value at byte {opening}")


def parse_fields(text: str, start: int, close: int) -> dict[str, BibField]:
    fields: dict[str, BibField] = {}
    index = start
    while index < close:
        while index < close and (text[index].isspace() or text[index] == ","):
            index += 1
        match = re.match(r"([A-Za-z][A-Za-z0-9_-]*)\s*=\s*", text[index:close])
        if not match:
            break
        name = match.group(1).lower()
        index += match.end()
        if index >= close:
            break
        if text[index] == "{":
            value_start, value_end, index = parse_braced_value(text, index, close)
        elif text[index] == '"':
            value_start, value_end, index = parse_quoted_value(text, index, close)
        else:
            value_start = index
            while index < close and text[index] not in ",\n\r":
                index += 1
            value_end = index
        fields[name] = BibField(name, text[value_start:value_end].strip(), value_start, value_end)
    return fields


def parse_bibtex(text: str) -> list[BibEntry]:
    entries: list[BibEntry] = []
    pattern = re.compile(r"@([A-Za-z]+)\s*\{", re.MULTILINE)
    for match in pattern.finditer(text):
        opening = match.end() - 1
        close = find_closing_brace(text, opening)
        comma = text.find(",", opening + 1, close)
        if comma < 0:
            continue
        key = text[opening + 1:comma].strip()
        entries.append(BibEntry(match.group(1).lower(), key, match.start(), close, parse_fields(text, comma + 1, close)))
    return entries


def clean_doi(value: str) -> str:
    value = html.unescape(value).strip().strip("{}")
    value = re.sub(r"^doi:\s*", "", value, flags=re.IGNORECASE)
    value = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", value, flags=re.IGNORECASE)
    return value.rstrip(".,; ").lower()


def plain_text(value: str) -> str:
    value = html.unescape(re.sub(r"<[^>]+>", " ", value or ""))
    value = re.sub(r"\\[A-Za-z]+\*?(?:\[[^]]*\])?", " ", value)
    value = value.replace("{", "").replace("}", "").replace("\\", "")
    value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    return " ".join(re.findall(r"[a-z0-9]+", value.lower()))


def title_similarity(left: str, right: str) -> float:
    a, b = plain_text(left), plain_text(right)
    if not a or not b:
        return 0.0
    sequence = SequenceMatcher(None, a, b).ratio()
    a_tokens, b_tokens = set(a.split()), set(b.split())
    jaccard = len(a_tokens & b_tokens) / max(1, len(a_tokens | b_tokens))
    return max(sequence, jaccard)


def bib_surnames(author_field: str) -> set[str]:
    result: set[str] = set()
    for author in re.split(r"\s+and\s+", author_field or ""):
        normalized = plain_text(author)
        if not normalized:
            continue
        raw = plain_text(author.split(",", 1)[0]) if "," in author else normalized.split()[-1]
        result.add(raw)
    return result


def api_surnames(payload: dict[str, Any]) -> set[str]:
    response = payload.get("abstracts-retrieval-response", {})
    authors = response.get("authors", {}).get("author", [])
    if isinstance(authors, dict):
        authors = [authors]
    return {plain_text(str(author.get("ce:surname", ""))) for author in authors if author.get("ce:surname")}


def author_overlap(bib_authors: str, payload: dict[str, Any]) -> float:
    left, right = bib_surnames(bib_authors), api_surnames(payload)
    if not left or not right:
        return 0.0
    return len(left & right) / max(1, min(len(left), len(right)))


def load_api_key() -> str:
    key = os.environ.get("ELSEVIER_API_KEY", "").strip()
    env_path = ROOT / ".env"
    if not key and env_path.exists():
        for line in env_path.read_text().splitlines():
            if line.strip().startswith("ELSEVIER_API_KEY="):
                key = line.split("=", 1)[1].strip().strip('"\'')
                break
    if not key:
        raise RuntimeError("Set ELSEVIER_API_KEY or add it to the untracked .env file.")
    return key


def cache_path(cache_dir: Path, doi: str) -> Path:
    return cache_dir / f"{hashlib.sha256(doi.encode()).hexdigest()}.json"


def request_json(url: str, api_key: str = "", retries: int = 3) -> tuple[int, dict[str, Any]]:
    headers = {"Accept": "application/json", "User-Agent": "PhysicalGrindingFusion-reference-audit/1.0"}
    if api_key:
        headers["X-ELS-APIKey"] = api_key
    request = Request(
        url,
        headers=headers,
    )
    status, payload = 0, {}
    for attempt in range(retries):
        try:
            with urlopen(request, timeout=30) as response:
                return response.status, json.loads(response.read().decode("utf-8"))
        except HTTPError as error:
            status = error.code
            raw = error.read().decode("utf-8", errors="replace")
            try:
                payload = json.loads(raw) if raw else {}
            except json.JSONDecodeError:
                payload = {"error": raw[:500]}
            if status == 429 and attempt + 1 < retries:
                time.sleep(2 ** (attempt + 1))
                continue
            return status, payload
        except URLError as error:
            payload = {"error": str(error.reason)}
            if attempt + 1 < retries:
                time.sleep(2 ** attempt)
    return status, payload


def retrieve_elsevier(doi: str, api_key: str, cache_dir: Path, refresh: bool, retries: int = 3) -> tuple[int, dict[str, Any], str]:
    path = cache_path(cache_dir, doi)
    if path.exists() and not refresh:
        cached = json.loads(path.read_text())
        return int(cached["http_status"]), cached.get("payload", {}), "cache"
    status, payload = request_json(API_TEMPLATE.format(doi=quote(doi, safe="")), api_key, retries)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"http_status": status, "payload": payload}, ensure_ascii=False, indent=2) + "\n")
    return status, payload, "api"


def search_elsevier(title: str, api_key: str, cache_dir: Path, refresh: bool) -> tuple[int, dict[str, Any], str]:
    normalized = plain_text(title)
    path = cache_dir / f"search-{hashlib.sha256(normalized.encode()).hexdigest()}.json"
    if path.exists() and not refresh:
        cached = json.loads(path.read_text())
        return int(cached["http_status"]), cached.get("payload", {}), "cache"
    query_title = re.sub(r'["()]', " ", plain_text(title))
    parameters = {"query": f'TITLE("{query_title}")', "count": 5, "view": "STANDARD"}
    url = f"{SEARCH_ENDPOINT}?{urlencode(parameters)}"
    status, payload = request_json(url, api_key)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"http_status": status, "payload": payload}, ensure_ascii=False, indent=2) + "\n")
    return status, payload, "api"


def retrieve_crossref(doi: str, cache_dir: Path, refresh: bool) -> tuple[int, dict[str, Any], str]:
    path = cache_dir / f"crossref-{hashlib.sha256(doi.encode()).hexdigest()}.json"
    if path.exists() and not refresh:
        cached = json.loads(path.read_text())
        return int(cached["http_status"]), cached.get("payload", {}), "crossref-cache"
    status, payload = request_json(CROSSREF_TEMPLATE.format(doi=quote(doi, safe="")))
    cache_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"http_status": status, "payload": payload}, ensure_ascii=False, indent=2) + "\n")
    return status, payload, "crossref-api"


def search_crossref(title: str, cache_dir: Path, refresh: bool) -> tuple[int, dict[str, Any], str]:
    normalized = plain_text(title)
    path = cache_dir / f"crossref-search-{hashlib.sha256(normalized.encode()).hexdigest()}.json"
    if path.exists() and not refresh:
        cached = json.loads(path.read_text())
        return int(cached["http_status"]), cached.get("payload", {}), "crossref-search-cache"
    parameters = {
        "query.title": title,
        "rows": 5,
        "select": "DOI,title,author,issued,container-title,volume,issue,page,ISSN",
    }
    status, payload = request_json(f"{CROSSREF_SEARCH_ENDPOINT}?{urlencode(parameters)}")
    cache_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"http_status": status, "payload": payload}, ensure_ascii=False, indent=2) + "\n")
    return status, payload, "crossref-search-api"


def crossref_search_candidates(payload: dict[str, Any], bib_title: str, bib_year: str) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for item in payload.get("message", {}).get("items", []):
        titles = item.get("title", [])
        doi = clean_doi(str(item.get("DOI", "")))
        if not titles or not doi:
            continue
        title = str(titles[0])
        date_parts = item.get("issued", {}).get("date-parts", [[]])
        year = str(date_parts[0][0]) if date_parts and date_parts[0] else ""
        similarity = title_similarity(bib_title, title)
        year_match = not bib_year or not year or bib_year.strip("{} ") == year
        candidates.append({
            "doi": doi,
            "title": title,
            "year": year,
            "similarity": similarity,
            "year_match": year_match,
            "score": similarity + (0.03 if year_match else -0.05),
        })
    return sorted(candidates, key=lambda item: float(item["score"]), reverse=True)


def search_candidates(payload: dict[str, Any], bib_title: str, bib_year: str) -> list[dict[str, str | float]]:
    entries = payload.get("search-results", {}).get("entry", [])
    if isinstance(entries, dict):
        entries = [entries]
    candidates: list[dict[str, str | float]] = []
    for result in entries:
        title = str(result.get("dc:title", ""))
        doi = clean_doi(str(result.get("prism:doi", "")))
        if not title or not doi:
            continue
        year = str(result.get("prism:coverDate", ""))[:4]
        similarity = title_similarity(bib_title, title)
        year_match = not bib_year or not year or bib_year.strip("{} ") == year
        score = similarity + (0.03 if year_match else -0.05)
        candidates.append({"doi": doi, "title": title, "year": year, "similarity": similarity, "score": score})
    return sorted(candidates, key=lambda item: float(item["score"]), reverse=True)


def core_metadata(payload: dict[str, Any], entry_type: str) -> dict[str, str]:
    core = payload.get("abstracts-retrieval-response", {}).get("coredata", {})
    publication_field = "booktitle" if entry_type in {"inproceedings", "conference"} else "journal"
    cover_date = str(core.get("prism:coverDate", ""))
    metadata = {
        "doi": clean_doi(str(core.get("prism:doi", ""))),
        "title": str(core.get("dc:title", "")).strip(),
        publication_field: str(core.get("prism:publicationName", "")).strip(),
        "year": cover_date[:4] if re.match(r"\d{4}", cover_date) else "",
        "volume": str(core.get("prism:volume", "")).strip(),
        "number": str(core.get("prism:issueIdentifier", "")).strip(),
        "pages": str(core.get("prism:pageRange", "")).strip(),
        "issn": str(core.get("prism:issn", "") or core.get("prism:eIssn", "")).strip(),
    }
    return {key: value for key, value in metadata.items() if value}


def crossref_metadata(payload: dict[str, Any], entry_type: str) -> dict[str, str]:
    message = payload.get("message", {})
    titles = message.get("title", [])
    containers = message.get("container-title", [])
    date_parts = message.get("issued", {}).get("date-parts", [[]])
    publication_field = "booktitle" if entry_type in {"inproceedings", "conference"} else "journal"
    metadata = {
        "doi": clean_doi(str(message.get("DOI", ""))),
        "title": str(titles[0]).strip() if titles else "",
        publication_field: str(containers[0]).strip() if containers else "",
        "year": str(date_parts[0][0]) if date_parts and date_parts[0] else "",
        "volume": str(message.get("volume", "")).strip(),
        "number": str(message.get("issue", "")).strip(),
        "pages": str(message.get("page", "")).strip(),
        "issn": str((message.get("ISSN") or [""])[0]).strip(),
    }
    return {key: value for key, value in metadata.items() if value}


def crossref_author_overlap(bib_authors: str, payload: dict[str, Any]) -> float:
    left = bib_surnames(bib_authors)
    right = {
        plain_text(str(author.get("family", "")))
        for author in payload.get("message", {}).get("author", [])
        if author.get("family")
    }
    if not left or not right:
        return 0.0
    return len(left & right) / max(1, min(len(left), len(right)))


def escape_bibtex(value: str) -> str:
    value = html.unescape(re.sub(r"<[^>]+>", "", value)).strip()
    for source, replacement in (("&", r"\&"), ("%", r"\%"), ("#", r"\#")):
        value = value.replace(source, replacement)
    return value


def apply_updates(text: str, entries: list[BibEntry], updates: dict[str, dict[str, str]]) -> str:
    by_key = {entry.key: entry for entry in entries}
    for key in sorted(updates, key=lambda item: by_key[item].start, reverse=True):
        entry = by_key[key]
        replacements: list[tuple[int, int, str]] = []
        missing: list[tuple[str, str]] = []
        for field_name, value in updates[key].items():
            value = escape_bibtex(value)
            if field_name in entry.fields:
                field = entry.fields[field_name]
                replacements.append((field.value_start, field.value_end, value))
            else:
                missing.append((field_name, value))
        close = entry.close + sum(len(value) - (end - start) for start, end, value in replacements)
        for start, end, value in sorted(replacements, reverse=True):
            text = text[:start] + value + text[end:]
        if missing:
            insertion_at = close
            while insertion_at > entry.start and text[insertion_at - 1].isspace():
                insertion_at -= 1
            separator = "" if text[insertion_at - 1] == "," else ","
            insertion = separator + "".join(f"\n  {name} = {{{value}}}," for name, value in missing)
            insertion = insertion.rstrip(",")
            text = text[:insertion_at] + insertion + text[insertion_at:]
    return text


def metadata_updates(entry: BibEntry, metadata: dict[str, str], doi: str, refresh_existing: bool) -> dict[str, str]:
    updates: dict[str, str] = {}
    existing_doi = entry.fields.get("doi")
    if not existing_doi or clean_doi(existing_doi.value) != doi:
        updates["doi"] = doi
    for field_name, value in metadata.items():
        if field_name not in SUPPORTED_FIELDS or field_name in {"doi", "title"}:
            continue
        if refresh_existing or field_name not in entry.fields:
            updates[field_name] = value
    if "title" not in entry.fields and metadata.get("title"):
        updates["title"] = metadata["title"]
    return updates


def audit(args: argparse.Namespace) -> int:
    bib_path = args.bib.resolve()
    original = bib_path.read_text()
    entries = parse_bibtex(original)
    api_key = load_api_key()
    rows: list[dict[str, Any]] = []
    updates: dict[str, dict[str, str]] = {}
    duplicate_dois: dict[str, list[str]] = {}
    empty = BibField("", "", 0, 0)

    for entry in entries:
        doi_field = entry.fields.get("doi")
        doi = clean_doi(doi_field.value) if doi_field else ""
        bib_title = entry.fields.get("title", empty).value
        bib_year = entry.fields.get("year", empty).value
        status_code, payload, source = 0, {}, ""
        metadata: dict[str, str] = {}
        if doi:
            duplicate_dois.setdefault(doi, []).append(entry.key)
            status_code, payload, source = retrieve_elsevier(doi, api_key, args.cache_dir, args.refresh_cache)
            metadata = core_metadata(payload, entry.entry_type) if status_code == 200 else {}
        api_doi = metadata.get("doi", "")
        api_title = metadata.get("title", "")
        similarity = title_similarity(bib_title, api_title)
        overlap = author_overlap(entry.fields.get("author", empty).value, payload)
        if not doi:
            status = "missing_doi"
        elif status_code == 200 and api_doi == doi and similarity >= args.title_threshold:
            status = "verified"
        elif status_code == 200 and api_doi and api_doi != doi:
            status = "doi_mismatch"
        elif status_code == 200:
            status = "metadata_mismatch"
        elif status_code == 404:
            status = "not_found_in_scopus"
        elif status_code in {401, 403}:
            status = "authentication_or_entitlement_error"
        elif status_code == 429:
            status = "quota_exceeded"
        else:
            status = "api_error"

        registry_status = 0
        registry_source = ""
        if status == "not_found_in_scopus":
            registry_status, registry_payload, registry_source = retrieve_crossref(
                doi, args.cache_dir, args.refresh_cache
            )
            registry_metadata = crossref_metadata(registry_payload, entry.entry_type) if registry_status == 200 else {}
            registry_similarity = title_similarity(bib_title, registry_metadata.get("title", ""))
            if (
                registry_status == 200
                and registry_metadata.get("doi") == doi
                and registry_similarity >= args.title_threshold
            ):
                metadata = registry_metadata
                api_doi = doi
                api_title = registry_metadata.get("title", "")
                similarity = registry_similarity
                overlap = crossref_author_overlap(entry.fields.get("author", empty).value, registry_payload)
                status = "verified_by_crossref"
                source = registry_source

        candidate_doi = ""
        candidate_title = ""
        candidate_similarity = 0.0
        search_status = 0
        needs_search = status in {"missing_doi", "not_found_in_scopus", "doi_mismatch", "metadata_mismatch"}
        if needs_search and bib_title:
            search_status, search_payload, search_source = search_elsevier(
                bib_title, api_key, args.cache_dir, args.refresh_cache
            )
            candidates = search_candidates(search_payload, bib_title, bib_year) if search_status == 200 else []
            if candidates:
                best = candidates[0]
                candidate_doi = str(best["doi"])
                candidate_title = str(best["title"])
                candidate_similarity = float(best["similarity"])
                if candidate_similarity >= args.search_title_threshold:
                    candidate_status, candidate_payload, candidate_source = retrieve_elsevier(
                        candidate_doi, api_key, args.cache_dir, args.refresh_cache
                    )
                    candidate_metadata = core_metadata(candidate_payload, entry.entry_type) if candidate_status == 200 else {}
                    candidate_overlap = author_overlap(entry.fields.get("author", empty).value, candidate_payload)
                    candidate_verified = (
                        candidate_status == 200
                        and candidate_metadata.get("doi") == candidate_doi
                        and title_similarity(bib_title, candidate_metadata.get("title", "")) >= args.search_title_threshold
                        and candidate_overlap >= args.author_threshold
                    )
                    if candidate_verified:
                        metadata = candidate_metadata
                        payload = candidate_payload
                        api_doi = candidate_doi
                        api_title = candidate_metadata.get("title", "")
                        similarity = title_similarity(bib_title, api_title)
                        overlap = candidate_overlap
                        status_code = candidate_status
                        source = "+".join(filter(None, (source, search_source, candidate_source)))
                        status = "verified_by_title_search" if not doi else "doi_repaired_by_title_search"
            unresolved_statuses = {"missing_doi", "not_found_in_scopus", "doi_mismatch", "metadata_mismatch"}
            crossref_search_status = 0
            crossref_search_source = ""
            if status in unresolved_statuses:
                crossref_search_status, crossref_payload, crossref_search_source = search_crossref(
                    bib_title, args.cache_dir, args.refresh_cache
                )
                crossref_candidates = (
                    crossref_search_candidates(crossref_payload, bib_title, bib_year)
                    if crossref_search_status == 200 else []
                )
                if crossref_candidates:
                    crossref_best = crossref_candidates[0]
                    crossref_similarity = float(crossref_best["similarity"])
                    if crossref_similarity > candidate_similarity:
                        candidate_doi = str(crossref_best["doi"])
                        candidate_title = str(crossref_best["title"])
                        candidate_similarity = crossref_similarity
                    if (
                        crossref_similarity >= args.search_title_threshold
                        and bool(crossref_best["year_match"])
                    ):
                        crossref_doi = str(crossref_best["doi"])
                        direct_status, direct_payload, direct_source = retrieve_crossref(
                            crossref_doi, args.cache_dir, args.refresh_cache
                        )
                        direct_metadata = (
                            crossref_metadata(direct_payload, entry.entry_type) if direct_status == 200 else {}
                        )
                        direct_similarity = title_similarity(bib_title, direct_metadata.get("title", ""))
                        direct_overlap = crossref_author_overlap(
                            entry.fields.get("author", empty).value, direct_payload
                        )
                        direct_authors = direct_payload.get("message", {}).get("author", [])
                        authors_verified = direct_overlap >= args.author_threshold
                        metadata_only_verified = (
                            not direct_authors
                            and direct_similarity >= 0.98
                            and bool(crossref_best["year_match"])
                        )
                        if (
                            direct_status == 200
                            and direct_metadata.get("doi") == crossref_doi
                            and direct_similarity >= args.search_title_threshold
                            and (authors_verified or metadata_only_verified)
                        ):
                            metadata = direct_metadata
                            api_doi = crossref_doi
                            api_title = direct_metadata.get("title", "")
                            similarity = direct_similarity
                            overlap = direct_overlap
                            registry_status = direct_status
                            registry_source = "+".join((crossref_search_source, direct_source))
                            source = registry_source
                            status = (
                                "verified_by_crossref_title_search"
                                if not doi else "doi_repaired_by_crossref_title_search"
                            )
            if status in unresolved_statuses and candidate_doi:
                status = f"{status}_candidate_only"
        else:
            crossref_search_status = 0
            crossref_search_source = ""

        populated: list[str] = []
        verified_statuses = {
            "verified", "verified_by_crossref", "verified_by_title_search",
            "doi_repaired_by_title_search", "verified_by_crossref_title_search",
            "doi_repaired_by_crossref_title_search",
        }
        if status in verified_statuses and args.apply:
            verified_doi = api_doi or doi
            entry_updates = metadata_updates(entry, metadata, verified_doi, args.refresh_existing)
            if entry_updates:
                updates[entry.key] = entry_updates
            populated = sorted(entry_updates)

        rows.append({
            "citation_key": entry.key, "entry_type": entry.entry_type, "bib_doi": doi,
            "api_doi": api_doi, "status": status, "http_status": status_code,
            "title_similarity": f"{similarity:.4f}", "author_overlap": f"{overlap:.4f}",
            "bib_title": bib_title, "api_title": api_title,
            "bib_year": bib_year, "api_year": metadata.get("year", ""),
            "candidate_doi": candidate_doi, "candidate_title": candidate_title,
            "candidate_title_similarity": f"{candidate_similarity:.4f}" if candidate_doi else "",
            "search_http_status": search_status or "", "source": source,
            "registry_http_status": registry_status or "", "registry_source": registry_source,
            "crossref_search_http_status": crossref_search_status or "",
            "crossref_search_source": crossref_search_source,
            "populated_fields": ";".join(populated), "notes": "",
        })
        if args.delay and source == "api":
            time.sleep(args.delay)

    if args.apply and updates:
        updated_text = apply_updates(original, entries, updates)
        if updated_text != original:
            bib_path.write_text(updated_text)

    fieldnames = [
        "citation_key", "entry_type", "bib_doi", "api_doi", "status", "http_status",
        "title_similarity", "author_overlap", "bib_title", "api_title", "bib_year", "api_year",
        "candidate_doi", "candidate_title", "candidate_title_similarity", "search_http_status",
        "source", "registry_http_status", "registry_source", "crossref_search_http_status",
        "crossref_search_source", "populated_fields", "notes",
    ]
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    counts: dict[str, int] = {}
    for row in rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "bibliography": str(bib_path.relative_to(ROOT)),
        "bibliography_sha256_before_update": hashlib.sha256(original.encode()).hexdigest(),
        "api_endpoint": API_TEMPLATE,
        "search_endpoint": SEARCH_ENDPOINT,
        "crossref_fallback_endpoint": CROSSREF_TEMPLATE,
        "crossref_search_endpoint": CROSSREF_SEARCH_ENDPOINT,
        "entries_total": len(entries),
        "entries_with_doi_before_update": sum(1 for entry in entries if "doi" in entry.fields),
        "entries_with_doi_after_update": sum(
            1 for entry in entries if "doi" in entry.fields or "doi" in updates.get(entry.key, {})
        ),
        "status_counts": counts,
        "duplicate_dois": {doi: keys for doi, keys in duplicate_dois.items() if len(keys) > 1},
        "title_similarity_threshold": args.title_threshold,
        "search_title_similarity_threshold": args.search_title_threshold,
        "author_overlap_threshold": args.author_threshold,
        "applied": bool(args.apply),
        "updated_entries": sorted(updates),
        "api_key_persisted_in_report": False,
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"status_counts": counts, "updated_entries": len(updates), "report": str(args.report)}, indent=2))
    hard_failures = ("authentication_or_entitlement_error", "quota_exceeded", "api_error")
    unresolved = sum(
        count for status, count in counts.items()
        if status.startswith(("not_found_in_scopus", "doi_mismatch", "metadata_mismatch"))
    )
    return 1 if unresolved or any(counts.get(status, 0) for status in hard_failures) else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bib", type=Path, default=DEFAULT_BIB)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--title-threshold", type=float, default=0.78)
    parser.add_argument("--search-title-threshold", type=float, default=0.93)
    parser.add_argument("--author-threshold", type=float, default=0.50)
    parser.add_argument("--delay", type=float, default=0.1)
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument("--apply", action="store_true", help="Normalize verified DOIs and populate missing metadata fields.")
    parser.add_argument("--refresh-existing", action="store_true", help="Refresh existing non-title metadata for verified entries.")
    return audit(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
