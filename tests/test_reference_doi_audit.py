import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from audit_references_elsevier import (  # noqa: E402
    apply_updates,
    clean_doi,
    crossref_metadata,
    crossref_search_candidates,
    parse_bibtex,
    search_candidates,
    title_similarity,
)


def test_parser_handles_nested_braces_and_allows_metadata_insertion() -> None:
    source = """@article{sample,
  title = {A {Nested} Title},
  author = {Doe, Jane},
  doi = {https://doi.org/10.1000/ABC}
}
"""
    entries = parse_bibtex(source)
    assert len(entries) == 1
    assert entries[0].fields["title"].value == "A {Nested} Title"

    updated = apply_updates(
        source,
        entries,
        {"sample": {"doi": "10.1000/xyz-longer", "year": "2025"}},
    )
    reparsed = parse_bibtex(updated)[0]
    assert reparsed.fields["doi"].value == "10.1000/xyz-longer"
    assert reparsed.fields["year"].value == "2025"


def test_doi_and_title_normalization() -> None:
    assert clean_doi(" DOI: https://doi.org/10.1000/ABC. ") == "10.1000/abc"
    assert title_similarity("A {Physics-Aware} Model", "A Physics-Aware Model") == 1.0


def test_search_candidates_rank_exact_title_first() -> None:
    payload = {
        "search-results": {
            "entry": [
                {"dc:title": "A related but different study", "prism:doi": "10.1/other", "prism:coverDate": "2024-01-01"},
                {"dc:title": "Exact Grinding Study", "prism:doi": "10.1/exact", "prism:coverDate": "2024-03-01"},
            ]
        }
    }
    candidates = search_candidates(payload, "Exact Grinding Study", "2024")
    assert candidates[0]["doi"] == "10.1/exact"
    assert candidates[0]["similarity"] == 1.0


def test_crossref_metadata_parsing() -> None:
    payload = {
        "message": {
            "DOI": "10.1000/ABC",
            "title": ["Verified title"],
            "container-title": ["Journal"],
            "issued": {"date-parts": [[2024, 2, 1]]},
            "volume": "12",
            "issue": "3",
            "page": "4-9",
            "ISSN": ["1234-5678"],
        }
    }
    metadata = crossref_metadata(payload, "article")
    assert metadata["doi"] == "10.1000/abc"
    assert metadata["title"] == "Verified title"
    assert metadata["journal"] == "Journal"
    assert metadata["year"] == "2024"

    candidates = crossref_search_candidates(
        {"message": {"items": [payload["message"]]}}, "Verified title", "2024"
    )
    assert candidates[0]["doi"] == "10.1000/abc"
    assert candidates[0]["year_match"] is True
