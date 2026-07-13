# DOI reference verification

The manuscript bibliography is audited with Elsevier's Scopus Abstract
Retrieval API using each entry's DOI:

```text
https://api.elsevier.com/content/abstract/doi/{doi}
```

## Credentials

The API credential is secret and must not be stored in source code,
documentation, generated reports, or Git history. Set it through either:

```bash
export ELSEVIER_API_KEY="..."
```

or an untracked `.env` file based on `.env.example`.

## Audit and population

```bash
python scripts/audit_references_elsevier.py
python scripts/audit_references_elsevier.py --apply
```

The first command validates DOI and title agreement and writes:

- `reports/evidence/tables/reference_doi_audit.csv`
- `reports/evidence/tables/reference_doi_audit_summary.json`

The `--apply` command normalizes verified DOI values and fills missing journal,
booktitle, year, volume, issue, page, and ISSN fields. Existing titles and
authors are not overwritten. Entries without a DOI are reported rather than
assigned an unverified related paper. API responses are cached under `.cache/`,
which is excluded from Git.

If DOI retrieval fails, the script queries the Elsevier Scopus Search API by
title. It automatically repairs or adds a DOI only when the candidate title
and authors exceed the strict documented thresholds and the candidate DOI is
then independently retrievable through the Abstract Retrieval API. Lower-score
related papers are written to the audit as review candidates and are never
silently substituted, because a different paper may not support the manuscript
claim attached to the original citation.

An entry is marked `verified` only when Elsevier returns the same normalized
DOI and the API title agrees with the BibTeX title above the documented
similarity threshold. If a DOI is absent from Scopus, the script checks the
Crossref DOI registry at `https://api.crossref.org/works/{doi}` and records a
matching title as `verified_by_crossref`. This registry fallback avoids
misclassifying valid non-Scopus publications. Crossref is not used to replace
a DOI-less citation with a merely related work.

The audit is intentionally conservative. A relevant search result is not an
interchangeable citation: replacement is automatic only when the title and
author match identifies the same work and its DOI can be retrieved directly.
All other candidates remain visible for manual review.

## DOI-free substitution review

After the automated audit, each DOI-free citation is reviewed in the context
of the manuscript claim it supports. A DOI-bearing source replaces it only
when the claim can be preserved without broadening or changing the evidence.
The decisions are recorded in
`reports/evidence/tables/non_doi_substitution_review.csv`.

Four sources were replaced by DOI-bearing publications after their citing
sentences were checked and, where necessary, narrowed. Four foundational
conference papers remain because no DOI version was found and the returned
DOI-bearing candidates were different works. Those retained references use
stable PMLR, NeurIPS, or arXiv locators. DOI coverage is therefore not pursued
at the cost of citation accuracy.

## Proposed-reference inclusion

Author-supplied candidate publications are handled separately:

```bash
python scripts/include_verified_related_references.py
python scripts/include_verified_related_references.py --apply
```

The script verifies every supplied DOI but inserts only candidates with an
explicit claim-level mapping in the manuscript. Verification alone is not a
reason to cite a paper. The generated
`reports/evidence/tables/suggested_reference_inclusion.csv` records the title
match, inclusion decision, and scientific rationale for every candidate.
Candidates without a final DOI or outside the direct scope remain excluded.
