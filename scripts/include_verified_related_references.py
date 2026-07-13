#!/usr/bin/env python3
"""Verify supplied publications and add only claim-relevant references.

The script is deliberately conservative: DOI verification does not imply that
a paper belongs in the manuscript. Each candidate has an explicit scope
decision, and citations are inserted only at predefined claims that the source
directly supports. Run without ``--apply`` to audit, then use ``--apply`` to
update ``overleaf/ref.bib`` and the mapped Related Work passages.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from audit_references_elsevier import (
    DEFAULT_CACHE,
    ROOT,
    apply_updates,
    clean_doi,
    core_metadata,
    escape_bibtex,
    load_api_key,
    parse_bibtex,
    retrieve_crossref,
    retrieve_elsevier,
    title_similarity,
)


REPORT = ROOT / "reports" / "evidence" / "tables" / "suggested_reference_inclusion.csv"
BIB_PATH = ROOT / "overleaf" / "ref.bib"


@dataclass(frozen=True)
class Candidate:
    key: str
    title: str
    doi: str = ""
    include: bool = False
    rationale: str = "Outside the direct scope of the grinding roughness benchmark."
    volume: str = ""
    number: str = ""
    pages: str = ""


CANDIDATES = (
    Candidate(
        "Chen2025PrecisionGrinding",
        "Predictive modelling of surface roughness in precision grinding based on hybrid algorithm",
        "10.1016/j.cirpj.2025.02.004",
        True,
        "Directly studies multimodal AE/vibration roughness prediction in precision grinding.",
        "59", "", "1--17",
    ),
    Candidate(
        "Wu2025ECMInterpretability",
        "Data-driven models with physical interpretability for real-time cavity profile prediction in electrochemical machining processes",
        "10.1016/j.engappai.2025.111807",
        True,
        "Direct manufacturing precedent for SHAP and Grad-CAM inspection of process predictions.",
        "160", "", "111807",
    ),
    Candidate(
        "Ge2025BLSVSG",
        "Tackling data scarcity in machine learning-based CFRP drilling performance prediction through a broad learning system with virtual sample generation (BLS-VSG)",
        "10.1016/j.compositesb.2025.112701",
        True,
        "Supports the discussion of small-data manufacturing prediction and augmentation.",
        "305", "", "112701",
    ),
    Candidate(
        "Shi2025FlexibleFixture",
        "Design and Manufacture of a Flexible Adaptive Fixture for Precision Grinding of Thin-Walled Bearing Rings",
        "10.3390/jmmp9050139",
        True,
        "Provides bearing-ring grinding context for clamping-induced deformation and fixture stability.",
        "9", "5", "139",
    ),
    Candidate(
        "Wu2022ECMProfile",
        "Profile prediction in ECM using machine learning",
        "10.1016/j.procir.2022.09.192",
        True,
        "Manufacturing precedent for combining process parameters and in-process data in profile prediction.",
        "113", "", "410--416",
    ),
    Candidate(
        "Wu2025LPBFAE",
        "Data-Driven Approach to Identify Acoustic Emission Source Motion and Positioning Effects in Laser Powder Bed Fusion with Frequency Analysis",
        "10.1016/j.procir.2025.02.091",
        True,
        "Supports the sensor-context caveat for frequency attribution in AE monitoring.",
        "133", "", "531--536",
    ),
    Candidate("Zhang2025MedicalSegmentation", "A Review of Non-Fully Supervised Deep Learning for Medical Image Segmentation", "10.3390/info16060433"),
    Candidate("Liu2020ECDGrinding", "Electrochemical Discharge Grinding of Metal Matrix Composites Using Shaped Abrasive Tools Formed by Sintered Bronze/diamond", "10.1515/secm-2020-0038"),
    Candidate("Wu2023MEJM", "Experimental and Numerical Investigations on Fabrication of Surface Microstructures Using Mask Electrolyte Jet Machining and Duckbill Nozzle", "10.1115/1.4056570"),
    Candidate("Wu2022MultiIonMEJM", "Multi-Ion-Based Modelling and Experimental Investigations on Consistent and High-Throughput Generation of a Micro Cavity Array by Mask Electrolyte Jet Machining", "10.3390/mi13122165"),
    Candidate("Wu2018JetECM", "Modeling and simulation of the material removal process in electrolyte jet machining of mass transfer in convection and electric migration", "10.1016/j.procir.2017.12.079"),
    Candidate("Wu2020MicroLetters", "Fabrication of Surface Micro Letters by Electrolyte Jet Mask Machining", "10.1016/j.procir.2020.02.261"),
    Candidate("Wu2025EDMGeometry", "Geometrical feature classification in electrical discharge machining using in-process monitoring and machine learning", "10.1016/j.procir.2025.07.010"),
    Candidate("Wu2025PulseClassification", "A Threshold-Free and Label-Free Pipeline for Adaptive Pulse Classification in Electrical Discharge Machining", "10.1016/j.procir.2025.02.118"),
    Candidate("Wu2025FFF", "Deep Learning-based characterization of fused filament fabrication from temporal thermal data", "10.1016/j.procir.2025.02.098"),
    Candidate("Yao2025RFDischarge", "Intelligent discharge state detection in micro-EDM process with cost-effective radio frequency (RF) radiation: Integrating machine learning and interpretable AI", rationale="No DOI or final bibliographic record was supplied."),
    Candidate("Yao2024Crater", "Prediction of crater morphology and its application for enhancing dimensional accuracy in micro-EDM", rationale="No DOI was supplied; outside the direct benchmark scope."),
    Candidate("LinSecureDet", "The Enhance-Fuse-Align Principle: A New Architectural Blueprint for Robust Object Detection, with Application to X-Ray Security", rationale="No final publication record or DOI was supplied."),
    Candidate("ShaoCFTADRC", "Composite finite-time ADRC for flexible-joint manipulators with frequency-domain separation", rationale="No final publication record or DOI was supplied."),
    Candidate("ShaoAeroPINN", "An aero-thermodynamic physics-informed neural network for small-sample performance prediction of variable-speed centrifugal Chillers", rationale="No final publication record or DOI was supplied."),
    Candidate("WuMaskElectrolyteJet", "Fabrication of surface microstructures by mask electrolyte jet machining", rationale="No DOI was supplied and the topic is outside the direct benchmark scope."),
)


INSERTIONS = (
    (
        ROOT / "overleaf" / "main" / "Related_work.tex",
        "In-process sensing is consequently attractive because it can provide information on process evolution before offline metrology is available.",
        " For thin-walled bearing rings, fixture-induced deformation is an additional source of geometric variation, and adaptive fixture design can improve clamping stability during precision grinding \\cite{Shi2025FlexibleFixture}.",
        ("Shi2025FlexibleFixture",),
    ),
    (
        ROOT / "overleaf" / "main" / "Related_work.tex",
        "Empirical grinding studies have shown that vibration features can correlate with surface roughness in cylindrical grinding, supporting the use of vibration as a roughness-prediction modality \\cite{Hassui2003}.",
        " A recent bearing outer-ring study combined vibration, AE, and grinding parameters in an optimized 1D-CNN--LSTM and reported strong within-study roughness-prediction accuracy \\cite{Chen2025PrecisionGrinding}.",
        ("Chen2025PrecisionGrinding",),
    ),
    (
        ROOT / "overleaf" / "main" / "Related_work.tex",
        "The present work therefore treats deep models as evaluated baselines, not as a class to be rejected. The comparison is intended to identify which representation--model combinations generalize under strict condition-level exclusion.",
        " In other manufacturing settings, virtual sample generation has been combined with broad learning systems to mitigate small experimental datasets, although synthetic samples do not replace grouped external validation \\cite{Ge2025BLSVSG}.",
        ("Ge2025BLSVSG",),
    ),
    (
        ROOT / "overleaf" / "main" / "Related_work.tex",
        "Broader XAI reviews emphasize that such methods improve model inspection but do not by themselves establish causal mechanisms \\cite{Arrieta2020}.",
        " Manufacturing studies have similarly combined process parameters and in-process signals for profile prediction \\cite{Wu2022ECMProfile} and used SHAP and Grad-CAM for global and local inspection of machining models \\cite{Wu2025ECMInterpretability}.",
        ("Wu2022ECMProfile", "Wu2025ECMInterpretability"),
    ),
    (
        ROOT / "overleaf" / "main" / "Related_work.tex",
        "Attributions above the nominal flat-response limit of the AE sensor should therefore be reported as model-relevant spectral regions requiring sensor-transfer-function verification rather than as direct evidence of high-frequency process physics.",
        " Related AE monitoring work has shown that source motion and sensor position can themselves produce classifiable frequency signatures, reinforcing the need to separate process information from acquisition context \\cite{Wu2025LPBFAE}.",
        ("Wu2025LPBFAE",),
    ),
)


def api_authors(payload: dict[str, Any]) -> str:
    authors = payload.get("abstracts-retrieval-response", {}).get("authors", {}).get("author", [])
    if isinstance(authors, dict):
        authors = [authors]
    authors = sorted(authors, key=lambda item: int(item.get("@seq", 9999)))
    formatted = []
    for author in authors:
        surname = str(author.get("ce:surname", "")).strip()
        given = str(author.get("ce:given-name", author.get("preferred-name", {}).get("ce:given-name", ""))).strip()
        if surname:
            formatted.append(f"{surname}, {given}" if given else surname)
    return " and ".join(formatted)


def desired_fields(candidate: Candidate, payload: dict[str, Any]) -> list[tuple[str, str]]:
    metadata = core_metadata(payload, "article")
    return [
        ("author", api_authors(payload)),
        ("title", metadata.get("title", candidate.title)),
        ("journal", metadata.get("journal", "")),
        ("year", metadata.get("year", "")),
        ("volume", candidate.volume or metadata.get("volume", "")),
        ("number", candidate.number or metadata.get("number", "")),
        ("pages", candidate.pages or metadata.get("pages", "")),
        ("doi", clean_doi(candidate.doi)),
        ("url", f"https://doi.org/{clean_doi(candidate.doi)}"),
    ]


def bibtex_entry(candidate: Candidate, payload: dict[str, Any]) -> str:
    fields = desired_fields(candidate, payload)
    lines = [f"@article{{{candidate.key},"]
    present = [(name, escape_bibtex(value)) for name, value in fields if value]
    for index, (name, value) in enumerate(present):
        comma = "," if index + 1 < len(present) else ""
        lines.append(f"  {name} = {{{value}}}{comma}")
    lines.append("}")
    return "\n".join(lines)


def apply_insertions(included_keys: set[str]) -> list[str]:
    changed: list[str] = []
    for path, anchor, addition, required_keys in INSERTIONS:
        if not set(required_keys).issubset(included_keys):
            continue
        text = path.read_text()
        if all(f"{{{key}}}" in text for key in required_keys):
            continue
        if anchor not in text:
            raise RuntimeError(f"Citation anchor not found in {path.relative_to(ROOT)}: {anchor[:80]}")
        path.write_text(text.replace(anchor, anchor + addition, 1))
        changed.append(str(path.relative_to(ROOT)))
    return changed


def run(args: argparse.Namespace) -> int:
    api_key = load_api_key()
    rows: list[dict[str, Any]] = []
    verified_payloads: dict[str, dict[str, Any]] = {}
    for candidate in CANDIDATES:
        status_code = 0
        source = ""
        api_title = ""
        similarity = 0.0
        verified = False
        if candidate.doi:
            status_code, payload, source = retrieve_elsevier(
                clean_doi(candidate.doi), api_key, args.cache_dir, args.refresh_cache
            )
            if status_code == 200:
                metadata = core_metadata(payload, "article")
                api_title = metadata.get("title", "")
                similarity = title_similarity(candidate.title, api_title)
                verified = metadata.get("doi") == clean_doi(candidate.doi) and similarity >= args.title_threshold
            elif status_code == 404:
                status_code, payload, source = retrieve_crossref(
                    clean_doi(candidate.doi), args.cache_dir, args.refresh_cache
                )
                message = payload.get("message", {}) if status_code == 200 else {}
                titles = message.get("title", [])
                api_title = str(titles[0]) if titles else ""
                similarity = title_similarity(candidate.title, api_title)
                verified = clean_doi(str(message.get("DOI", ""))) == clean_doi(candidate.doi) and similarity >= args.title_threshold
            if verified:
                verified_payloads[candidate.key] = payload
        decision = "include" if candidate.include and verified else "exclude"
        if candidate.include and not verified:
            decision = "blocked_unverified"
        rows.append({
            "citation_key": candidate.key,
            "doi": clean_doi(candidate.doi),
            "expected_title": candidate.title,
            "api_title": api_title,
            "title_similarity": f"{similarity:.4f}" if candidate.doi else "",
            "http_status": status_code or "",
            "verified": verified,
            "decision": decision,
            "source": source,
            "rationale": candidate.rationale,
        })

    blocked = [row["citation_key"] for row in rows if row["decision"] == "blocked_unverified"]
    if blocked:
        raise RuntimeError(f"Required references failed DOI/title verification: {blocked}")

    changed: list[str] = []
    if args.apply:
        bib_text = BIB_PATH.read_text()
        parsed_entries = parse_bibtex(bib_text)
        entries_by_key = {entry.key: entry for entry in parsed_entries}
        existing = set(entries_by_key)
        additions = []
        field_updates: dict[str, dict[str, str]] = {}
        for candidate in CANDIDATES:
            if not candidate.include:
                continue
            if candidate.key not in existing:
                additions.append(bibtex_entry(candidate, verified_payloads[candidate.key]))
                continue
            entry = entries_by_key[candidate.key]
            missing = {
                name: value for name, value in desired_fields(candidate, verified_payloads[candidate.key])
                if value and name not in entry.fields
            }
            if missing:
                field_updates[candidate.key] = missing
        if field_updates:
            bib_text = apply_updates(bib_text, parsed_entries, field_updates)
        if additions:
            bib_text = bib_text.rstrip() + "\n\n" + "\n\n".join(additions) + "\n"
        if field_updates or additions:
            BIB_PATH.write_text(bib_text)
            changed.append(str(BIB_PATH.relative_to(ROOT)))
        included_keys = {candidate.key for candidate in CANDIDATES if candidate.include}
        changed.extend(apply_insertions(included_keys))

    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print({
        "verified_doi_records": sum(bool(row["verified"]) for row in rows),
        "included_records": sum(row["decision"] == "include" for row in rows),
        "changed_files": sorted(set(changed)),
        "report": str(args.report),
    })
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--report", type=Path, default=REPORT)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument("--title-threshold", type=float, default=0.90)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
