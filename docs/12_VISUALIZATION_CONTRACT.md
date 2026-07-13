# Visualization Contract

## Purpose

This document defines structured visualization contracts for scientific and engineering projects.

Visualizations are scientific artifacts. They should be generated through validated requests rather than arbitrary plotting code.

The central rule is:

> A figure should have a hypothesis link, caption, units, provenance, and validated output format before it becomes evidence.

## 1. Why visualization requires a contract

Visualizations are often used for:

- dataset description;
- data-quality review;
- hypothesis exploration;
- model validation;
- failure-case analysis;
- novelty and impact communication;
- publication figures.

Without a contract, figures can become inconsistent, undocumented, or scientifically ambiguous.

## 2. Core data models

```python
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence, Protocol


@dataclass(frozen=True)
class VisualizationRequest:
    hypothesis_id: str
    title: str
    visual_type: str
    data: Mapping[str, Any]
    caption: str
    output_basename: str
    axis_labels: Mapping[str, str] = field(default_factory=dict)
    units: Mapping[str, str] = field(default_factory=dict)
    preferred_format: str | None = None
    latex_required: bool | None = None


@dataclass(frozen=True)
class VisualizationArtifact:
    path: Path
    format: str
    dpi: int | None = None
    latex_mode: str | None = None
    notes: Sequence[str] = field(default_factory=tuple)


@dataclass(frozen=True)
class VisualizationResult:
    request: VisualizationRequest
    artifacts: Sequence[VisualizationArtifact]
    compliance_notes: Sequence[str] = field(default_factory=tuple)


class VisualizationRenderer(Protocol):
    def render(
        self,
        request: VisualizationRequest,
        config: Mapping[str, Any],
    ) -> VisualizationResult:
        ...
```

## 3. Validation function

```python
def validate_visualization_request(
    request: VisualizationRequest,
    config: Mapping[str, Any],
) -> list[str]:
    errors: list[str] = []

    visualization_cfg = config.get("visualization", {})
    evidence_cfg = config.get("evidence", {})
    interface_cfg = visualization_cfg.get("interface", {})

    allowed_visual_types = set(interface_cfg.get("allowed_visual_types", []))
    if allowed_visual_types and request.visual_type not in allowed_visual_types:
        errors.append(
            f"visual_type '{request.visual_type}' is not allowed; "
            f"expected one of {sorted(allowed_visual_types)}"
        )

    allowed_formats = set(evidence_cfg.get("allowed_artifact_types", []))
    if request.preferred_format and allowed_formats and request.preferred_format not in allowed_formats:
        errors.append(
            f"preferred_format '{request.preferred_format}' is not allowed; "
            f"expected one of {sorted(allowed_formats)}"
        )

    required_fields = interface_cfg.get("required_request_fields", [])
    for field_name in required_fields:
        value = getattr(request, field_name, None)
        if value in (None, "", {}, []):
            errors.append(f"missing required visualization request field: {field_name}")

    if visualization_cfg.get("nature_guidance", {}).get("require_axis_units"):
        for axis_name in request.axis_labels:
            if axis_name not in request.units:
                errors.append(f"missing unit for axis '{axis_name}'")

    return errors
```

## 4. Supported visual types

Recommended generic visual types:

```text
line_plot
scatter_plot
bar_chart
heatmap
histogram
box_plot
violin_plot
residual_plot
calibration_plot
confusion_matrix
embedding_plot
time_series
spectrogram
feature_importance
```

The renderer does not need to support all types. It must fail clearly when a type is unsupported.

## 5. Renderer contract

A renderer should:

```text
1. validate the request before rendering.
2. use project_config.yaml for output directory, format, dpi, and style constraints.
3. write artifacts only under allowed evidence or figure directories.
4. return a VisualizationResult.
5. avoid hard-coded project-specific assumptions.
6. close figure handles after rendering.
7. preserve caption, units, and hypothesis link.
8. never invent data.
```

## 6. Matplotlib renderer contract

```python
class MatplotlibPlotRenderer(VisualizationRenderer):
    supported_visual_types = {
        "line_plot",
        "scatter_plot",
        "bar_chart",
        "heatmap",
        "histogram",
        "box_plot",
    }

    def render(
        self,
        request: VisualizationRequest,
        config: Mapping[str, Any],
    ) -> VisualizationResult:
        ...
```

### Implementation guidance

- Use `matplotlib.use("Agg")` for non-interactive rendering.
- Use project configuration for output format and DPI.
- Use project configuration for style and color policy.
- Do not hard-code a domain-specific colormap unless the project config allows it.
- Raise `ValueError` for unsupported visual types.
- Return explicit artifact metadata.

## 7. Visualization artifact as evidence

A visualization becomes scientific evidence only when linked to records.

Recommended links:

```yaml
schema_version: 1
record_type: VisualizationEvidence
record_id: VIZ-0001

hypothesis_id: H-0001
visualization_request_id: VIZREQ-0001
artifact_paths:
  - reports/evidence/plots/H-0001_residual_plot.png

caption: ""
interpretation: ""
review_status:
  options:
    - draft
    - reviewed
    - accepted
    - rejected
```

## 8. Human review

Figures used for scientific claims should be reviewable.

Human review questions:

```text
Is the visual type appropriate?
Are axes and units correct?
Is the caption accurate?
Does the figure support the linked hypothesis?
Are uncertainty and sample size visible where needed?
Is the figure misleading or over-interpreted?
```

## 9. Project configuration

Visualization should be configured through `project_config.yaml`.

```yaml
visualization:
  interface:
    allowed_visual_types:
      - line_plot
      - scatter_plot
      - bar_chart
      - heatmap
      - histogram
      - box_plot
      - residual_plot
      - calibration_plot
    required_request_fields:
      - hypothesis_id
      - title
      - visual_type
      - data
      - caption
      - output_basename

  output:
    default_format: pdf
    allowed_formats:
      - pdf
      - eps
      - png
      - svg
    dpi: 900
    vector_graphics_preferred: true

  style:
    use_project_style: true
    hardcoded_colors_allowed: false
    journal_style: nature

  latex:
    mode: required
    nature_preamble: |
      \usepackage{cmbright}

  nature_guidance:
    require_axis_units: true
    require_caption: true
    require_provenance: true
    font_family: sans-serif
    font_stack: [Arial, Helvetica, DejaVu Sans]
    text_size_min_pt: 5
    text_size_max_pt: 7
    panel_label_size_pt: 8
    panel_label_weight: bold
    font_embedding: 42
    max_height_mm: 170
    single_column_width_mm: 89
    double_column_width_mm: 183

evidence:
  allowed_artifact_types:
    - png
    - svg
    - pdf
    - json
    - yaml
    - csv
    - md
  plots_dir: reports/evidence/plots
```

## 10. Tests

```text
tests/test_visualization_interface.py
tests/test_matplotlib_renderer.py
tests/test_visualization_config.py
```

Minimal tests:

```text
1. valid request passes validation.
2. missing required field fails.
3. disallowed visual type fails.
4. disallowed format fails.
5. missing axis unit fails when required.
6. renderer writes artifact under configured directory.
7. unsupported visual type raises ValueError.
8. renderer closes figure handles.
9. result contains request and artifact metadata.
10. visualization evidence links to a hypothesis.
```

## 11. Implementation requirements

Do not accept arbitrary plotting scripts as final evidence. Exploratory figures may be generated separately, but accepted figures must pass through this visualization contract.

---

# Appendix: Hardcoded Nature-style figure guidelines

This appendix hard-codes the publication figure style used by this project.
The official Nature Research figure guide and Nature final-submission
specifications are the controlling sources. Where an older local convention
conflicts with those sources, the Nature requirement wins.

## A.1 Scope

All final figures for manuscripts, reports, and evidence must follow these rules unless a journal explicitly requires a different style. Exploratory plots are exempt but should be clearly marked as draft.

## A.2 Typography

| Element | Specification |
|---|---|
| Font family | Sans-serif: **Arial / Helvetica / DejaVu Sans** |
| Greek/math | LaTeX rendering with `\usepackage{cmbright}` |
| Base text size | **7 pt** (Nature maximum; never exceed 7 pt except panel labels) |
| Axis labels | **7 pt** |
| Tick labels | **6 pt** |
| Legend text | **6 pt** |
| Panel labels | **8 pt bold**, upright, lowercase: **a, b, c, …** |
| Text color | Black or dark gray only — **no colored text** |
| Outlining | **Never** outline text; keep editable for vector formats |
| Font embedding | `pdf.fonttype = 42` (TrueType 42) |

## A.3 Figure dimensions

| Layout | Width | Notes |
|---|---|---|
| Single column | **89 mm** | Default |
| Double column | **183 mm** | Wide multi-panel figures |
| Maximum height | **170 mm** | Leave room for caption |
| Aspect ratio | 0.618 (golden) unless content dictates otherwise |

### A.3.1 Dense ranking figures

Dense rankings of 15--25 labelled items must remain a single figure only when
they use the dedicated double-column profile:

| Requirement | Rule |
|---|---|
| Export width | **183 mm** |
| Maximum height | **170 mm** |
| Tick labels | **6 pt** at final print size |
| Value labels | **6--7 pt** at final print size, outside error-bar paths |
| Labels | Three lines: `[model]`, `vs`, `[configuration]`; abbreviate before shrinking text |
| LaTeX inclusion | `width=\linewidth`; the exported PDF retains the 183 mm design size |

### A.3.2 Three- and four-panel horizontal rows

Horizontal small-multiple layouts are valid only as explicit profiles. They
are not ad hoc exceptions to the standard dimensions.

| Profile | Width | Required constraints |
|---|---:|---|
| `three_panel_row` | **183 mm** | At most one legend and one colourbar; use shared outer labels where possible; no more than six tick labels per axis. |
| `four_panel_row` | **183 mm** | Shared x and y axes required; at most one legend and one colourbar; no more than five tick labels per axis; compact repeated measures only. |
| `top_span_two_bottom` | **183 mm** | One full-width overview panel above two equal diagnostic panels; use when the overview needs the full horizontal range and the diagnostics are compact. |

A four-panel row must be rejected in favour of a 2×2 grid when it needs long
tick labels, independent axis labels, individual legends, or more than one
colourbar. All panel labels remain 8 pt; axis labels remain 7 pt and ticks 6
pt at final print size. A row profile must be included at its intended width,
without `0.75`, `0.85`, or other LaTeX scale factors.

### A.3.3 Required OOP rendering path

Every final Python-generated figure must inherit from `PublicationFigure` or
be instantiated as a registered `MutableFigure` with a named `FigureProfile`.
The base class owns style setup, physical dimensions, export, and provenance;
individual figures may mutate only their data artists and justified layout
details. Direct standalone `matplotlib.pyplot.subplots` plus direct
`savefig` calls are not permitted in final figure generators.

The ancestor automatically labels every visible axis in a multi-panel figure
with 8 pt bold upright lowercase letters in row-major order. Do not add panel
titles or prose subcaptions inside a panel when the letter and the main figure
caption are sufficient; the ancestor clears axis titles at export to enforce
this rule. Each panel must be described in the LaTeX caption as
`\textbf{a}, ...`, `\textbf{b}, ...`. Legends must be outside data regions;
use one shared legend for repeated series across panels.

## A.4 File format and resolution

- **Preferred format**: **PDF** with embedded fonts.
- **Alternative**: EPS with embedded fonts.
- **Raster fallback**: PNG at **900 DPI** if vector output is impossible.
- Minimum acceptable raster resolution: **300 DPI**.
- Prefer vector graphics for line art, diagrams, and charts.

## A.5 Color and style

- Use accessible, colorblind-safe palettes.
- Heatmap annotation colour must be selected from rendered cell luminance so
  every label contrasts with its background; do not infer text colour from
  whether a value is numerically high or low.
- Use solid colors; **no patterns, hatching, or textures**.
- **No background gridlines** (exploratory gridlines may be removed before publication).
- **No drop shadows, gradients, or decorative icons**.
- Ensure sufficient contrast on white backgrounds.
- Avoid colored text labels; use black text with keys or keylines.

## A.6 Axes and labels

- Include axis lines and tick marks.
- Label every axis; place units in parentheses, e.g., `Roughness (µm)`.
- Avoid overlapping text and labels.
- Pairwise or model--configuration labels must use the shared three-line form
  `[A]`, `vs`, `[B]` on the categorical axis.
- Statistical figures that show the same pairwise family must use the shared
  comparison palette: blue (`PublicationPalette.OBSERVED`) for Model A or
  confidence intervals and orange
  (`PublicationPalette.MODEL_FAMILY["RidgeRegressionModel"]`) for Model B or
  paired point estimates. Do not introduce a separate significance palette;
  use symbols, opacity, or annotations for significance.
- Panel labels must be in alphabetical order, top-left of each panel.

## A.7 Recommended Matplotlib boilerplate

All Python figure-generation scripts must apply this configuration before plotting:

```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams.update({
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{cmbright}",
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7,
    "axes.labelsize": 7,
    "axes.titlesize": 7,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "legend.fontsize": 6,
    "figure.dpi": 900,
    "savefig.dpi": 900,
    "pdf.fonttype": 42,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "lines.linewidth": 0.8,
})
```

Use the helper below for panel labels:

```python
def add_panel_label(ax, label, x=-0.1, y=1.1):
    ax.text(x, y, r"\textbf{" + label + "}",
            fontsize=8, style="normal", va="top", ha="right",
            transform=ax.transAxes)
```

## A.8 Figure sizing helper

```python
def fig_size(width_mm=89, ratio=0.618, max_height_mm=170):
    width_in = width_mm / 25.4
    height_in = width_in * ratio
    if height_in * 25.4 > max_height_mm:
        raise ValueError(f"Figure height exceeds {max_height_mm} mm")
    return (width_in, height_in)
```

## A.9 Pre-submission checklist

- [ ] All text uses Arial / Helvetica / DejaVu Sans (sans-serif).
- [ ] No text exceeds 7 pt (panel labels 8 pt bold are allowed).
- [ ] No text is outlined; vector text remains editable.
- [ ] Fonts are embedded (`pdf.fonttype = 42`).
- [ ] No colored text labels.
- [ ] No background gridlines.
- [ ] No patterns, drop shadows, or decorative elements.
- [ ] Axes labelled with units in parentheses.
- [ ] Accessible color palette used.
- [ ] Figure width is 89 mm (single) or 183 mm (double).
- [ ] Figure height does not exceed 170 mm.
- [ ] Output is PDF/EPS (or PNG at 900 DPI if raster is unavoidable).

## A.10 Controlling sources

These are hard constraints derived from the official sources below:

- Nature Research, *Building and exporting figure panels*:
  https://research-figure-guide.nature.com/figures/building-and-exporting-figure-panels/
- Nature Research, *Preparing figures: our specifications*:
  https://research-figure-guide.nature.com/figures/preparing-figures-our-specifications/
- Nature, *Final submission*:
  https://www.nature.com/nature/for-authors/final-submission
- Nature Reviews, *Figure guidelines*:
  https://www.nature.com/documents/natrev-figure-guidelines-v1.pdf

Required interpretation: 89 mm and 183 mm are the only final widths; ordinary
text is 5--7 pt; panel letters are 8 pt bold upright lowercase; strokes are
0.25--1 pt; text remains editable and fonts embedded; panels are compact,
alphabetical, and free of overlap. This contract applies to Python figures and
native TikZ diagrams.

For Python multi-panel figures, the OOP ancestor reserves a gutter above each
axis and places the panel letter at the upper-left axes corner with a fixed
point offset. Panel letters must never share the y-tick-label region. Every
multi-panel manuscript caption must describe each panel explicitly using
`\textbf{a}`, `\textbf{b}`, and subsequent lowercase letters in row-major
order.
