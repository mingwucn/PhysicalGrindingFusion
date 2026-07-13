"""Hard-coded Nature-style publication figure utilities.

This module provides a reusable ``PublicationPlotter`` class that configures
matplotlib according to the project's hard-coded figure guidelines (derived
from ``docs/12_VISUALIZATION_CONTRACT.md`` and the ``MyVis`` repository).

Typical usage::

    from src.grinding_physic_fusion.visualization import PublicationPlotter

    PublicationPlotter.set_style()
    fig, ax = PublicationPlotter.create_figure(width_mm=89)
    ax.plot(...)
    PublicationPlotter.savefig(fig, "my_figure.png")
"""

from __future__ import annotations

import shutil
import warnings
import json
from pathlib import Path
from typing import Iterable, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure


class PublicationPlotter:
    """Configure matplotlib and save figures for publication."""

    _ROOT = Path(__file__).resolve().parents[3]
    _REPORTS_PLOTS = _ROOT / "reports" / "evidence" / "plots"
    _OVERLEAF_IMAGES = _ROOT / "overleaf" / "images"

    # Nature-style color cycle (colorblind-friendly, solid)
    _COLOR_CYCLE = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]
    LEGEND_SIZE = 6
    PANEL_LABEL_SIZE = 8
    POINT_SIZE = 78
    BUBBLE_LABEL_SIZE = 7

    @classmethod
    def color_cycle(cls) -> tuple[str, ...]:
        """Return the project-wide categorical colour cycle."""
        return tuple(cls._COLOR_CYCLE)

    @classmethod
    def set_style(cls, journal: str = "nature", usetex: bool = False) -> None:
        """Configure matplotlib for publication-quality output.

        Parameters
        ----------
        journal:
            Target journal style. Currently only ``"nature"`` is fully
            supported.
        usetex:
            Whether to use LaTeX text rendering. Defaults to ``False`` because
            the ``ai`` conda environment on this machine does not have a
            working LaTeX format. When a working LaTeX installation is
            available, set this to ``True`` to enable the ``cmbright`` sans-serif
            math font.
        """
        if journal not in {"nature", "ieee", "default"}:
            raise ValueError(f"Unsupported journal style: {journal}")

        matplotlib.rcParams.update(matplotlib.rcParamsDefault)

        base_preamble = r"\usepackage{cmbright}"

        settings: dict = {
            # Font settings (Nature: 5-7 pt, panel labels 8 pt bold)
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7,
            "axes.labelsize": 7,
            "axes.titlesize": 7,
            "legend.fontsize": 6,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            # Math rendering
            "text.usetex": usetex,
            "mathtext.fontset": "dejavusans",
            "mathtext.default": "regular",
            # High-resolution output
            "figure.dpi": 900,
            "savefig.dpi": 900,
            # Font embedding (TrueType 42)
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            # Line and axis styling
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.minor.width": 0.4,
            "ytick.minor.width": 0.4,
            "lines.linewidth": 0.8,
            "patch.linewidth": 0.5,
            # Accessible color cycle
            "axes.prop_cycle": plt.cycler(color=cls._COLOR_CYCLE),
            # Layout (do not force constrained_layout; legacy tight_layout calls are common)
            "figure.constrained_layout.use": False,
            "figure.autolayout": False,
            # Avoid noisy grids by default
            "axes.grid": False,
        }

        if usetex:
            settings["text.latex.preamble"] = base_preamble

        matplotlib.rcParams.update(settings)

    @staticmethod
    def fig_size(
        width_mm: float = 89,
        ratio: float = 0.618,
        max_height_mm: float = 170,
    ) -> Tuple[float, float]:
        """Return figure size in inches for a given column width.

        Nature standard widths: 89 mm (single), 183 mm (double).
        """
        if width_mm not in (89, 183):
            warnings.warn(
                f"Width {width_mm} mm is not a standard Nature width "
                "(89 or 183 mm).",
                UserWarning,
                stacklevel=2,
            )

        width_in = width_mm / 25.4
        height_in = width_in * ratio
        height_mm = height_in * 25.4

        if height_mm > max_height_mm:
            raise ValueError(
                f"Figure height {height_mm:.1f} mm exceeds the maximum "
                f"{max_height_mm} mm. Reduce width or ratio."
            )

        return width_in, height_in

    @classmethod
    def create_figure(
        cls,
        width_mm: float = 89,
        ratio: float = 0.618,
        nrows: int = 1,
        ncols: int = 1,
        **kwargs,
    ) -> Tuple[Figure, ...]:
        """Create a figure with publication-standard dimensions."""
        figsize = cls.fig_size(width_mm, ratio)
        return plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, **kwargs)

    @staticmethod
    def add_panel_label(
        ax,
        label: str,
        x: float = -0.12,
        y: float = 1.04,
        **kwargs,
    ) -> None:
        """Add an 8-pt bold lowercase panel label (a, b, c, ...)."""
        defaults = {
            "fontsize": PublicationPlotter.PANEL_LABEL_SIZE,
            "fontweight": "bold",
            "style": "normal",
            "va": "top",
            "ha": "right",
            "transform": ax.transAxes,
            "clip_on": False,
            "color": "black",
        }
        defaults.update(kwargs)
        ax.text(x, y, label, **defaults)

    @staticmethod
    def comparison_label(model_a: str, model_b: str) -> str:
        """Return the project-wide compact pair label used on categorical axes."""
        return f"[{model_a}]\nvs\n[{model_b}]"

    @staticmethod
    def figure_legend_below(fig: Figure, handles: Sequence, labels: Sequence[str], *, ncol: int) -> None:
        """Place one shared legend below the axes, outside all data regions."""
        fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.01),
            ncol=ncol,
            frameon=False,
            fontsize=PublicationPlotter.LEGEND_SIZE,
            handlelength=1.8,
            columnspacing=1.2,
        )

    @classmethod
    def savefig(
        cls,
        fig: Figure,
        name: str,
        out_dir: Path | None = None,
        overleaf_dir: Path | None = None,
        formats: Sequence[str] = ("pdf", "png"),
        dpi: int = 900,
        facecolor: str = "white",
        close: bool = True,
    ) -> dict:
        """Save a figure in publication formats and copy PNG to Overleaf.

        Parameters
        ----------
        fig:
            The matplotlib figure to save.
        name:
            Output filename. If it contains an extension, that extension is
            ignored and replaced by ``formats``.
        out_dir:
            Directory for the canonical evidence plots. Defaults to
            ``reports/evidence/plots``.
        overleaf_dir:
            Directory to copy the PNG into for Overleaf compilation. Defaults
            to ``overleaf/images``.
        formats:
            Output formats. PDF is kept as the editable publication master;
            PNG is copied to Overleaf so existing LaTeX references stay valid.
        dpi:
            Resolution for raster output. The default is 900 DPI per the
            project guidelines.
        facecolor:
            Figure background color.
        close:
            Whether to close the figure after saving.

        Returns
        -------
        dict:
            Mapping of format to saved path.
        """
        out_dir = (out_dir or cls._REPORTS_PLOTS).resolve()
        overleaf_dir = overleaf_dir or cls._OVERLEAF_IMAGES
        out_dir.mkdir(parents=True, exist_ok=True)
        overleaf_dir.mkdir(parents=True, exist_ok=True)

        stem = Path(name).stem
        saved: dict = {}

        for fmt in formats:
            path = out_dir / f"{stem}.{fmt}"
            fig.savefig(
                path,
                dpi=dpi if fmt == "png" else None,
                bbox_inches=None,
                facecolor=facecolor,
                format=fmt,
            )
            saved[fmt] = path
            print(f"[PublicationPlotter] Saved {path}")

            if fmt == "png":
                overleaf_path = overleaf_dir / f"{stem}.png"
                shutil.copyfile(path, overleaf_path)
                print(f"[PublicationPlotter] Copied PNG to {overleaf_path}")

        if close:
            plt.close(fig)

        return saved

    @classmethod
    def savefig_to_category(
        cls,
        fig: Figure,
        name: str,
        category: str,
        **kwargs,
    ) -> dict:
        """Convenience wrapper that saves under ``reports/evidence/plots/<category>``.

        The PNG is still copied to ``overleaf/images``.
        """
        out_dir = cls._REPORTS_PLOTS / category
        return cls.savefig(fig, name, out_dir=out_dir, **kwargs)

    @classmethod
    def record_figure_metadata(
        cls,
        name: str,
        *,
        profile: dict,
        metadata: dict | None = None,
        out_dir: Path | None = None,
    ) -> Path:
        """Persist render provenance used by the figure-style compliance check."""
        out_dir = (out_dir or cls._REPORTS_PLOTS).resolve()
        manifest_dir = cls._ROOT / "reports" / "evidence" / "figure_metadata"
        manifest_dir.mkdir(parents=True, exist_ok=True)
        path = manifest_dir / f"{Path(name).stem}.json"
        payload = {
            "artifact": Path(name).stem,
            "profile": profile,
            "output_directory": str(out_dir.relative_to(cls._ROOT)),
            "palette": "PublicationPalette",
            "metadata": metadata or {},
        }
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return path
