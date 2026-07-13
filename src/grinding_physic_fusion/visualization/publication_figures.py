"""Object-oriented publication figure framework.

Final figures are built through this module so their physical size,
typography, palette, layout, and export metadata are governed in one place.
Concrete figure classes override :meth:`draw`; layout-specific subclasses
enforce the few sanctioned mutations needed for dense rankings and panel rows.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from .publication_plotter import PublicationPlotter


@dataclass(frozen=True)
class FigureProfile:
    """Validated physical layout for a final figure."""

    name: str
    width_mm: int
    ratio: float
    nrows: int = 1
    ncols: int = 1
    sharex: bool = False
    sharey: bool = False
    max_legends: int = 1
    max_colorbars: int = 1
    max_ticks_per_axis: int | None = None
    require_shared_labels: bool = False


class FigureProfiles:
    """The only physical layouts permitted for final figures."""

    SINGLE = FigureProfile("single", 89, 0.618)
    SQUARE = FigureProfile("square", 89, 1.0)
    WIDE = FigureProfile("wide", 183, 0.50)
    DOUBLE = FigureProfile("double", 183, 0.618)
    DOUBLE_TALL = FigureProfile("double_tall", 183, 0.78)
    TWO_PANEL_ROW = FigureProfile("two_panel_row", 183, 0.46, ncols=2, max_legends=1, max_colorbars=1)
    TWO_PANEL_TALL = FigureProfile("two_panel_tall", 183, 0.75, ncols=2, max_legends=1, max_colorbars=1)
    DENSE_RANKING = FigureProfile("dense_ranking", 183, 0.89, max_ticks_per_axis=20)
    THREE_PANEL_ROW = FigureProfile(
        "three_panel_row", 183, 0.40, ncols=3, sharey=False,
        max_legends=1, max_colorbars=1, max_ticks_per_axis=6,
        require_shared_labels=True,
    )
    THREE_PANEL_ROW_SHARED = FigureProfile(
        "three_panel_row_shared", 183, 0.40, ncols=3, sharex=True, sharey=True,
        max_legends=1, max_colorbars=1, max_ticks_per_axis=6,
        require_shared_labels=True,
    )
    FOUR_PANEL_ROW = FigureProfile(
        "four_panel_row", 183, 0.34, ncols=4, sharex=True, sharey=True,
        max_legends=1, max_colorbars=1, max_ticks_per_axis=5,
        require_shared_labels=True,
    )
    TWO_BY_TWO = FigureProfile("two_by_two", 183, 0.70, nrows=2, ncols=2)
    TOP_SPAN_TWO_BOTTOM = FigureProfile(
        "top_span_two_bottom", 183, 0.78, nrows=2, ncols=2,
        max_legends=2, max_colorbars=1,
    )
    VERTICAL_TRIPTYCH = FigureProfile("vertical_triptych", 183, 0.75, nrows=3, ncols=1, sharex=True)
    VERTICAL_DUO = FigureProfile("vertical_duo", 183, 0.62, nrows=2, ncols=1, sharex=True)
    VERTICAL_QUAD = FigureProfile("vertical_quad", 183, 0.88, nrows=4, ncols=1, sharex=True)
    DIAGNOSTIC_GRID = FigureProfile("diagnostic_grid", 183, 0.88, nrows=3, ncols=3)

    @classmethod
    def by_name(cls, name: str) -> FigureProfile:
        for value in vars(cls).values():
            if isinstance(value, FigureProfile) and value.name == name:
                return value
        raise KeyError(f"Unknown publication figure profile: {name}")


class PublicationPalette:
    """Shared semantic palette for categorical and continuous figures."""

    MODEL_FAMILY = {
        "RandomForestModel": "#1F77B4",
        "RidgeRegressionModel": "#FF7F0E",
        "LightGBMModel": "#2CA02C",
        "ShallowMLPModel": "#9467BD",
        "TrajectoryCNN": "#8C564B",
        "ResNetVibCNN": "#D62728",
        "ResNetAECNN": "#D62728",
        "ResNetFusion": "#D62728",
        "BilinearFusionNetwork": "#D62728",
    }
    CONDITION_7 = "#D55E00"
    OBSERVED = "#1F77B4"
    PREDICTED = "#D62728"
    UNCERTAINTY = "#56B4E9"
    NEUTRAL = "#595959"
    TRAIN = "#D9EAF7"
    VALIDATION = "#E69F00"
    TEST = "#D55E00"
    CONTINUOUS = "viridis"
    SEQUENTIAL = "cividis"
    DIVERGING = "coolwarm"

    @classmethod
    def model(cls, model: str, fallback_index: int = 0) -> str:
        return cls.MODEL_FAMILY.get(
            model,
            PublicationPlotter.color_cycle()[fallback_index % len(PublicationPlotter.color_cycle())],
        )


class PublicationFigure(ABC):
    """Abstract ancestor for every final Python-generated figure."""

    profile: FigureProfile = FigureProfiles.SINGLE
    palette = PublicationPalette

    def __init__(
        self,
        name: str,
        *,
        profile: FigureProfile | None = None,
        out_dir: Path | None = None,
        overleaf_dir: Path | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self.name = name
        self.profile = profile or self.profile
        self.out_dir = out_dir
        self.overleaf_dir = overleaf_dir
        self.metadata = dict(metadata or {})
        self.fig: Figure | None = None
        self.axes: Any = None

    def create(self, **kwargs: Any) -> tuple[Figure, Any]:
        """Create a figure at its final physical print size."""
        PublicationPlotter.set_style()
        self._validate_profile()
        self.fig, self.axes = PublicationPlotter.create_figure(
            width_mm=self.profile.width_mm,
            ratio=self.profile.ratio,
            nrows=self.profile.nrows,
            ncols=self.profile.ncols,
            sharex=self.profile.sharex,
            sharey=self.profile.sharey,
            **kwargs,
        )
        return self.fig, self.axes

    def _validate_profile(self) -> None:
        if self.profile.width_mm not in {89, 183}:
            raise ValueError("Nature figures must use the 89 mm or 183 mm final-width profile.")
        if self.profile.ncols >= 3 and self.profile.width_mm != 183:
            raise ValueError("Three- and four-panel rows must use the 183 mm double-column profile.")
        if self.profile.ncols == 4 and not (self.profile.sharex and self.profile.sharey):
            raise ValueError("Four-panel rows require shared axes; use a 2x2 profile otherwise.")

    def apply_panel_row_rules(self, axes: Sequence[Any]) -> None:
        """Apply common typography and shared-label discipline for horizontal rows."""
        if not self.profile.require_shared_labels:
            return
        flat = list(axes)
        for index, ax in enumerate(flat):
            if index:
                ax.set_ylabel("")
            ax.tick_params(labelsize=6)
        self.fig.subplots_adjust(wspace=0.28, bottom=0.23, left=0.07, right=0.98)

    @abstractmethod
    def draw(self) -> None:
        """Draw the figure after :meth:`create` has established its layout."""

    def render(self) -> Figure:
        self.create()
        self.draw()
        assert self.fig is not None
        return self.fig

    def save(self, formats: Sequence[str] = ("pdf", "png")) -> dict[str, Path]:
        if self.fig is None:
            raise RuntimeError("Call render() before save().")
        self.apply_nature_panel_labels()
        saved = PublicationPlotter.savefig(
            self.fig,
            self.name,
            out_dir=self.out_dir,
            overleaf_dir=self.overleaf_dir,
            formats=formats,
            close=True,
        )
        PublicationPlotter.record_figure_metadata(
            self.name,
            profile=asdict(self.profile),
            metadata=self.metadata,
            out_dir=self.out_dir,
        )
        return saved

    def primary_axes(self) -> list[Any]:
        """Return visible data axes in deterministic row-major order."""
        if self.axes is None:
            return []
        try:
            axes = list(self.axes.flat)
        except AttributeError:
            axes = list(self.axes) if isinstance(self.axes, (list, tuple)) else [self.axes]
        return [ax for ax in axes if getattr(ax, "get_visible", lambda: True)()]

    def apply_nature_panel_labels(self) -> None:
        """Add only bold upright lowercase letters to multi-panel figures."""
        axes = self.primary_axes()
        if len(axes) <= 1:
            return
        for index, ax in enumerate(axes):
            ax.set_title("")
            PublicationPlotter.add_panel_label(ax, chr(ord("a") + index))
        self.metadata["panel_labels"] = "8 pt bold upright lowercase; row-major"
        self.metadata["panel_titles"] = "caption-only; axes titles cleared by ancestor"


class SingleAxesFigure(PublicationFigure):
    """Convenience ancestor for one-axis charts."""

    profile = FigureProfiles.SINGLE

    @property
    def ax(self) -> Any:
        if self.axes is None:
            raise RuntimeError("Figure has not been created.")
        return self.axes


class PanelGridFigure(PublicationFigure):
    """Ancestor for figures with a regular panel grid."""

    profile = FigureProfiles.TWO_BY_TWO

    def flat_axes(self) -> list[Any]:
        if self.axes is None:
            raise RuntimeError("Figure has not been created.")
        try:
            return list(self.axes.flat)
        except AttributeError:
            return [self.axes]


class MutableFigure(PublicationFigure):
    """A registered OOP figure for script-local drawing mutations.

    A generator instantiates this class with a profile, calls ``create()``,
    draws its domain-specific artists, and calls ``save()``. It provides a
    controlled migration path for complex existing figures while retaining the
    same validation and provenance guarantees as specialised subclasses.
    """

    def draw(self) -> None:
        return None

    @classmethod
    def adopt(
        cls,
        fig: Figure,
        name: str,
        *,
        profile: FigureProfile = FigureProfiles.DOUBLE,
        out_dir: Path | None = None,
        overleaf_dir: Path | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> "MutableFigure":
        """Register a figure created by a third-party plotting API (for example SHAP)."""
        instance = cls(name, profile=profile, out_dir=out_dir, overleaf_dir=overleaf_dir, metadata=metadata)
        instance.fig = fig
        instance.axes = fig.axes
        return instance


class DenseRankingFigure(SingleAxesFigure):
    """One tall full-width ranking figure with readable final-size labels."""

    profile = FigureProfiles.DENSE_RANKING


class ThreePanelRowFigure(PanelGridFigure):
    """Validated 1x3 profile for related compact panels."""

    profile = FigureProfiles.THREE_PANEL_ROW

    def render(self) -> Figure:
        fig = super().render()
        self.apply_panel_row_rules(self.flat_axes())
        return fig


class FourPanelRowFigure(PanelGridFigure):
    """Validated 1x4 profile for compact small multiples with shared axes."""

    profile = FigureProfiles.FOUR_PANEL_ROW

    def render(self) -> Figure:
        fig = super().render()
        self.apply_panel_row_rules(self.flat_axes())
        return fig
