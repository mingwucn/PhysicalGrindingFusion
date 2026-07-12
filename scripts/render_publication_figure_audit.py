#!/usr/bin/env python3
"""Render labelled contact sheets for Python figures included by the manuscript."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
MAIN_DIR = ROOT / "overleaf" / "main"
DEFAULT_OUTPUT = Path("/tmp/vibegrinding-python-figure-audit")
INCLUDE_RE = re.compile(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}")


def included_pngs() -> list[Path]:
    """Resolve PNG graphics referenced by manuscript TeX sources."""
    references: list[str] = []
    for tex_path in sorted(MAIN_DIR.glob("*.tex")):
        references.extend(INCLUDE_RE.findall(tex_path.read_text(encoding="utf-8")))

    resolved: list[Path] = []
    for reference in dict.fromkeys(references):
        if not reference.lower().endswith(".png"):
            continue
        candidates = [
            ROOT / reference,
            ROOT / "overleaf" / reference,
            ROOT / "overleaf" / "images" / Path(reference).name,
        ]
        path = next((candidate for candidate in candidates if candidate.is_file()), None)
        if path is None:
            raise FileNotFoundError(f"Cannot resolve manuscript figure: {reference}")
        resolved.append(path)
    return resolved


def render_contact_sheets(figures: list[Path], output_dir: Path, columns: int = 3) -> list[Path]:
    """Create scaled contact sheets while preserving the source aspect ratios."""
    output_dir.mkdir(parents=True, exist_ok=True)
    tile_width, tile_height, header_height = 560, 410, 34
    rows_per_page = 3
    page_count = (len(figures) + columns * rows_per_page - 1) // (columns * rows_per_page)
    font = ImageFont.load_default()
    outputs: list[Path] = []

    for page_index in range(page_count):
        page_figures = figures[page_index * columns * rows_per_page : (page_index + 1) * columns * rows_per_page]
        rows = (len(page_figures) + columns - 1) // columns
        page = Image.new("RGB", (columns * tile_width, rows * (tile_height + header_height)), "white")
        draw = ImageDraw.Draw(page)
        for index, figure_path in enumerate(page_figures):
            row, column = divmod(index, columns)
            x0, y0 = column * tile_width, row * (tile_height + header_height)
            with Image.open(figure_path) as source:
                image = source.convert("RGB")
            image.thumbnail((tile_width - 16, tile_height - 16), Image.Resampling.LANCZOS)
            x = x0 + (tile_width - image.width) // 2
            y = y0 + header_height + (tile_height - image.height) // 2
            page.paste(image, (x, y))
            draw.text((x0 + 8, y0 + 9), figure_path.name, fill="black", font=font)

        output_path = output_dir / f"python_figure_audit_{page_index + 1:02d}.png"
        page.save(output_path)
        outputs.append(output_path)
    return outputs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    figures = included_pngs()
    outputs = render_contact_sheets(figures, args.output_dir)
    print(f"Rendered {len(figures)} manuscript PNG figures into {len(outputs)} contact sheets:")
    print("\n".join(str(output) for output in outputs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
