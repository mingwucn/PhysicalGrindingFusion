#!/usr/bin/env python3
"""Render isolated native TikZ figures to local PNG previews.

Only the selected TikZ source and shared style layer are sent to the remote
LaTeX renderer. The manuscript, data, and cached model artefacts are not sent.
"""

from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path

import fitz


ROOT = Path(__file__).resolve().parents[1]
TIKZ_DIR = ROOT / "overleaf" / "tikz"
STYLE_FILE = TIKZ_DIR / "publication_styles.tex"
RENDER_URL = "https://latexonline.cc/compile"
DEFAULT_FIGURES = (
    "preprocessing_pipeline",
    "wst_pipeline",
    "statistical_workflow",
    "model_taxonomy",
    "logo_scheme",
    "resnetvibcnn_architecture",
)
PREAMBLE = r"""\documentclass[tikz,border=4pt]{standalone}
\usepackage{graphicx}
\usepackage{textcomp}
\usepackage{upgreek}
\usetikzlibrary{arrows.meta,positioning,shapes.multipart,fit,backgrounds,calc}
"""


def tex_document(figure: str) -> str:
    source = TIKZ_DIR / f"{figure}.tex"
    if not source.exists():
        raise FileNotFoundError(f"Unknown TikZ figure: {source}")
    return "\n".join(
        (PREAMBLE, STYLE_FILE.read_text(), r"\begin{document}", source.read_text(), r"\end{document}")
    )


def render(figure: str, output_dir: Path, scale: float) -> tuple[Path, Path]:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".tex", encoding="utf-8") as tex_file:
        tex_file.write(tex_document(figure))
        tex_file.flush()
        result = subprocess.run(
            [
                "curl",
                "--fail",
                "--silent",
                "--show-error",
                "--location",
                "--max-time",
                "120",
                "--get",
                "--data-urlencode",
                f"text@{tex_file.name}",
                RENDER_URL,
            ],
            capture_output=True,
        )
    if result.returncode:
        message = result.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(
            f"Remote TikZ rendering failed for {figure}. "
            "The public renderer accepts isolated small diagrams only; "
            "large schematics may exceed its request-size limit."
            + (f" curl: {message}" if message else "")
        )
    pdf_bytes = result.stdout

    pdf_path = output_dir / f"{figure}.pdf"
    png_path = output_dir / f"{figure}.png"
    pdf_path.write_bytes(pdf_bytes)
    with fitz.open(stream=pdf_bytes, filetype="pdf") as document:
        pixmap = document[0].get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
        pixmap.save(png_path)
    return pdf_path, png_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("figures", nargs="*", default=DEFAULT_FIGURES, help="TikZ basenames without .tex")
    parser.add_argument("--output", type=Path, default=Path("/tmp/vibegrinding-tikz-previews"))
    parser.add_argument("--scale", type=float, default=2.0, help="Raster scale for PNG inspection")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    for figure in args.figures:
        pdf_path, png_path = render(figure, args.output, args.scale)
        print(f"{figure}: {pdf_path} {png_path}")


if __name__ == "__main__":
    main()
