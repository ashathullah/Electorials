"""
Extract "source" assets from PDFs (usually page images) in one step.

Defaults:
  - Input PDFs:   pdf-converter/pdfs/*.pdf
  - Output:       pdf-converter/extracted/<pdf-stem>/

Examples:
    python pdf-converter/extract_source.py
    python pdf-converter/extract_source.py --extract render --dpi 200
    python pdf-converter/extract_source.py --extract both --dpi 200
    python pdf-converter/extract_source.py --write-text
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import fitz  # PyMuPDF


@dataclass(frozen=True)
class ExtractedImage:
    page: int
    index: int
    path: str
    ext: str
    width: int
    height: int


def _iter_pdfs(input_dir: Path) -> Iterable[Path]:
    for path in sorted(input_dir.glob("*.pdf")):
        if path.is_file():
            yield path


def _safe_stem(path: Path) -> str:
    # Keep filenames stable and filesystem-friendly.
    return "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in path.stem)


def _extract_embedded_images(doc: fitz.Document, out_dir: Path) -> list[ExtractedImage]:
    images: list[ExtractedImage] = []
    out_dir.mkdir(parents=True, exist_ok=True)

    for page_index in range(doc.page_count):
        page = doc.load_page(page_index)
        page_images = page.get_images(full=True)
        for image_index, image_tuple in enumerate(page_images, start=1):
            xref = int(image_tuple[0])
            info = doc.extract_image(xref)
            raw = info.get("image")
            if not raw:
                continue

            ext = str(info.get("ext") or "bin").lower()
            filename = f"page-{page_index + 1:03d}-img-{image_index:02d}.{ext}"
            output_path = out_dir / filename
            output_path.write_bytes(raw)

            images.append(
                ExtractedImage(
                    page=page_index + 1,
                    index=image_index,
                    path=str(output_path.as_posix()),
                    ext=ext,
                    width=int(info.get("width") or 0),
                    height=int(info.get("height") or 0),
                )
            )

    return images


def _render_pages(doc: fitz.Document, out_dir: Path, dpi: int) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    rendered: list[str] = []

    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)
    for page_index in range(doc.page_count):
        page = doc.load_page(page_index)
        pix = page.get_pixmap(matrix=matrix, alpha=False)
        output_path = out_dir / f"page-{page_index + 1:03d}.png"
        pix.save(output_path)
        rendered.append(str(output_path.as_posix()))

    return rendered


def _write_text(doc: fitz.Document, out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    combined_lines: list[str] = []
    per_page: list[dict[str, Any]] = []

    for page_index in range(doc.page_count):
        page = doc.load_page(page_index)
        text = page.get_text("text") or ""
        per_page_path = out_dir / f"page-{page_index + 1:03d}.txt"
        per_page_path.write_text(text, encoding="utf-8")
        combined_lines.append(text)
        per_page.append(
            {
                "page": page_index + 1,
                "path": str(per_page_path.as_posix()),
                "chars": len(text),
            }
        )

    combined_path = out_dir / "text.txt"
    combined_path.write_text("\n".join(combined_lines), encoding="utf-8")
    return {"combined": str(combined_path.as_posix()), "pages": per_page}


def main() -> int:
    script_dir = Path(__file__).resolve().parent
    default_input = script_dir / "pdfs"
    default_output = script_dir / "extracted"

    parser = argparse.ArgumentParser(description="Extract source assets from PDFs.")
    parser.add_argument("--input", type=Path, default=default_input, help="Directory containing PDFs.")
    parser.add_argument("--output", type=Path, default=default_output, help="Output directory.")
    parser.add_argument(
        "--extract",
        choices=("images", "render", "both"),
        default="render",
        help="What to extract from PDFs.",
    )
    parser.add_argument("--dpi", type=int, default=200, help="DPI for --extract render/both.")
    parser.add_argument("--write-text", action="store_true", help="Also write extracted text files.")
    parser.add_argument("--limit", type=int, default=0, help="Process only first N PDFs (0 = all).")
    args = parser.parse_args()

    input_dir: Path = args.input
    output_dir: Path = args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    pdfs = list(_iter_pdfs(input_dir))
    if args.limit and args.limit > 0:
        pdfs = pdfs[: args.limit]

    if not pdfs:
        print(f"No PDFs found in: {input_dir}")
        return 2

    for pdf_path in pdfs:
        pdf_name = _safe_stem(pdf_path)
        pdf_out = output_dir / pdf_name
        pdf_out.mkdir(parents=True, exist_ok=True)

        doc = fitz.open(pdf_path)
        try:
            manifest: dict[str, Any] = {
                "input_pdf": str(pdf_path.as_posix()),
                "pages": doc.page_count,
                "metadata": {k: v for k, v in (doc.metadata or {}).items() if v},
            }

            if args.extract in ("images", "both"):
                # Embedded-image extraction can yield multiple images per page.
                # Keep it separate so rendered per-page PNGs can live under images/.
                images = _extract_embedded_images(doc, pdf_out / "embedded_images")
                manifest["images"] = [asdict(img) for img in images]

            if args.extract in ("render", "both"):
                # One PNG per page, always named page-XXX.png
                manifest["rendered_pages"] = _render_pages(doc, pdf_out / "images", dpi=args.dpi)

            if args.write_text:
                manifest["text"] = _write_text(doc, pdf_out / "text")

            (pdf_out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            print(f"OK: {pdf_path.name} -> {pdf_out}")
        finally:
            doc.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

