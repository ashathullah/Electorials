"""Raw OCR (Tesseract) dump to Markdown.

This is meant for debugging/QA: it writes what Tesseract returned
BEFORE any downstream parsing/cleaning.

By default, we keep this lightweight: "text by line" only.
No OCR dependencies are imported here; pass in the `image_to_data` Output.DICT.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def _md_escape(value: Any) -> str:
    if value is None:
        return ""
    s = str(value)
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = s.replace("|", "\\|")
    s = s.replace("\n", "<br>")
    return s.strip()


def _safe_int(value: Any, default: int = -1) -> int:
    try:
        return int(float(value))
    except Exception:
        return default


def _group_lines(data: dict[str, Any]) -> list[tuple[tuple[int, int, int], list[dict[str, Any]]]]:
    # Groups by (block_num, par_num, line_num) and sorts within the line by x (left)
    texts = data.get("text") or []
    n = len(texts)

    lines: dict[tuple[int, int, int], list[dict[str, Any]]] = {}
    for i in range(n):
        txt = (texts[i] or "").strip()
        if not txt:
            continue

        item = {
            "text": txt,
            "conf": _safe_int((data.get("conf") or [])[i] if (data.get("conf") or None) else -1),
            "left": _safe_int((data.get("left") or [])[i] if (data.get("left") or None) else 0, 0),
            "top": _safe_int((data.get("top") or [])[i] if (data.get("top") or None) else 0, 0),
            "width": _safe_int((data.get("width") or [])[i] if (data.get("width") or None) else 0, 0),
            "height": _safe_int((data.get("height") or [])[i] if (data.get("height") or None) else 0, 0),
            "block_num": _safe_int((data.get("block_num") or [])[i] if (data.get("block_num") or None) else 0, 0),
            "par_num": _safe_int((data.get("par_num") or [])[i] if (data.get("par_num") or None) else 0, 0),
            "line_num": _safe_int((data.get("line_num") or [])[i] if (data.get("line_num") or None) else 0, 0),
            "word_num": _safe_int((data.get("word_num") or [])[i] if (data.get("word_num") or None) else 0, 0),
        }

        key = (item["block_num"], item["par_num"], item["line_num"])
        lines.setdefault(key, []).append(item)

    grouped: list[tuple[tuple[int, int, int], list[dict[str, Any]]]] = []
    for key, items in lines.items():
        items.sort(key=lambda t: (t["left"], t["word_num"]))
        grouped.append((key, items))

    grouped.sort(key=lambda t: t[0])
    return grouped


def render_raw_ocr_text_only_markdown(
    *,
    file: str,
    page: str,
    image: str,
    languages: str,
    tesseract_config: str,
    ocr_data: dict[str, Any],
) -> str:
    lines: list[str] = []
    lines.append(f"## { _md_escape(page) } / { _md_escape(image) }")
    lines.append("")
    lines.append(f"- **File**: {_md_escape(file)}")
    lines.append(f"- **Languages**: {_md_escape(languages)}")
    lines.append(f"- **Tesseract config**: `{tesseract_config}`")
    lines.append("")
    lines.append("### Text by line")
    lines.append("")
    for (b, p, l), items in _group_lines(ocr_data):
        joined = " ".join(i["text"] for i in items)
        lines.append(f"- **b{b}/p{p}/l{l}**: { _md_escape(joined) }")
    lines.append("")
    return "\n".join(lines)


def render_raw_ocr_detailed_markdown(
    *,
    file: str,
    page: str,
    image: str,
    languages: str,
    tesseract_config: str,
    ocr_data: dict[str, Any],
) -> str:
    texts = ocr_data.get("text") or []
    n = len(texts)

    confs = [
        _safe_int(c, -1)
        for c in (ocr_data.get("conf") or [])
        if str(c).strip() not in {"", "-1"}
    ]
    conf_min = min(confs) if confs else None
    conf_max = max(confs) if confs else None

    lines: list[str] = []
    lines.append("# Raw OCR Dump (Detailed)")
    lines.append("")
    lines.append(f"- **File**: {_md_escape(file)}")
    lines.append(f"- **Page**: {_md_escape(page)}")
    lines.append(f"- **Image**: {_md_escape(image)}")
    lines.append(f"- **Languages**: {_md_escape(languages)}")
    lines.append(f"- **Tesseract config**: `{tesseract_config}`")
    lines.append(f"- **Token rows**: {n}")
    if conf_min is not None and conf_max is not None:
        lines.append(f"- **Conf (min..max)**: {conf_min}..{conf_max}")
    lines.append("")

    lines.append("## Text by line (as Tesseract tokens)")
    lines.append("")
    for (b, p, l), items in _group_lines(ocr_data):
        joined = " ".join(i["text"] for i in items)
        lines.append(f"- **b{b}/p{p}/l{l}**: {_md_escape(joined)}")
    lines.append("")

    lines.append("## Tokens")
    lines.append("")
    lines.append("| i | text | conf | left | top | width | height | block | par | line | word |")
    lines.append("|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")

    for i in range(n):
        txt = (texts[i] or "").strip()
        if not txt:
            continue
        conf = (ocr_data.get("conf") or [])[i] if (ocr_data.get("conf") or None) else ""
        left = (ocr_data.get("left") or [])[i] if (ocr_data.get("left") or None) else ""
        top = (ocr_data.get("top") or [])[i] if (ocr_data.get("top") or None) else ""
        width = (ocr_data.get("width") or [])[i] if (ocr_data.get("width") or None) else ""
        height = (ocr_data.get("height") or [])[i] if (ocr_data.get("height") or None) else ""
        block = (ocr_data.get("block_num") or [])[i] if (ocr_data.get("block_num") or None) else ""
        par = (ocr_data.get("par_num") or [])[i] if (ocr_data.get("par_num") or None) else ""
        line = (ocr_data.get("line_num") or [])[i] if (ocr_data.get("line_num") or None) else ""
        word = (ocr_data.get("word_num") or [])[i] if (ocr_data.get("word_num") or None) else ""

        lines.append(
            "| "
            + " | ".join(
                [
                    str(i),
                    _md_escape(txt),
                    _md_escape(conf),
                    _md_escape(left),
                    _md_escape(top),
                    _md_escape(width),
                    _md_escape(height),
                    _md_escape(block),
                    _md_escape(par),
                    _md_escape(line),
                    _md_escape(word),
                ]
            )
            + " |"
        )

    lines.append("")
    return "\n".join(lines)


def init_raw_ocr_markdown(path: Path, *, file: str, languages: str, tesseract_config: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = "\n".join(
        [
            "# Raw OCR (Text Only)",
            "",
            f"- **File**: {_md_escape(file)}",
            f"- **Languages**: {_md_escape(languages)}",
            f"- **Tesseract config**: `{tesseract_config}`",
            "",
        ]
    )
    path.write_text(header, encoding="utf-8")


def append_raw_ocr_text_only(
    *,
    md_path: Path,
    file: str,
    page: str,
    image: str,
    languages: str,
    tesseract_config: str,
    ocr_data: dict[str, Any],
) -> None:
    md_path.parent.mkdir(parents=True, exist_ok=True)
    chunk = render_raw_ocr_text_only_markdown(
        file=file,
        page=page,
        image=image,
        languages=languages,
        tesseract_config=tesseract_config,
        ocr_data=ocr_data,
    )
    with md_path.open("a", encoding="utf-8") as f:
        f.write(chunk)


def write_raw_ocr_detailed_for_image(
    *,
    out_path: Path,
    file: str,
    page: str,
    image: str,
    languages: str,
    tesseract_config: str,
    ocr_data: dict[str, Any],
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    md = render_raw_ocr_detailed_markdown(
        file=file,
        page=page,
        image=image,
        languages=languages,
        tesseract_config=tesseract_config,
        ocr_data=ocr_data,
    )
    out_path.write_text(md, encoding="utf-8")
