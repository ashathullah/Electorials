"""Markdown preview helpers for OCR output.

Purpose:
- Write a human-readable Markdown snapshot of what OCR recognized
- Store it next to JSON outputs (same basename)

This is intentionally dependency-free (no OCR libs imported here).
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


def _yn(value: Any) -> str:
    return "Y" if bool(value) else "N"


def render_page_markdown(payload: dict[str, Any]) -> str:
    records = payload.get("records") or []
    total = len(records)

    epic_valid_count = 0
    missing_name = 0
    missing_epic = 0
    missing_house = 0
    missing_age = 0
    missing_gender = 0

    for r in records:
        if r.get("epic_valid"):
            epic_valid_count += 1
        if not (r.get("name") or "").strip():
            missing_name += 1
        if not (r.get("epic_no") or "").strip():
            missing_epic += 1
        if not (r.get("house_no") or "").strip():
            missing_house += 1
        if not (r.get("age") or "").strip():
            missing_age += 1
        if not (r.get("gender") or "").strip():
            missing_gender += 1

    lines: list[str] = []
    lines.append(f"# OCR Preview")
    lines.append("")
    lines.append(f"- **File**: {_md_escape(payload.get('file', ''))}")
    lines.append(f"- **Page**: {_md_escape(payload.get('page', ''))}")
    lines.append(f"- **Images**: {_md_escape(payload.get('images_count', total))}")
    lines.append(f"- **Languages**: {_md_escape(payload.get('languages', ''))}")
    lines.append(f"- **Allow next line**: {_yn(payload.get('allow_next_line'))}")
    lines.append(f"- **Records**: {total}")
    lines.append("")

    lines.append("## Quick stats")
    lines.append("")
    lines.append(f"- EPIC valid: {epic_valid_count}/{total}")
    lines.append(f"- Missing name: {missing_name}/{total}")
    lines.append(f"- Missing EPIC: {missing_epic}/{total}")
    lines.append(f"- Missing house no: {missing_house}/{total}")
    lines.append(f"- Missing age: {missing_age}/{total}")
    lines.append(f"- Missing gender: {missing_gender}/{total}")
    lines.append("")

    lines.append("## Records")
    lines.append("")
    # Markdown table (wide but quick to scan)
    lines.append(
        "| # | image | serial | epic | epic_valid | name | relation_type | relation_name | house_no | age | gender |"
    )
    lines.append(
        "|---:|---|---|---|:---:|---|---|---|---|---|---|"
    )

    for i, r in enumerate(records, start=1):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(i),
                    _md_escape(r.get("image", "")),
                    _md_escape(r.get("serial_no", "")),
                    _md_escape(r.get("epic_no", "")),
                    _yn(r.get("epic_valid")),
                    _md_escape(r.get("name", "")),
                    _md_escape(r.get("relation_type", "")),
                    _md_escape(r.get("relation_name", "")),
                    _md_escape(r.get("house_no", "")),
                    _md_escape(r.get("age", "")),
                    _md_escape(r.get("gender", "")),
                ]
            )
            + " |"
        )

    lines.append("")
    return "\n".join(lines)


def render_file_markdown(file_payload: dict[str, Any]) -> str:
    pages = file_payload.get("pages") or []

    lines: list[str] = []
    lines.append("# OCR Preview (File)")
    lines.append("")
    lines.append(f"- **Folder**: {_md_escape(file_payload.get('folder', ''))}")
    lines.append(f"- **Pages**: {_md_escape(file_payload.get('pages_count', len(pages)))}")
    lines.append(f"- **Images**: {_md_escape(file_payload.get('images_count', ''))}")
    lines.append(f"- **Languages**: {_md_escape(file_payload.get('languages', ''))}")
    lines.append(f"- **Allow next line**: {_yn(file_payload.get('allow_next_line'))}")
    lines.append("")

    lines.append("## Pages")
    lines.append("")
    lines.append("| page | images_processed | avg_seconds_per_image | records | page_json |")
    lines.append("|---|---:|---:|---:|---|")

    for p in pages:
        records = p.get("records") or []
        lines.append(
            "| "
            + " | ".join(
                [
                    _md_escape(p.get("page", "")),
                    _md_escape(p.get("images_processed", "")),
                    _md_escape(p.get("avg_seconds_per_image", "")),
                    str(len(records)),
                    _md_escape(p.get("page_output_path", "")),
                ]
            )
            + " |"
        )

    lines.append("")
    return "\n".join(lines)


def write_markdown(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def write_page_markdown(page_md_path: Path, payload: dict[str, Any]) -> None:
    write_markdown(page_md_path, render_page_markdown(payload))


def write_file_markdown(file_md_path: Path, file_payload: dict[str, Any]) -> None:
    write_markdown(file_md_path, render_file_markdown(file_payload))
