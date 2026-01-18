#!/usr/bin/env python3
"""AI-assisted repair for metadata-regeneration/extracted_numbers/*.json.

This script is intentionally separate from extract_numbers.py.

What it does:
- Loads one or more JSON files produced by extract_numbers.py.
- Finds records with missing/invalid pincode or voters_end.
- Uses an AI vision model (OpenAI-compatible endpoint; Groq works) to extract digits
  from a compact stitched ROI image (pincode ROI + voters_end ROI).
- Updates the JSON in-place (with timestamped .bak backup) or writes to an output dir.

It also supports a no-AI fallback mode that fills missing pincodes using the
most-common pincode within each JSON file.

Requirements:
- AI mode: AI_API_KEY must be set (see src/config.py), and openai>=1.12 installed.

Example:
  python ai_fix_extracted_numbers.py \
    --images extracted_images \
    --json extracted_numbers \
    --only missing_pincode,invalid_voters_end \
    --limit 50

"""

from __future__ import annotations

import argparse
import base64
import json
import re
import shutil
import sys
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
from openai import OpenAI

from rois import PINCODE_ROI, VOTERS_END_ROI

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Pulls env + .env via module import
from src.config import Config


PIN_RE = re.compile(r"^6\d{5}$")
VOTERS_RE = re.compile(r"^\d{2,4}$")


def _denormalize_roi(roi: Tuple[float, float, float, float], h: int, w: int) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = roi
    return int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)


def _remove_table_lines_inpaint(bgr_img: np.ndarray) -> np.ndarray:
    """Remove table/grid lines via masking + inpainting; tends to preserve digits better."""
    gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)

    bw = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        15,
        10,
    )

    horiz_len = max(25, min(120, bgr_img.shape[1] // 2))
    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (horiz_len, 1))
    horizontal = cv2.erode(bw, horiz_kernel, iterations=1)
    horizontal = cv2.dilate(horizontal, horiz_kernel, iterations=1)

    vert_len = max(25, min(120, bgr_img.shape[0] // 2))
    vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vert_len))
    vertical = cv2.erode(bw, vert_kernel, iterations=1)
    vertical = cv2.dilate(vertical, vert_kernel, iterations=1)

    line_mask = cv2.bitwise_or(horizontal, vertical)
    line_mask = cv2.dilate(line_mask, np.ones((3, 3), np.uint8), iterations=1)

    cleaned = cv2.inpaint(bgr_img, line_mask, 3, cv2.INPAINT_TELEA)
    return cleaned


def _encode_bgr_as_data_url_png(bgr: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", bgr)
    if not ok:
        raise RuntimeError("Failed to PNG-encode ROI snippet")
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def _make_stitched_snippet(full_bgr: np.ndarray) -> np.ndarray:
    """Create a compact image that contains both ROIs side-by-side."""
    h, w = full_bgr.shape[:2]
    pc = _denormalize_roi(PINCODE_ROI, h, w)
    ve = _denormalize_roi(VOTERS_END_ROI, h, w)

    x1, y1, x2, y2 = pc
    pincode_roi = full_bgr[y1:y2, x1:x2]

    x1, y1, x2, y2 = ve
    voters_roi = full_bgr[y1:y2, x1:x2]
    voters_roi = _remove_table_lines_inpaint(voters_roi)

    target_h = max(pincode_roi.shape[0], voters_roi.shape[0])

    def pad_to_h(img: np.ndarray, new_h: int) -> np.ndarray:
        if img.shape[0] == new_h:
            return img
        pad = new_h - img.shape[0]
        return cv2.copyMakeBorder(img, 0, pad, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))

    pincode_roi = pad_to_h(pincode_roi, target_h)
    voters_roi = pad_to_h(voters_roi, target_h)

    sep_w = 28
    separator = np.ones((target_h, sep_w, 3), dtype=np.uint8) * 255
    stitched = cv2.hconcat([pincode_roi, separator, voters_roi])

    # Upscale for AI readability
    stitched = cv2.resize(stitched, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    return stitched


def _extract_json_from_text(text: str) -> Dict[str, Any]:
    """Best-effort parse for models that ignore response_format."""
    text = (text or "").strip()
    if not text:
        return {}

    # Try strict JSON first
    try:
        return json.loads(text)
    except Exception:
        pass

    # Try to find a JSON object substring
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass

    # Very last resort: regex digits
    out: Dict[str, Any] = {}
    m_pc = re.search(r"(6\d{5})", text)
    if m_pc:
        out["pincode"] = m_pc.group(1)

    # voters_end: 2-4 digits; prefer the last occurrence
    m_ve = re.findall(r"\b(\d{2,4})\b", text)
    if m_ve:
        out["voters_end"] = m_ve[-1]

    return out


def _needs_fix(result: Dict[str, Any], only: set[str]) -> bool:
    pc = result.get("pincode")
    ve = result.get("voters_end")

    missing_pc = not (isinstance(pc, str) and PIN_RE.fullmatch(pc))
    invalid_ve = not (isinstance(ve, str) and VOTERS_RE.fullmatch(ve))

    if "missing_pincode" in only and missing_pc:
        return True
    if "invalid_voters_end" in only and invalid_ve:
        return True
    if "missing_voters_end" in only and (ve is None or ve == ""):
        return True
    return False


def _recompute_status(pincode: Optional[str], voters_end: Optional[str]) -> str:
    if not pincode:
        return "missing_pincode"
    if not voters_end:
        return "missing_voters_end"
    return "success"


@dataclass
class AIFixStats:
    scanned: int = 0
    attempted: int = 0
    updated: int = 0
    ai_errors: int = 0
    skipped_no_image: int = 0


def _call_ai_extract_digits(client: OpenAI, model: str, img_data_url: str, want_json_object: bool) -> Dict[str, Any]:
    system = (
        "You extract numeric fields from a cropped election roll image snippet. "
        "Return STRICT JSON with keys: pincode, voters_end. "
        "Rules: pincode must be exactly 6 digits and start with '6'. "
        "voters_end must be 2 to 4 digits. If you are not confident, use null."
    )

    payload: Dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Extract pincode and voters_end."},
                    {"type": "image_url", "image_url": {"url": img_data_url}},
                ],
            },
        ],
    }

    if want_json_object:
        payload["response_format"] = {"type": "json_object"}

    resp = client.chat.completions.create(**payload)
    content = str(resp.choices[0].message.content or "")
    return _extract_json_from_text(content)


def _best_effort_fill_pincode(data: Dict[str, Any]) -> int:
    """Fill missing pincodes using mode within same JSON file."""
    results: List[Dict[str, Any]] = data.get("results") or []
    pincodes = [r.get("pincode") for r in results if isinstance(r.get("pincode"), str) and PIN_RE.fullmatch(r["pincode"])]
    if not pincodes:
        return 0

    mode, _ = Counter(pincodes).most_common(1)[0]
    updated = 0
    for r in results:
        pc = r.get("pincode")
        if not (isinstance(pc, str) and PIN_RE.fullmatch(pc)):
            r["pincode"] = mode
            r["status"] = _recompute_status(r.get("pincode"), r.get("voters_end"))
            updated += 1
    return updated


def _backup_file(path: Path) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = path.with_suffix(path.suffix + f".bak-{ts}")
    shutil.copy2(path, backup)
    return backup


def main() -> int:
    parser = argparse.ArgumentParser(description="AI-assisted fix for extracted_numbers JSON")
    parser.add_argument("--images", default="extracted_images", help="Input images root (default: extracted_images)")
    parser.add_argument("--json", dest="json_dir", default="extracted_numbers", help="Input JSON dir (default: extracted_numbers)")
    parser.add_argument("--out", default=None, help="Optional output dir (if omitted, edits JSON files in-place)")
    parser.add_argument(
        "--only",
        default="missing_pincode,invalid_voters_end",
        help="Comma-list: missing_pincode,missing_voters_end,invalid_voters_end",
    )
    parser.add_argument("--mode", default="auto", choices=["auto", "ai", "heuristic"], help="Fix mode")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of records to AI-process (0 = no limit)")
    parser.add_argument("--dry-run", action="store_true", help="Do not write any files")
    parser.add_argument("--debug-snips", default=None, help="Optional directory to save stitched snippets sent to AI")

    args = parser.parse_args()

    script_dir = Path(__file__).parent
    images_root = Path(args.images)
    if not images_root.is_absolute():
        images_root = script_dir / images_root

    json_dir = Path(args.json_dir)
    if not json_dir.is_absolute():
        json_dir = script_dir / json_dir

    out_dir = None
    if args.out:
        out_dir = Path(args.out)
        if not out_dir.is_absolute():
            out_dir = script_dir / out_dir
        out_dir.mkdir(parents=True, exist_ok=True)

    only = {s.strip() for s in (args.only or "").split(",") if s.strip()}

    json_files = sorted(json_dir.glob("*.json"))
    if not json_files:
        print(f"❌ No JSON files found in: {json_dir}")
        return 1

    cfg = Config()
    ai_key = (cfg.ai.api_key or "").strip()
    use_ai = args.mode in ("ai", "auto") and bool(ai_key)
    use_heuristic = args.mode in ("heuristic", "auto") and not use_ai

    client: Optional[OpenAI] = None
    want_json_object = (cfg.ai.response_format == "json_object")
    if use_ai:
        base_url = cfg.ai.get_normalized_base_url() or None
        client = OpenAI(api_key=cfg.ai.api_key, base_url=base_url, timeout=cfg.ai.timeout_sec)
        print(f"🤖 AI mode enabled | provider={cfg.ai.provider} model={cfg.ai.model}")
    else:
        if args.mode == "ai":
            print("❌ --mode ai requested but AI_API_KEY is missing")
            return 2
        print("🧩 Heuristic mode (no AI_API_KEY detected): fill pincodes by majority vote")

    debug_snips_dir: Optional[Path] = None
    if args.debug_snips:
        debug_snips_dir = Path(args.debug_snips)
        if not debug_snips_dir.is_absolute():
            debug_snips_dir = script_dir / debug_snips_dir
        debug_snips_dir.mkdir(parents=True, exist_ok=True)

    total_stats = AIFixStats()
    ai_processed = 0

    for jf in json_files:
        data = json.loads(jf.read_text(encoding="utf-8"))
        directory = (data.get("metadata") or {}).get("directory")
        if not directory:
            print(f"⚠️  Skipping {jf.name}: missing metadata.directory")
            continue

        results: List[Dict[str, Any]] = data.get("results") or []
        to_fix = [r for r in results if _needs_fix(r, only)]

        if use_heuristic:
            updated = _best_effort_fill_pincode(data)
            if updated:
                (data.setdefault("metadata", {})).setdefault("ai_fix", {})
                data["metadata"]["ai_fix"].update(
                    {
                        "mode": "heuristic",
                        "timestamp": datetime.now().isoformat(),
                        "filled_pincode_count": updated,
                    }
                )
                if not args.dry_run:
                    out_path = (out_dir / jf.name) if out_dir else jf
                    if out_path == jf:
                        _backup_file(jf)
                    out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
                print(f"✅ {jf.name}: filled {updated} pincodes (heuristic)")
            continue

        # AI mode
        if not to_fix:
            continue

        print(f"🔎 {jf.name}: {len(to_fix)} records need fix")

        for r in to_fix:
            total_stats.scanned += 1
            if args.limit and ai_processed >= args.limit:
                break

            doc_id = r.get("document_id")
            if not doc_id:
                continue

            img_path = images_root / directory / f"{doc_id}.png"
            if not img_path.exists():
                total_stats.skipped_no_image += 1
                continue

            total_stats.attempted += 1
            start = time.time()
            try:
                full = cv2.imread(str(img_path))
                if full is None:
                    raise RuntimeError("cv2.imread failed")

                snippet = _make_stitched_snippet(full)

                if debug_snips_dir:
                    safe = re.sub(r"[^A-Za-z0-9_.()-]+", "_", doc_id)
                    cv2.imwrite(str(debug_snips_dir / f"{directory}__{safe}.png"), snippet)

                img_url = _encode_bgr_as_data_url_png(snippet)

                assert client is not None
                parsed = _call_ai_extract_digits(client, cfg.ai.model, img_url, want_json_object)

                pc = parsed.get("pincode")
                ve = parsed.get("voters_end")

                # Normalize
                if isinstance(pc, int):
                    pc = str(pc)
                if isinstance(ve, int):
                    ve = str(ve)

                if isinstance(pc, str):
                    pc = re.sub(r"\D", "", pc)
                if isinstance(ve, str):
                    ve = re.sub(r"\D", "", ve)

                if not (isinstance(pc, str) and PIN_RE.fullmatch(pc)):
                    pc = None
                if not (isinstance(ve, str) and VOTERS_RE.fullmatch(ve)):
                    ve = None

                changed = False
                if pc and r.get("pincode") != pc:
                    r["pincode"] = pc
                    changed = True
                if ve and r.get("voters_end") != ve:
                    r["voters_end"] = ve
                    changed = True

                new_status = _recompute_status(r.get("pincode"), r.get("voters_end"))
                if r.get("status") != new_status:
                    r["status"] = new_status
                    changed = True

                if changed:
                    total_stats.updated += 1

                ai_processed += 1
                elapsed_ms = int((time.time() - start) * 1000)
                print(
                    f"  {'✅' if changed else '➖'} {doc_id} | PC:{r.get('pincode') or 'None'} VE:{r.get('voters_end') or 'None'} | {elapsed_ms}ms"
                )

            except Exception as e:
                total_stats.ai_errors += 1
                print(f"  ❌ {doc_id}: {e}")

        # Write back this JSON file
        (data.setdefault("metadata", {})).setdefault("ai_fix", {})
        data["metadata"]["ai_fix"].update(
            {
                "mode": "ai",
                "timestamp": datetime.now().isoformat(),
                "model": cfg.ai.model,
                "provider": cfg.ai.provider,
                "attempted": total_stats.attempted,
                "updated": total_stats.updated,
                "errors": total_stats.ai_errors,
            }
        )

        if not args.dry_run:
            out_path = (out_dir / jf.name) if out_dir else jf
            if out_path == jf:
                _backup_file(jf)
            out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

        if args.limit and ai_processed >= args.limit:
            break

    print(
        f"\nDone. scanned={total_stats.scanned} attempted={total_stats.attempted} "
        f"updated={total_stats.updated} errors={total_stats.ai_errors} no_image={total_stats.skipped_no_image}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
