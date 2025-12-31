"""send_front_back_to_ai.py

Batch-extract metadata from Indian Electoral Roll PDFs by sending the FIRST and LAST
(relevant) page images of each extracted folder to a multimodal AI model.

Input folders:
  extracted/<file_name>/images/*.(png|jpg|jpeg|tif|tiff|bmp|webp)

Output:
  extracted/<file_name>/output/<file_name>.json

Back-page selection:
  Sometimes the very last image is blank or lacks the "SUMMARY OF ELECTORS" table.
  This script uses OpenCV heuristics (ink density + table line presence) to pick the
  best back page by scanning backwards from the end.

AI call:
  Uses an OpenAI-compatible Chat Completions endpoint via HTTP.

Configuration:
    - Put secrets/config in a local .env (recommended) or export env vars.
    - For Gemini (OpenAI-compat): set AI_PROVIDER=gemini and AI_API_KEY.

Optional environment variables:
    - AI_PROVIDER       default: (empty) => OpenAI-style defaults
    - AI_BASE_URL       provider default (Gemini/OpenAI)
    - AI_MODEL          provider default (Gemini/OpenAI)
  - AI_TIMEOUT_SEC    default: 120
    - AI_RESPONSE_FORMAT  set to json_object to request JSON mode (if supported)

Install deps:
  pip install opencv-python numpy requests

Usage:
  python send_front_back_to_ai.py
  python send_front_back_to_ai.py --limit 5
  python send_front_back_to_ai.py --force
  python send_front_back_to_ai.py --prompt prompt.md

Gemini quickstart:
    1) Copy .env.example -> .env
    2) Set AI_PROVIDER=gemini, AI_API_KEY=..., AI_MODEL=gemini-2.5-flash
    3) Run: python send_front_back_to_ai.py

Notes:
  - The prompt should instruct the model to output JSON only.
  - On parse failure, the raw response is stored alongside as .raw.txt
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
from openai import OpenAI


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}


def _load_dotenv(dotenv_path: Path) -> None:
    """Minimal .env loader (no external dependency).

    Supports KEY=VALUE, ignores blank lines and comments (#).
    Does not override existing environment variables.
    """
    if not dotenv_path.exists() or not dotenv_path.is_file():
        return

    for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if not key:
            continue
        if os.getenv(key) is None:
            os.environ[key] = value


@dataclass(frozen=True)
class ImageScore:
    path: Path
    ink_ratio: float
    table_line_ratio: float


def _iter_extracted_folders(extracted_dir: Path) -> list[Path]:
    if not extracted_dir.exists():
        return []
    folders: list[Path] = []
    for p in sorted(extracted_dir.iterdir()):
        if not p.is_dir():
            continue
        images_dir = p / "images"
        if images_dir.exists() and images_dir.is_dir():
            folders.append(p)
    return folders


def _sorted_images(images_dir: Path) -> list[Path]:
    imgs = [
        p
        for p in images_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    ]
    return sorted(imgs, key=lambda p: p.name.lower())


def _read_gray_for_scoring(image_path: Path) -> np.ndarray:
    img = cv2.imdecode(np.fromfile(str(image_path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Failed to read image: {image_path}")

    # Downscale aggressively for speed while preserving line structure.
    h, w = img.shape[:2]
    target_w = 900
    if w > target_w:
        scale = target_w / float(w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    return img


def _binarize(gray: np.ndarray) -> np.ndarray:
    # Otsu gives stable binarization for scanned pages.
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return bw


def _score_back_page_candidate(image_path: Path) -> ImageScore:
    gray = _read_gray_for_scoring(image_path)
    bw = _binarize(gray)

    # "Ink" = non-white pixels after binarization.
    ink = (bw < 250).astype(np.uint8)
    ink_ratio = float(ink.mean())

    # Detect table-like line structures (horizontal + vertical) using morphology.
    inv = (255 - bw)

    h, w = inv.shape[:2]
    # Kernel sizes scale with image; keep minimum to avoid 0.
    hk = max(20, w // 30)
    vk = max(20, h // 30)

    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (hk, 1))
    vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vk))

    horiz = cv2.morphologyEx(inv, cv2.MORPH_OPEN, horiz_kernel, iterations=1)
    vert = cv2.morphologyEx(inv, cv2.MORPH_OPEN, vert_kernel, iterations=1)
    lines = cv2.bitwise_or(horiz, vert)

    table_line_ratio = float((lines > 0).mean())

    return ImageScore(path=image_path, ink_ratio=ink_ratio, table_line_ratio=table_line_ratio)


def _pick_back_page(images: list[Path], *, max_lookback: int = 6) -> Path:
    if not images:
        raise ValueError("No images provided")
    if len(images) == 1:
        return images[0]

    # Heuristics tuned for scanned summary tables:
    # - Blank/near-blank pages have very low ink.
    # - Summary-of-electors pages typically have strong table lines.
    ink_min = 0.006  # ~0.6% pixels are ink
    table_min = 0.002  # ~0.2% pixels are line structures

    candidates = list(reversed(images[-max_lookback:]))
    scored: list[ImageScore] = []

    for p in candidates:
        try:
            s = _score_back_page_candidate(p)
        except Exception:
            continue
        scored.append(s)

        # Prefer first strong match scanning backwards.
        if s.ink_ratio >= ink_min and s.table_line_ratio >= table_min:
            return p

    # Fallback: choose the best looking candidate by table_line_ratio, then ink.
    if scored:
        scored_sorted = sorted(scored, key=lambda s: (s.table_line_ratio, s.ink_ratio), reverse=True)
        best = scored_sorted[0]
        # If even the best is basically blank, fall back to second-to-last image.
        if best.ink_ratio < ink_min and len(images) >= 2:
            return images[-2]
        return best.path

    # If scoring fails, use second-to-last (safer than last).
    return images[-2]


def _encode_image_data_url(image_path: Path) -> str:
    raw = image_path.read_bytes()
    b64 = base64.b64encode(raw).decode("ascii")
    ext = image_path.suffix.lower().lstrip(".")
    mime = "image/png" if ext == "png" else "image/jpeg" if ext in {"jpg", "jpeg"} else f"image/{ext}"
    return f"data:{mime};base64,{b64}"


def _extract_first_json_object(text: str) -> Any:
    """Extract the first JSON object/array from a text response.

    Handles common patterns like ```json ... ``` wrappers.
    """
    if text is None:
        raise ValueError("Empty response")

    t = text.strip()

    # Strip fenced code blocks if present.
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", t)
        t = re.sub(r"\s*```$", "", t).strip()

    # Fast path: full JSON.
    try:
        return json.loads(t)
    except Exception:
        pass

    # Try to locate first object/array by scanning for balanced braces/brackets.
    start_candidates = [i for i, ch in enumerate(t) if ch in "[{"]
    for start in start_candidates:
        stack: list[str] = []
        in_str = False
        escape = False
        for i in range(start, len(t)):
            ch = t[i]
            if in_str:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_str = False
                continue

            if ch == '"':
                in_str = True
                continue

            if ch in "[{":
                stack.append(ch)
            elif ch in "]}":
                if not stack:
                    break
                opener = stack.pop()
                if (opener == "[" and ch != "]") or (opener == "{" and ch != "}"):
                    break
                if not stack:
                    snippet = t[start : i + 1]
                    try:
                        return json.loads(snippet)
                    except Exception:
                        break

    raise ValueError("Could not parse JSON from model response")


def _normalize_chat_completions_url(base_url: str) -> str:
    """Normalize OpenAI-compatible base URL to a chat/completions endpoint.

    Gemini OpenAI-compat docs often show:
      https://generativelanguage.googleapis.com/v1beta/openai/
    while REST examples use:
      https://generativelanguage.googleapis.com/v1beta/openai/chat/completions

    This script uses raw HTTP, so we ensure the final URL targets chat/completions.
    """
    u = (base_url or "").strip()
    if not u:
        return u

    u = u.rstrip("/")
    if u.endswith("/chat/completions"):
        return u

    # If user provided the OpenAI-compat base (ending in /openai), append endpoint.
    if u.endswith("/openai"):
        return u + "/chat/completions"

    # Otherwise keep as-is.
    return u


def _normalize_openai_sdk_base_url(base_url: str) -> str:
    """Normalize base_url for the OpenAI Python SDK.

    The OpenAI SDK expects a *base* URL, e.g.:
      https://generativelanguage.googleapis.com/v1beta/openai/

    If a user configured the full endpoint (/chat/completions), we strip it.
    """
    u = (base_url or "").strip()
    if not u:
        return u

    u = u.rstrip("/")
    if u.endswith("/chat/completions"):
        u = u[: -len("/chat/completions")]

    # Keep trailing slash for consistency with docs.
    return u.rstrip("/") + "/"


def _call_openai_compatible_vision(*, prompt_text: str, front_image: Path, back_image: Path) -> str:
    api_key = os.getenv("AI_API_KEY") or os.getenv("AI_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing AI_API_KEY / AI_API_KEY / OPENAI_API_KEY environment variable")

    provider = (os.getenv("AI_PROVIDER") or "").strip().lower()

    default_base_url = "https://api.openai.com/v1/chat/completions"
    default_model = "gpt-4o-mini"

    if provider == "gemini":
        default_base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
        default_model = "gemini-2.5-flash"

    base_url = os.getenv("AI_BASE_URL") or default_base_url
    base_url = _normalize_openai_sdk_base_url(base_url)
    model = os.getenv("AI_MODEL") or default_model
    timeout_sec = int(os.getenv("AI_TIMEOUT_SEC") or "120")

    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=timeout_sec,
    )

    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": prompt_text},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Front page and back page images."},
                    {"type": "image_url", "image_url": {"url": _encode_image_data_url(front_image)}},
                    {"type": "image_url", "image_url": {"url": _encode_image_data_url(back_image)}},
                ],
            },
        ],
    }

    # Some providers accept OpenAI-style structured output; others may reject it.
    # Only include it when explicitly requested.
    resp_format = (os.getenv("AI_RESPONSE_FORMAT") or "").strip().lower()
    if resp_format == "json_object":
        payload["response_format"] = {"type": "json_object"}

    resp = client.chat.completions.create(**payload)
    # OpenAI-compatible shape
    try:
        return str(resp.choices[0].message.content)
    except Exception as e:
        raise RuntimeError(f"Unexpected response shape from AI SDK: {e}")


def main() -> int:
    script_dir = Path(__file__).resolve().parent
    default_extracted = script_dir / "extracted"
    default_prompt = script_dir / "prompt.md"

    # Load .env if present (local dev convenience)
    _load_dotenv(script_dir / ".env")

    parser = argparse.ArgumentParser(
        description="Send first + back-page images to AI and store extracted JSON per folder."
    )
    parser.add_argument("--extracted", type=Path, default=default_extracted, help="Root extracted directory")
    parser.add_argument("--prompt", type=Path, default=default_prompt, help="Path to prompt markdown")
    parser.add_argument("--limit", type=int, default=0, help="Process only first N folders (0 = all)")
    parser.add_argument("--force", action="store_true", help="Overwrite existing outputs")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print selected images + resolved endpoint/model without calling the AI.",
    )
    args = parser.parse_args()

    extracted_dir: Path = args.extracted
    prompt_path: Path = args.prompt

    if not prompt_path.exists():
        print(f"Prompt not found: {prompt_path}")
        return 2

    prompt_text = prompt_path.read_text(encoding="utf-8")

    folders = _iter_extracted_folders(extracted_dir)
    if args.limit and args.limit > 0:
        folders = folders[: args.limit]

    if not folders:
        print(f"No extracted folders found under: {extracted_dir}")
        return 2

    ok = 0
    failed = 0

    for folder in folders:
        file_name = folder.name
        images_dir = folder / "images"
        out_dir = folder / "output"
        out_path = out_dir / f"{file_name}.json"

        if out_path.exists() and not args.force:
            print(f"SKIP (exists): {out_path}")
            continue

        imgs = _sorted_images(images_dir)
        if len(imgs) < 2:
            print(f"SKIP (need >=2 images): {images_dir}")
            continue

        front = imgs[0]
        back = _pick_back_page(imgs)

        out_dir.mkdir(parents=True, exist_ok=True)

        if args.dry_run:
            # Resolve provider defaults the same way as the call path.
            provider = (os.getenv("AI_PROVIDER") or "").strip().lower() or "(default)"
            default_base_url = "https://api.openai.com/v1/chat/completions"
            default_model = "gpt-4o-mini"
            if provider == "gemini":
                default_base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
                default_model = "gemini-2.5-flash"
            base_url = _normalize_openai_sdk_base_url(os.getenv("AI_BASE_URL") or default_base_url)
            model = os.getenv("AI_MODEL") or default_model
            print(
                f"DRYRUN: {file_name} -> {out_path.name} "
                f"(front={front.name}, back={back.name}, provider={provider}, model={model})\n"
                f"        endpoint={base_url}"
            )
            continue

        content: Optional[str] = None
        try:
            content = _call_openai_compatible_vision(
                prompt_text=prompt_text,
                front_image=front,
                back_image=back,
            )

            obj = _extract_first_json_object(content)
            out_path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
            ok += 1
            print(f"OK: {file_name} -> {out_path.name} (front={front.name}, back={back.name})")

        except Exception as e:
            failed += 1
            raw_path = out_dir / f"{file_name}.raw.txt"
            try:
                with raw_path.open("w", encoding="utf-8") as f:
                    f.write(
                        f"ERROR: {e}\n"
                        f"front={front}\n"
                        f"back={back}\n"
                    )
                    if content:
                        f.write("\n--- MODEL OUTPUT (raw) ---\n")
                        f.write(content)
            except Exception:
                pass
            print(f"FAIL: {file_name}: {e}")

    print(f"Done. OK={ok} FAIL={failed}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
