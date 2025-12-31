import os
import glob
import cv2
import numpy as np
import argparse
import re
import time

# ========== PATHS ==========
# New layout:
#   extracted/<file_name>/images/<page image>
# Output:
#   extracted/<file_name>/crops/<image_name>/<crop_name>
EXTRACTED_DIR = "extracted"
# ===========================

CANON_W, CANON_H = 1187, 1679

# Box filters (detection)
MIN_BOX_AREA_FRAC = 0.006
MAX_BOX_AREA_FRAC = 0.25
MIN_AR, MAX_AR = 0.55, 2.8

# Crop padding on original
PAD = 3

# Grid-line extraction parameters (detection)
HLINE_SCALE = 25
VLINE_SCALE = 25


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def list_images(input_dir: str):
    exts = ("*.png", "*.jpg", "*.jpeg", "*.webp", "*.tif", "*.tiff", "*.bmp")
    files = []
    for e in exts:
        files.extend(glob.glob(os.path.join(input_dir, e)))
        files.extend(glob.glob(os.path.join(input_dir, e.upper())))
    return sorted(set(files))


def list_extracted_image_dirs(extracted_root: str):
    if not os.path.isdir(extracted_root):
        return []
    out = []
    for name in sorted(os.listdir(extracted_root)):
        p = os.path.join(extracted_root, name)
        if not os.path.isdir(p):
            continue
        images_dir = os.path.join(p, "images")
        if os.path.isdir(images_dir):
            out.append((name, images_dir))
    return out


def derive_page_id(image_path: str) -> str:
    """Derive a stable page id from an image filename.

    Examples:
      page-003-img-01.png -> page-003
      page-003_img_01.jpg -> page-003
    """
    base = os.path.splitext(os.path.basename(image_path))[0]
    return re.sub(r"(?:[-_]img[-_]?\d+)$", "", base, flags=re.IGNORECASE)


def sort_boxes_reading_order(boxes):
    if not boxes:
        return boxes
    heights = sorted(h for _, _, _, h in boxes)
    median_h = heights[len(heights) // 2]
    row_thresh = max(10, int(median_h * 0.55))

    boxes_sorted = sorted(boxes, key=lambda b: (b[1], b[0]))
    rows, cur = [], [boxes_sorted[0]]
    for b in boxes_sorted[1:]:
        if abs(b[1] - cur[-1][1]) <= row_thresh:
            cur.append(b)
        else:
            rows.append(sorted(cur, key=lambda x: x[0]))
            cur = [b]
    rows.append(sorted(cur, key=lambda x: x[0]))

    out = []
    for r in rows:
        out.extend(r)
    return out


def dedupe_boxes(boxes, tol=6):
    boxes = sorted(boxes, key=lambda b: (b[0], b[1], b[2], b[3]))
    out = []
    for x, y, w, h in boxes:
        dup = False
        for x2, y2, w2, h2 in out:
            if abs(x - x2) < tol and abs(y - y2) < tol and abs(w - w2) < tol and abs(h - h2) < tol:
                dup = True
                break
        if not dup:
            out.append((x, y, w, h))
    return out


def detect_boxes_using_grid_lines(img_bgr_canon):
    H, W = img_bgr_canon.shape[:2]
    page_area = W * H

    gray = cv2.cvtColor(img_bgr_canon, cv2.COLOR_BGR2GRAY)
    bw = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        31, 12
    )

    # horizontal lines
    h_kernel_len = max(10, W // HLINE_SCALE)
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_kernel_len, 1))
    horiz = cv2.erode(bw, h_kernel, iterations=1)
    horiz = cv2.dilate(horiz, h_kernel, iterations=1)

    # vertical lines
    v_kernel_len = max(10, H // VLINE_SCALE)
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_kernel_len))
    vert = cv2.erode(bw, v_kernel, iterations=1)
    vert = cv2.dilate(vert, v_kernel, iterations=1)

    grid = cv2.bitwise_or(horiz, vert)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    grid = cv2.morphologyEx(grid, cv2.MORPH_CLOSE, k, iterations=2)

    contours, _ = cv2.findContours(grid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        if area < page_area * MIN_BOX_AREA_FRAC:
            continue
        if area > page_area * MAX_BOX_AREA_FRAC:
            continue
        ar = w / float(h)
        if not (MIN_AR <= ar <= MAX_AR):
            continue

        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) < 4:
            continue

        boxes.append((x, y, w, h))

    boxes = sort_boxes_reading_order(dedupe_boxes(boxes))
    return boxes


def scale_boxes(boxes_canon, sx, sy):
    out = []
    for x, y, w, h in boxes_canon:
        ox = int(round(x * sx))
        oy = int(round(y * sy))
        ow = int(round(w * sx))
        oh = int(round(h * sy))
        out.append((ox, oy, ow, oh))
    return out


# -------- OCR preprocessing --------

def estimate_skew_angle(gray):
    """
    Estimate skew using Hough lines.
    Returns angle in degrees. Small angles only; safe for documents.
    """
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=80,
                            minLineLength=max(50, gray.shape[1]//3),
                            maxLineGap=10)
    if lines is None:
        return 0.0

    angles = []
    for x1, y1, x2, y2 in lines[:, 0]:
        dx, dy = (x2 - x1), (y2 - y1)
        if dx == 0:
            continue
        ang = np.degrees(np.arctan2(dy, dx))
        # keep near-horizontal lines only
        if -30 <= ang <= 30:
            angles.append(ang)

    if not angles:
        return 0.0

    # median is robust
    return float(np.median(angles))


def rotate_image(img, angle_deg):
    if abs(angle_deg) < 0.2:
        return img
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), angle_deg, 1.0)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REPLICATE)


def ocr_preprocess(crop_bgr):
    # grayscale
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)

    # deskew
    angle = estimate_skew_angle(gray)
    if abs(angle) > 0.2:
        gray = rotate_image(gray, angle)

    # upscale AFTER crop
    gray = cv2.resize(
        gray, None,
        fx=2.0, fy=2.0,
        interpolation=cv2.INTER_CUBIC
    )

    # light denoise (does NOT thicken strokes)
    gray = cv2.fastNlMeansDenoising(
        gray, None,
        h=8,                # lower = thinner strokes
        templateWindowSize=7,
        searchWindowSize=21
    )

    # contrast normalization (safe)
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)

    return gray


def parse_args():
    p = argparse.ArgumentParser(
        description="Crop voter boxes from extracted/<file>/images and save into extracted/<file>/crops."
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit processing to first N extracted folders (sorted).",
    )
    return p.parse_args()


def main(limit=None):
    print("Working dir:", os.getcwd())
    extracted_root = os.path.abspath(EXTRACTED_DIR)
    print("Input root:", extracted_root)

    extracted = list_extracted_image_dirs(EXTRACTED_DIR)
    print(f"Found {len(extracted)} extracted folder(s) with images/.")
    if not extracted:
        return

    if limit is not None:
        if limit <= 0:
            print("Limit <= 0; nothing to do.")
            return
        extracted = extracted[:limit]
        print(f"Limiting to {len(extracted)} extracted folder(s).")

    for file_name, images_dir in extracted:
        page_images = list_images(images_dir)
        print(f"\n[{file_name}] Found {len(page_images)} page image(s) in {os.path.abspath(images_dir)}")
        if not page_images:
            continue

        for p in page_images:
            page_t0 = time.perf_counter()
            img_orig = cv2.imread(p)
            if img_orig is None:
                print(f"[skip] unreadable: {p}")
                continue

            H0, W0 = img_orig.shape[:2]
            img_canon = cv2.resize(img_orig, (CANON_W, CANON_H), interpolation=cv2.INTER_AREA)

            boxes_canon = detect_boxes_using_grid_lines(img_canon)
            page_base = os.path.splitext(os.path.basename(p))[0]
            page_id = derive_page_id(p)
            print(f"  {os.path.basename(p)} ({W0}x{H0}): detected {len(boxes_canon)} box(es)")
            if not boxes_canon:
                elapsed_s = time.perf_counter() - page_t0
                print(f"    -> Page time: {elapsed_s:.2f}s (no crops saved)")
                continue

            sx = W0 / float(CANON_W)
            sy = H0 / float(CANON_H)
            boxes_orig = scale_boxes(boxes_canon, sx, sy)

            crops_dir = os.path.join(EXTRACTED_DIR, file_name, "crops", page_id)
            ensure_dir(crops_dir)

            saved = 0
            for i, (x, y, w, h) in enumerate(boxes_orig, start=1):
                x1 = max(0, x - PAD)
                y1 = max(0, y - PAD)
                x2 = min(W0, x + w + PAD)
                y2 = min(H0, y + h + PAD)
                if x2 <= x1 or y2 <= y1:
                    continue

                crop = img_orig[y1:y2, x1:x2]

                ocr_img = ocr_preprocess(crop)

                # Default: page-003-001.png
                crop_name = f"{page_id}-{i:03d}.png"
                out_path = os.path.join(crops_dir, crop_name)

                # If multiple images map to the same page_id (e.g. page-003-img-01 and
                # page-003-img-02), prevent overwrites by adding a disambiguator.
                if os.path.exists(out_path) and page_base != page_id:
                    extra = page_base[len(page_id):].lstrip("-_")
                    if extra:
                        crop_name = f"{page_id}-{extra}-{i:03d}.png"
                        out_path = os.path.join(crops_dir, crop_name)
                # lossless PNG, minimal compression artifacts (compression=0 is largest files)
                cv2.imwrite(out_path, ocr_img, [cv2.IMWRITE_PNG_COMPRESSION, 0])
                saved += 1

            print(f"    -> Saved {saved} OCR-ready crops into {os.path.abspath(crops_dir)}")
            elapsed_s = time.perf_counter() - page_t0
            per_crop_ms = (elapsed_s / saved * 1000.0) if saved else 0.0
            print(f"    -> Page time: {elapsed_s:.2f}s ({per_crop_ms:.1f} ms/crop)")

    print("Done.")


if __name__ == "__main__":
    args = parse_args()
    main(limit=args.limit)
