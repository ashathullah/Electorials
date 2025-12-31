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


# -------- Non-voter (diagram) crop filtering --------
# Heuristic: voter info boxes tend to have lots of small connected components (text)
# once long straight lines are removed, while diagrams/maps/photos are often
# line-dominant and have few small components.
VOTER_INK_FRAC_MIN = 0.008
VOTER_INK_FRAC_MAX = 0.28
VOTER_MAX_LINE_RATIO = 0.60
VOTER_MIN_SMALL_COMPONENTS = 35

# Photos and many diagrams tend to have a few dominant blobs after line removal,
# unlike voter boxes which have many similarly-sized glyph components.
VOTER_MAX_LARGEST_CC_RATIO = 0.22

# Voter boxes contain dense text edges; photos often have smoother gradients.
VOTER_MIN_EDGE_FRAC = 0.02

# Final safety check on the OCR-preprocessed grayscale (closer to what OCR sees).
POST_OCR_MIN_EDGE_FRAC = 0.02

# If auto-filter rejects everything, treat the page as "confidently non-voter"
# (skip page) when the average metrics are clearly non-text-like.
AUTO_SKIP_MAX_SMALL_COMPONENTS = 14
AUTO_SKIP_MIN_LINE_RATIO = 0.70
AUTO_SKIP_MIN_LARGEST_CC_RATIO = 0.35
AUTO_SKIP_MAX_EDGE_FRAC = 0.015


def _extract_hv_lines(bw_inv):
    """Extract horizontal/vertical lines from a binary-inverted image."""
    h, w = bw_inv.shape[:2]

    # Important: keep this conservative (long kernels) so regular text strokes
    # don't get misclassified as "lines" and inflate line_ratio.
    h_kernel_len = max(40, w // 6)
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_kernel_len, 1))
    horiz = cv2.erode(bw_inv, h_kernel, iterations=1)
    horiz = cv2.dilate(horiz, h_kernel, iterations=1)

    v_kernel_len = max(40, h // 6)
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_kernel_len))
    vert = cv2.erode(bw_inv, v_kernel, iterations=1)
    vert = cv2.dilate(vert, v_kernel, iterations=1)

    lines = cv2.bitwise_or(horiz, vert)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    lines = cv2.morphologyEx(lines, cv2.MORPH_CLOSE, k, iterations=1)
    return lines


def is_voter_info_crop(crop_bgr):
    """Return True if crop looks like a voter info box; False for diagrams/photos."""
    is_voter, _ = classify_crop_metrics(crop_bgr)
    return is_voter


def classify_crop_metrics(crop_bgr):
    """Return (is_voter, metrics dict) used by auto mode for confidence decisions."""
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    edges = cv2.Canny(gray, 50, 150)
    edge_frac = float(np.count_nonzero(edges)) / float(edges.size)

    bw = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        12,
    )

    ink_px = int(np.count_nonzero(bw))
    ink_frac = float(ink_px) / float(bw.size)

    lines = _extract_hv_lines(bw)
    line_px = int(np.count_nonzero(lines))
    line_ratio = float(line_px) / float(max(1, ink_px))

    # Remove long lines; count small components (text glyphs)
    bw_wo_lines = cv2.bitwise_and(bw, cv2.bitwise_not(lines))

    # Reduce salt/pepper noise that can make photos look like "lots of text".
    k2 = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    bw_wo_lines = cv2.morphologyEx(bw_wo_lines, cv2.MORPH_OPEN, k2, iterations=1)

    nlabels, _, stats, _ = cv2.connectedComponentsWithStats(bw_wo_lines, connectivity=8)

    area_total = bw_wo_lines.shape[0] * bw_wo_lines.shape[1]
    # dynamic upper bound: avoid counting big diagram strokes; allow glyph-sized blobs
    max_small_area = max(80, int(area_total * 0.002))
    min_small_area = 10
    small_components = 0
    largest_cc_area = 0
    for i in range(1, nlabels):  # 0 is background
        a = int(stats[i, cv2.CC_STAT_AREA])
        if a > largest_cc_area:
            largest_cc_area = a
        if min_small_area <= a <= max_small_area:
            small_components += 1

    ink_wo_lines_px = int(np.count_nonzero(bw_wo_lines))
    largest_cc_ratio = float(largest_cc_area) / float(max(1, ink_wo_lines_px))

    is_voter = True
    if not (VOTER_INK_FRAC_MIN <= ink_frac <= VOTER_INK_FRAC_MAX):
        is_voter = False
    if line_ratio > VOTER_MAX_LINE_RATIO:
        is_voter = False
    if small_components < VOTER_MIN_SMALL_COMPONENTS:
        is_voter = False
    if largest_cc_ratio > VOTER_MAX_LARGEST_CC_RATIO:
        is_voter = False
    if edge_frac < VOTER_MIN_EDGE_FRAC:
        is_voter = False

    metrics = {
        "ink_frac": ink_frac,
        "line_ratio": line_ratio,
        "small_components": small_components,
        "largest_cc_ratio": largest_cc_ratio,
        "edge_frac": edge_frac,
    }
    return is_voter, metrics


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


def remove_contained_boxes(boxes, margin=6, max_area_ratio=0.70):
    """Remove boxes fully contained within another box.

    This helps avoid cropping nested sub-cells like the photo placeholder square
    inside a larger voter-info row box.
    """
    if not boxes:
        return boxes

    boxes_sorted = sorted(boxes, key=lambda b: (b[2] * b[3]), reverse=True)
    kept = []
    for x, y, w, h in boxes_sorted:
        x2, y2 = x + w, y + h
        area = w * h
        contained = False
        for X, Y, W, H in kept:
            X2, Y2 = X + W, Y + H
            if (
                x >= X + margin
                and y >= Y + margin
                and x2 <= X2 - margin
                and y2 <= Y2 - margin
            ):
                big_area = W * H
                if area / float(max(1, big_area)) <= max_area_ratio:
                    contained = True
                    break
        if not contained:
            kept.append((x, y, w, h))

    # Return in original-ish order; caller sorts into reading order later.
    return sorted(kept, key=lambda b: (b[1], b[0]))


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

    boxes = dedupe_boxes(boxes)
    boxes = remove_contained_boxes(boxes)
    boxes = sort_boxes_reading_order(boxes)
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
    p.add_argument(
        "--diagram-filter",
        choices=("auto", "on", "off"),
        default="auto",
        help=(
            "Filter out diagram/photo-like boxes. "
            "auto=filter but fall back to saving all boxes if it rejects everything on a page; "
            "on=strict filter (may skip entire pages); off=disable filter."
        ),
    )
    return p.parse_args()


def main(limit=None, diagram_filter="auto"):
    run_t0 = time.perf_counter()
    total_folders = 0
    planned_pages = 0
    processed_pages = 0
    unreadable_pages = 0
    skipped_pages = 0
    total_saved_crops = 0
    total_elapsed_pages_s = 0.0

    print("Working dir:", os.getcwd())
    extracted_root = os.path.abspath(EXTRACTED_DIR)
    print("Input root:", extracted_root)
    print("Diagram filter mode:", diagram_filter)

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

    # Pre-scan page counts so we can print a reliable total runtime summary.
    work = []  # list[(file_name, images_dir, page_images)]
    for file_name, images_dir in extracted:
        page_images = list_images(images_dir)
        work.append((file_name, images_dir, page_images))
        planned_pages += len(page_images)

    for file_name, images_dir, page_images in work:
        total_folders += 1
        print(f"\n[{file_name}] Found {len(page_images)} page image(s) in {os.path.abspath(images_dir)}")
        if not page_images:
            continue

        for p in page_images:
            page_t0 = time.perf_counter()
            img_orig = cv2.imread(p)
            if img_orig is None:
                print(f"[skip] unreadable: {p}")
                unreadable_pages += 1
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
                processed_pages += 1
                skipped_pages += 1
                total_elapsed_pages_s += elapsed_s
                continue

            sx = W0 / float(CANON_W)
            sy = H0 / float(CANON_H)
            boxes_orig = scale_boxes(boxes_canon, sx, sy)

            # First pass: classify boxes, but don't create the page folder unless
            # at least one voter-like crop exists.
            voter_candidates = []  # list[(i, x1, y1, x2, y2)]
            skipped_non_voter = 0
            rejected_metrics = []
            for i, (x, y, w, h) in enumerate(boxes_orig, start=1):
                x1 = max(0, x - PAD)
                y1 = max(0, y - PAD)
                x2 = min(W0, x + w + PAD)
                y2 = min(H0, y + h + PAD)
                if x2 <= x1 or y2 <= y1:
                    continue

                if diagram_filter == "off":
                    voter_candidates.append((i, x1, y1, x2, y2))
                    continue

                crop = img_orig[y1:y2, x1:x2]
                ok, metrics = classify_crop_metrics(crop)
                if ok:
                    voter_candidates.append((i, x1, y1, x2, y2))
                else:
                    skipped_non_voter += 1
                    rejected_metrics.append(metrics)

            # Auto mode: if everything is rejected, only "fail-open" when uncertain.
            if diagram_filter == "auto" and not voter_candidates and boxes_orig:
                avg_small = 0.0
                avg_line_ratio = 0.0
                avg_largest_cc = 0.0
                avg_edge = 0.0
                if rejected_metrics:
                    avg_small = float(np.mean([m["small_components"] for m in rejected_metrics]))
                    avg_line_ratio = float(np.mean([m["line_ratio"] for m in rejected_metrics]))
                    avg_largest_cc = float(np.mean([m["largest_cc_ratio"] for m in rejected_metrics]))
                    avg_edge = float(np.mean([m.get("edge_frac", 0.0) for m in rejected_metrics]))

                confidently_non_voter = (
                    rejected_metrics
                    and (
                        avg_small <= AUTO_SKIP_MAX_SMALL_COMPONENTS
                        or avg_line_ratio >= AUTO_SKIP_MIN_LINE_RATIO
                        or avg_largest_cc >= AUTO_SKIP_MIN_LARGEST_CC_RATIO
                        or avg_edge <= AUTO_SKIP_MAX_EDGE_FRAC
                    )
                )

                if not confidently_non_voter:
                    voter_candidates = []
                    for i, (x, y, w, h) in enumerate(boxes_orig, start=1):
                        x1 = max(0, x - PAD)
                        y1 = max(0, y - PAD)
                        x2 = min(W0, x + w + PAD)
                        y2 = min(H0, y + h + PAD)
                        if x2 <= x1 or y2 <= y1:
                            continue
                        voter_candidates.append((i, x1, y1, x2, y2))
                    print(
                        f"    -> Warning: diagram filter rejected all boxes but looks uncertain "
                        f"(avg_small={avg_small:.1f}, avg_line_ratio={avg_line_ratio:.2f}, avg_largest_cc={avg_largest_cc:.2f}, avg_edge={avg_edge:.3f}); "
                        f"saving all {len(voter_candidates)} crop(s). "
                        "Use --diagram-filter on to force strict skipping."
                    )
                    skipped_non_voter = 0

            if not voter_candidates:
                elapsed_s = time.perf_counter() - page_t0
                if skipped_non_voter:
                    print(f"    -> Skipped page (all {skipped_non_voter} box(es) were diagram/photo-like)")
                else:
                    print("    -> Skipped page (no valid crops)")
                print(f"    -> Page time: {elapsed_s:.2f}s (no crops saved)")
                processed_pages += 1
                skipped_pages += 1
                total_elapsed_pages_s += elapsed_s
                continue

            crops_dir = os.path.join(EXTRACTED_DIR, file_name, "crops", page_id)
            saved = 0
            skipped_post_ocr = 0
            created_dir = False
            for i, x1, y1, x2, y2 in voter_candidates:
                crop = img_orig[y1:y2, x1:x2]
                ocr_img = ocr_preprocess(crop)

                if diagram_filter != "off":
                    edges = cv2.Canny(ocr_img, 50, 150)
                    edge_frac_post = float(np.count_nonzero(edges)) / float(edges.size)
                    if edge_frac_post < POST_OCR_MIN_EDGE_FRAC:
                        skipped_post_ocr += 1
                        continue

                if not created_dir:
                    ensure_dir(crops_dir)
                    created_dir = True

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

            if saved:
                print(f"    -> Saved {saved} OCR-ready crops into {os.path.abspath(crops_dir)}")
            else:
                if created_dir:
                    try:
                        os.rmdir(crops_dir)
                    except OSError:
                        pass
                print("    -> Saved 0 OCR-ready crops (all candidates looked non-text after OCR preprocessing)")
            elapsed_s = time.perf_counter() - page_t0
            processed_pages += 1
            total_saved_crops += saved
            total_elapsed_pages_s += elapsed_s
            per_crop_ms = (elapsed_s / saved * 1000.0) if saved else 0.0
            if skipped_non_voter:
                print(f"    -> Skipped {skipped_non_voter} non-voter box(es) (diagram/photo-like)")
            if skipped_post_ocr:
                print(f"    -> Skipped {skipped_post_ocr} non-text crop(s) after OCR preprocessing")
            print(f"    -> Page time: {elapsed_s:.2f}s ({per_crop_ms:.1f} ms/crop)")

    run_elapsed_s = time.perf_counter() - run_t0
    avg_page_s = (total_elapsed_pages_s / processed_pages) if processed_pages else 0.0
    avg_crop_ms = (total_elapsed_pages_s / total_saved_crops * 1000.0) if total_saved_crops else 0.0

    print("\nDone.")
    print(
        "Summary: "
        f"folders={total_folders}, planned_pages={planned_pages}, processed_pages={processed_pages}, "
        f"unreadable_pages={unreadable_pages}, skipped_pages={skipped_pages}, saved_crops={total_saved_crops}"
    )
    print(
        "Timing: "
        f"total_elapsed={run_elapsed_s:.2f}s, avg_page={avg_page_s:.2f}s/page, avg_crop={avg_crop_ms:.1f} ms/crop"
    )


if __name__ == "__main__":
    args = parse_args()
    main(limit=args.limit, diagram_filter=args.diagram_filter)
