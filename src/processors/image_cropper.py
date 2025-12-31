"""
Image Cropper processor.

Detects and crops voter information boxes from page images
using grid-line detection and morphological operations.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple, Optional, Any

import cv2
import numpy as np

from .base import BaseProcessor, ProcessingContext
from ..exceptions import CropExtractionError
from ..utils.file_utils import derive_page_id


# Canonical image dimensions for consistent detection
CANON_W, CANON_H = 1187, 1679

# Box detection filters
MIN_BOX_AREA_FRAC = 0.006
MAX_BOX_AREA_FRAC = 0.25
MIN_AR, MAX_AR = 0.55, 2.8

# Crop padding
PAD = 3

# Grid-line extraction parameters
HLINE_SCALE = 25
VLINE_SCALE = 25

# Voter info box detection thresholds
VOTER_INK_FRAC_MIN = 0.008  # Restored - works for all voter pages
VOTER_INK_FRAC_MAX = 0.28   # Restored - works for all voter pages
VOTER_MAX_LINE_RATIO = 0.55  # Slightly relaxed from 0.60 - voter pages have ~0.41-0.50
VOTER_MIN_SMALL_COMPONENTS = 15  # Lowered from 35 - voter pages have 18-38, some have 18
VOTER_MAX_LARGEST_CC_RATIO = 0.25  # Slightly raised from 0.22 - voter pages have max ~0.20
VOTER_MIN_EDGE_FRAC = 0.07  # Key differentiator: voter=0.08+, non-voter=0.03-0.06
POST_OCR_MIN_EDGE_FRAC = 0.03  # Lower threshold for preprocessed images (upscaling reduces edge density)

# Auto-skip thresholds for non-voter pages
AUTO_SKIP_MAX_SMALL_COMPONENTS = 14
AUTO_SKIP_MIN_LINE_RATIO = 0.70
AUTO_SKIP_MIN_LARGEST_CC_RATIO = 0.35
AUTO_SKIP_MAX_EDGE_FRAC = 0.015


@dataclass
class CropMetrics:
    """Metrics for classifying crops as voter/non-voter."""
    ink_frac: float
    line_ratio: float
    small_components: int
    largest_cc_ratio: float
    edge_frac: float


@dataclass
class CroppedBox:
    """Information about a cropped voter box."""
    page_id: str
    box_index: int
    x1: int
    y1: int
    x2: int
    y2: int
    output_path: Optional[Path] = None
    
    @property
    def width(self) -> int:
        return self.x2 - self.x1
    
    @property
    def height(self) -> int:
        return self.y2 - self.y1


@dataclass
class PageCropResult:
    """Result of cropping a single page."""
    page_id: str
    page_path: Path
    total_boxes_detected: int
    crops_saved: int
    skipped_diagram: int
    skipped_post_ocr: int
    elapsed_seconds: float
    crops: List[CroppedBox] = field(default_factory=list)


@dataclass
class CropSummary:
    """Summary of cropping results for a document."""
    pdf_name: str
    total_pages: int
    processed_pages: int
    skipped_pages: int
    unreadable_pages: int
    total_crops: int
    elapsed_seconds: float


class ImageCropper(BaseProcessor):
    """
    Crop voter information boxes from page images.
    
    Uses grid-line detection and morphological operations to:
    1. Detect table cell boundaries
    2. Filter for voter info boxes (vs diagrams/photos)
    3. Apply OCR preprocessing
    4. Save cropped images
    """
    
    name = "ImageCropper"
    
    def __init__(
        self,
        context: ProcessingContext,
        diagram_filter: str = "auto",
    ):
        """
        Initialize cropper.
        
        Args:
            context: Processing context
            diagram_filter: "auto", "on", or "off"
        """
        super().__init__(context)
        self.diagram_filter = diagram_filter
        self.page_results: List[PageCropResult] = []
        self.summary: Optional[CropSummary] = None
    
    def validate(self) -> bool:
        """Validate prerequisites."""
        if not self.context.images_dir:
            self.log_error("Images directory not set")
            return False
        
        if not self.context.images_dir.exists():
            self.log_error(f"Images directory not found: {self.context.images_dir}")
            return False
        
        return True
    
    def process(self) -> bool:
        """
        Process all page images in the document.
        
        Returns:
            True if processing succeeded
        """
        import time
        
        images_dir = self.context.images_dir
        page_images = self._list_images(images_dir)
        
        if not page_images:
            self.log_warning(f"No images found in {images_dir}")
            return False
        
        self.log_info(f"Found {len(page_images)} page image(s)")
        
        total_pages = len(page_images)
        processed = 0
        skipped = 0
        unreadable = 0
        total_crops = 0
        
        run_start = time.perf_counter()
        
        for img_path in page_images:
            result = self._process_page(img_path)
            
            if result is None:
                unreadable += 1
                continue
            
            self.page_results.append(result)
            processed += 1
            
            if result.crops_saved == 0:
                skipped += 1
            else:
                total_crops += result.crops_saved
        
        elapsed = time.perf_counter() - run_start
        
        self.summary = CropSummary(
            pdf_name=self.context.pdf_name,
            total_pages=total_pages,
            processed_pages=processed,
            skipped_pages=skipped,
            unreadable_pages=unreadable,
            total_crops=total_crops,
            elapsed_seconds=elapsed,
        )
        
        self.log_info(
            f"Cropping complete",
            pages=processed,
            crops=total_crops,
            skipped=skipped,
            time=f"{elapsed:.2f}s"
        )
        
        return True
    
    def _list_images(self, input_dir: Path) -> List[Path]:
        """List all image files in directory."""
        exts = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".bmp"}
        images = [
            p for p in input_dir.iterdir()
            if p.is_file() and p.suffix.lower() in exts
        ]
        return sorted(images, key=lambda p: p.name.lower())
    
    def _process_page(self, img_path: Path) -> Optional[PageCropResult]:
        """
        Process a single page image.
        
        Args:
            img_path: Path to page image
        
        Returns:
            PageCropResult or None if unreadable
        """
        import time
        
        page_start = time.perf_counter()
        page_id = derive_page_id(img_path)
        
        # Read image
        img_orig = cv2.imdecode(
            np.fromfile(str(img_path), dtype=np.uint8),
            cv2.IMREAD_COLOR
        )
        
        if img_orig is None:
            self.log_warning(f"Unreadable image: {img_path.name}")
            return None
        
        H0, W0 = img_orig.shape[:2]
        
        # Resize to canonical dimensions for detection
        img_canon = cv2.resize(img_orig, (CANON_W, CANON_H), interpolation=cv2.INTER_AREA)
        
        # Detect boxes
        boxes_canon = self._detect_boxes(img_canon)
        
        self.log_debug(
            f"Page {page_id}",
            size=f"{W0}x{H0}",
            boxes=len(boxes_canon)
        )
        
        if not boxes_canon:
            elapsed = time.perf_counter() - page_start
            return PageCropResult(
                page_id=page_id,
                page_path=img_path,
                total_boxes_detected=0,
                crops_saved=0,
                skipped_diagram=0,
                skipped_post_ocr=0,
                elapsed_seconds=elapsed,
            )
        
        # Scale boxes back to original dimensions
        sx = W0 / float(CANON_W)
        sy = H0 / float(CANON_H)
        boxes_orig = self._scale_boxes(boxes_canon, sx, sy)
        
        # Classify boxes
        voter_candidates, skipped_diagram, rejected_metrics = self._classify_boxes(
            img_orig, boxes_orig, W0, H0, page_id
        )
        
        self.log_debug(
            f"Page {page_id}: detected={len(boxes_orig)}, accepted={len(voter_candidates)}, rejected={skipped_diagram}"
        )
        
        # Auto mode: fail-open if uncertain
        if self.diagram_filter == "auto" and not voter_candidates and boxes_orig:
            if not self._is_confidently_non_voter(rejected_metrics):
                voter_candidates = self._all_boxes_as_candidates(boxes_orig, W0, H0)
                skipped_diagram = 0
                self.log_debug(
                    f"Auto mode: saving all {len(voter_candidates)} boxes (uncertain rejection)"
                )
        
        if not voter_candidates:
            elapsed = time.perf_counter() - page_start
            if skipped_diagram:
                self.log_debug(f"Skipped page {page_id}: all {skipped_diagram} boxes diagram-like")
            return PageCropResult(
                page_id=page_id,
                page_path=img_path,
                total_boxes_detected=len(boxes_orig),
                crops_saved=0,
                skipped_diagram=skipped_diagram,
                skipped_post_ocr=0,
                elapsed_seconds=elapsed,
            )
        
        # Create crops directory
        crops_dir = self.context.crops_dir / page_id
        crops_saved = 0
        skipped_post_ocr = 0
        saved_crops: List[CroppedBox] = []
        
        for box in voter_candidates:
            crop = img_orig[box.y1:box.y2, box.x1:box.x2]
            ocr_img = self._ocr_preprocess(crop)
            
            # Post-OCR edge check
            if self.diagram_filter != "off":
                edges = cv2.Canny(ocr_img, 50, 150)
                edge_frac = float(np.count_nonzero(edges)) / float(edges.size)
                if edge_frac < POST_OCR_MIN_EDGE_FRAC:
                    skipped_post_ocr += 1
                    continue
            
            # Create directory on first save
            if not crops_dir.exists():
                crops_dir.mkdir(parents=True)
            
            # Save crop
            crop_name = f"{page_id}-{box.box_index:03d}.png"
            out_path = crops_dir / crop_name
            
            cv2.imwrite(str(out_path), ocr_img, [cv2.IMWRITE_PNG_COMPRESSION, 0])
            
            box.output_path = out_path
            saved_crops.append(box)
            crops_saved += 1
        
        elapsed = time.perf_counter() - page_start
        
        self.log_debug(
            f"Page {page_id}: saved {crops_saved} crops",
            skipped_diagram=skipped_diagram,
            skipped_post_ocr=skipped_post_ocr,
            time=f"{elapsed:.2f}s"
        )
        
        return PageCropResult(
            page_id=page_id,
            page_path=img_path,
            total_boxes_detected=len(boxes_orig),
            crops_saved=crops_saved,
            skipped_diagram=skipped_diagram,
            skipped_post_ocr=skipped_post_ocr,
            elapsed_seconds=elapsed,
            crops=saved_crops,
        )
    
    def _detect_boxes(self, img_canon: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect voter boxes using grid-line detection.
        
        Args:
            img_canon: Canonical-sized image
        
        Returns:
            List of (x, y, w, h) boxes
        """
        H, W = img_canon.shape[:2]
        page_area = W * H
        
        gray = cv2.cvtColor(img_canon, cv2.COLOR_BGR2GRAY)
        bw = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY_INV,
            31, 12
        )
        
        # Extract horizontal lines
        h_kernel_len = max(10, W // HLINE_SCALE)
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_kernel_len, 1))
        horiz = cv2.erode(bw, h_kernel, iterations=1)
        horiz = cv2.dilate(horiz, h_kernel, iterations=1)
        
        # Extract vertical lines
        v_kernel_len = max(10, H // VLINE_SCALE)
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_kernel_len))
        vert = cv2.erode(bw, v_kernel, iterations=1)
        vert = cv2.dilate(vert, v_kernel, iterations=1)
        
        # Combine and close gaps
        grid = cv2.bitwise_or(horiz, vert)
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        grid = cv2.morphologyEx(grid, cv2.MORPH_CLOSE, k, iterations=2)
        
        # Find contours
        contours, _ = cv2.findContours(grid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        boxes = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            area = w * h
            
            # Filter by area
            if area < page_area * MIN_BOX_AREA_FRAC:
                continue
            if area > page_area * MAX_BOX_AREA_FRAC:
                continue
            
            # Filter by aspect ratio
            ar = w / float(h)
            if not (MIN_AR <= ar <= MAX_AR):
                continue
            
            # Filter by polygon complexity
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            if len(approx) < 4:
                continue
            
            boxes.append((x, y, w, h))
        
        # Post-process
        boxes = self._dedupe_boxes(boxes)
        boxes = self._remove_contained_boxes(boxes)
        boxes = self._sort_reading_order(boxes)
        
        return boxes
    
    def _scale_boxes(
        self,
        boxes: List[Tuple[int, int, int, int]],
        sx: float,
        sy: float
    ) -> List[Tuple[int, int, int, int]]:
        """Scale boxes from canonical to original dimensions."""
        return [
            (int(round(x * sx)), int(round(y * sy)),
             int(round(w * sx)), int(round(h * sy)))
            for x, y, w, h in boxes
        ]
    
    def _dedupe_boxes(
        self,
        boxes: List[Tuple[int, int, int, int]],
        tol: int = 6
    ) -> List[Tuple[int, int, int, int]]:
        """Remove near-duplicate boxes."""
        boxes = sorted(boxes, key=lambda b: (b[0], b[1], b[2], b[3]))
        out = []
        for x, y, w, h in boxes:
            dup = False
            for x2, y2, w2, h2 in out:
                if (abs(x - x2) < tol and abs(y - y2) < tol and
                    abs(w - w2) < tol and abs(h - h2) < tol):
                    dup = True
                    break
            if not dup:
                out.append((x, y, w, h))
        return out
    
    def _remove_contained_boxes(
        self,
        boxes: List[Tuple[int, int, int, int]],
        margin: int = 6,
        max_area_ratio: float = 0.70
    ) -> List[Tuple[int, int, int, int]]:
        """Remove boxes fully contained within another."""
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
                if (x >= X + margin and y >= Y + margin and
                    x2 <= X2 - margin and y2 <= Y2 - margin):
                    big_area = W * H
                    if area / float(max(1, big_area)) <= max_area_ratio:
                        contained = True
                        break
            
            if not contained:
                kept.append((x, y, w, h))
        
        return sorted(kept, key=lambda b: (b[1], b[0]))
    
    def _sort_reading_order(
        self,
        boxes: List[Tuple[int, int, int, int]]
    ) -> List[Tuple[int, int, int, int]]:
        """Sort boxes in reading order (top-to-bottom, left-to-right)."""
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
    
    def _classify_boxes(
        self,
        img_orig: np.ndarray,
        boxes: List[Tuple[int, int, int, int]],
        W: int,
        H: int,
        page_id: str = ""
    ) -> Tuple[List[CroppedBox], int, List[CropMetrics]]:
        """
        Classify boxes as voter/non-voter.
        
        Returns:
            Tuple of (voter candidates, skipped count, rejected metrics)
        """
        voter_candidates = []
        skipped = 0
        rejected_metrics = []
        
        for i, (x, y, w, h) in enumerate(boxes, start=1):
            x1 = max(0, x - PAD)
            y1 = max(0, y - PAD)
            x2 = min(W, x + w + PAD)
            y2 = min(H, y + h + PAD)
            
            if x2 <= x1 or y2 <= y1:
                continue
            
            box = CroppedBox(
                page_id="",  # Set later
                box_index=i,
                x1=x1, y1=y1, x2=x2, y2=y2
            )
            
            if self.diagram_filter == "off":
                voter_candidates.append(box)
                continue
            
            crop = img_orig[y1:y2, x1:x2]
            is_voter, metrics = self._classify_crop_metrics(crop)
            
            if is_voter:
                voter_candidates.append(box)
            else:
                skipped += 1
                rejected_metrics.append(metrics)
                # Debug log rejection reasons
                reasons = getattr(metrics, '_rejection_reasons', [])
                if reasons:
                    self.log_debug(f"{page_id} box {i} rejected: {'; '.join(reasons)}")
        
        return voter_candidates, skipped, rejected_metrics
    
    def _classify_crop_metrics(self, crop_bgr: np.ndarray) -> Tuple[bool, CropMetrics]:
        """
        Classify a crop as voter info or diagram/photo.
        
        Returns:
            Tuple of (is_voter, metrics)
        """
        gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # Edge detection
        edges = cv2.Canny(gray, 50, 150)
        edge_frac = float(np.count_nonzero(edges)) / float(edges.size)
        
        # Binarize
        bw = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY_INV,
            31, 12
        )
        
        ink_px = int(np.count_nonzero(bw))
        ink_frac = float(ink_px) / float(bw.size)
        
        # Extract lines
        lines = self._extract_hv_lines(bw)
        line_px = int(np.count_nonzero(lines))
        line_ratio = float(line_px) / float(max(1, ink_px))
        
        # Remove lines and count small components
        bw_wo_lines = cv2.bitwise_and(bw, cv2.bitwise_not(lines))
        
        k2 = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        bw_wo_lines = cv2.morphologyEx(bw_wo_lines, cv2.MORPH_OPEN, k2, iterations=1)
        
        nlabels, _, stats, _ = cv2.connectedComponentsWithStats(bw_wo_lines, connectivity=8)
        
        area_total = bw_wo_lines.shape[0] * bw_wo_lines.shape[1]
        max_small_area = max(80, int(area_total * 0.002))
        min_small_area = 10
        small_components = 0
        largest_cc_area = 0
        
        for i in range(1, nlabels):
            a = int(stats[i, cv2.CC_STAT_AREA])
            if a > largest_cc_area:
                largest_cc_area = a
            if min_small_area <= a <= max_small_area:
                small_components += 1
        
        ink_wo_lines_px = int(np.count_nonzero(bw_wo_lines))
        largest_cc_ratio = float(largest_cc_area) / float(max(1, ink_wo_lines_px))
        
        metrics = CropMetrics(
            ink_frac=ink_frac,
            line_ratio=line_ratio,
            small_components=small_components,
            largest_cc_ratio=largest_cc_ratio,
            edge_frac=edge_frac,
        )
        
        # Classification logic - track rejection reasons
        is_voter = True
        rejection_reasons = []
        
        if not (VOTER_INK_FRAC_MIN <= ink_frac <= VOTER_INK_FRAC_MAX):
            is_voter = False
            rejection_reasons.append(f"ink_frac={ink_frac:.4f} not in [{VOTER_INK_FRAC_MIN}, {VOTER_INK_FRAC_MAX}]")
        if line_ratio > VOTER_MAX_LINE_RATIO:
            is_voter = False
            rejection_reasons.append(f"line_ratio={line_ratio:.4f} > {VOTER_MAX_LINE_RATIO}")
        if small_components < VOTER_MIN_SMALL_COMPONENTS:
            is_voter = False
            rejection_reasons.append(f"small_components={small_components} < {VOTER_MIN_SMALL_COMPONENTS}")
        if largest_cc_ratio > VOTER_MAX_LARGEST_CC_RATIO:
            is_voter = False
            rejection_reasons.append(f"largest_cc_ratio={largest_cc_ratio:.4f} > {VOTER_MAX_LARGEST_CC_RATIO}")
        if edge_frac < VOTER_MIN_EDGE_FRAC:
            is_voter = False
            rejection_reasons.append(f"edge_frac={edge_frac:.4f} < {VOTER_MIN_EDGE_FRAC}")
        
        # Store rejection reasons for debugging
        if rejection_reasons:
            metrics._rejection_reasons = rejection_reasons
        
        return is_voter, metrics
    
    def _extract_hv_lines(self, bw_inv: np.ndarray) -> np.ndarray:
        """Extract horizontal/vertical lines from binary-inverted image."""
        h, w = bw_inv.shape[:2]
        
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
    
    def _is_confidently_non_voter(self, rejected_metrics: List[CropMetrics]) -> bool:
        """Check if rejected boxes are confidently non-voter."""
        if not rejected_metrics:
            return False
        
        avg_small = float(np.mean([m.small_components for m in rejected_metrics]))
        avg_line_ratio = float(np.mean([m.line_ratio for m in rejected_metrics]))
        avg_largest_cc = float(np.mean([m.largest_cc_ratio for m in rejected_metrics]))
        avg_edge = float(np.mean([m.edge_frac for m in rejected_metrics]))
        
        return (
            avg_small <= AUTO_SKIP_MAX_SMALL_COMPONENTS or
            avg_line_ratio >= AUTO_SKIP_MIN_LINE_RATIO or
            avg_largest_cc >= AUTO_SKIP_MIN_LARGEST_CC_RATIO or
            avg_edge <= AUTO_SKIP_MAX_EDGE_FRAC
        )
    
    def _all_boxes_as_candidates(
        self,
        boxes: List[Tuple[int, int, int, int]],
        W: int,
        H: int
    ) -> List[CroppedBox]:
        """Convert all boxes to candidates (for auto mode fail-open)."""
        candidates = []
        for i, (x, y, w, h) in enumerate(boxes, start=1):
            x1 = max(0, x - PAD)
            y1 = max(0, y - PAD)
            x2 = min(W, x + w + PAD)
            y2 = min(H, y + h + PAD)
            
            if x2 <= x1 or y2 <= y1:
                continue
            
            candidates.append(CroppedBox(
                page_id="",
                box_index=i,
                x1=x1, y1=y1, x2=x2, y2=y2
            ))
        
        return candidates
    
    def _ocr_preprocess(self, crop_bgr: np.ndarray) -> np.ndarray:
        """
        Preprocess crop for OCR.
        
        Applies:
        - Grayscale conversion
        - Deskewing
        - Upscaling (2x)
        - Denoising
        - Contrast normalization
        """
        # Grayscale
        gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
        
        # Deskew
        angle = self._estimate_skew(gray)
        if abs(angle) > 0.2:
            gray = self._rotate_image(gray, angle)
        
        # Upscale
        gray = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
        
        # Denoise
        gray = cv2.fastNlMeansDenoising(
            gray, None,
            h=8,
            templateWindowSize=7,
            searchWindowSize=21
        )
        
        # Contrast normalization
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
        
        return gray
    
    def _estimate_skew(self, gray: np.ndarray) -> float:
        """Estimate skew angle using Hough lines."""
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(
            edges, 1, np.pi / 180,
            threshold=80,
            minLineLength=max(50, gray.shape[1] // 3),
            maxLineGap=10
        )
        
        if lines is None:
            return 0.0
        
        angles = []
        for x1, y1, x2, y2 in lines[:, 0]:
            dx, dy = x2 - x1, y2 - y1
            if dx == 0:
                continue
            ang = np.degrees(np.arctan2(dy, dx))
            if -30 <= ang <= 30:
                angles.append(ang)
        
        if not angles:
            return 0.0
        
        return float(np.median(angles))
    
    def _rotate_image(self, img: np.ndarray, angle_deg: float) -> np.ndarray:
        """Rotate image by angle."""
        if abs(angle_deg) < 0.2:
            return img
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle_deg, 1.0)
        return cv2.warpAffine(
            img, M, (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE
        )


def crop_document(
    extracted_dir: Path,
    diagram_filter: str = "auto",
) -> CropSummary:
    """
    Convenience function to crop all pages in an extracted folder.
    
    Args:
        extracted_dir: Path to extracted folder
        diagram_filter: "auto", "on", or "off"
    
    Returns:
        Crop summary
    """
    from ..config import Config
    
    config = Config()
    context = ProcessingContext(config=config)
    context.setup_paths_from_extracted(extracted_dir)
    
    cropper = ImageCropper(context, diagram_filter=diagram_filter)
    
    if not cropper.run():
        raise CropExtractionError("Cropping failed", str(extracted_dir))
    
    return cropper.summary
