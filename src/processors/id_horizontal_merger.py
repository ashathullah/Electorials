"""
ID Horizontal Merger.

Stitches ID field crops from multiple pages horizontally for batch AI processing.
This allows AI to process multiple pages' ID fields in a single image.

Output structure:
    id_merged_horizontal/
        batch-001/      # Each batch contains 5 horizontal strips
            strip-001.png   # 5 pages stitched horizontally (1 row per voter)
            strip-002.png   # Next 5 pages stitched horizontally
            ...
        batch-002/
            ...

Each strip contains: [Page1_Voter1] [Sep] [Page2_Voter1] [Sep] [Page3_Voter1] [Sep] [Page4_Voter1] [Sep] [Page5_Voter1]
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Tuple

import cv2
import numpy as np

from .base import BaseProcessor, ProcessingContext
from ..logger import get_logger

logger = get_logger("id_horizontal_merger")

# Constants
HORIZONTAL_STITCH_SPACING = 30  # Pixels between horizontally stitched images (increased for visibility)
STITCH_BG_COLOR = 255  # White background
VERTICAL_SPACING = 5  # Pixels between vertically stacked rows

def _create_divider(height: int, width: int = HORIZONTAL_STITCH_SPACING) -> np.ndarray:
    """
    Create a visible divider pattern to separate pages.
    
    Creates a vertical striped pattern (black and white alternating stripes)
    to clearly mark page boundaries.
    """
    divider = np.zeros((height, width), dtype=np.uint8)
    
    # Create vertical stripes pattern (2 pixels black, 2 pixels white)
    stripe_width = 3
    for x in range(0, width, stripe_width * 2):
        # White stripe
        end_x = min(x + stripe_width, width)
        divider[:, x:end_x] = 255
        # Black stripe is already 0 from initialization
    
    # Add thin black borders on both edges for emphasis
    divider[:, 0:2] = 0  # Left border
    divider[:, -2:] = 0  # Right border
    
    return divider



@dataclass
class HorizontalStitchConfig:
    """Configuration for horizontal stitching."""
    pages_per_stitch: int = 5  # Number of pages to stitch horizontally
    strips_per_batch: int = 5  # Number of horizontal strips per batch (for AI request)
    max_rows_per_strip: int = 30  # Maximum voter rows in a single strip image


@dataclass
class HorizontalMergeSummary:
    """Summary of horizontal merge operation."""
    total_pages: int = 0
    total_strips_created: int = 0
    total_batches: int = 0
    elapsed_seconds: float = 0.0


class IdHorizontalMerger(BaseProcessor):
    """
    Merges ID field crops horizontally across pages.
    
    Flow:
    1. Collects ID crops from multiple pages (default: 5 pages)
    2. Stitches voter rows horizontally: [P1V1] [sep] [P2V1] [sep] [P3V1] [sep] [P4V1] [sep] [P5V1]
    3. Stacks rows vertically into a single strip image
    4. Groups strips into batches for AI processing
    """
    
    name = "IdHorizontalMerger"
    
    def __init__(self, context: ProcessingContext, config: Optional[HorizontalStitchConfig] = None):
        super().__init__(context)
        self.stitch_config = config or HorizontalStitchConfig()
        self.summary: Optional[HorizontalMergeSummary] = None
        
        # Input: individual ID crops
        self.id_crops_dir = self.context.id_crops_dir
        
        # Output: horizontally merged batches
        self.output_dir = self.context.extracted_dir / "id_merged_horizontal" if self.context.extracted_dir else None
        
    def validate(self) -> bool:
        """Validate prerequisites."""
        if not self.id_crops_dir:
            self.log_error("ID crops directory not set")
            return False
            
        if not self.id_crops_dir.exists():
            self.log_error(f"ID crops directory not found: {self.id_crops_dir}")
            return False
            
        if not self.output_dir:
            self.log_error("Output directory not set (extracted_dir not set)")
            return False
            
        return True
    
    def process(self) -> bool:
        """Process all ID crops and create horizontal merges."""
        start_time = time.perf_counter()
        
        # Get all page directories sorted
        page_dirs = sorted([
            d for d in self.id_crops_dir.iterdir()
            if d.is_dir() and d.name.startswith("page-")
        ])
        
        if not page_dirs:
            self.log_warning(f"No page directories found in {self.id_crops_dir}")
            return False
        
        self.log_info(f"Found {len(page_dirs)} pages with ID crops")
        self.log_info(f"Config: {self.stitch_config.pages_per_stitch} pages/stitch, "
                     f"{self.stitch_config.strips_per_batch} strips/batch")
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        total_strips = 0
        total_batches = 0
        
        # Process pages in groups
        pages_per_stitch = self.stitch_config.pages_per_stitch
        strips_per_batch = self.stitch_config.strips_per_batch
        
        # Collect all page data: page_id -> list of image paths (sorted)
        page_images: Dict[str, List[Path]] = {}
        for page_dir in page_dirs:
            images = sorted([
                p for p in page_dir.iterdir()
                if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg"}
            ])
            if images:
                page_images[page_dir.name] = images
        
        if not page_images:
            self.log_warning("No ID crop images found in any page directory")
            return False
        
        # Group pages for horizontal stitching
        page_ids = sorted(page_images.keys())
        
        # Create strips by stitching pages horizontally
        all_strips: List[Tuple[np.ndarray, List[str]]] = []  # (image, [page_ids])
        
        for i in range(0, len(page_ids), pages_per_stitch):
            chunk_page_ids = page_ids[i:i + pages_per_stitch]
            
            # Get images for this chunk of pages
            chunk_images = [page_images[pid] for pid in chunk_page_ids]
            
            # Find the maximum number of voters across these pages
            max_voters = max(len(imgs) for imgs in chunk_images)
            
            # Create horizontal strips for each voter index
            strips_for_chunk = self._create_horizontal_strips(
                chunk_page_ids, chunk_images, max_voters
            )
            
            for strip_img in strips_for_chunk:
                all_strips.append((strip_img, chunk_page_ids))
            
            self.log_debug(f"Created {len(strips_for_chunk)} strips for pages: {chunk_page_ids}")
        
        # Now group strips into batches
        for batch_idx in range(0, len(all_strips), strips_per_batch):
            batch_strips = all_strips[batch_idx:batch_idx + strips_per_batch]
            batch_num = (batch_idx // strips_per_batch) + 1
            
            batch_dir = self.output_dir / f"batch-{batch_num:03d}"
            batch_dir.mkdir(parents=True, exist_ok=True)
            
            # Save each strip in the batch
            for strip_idx, (strip_img, strip_page_ids) in enumerate(batch_strips, start=1):
                strip_path = batch_dir / f"strip-{strip_idx:03d}.png"
                
                success, encoded = cv2.imencode(".png", strip_img)
                if success:
                    encoded.tofile(str(strip_path))
                    total_strips += 1
                    self.log_debug(f"Saved {strip_path.name} for pages: {strip_page_ids}")
                else:
                    self.log_error(f"Failed to encode strip {strip_path}")
            
            # Save metadata for this batch
            metadata = {
                "batch_num": batch_num,
                "strips": [
                    {"strip_num": i + 1, "pages": strip_page_ids}
                    for i, (_, strip_page_ids) in enumerate(batch_strips)
                ]
            }
            self._save_batch_metadata(batch_dir, metadata)
            
            total_batches += 1
            self.log_info(f"Created batch {batch_num} with {len(batch_strips)} strips")
        
        elapsed = time.perf_counter() - start_time
        
        self.summary = HorizontalMergeSummary(
            total_pages=len(page_dirs),
            total_strips_created=total_strips,
            total_batches=total_batches,
            elapsed_seconds=elapsed,
        )
        
        self.log_info(
            f"Horizontal merge complete: {total_strips} strips in {total_batches} batches "
            f"from {len(page_dirs)} pages in {elapsed:.2f}s"
        )
        
        return True
    
    def _create_horizontal_strips(
        self,
        page_ids: List[str],
        page_images: List[List[Path]],
        max_voters: int
    ) -> List[np.ndarray]:
        """
        Create horizontal strips by stitching images across pages.
        
        Args:
            page_ids: List of page IDs being processed
            page_images: List of image path lists (one per page)
            max_voters: Maximum number of voters in any page
            
        Returns:
            List of strip images (each strip contains multiple rows,
            up to max_rows_per_strip)
        """
        max_rows = self.stitch_config.max_rows_per_strip
        strips = []
        
        # Process voters in chunks of max_rows_per_strip
        for voter_start in range(0, max_voters, max_rows):
            voter_end = min(voter_start + max_rows, max_voters)
            
            rows = []
            for voter_idx in range(voter_start, voter_end):
                row = self._create_horizontal_row(page_ids, page_images, voter_idx)
                if row is not None:
                    rows.append(row)
            
            if rows:
                # Stack rows vertically with spacing
                strip = self._stack_rows_vertically(rows)
                strips.append(strip)
        
        return strips
    
    def _create_horizontal_row(
        self,
        page_ids: List[str],
        page_images: List[List[Path]],
        voter_idx: int
    ) -> Optional[np.ndarray]:
        """
        Create a single horizontal row by stitching images at voter_idx from each page.
        
        Returns:
            Horizontally stitched image or None if all pages are empty at this index
        """
        images_to_stitch = []
        max_height = 0
        
        for page_idx, page_imgs in enumerate(page_images):
            if voter_idx < len(page_imgs):
                img = cv2.imdecode(
                    np.fromfile(str(page_imgs[voter_idx]), dtype=np.uint8),
                    cv2.IMREAD_GRAYSCALE
                )
                if img is not None:
                    images_to_stitch.append(img)
                    if img.shape[0] > max_height:
                        max_height = img.shape[0]
                else:
                    # Failed to load - use placeholder
                    images_to_stitch.append(None)
            else:
                # This page doesn't have a voter at this index
                images_to_stitch.append(None)
        
        # Check if all images are None
        if all(img is None for img in images_to_stitch):
            return None
        
        # Normalize heights and stitch horizontally
        stitched_parts = []
        
        for page_idx, img in enumerate(images_to_stitch):
            if img is not None:
                # Pad vertically to match max_height
                if img.shape[0] < max_height:
                    top = (max_height - img.shape[0]) // 2
                    bottom = max_height - img.shape[0] - top
                    img = np.vstack([
                        np.full((top, img.shape[1]), STITCH_BG_COLOR, dtype=np.uint8),
                        img,
                        np.full((bottom, img.shape[1]), STITCH_BG_COLOR, dtype=np.uint8)
                    ])
                stitched_parts.append(img)
            else:
                # Create placeholder for missing voter
                # Use width of first available image or default
                placeholder_width = 100
                for other_img in images_to_stitch:
                    if other_img is not None:
                        placeholder_width = other_img.shape[1]
                        break
                placeholder = np.full(
                    (max_height, placeholder_width),
                    STITCH_BG_COLOR,
                    dtype=np.uint8
                )
                stitched_parts.append(placeholder)
            
            # Add horizontal separator between pages (not after last)
            if page_idx < len(images_to_stitch) - 1:
                separator = _create_divider(max_height)
                stitched_parts.append(separator)
        
        return np.hstack(stitched_parts)
    
    def _stack_rows_vertically(self, rows: List[np.ndarray]) -> np.ndarray:
        """Stack horizontal rows vertically with spacing between them."""
        if not rows:
            return np.array([])
        
        # Find max width
        max_width = max(row.shape[1] for row in rows)
        
        # Pad rows to max width and add vertical spacing
        stacked_parts = []
        for i, row in enumerate(rows):
            # Pad width if needed
            if row.shape[1] < max_width:
                padding = np.full(
                    (row.shape[0], max_width - row.shape[1]),
                    STITCH_BG_COLOR,
                    dtype=np.uint8
                )
                row = np.hstack([row, padding])
            
            stacked_parts.append(row)
            
            # Add vertical separator between rows (not after last)
            if i < len(rows) - 1:
                separator = np.full(
                    (VERTICAL_SPACING, max_width),
                    STITCH_BG_COLOR,
                    dtype=np.uint8
                )
                stacked_parts.append(separator)
        
        return np.vstack(stacked_parts)
    
    def _save_batch_metadata(self, batch_dir: Path, metadata: dict):
        """Save batch metadata as JSON."""
        import json
        
        metadata_path = batch_dir / "metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
