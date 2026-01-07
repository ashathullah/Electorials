"""
AI ID Processor for Horizontal Strips.

Extracts Serial Numbers and House Numbers from horizontally stitched ID strip images.
Each strip contains multiple pages stitched side by side.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional

from .base import BaseProcessor, ProcessingContext
from ..models import AIUsage
from ..logger import get_logger

logger = get_logger("ai_id_horizontal_processor")


@dataclass
class IdExtractionResult:
    """Result from AI ID extraction."""
    serial_no: str
    house_no: str
    page_id: str = ""  # Track which page this belongs to


@dataclass
class BatchResult:
    """Results for a batch of strips."""
    batch_num: int
    page_results: Dict[str, List[IdExtractionResult]] = field(default_factory=dict)


class AIIdHorizontalProcessor(BaseProcessor):
    """
    Extracts ID data from horizontally merged strips using AI.
    
    Input: id_merged_horizontal/batch-XXX/strip-XXX.png
    Each strip contains rows of ID fields stitched horizontally across pages.
    """
    
    name = "AIIdHorizontalProcessor"
    
    def __init__(self, context: ProcessingContext):
        super().__init__(context)
        self.client = None
        self.model = self.config.ai.model
        self.batch_size = self.config.ai.id_batch_size  # Strips per API call (default: 5)
        self.log_info(f"Initialized with ID Batch Size: {self.batch_size}")
        
        # Results storage
        self.page_results: Dict[str, List[IdExtractionResult]] = {}
        
        # Input directory
        self.input_dir = self.context.extracted_dir / "id_merged_horizontal" if self.context.extracted_dir else None

    def validate(self) -> bool:
        """Validate prerequisites."""
        if not self.config.ai.api_key:
            self.log_error("AI API Key not set")
            return False
            
        if not self.input_dir:
            self.log_error("Input directory not set")
            return False
            
        if not self.input_dir.exists():
            self.log_error(f"Horizontal merged directory not found: {self.input_dir}. "
                          "Run 'id-horizontal-merge' step first.")
            return False
            
        return True
        
    def _init_client(self):
        """Initialize AI client (using Groq/OpenAI compatible interface)."""
        if self.client:
            return
            
        try:
            from openai import OpenAI
            
            base_url = self.config.ai.get_normalized_base_url()
            self.client = OpenAI(
                api_key=self.config.ai.api_key,
                base_url=base_url if base_url else None
            )
            self.log_info(f"Initialized AI client with model {self.model}")
        except ImportError:
            self.log_error("openai package not installed. Please install it: pip install openai")
            raise

    def process(self) -> bool:
        """Process all horizontal ID batches."""
        self._init_client()
        
        # Find all batch directories
        batch_dirs = sorted([
            d for d in self.input_dir.iterdir()
            if d.is_dir() and d.name.startswith("batch-")
        ])
        
        if not batch_dirs:
            self.log_warning(f"No batch directories found in {self.input_dir}")
            return False
            
        self.log_info(f"Found {len(batch_dirs)} batches to process")
        
        total_processed = 0
        
        for batch_dir in batch_dirs:
            batch_num = int(batch_dir.name.split("-")[1])
            
            # Load batch metadata
            metadata = self._load_batch_metadata(batch_dir)
            if not metadata:
                self.log_warning(f"No metadata found for {batch_dir.name}")
                continue
            
            # Get strips for this batch
            strips = sorted([
                p for p in batch_dir.iterdir()
                if p.is_file() and p.name.startswith("strip-") and p.suffix.lower() == ".png"
            ])
            
            if not strips:
                continue
            
            self.log_info(f"Processing {batch_dir.name} with {len(strips)} strips...")
            
            # Process this batch (all strips in one API call)
            results = self._process_strips_batch(strips, metadata)
            
            # Merge results into page_results
            for page_id, page_results in results.items():
                if page_id not in self.page_results:
                    self.page_results[page_id] = []
                self.page_results[page_id].extend(page_results)
            
            total_processed += len(strips)
        
        # Save debug info for each page
        for page_id, results in self.page_results.items():
            self.save_debug_info(f"id_horizontal_extract_{page_id}", [
                {"serial_no": r.serial_no, "house_no": r.house_no}
                for r in results
            ])
            self.log_info(f"Extracted {len(results)} voter records for {page_id}")
        
        self.log_info(f"Total: Processed {total_processed} strips, "
                     f"extracted data for {len(self.page_results)} pages")
            
        return True
    
    def _load_batch_metadata(self, batch_dir: Path) -> Optional[Dict]:
        """Load batch metadata JSON."""
        metadata_path = batch_dir / "metadata.json"
        if not metadata_path.exists():
            return None
            
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            self.log_error(f"Failed to load metadata from {metadata_path}: {e}")
            return None

    def _process_strips_batch(
        self, 
        strip_paths: List[Path], 
        metadata: Dict
    ) -> Dict[str, List[IdExtractionResult]]:
        """
        Send a batch of horizontal strip images to AI and parse response.
        
        Args:
            strip_paths: Paths to strip images
            metadata: Batch metadata containing page mappings
            
        Returns:
            Dictionary mapping page_id to list of extraction results
        """
        import base64
        
        # Build page mapping info from metadata
        strip_page_mappings = {}  # strip_num -> [page_ids]
        for strip_info in metadata.get("strips", []):
            strip_num = strip_info["strip_num"]
            pages = strip_info["pages"]
            strip_page_mappings[strip_num] = pages
        
        # Build description for AI
        strip_desc_lines = []
        for strip_num, pages in strip_page_mappings.items():
            pages_str = ", ".join(pages)
            strip_desc_lines.append(f"Strip {strip_num}: Pages {pages_str} (left to right)")
        strip_description = "\n".join(strip_desc_lines)
        
        num_pages_per_strip = len(next(iter(strip_page_mappings.values()), []))
        
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text", 
                        "text": f"""
Extract serial numbers and house numbers from electoral roll ID strip images.

Each image shows multiple rows. Each row contains {num_pages_per_strip} voter ID strips stitched HORIZONTALLY (left to right).
Each strip shows: [Serial Number on left] [House Number on right]
Pages are separated by VERTICAL STRIPED DIVIDERS (black and white stripes).

Image mapping:
{strip_description}

IMPORTANT: 
- Each ROW contains {num_pages_per_strip} voters from different pages (stitched horizontally)
- Pages are clearly separated by vertical striped dividers
- Read LEFT to RIGHT across each row to get voters from different pages
- Then move DOWN to the next row
- Serial numbers within each page should be sequential

Return data in TOML format. DO NOT use markdown code blocks. Return raw TOML only.

Format for each page:
[page-XXX]
voter_records = [
  ["serial1", "house1"],
  ["serial2", "house2"]
]

Example (with 3 pages per strip, 2 rows):
[page-001]
voter_records = [
  ["1", "12A"],
  ["2", "15B"]
]

[page-002]
voter_records = [
  ["1", "34"],
  ["2", "56-C"]
]

[page-003]
voter_records = [
  ["1", "78"],
  ["2", "90"]
]

Use "" for empty/illegible fields. Maintain exact sequence.
                        """
                    }
                ]
            }
        ]
        
        # Append images
        for strip_path in strip_paths:
            try:
                with open(strip_path, "rb") as f:
                    base64_image = base64.b64encode(f.read()).decode('utf-8')
                    
                messages[0]["content"].append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{base64_image}"
                    }
                })
            except Exception as e:
                self.log_error(f"Failed to read image {strip_path}: {e}")
                
        max_retries = self.config.ai.max_retries
        retry_delay = self.config.ai.retry_delay_sec
        
        for attempt in range(max_retries + 1):
            try:
                start_time = time.perf_counter()
                
                completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0,
                    max_completion_tokens=16384  # Increased for horizontal format (5 strips × 30 rows × 5 pages = up to 750 records)
                )
                
                elapsed = time.perf_counter() - start_time
                
                # Track usage
                if hasattr(completion, 'usage') and completion.usage:
                    input_tokens = completion.usage.prompt_tokens
                    output_tokens = completion.usage.completion_tokens
                    cost = self.config.ai.estimate_cost(input_tokens, output_tokens)
                    self.context.ai_usage.add_call(input_tokens, output_tokens, cost)
                
                content = completion.choices[0].message.content
                
                # Save raw response for debugging
                raw_response_debug = {
                    "raw_response": content,
                    "strip_count": len(strip_paths),
                    "metadata": metadata,
                    "elapsed": elapsed
                }
                debug_filename = f"ai_horizontal_raw_{metadata.get('batch_num', 0)}"
                self.save_debug_info(debug_filename, raw_response_debug)
                
                # Strip markdown code blocks if present
                content = content.strip()
                if content.startswith("```"):
                    lines = content.split('\n')
                    if lines[0].startswith("```"):
                        lines = lines[1:]
                    if lines and lines[-1].strip() == "```":
                        lines = lines[:-1]
                    content = '\n'.join(lines).strip()
                
                # Parse TOML response
                try:
                    import toml
                    data = toml.loads(content)
                    
                    results_dict: Dict[str, List[IdExtractionResult]] = {}
                    
                    for page_id, page_data in data.items():
                        if not isinstance(page_data, dict):
                            continue
                        
                        voter_records = page_data.get("voter_records", [])
                        if not isinstance(voter_records, list):
                            continue
                        
                        page_results = []
                        for record in voter_records:
                            if isinstance(record, list) and len(record) >= 2:
                                serial_no = str(record[0]).strip()
                                house_no = str(record[1]).strip()
                                page_results.append(IdExtractionResult(
                                    serial_no=serial_no,
                                    house_no=house_no,
                                    page_id=page_id
                                ))
                            elif isinstance(record, list) and len(record) == 1:
                                page_results.append(IdExtractionResult(
                                    serial_no="",
                                    house_no=str(record[0]).strip(),
                                    page_id=page_id
                                ))
                        
                        results_dict[page_id] = page_results
                    
                    if results_dict:
                        total_extracted = sum(len(v) for v in results_dict.values())
                        self.log_info(f"Extracted {total_extracted} voter records "
                                     f"from {len(strip_paths)} strips "
                                     f"across {len(results_dict)} pages in {elapsed:.2f}s")
                        return results_dict
                    else:
                        self.log_error("No valid page data found in TOML response")
                        return {}
                    
                except Exception as e:
                    self.log_error(f"Failed to parse AI response as TOML: {e}")
                    self.log_debug(f"Response content (first 500 chars): {content[:500]}...")
                    return {}
                    
            except Exception as e:
                is_last_attempt = attempt == max_retries
                error_msg = f"AI API call failed (attempt {attempt+1}/{max_retries+1}): {e}"
                
                if is_last_attempt:
                    self.log_error(f"AI API call completely failed after {max_retries+1} attempts: {e}")
                    return {}
                else:
                    self.log_warning(f"{error_msg}. Retrying in {retry_delay}s...")
                    time.sleep(retry_delay)
                    retry_delay *= 2

    def get_results_for_page(self, page_id: str) -> List[IdExtractionResult]:
        """Get extraction results for a specific page."""
        return self.page_results.get(page_id, [])
    
    def get_all_results(self) -> Dict[str, List[IdExtractionResult]]:
        """Get all extraction results organized by page."""
        return self.page_results
