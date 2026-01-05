"""
AI ID Processor.

Extracts Serial, EPIC, and House Number from stitched ID crop strips using AI Vision.
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

logger = get_logger("ai_id_processor")


@dataclass
class IdExtractionResult:
    """Result from AI ID extraction."""
    serial_no: str
    epic_no: str
    house_no: str


@dataclass
class PageIdResult:
    """ID extraction results for a page."""
    page_id: str
    results: List[IdExtractionResult] = field(default_factory=list)


class AIIdProcessor(BaseProcessor):
    """
    Extracts ID data from merged ID strips using AI.
    """
    
    name = "AIIdProcessor"
    
    def __init__(self, context: ProcessingContext):
        super().__init__(context)
        self.client = None
        self.model = self.config.ai.model
        self.batch_size = self.config.ai.id_batch_size
        self.log_info(f"Initialized with ID Batch Size: {self.batch_size}")
        
        # We need to map extracted data back to voters
        self.page_results: Dict[str, List[IdExtractionResult]] = {}

    def validate(self) -> bool:
        """Validate prerequisites."""
        if not self.config.ai.api_key:
            self.log_error("AI API Key not set")
            return False
            
        if not self.context.id_merged_dir:
            self.log_error("ID merged directory not set")
            return False
            
        if not self.context.id_merged_dir.exists():
            self.log_error(f"ID merged directory not found: {self.context.id_merged_dir}")
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
        """Process all ID merged batches."""
        self._init_client()
        
        id_merged_dir = self.context.id_merged_dir
        merged_pages = sorted([p for p in id_merged_dir.iterdir() if p.is_dir()])
        
        if not merged_pages:
            self.log_warning(f"No pages found in {id_merged_dir}")
            return False
            
        total_pages = len(merged_pages)
        self.log_info(f"Processing ID extraction for {total_pages} pages using AI...")
        
        for i, page_dir in enumerate(merged_pages, 1):
            self.log_info(f"Processing page {i}/{total_pages}: {page_dir.name}")
            self._process_page(page_dir)
            
        return True

    def _process_page(self, page_dir: Path):
        """Process a single page of merged batches."""
        page_id = page_dir.name
        
        # Get all batch images
        batch_images = sorted([
            p for p in page_dir.iterdir() 
            if p.is_file() and p.name.startswith("batch-") and p.suffix.lower() == ".png"
        ])
        
        if not batch_images:
            return
            
        all_results = []
        
        # Process in chunks (batched API requests)
        # However, the user said "send multiple images in a single request"
        # So we group the batch_images themselves
        
        for i in range(0, len(batch_images), self.batch_size):
            chunk = batch_images[i:i + self.batch_size]
            results = self._process_image_batch(chunk)
            all_results.extend(results)
            
        self.page_results[page_id] = all_results
        
        # Save intermediate results to debug/recover
        self.save_debug_info(f"id_extract_{page_id}", [
            {"serial": r.serial_no, "epic": r.epic_no, "house": r.house_no}
            for r in all_results
        ])

    def _process_image_batch(self, image_paths: List[Path]) -> List[IdExtractionResult]:
        """Send a batch of images to AI and parse response."""
        import base64
        
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text", 
                        "text": """
You are an expert OCR system for Indian Electoral Rolls.
You are provided with images containing rows of cropped ID fields.
Each row consists of 3 fields separated by vertical black lines:
1. Serial Number (Left) - usually a simple number
2. EPIC Number (Middle) - Alphanumeric ID (e.g., TN/23/123/456789 or ABC1234567)
3. House Number (Right) - Can be alphanumeric (e.g., 12, 12/A, 4-23). Can be empty.

The images are batches containing multiple rows (voters).
For EACH row in EACH image provided, extract these 3 values.

Return ONLY a valid JSON object with the following structure:
{
  "voters": [
    {
       "serial_no": "...",
       "epic_no": "...",
       "house_no": "..."
    },
    ...
  ]
}

- Maintain the EXACT order of rows as they appear in the images (Top to Bottom).
- If multiple images are provided, process them in order (Image 1 top-to-bottom, then Image 2...).
- If a field is illegible or empty, use an empty string "".
- Do not include markdown formatting (```json). Just the raw JSON.
                        """
                    }
                ]
            }
        ]
        
        # Append images
        for img_path in image_paths:
            try:
                with open(img_path, "rb") as f:
                    base64_image = base64.b64encode(f.read()).decode('utf-8')
                    
                messages[0]["content"].append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{base64_image}"
                    }
                })
            except Exception as e:
                self.log_error(f"Failed to read image {img_path}: {e}")
                
        max_retries = self.config.ai.max_retries
        retry_delay = self.config.ai.retry_delay_sec
        
        for attempt in range(max_retries + 1):
            try:
                start_time = time.perf_counter()
                
                completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0,
                    max_completion_tokens=4096,
                    response_format={"type": "json_object"}
                )
                
                elapsed = time.perf_counter() - start_time
                
                # Track usage
                if hasattr(completion, 'usage') and completion.usage:
                    self.context.ai_usage.add_call(
                        input_tokens=completion.usage.prompt_tokens,
                        output_tokens=completion.usage.completion_tokens,
                        cost_usd=self.config.ai.estimate_cost(
                            completion.usage.prompt_tokens,
                            completion.usage.completion_tokens
                        )
                    )
                
                content = completion.choices[0].message.content
                
                # Parse JSON
                try:
                    data = json.loads(content)
                    voters_data = data.get("voters", [])
                    
                    results = []
                    for v in voters_data:
                        results.append(IdExtractionResult(
                            serial_no=str(v.get("serial_no", "")).strip(),
                            epic_no=str(v.get("epic_no", "")).strip(),
                            house_no=str(v.get("house_no", "")).strip()
                        ))
                    
                    self.log_info(f"Extracted {len(results)} records from {len(image_paths)} images in {elapsed:.2f}s")
                    return results
                    
                except json.JSONDecodeError:
                    self.log_error(f"Failed to parse AI response as JSON: {content[:100]}...")
                    # Don't retry on parsing errors as the model output is likely deterministic or the issue is with the response handling
                    return []
                    
            except Exception as e:
                is_last_attempt = attempt == max_retries
                error_msg = f"AI API call failed (attempt {attempt+1}/{max_retries+1}): {e}"
                
                if is_last_attempt:
                    self.log_error(f"AI API call completely failed after {max_retries+1} attempts: {e}")
                    return []
                else:
                    self.log_warning(f"{error_msg}. Retrying in {retry_delay}s...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff

    def get_results_for_page(self, page_id: str) -> List[IdExtractionResult]:
        return self.page_results.get(page_id, [])
