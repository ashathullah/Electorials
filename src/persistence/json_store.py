"""
JSON file-based storage implementation.

Provides persistent storage for processed documents using JSON files.
Designed to be easily migratable to a database in the future.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, List, Any, Union
from datetime import datetime

from ..models import ProcessedDocument, Voter, DocumentMetadata, ProcessingStats


class JSONStore:
    """
    JSON file-based document storage.
    
    Stores processed documents as JSON files with the following structure:
    - <base_dir>/<pdf_name>/output/<pdf_name>.json (combined output)
    - <base_dir>/<pdf_name>/output/<pdf_name>-metadata.json (AI metadata)
    - <base_dir>/<pdf_name>/output/page_wise/<page>.json (per-page data)
    """
    
    def __init__(self, base_dir: Path):
        """
        Initialize JSON store.
        
        Args:
            base_dir: Base directory for extracted files
        """
        self.base_dir = Path(base_dir)
    
    def _get_output_dir(self, pdf_name: str) -> Path:
        """Get output directory for a PDF."""
        output_dir = self.base_dir / pdf_name / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir
    
    def _get_page_wise_dir(self, pdf_name: str) -> Path:
        """Get page-wise output directory."""
        page_dir = self._get_output_dir(pdf_name) / "page_wise"
        page_dir.mkdir(parents=True, exist_ok=True)
        return page_dir
    
    def save_document(self, document: ProcessedDocument) -> Path:
        """
        Save complete processed document.
        
        Creates unified JSON with all document data.
        
        Args:
            document: Processed document to save
        
        Returns:
            Path to saved file
        """
        output_dir = self._get_output_dir(document.pdf_name)
        output_path = output_dir / f"{document.pdf_name}.json"
        
        data = document.to_combined_json()
        
        output_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
        
        return output_path
    
    def save_metadata(self, pdf_name: str, metadata: DocumentMetadata) -> Path:
        """
        Save document metadata separately.
        
        Args:
            pdf_name: PDF name
            metadata: Metadata to save
        
        Returns:
            Path to saved file
        """
        output_dir = self._get_output_dir(pdf_name)
        output_path = output_dir / f"{pdf_name}-metadata.json"
        
        output_path.write_text(
            json.dumps(metadata.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
        
        return output_path
    
    def save_page(
        self,
        pdf_name: str,
        page_id: str,
        voters: List[Voter],
        extra_data: Optional[dict[str, Any]] = None,
        page_processing_seconds: Optional[float] = None,
    ) -> Path:
        """
        Save page-wise voter data.
        
        Args:
            pdf_name: PDF name
            page_id: Page identifier
            voters: List of voters on this page
            extra_data: Additional data to include
            page_processing_seconds: How long the page took to process
        
        Returns:
            Path to saved file
        """
        page_dir = self._get_page_wise_dir(pdf_name)
        output_path = page_dir / f"{page_id}.json"
        
        data = {
            "file": pdf_name,
            "page": page_id,
            "generated_at_epoch": int(datetime.utcnow().timestamp()),
            "page_processing_seconds": round(page_processing_seconds, 3) if page_processing_seconds is not None else None,
            "records": [v.to_dict() for v in voters],
        }
        
        # Remove None values
        data = {k: v for k, v in data.items() if v is not None}
        
        if extra_data:
            data.update(extra_data)
        
        output_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
        
        return output_path
    
    def save_stats(self, pdf_name: str, stats: ProcessingStats) -> Path:
        """
        Save processing statistics.
        
        Args:
            pdf_name: PDF name
            stats: Processing statistics
        
        Returns:
            Path to saved file
        """
        output_dir = self._get_output_dir(pdf_name)
        output_path = output_dir / f"{pdf_name}-stats.json"
        
        output_path.write_text(
            json.dumps(stats.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
        
        return output_path
    
    def load_document(self, pdf_name: str) -> Optional[dict[str, Any]]:
        """
        Load processed document data.
        
        Args:
            pdf_name: PDF name
        
        Returns:
            Document data as dictionary, or None if not found
        """
        output_dir = self._get_output_dir(pdf_name)
        output_path = output_dir / f"{pdf_name}.json"
        
        if not output_path.exists():
            return None
        
        return json.loads(output_path.read_text(encoding="utf-8"))
    
    def load_metadata(self, pdf_name: str) -> Optional[dict[str, Any]]:
        """
        Load document metadata.
        
        Args:
            pdf_name: PDF name
        
        Returns:
            Metadata as dictionary, or None if not found
        """
        output_dir = self._get_output_dir(pdf_name)
        output_path = output_dir / f"{pdf_name}-metadata.json"
        
        if not output_path.exists():
            return None
        
        return json.loads(output_path.read_text(encoding="utf-8"))
    
    def load_page(self, pdf_name: str, page_id: str) -> Optional[dict[str, Any]]:
        """
        Load page data.
        
        Args:
            pdf_name: PDF name
            page_id: Page identifier
        
        Returns:
            Page data as dictionary, or None if not found
        """
        page_dir = self._get_page_wise_dir(pdf_name)
        page_path = page_dir / f"{page_id}.json"
        
        if not page_path.exists():
            return None
        
        return json.loads(page_path.read_text(encoding="utf-8"))
    
    def document_exists(self, pdf_name: str) -> bool:
        """Check if a processed document exists."""
        output_dir = self._get_output_dir(pdf_name)
        output_path = output_dir / f"{pdf_name}.json"
        return output_path.exists()
    
    def metadata_exists(self, pdf_name: str) -> bool:
        """Check if metadata exists for a document."""
        output_dir = self._get_output_dir(pdf_name)
        output_path = output_dir / f"{pdf_name}-metadata.json"
        return output_path.exists()
    
    def list_processed(self) -> List[str]:
        """
        List all processed document names.
        
        Returns:
            List of PDF names that have been processed
        """
        processed = []
        
        if not self.base_dir.exists():
            return processed
        
        for folder in self.base_dir.iterdir():
            if not folder.is_dir():
                continue
            
            output_file = folder / "output" / f"{folder.name}.json"
            if output_file.exists():
                processed.append(folder.name)
        
        return sorted(processed)
    
    def get_all_voters(self, pdf_name: str) -> List[dict[str, Any]]:
        """
        Get all voters from a processed document.
        
        Args:
            pdf_name: PDF name
        
        Returns:
            List of voter records
        """
        doc_data = self.load_document(pdf_name)
        if not doc_data:
            return []
        
        return doc_data.get("records", [])
    
    def get_summary(self, pdf_name: str) -> Optional[dict[str, Any]]:
        """
        Get summary statistics for a processed document.
        
        Args:
            pdf_name: PDF name
        
        Returns:
            Summary statistics dictionary
        """
        doc_data = self.load_document(pdf_name)
        if not doc_data:
            return None
        
        return {
            "pdf_name": pdf_name,
            "status": doc_data.get("status", "unknown"),
            "pages_count": doc_data.get("pages_count", 0),
            "total_voters": doc_data.get("total_voters", 0),
            "valid_voters": doc_data.get("valid_voters", 0),
            "timing": doc_data.get("timing", {}),
            "ai_usage": doc_data.get("ai_usage", {}),
        }

    def save_to_csv(self, document: Union[ProcessedDocument, dict[str, Any]]) -> List[Path]:
        """
        Save document data to CSV files.
        
        Creates two files:
        - <base_dir>/<pdf_name>/output/csv/<pdf_name>_voters.csv
        - <base_dir>/<pdf_name>/output/csv/<pdf_name>_metadata.csv
        
        Args:
            document: Processed document object or dict (already in combined format)
            
        Returns:
            List of paths to saved files
        """
        import csv
        
        if isinstance(document, dict):
            data = document
            # Extract pdf_name from data or error?
            # dictionary "folder" is usually pdf_name in to_combined_json
            pdf_name = data.get("folder") or data.get("document", {}).get("pdf_name")
            if not pdf_name:
                 raise ValueError("Could not determine pdf_name from dictionary data")
        else:
            pdf_name = document.pdf_name
            data = document.to_combined_json()
        
        output_dir = self._get_output_dir(pdf_name) / "csv"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        saved_paths = []
        
        # 1. Save Voters CSV
        voters_path = output_dir / f"{pdf_name}_voters.csv"
        saved_paths.append(voters_path)
        records = data.get("records", [])
        
        if records:
            # Collect all keys dynamically
            keys = set()
            for r in records:
                keys.update(r.keys())
            
            # Exclude unwanted voter keys
            voter_exclude = {
                "processing_time_ms", "page_id", "sequence_in_page", 
                "sequence_in_document", "image_file"
            }
            final_keys = {k for k in keys if k not in voter_exclude}
            
            # Prioritize standard keys order for better readability
            ordered_keys = [
                "serial_no", "epic_no", "name", "relation_type", "relation_name",
                "age", "gender", "house_no", 
                "assembly_constituency_number_and_name",
                "section_number_and_name", "part_number"
            ]
            
            # Add remaining keys sorted
            remaining = sorted([k for k in final_keys if k not in ordered_keys])
            fieldnames = [k for k in ordered_keys if k in final_keys] + remaining
            
            try:
                with open(voters_path, "w", newline="", encoding="utf-8-sig") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
                    writer.writeheader()
                    writer.writerows(records)
            except PermissionError:
                print(f"ERROR: Permission denied when writing to {voters_path}")
                print("       Please close the file if it is open in another program (like Excel) and try again.")
        else:
            # Create empty CSV with headers if possible, or just empty file
            try:
                with open(voters_path, "w", newline="", encoding="utf-8-sig") as f:
                    pass
            except PermissionError:
                print(f"ERROR: Permission denied when writing to {voters_path}")
                print("       Please close the file if it is open in another program (like Excel) and try again.")

        # 2. Save Metadata CSV
        metadata_path = output_dir / f"{pdf_name}_metadata.csv"
        saved_paths.append(metadata_path)
        
        # Flatten metadata helper
        def flatten_dict(d: dict, parent_key: str = '', sep: str = '_') -> dict:
            items = []
            for k, v in d.items():
                new_key = f"{parent_key}{sep}{k}" if parent_key else k
                if isinstance(v, dict):
                    items.extend(flatten_dict(v, new_key, sep=sep).items())
                elif isinstance(v, list):
                    # Convert list to string representation or join if simple
                    # specific handling for sections list? 
                    # For general case, keep lists as strings to avoid explosion
                    # But user said "include json data can make them flat"
                    # If it's a list of dicts, standard csv flatten is hard.
                    # We will keep list as string for now unless specific structure is known.
                    items.append((new_key, json.dumps(v, ensure_ascii=False)))
                else:
                    items.append((new_key, v))
            return dict(items)

        meta_row = {}
        
        # Top level fields (excluding complex structures)
        exclude_fields = ["records", "pages", "metadata", "timing", "stats", "ai_usage"]
        for k, v in data.items():
            if k not in exclude_fields:
                meta_row[k] = v
        
        # Flatten nested 'metadata' object recursively
        # User requested removing 'metadata_' prefix, so we pass parent_key=''
        # Fallback: Validation if metadata is missing in main dict, try to reload from sidecar
        if not data.get("metadata"):
            try:
                sidecar = self.load_metadata(pdf_name)
                if sidecar:
                    data["metadata"] = sidecar
            except Exception:
                pass # Ignore if failed, just proceed

        if data.get("metadata"):
            flat_metadata = flatten_dict(data["metadata"], parent_key='')
            meta_row.update(flat_metadata)

        # Flatten 'timing'
        if data.get("timing"):
             flat_timing = flatten_dict(data["timing"], parent_key='timing')
             meta_row.update(flat_timing)
        
        # Filter unwanted metadata keys
        # Note: keys no longer have 'metadata_' prefix
        metadata_exclude = {
            "folder", "document_id", "status", "created_at", "processed_at",
            "language_detected", "page_number_current",
            "ai_metadata_provider", "ai_metadata_model",
            "metadata_document_id", # old key
            "document_id" # possible collision if inside metadata
        }
        
        final_meta_row = {k: v for k, v in meta_row.items() if k not in metadata_exclude}
                
        # Write metadata CSV (single row)
        if final_meta_row:
            try:
                with open(metadata_path, "w", newline="", encoding="utf-8-sig") as f:
                    writer = csv.DictWriter(f, fieldnames=sorted(final_meta_row.keys()))
                    writer.writeheader()
                    writer.writerow(final_meta_row)
            except PermissionError:
                print(f"ERROR: Permission denied when writing to {metadata_path}")
                print("       Please close the file if it is open in another program (like Excel) and try again.")
        
        return saved_paths
