"""
JSON file-based storage implementation.

Provides persistent storage for processed documents using JSON files.
Designed to be easily migratable to a database in the future.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, List, Any
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
        extra_data: Optional[dict[str, Any]] = None
    ) -> Path:
        """
        Save page-wise voter data.
        
        Args:
            pdf_name: PDF name
            page_id: Page identifier
            voters: List of voters on this page
            extra_data: Additional data to include
        
        Returns:
            Path to saved file
        """
        page_dir = self._get_page_wise_dir(pdf_name)
        output_path = page_dir / f"{page_id}.json"
        
        data = {
            "file": pdf_name,
            "page": page_id,
            "generated_at_epoch": int(datetime.utcnow().timestamp()),
            "records": [v.to_dict() for v in voters],
        }
        
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
