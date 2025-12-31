"""
Document models.

Represents the complete processed document with all associated data.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional, List, Any
from datetime import datetime
import uuid

from .voter import Voter
from .metadata import DocumentMetadata
from .processing_stats import ProcessingStats


@dataclass
class PageData:
    """
    Data for a single page of the electoral roll.
    
    Contains all voters extracted from this page.
    """
    
    # Page identification
    page_id: str = ""  # e.g., "page-004"
    page_number: int = 0  # 1-based
    
    # Source files
    image_path: str = ""
    
    # Voters on this page (in order)
    voters: List[Voter] = field(default_factory=list)
    
    # Processing info
    crops_count: int = 0
    processing_time_sec: float = 0.0
    
    @property
    def voters_count(self) -> int:
        return len(self.voters)
    
    @property
    def valid_voters_count(self) -> int:
        return sum(1 for v in self.voters if v.epic_valid)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "page_id": self.page_id,
            "page_number": self.page_number,
            "image_path": self.image_path,
            "crops_count": self.crops_count,
            "voters_count": self.voters_count,
            "valid_voters_count": self.valid_voters_count,
            "processing_time_sec": round(self.processing_time_sec, 4),
            "voters": [v.to_dict() for v in self.voters],
        }


@dataclass
class ProcessedDocument:
    """
    Complete processed document with all data.
    
    This is the main data structure that combines:
    - Document identification
    - Metadata from AI extraction
    - All voter data from OCR
    - Processing statistics
    
    Designed to be database-ready with clear relationships.
    """
    
    # Primary identification
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    pdf_name: str = ""
    pdf_path: str = ""
    
    # Status
    status: str = "pending"  # pending, processing, completed, failed
    
    # Timestamps
    created_at: str = field(
        default_factory=lambda: datetime.utcnow().isoformat() + "Z"
    )
    processed_at: str = ""
    
    # Metadata (from AI extraction)
    metadata: Optional[DocumentMetadata] = None
    
    # Pages with voter data
    pages: List[PageData] = field(default_factory=list)
    
    # Processing statistics
    stats: ProcessingStats = field(default_factory=ProcessingStats)
    
    @property
    def total_voters(self) -> int:
        """Total number of voters across all pages."""
        return sum(p.voters_count for p in self.pages)
    
    @property
    def valid_voters(self) -> int:
        """Number of voters with valid EPIC."""
        return sum(p.valid_voters_count for p in self.pages)
    
    @property
    def all_voters(self) -> List[Voter]:
        """Get all voters in document order."""
        voters = []
        for page in self.pages:
            voters.extend(page.voters)
        return voters
    
    def add_page(self, page_data: PageData) -> None:
        """Add a processed page with voters."""
        # Update sequence numbers
        current_doc_sequence = self.total_voters
        for i, voter in enumerate(page_data.voters):
            voter.sequence_in_page = i + 1
            voter.sequence_in_document = current_doc_sequence + i + 1
        
        self.pages.append(page_data)
    
    def add_voters(self, voters: List[Voter]) -> None:
        """
        Add voters from a flat list, grouping by source_image.
        
        This is a convenience method for adding voters when page data
        is not already structured.
        """
        from collections import defaultdict
        
        # Group voters by page (derived from source_image)
        by_page = defaultdict(list)
        for voter in voters:
            # Extract page_id from source_image (e.g., "page-004-001.png" -> "page-004")
            if voter.source_image:
                parts = voter.source_image.split("-")
                if len(parts) >= 2:
                    page_id = f"{parts[0]}-{parts[1]}"
                else:
                    page_id = voter.source_image
            else:
                page_id = "unknown"
            by_page[page_id].append(voter)
        
        # Sort and add each page
        for page_id in sorted(by_page.keys()):
            page_voters = by_page[page_id]
            # Try to get page number from page_id
            page_num = 0
            try:
                page_num = int(page_id.split("-")[1])
            except (IndexError, ValueError):
                pass
            
            page = PageData(
                page_id=page_id,
                page_number=page_num,
                voters=page_voters,
                crops_count=len(page_voters),
            )
            self.add_page(page)
    
    def complete(self) -> None:
        """Mark document as completed."""
        self.status = "completed"
        self.processed_at = datetime.utcnow().isoformat() + "Z"
        self.stats.complete()
    
    def fail(self, error: str) -> None:
        """Mark document as failed."""
        self.status = "failed"
        self.processed_at = datetime.utcnow().isoformat() + "Z"
        self.stats.fail(error)
    
    def to_dict(self) -> dict[str, Any]:
        """
        Convert to dictionary for JSON serialization.
        
        This is the unified output format that can be:
        1. Saved as JSON
        2. Mapped to SQL tables
        """
        return {
            "document": {
                "id": self.id,
                "pdf_name": self.pdf_name,
                "pdf_path": self.pdf_path,
                "status": self.status,
                "created_at": self.created_at,
                "processed_at": self.processed_at,
                "total_pages": len(self.pages),
                "total_voters": self.total_voters,
                "valid_voters": self.valid_voters,
            },
            
            "metadata": self.metadata.to_dict() if self.metadata else None,
            
            "pages": [p.to_dict() for p in self.pages],
            
            "voters": [v.to_dict() for v in self.all_voters],
            
            "stats": self.stats.to_dict(),
        }
    
    def to_combined_json(self) -> dict[str, Any]:
        """
        Generate combined JSON matching original output format.
        
        This maintains backward compatibility with the existing
        output structure while adding new tracking fields.
        """
        # Combine voters across all pages
        all_records = []
        for page in self.pages:
            for voter in page.voters:
                all_records.append(voter.to_dict())
        
        return {
            "folder": self.pdf_name,
            "document_id": self.id,
            "status": self.status,
            "created_at": self.created_at,
            "processed_at": self.processed_at,
            
            # Metadata if available
            "metadata": self.metadata.to_dict() if self.metadata else None,
            
            # Summary counts
            "pages_count": len(self.pages),
            "images_count": sum(p.crops_count for p in self.pages),
            "total_voters": self.total_voters,
            "valid_voters": self.valid_voters,
            
            # Timing
            "timing": {
                "total_time_sec": round(self.stats.total_time_sec, 4),
                "avg_time_per_voter_ms": round(self.stats.avg_time_per_voter_ms, 2),
            },
            
            # AI tracking
            "ai_usage": self.stats.ai_usage.to_dict(),
            
            # Page summaries
            "pages": [
                {
                    "page": p.page_id,
                    "page_number": p.page_number,
                    "images_processed": p.crops_count,
                    "voters_count": p.voters_count,
                    "processing_time_sec": round(p.processing_time_sec, 4),
                }
                for p in self.pages
            ],
            
            # All voter records
            "records": all_records,
        }


@dataclass
class ProcessingResult:
    """
    Result of a processing operation.
    
    Used to communicate success/failure between processors.
    """
    success: bool = True
    message: str = ""
    error: Optional[str] = None
    error_type: Optional[str] = None
    data: Any = None
    timing_sec: float = 0.0
    
    @classmethod
    def ok(cls, data: Any = None, message: str = "", timing_sec: float = 0.0) -> "ProcessingResult":
        """Create successful result."""
        return cls(success=True, data=data, message=message, timing_sec=timing_sec)
    
    @classmethod
    def error(
        cls,
        error: str,
        error_type: str = "UnknownError",
        timing_sec: float = 0.0
    ) -> "ProcessingResult":
        """Create error result."""
        return cls(
            success=False,
            error=error,
            error_type=error_type,
            timing_sec=timing_sec
        )
