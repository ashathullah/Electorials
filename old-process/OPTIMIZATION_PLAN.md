# Electoral Roll PDF Processing - Code Optimization Plan

## Executive Summary

This document outlines a comprehensive plan to refactor and optimize the Electoral Roll PDF Processing application. The current codebase consists of several independent scripts that need to be executed separately. The goal is to create a unified, maintainable, and production-ready application with proper code structure, error handling, logging, and database-ready data persistence.

---

## Table of Contents

1. [Current State Analysis](#1-current-state-analysis)
2. [Proposed Architecture](#2-proposed-architecture)
3. [Module Structure](#3-module-structure)
4. [Code Optimization Strategy](#4-code-optimization-strategy)
5. [Data Persistence & Database Readiness](#5-data-persistence--database-readiness)
6. [Logging & Debugging Strategy](#6-logging--debugging-strategy)
7. [Configuration Management](#7-configuration-management)
8. [Error Handling Strategy](#8-error-handling-strategy)
9. [Performance Optimization](#9-performance-optimization)
10. [CLI Interface Design](#10-cli-interface-design)
11. [Implementation Phases](#11-implementation-phases)
12. [Migration Guide](#12-migration-guide)

---

## 1. Current State Analysis

### 1.1 Existing Files & Their Responsibilities

| File | Purpose | Lines | Issues |
|------|---------|-------|--------|
| `extract_source.py` | Extract images from PDFs | ~180 | Good structure, standalone |
| `meta_info_capture_ai.py` | AI extraction of front/back page metadata | ~580 | Good, needs integration |
| `crop_voters_for_ocr.py` | Crop voter boxes from page images | ~644 | Complex, needs refactoring |
| `ocr_processor.py` | OCR processing of cropped images | ~1558 | Very large, needs splitting |
| `markdown_preview.py` | Generate Markdown previews | ~160 | Helper module, OK |
| `raw_ocr_dump.py` | Raw OCR debugging dumps | ~235 | Helper module, OK |

### 1.2 Current Problems

1. **No Unified Entry Point**: Each script runs independently
2. **Duplicate Code**: Similar file/folder iteration logic in multiple files
3. **Inconsistent Configuration**: Hardcoded paths, magic numbers
4. **No Centralized Logging**: Print statements scattered throughout
5. **No Environment-based Debug Mode**: Manual flag passing required
6. **Incomplete Requirements**: Missing packages in `requirements.txt`
7. **Data Scattered**: Metadata, voters, timing stored in separate files
8. **No Database-Ready Schema**: JSON structure not normalized
9. **Order Maintenance**: Not explicitly tracked during parallel processing
10. **Expensive AI Calls**: No caching or cost tracking aggregation

### 1.3 Current Data Flow

```
PDF Files (pdfs/)
       │
       ▼
[extract_source.py] ──► extracted/<name>/images/ + manifest.json
       │
       ├──► [meta_info_capture_ai.py] ──► output/<name>-metadata.json
       │
       ▼
[crop_voters_for_ocr.py] ──► extracted/<name>/crops/<page>/
       │
       ▼
[ocr_processor.py] ──► output/page_wise/<page>.json
                   ──► output/<name>.json
                   ──► output/<name>.md
```

---

## 2. Proposed Architecture

### 2.1 High-Level Architecture

```
electorials/
├── main.py                      # Unified entry point
├── config.py                    # Centralized configuration
├── logger.py                    # Logging setup
├── models/                      # Data models (dataclasses/pydantic)
│   ├── __init__.py
│   ├── pdf_document.py          # PDF document model
│   ├── voter.py                 # Voter data model
│   ├── metadata.py              # Metadata model
│   └── processing_stats.py      # Timing/cost tracking model
├── processors/                  # Core processing modules
│   ├── __init__.py
│   ├── base.py                  # Base processor class
│   ├── pdf_extractor.py         # PDF to images
│   ├── metadata_extractor.py    # AI-based metadata extraction
│   ├── image_cropper.py         # Voter box cropping
│   └── ocr_processor.py         # OCR processing
├── utils/                       # Utility functions
│   ├── __init__.py
│   ├── file_utils.py            # File/path operations
│   ├── image_utils.py           # OpenCV helpers
│   ├── ocr_utils.py             # Tesseract helpers
│   └── markdown_utils.py        # Markdown generation
├── persistence/                 # Data persistence layer
│   ├── __init__.py
│   ├── json_store.py            # JSON file storage
│   ├── models.py                # SQLAlchemy models (future)
│   └── repository.py            # Data access layer
├── cli/                         # CLI interface
│   ├── __init__.py
│   └── commands.py              # Click/argparse commands
├── .env.example                 # Environment variables template
├── requirements.txt             # Complete dependencies
└── README.md                    # Project documentation
```

### 2.2 Data Flow (New)

```
┌─────────────────────────────────────────────────────────────────┐
│                        main.py (Orchestrator)                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  PDF File ──► PDFExtractor ──► MetadataExtractor ──► ImageCropper ──► OCRProcessor
│                    │                │                    │              │
│                    ▼                ▼                    ▼              ▼
│              manifest.json    metadata.json         crops/        voters.json
│                    │                │                    │              │
│                    └────────────────┴────────────────────┴──────────────┘
│                                          │
│                                          ▼
│                              ProcessingResult (unified)
│                                          │
│                                          ▼
│                              JSONStore / DatabaseStore
│                                          │
│                                          ▼
│                              final_output.json (combined)
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. Module Structure

### 3.1 Models (`models/`)

```python
# models/voter.py
@dataclass
class Voter:
    serial_no: str
    epic_no: str
    epic_valid: bool
    name: str
    relation_type: str  # father/mother/husband
    relation_name: str
    house_no: str
    age: str
    gender: str
    
    # Metadata for tracking
    image_file: str
    page_id: str
    processing_time_ms: float
    confidence_scores: dict

# models/metadata.py
@dataclass
class DocumentMetadata:
    state: str
    electoral_roll_year: int
    revision_type: str
    qualifying_date: str
    publication_date: str
    # ... (all fields from AI extraction)
    
    # Tracking
    ai_provider: str
    ai_model: str
    ai_cost_usd: float
    ai_tokens_used: dict

# models/processing_stats.py
@dataclass
class ProcessingStats:
    pdf_name: str
    started_at: datetime
    completed_at: datetime
    total_pages: int
    total_voters: int
    
    # Timing breakdown
    extraction_time_sec: float
    metadata_time_sec: float
    cropping_time_sec: float
    ocr_time_sec: float
    
    # Cost tracking
    ai_calls_count: int
    ai_total_cost_usd: float
    ai_tokens_breakdown: dict
    
    # Per-page timing
    page_timings: List[PageTiming]
```

### 3.2 Processors (`processors/`)

```python
# processors/base.py
class BaseProcessor(ABC):
    def __init__(self, config: Config, logger: Logger):
        self.config = config
        self.logger = logger
    
    @abstractmethod
    def process(self, input_data: Any) -> ProcessingResult:
        pass
    
    def _log_timing(self, operation: str, duration: float):
        self.logger.debug(f"{operation}: {duration:.2f}s")

# processors/pdf_extractor.py
class PDFExtractor(BaseProcessor):
    def process(self, pdf_path: Path) -> ExtractionResult:
        # Extract images, create manifest
        pass

# processors/metadata_extractor.py
class MetadataExtractor(BaseProcessor):
    def process(self, images_dir: Path) -> MetadataResult:
        # Call AI for front/back page
        pass

# processors/image_cropper.py
class ImageCropper(BaseProcessor):
    def process(self, images_dir: Path) -> CroppingResult:
        # Crop voter boxes
        pass

# processors/ocr_processor.py
class OCRProcessor(BaseProcessor):
    def process(self, crops_dir: Path) -> OCRResult:
        # Run OCR on crops
        pass
```

### 3.3 Persistence (`persistence/`)

```python
# persistence/json_store.py
class JSONStore:
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
    
    def save_document(self, doc: ProcessedDocument) -> Path:
        """Save complete document with all data"""
        pass
    
    def save_voters(self, pdf_name: str, voters: List[Voter]) -> Path:
        """Save voters list"""
        pass
    
    def save_stats(self, stats: ProcessingStats) -> Path:
        """Save processing statistics"""
        pass
    
    def load_document(self, pdf_name: str) -> Optional[ProcessedDocument]:
        """Load previously processed document"""
        pass

# persistence/repository.py (database-ready interface)
class DocumentRepository(ABC):
    @abstractmethod
    def save(self, document: ProcessedDocument) -> str:
        """Returns document ID"""
        pass
    
    @abstractmethod
    def get(self, document_id: str) -> Optional[ProcessedDocument]:
        pass
    
    @abstractmethod
    def list_all(self) -> List[ProcessedDocument]:
        pass

class JSONRepository(DocumentRepository):
    """JSON file-based implementation"""
    pass

class SQLRepository(DocumentRepository):
    """SQL database implementation (future)"""
    pass
```

---

## 4. Code Optimization Strategy

### 4.1 DRY (Don't Repeat Yourself) Violations to Fix

| Location | Duplicate Code | Solution |
|----------|---------------|----------|
| `extract_source.py`, `crop_voters_for_ocr.py`, `ocr_processor.py` | PDF/folder iteration | `utils/file_utils.py` |
| `crop_voters_for_ocr.py`, `ocr_processor.py` | Image preprocessing | `utils/image_utils.py` |
| `ocr_processor.py` | ROI extraction (EPIC, Serial, House) | Unified `ROIExtractor` class |
| Multiple files | Path construction | `config.py` with path templates |
| Multiple files | Timing measurement | `utils/timing.py` decorator |

### 4.2 Code Smells to Address

1. **Long Functions**: `main()` in `crop_voters_for_ocr.py` (~200 lines) → split into smaller functions
2. **Magic Numbers**: Hardcoded ROI coordinates → move to config
3. **God Module**: `ocr_processor.py` (1558 lines) → split into:
   - `ocr_utils.py` - Tesseract wrappers
   - `field_extractors.py` - Field-specific extraction
   - `line_parser.py` - Line reconstruction
   - `roi_extractor.py` - ROI-based extraction

### 4.3 Utility Functions to Extract

```python
# utils/file_utils.py
def iter_pdfs(input_dir: Path) -> Iterator[Path]
def iter_extracted_folders(extracted_dir: Path) -> Iterator[Path]
def iter_images(images_dir: Path) -> Iterator[Path]
def safe_stem(path: Path) -> str
def ensure_dir(path: Path) -> Path

# utils/image_utils.py
def load_image(path: Path) -> np.ndarray
def save_image(image: np.ndarray, path: Path, quality: int = 100) -> None
def crop_relative(image: np.ndarray, roi: ROI) -> np.ndarray
def preprocess_for_ocr(image: np.ndarray) -> np.ndarray
def estimate_skew(image: np.ndarray) -> float
def deskew(image: np.ndarray, angle: float) -> np.ndarray

# utils/timing.py
@contextmanager
def timed_operation(name: str, logger: Logger) -> Iterator[TimingContext]
def timing_decorator(logger: Logger) -> Callable
```

---

## 5. Data Persistence & Database Readiness

### 5.1 Unified Output Schema (JSON → SQL ready)

```python
# Final output structure that maps to SQL tables

{
    "document": {
        "id": "uuid-or-hash",
        "pdf_name": "2025-EROLLGEN-S22-114-...",
        "pdf_path": "pdfs/original.pdf",
        "created_at": "2025-01-01T00:00:00Z",
        "processed_at": "2025-01-01T01:00:00Z",
        "status": "completed"  # pending, processing, completed, failed
    },
    
    "metadata": {
        "document_id": "fk-to-document",
        "state": "Tamil Nadu",
        "electoral_roll_year": 2025,
        "assembly_constituency_number": 114,
        "assembly_constituency_name": "திருப்பூர் (தெற்கு)",
        # ... all metadata fields
        
        # AI tracking (for cost analysis)
        "ai_provider": "gemini",
        "ai_model": "gemini-2.5-flash",
        "ai_input_tokens": 1234,
        "ai_output_tokens": 567,
        "ai_cost_usd": 0.0012
    },
    
    "pages": [
        {
            "document_id": "fk-to-document",
            "page_number": 4,
            "page_id": "page-004",
            "image_path": "images/page-004.png",
            "crops_count": 30,
            "voters_count": 30,
            "processing_time_sec": 45.2
        }
    ],
    
    "voters": [
        {
            "id": "uuid",
            "document_id": "fk-to-document",
            "page_id": "page-004",
            "sequence_in_page": 1,  # Maintains order!
            "sequence_in_document": 91,  # Global order
            
            "serial_no": "91",
            "epic_no": "XYZ1234567",
            "epic_valid": true,
            "name": "குமார் ச",
            "relation_type": "father",
            "relation_name": "சரவணன்",
            "house_no": "123/A",
            "age": "45",
            "gender": "Male",
            
            # Tracking
            "image_file": "page-004-001.png",
            "processing_time_ms": 1234,
            "extraction_confidence": 0.95
        }
    ],
    
    "processing_stats": {
        "document_id": "fk-to-document",
        "total_time_sec": 300.5,
        "extraction_time_sec": 10.2,
        "metadata_time_sec": 5.3,
        "cropping_time_sec": 45.0,
        "ocr_time_sec": 240.0,
        
        "total_pages": 55,
        "total_crops": 1650,
        "total_voters": 1400,
        "avg_time_per_voter_ms": 171.4,
        
        "ai_total_cost_usd": 0.0012,
        "errors_count": 0
    }
}
```

### 5.2 SQL Schema (Future Reference)

```sql
-- Documents table
CREATE TABLE documents (
    id UUID PRIMARY KEY,
    pdf_name VARCHAR(255) NOT NULL,
    pdf_path VARCHAR(500),
    status VARCHAR(50) DEFAULT 'pending',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    processed_at TIMESTAMP
);

-- Document metadata
CREATE TABLE document_metadata (
    id UUID PRIMARY KEY,
    document_id UUID REFERENCES documents(id),
    state VARCHAR(100),
    electoral_roll_year INTEGER,
    assembly_constituency_number INTEGER,
    assembly_constituency_name VARCHAR(255),
    -- ... other fields
    ai_provider VARCHAR(50),
    ai_model VARCHAR(100),
    ai_cost_usd DECIMAL(10, 6)
);

-- Pages
CREATE TABLE pages (
    id UUID PRIMARY KEY,
    document_id UUID REFERENCES documents(id),
    page_number INTEGER,
    page_id VARCHAR(50),
    voters_count INTEGER,
    processing_time_sec DECIMAL(10, 4)
);

-- Voters (main data)
CREATE TABLE voters (
    id UUID PRIMARY KEY,
    document_id UUID REFERENCES documents(id),
    page_id VARCHAR(50),
    sequence_in_page INTEGER,
    sequence_in_document INTEGER,
    
    serial_no VARCHAR(20),
    epic_no VARCHAR(20),
    epic_valid BOOLEAN,
    name VARCHAR(255),
    relation_type VARCHAR(20),
    relation_name VARCHAR(255),
    house_no VARCHAR(50),
    age VARCHAR(10),
    gender VARCHAR(20),
    
    processing_time_ms INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    INDEX idx_document (document_id),
    INDEX idx_epic (epic_no),
    INDEX idx_name (name)
);

-- Processing statistics
CREATE TABLE processing_stats (
    id UUID PRIMARY KEY,
    document_id UUID REFERENCES documents(id),
    total_time_sec DECIMAL(10, 4),
    ai_total_cost_usd DECIMAL(10, 6),
    -- ... other stats
);
```

---

## 6. Logging & Debugging Strategy

### 6.1 Logging Configuration

```python
# logger.py
import logging
import os
from pathlib import Path
from datetime import datetime

def setup_logger(
    name: str = "electorials",
    log_dir: Path = Path("logs"),
    debug: bool = None
) -> logging.Logger:
    """
    Setup logger with file and console handlers.
    
    Debug mode controlled by:
    1. Explicit parameter
    2. Environment variable DEBUG=1/true
    """
    if debug is None:
        debug = os.getenv("DEBUG", "").lower() in ("1", "true", "yes")
    
    level = logging.DEBUG if debug else logging.INFO
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Console handler
    console = logging.StreamHandler()
    console.setLevel(level)
    console_format = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%H:%M:%S'
    )
    console.setFormatter(console_format)
    logger.addHandler(console)
    
    # File handler (always DEBUG level for troubleshooting)
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / f"{datetime.now():%Y%m%d_%H%M%S}.log"
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)s | %(funcName)s:%(lineno)d | %(message)s'
    )
    file_handler.setFormatter(file_format)
    logger.addHandler(file_handler)
    
    return logger
```

### 6.2 Debug Mode Features

When `DEBUG=1` is set in environment:

1. **Verbose Logging**: All debug messages printed to console
2. **Raw OCR Dumps**: Automatically create markdown dumps
3. **ROI Visualization**: Save debug images with ROI overlays
4. **Timing Details**: Per-operation timing breakdown
5. **Intermediate Files**: Keep all intermediate processing files

```python
# config.py
@dataclass
class Config:
    debug: bool = field(default_factory=lambda: os.getenv("DEBUG", "").lower() in ("1", "true"))
    
    @property
    def dump_raw_ocr(self) -> bool:
        return self.debug
    
    @property
    def save_roi_overlays(self) -> bool:
        return self.debug
    
    @property
    def keep_intermediate_files(self) -> bool:
        return self.debug
```

---

## 7. Configuration Management

### 7.1 Environment Variables

```bash
# .env.example

# === Processing ===
DEBUG=0                           # Enable debug mode (1/true/yes)
EXTRACTED_DIR=extracted           # Output directory for extracted files
PDFS_DIR=pdfs                     # Input directory for PDFs

# === PDF Extraction ===
RENDER_DPI=200                    # DPI for rendering PDF pages

# === AI Metadata Extraction ===
AI_PROVIDER=gemini                # AI provider (gemini/openai)
AI_API_KEY=your-api-key           # API key
AI_MODEL=gemini-2.5-flash         # Model name
AI_BASE_URL=                      # Custom base URL (optional)
AI_TIMEOUT_SEC=120                # Request timeout

# === AI Cost Tracking ===
AI_INPUT_COST_PER_1M_USD=0.075    # Input token cost
AI_OUTPUT_COST_PER_1M_USD=0.30    # Output token cost
AI_COST_CURRENCY=USD

# === OCR ===
OCR_LANGUAGES=eng+tam             # Tesseract languages
TESSERACT_PATH=                   # Custom Tesseract path (Windows)

# === Logging ===
LOG_LEVEL=INFO                    # Logging level
LOG_DIR=logs                      # Log files directory
```

### 7.2 Configuration Class

```python
# config.py
from dataclasses import dataclass, field
from pathlib import Path
import os

def _load_dotenv():
    """Load .env file if exists"""
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                if os.getenv(key.strip()) is None:
                    os.environ[key.strip()] = value.strip().strip('"\'')

_load_dotenv()

@dataclass
class Config:
    # Directories
    base_dir: Path = field(default_factory=lambda: Path(__file__).parent)
    pdfs_dir: Path = field(default_factory=lambda: Path(os.getenv("PDFS_DIR", "pdfs")))
    extracted_dir: Path = field(default_factory=lambda: Path(os.getenv("EXTRACTED_DIR", "extracted")))
    logs_dir: Path = field(default_factory=lambda: Path(os.getenv("LOG_DIR", "logs")))
    
    # Debug
    debug: bool = field(default_factory=lambda: os.getenv("DEBUG", "").lower() in ("1", "true", "yes"))
    
    # PDF extraction
    render_dpi: int = field(default_factory=lambda: int(os.getenv("RENDER_DPI", "200")))
    
    # AI
    ai_provider: str = field(default_factory=lambda: os.getenv("AI_PROVIDER", "gemini"))
    ai_api_key: str = field(default_factory=lambda: os.getenv("AI_API_KEY", ""))
    ai_model: str = field(default_factory=lambda: os.getenv("AI_MODEL", "gemini-2.5-flash"))
    ai_base_url: str = field(default_factory=lambda: os.getenv("AI_BASE_URL", ""))
    ai_timeout_sec: int = field(default_factory=lambda: int(os.getenv("AI_TIMEOUT_SEC", "120")))
    
    # OCR
    ocr_languages: str = field(default_factory=lambda: os.getenv("OCR_LANGUAGES", "eng+tam"))
    tesseract_path: str = field(default_factory=lambda: os.getenv("TESSERACT_PATH", ""))
    
    # ROI configurations (relative coordinates)
    epic_roi: tuple = (0.65, 0.05, 0.98, 0.20)
    serial_roi: tuple = (0.15, 0.06, 0.329, 0.2)
    house_roi: tuple = (0.02, 0.42, 0.40, 0.54)
    
    def __post_init__(self):
        # Resolve paths relative to base_dir
        self.pdfs_dir = self.base_dir / self.pdfs_dir
        self.extracted_dir = self.base_dir / self.extracted_dir
        self.logs_dir = self.base_dir / self.logs_dir
```

---

## 8. Error Handling Strategy

### 8.1 Custom Exceptions

```python
# exceptions.py

class ElectorialsError(Exception):
    """Base exception for all application errors"""
    pass

class PDFExtractionError(ElectorialsError):
    """Failed to extract images from PDF"""
    pass

class MetadataExtractionError(ElectorialsError):
    """Failed to extract metadata via AI"""
    pass

class CroppingError(ElectorialsError):
    """Failed to crop voter boxes"""
    pass

class OCRError(ElectorialsError):
    """OCR processing failed"""
    pass

class ConfigurationError(ElectorialsError):
    """Invalid configuration"""
    pass
```

### 8.2 Error Recovery

```python
# processors/base.py

class BaseProcessor:
    def process_with_recovery(self, input_data: Any) -> ProcessingResult:
        """Process with automatic retry and error capture"""
        max_retries = 3
        last_error = None
        
        for attempt in range(max_retries):
            try:
                return self.process(input_data)
            except TransientError as e:
                self.logger.warning(f"Attempt {attempt + 1} failed: {e}")
                last_error = e
                time.sleep(2 ** attempt)  # Exponential backoff
            except PermanentError as e:
                self.logger.error(f"Permanent error: {e}")
                return ProcessingResult(
                    success=False,
                    error=str(e),
                    error_type=type(e).__name__
                )
        
        return ProcessingResult(
            success=False,
            error=f"Failed after {max_retries} attempts: {last_error}"
        )
```

---

## 9. Performance Optimization

### 9.1 Cropping Optimization

Current bottleneck: Sequential image processing

```python
# Option 1: Batch processing with multiprocessing (RECOMMENDED for order maintenance)
from concurrent.futures import ProcessPoolExecutor, as_completed

def process_pages_parallel(pages: List[Path], max_workers: int = 4) -> List[CropResult]:
    """Process pages in parallel while maintaining order"""
    results = [None] * len(pages)
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit with index to maintain order
        futures = {
            executor.submit(process_single_page, page): idx 
            for idx, page in enumerate(pages)
        }
        
        for future in as_completed(futures):
            idx = futures[future]
            results[idx] = future.result()
    
    return results

# Option 2: Async I/O for AI calls
import asyncio
import aiohttp

async def extract_metadata_batch(folders: List[Path]) -> List[MetadataResult]:
    """Extract metadata for multiple folders concurrently"""
    async with aiohttp.ClientSession() as session:
        tasks = [extract_single_metadata(session, folder) for folder in folders]
        return await asyncio.gather(*tasks)
```

### 9.2 OCR Optimization

```python
# Pre-load Tesseract data once
class TesseractManager:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        # Initialize Tesseract once
        self.initialized = True
```

### 9.3 Memory Optimization

```python
# Process images in batches to avoid memory issues
def process_images_batched(images: List[Path], batch_size: int = 50):
    """Process images in batches to manage memory"""
    for i in range(0, len(images), batch_size):
        batch = images[i:i + batch_size]
        yield from process_batch(batch)
        gc.collect()  # Explicit garbage collection between batches
```

---

## 10. CLI Interface Design

### 10.1 Main Entry Point

```python
# main.py
"""
Electoral Roll PDF Processor

Usage:
    # Process all PDFs in default directory
    python main.py
    
    # Process specific PDFs
    python main.py path/to/file1.pdf path/to/file2.pdf
    
    # Run specific step only
    python main.py --step extract
    python main.py --step metadata
    python main.py --step crop
    python main.py --step ocr
    python main.py --step combine
    
    # Debug mode
    DEBUG=1 python main.py
    
    # Limit processing
    python main.py --limit 5 --limit-pages 10
"""

import argparse
import sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(
        description="Electoral Roll PDF Processor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                          # Process all PDFs
  python main.py file1.pdf file2.pdf      # Process specific PDFs
  python main.py --step extract           # Extract images only
  python main.py --step ocr --folder XYZ  # Run OCR on specific folder
  DEBUG=1 python main.py                  # Enable debug mode
        """
    )
    
    parser.add_argument(
        "pdfs",
        nargs="*",
        help="PDF files to process (default: all in pdfs/)"
    )
    
    parser.add_argument(
        "--step",
        choices=["all", "extract", "metadata", "crop", "ocr", "combine"],
        default="all",
        help="Processing step to run"
    )
    
    parser.add_argument(
        "--folder",
        type=str,
        help="Process specific extracted folder"
    )
    
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit number of PDFs/folders to process"
    )
    
    parser.add_argument(
        "--limit-pages",
        type=int,
        default=0,
        help="Limit pages per PDF"
    )
    
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing outputs"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be processed without executing"
    )
    
    args = parser.parse_args()
    
    # Initialize
    config = Config()
    logger = setup_logger(debug=config.debug)
    
    # Run orchestrator
    orchestrator = ProcessingOrchestrator(config, logger)
    
    try:
        if args.pdfs:
            # Process specific PDFs
            pdf_paths = [Path(p) for p in args.pdfs]
            result = orchestrator.process_pdfs(pdf_paths, step=args.step)
        else:
            # Process all PDFs in directory
            result = orchestrator.process_all(
                step=args.step,
                folder=args.folder,
                limit=args.limit,
                limit_pages=args.limit_pages,
                force=args.force,
                dry_run=args.dry_run
            )
        
        return 0 if result.success else 1
        
    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
        return 130
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
```

### 10.2 Backward Compatibility

Keep individual scripts runnable:

```python
# extract_source.py (modified)
from processors.pdf_extractor import PDFExtractor
from config import Config
from logger import setup_logger

def main():
    # Parse original arguments for backward compatibility
    args = parse_args()
    
    config = Config()
    logger = setup_logger()
    
    extractor = PDFExtractor(config, logger)
    extractor.run_standalone(args)

if __name__ == "__main__":
    main()
```

---

## 11. Implementation Phases

### Phase 1: Foundation (Week 1)
- [ ] Create new directory structure
- [ ] Implement `config.py` with environment loading
- [ ] Implement `logger.py` with debug mode
- [ ] Create base models (`models/`)
- [ ] Create utility modules (`utils/`)
- [ ] Update `requirements.txt`

### Phase 2: Refactor Processors (Week 2)
- [ ] Create `BaseProcessor` class
- [ ] Refactor `PDFExtractor` from `extract_source.py`
- [ ] Refactor `MetadataExtractor` from `meta_info_capture_ai.py`
- [ ] Refactor `ImageCropper` from `crop_voters_for_ocr.py`
- [ ] Split and refactor `OCRProcessor` from `ocr_processor.py`

### Phase 3: Data Persistence (Week 3)
- [ ] Implement `JSONStore`
- [ ] Create unified output schema
- [ ] Implement sequence/order tracking
- [ ] Implement data combination logic

### Phase 4: Integration (Week 4)
- [ ] Create `ProcessingOrchestrator`
- [ ] Implement `main.py` CLI
- [ ] Add backward compatibility wrappers
- [ ] Comprehensive testing

### Phase 5: Documentation & Polish (Week 5)
- [ ] Write README.md
- [ ] Create .env.example
- [ ] Add inline documentation
- [ ] Performance testing and optimization

---

## 12. Migration Guide

### 12.1 File Changes

| Old File | New Location | Notes |
|----------|--------------|-------|
| `extract_source.py` | `processors/pdf_extractor.py` | Wrapped in class |
| `meta_info_capture_ai.py` | `processors/metadata_extractor.py` | Wrapped in class |
| `crop_voters_for_ocr.py` | `processors/image_cropper.py` | Wrapped in class |
| `ocr_processor.py` | `processors/ocr_processor.py` + `utils/ocr_utils.py` | Split |
| `markdown_preview.py` | `utils/markdown_utils.py` | Moved |
| `raw_ocr_dump.py` | `utils/ocr_debug.py` | Moved |

### 12.2 Breaking Changes

1. **Output Structure**: JSON output format changes to unified schema
2. **CLI Arguments**: New unified CLI, old scripts still work standalone
3. **Configuration**: Environment variables now required for some features

### 12.3 Backward Compatibility

- All original scripts remain runnable with same arguments
- Old output format available via `--legacy-output` flag
- Existing `extracted/` folders will be migrated automatically

---

## Summary

This optimization plan transforms the Electoral Roll PDF Processing application from a collection of independent scripts into a well-structured, maintainable application with:

1. **Unified Entry Point**: Single `main.py` with flexible CLI
2. **Modular Architecture**: Clear separation of concerns
3. **Environment-based Configuration**: Easy deployment and debugging
4. **Proper Logging**: Centralized, configurable logging with debug mode
5. **Database-Ready Schema**: Normalized data structure for future SQL migration
6. **Order Preservation**: Explicit sequence tracking for voters
7. **Cost Tracking**: AI usage and timing statistics
8. **Error Handling**: Graceful recovery and detailed error reporting
9. **Performance**: Parallel processing where safe
10. **Documentation**: Comprehensive README and inline docs

The implementation follows Python best practices including DRY, SOLID principles, and proper use of dataclasses/type hints.
