"""
Document processors module.

Contains all processing components for the Electoral Roll pipeline:
- PDFExtractor: Extract images from PDF files
- MetadataExtractor: Extract metadata using AI
- ImageCropper: Crop voter boxes from page images
- ImageMerger: Merge cropped images for batch OCR
- OCRProcessor: Extract voter data from cropped images
"""

from .base import BaseProcessor, ProcessingContext
from .pdf_extractor import PDFExtractor
from .metadata_extractor import MetadataExtractor
from .image_cropper import ImageCropper
from .image_merger import ImageMerger
from .ocr_processor import OCRProcessor

__all__ = [
    "BaseProcessor",
    "ProcessingContext",
    "PDFExtractor",
    "MetadataExtractor",
    "ImageCropper",
    "ImageMerger",
    "OCRProcessor",
]
