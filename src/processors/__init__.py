"""
Document processors module.

Contains all processing components for the Electoral Roll pipeline:
- PDFExtractor: Extract images from PDF files
- MetadataExtractor: Extract metadata using AI
- ImageCropper: Crop voter boxes from page images
- ImageMerger: Merge cropped images for batch OCR
- OCRProcessor: Extract voter data from cropped images
- HeaderExtractor: Extract page header metadata (assembly, section, part)
- FieldCropper: Crop specific data fields and stitch into compact images
"""

from .base import BaseProcessor, ProcessingContext
from .pdf_extractor import PDFExtractor
from .metadata_extractor import MetadataExtractor
from .image_cropper import ImageCropper
from .image_merger import ImageMerger
from .ocr_processor import OCRProcessor
from .header_extractor import HeaderExtractor
from .field_cropper import FieldCropper

__all__ = [
    "BaseProcessor",
    "ProcessingContext",
    "PDFExtractor",
    "MetadataExtractor",
    "ImageCropper",
    "ImageMerger",
    "OCRProcessor",
    "HeaderExtractor",
    "FieldCropper",
]
