
"""
Document processors module.

Contains all processing components for the Electoral Roll pipeline:
- PDFExtractor: Extract images from PDF files
- MetadataExtractor: Extract metadata using AI
- ImageCropper: Crop voter boxes from page images
- ImageMerger: Merge cropped images for batch OCR
- OCRProcessor: Extract voter data from cropped images
- AIOCRProcessor: Extract voter data using AI (Groq) Vision
- HeaderExtractor: Extract page header metadata (assembly, section, part)
- FieldCropper: Crop specific data fields and stitch into compact images
- CropTopMerger: Merge crop-top images with filename labels for batch processing
"""

from .base import BaseProcessor, ProcessingContext
from .pdf_extractor import PDFExtractor
from .metadata_extractor import MetadataExtractor
from .image_cropper import ImageCropper
from .image_merger import ImageMerger
from .ocr_processor import OCRProcessor
from .ai_ocr_processor import AIOCRProcessor, FailedImage
from .header_extractor import HeaderExtractor
from .field_cropper import FieldCropper
from .crop_top_merger import CropTopMerger

__all__ = [
    "BaseProcessor",
    "ProcessingContext",
    "PDFExtractor",
    "MetadataExtractor",
    "ImageCropper",
    "ImageMerger",
    "OCRProcessor",
    "AIOCRProcessor",
    "FailedImage",
    "HeaderExtractor",
    "FieldCropper",
    "CropTopMerger",
]

