#!/usr/bin/env python3
"""Test script for header extraction."""

import sys
import io
from pathlib import Path

# Set UTF-8 encoding for stdout
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from src.processors import HeaderExtractor, ProcessingContext
from src.config import Config

def test_header_extraction(folder_name: str = "engilsh_removed", languages: str = "eng+tam"):
    config = Config()
    ctx = ProcessingContext(config=config)
    
    # Test on existing extracted folder
    extracted_folder = Path(f'extracted/{folder_name}')
    
    if not extracted_folder.exists():
        print(f"Folder not found: {extracted_folder}")
        return
    
    ctx.setup_paths_from_extracted(extracted_folder)
    print(f"Testing header extraction on: {extracted_folder.name}")
    print(f"Images dir: {ctx.images_dir}")
    print(f"Crop-top dir: {ctx.crop_top_dir}")
    
    extractor = HeaderExtractor(ctx, languages=languages)
    
    if not extractor.validate():
        print("Validation failed")
        return
    
    print("Running header extraction...")
    result = extractor.run()
    print(f"Extraction result: {result}")
    
    headers = extractor.get_all_headers()
    print(f"Found {len(headers)} headers")
    
    # Print all headers
    for page_id, header in sorted(headers.items()):
        print(f"\n{page_id}:")
        print(f"  Assembly: {header.assembly_constituency_number_and_name}")
        print(f"  Section: {header.section_number_and_name}")
        print(f"  Part: {header.part_number}")
        print(f"  Raw text preview: {header.raw_text[:150] if header.raw_text else 'None'}...")

if __name__ == "__main__":
    folder = sys.argv[1] if len(sys.argv) > 1 else "engilsh_removed"
    lang = sys.argv[2] if len(sys.argv) > 2 else "eng+tam"
    test_header_extraction(folder, lang)
