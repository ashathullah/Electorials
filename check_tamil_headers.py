"""Check header extraction results."""
import json
from pathlib import Path
import sys

sys.path.insert(0, '.')
from src.processors.header_extractor import HeaderExtractor
from src.processors.base import ProcessingContext
from src.config import Config

def main():
    folder_name = sys.argv[1] if len(sys.argv) > 1 else 'tamil_removed'
    languages = sys.argv[2] if len(sys.argv) > 2 else 'tam+eng'
    
    config = Config()
    context = ProcessingContext(config=config)
    extracted_dir = Path(f'extracted/{folder_name}')
    context.setup_paths_from_extracted(extracted_dir)

    extractor = HeaderExtractor(context, languages=languages)
    extractor.run()

    results = {}
    for page_id, header in sorted(extractor.get_all_headers().items()):
        results[page_id] = {
            'assembly': header.assembly_constituency_number_and_name,
            'section': header.section_number_and_name,
            'part': header.part_number
        }

    output_file = f'header_results_{folder_name}.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Results saved to {output_file}")

if __name__ == '__main__':
    main()
