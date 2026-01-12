import os
import json
from pathlib import Path
from urllib.parse import quote

def extract_state_and_constituency(filename):
    """
    Extract state and constituency from filename
    Example: Tamil Nadu-(S22)_Manachanallur-(AC144)_2
    Returns: state, constituency
    """
    # Remove _voters.json suffix if present
    if filename.endswith('_voters.json'):
        filename = filename[:-12]
    elif filename.endswith('.json'):
        filename = filename[:-5]
    
    # Split by -(S and -(AC to extract parts
    parts = filename.split('-(S')
    if len(parts) < 2:
        return None, None
    
    state = parts[0]
    
    # Now split the second part by )_
    remaining = parts[1]
    s_code_and_rest = remaining.split(')_', 1)
    if len(s_code_and_rest) < 2:
        return None, None
    
    # Extract constituency from the remaining part
    constituency_part = s_code_and_rest[1]
    constituency_parts = constituency_part.split('-(AC')
    if len(constituency_parts) < 2:
        return None, None
    
    constituency = constituency_parts[0]
    
    return state, constituency

def generate_download_link(filename):
    """
    Generate S3 URI for the PDF
    Format: s3://264676382451-eci-download/2026/1/S22/pdfs/{State}/{Constituency}/{filename}.pdf
    """
    # Remove _voters.json suffix
    clean_filename = filename.replace('_voters.json', '').replace('.json', '')
    
    # Extract state and constituency
    state, constituency = extract_state_and_constituency(filename)
    
    if not state or not constituency:
        return None
    
    # Construct the S3 URI (no URL encoding needed, use spaces as-is)
    download_link = f"s3://264676382451-eci-download/2026/1/S22/pdfs/{state}/{constituency}/{clean_filename}.pdf"
    
    return download_link

def find_voters_with_missing_names(json_file_path):
    """
    Find all voters with missing names in a JSON file
    Returns: list of serial numbers
    """
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            voters = json.load(f)
        
        missing_name_serial_nos = []
        
        for voter in voters:
            name = voter.get('name', '').strip()
            serial_no = voter.get('serial_no', '')
            
            # Check if name is empty or missing
            if not name:
                missing_name_serial_nos.append(serial_no)
        
        return missing_name_serial_nos
    
    except Exception as e:
        print(f"Error processing {json_file_path}: {e}")
        return []

def main():
    # Directory containing the voter JSON files
    input_dir = Path("e:/Raja_mohaemd/projects/voter-shield-data-cleanup/valid_voter_sections_json")
    
    # Output file
    output_file = Path("e:/Raja_mohaemd/projects/voter-shield-data-cleanup/voters_missing_names.json")
    
    results = []
    
    # Process all JSON files in the directory
    json_files = list(input_dir.glob("*.json"))
    print(f"Found {len(json_files)} JSON files to process...")
    
    processed_count = 0
    files_with_missing_names = 0
    
    for json_file in json_files:
        filename = json_file.name
        
        # Skip any output or temporary files
        if filename.startswith('test_'):
            continue
        
        # Find voters with missing names
        missing_serial_nos = find_voters_with_missing_names(json_file)
        
        if missing_serial_nos:
            # Remove _voters suffix from filename
            clean_filename = filename.replace('_voters.json', '')
            
            # Generate download link
            download_link = generate_download_link(filename)
            
            if download_link:
                result = {
                    "file_name": clean_filename,
                    "missing_name_serial_numbers": missing_serial_nos,
                    "download_link": download_link
                }
                
                results.append(result)
                files_with_missing_names += 1
                print(f"✓ {clean_filename}: {len(missing_serial_nos)} voters with missing names")
        
        processed_count += 1
        if processed_count % 50 == 0:
            print(f"Processed {processed_count}/{len(json_files)} files...")
    
    # Write results to output file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*60}")
    print(f"Processing complete!")
    print(f"Total files processed: {processed_count}")
    print(f"Files with missing names: {files_with_missing_names}")
    print(f"Total voters with missing names: {sum(len(r['missing_name_serial_numbers']) for r in results)}")
    print(f"\nResults saved to: {output_file}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
