
import os

def fix_json_keys(directory):
    # The character identified is U+FEFF (BOM) inside the key
    target = '"\ufeffserial_no":'
    replacement = '"serial_no":'
    
    count = 0
    processed = 0
    
    print(f"Scanning {directory}...")
    
    files = [f for f in os.listdir(directory) if f.endswith(".json")]
    total_files = len(files)
    
    for filename in files:
        filepath = os.path.join(directory, filename)
        processed += 1
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if target in content:
                new_content = content.replace(target, replacement)
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(new_content)
                print(f"Fixed: {filename}")
                count += 1
                
        except Exception as e:
            print(f"Error processing {filename}: {e}")

    print(f"Processing complete.")
    print(f"Total files scanned: {total_files}")
    print(f"Total files fixed: {count}")

if __name__ == "__main__":
    directory = r"e:\Raja_mohaemd\projects\voter-shield-data-cleanup\valid_voter_sections_json"
    if os.path.exists(directory):
        fix_json_keys(directory)
    else:
        print(f"Directory not found: {directory}")
