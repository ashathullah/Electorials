import os
import csv
import json

def convert_csv_to_json(source_dir, dest_dir):
    """
    Converts all CSV files in the source directory to JSON files in the destination directory.
    Uses the first line of each CSV as the header/structure.
    """
    
    # Create destination directory if it doesn't exist
    if not os.path.exists(dest_dir):
        print(f"Creating destination directory: {dest_dir}")
        os.makedirs(dest_dir)

    # List all files in the source directory
    files = os.listdir(source_dir)
    
    # Filter for CSV files
    csv_files = [f for f in files if f.endswith('.csv')]
    
    print(f"Found {len(csv_files)} CSV files to process.")

    for filename in csv_files:
        source_path = os.path.join(source_dir, filename)
        dest_filename = filename.replace('.csv', '.json')
        dest_path = os.path.join(dest_dir, dest_filename)

        try:
            data = []
            with open(source_path, mode='r', encoding='utf-8') as csv_file:
                # Use DictReader to automatically iterate over the CSV using the first line as keys
                csv_reader = csv.DictReader(csv_file)
                
                # Convert each row to a dictionary and add to data list
                for row in csv_reader:
                    data.append(row)
            
            # Write the data to a JSON file
            with open(dest_path, mode='w', encoding='utf-8') as json_file:
                json.dump(data, json_file, indent=4, ensure_ascii=False)
            
            print(f"Converted {filename} to {dest_filename}")

        except Exception as e:
            print(f"Error processing {filename}: {e}")

if __name__ == "__main__":
    # Define source and destination directories relative to the project root or absolute
    # Using absolute paths based on the context provided
    SOURCE_DIR = r"e:\Raja_mohaemd\projects\voter-shield-data-cleanup\extraction_results"
    DEST_DIR = r"e:\Raja_mohaemd\projects\voter-shield-data-cleanup\valid_voter_sections_json"

    convert_csv_to_json(SOURCE_DIR, DEST_DIR)
