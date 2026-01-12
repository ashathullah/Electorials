import os
import shutil
from pathlib import Path

def move_metadata_csvs():
    """
    Move all *_metadata.csv files from extraction_results to metadata_csvs directory.
    """
    # Define source and destination directories
    source_dir = Path("extraction_results")
    dest_dir = Path("metadata_csvs")
    
    # Create destination directory if it doesn't exist
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if source directory exists
    if not source_dir.exists():
        print(f"❌ Source directory '{source_dir}' does not exist!")
        return
    
    # Find all *_metadata.csv files
    metadata_files = list(source_dir.glob("*_metadata.csv"))
    
    if not metadata_files:
        print(f"ℹ️  No *_metadata.csv files found in '{source_dir}'")
        return
    
    print(f"📁 Found {len(metadata_files)} metadata CSV file(s)")
    
    # Move each file
    moved_count = 0
    for file_path in metadata_files:
        try:
            dest_path = dest_dir / file_path.name
            
            # Check if file already exists in destination
            if dest_path.exists():
                print(f"⚠️  Skipping '{file_path.name}' - already exists in destination")
                continue
            
            # Move the file
            shutil.move(str(file_path), str(dest_path))
            print(f"✅ Moved: {file_path.name}")
            moved_count += 1
            
        except Exception as e:
            print(f"❌ Error moving '{file_path.name}': {e}")
    
    print(f"\n🎉 Successfully moved {moved_count} file(s) to '{dest_dir}'")

if __name__ == "__main__":
    move_metadata_csvs()
