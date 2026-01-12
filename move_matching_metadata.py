import os
import shutil
from pathlib import Path

def move_matching_metadata():
    """
    Move _metadata.json files from /good_metadata to /metadata 
    by comparing the names with /voters directory.
    
    The idea is to gather available valid metadata and voters in their respective directories
    and later add these data to the database.
    """
    
    # Define directory paths
    base_dir = Path(__file__).parent
    voters_dir = base_dir / "voters"
    good_metadata_dir = base_dir / "good_metadata"
    metadata_dir = base_dir / "metadata"
    
    # Create metadata directory if it doesn't exist
    metadata_dir.mkdir(exist_ok=True)
    
    # Track statistics
    total_voters = 0
    matched_files = 0
    moved_files = 0
    not_found = 0
    
    print(f"Starting metadata file move operation...")
    print(f"Voters directory: {voters_dir}")
    print(f"Source directory (good_metadata): {good_metadata_dir}")
    print(f"Destination directory (metadata): {metadata_dir}")
    print("-" * 80)
    
    # Get all voter files
    voter_files = list(voters_dir.glob("*_voters.json"))
    total_voters = len(voter_files)
    
    print(f"Found {total_voters} voter files in /voters directory\n")
    
    # Process each voter file
    for voter_file in voter_files:
        voter_filename = voter_file.name
        
        # Extract base name by removing "_voters.json" suffix
        # Example: "Tamil Nadu-(S22)_Coimbatore (North)-(AC118)_1_voters.json"
        # becomes: "Tamil Nadu-(S22)_Coimbatore (North)-(AC118)_1"
        base_name = voter_filename.replace("_voters.json", "")
        
        # Construct expected metadata filename
        metadata_filename = f"{base_name}_metadata.json"
        metadata_source_path = good_metadata_dir / metadata_filename
        metadata_dest_path = metadata_dir / metadata_filename
        
        # Check if matching metadata file exists
        if metadata_source_path.exists():
            matched_files += 1
            
            try:
                # Move the file
                shutil.move(str(metadata_source_path), str(metadata_dest_path))
                moved_files += 1
                print(f"[OK] Moved: {metadata_filename}")
            except Exception as e:
                print(f"[ERROR] Error moving {metadata_filename}: {str(e)}")
        else:
            not_found += 1
            print(f"[MISSING] Not found in good_metadata: {metadata_filename}")
    
    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total voter files found:           {total_voters}")
    print(f"Matching metadata files found:     {matched_files}")
    print(f"Successfully moved files:          {moved_files}")
    print(f"Metadata files not found:          {not_found}")
    print(f"Remaining files in good_metadata:  {len(list(good_metadata_dir.glob('*_metadata.json')))}")
    print(f"Total files in metadata directory: {len(list(metadata_dir.glob('*_metadata.json')))}")
    print("=" * 80)
    
    return {
        'total_voters': total_voters,
        'matched_files': matched_files,
        'moved_files': moved_files,
        'not_found': not_found
    }

if __name__ == "__main__":
    results = move_matching_metadata()
    print("\nOperation completed successfully!")
