import os
import shutil
import sys
from pathlib import Path

def clean_directory(target_dir):
    """
    Cleans all contents of the target directory except the 'output' folder.
    """
    # Check if absolute path was passed or relative
    target_path = Path(target_dir)
    
    # If path doesn't exist, try looking in 'extracted' directory relative to script
    if not target_path.exists():
        potential_path = Path("extracted") / target_dir
        if potential_path.exists():
            target_path = potential_path
        else:
            print(f"Error: Directory '{target_dir}' not found.")
            return

    print(f"Cleaning directory: {target_path}")
    
    if not target_path.is_dir():
        print(f"Error: '{target_path}' is not a directory.")
        return

    for item in target_path.iterdir():
        if item.name == "output":
            print(f"Skipping preserved folder: {item.name}")
            continue
        
        try:
            if item.is_file() or item.is_symlink():
                item.unlink()
                print(f"Deleted file: {item.name}")
            elif item.is_dir():
                shutil.rmtree(item)
                print(f"Deleted directory: {item.name}")
        except Exception as e:
            print(f"Failed to delete {item.name}. Reason: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python clean_extracted.py <folder_name_or_path>")
        print("Example: python clean_extracted.py \"Tamil Nadu-(S22)_Tiruppur (South)-(AC114)_1\"")
        print("         python clean_extracted.py all  <-- To clean ALL subdirectories in 'extracted'")
        sys.exit(1)

    input_arg = sys.argv[1]

    if input_arg.lower() == "all":
        base_path = Path("extracted")
        if not base_path.exists():
            print("Error: 'extracted' directory not found in current location.")
            sys.exit(1)
            
        print("Cleaning ALL directories in 'extracted'...")
        for sub_dir in base_path.iterdir():
            if sub_dir.is_dir():
                clean_directory(sub_dir)
    else:
        clean_directory(input_arg)
