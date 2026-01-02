import json
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path.cwd()))

from src.persistence.json_store import JSONStore

def patch_and_export():
    pdf_name = "tamil_removed"
    store = JSONStore(Path("extracted"))
    
    # 1. Load the main document
    doc_path = store._get_output_dir(pdf_name) / f"{pdf_name}.json"
    if not doc_path.exists():
        print("Document not found")
        return

    with open(doc_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # 2. Load the metadata file
    meta_path = store._get_output_dir(pdf_name) / f"{pdf_name}-metadata.json"
    if not meta_path.exists():
        print("Metadata file not found")
        return
        
    with open(meta_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
        
    # 3. Patch the metadata into the main data
    print("Patching metadata into document...")
    data["metadata"] = metadata
    
    # 4. Save the patched document back (optional, but good for consistency)
    # verify we aren't overwriting with null again
    if data["metadata"] is None:
        print("Error: Loaded metadata is None!")
        return

    with open(doc_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        
    # 5. Export to CSV
    print("Exporting to CSV...")
    paths = store.save_to_csv(data)
    print(f"Exported CSVs to: {[str(p) for p in paths]}")

if __name__ == "__main__":
    try:
        patch_and_export()
    except Exception as e:
        print(f"Error: {e}")
