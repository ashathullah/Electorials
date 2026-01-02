import sys
from pathlib import Path
import traceback

sys.path.append(str(Path.cwd()))

# Force UTF-8 for log file
def log(msg):
    try:
        with open("test_log.txt", "a", encoding="utf-8") as f:
            f.write(str(msg) + "\n")
    except Exception:
        pass

try:
    with open("test_log.txt", "w") as f: f.write("Init\n")
    log("Starting test...")
    from src.persistence.json_store import JSONStore
    log("Imported JSONStore")
    
    store = JSONStore(Path("extracted"))
    pdf_name = "tamil_removed"
    
    log(f"Loading {pdf_name}")
    doc = store.load_document(pdf_name)
    if doc:
        log("Doc loaded, calling save_to_csv")
        # Ensure 'doc' is treated as dict if it is one (load_document returns dict)
        paths = store.save_to_csv(doc)
        log(f"Success! Paths: {paths}")
    else:
        log("Doc not found")

except Exception as e:
    log(f"EXCEPTION: {e}")
    log(traceback.format_exc())
