
import csv
import os
import glob
import sys
import io

# Force utf-8 for stdout
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

SOURCE_DIR = r"e:\Raja_mohaemd\projects\voter-shield-data-cleanup\invalid_voter_sections"

def diagnose():
    print(f"Scanning {SOURCE_DIR}...")
    csv_files = glob.glob(os.path.join(SOURCE_DIR, "*_voters.csv"))
    print(f"Found {len(csv_files)} files.")

    stats = {
        "total": 0,
        "empty_file": 0,
        "only_header": 0,
        "malformed_row": 0,
        "empty_epic_only": 0,
        "empty_name_only": 0,
        "empty_both": 0,
        "valid": 0,
        "read_error": 0
    }

    suspicious_samples = []

    for file_path in csv_files:
        stats["total"] += 1
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                header = next(reader, None)
                if not header:
                    stats["empty_file"] += 1
                    continue
                
                first_record = next(reader, None)
                if not first_record:
                    stats["only_header"] += 1
                    continue
                
                if len(first_record) < 3:
                    stats["malformed_row"] += 1
                    if len(suspicious_samples) < 10:
                        suspicious_samples.append((os.path.basename(file_path), "MALFORMED", first_record))
                    continue

                epic = first_record[1].strip()
                name = first_record[2].strip()

                is_epic_empty = not epic
                is_name_empty = not name

                if is_epic_empty and is_name_empty:
                    stats["empty_both"] += 1
                    if len(suspicious_samples) < 10:
                        suspicious_samples.append((os.path.basename(file_path), "BOTH_EMPTY", first_record))
                elif is_epic_empty:
                    stats["empty_epic_only"] += 1
                    if len(suspicious_samples) < 10:
                        suspicious_samples.append((os.path.basename(file_path), "EPIC_EMPTY", first_record))
                elif is_name_empty:
                    stats["empty_name_only"] += 1
                    if len(suspicious_samples) < 10:
                        suspicious_samples.append((os.path.basename(file_path), "NAME_EMPTY", first_record))
                else:
                    stats["valid"] += 1

        except Exception as e:
            stats["read_error"] += 1
            print(f"Error reading {file_path}: {e}")

    with open("diagnostic_output.txt", "w", encoding="utf-8") as f:
        f.write(f"Scanning {SOURCE_DIR}...\n")
        f.write(f"Found {len(csv_files)} files.\n")

        f.write("\n--- Diagnosis Report ---\n")
        for k, v in stats.items():
            f.write(f"{k}: {v}\n")
        
        f.write("\n--- Suspicious Samples (First 10) ---\n")
        for sample in suspicious_samples:
            f.write(f"{sample}\n")
            
    print("Diagnosis complete. usage written to diagnostic_output.txt")

if __name__ == "__main__":
    diagnose()
