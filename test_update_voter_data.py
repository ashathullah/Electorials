import json
import shutil
from pathlib import Path

def test_update_voter_data():
    """
    Test the update logic on a single file before running on all files.
    Tests with: Tamil Nadu-(S22)_Manachanallur-(AC144)_2.json
    """
    
    # Define test files
    source_file = Path("results/Tamil Nadu-(S22)_Manachanallur-(AC144)_2.json")
    target_file = Path("valid_voter_sections_json/Tamil Nadu-(S22)_Manachanallur-(AC144)_2_voters.json")
    backup_file = Path("valid_voter_sections_json/Tamil Nadu-(S22)_Manachanallur-(AC144)_2_voters.json.backup")
    
    print("="*60)
    print("TEST: Updating Voter Data - Single File")
    print("="*60)
    print(f"\nSource file: {source_file}")
    print(f"Target file: {target_file}")
    
    # Check if files exist
    if not source_file.exists():
        print(f"\n❌ Source file not found: {source_file}")
        return False
    
    if not target_file.exists():
        print(f"\n❌ Target file not found: {target_file}")
        return False
    
    # Create backup
    print(f"\n📦 Creating backup: {backup_file}")
    shutil.copy(target_file, backup_file)
    
    # Load source data
    with open(source_file, 'r', encoding='utf-8') as f:
        source_data = json.load(f)
    
    # Load target data
    with open(target_file, 'r', encoding='utf-8') as f:
        target_data = json.load(f)
    
    print(f"\n📊 Source file contains: {len(source_data)} record(s)")
    print(f"📊 Target file contains: {len(target_data)} records")
    
    # Print sample source record
    print("\n" + "-"*60)
    print("SAMPLE SOURCE RECORD (from /results):")
    print("-"*60)
    sample_source = source_data[0]
    print(json.dumps(sample_source, indent=2, ensure_ascii=False))
    
    # Find matching record in target (serial_no 271)
    target_record_271_before = None
    target_record_271_index = None
    for idx, record in enumerate(target_data):
        if str(record.get("serial_no")) == "271":
            target_record_271_before = record.copy()
            target_record_271_index = idx
            break
    
    if target_record_271_before:
        print("\n" + "-"*60)
        print("BEFORE UPDATE - Target Record (serial_no 271):")
        print("-"*60)
        print(f"Name: '{target_record_271_before.get('name')}'")
        print(f"Father Name: '{target_record_271_before.get('father_name')}'")
        print(f"Husband Name: '{target_record_271_before.get('husband_name')}'")
        print(f"House No: '{target_record_271_before.get('house_no')}'")
    
    # Create lookup for source data
    source_lookup = {}
    for record in source_data:
        serial_no = str(record.get("serial_no", ""))
        source_lookup[serial_no] = record
    
    # Update target data
    records_updated = 0
    for target_record in target_data:
        serial_no = str(target_record.get("serial_no", ""))
        
        if serial_no in source_lookup:
            source_record = source_lookup[serial_no]
            
            # Update name
            if "name" in source_record:
                target_record["name"] = source_record["name"]
            
            # Update house_no
            if "house_no" in source_record:
                target_record["house_no"] = source_record["house_no"]
            
            # Update relation_name based on relation_type
            relation_type = source_record.get("relation_type", "").lower()
            relation_name = source_record.get("relation_name", "")
            
            if relation_type == "father":
                target_record["father_name"] = relation_name
            elif relation_type == "husband":
                target_record["husband_name"] = relation_name
            elif relation_type == "mother":
                target_record["mother_name"] = relation_name
            elif relation_type == "other":
                target_record["other_name"] = relation_name
            
            records_updated += 1
    
    # Show updated record
    target_record_271_after = target_data[target_record_271_index] if target_record_271_index is not None else None
    
    if target_record_271_after:
        print("\n" + "-"*60)
        print("AFTER UPDATE - Target Record (serial_no 271):")
        print("-"*60)
        print(f"Name: '{target_record_271_after.get('name')}'")
        print(f"Father Name: '{target_record_271_after.get('father_name')}'")
        print(f"Husband Name: '{target_record_271_after.get('husband_name')}'")
        print(f"House No: '{target_record_271_after.get('house_no')}'")
        
        # Highlight changes
        print("\n" + "="*60)
        print("CHANGES DETECTED:")
        print("="*60)
        if target_record_271_before.get('name') != target_record_271_after.get('name'):
            print(f"✏️  Name: '{target_record_271_before.get('name')}' → '{target_record_271_after.get('name')}'")
        if target_record_271_before.get('father_name') != target_record_271_after.get('father_name'):
            print(f"✏️  Father Name: '{target_record_271_before.get('father_name')}' → '{target_record_271_after.get('father_name')}'")
        if target_record_271_before.get('house_no') != target_record_271_after.get('house_no'):
            print(f"✏️  House No: '{target_record_271_before.get('house_no')}' → '{target_record_271_after.get('house_no')}'")
    
    # Save updated data (for test purposes)
    print(f"\n💾 Saving updated data to target file...")
    with open(target_file, 'w', encoding='utf-8') as f:
        json.dump(target_data, f, ensure_ascii=False, indent=4)
    
    print(f"\n✅ Test completed! {records_updated} record(s) updated")
    print(f"\n📌 Backup saved at: {backup_file}")
    print(f"📌 To restore backup: copy '{backup_file}' to '{target_file}'")
    
    return True

if __name__ == "__main__":
    print("\n🧪 Running test update...\n")
    success = test_update_voter_data()
    if success:
        print("\n✅ Test passed! You can now run the full update script.")
    else:
        print("\n❌ Test failed! Please check the errors above.")
