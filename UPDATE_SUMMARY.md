# Voter Data Update Script - Summary

## Overview
This script updates voter data from the `/results` folder into the `/valid_voter_sections_json` folder.

## What I Understood

### Source Data Location
- **Folder**: `/results`
- **File naming**: `State_Assembly_PartNo.json` (e.g., `Tamil Nadu-(S22)_Manachanallur-(AC144)_2.json`)

### Target Data Location
- **Folder**: `/valid_voter_sections_json`
- **File naming**: `State_Assembly_PartNo_voters.json` (e.g., `Tamil Nadu-(S22)_Manachanallur-(AC144)_2_voters.json`)

### Update Logic
The script:
1. Loops through each JSON file in `/results`
2. Finds corresponding file in `/valid_voter_sections_json` by adding `_voters` suffix
3. Matches voters by `serial_no` field
4. Updates the following fields from source to target:
   - **name** → `name`
   - **house_no** → `house_no`
   - **relation_name** → maps to appropriate field based on `relation_type`:
     - If `relation_type` = "father" → updates `father_name`
     - If `relation_type` = "husband" → updates `husband_name`
     - If `relation_type` = "mother" → updates `mother_name`
     - If `relation_type` = "other" → updates `other_name`

## Files Created

### 1. `update_voter_data.py`
- Main script that processes all files
- Provides detailed statistics and error reporting
- Updates all matching records in bulk

### 2. `test_update_voter_data.py`
- Test script that processes only ONE file for verification
- Shows before/after comparison
- Creates automatic backup before updating
- Safe to run for testing

## Test Results ✅

**Test file**: `Tamil Nadu-(S22)_Manachanallur-(AC144)_2.json`

**Sample update (serial_no 271)**:
- **Before**: 
  - name: "" (empty)
  - father_name: "நாராயணசாமி"
  - house_no: "4/92"

- **After**:
  - name: "சிவஞானம்" ✅ (updated from source)
  - father_name: "நாராயணசாமி" ✅ (matched)
  - house_no: "4/92" ✅ (confirmed)

**Backup created**: `Tamil Nadu-(S22)_Manachanallur-(AC144)_2_voters.json.backup`

## How to Use

### To test on a single file:
```bash
python test_update_voter_data.py
```

### To run full update on all files:
```bash
python update_voter_data.py
```

## Statistics Expected (from /results folder):
- Total files in `/results`: **633 files**
- Files to be updated in `/valid_voter_sections_json`: depends on matching filenames

## Safety Features

1. **Backup**: Test script creates `.backup` files
2. **Validation**: Only updates records with matching `serial_no`
3. **Statistics**: Provides detailed count of:
   - Files processed
   - Files not found
   - Records updated
   - Records not matched
   - Errors encountered

## Next Steps

1. ✅ **Test completed successfully** - verified with one file
2. ⏳ **Ready to execute full update** - await your confirmation
3. 📊 **Review statistics** - after full execution

## Restore Instructions (if needed)

To restore the test file from backup:
```bash
Copy-Item "valid_voter_sections_json\Tamil Nadu-(S22)_Manachanallur-(AC144)_2_voters.json.backup" "valid_voter_sections_json\Tamil Nadu-(S22)_Manachanallur-(AC144)_2_voters.json"
```
