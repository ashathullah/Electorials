# Database Population Script - Documentation

## Overview
This script populates the `metadata_stage` and `voters_stage` tables from JSON files in the `/metadata` and `/voters` directories.

## Features

### ✅ UPSERT Logic (Prevents Duplicates)
- Uses PostgreSQL's `ON CONFLICT ... DO UPDATE` clause
- If a record already exists (based on primary key), it **updates** the existing record
- If you accidentally run this script multiple times, it will NOT create duplicate entries
- Existing voter UUIDs are preserved when updating records

### ✅ Automatic Processing
- Processes all files in batch mode without manual intervention
- Automatically commits changes after each file is successfully processed
- Continues to next file if an error occurs
- No confirmation prompts - just run and let it process

### ✅ UUID-Based Voter IDs
- Each voter gets a unique UUID identifier (e.g., `d30875a7-6d80-4335-83ad-51dd2aee1ec4`)
- UUIDs are preserved when re-processing existing voters
- Ensures globally unique identifiers across the database

### ✅ Detailed Logging
- Shows progress for each file being processed
- Displays the number of rows inserted/updated for:
  - Metadata table (1 row per file)
  - Voters table (N rows per file, where N = number of voters)
- Final summary with total row counts

### ✅ Transaction Safety
- Each file is processed in its own transaction
- If an error occurs, the transaction is rolled back
- Successful files are committed automatically

## Data Mapping

### Metadata Table (`metadata_stage`)
- **document_id**: Filename without `_metadata.json` suffix
- **pdf_name**: Same as document_id
- **total_voters_stage_extracted**: Count of voters in corresponding `_voters.json` file
- All other fields are mapped from the JSON structure

Example:
```
File: Tamil Nadu-(S22)_Coimbatore (North)-(AC118)_1_metadata.json
→ document_id: Tamil Nadu-(S22)_Coimbatore (North)-(AC118)_1
→ pdf_name: Tamil Nadu-(S22)_Coimbatore (North)-(AC118)_1
```

### Voters Table (`voters_stage`)
- **id**: UUID (e.g., `d30875a7-6d80-4335-83ad-51dd2aee1ec4`)
  - New voters get a randomly generated UUID
  - Existing voters preserve their UUID when updated
- **document_id**: Filename without `_voters.json` suffix
- All voter fields are mapped from the JSON array

Example:
```
File: Tamil Nadu-(S22)_Coimbatore (North)-(AC118)_1_voters.json
→ document_id: Tamil Nadu-(S22)_Coimbatore (North)-(AC118)_1
→ For voter with serial_no "1":
   id: d30875a7-6d80-4335-83ad-51dd2aee1ec4 (random UUID)
```

## Usage

### Prerequisites
1. PostgreSQL database must be running
2. Database credentials in `.env` file:
   ```
   DB_HOST=localhost
   DB_PORT=5432
   DB_NAME=voter_shield
   DB_USER=postgres
   DB_PASSWORD=your_password
   ```
3. Schema tables must exist (run `schema.sql` first)

### Running the Script
```bash
python populate_database.py
```

### Script Flow
1. Scans `/metadata` folder for `*_metadata.json` files
2. For each metadata file:
   - Extracts document ID from filename
   - Loads metadata JSON
   - Finds corresponding voters JSON in `/voters` folder
   - Counts voters
   - Inserts/updates metadata record (1 row)
   - Inserts/updates voter records (N rows)
   - **Automatically commits changes**
   - Proceeds to next file

### Example Output
```
================================================================================
🚀 VOTER SHIELD DATABASE POPULATION SCRIPT
================================================================================
📁 Found 72 metadata files to process
================================================================================

✅ Database connection established

────────────────────────────────────────────────────────────────────────────────
📄 Processing file 1/72: Tamil Nadu-(S22)_Coimbatore (North)-(AC118)_1_metadata.json
────────────────────────────────────────────────────────────────────────────────
📋 Document ID: Tamil Nadu-(S22)_Coimbatore (North)-(AC118)_1
👥 Total voters found: 890
✅ Metadata: 1 row(s) inserted/updated
✅ Voters: 890 row(s) inserted/updated
✅ Changes committed successfully!

────────────────────────────────────────────────────────────────────────────────
📄 Processing file 2/72: Tamil Nadu-(S22)_Manachanallur-(AC144)_100_metadata.json
────────────────────────────────────────────────────────────────────────────────
📋 Document ID: Tamil Nadu-(S22)_Manachanallur-(AC144)_100
👥 Total voters found: 1023
✅ Metadata: 1 row(s) inserted/updated
✅ Voters: 1023 row(s) inserted/updated
✅ Changes committed successfully!

...

================================================================================
🏁 PROCESSING COMPLETE
================================================================================
✅ Successfully processed: 72 files
⏭️  Skipped: 0 files
📁 Total files: 72
📊 Total metadata rows inserted/updated: 72
📊 Total voter rows inserted/updated: 64230
================================================================================
```

## Safety Features

### 1. No Duplicates
If you run the script again on the same files, it will **update** existing records instead of creating duplicates. Voter UUIDs are preserved when updating.

### 2. Transaction Rollback
If an error occurs during processing of a file, all changes for that file are rolled back. Successfully processed files remain committed.

### 3. Graceful Error Handling
- If a metadata file is missing its corresponding voters file, the script continues with metadata only (0 voters)
- If an error occurs processing a file, the script logs the error and continues to the next file
- Press Ctrl+C at any time to safely exit (current transaction will be rolled back)

### 4. UUID Preservation
When re-running the script on existing data, voter UUIDs are preserved to maintain referential integrity across your database.

## Troubleshooting

### Database Connection Error
```
❌ Failed to connect to database: ...
```
**Solution**: Check your `.env` file and ensure PostgreSQL is running.

### File Not Found Error
```
⚠️  Warning: Voters file not found: ...
```
**Solution**: Ensure corresponding `_voters.json` file exists in `/voters` directory, or choose to continue without it.

### Constraint Violation
```
❌ Error processing file: ... violates foreign key constraint
```
**Solution**: Ensure the schema is properly created with the correct constraints.

## Database Schema Reference

The script expects these tables to exist:

```sql
-- Metadata table
CREATE TABLE metadata_stage (
    document_id TEXT PRIMARY KEY,
    pdf_name TEXT NOT NULL,
    -- ... other columns
);

-- Voters table
CREATE TABLE voters_stage (
    id TEXT PRIMARY KEY,
    document_id TEXT REFERENCES metadata_stage(document_id),
    -- ... other columns
);
```

Run `schema.sql` before using this script if tables don't exist.
