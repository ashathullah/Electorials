"""
Database Population Script for Voter Shield Data
This script populates the metadata_stage and voters_stage tables from JSON files
in the /metadata and /voters directories.
"""

import os
import json
import psycopg2
from psycopg2.extras import execute_values
from dotenv import load_dotenv
import sys
import uuid

# Load environment variables
load_dotenv()

# Database connection parameters
DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': os.getenv('DB_PORT', '5432'),
    'database': os.getenv('DB_NAME', 'voter_shield'),
    'user': os.getenv('DB_USER', 'postgres'),
    'password': os.getenv('DB_PASSWORD', '')
}

# Directory paths
METADATA_DIR = os.path.join(os.path.dirname(__file__), 'metadata')
VOTERS_DIR = os.path.join(os.path.dirname(__file__), 'voters')


def get_document_id_from_filename(filename, suffix):
    """
    Extract document ID from filename by removing the suffix
    E.g., 'Tamil Nadu-(S22)_Coimbatore (North)-(AC118)_1_metadata.json' 
    -> 'Tamil Nadu-(S22)_Coimbatore (North)-(AC118)_1'
    """
    if filename.endswith(suffix):
        return filename[:-len(suffix)]
    return filename


def load_json_file(filepath):
    """Load and parse JSON file"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"❌ Error loading {filepath}: {e}")
        return None


def insert_metadata(conn, document_id, metadata_json, total_voters):
    """
    Insert or update metadata record into metadata_stage table
    Uses UPSERT to prevent duplicates
    """
    cursor = conn.cursor()
    
    # Extract nested structures
    doc_meta = metadata_json.get('document_metadata', {})
    constituency = metadata_json.get('constituency_details', {})
    admin_address = metadata_json.get('administrative_address', {})
    polling = metadata_json.get('part_and_polling_details', {})
    elector_summary = metadata_json.get('detailed_elector_summary', {})
    auth_verification = metadata_json.get('authority_verification', {})
    
    # SQL UPSERT query
    query = """
    INSERT INTO metadata_stage (
        document_id, pdf_name, state, year, revision_type, qualifying_date,
        publication_date, roll_type, roll_identification, total_pages,
        total_voters_extracted, town_or_village, main_town_or_village,
        ward_number, post_office, police_station, taluk_or_block, subdivision,
        district, pin_code, panchayat_name, constituency_details,
        administrative_address, polling_details, detailed_elector_summary,
        authority_verification, output_identifier
    ) VALUES (
        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 
        %s, %s, %s, %s, %s, %s, %s, %s, %s
    )
    ON CONFLICT (document_id) 
    DO UPDATE SET
        pdf_name = EXCLUDED.pdf_name,
        state = EXCLUDED.state,
        year = EXCLUDED.year,
        revision_type = EXCLUDED.revision_type,
        qualifying_date = EXCLUDED.qualifying_date,
        publication_date = EXCLUDED.publication_date,
        roll_type = EXCLUDED.roll_type,
        roll_identification = EXCLUDED.roll_identification,
        total_pages = EXCLUDED.total_pages,
        total_voters_extracted = EXCLUDED.total_voters_extracted,
        town_or_village = EXCLUDED.town_or_village,
        main_town_or_village = EXCLUDED.main_town_or_village,
        ward_number = EXCLUDED.ward_number,
        post_office = EXCLUDED.post_office,
        police_station = EXCLUDED.police_station,
        taluk_or_block = EXCLUDED.taluk_or_block,
        subdivision = EXCLUDED.subdivision,
        district = EXCLUDED.district,
        pin_code = EXCLUDED.pin_code,
        panchayat_name = EXCLUDED.panchayat_name,
        constituency_details = EXCLUDED.constituency_details,
        administrative_address = EXCLUDED.administrative_address,
        polling_details = EXCLUDED.polling_details,
        detailed_elector_summary = EXCLUDED.detailed_elector_summary,
        authority_verification = EXCLUDED.authority_verification,
        output_identifier = EXCLUDED.output_identifier,
        updated_at = CURRENT_TIMESTAMP
    """
    
    # Extract values
    values = (
        document_id,  # document_id
        document_id,  # pdf_name (same as document_id)
        doc_meta.get('state'),
        doc_meta.get('electoral_roll_year'),
        doc_meta.get('revision_type'),
        doc_meta.get('qualifying_date'),
        doc_meta.get('publication_date'),
        doc_meta.get('roll_type'),
        doc_meta.get('roll_identification'),
        doc_meta.get('total_pages'),
        total_voters,  # total_voters_extracted
        admin_address.get('town_or_village'),
        admin_address.get('main_town_or_village'),
        str(admin_address.get('ward_number')) if admin_address.get('ward_number') else None,
        admin_address.get('post_office'),
        admin_address.get('police_station'),
        admin_address.get('taluk_or_block'),
        admin_address.get('subdivision'),
        admin_address.get('district'),
        str(admin_address.get('pin_code')) if admin_address.get('pin_code') else None,
        admin_address.get('panchayat_name'),
        json.dumps(constituency, ensure_ascii=False),
        json.dumps(admin_address, ensure_ascii=False),
        json.dumps(polling, ensure_ascii=False),
        json.dumps(elector_summary, ensure_ascii=False),
        json.dumps(auth_verification, ensure_ascii=False),
        metadata_json.get('ai_metadata', {}).get('model')
    )
    
    cursor.execute(query, values)
    return cursor.rowcount


def insert_voters(conn, document_id, voters_json):
    """
    Insert or update voter records into voters_stage table
    Uses UPSERT to prevent duplicates
    """
    if not voters_json or not isinstance(voters_json, list):
        return 0
    
    cursor = conn.cursor()
    
    # First, we need to check if voters already exist to preserve their UUIDs
    # Get existing voter UUIDs for this document
    cursor.execute("""
        SELECT serial_no, id FROM voters_stage 
        WHERE document_id = %s
    """, (document_id,))
    existing_voters = {row[0]: row[1] for row in cursor.fetchall()}
    
    # Prepare data for batch insert
    voter_records = []
    for voter in voters_json:
        # Determine relation type and names
        relation_type = None
        relation_name = None
        
        if voter.get('father_name'):
            relation_type = 'Father'
            relation_name = voter.get('father_name')
        elif voter.get('husband_name'):
            relation_type = 'Husband'
            relation_name = voter.get('husband_name')
        elif voter.get('mother_name'):
            relation_type = 'Mother'
            relation_name = voter.get('mother_name')
        elif voter.get('other_name'):
            relation_type = 'Other'
            relation_name = voter.get('other_name')
        
        # Use existing UUID if voter exists, otherwise generate new UUID
        serial_no = voter.get('serial_no')
        if serial_no in existing_voters:
            voter_id = existing_voters[serial_no]
        else:
            voter_id = str(uuid.uuid4())
        
        voter_record = (
            voter_id,  # id (UUID)
            document_id,  # document_id
            voter.get('serial_no'),
            voter.get('epic_id'),
            voter.get('name'),
            relation_type,
            relation_name,
            voter.get('father_name'),
            voter.get('mother_name'),
            voter.get('husband_name'),
            voter.get('other_name'),
            voter.get('house_no'),
            voter.get('age'),
            voter.get('gender'),
            voter.get('street_names_and_numbers'),
            voter.get('part_no'),
            voter.get('assembly'),
            None,  # critical_flag
            None,  # non_critical_flag
            None,  # duplication_flag
            None,  # flag_details
            None,  # duplication_details
            None,  # page_id
            None,  # sequence_in_page
            None,  # epic_valid
            voter.get('deleted', '')  # deleted
        )
        voter_records.append(voter_record)
    
    # UPSERT query
    query = """
    INSERT INTO voters_stage (
        id, document_id, serial_no, epic_no, name, relation_type, relation_name,
        father_name, mother_name, husband_name, other_name, house_no, age, gender,
        street_names_and_numbers, part_no, assembly, critical_flag, non_critical_flag,
        duplication_flag, flag_details, duplication_details, page_id, sequence_in_page,
        epic_valid, deleted
    ) VALUES %s
    ON CONFLICT (id) 
    DO UPDATE SET
        document_id = EXCLUDED.document_id,
        serial_no = EXCLUDED.serial_no,
        epic_no = EXCLUDED.epic_no,
        name = EXCLUDED.name,
        relation_type = EXCLUDED.relation_type,
        relation_name = EXCLUDED.relation_name,
        father_name = EXCLUDED.father_name,
        mother_name = EXCLUDED.mother_name,
        husband_name = EXCLUDED.husband_name,
        other_name = EXCLUDED.other_name,
        house_no = EXCLUDED.house_no,
        age = EXCLUDED.age,
        gender = EXCLUDED.gender,
        street_names_and_numbers = EXCLUDED.street_names_and_numbers,
        part_no = EXCLUDED.part_no,
        assembly = EXCLUDED.assembly,
        critical_flag = EXCLUDED.critical_flag,
        non_critical_flag = EXCLUDED.non_critical_flag,
        duplication_flag = EXCLUDED.duplication_flag,
        flag_details = EXCLUDED.flag_details,
        duplication_details = EXCLUDED.duplication_details,
        page_id = EXCLUDED.page_id,
        sequence_in_page = EXCLUDED.sequence_in_page,
        epic_valid = EXCLUDED.epic_valid,
        deleted = EXCLUDED.deleted
    """
    
    execute_values(cursor, query, voter_records)
    return len(voter_records)


def process_files():
    """Main processing function"""
    # Get list of metadata files
    metadata_files = [f for f in os.listdir(METADATA_DIR) if f.endswith('_metadata.json')]
    total_files = len(metadata_files)
    
    print(f"\n{'='*80}")
    print(f"🚀 VOTER SHIELD DATABASE POPULATION SCRIPT")
    print(f"{'='*80}")
    print(f"📁 Found {total_files} metadata files to process")
    print(f"{'='*80}\n")
    
    if total_files == 0:
        print("❌ No metadata files found. Exiting.")
        return
    
    # Connect to database
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        conn.autocommit = False  # Use manual transactions
        print("✅ Database connection established\n")
    except Exception as e:
        print(f"❌ Failed to connect to database: {e}")
        return
    
    processed_count = 0
    skipped_count = 0
    total_metadata_rows = 0
    total_voter_rows = 0
    
    try:
        for idx, metadata_file in enumerate(metadata_files, 1):
            print(f"\n{'─'*80}")
            print(f"📄 Processing file {idx}/{total_files}: {metadata_file}")
            print(f"{'─'*80}")
            
            # Extract document ID
            document_id = get_document_id_from_filename(metadata_file, '_metadata.json')
            print(f"📋 Document ID: {document_id}")
            
            # Load metadata JSON
            metadata_path = os.path.join(METADATA_DIR, metadata_file)
            metadata_json = load_json_file(metadata_path)
            
            if not metadata_json:
                print(f"⚠️  Skipping - could not load metadata file")
                skipped_count += 1
                continue
            
            # Find corresponding voters file
            voters_file = f"{document_id}_voters.json"
            voters_path = os.path.join(VOTERS_DIR, voters_file)
            
            if not os.path.exists(voters_path):
                print(f"⚠️  Warning: Voters file not found: {voters_file}")
                print(f"⏭️  Continuing with metadata only (0 voters)")
                voters_json = []
                total_voters = 0
            else:
                # Load voters JSON
                voters_json = load_json_file(voters_path)
                if not voters_json:
                    print(f"⚠️  Skipping - could not load voters file")
                    skipped_count += 1
                    continue
                total_voters = len(voters_json) if isinstance(voters_json, list) else 0
                print(f"👥 Total voters found: {total_voters}")
            
            # Start transaction for this file
            try:
                # Insert metadata
                metadata_rows = insert_metadata(conn, document_id, metadata_json, total_voters)
                print(f"✅ Metadata: {metadata_rows} row(s) inserted/updated")
                
                # Insert voters
                voter_rows = insert_voters(conn, document_id, voters_json)
                print(f"✅ Voters: {voter_rows} row(s) inserted/updated")
                
                # Commit automatically
                conn.commit()
                processed_count += 1
                total_metadata_rows += metadata_rows
                total_voter_rows += voter_rows
                print(f"✅ Changes committed successfully!")
                    
            except Exception as e:
                print(f"❌ Error processing file: {e}")
                conn.rollback()
                print(f"↩️  Transaction rolled back")
                skipped_count += 1
    
    finally:
        conn.close()
        print(f"\n{'='*80}")
        print(f"🏁 PROCESSING COMPLETE")
        print(f"{'='*80}")
        print(f"✅ Successfully processed: {processed_count} files")
        print(f"⏭️  Skipped: {skipped_count} files")
        print(f"📁 Total files: {total_files}")
        print(f"📊 Total metadata rows inserted/updated: {total_metadata_rows}")
        print(f"📊 Total voter rows inserted/updated: {total_voter_rows}")
        print(f"{'='*80}\n")


if __name__ == "__main__":
    try:
        process_files()
    except KeyboardInterrupt:
        print("\n\n⚠️  Script interrupted by user. Exiting...")
        sys.exit(0)
