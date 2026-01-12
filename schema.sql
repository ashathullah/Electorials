-- Database Schema for Electoral Roll Processing
-- Run this in your PostgreSQL database to create the necessary tables manually
-- (Note: The application will attempt to create these automatically if they don't exist)

-- 1. Metadata_stage Table
-- Stores document-level information extracted from the PDF
CREATE TABLE IF NOT EXISTS metadata_stage (
    document_id TEXT PRIMARY KEY,
    pdf_name TEXT NOT NULL,
    state TEXT,
    year INTEGER,
    revision_type TEXT,
    qualifying_date TEXT,
    publication_date TEXT,
    roll_type TEXT,
    roll_identification TEXT,
    total_pages INTEGER,
    total_voters_extracted INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    -- Main administrative fields (promoted from JSONB for easier access)
    town_or_village TEXT,
    main_town_or_village TEXT,
    ward_number TEXT,
    post_office TEXT,
    police_station TEXT,
    taluk_or_block TEXT,
    subdivision TEXT,
    district TEXT,
    pin_code TEXT,
    panchayat_name TEXT,

    -- Nested structures stored as JSONB for flexibility
    constituency_details JSONB DEFAULT '{}',
    administrative_address JSONB DEFAULT '{}',
    polling_details JSONB DEFAULT '{}',
    detailed_elector_summary JSONB DEFAULT '{}',
    authority_verification JSONB DEFAULT '{}',
    output_identifier TEXT
);

-- 2. Voters_stage Table
-- Stores individual voter records linked to metadata_stage
CREATE TABLE IF NOT EXISTS voters_stage (
    id TEXT PRIMARY KEY,
    document_id TEXT REFERENCES metadata_stage(document_id),
    serial_no TEXT,
    epic_no TEXT,
    name TEXT,
    
    -- Relation fields
    relation_type TEXT,
    relation_name TEXT,
    father_name TEXT,
    mother_name TEXT,
    husband_name TEXT,
    other_name TEXT,
    
    -- Address/Details
    house_no TEXT,
    age TEXT,
    gender TEXT,
    street_names_and_numbers TEXT,
    part_no TEXT,
    assembly TEXT,
    
    -- Flags & duplication details
    critical_flag INTEGER,
    non_critical_flag INTEGER,
    duplication_flag TEXT,
    flag_details TEXT,
    duplication_details TEXT,
    
    -- Metadata_stage fields
    page_id TEXT,
    sequence_in_page INTEGER,
    epic_valid BOOLEAN,
    deleted TEXT,  -- Empty string = not deleted, 'true' = deleted
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);


-- 3. Indexes
-- Improve query performance
CREATE INDEX IF NOT EXISTS idx_voters_stage_document_id ON voters_stage(document_id);
CREATE INDEX IF NOT EXISTS idx_voters_stage_epic_no ON voters_stage(epic_no);
