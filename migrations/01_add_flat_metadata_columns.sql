-- Migration: Add Flat Metadata Columns
-- Date: 2026-01-15
-- Description: Adds language_detected and total columns for simplified flat metadata structure

-- Step 1: Add new columns if they don't exist
ALTER TABLE metadata 
ADD COLUMN IF NOT EXISTS language_detected JSONB DEFAULT '[]';

ALTER TABLE metadata 
ADD COLUMN IF NOT EXISTS total INTEGER;

-- Step 2: Migrate existing data from nested structure to flat columns
-- Migrate language_detected from document_metadata.language_detected
UPDATE metadata 
SET language_detected = COALESCE(
    detailed_elector_summary -> 'document_metadata' -> 'language_detected',
    '[]'::jsonb
)
WHERE language_detected = '[]'::jsonb OR language_detected IS NULL;

-- Migrate total from detailed_elector_summary.net_total.total
UPDATE metadata 
SET total = (detailed_elector_summary -> 'net_total' ->> 'total')::int
WHERE total IS NULL 
  AND detailed_elector_summary -> 'net_total' ->> 'total' IS NOT NULL;

-- Step 3: Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_metadata_total ON metadata(total);
CREATE INDEX IF NOT EXISTS idx_metadata_language_detected ON metadata USING GIN(language_detected);

-- Step 4: Verify migration
-- Run this to check how many rows were migrated:
-- SELECT 
--     COUNT(*) as total_rows,
--     COUNT(language_detected) as with_language,
--     COUNT(total) as with_total,
--     COUNT(CASE WHEN language_detected != '[]'::jsonb THEN 1 END) as language_populated,
--     COUNT(CASE WHEN total IS NOT NULL THEN 1 END) as total_populated
-- FROM metadata;

-- Optional Step 5: After confirming the migration works, you can drop old JSONB columns
-- WARNING: Only run these after thorough testing and backup
-- ALTER TABLE metadata DROP COLUMN IF EXISTS constituency_details;
-- ALTER TABLE metadata DROP COLUMN IF EXISTS administrative_address;
-- ALTER TABLE metadata DROP COLUMN IF EXISTS polling_details;
-- ALTER TABLE metadata DROP COLUMN IF EXISTS detailed_elector_summary;
-- ALTER TABLE metadata DROP COLUMN IF EXISTS authority_verification;
