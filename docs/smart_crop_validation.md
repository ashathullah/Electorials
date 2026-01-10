# Smart Crop Validation - Implementation Documentation

## Overview
This document describes the smart crop validation logic implemented to handle variable Tamil PDF structures where voters may start on page 2 or page 3.

## Problem Statement
Tamil electoral roll PDFs have inconsistent structures:
- Some PDFs: Page 1 (metadata) → Page 2 (voters start) → Page 3 (more voters)
- Other PDFs: Page 1 (metadata) → Page 2 (polling station photos) → Page 3 (voters start) → Page 4 (more voters)

Previously, the system would skip 3 pages for Tamil documents, potentially missing voters on pages 2-3.

## Solution: Smart Validation with Fallback

### Logic Flow

```
1. Language Detection (Page 1)
   └─> Tamil detected? → Skip 3 pages (default)

2. First Pass Cropping
   └─> Process pages 4 onwards
   └─> Count total crops

3. Validation Check (Tamil only)
   ├─> Expected total from metadata: net_total.total
   ├─> Actual crops: count from cropping
   └─> Match? 
       ├─> YES → ✓ Continue to next step
       └─> NO  → Go to step 4

4. Fallback: Re-process Page 3
   └─> Add page 3 to cropping
   └─> Count additional crops
   └─> New total = previous + additional

5. Final Validation
   ├─> Crop count == Expected total?
   │   ├─> YES → ✓ Processing complete
   │   └─> NO  → ❌ TERMINATE (log critical error)
```

## Implementation Details

### Modified Files

#### 1. `prompt.md`
**Change**: Added AI validation rule for `net_total.total`

```markdown
- **CRITICAL**: The "net_total.total" field MUST always be greater than 1. 
  Electoral rolls NEVER contain only a single voter. If you detect a value 
  of 1, re-examine the document carefully.
```

**Rationale**: Prevents AI from extracting incorrect total values (e.g., reading page numbers as totals)

#### 2. `src/processors/image_cropper.py`

##### New Helper Methods

**`_get_expected_voter_count() -> Optional[int]`**
- Reads metadata JSON file
- Extracts `detailed_elector_summary.net_total.total`
- Returns expected voter count or None

**`_is_tamil_document() -> bool`**
- Checks `language_detected` array from metadata
- Returns True if primary language contains "tamil"

**`_process_pages_batch(page_images: List[Path]) -> int`**
- Processes a batch of pages in parallel
- Encapsulates ThreadPoolExecutor logic
- Returns total crop count

##### Modified Method

**`process() -> bool`**
- **Before**: Simple skip + process all remaining pages
- **After**: Smart validation with fallback

Key changes:
1. Store `skipped_pages` array for potential re-processing
2. Get `expected_total` from metadata
3. Process first batch (pages after skip)
4. **Tamil validation**:
   - If `crops < expected_total` AND page 3 exists
   - Re-process page 3
   - Add additional crops to total
5. **Final check**:
   - If still `crops != expected_total` → Return `False` (terminate)
   - Else → Continue normally

## Example Scenarios

### Scenario 1: Voters Start on Page 4 (Standard Tamil PDF)

```
PDF Structure:
- Page 1: Metadata
- Page 2: Polling station photos
- Page 3: Maps/diagrams
- Page 4-10: Voters (557 total)

Execution:
1. Skip pages 1-3
2. Process pages 4-10 → 557 crops
3. Validation: 557 == 557 ✓
4. Result: SUCCESS
```

### Scenario 2: Voters Start on Page 3 (Your Problem Case)

```
PDF Structure:
- Page 1: Metadata
- Page 2: Polling station photos
- Page 3-10: Voters (557 total)

Execution:
1. Skip pages 1-3
2. Process pages 4-10 → 527 crops
3. Validation: 527 < 557 ❌
4. Fallback: Re-process page 3 → +30 crops
5. New total: 557
6. Final validation: 557 == 557 ✓
7. Result: SUCCESS
```

### Scenario 3: Corrupted/Invalid PDF

```
PDF Structure:
- Page 1: Metadata (total: 557)
- Page 2-10: Only 400 actual voter boxes detected

Execution:
1. Skip pages 1-3
2. Process pages 4-10 → 370 crops
3. Validation: 370 < 557 ❌
4. Fallback: Re-process page 3 → +30 crops
5. New total: 400
6. Final validation: 400 != 557 ❌
7. Log: "CRITICAL: Crop count still mismatched. Expected: 557, Got: 400"
8. Result: TERMINATED (return False)
```

## Benefits

✅ **Handles both PDF structures** automatically  
✅ **Self-correcting**: Adds page 3 only when needed  
✅ **Validates data integrity**: Ensures crop count matches metadata  
✅ **Fail-safe**: Terminates on persistent mismatches to prevent bad data  
✅ **Efficient**: Only re-processes page 3 when necessary  
✅ **Logged**: Clear logging at each validation step  

## Edge Cases Handled

1. **No metadata available**: Validation skipped, normal processing continues
2. **Invalid total in metadata**: Validation skipped (expected_total = None)
3. **English documents**: Validation skipped (not Tamil)
4. **Crops > expected**: Allowed (not considered an error)
5. **Less than 3 pages skipped**: Validation skipped (page 3 doesn't exist)

## Logs to Monitor

### Success Case
```
[INFO] Found 10 page image(s)
[INFO] Tamil document detected, will skip first 3 pages
[INFO] Skipping first 3 pages (non-voter pages)...
[DEBUG] Expected voter count from metadata: 557
[INFO] Validation check: expected=557, cropped=527, match=False
[WARN] Crop count mismatch: 527 < 557. Re-processing page 3: page-003.png
[INFO] Page 3 re-processed: additional_crops=30, new_total=557, time=1.23s
[INFO] ✓ Crop count validated successfully after page 3 re-processing: 557
[INFO] Cropping complete: pages=8, crops=557, time=12.45s
```

### Failure Case
```
[INFO] Validation check: expected=557, cropped=400, match=False
[WARN] Crop count mismatch: 400 < 557. Re-processing page 3: page-003.png
[INFO] Page 3 re-processed: additional_crops=30, new_total=430, time=1.23s
[ERROR] CRITICAL: Crop count still mismatched after retry. Expected: 557, Got: 430. Terminating processing for this document.
```

## Testing Checklist

- [ ] Tamil PDF with voters starting on page 4 (standard case)
- [ ] Tamil PDF with voters starting on page 3 (your problem case)
- [ ] Tamil PDF with voters starting on page 2 (edge case)
- [ ] English PDF (should skip validation)
- [ ] PDF with missing/invalid metadata total
- [ ] PDF with actual mismatch (corrupted data)

## Future Enhancements

Potential improvements:
1. Support for dynamic page range detection (try pages 2, 3, 4 until match)
2. Configurable tolerance (allow ±N crops difference)
3. Page-by-page validation (detect which exact pages are missing)
4. AI-based page content classification (detect voter pages without cropping)
