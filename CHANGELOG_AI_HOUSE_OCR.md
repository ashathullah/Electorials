# AI House Number OCR Improvements

## Summary of Changes

This update improves the AI-based house number extraction system with:
1. **Continued use of TOML format** (for lower token count)
2. **Improved page data isolation** to prevent data bleeding between pages
3. **Added retry strategy** for house number extraction
4. **Enhanced consistency** in AI response handling
5. **CRITICAL FIX: Serial number-based matching** instead of position-based matching

## Changes Made

### 1. AI ID Processor (`src/processors/ai_id_processor.py`)

#### A. Optimized TOML Prompt
- Kept TOML format for lower token usage
- Simplified prompt while maintaining clarity
- Added explicit instructions for complete house number extraction (including letters, numbers, /, -, parentheses)

#### B. Enhanced TOML Parsing
- Added validation to check response structure before processing
- Added page key validation (must start with "page-")
- Support for both `records` and `voter_records` keys (backward compatible)
- Enhanced error logging with more details (500 chars instead of 200)
- **CRITICAL FIX**: Only add page results if valid data exists for THAT specific page
- Added warning when no valid records extracted for a page (prevents carrying over previous page data)

#### C. Improved Page Data Isolation
```python
# Only add to DICT if we got valid data for THIS page
if page_results:
    results_dict[page_id] = page_results
    self.log_debug(f"Extracted {len(page_results)} records for {page_id}")
else:
    # Empty results for this page - don't carry over from previous page
    self.log_warning(f"No valid records extracted for {page_id}")
```

### 2. OCR Processor (`src/processors/ocr_processor.py`)

#### A. Enhanced Page Data Application
- Added explicit comments about page data isolation
- Use `range(min(len(page_ai_results), len(page_records)))` to ensure bounds checking
- Added logging for voters beyond AI results count
- Added fallback logging when no AI results exist for a page

#### B. Added House Number Retry Strategy
New method `_retry_house_from_crop()`:
- Retries house number extraction from individual crop images
- Falls back when merged image OCR fails
- Validates results using `_is_valid_house_number()`
- Only runs if AI ID processor is NOT being used (to avoid conflicts)

#### C. Integrated Retry in Batch Processing
Added house number retry in both processing paths:
- **Tesseract path**: After age retry, before logging (line ~690)
- **Tamil OCR path**: After age retry, before timing adjustment (line ~730)

Logic:
```python
# Retry house number extraction from individual crop if invalid/empty
# Only retry if AI processor is NOT being used (AI takes precedence later)
if not self.ai_id_processor and not self._is_valid_house_number(result.house_no):
    retry_house = self._retry_house_from_crop(page_id, image_name)
    if retry_house:
        result.house_no = retry_house
        self.log_debug(f"House number retry successful: {retry_house}")
```

#### D. **CRITICAL FIX: Serial Number-Based Matching**

**Problem Identified:**
- AI was returning correct serial numbers but assigning them to wrong page IDs
- Example: Serial #410 belongs to page-017, but AI labeled it as page-018
- Position-based matching caused wrong house numbers to be applied

**Solution Implemented:**

1. **Added global serial-to-house map** in `ai_id_processor.py`:
```python
def get_global_serial_house_map(self) -> Dict[str, str]:
    """
    Get a global mapping of serial_no -> house_no across ALL pages.
    
    This is needed because AI may assign records to wrong page IDs,
    but the serial numbers are usually correct.
    """
    serial_map = {}
    for page_id, results in self.page_results.items():
        for result in results:
            if result.serial_no and result.serial_no.strip():
                serial_no = result.serial_no.strip()
                if serial_no not in serial_map:
                    serial_map[serial_no] = result.house_no.strip() if result.house_no else ""
    return serial_map
```

2. **Changed matching logic** in `ocr_processor.py`:
   - Build global map before processing pages
   - Match by serial_no instead of position
   - Works regardless of which page AI assigned the record to

```python
# Build global serial_no -> house_no map from AI results
global_ai_house_map = {}
if self.ai_id_processor:
    global_ai_house_map = self.ai_id_processor.get_global_serial_house_map()

# Later, when processing each page:
for record in page_records:
    record_serial = record.serial_no.strip() if record.serial_no else ""
    
    # Match by serial number from GLOBAL map
    if record_serial and record_serial in global_ai_house_map:
        ai_house = global_ai_house_map[record_serial]
        if ai_house:
            record.house_no = ai_house
```

**Impact:**
- Fixes issue where serial #410 on page-017 was getting house number from page-018's position
- Now correctly matches serial #410 to its proper house number regardless of page assignment
- Eliminates data misalignment between OCR and AI results


## Benefits

### 1. Fixed Page Data Bleeding
- **Before**: If AI didn't return data for a page, previous page's data could leak into the current page
- **After**: Each page is strictly isolated; missing data stays missing (uses OCR fallback)

### 2. Increased Consistency
- Better validation at every step
- Clear error messages for debugging
- Proper bounds checking to prevent index errors

### 3. Retry Strategy
- **3-tier fallback**:
  1. Primary: Merged image OCR
  2. Retry: Individual crop image OCR (new!)
  3. AI Override: AI ID extraction (if enabled)
  
- Only retries when AI processor is not being used (avoids double-processing)
- Validates results before accepting them

### 4. Lower Token Usage
- TOML format uses fewer tokens than JSON
- Shortened prompt while maintaining clarity
- Cost savings on AI API calls

## Testing Recommendations

1. **Test page boundaries**: Verify that voter records don't bleed across pages
2. **Test AI failures**: Confirm OCR fallback works when AI returns empty/invalid data
3. **Test retry logic**: Verify house number retry kicks in when primary OCR fails
4. **Monitor logs**: Check for the new debug messages:
   - "Extracted N records for page-XXX"
   - "No valid records extracted for page-XXX"
   - "Using OCR for N voters beyond AI results on page-XXX"
   - "House number retry successful: XXX"

## Backward Compatibility

- Supports both `records` and `voter_records` TOML keys
- Existing configurations continue to work
- No database schema changes required
