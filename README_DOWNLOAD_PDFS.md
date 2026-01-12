# Download Missing Voters PDFs

This script downloads PDFs for voters with missing names from the `voters_missing_names.json` file using authenticated AWS S3 access.

## Features

✅ **AWS S3 Authentication** - Uses credentials from `.env` file for secure S3 access  
✅ **Progress Bar** - Shows real-time download progress in the console  
✅ **Resume Support** - Automatically resumes from where it left off if interrupted  
✅ **Status Tracking** - Tracks download status (pending/success/failed) in the JSON file  
✅ **Auto-Retry** - Retries failed downloads up to 3 times per file  
✅ **Retry Failed** - Use `--retry-failed` flag to retry all failed downloads  
✅ **Periodic Saves** - Saves progress every 10 files to prevent data loss  

## Prerequisites

### Required Environment Variables

The script requires AWS credentials in your `.env` file:

```env
AWS_ACCESS_KEY_ID=your_access_key_here
AWS_SECRET_ACCESS_KEY=your_secret_key_here
AWS_REGION=ap-southeast-2
```

✅ These are already configured in your `.env` file!

### Required Python Packages

- `boto3` - AWS SDK for Python  
- `python-dotenv` - Load environment variables from .env
- `tqdm` - Progress bar display

✅ All packages are already installed!

### Basic Usage

Download all pending PDFs:
```bash
python download_missing_voters_pdfs.py
```

### Retry Failed Downloads

If some downloads failed, retry them:
```bash
python download_missing_voters_pdfs.py --retry-failed
```

### Custom Paths

Specify custom JSON file and output directory:
```bash
python download_missing_voters_pdfs.py --json custom.json --output-dir custom_pdfs
```

### Custom Retry Count

Change the maximum number of retries per file (default is 3):
```bash
python download_missing_voters_pdfs.py --max-retries 5
```

### Help

View all available options:
```bash
python download_missing_voters_pdfs.py --help
```

## How It Works

1. **Reads JSON** - Loads the voters data from `voters_missing_names.json`
2. **Checks Status** - For each entry, checks the `download_status` field:
   - `pending` - Will be downloaded
   - `success` - Skipped (already downloaded)
   - `failed` - Skipped (unless `--retry-failed` is used)
3. **Downloads PDFs** - Downloads each PDF with progress tracking
4. **Updates Status** - Updates the JSON file with success/failure status
5. **Saves Progress** - Saves every 10 files and at the end

## Status Tracking

The script adds two fields to each entry in the JSON:

- `download_status`: One of `pending`, `success`, or `failed`
- `download_error`: Error message if download failed (null if successful)

## Example Output

```
Loading data from voters_missing_names.json...
Total entries in JSON: 633
Files to download: 633

Downloading PDFs to missing_voters_pdfs/

Overall Progress: 100%|████████████| 633/633 [15:30<00:00, 1.47s/file]

Saving final status to JSON...

============================================================
DOWNLOAD SUMMARY
============================================================
Total processed: 633
✓ Successful:    625
✗ Failed:        8
============================================================

⚠ 8 downloads failed. Run with --retry-failed to retry them.

Overall status:
  failed: 8
  success: 625

PDFs saved to: E:\Raja_mohaemd\projects\voter-shield-data-cleanup\missing_voters_pdfs
```

## Resume Functionality

If the script is interrupted (Ctrl+C, connection loss, etc.):

1. The progress is automatically saved in the JSON file
2. Simply run the script again - it will resume from where it left off
3. Already downloaded PDFs are skipped automatically

## Dependencies

- `requests` - For downloading PDFs
- `tqdm` - For progress bar display

Both are already in your requirements.txt and installed.

## Output Structure

PDFs are saved with the same name as the `file_name` field in the JSON:

```
missing_voters_pdfs/
├── Tamil Nadu-(S22)_Manachanallur-(AC144)_100.pdf
├── Tamil Nadu-(S22)_Manachanallur-(AC144)_103.pdf
├── Tamil Nadu-(S22)_Manachanallur-(AC144)_107.pdf
└── ...
```

## Troubleshooting

### All downloads are failing

- Check your internet connection
- Verify the S3 URLs are accessible
- Try increasing retries: `--max-retries 5`

### Script stops unexpectedly

- Check disk space (PDFs can be large)
- Review the error messages in the console
- Check the `download_error` field in the JSON for specific errors

### Want to reset and re-download everything

Manually edit `voters_missing_names.json` and remove the `download_status` and `download_error` fields from all entries, or delete them and the script will treat all as pending.
