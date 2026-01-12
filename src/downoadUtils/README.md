# S3 CSV Downloader

A robust Python script to download CSV files from AWS S3 with progress tracking, resume capability, and retry mechanism.

## Features

✅ **Progress Bar** - Visual progress indicator using `tqdm`  
✅ **Resumable Downloads** - Automatically continues from where it left off if interrupted  
✅ **Status Tracking** - JSON file stores download status for each file  
✅ **Retry Failed** - Re-download only failed files with `--retry` flag  
✅ **Force Re-download** - Override existing files with `--force` flag  
✅ **Smart Skip** - Automatically skips files that are already downloaded  

## Installation

Make sure you have the required dependencies:

```bash
pip install boto3 python-dotenv tqdm
```

## Configuration

AWS credentials should be set in the `.env` file at the project root:

```bash
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_REGION=ap-southeast-2
```

## Usage

### Basic Download

Download all CSV files from the default S3 location:

```bash
python src/downoadUtils/download_s3_csvs.py
```

### Resume Interrupted Download

If the download was interrupted, simply run the same command again. The script will automatically skip already downloaded files and continue from where it left off:

```bash
python src/downoadUtils/download_s3_csvs.py
```

### Retry Failed Downloads

To retry only the files that failed in previous runs:

```bash
python src/downoadUtils/download_s3_csvs.py --retry
```

### Force Re-download All Files

To re-download all files even if they already exist locally:

```bash
python src/downoadUtils/download_s3_csvs.py --force
```

### Check Download Status

To see the current status without downloading anything:

```bash
python src/downoadUtils/download_s3_csvs.py --status
```

### Custom S3 Location

Specify a different bucket and prefix:

```bash
python src/downoadUtils/download_s3_csvs.py --bucket your-bucket --prefix path/to/csvs/
```

### Custom Output Directory

Specify a different output directory:

```bash
python src/downoadUtils/download_s3_csvs.py --output /path/to/output
```

## File Status

Each CSV file can have one of the following statuses:

- **`downloaded`** - Successfully downloaded
- **`skipped`** - Already exists locally with correct size
- **`failed`** - Download failed (use `--retry` to re-attempt)
- **`pending`** - Not yet processed

## Tracking File

Download progress is tracked in a JSON file stored in `meta_logs/`:

```
meta_logs/download_status_2026_1_S22_extraction_results.json
```

This file contains:
- File list with S3 keys
- Download status for each file
- Error messages for failed downloads
- File sizes
- Timestamps

## Output Structure

Downloaded files maintain the S3 directory structure:

```
extraction_results/
├── file1_metadata.csv
├── file1_voters.csv
├── file2_metadata.csv
├── file2_voters.csv
└── ...
```

## Command Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--bucket` | S3 bucket name | `264676382451-eci-download` |
| `--prefix` | S3 prefix/path | `2026/1/S22/extraction_results/` |
| `--output` | Output directory | `../../extraction_results` |
| `--tracking` | Tracking directory | `../../meta_logs` |
| `--force` | Re-download all files | `False` |
| `--retry` | Retry only failed files | `False` |
| `--status` | Show status and exit | `False` |

## Error Handling

- **Interrupted Download**: Press `Ctrl+C` to stop. Progress is saved automatically.
- **Network Errors**: Failed downloads are tracked. Use `--retry` to re-attempt.
- **AWS Credentials**: Make sure `.env` has valid AWS credentials.

## Examples

1. **First time download:**
   ```bash
   python src/downoadUtils/download_s3_csvs.py
   ```

2. **Download was interrupted, continue:**
   ```bash
   python src/downoadUtils/download_s3_csvs.py
   ```

3. **Some files failed, retry them:**
   ```bash
   python src/downoadUtils/download_s3_csvs.py --retry
   ```

4. **Check what's been downloaded:**
   ```bash
   python src/downoadUtils/download_s3_csvs.py --status
   ```

5. **Force re-download everything:**
   ```bash
   python src/downoadUtils/download_s3_csvs.py --force
   ```
