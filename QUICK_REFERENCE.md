# Quick Reference - PDF Download Script

## Common Commands

```bash
# Start/resume downloading
python download_missing_voters_pdfs.py

# Retry failed downloads
python download_missing_voters_pdfs.py --retry-failed

# Use different output directory
python download_missing_voters_pdfs.py --output-dir my_pdfs

# Increase retry attempts
python download_missing_voters_pdfs.py --max-retries 5
```

## What the script does

1. ✅ Downloads PDFs from the `download_link` field in `voters_missing_names.json`
2. ✅ Saves them to `missing_voters_pdfs/` directory
3. ✅ Shows progress bar with download status
4. ✅ Updates JSON with `download_status` and `download_error` fields
5. ✅ Auto-saves every 10 files (prevents data loss)
6. ✅ Can resume if interrupted (just run again)

## Status Values in JSON

- `pending` - Not yet downloaded or needs to be re-attempted
- `success` - Successfully downloaded
- `failed` - Download failed after all retries

## Flags

| Flag | Description |
|------|-------------|
| `--json <path>` | Use different JSON file (default: voters_missing_names.json) |
| `--output-dir <dir>` | Save PDFs to different directory (default: missing_voters_pdfs) |
| `--retry-failed` | Retry all failed downloads |
| `--max-retries <N>` | Set max retries per file (default: 3) |
| `--help` | Show help message |

## Example Workflow

1. **First run** - Download all PDFs:
   ```bash
   python download_missing_voters_pdfs.py
   ```

2. **If interrupted** - Resume:
   ```bash
   python download_missing_voters_pdfs.py
   ```
   (Script automatically skips already downloaded files)

3. **Retry failures** - If some failed:
   ```bash
   python download_missing_voters_pdfs.py --retry-failed
   ```

4. **Check results** - Look at the summary:
   - Successful count
   - Failed count
   - Overall status breakdown
