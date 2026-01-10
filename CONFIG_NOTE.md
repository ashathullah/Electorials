# Important Configuration Note

## AWS Region Configuration

Your S3 bucket is located in **ap-southeast-2** (Sydney), NOT ap-south-1 (Mumbai).

Please ensure your `.env` file has the correct region:

```env
AWS_REGION=ap-southeast-2
```

## Correct S3 Details

Based on the AWS Console URL you provided:

- **Bucket Name**: `264676382451-eci-download`
- **Region**: `ap-southeast-2`
- **Prefix/Path**: `2026/1/S22/extraction_results/`

## Scripts Updated

The following scripts have been updated with the correct defaults:

1. ✅ `sync_s3_to_db.py` - Uses correct bucket and prefix
2. ✅ `check_s3_connection.py` - Tests correct bucket

## Testing Results

✅ The dry-run test completed successfully! This confirms:
- S3 connection is working
- CSV files are accessible
- File naming conversion is working correctly

## Next Steps

Now that the configuration is correct, you can:

1. **Run verification** (optional):
   ```bash
   python verify_sync_setup.py
   ```

2. **Sync a small batch**:
   ```bash
   python sync_s3_to_db.py --limit 10
   ```

3. **Run full sync**:
   ```bash
   python sync_s3_to_db.py
   ```

## Database pdf_name Format

The script correctly converts between:
- **CSV format**: `Tamil Nadu-(S22)_Coimbatore (North)-(AC118)_1`
- **DB format**: `Tamil Nadu-(S22)_Coimbatore (North)-(AC118_1`

Note the difference: closing parenthesis is removed before the underscore and number.
