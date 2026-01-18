"""Upload extracted_numbers JSON into Postgres metadata table.

This script reads JSON files produced under `metadata-regeneration/extracted_numbers/`.
Each JSON file is expected to have a top-level `results` list with items like:

    {
      "document_id": "Tamil Nadu-(S22)_Alangudi-(AC182)_1",
      "pincode": "622303",
      "voters_end": "1020",
      "status": "success"
    }

It will upsert into the `metadata` table:
- document_id
- state (default: Tamil Nadu)
- year (default: 2026)
- pin_code
- voter_end (int)

If your existing `metadata` table schema requires `pdf_name` (NOT NULL),
this script will populate it with the document_id to satisfy constraints.

Usage:
  python upload_extracted_numbers_to_db.py \
    --input-dir metadata-regeneration/extracted_numbers \
    --state "Tamil Nadu" \
    --year 2026

Requires DB_* env vars (same as the rest of the project):
  DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD, DB_SCHEMA (optional)

"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import re

import psycopg2
from psycopg2.extras import execute_values

from src.config import Config


@dataclass(frozen=True)
class ColumnInfo:
    name: str
    is_nullable: bool
    has_default: bool


def _parse_int(value: object) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if value.is_integer():
            return int(value)
        return None
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return None
        # Be tolerant of commas/whitespace or minor OCR noise.
        # Extract the first digit-run (e.g. '1,020' -> 1 then 020 would be wrong),
        # so instead remove common separators and then parse if remaining is digits.
        compact = s.replace(",", "").replace(" ", "")
        if compact.isdigit():
            return int(compact)
        m = re.search(r"\d+", s)
        if m:
            try:
                return int(m.group(0))
            except ValueError:
                return None
        return None
    return None


def _read_results_from_file(path: Path) -> Iterable[Tuple[str, Optional[str], Optional[int]]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    results = data.get("results")
    if not isinstance(results, list):
        return []

    out: List[Tuple[str, Optional[str], Optional[int]]] = []
    for item in results:
        if not isinstance(item, dict):
            continue
        document_id = (item.get("document_id") or "").strip()
        if not document_id:
            continue
        pin_code = item.get("pincode") or item.get("pin_code")
        if isinstance(pin_code, int):
            pin_code = str(pin_code)
        if isinstance(pin_code, str):
            pin_code = pin_code.strip() or None
        else:
            pin_code = None

        voter_end = _parse_int(item.get("voters_end") if "voters_end" in item else item.get("voter_end"))
        out.append((document_id, pin_code, voter_end))

    return out


def _get_connection(config: Config):
    if not config.db.is_configured:
        raise RuntimeError(
            "Database is not configured. Set DB_HOST, DB_NAME, DB_USER, DB_PASSWORD (and optionally DB_PORT, DB_SCHEMA)."
        )

    return psycopg2.connect(
        host=config.db.host,
        port=config.db.port,
        dbname=config.db.name,
        user=config.db.user,
        password=config.db.password,
        sslmode=config.db.ssl_mode,
    )


def _fetch_metadata_columns(cur, schema: str) -> Dict[str, ColumnInfo]:
    cur.execute(
        """
        SELECT
          column_name,
          is_nullable,
          column_default
        FROM information_schema.columns
        WHERE table_schema = %s AND table_name = 'metadata'
        """,
        (schema,),
    )
    rows = cur.fetchall()
    cols: Dict[str, ColumnInfo] = {}
    for name, is_nullable, default in rows:
        cols[name] = ColumnInfo(
            name=name,
            is_nullable=(is_nullable == "YES"),
            has_default=(default is not None),
        )
    return cols


def _ensure_metadata_table(cur, schema: str) -> None:
    # Create a minimal metadata table if it doesn't exist.
    # If a richer schema already exists, this is a no-op.
    cur.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {schema}.metadata (
            document_id TEXT PRIMARY KEY,
            state TEXT,
            year INTEGER,
            pin_code TEXT,
            voter_end INTEGER
        );
        """
    )

    # Ensure required columns exist (works even with the richer schema).
    for col_sql in (
        "ALTER TABLE {schema}.metadata ADD COLUMN IF NOT EXISTS state TEXT;",
        "ALTER TABLE {schema}.metadata ADD COLUMN IF NOT EXISTS year INTEGER;",
        "ALTER TABLE {schema}.metadata ADD COLUMN IF NOT EXISTS pin_code TEXT;",
        "ALTER TABLE {schema}.metadata ADD COLUMN IF NOT EXISTS voter_end INTEGER;",
    ):
        cur.execute(col_sql.format(schema=schema))


def _build_upsert_sql(schema: str, columns: Dict[str, ColumnInfo]) -> Tuple[str, List[str]]:
    needs_pdf_name = False
    pdf_info = columns.get("pdf_name")
    if pdf_info is not None and (not pdf_info.is_nullable) and (not pdf_info.has_default):
        needs_pdf_name = True

    insert_cols = ["document_id", "state", "year", "pin_code", "voter_end"]
    if needs_pdf_name:
        insert_cols.insert(1, "pdf_name")

    set_cols = [
        "state = EXCLUDED.state",
        "year = EXCLUDED.year",
        "pin_code = EXCLUDED.pin_code",
        "voter_end = EXCLUDED.voter_end",
    ]
    if needs_pdf_name:
        set_cols.insert(0, "pdf_name = EXCLUDED.pdf_name")

    sql = (
        f"INSERT INTO {schema}.metadata ("
        + ", ".join(insert_cols)
        + ") VALUES %s "
        + "ON CONFLICT (document_id) DO UPDATE SET "
        + ", ".join(set_cols)
    )

    return sql, insert_cols


def _collect_rows(input_dir: Path) -> Dict[str, Tuple[Optional[str], Optional[int]]]:
    rows: Dict[str, Tuple[Optional[str], Optional[int]]] = {}
    for path in sorted(input_dir.glob("*.json")):
        if ".bak-" in path.name:
            continue
        for document_id, pin_code, voter_end in _read_results_from_file(path):
            # Upload "as-is": keep the latest occurrence encountered.
            rows[document_id] = (pin_code, voter_end)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Upload extracted_numbers JSON to Postgres metadata table")
    parser.add_argument(
        "--input-dir",
        default="metadata-regeneration/extracted_numbers",
        help="Directory containing extracted_numbers JSON files",
    )
    parser.add_argument("--state", default="Tamil Nadu")
    parser.add_argument("--year", type=int, default=2026)
    parser.add_argument("--schema", default=None, help="Override DB schema (defaults to DB_SCHEMA or public)")
    parser.add_argument("--dry-run", action="store_true", help="Parse and report, but do not write to DB")
    parser.add_argument("--batch-size", type=int, default=5000)

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists() or not input_dir.is_dir():
        raise SystemExit(f"Input dir not found: {input_dir}")

    config = Config()
    schema = (args.schema or config.db.schema or "public").strip() or "public"

    doc_map = _collect_rows(input_dir)
    if not doc_map:
        print(f"No records found in {input_dir}")
        return 0

    print(f"Parsed {len(doc_map)} unique document_id rows from {input_dir}")

    if args.dry_run:
        missing_pin = sum(1 for _, (pin, _) in doc_map.items() if not pin)
        missing_end = sum(1 for _, (_, end) in doc_map.items() if end is None)
        print(
            f"Dry run: missing pin_code={missing_pin}, missing voter_end={missing_end} (this is OK)"
        )
        return 0

    conn = _get_connection(config)
    try:
        with conn:
            with conn.cursor() as cur:
                _ensure_metadata_table(cur, schema)
                columns = _fetch_metadata_columns(cur, schema)
                upsert_sql, insert_cols = _build_upsert_sql(schema, columns)

                # Build values list in insert_cols order
                values: List[Tuple[object, ...]] = []
                for document_id, (pin_code, voter_end) in doc_map.items():
                    row: Dict[str, object] = {
                        "document_id": document_id,
                        "state": args.state,
                        "year": args.year,
                        "pin_code": pin_code,
                        "voter_end": voter_end,
                    }
                    if "pdf_name" in insert_cols:
                        # Satisfy NOT NULL constraint when using the richer schema.
                        row["pdf_name"] = document_id

                    values.append(tuple(row[c] for c in insert_cols))

                total = len(values)
                print(f"Upserting {total} rows into {schema}.metadata ...")

                for start in range(0, total, args.batch_size):
                    chunk = values[start : start + args.batch_size]
                    execute_values(cur, upsert_sql, chunk, page_size=min(len(chunk), 1000))
                    print(f"  wrote {min(start + len(chunk), total)}/{total}")

        print("Done.")
        return 0
    finally:
        try:
            conn.close()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
