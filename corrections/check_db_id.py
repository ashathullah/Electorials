import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.config import Config
from src.persistence.postgres import PostgresRepository

def check_one_id():
    config = Config()
    repo = PostgresRepository(config.db) # This might fail if Config expects different args for PostgresRepository, let's check repo init
    # repo = PostgresRepository(config.db) is correct based on code I read.
    
    conn = repo._get_connection()
    with conn.cursor() as cur:
        cur.execute("SELECT document_id, pdf_name FROM metadata LIMIT 1")
        row = cur.fetchone()
        if row:
            print(f"ID: {row[0]}")
            print(f"PDF: {row[1]}")
        else:
            print("No metadata found.")

if __name__ == "__main__":
    check_one_id()
