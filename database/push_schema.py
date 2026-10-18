import os
from pathlib import Path

import psycopg
from dotenv import load_dotenv

load_dotenv()
with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
    conn.execute(Path(__file__).with_name("schema.sql").read_text())
print("Schema applied")