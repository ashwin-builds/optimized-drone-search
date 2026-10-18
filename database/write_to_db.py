import os
from datetime import datetime, timezone

import psycopg
from dotenv import load_dotenv

load_dotenv()
URL = os.environ["DATABASE_URL"]


def write_config(total_drones, latitude, longitude, depot_latitude, depot_longitude,
                 depot_stock, drone_capacity, restock_threshold=0, restock_time_s=60):
    """Replace the single static_config row."""
    with psycopg.connect(URL) as conn:
        conn.execute("DELETE FROM static_config")
        conn.execute(
            "INSERT INTO static_config (total_drones, latitude, longitude, depot_latitude, "
            "depot_longitude, depot_stock, drone_capacity, restock_threshold, restock_time_s) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)",
            (total_drones, latitude, longitude, depot_latitude, depot_longitude,
             depot_stock, drone_capacity, restock_threshold, restock_time_s),
        )


def write_readings(rows):
    """rows = [(drone_id, ts, latitude, longitude, supplies, status), ...]"""
    with psycopg.connect(URL) as conn:
        conn.cursor().executemany(
            "INSERT INTO input_data (drone_id, ts, latitude, longitude, supplies, status) "
            "VALUES (%s, %s, %s, %s, %s, %s) ON CONFLICT DO NOTHING",
            rows,
        )


if __name__ == "__main__":
    write_config(4, 47.6553, -122.3035, 47.6540, -122.3080, depot_stock=100, drone_capacity=5)
    now = datetime.now(timezone.utc)
    write_readings([
        (1, now, 47.6555, -122.3040, 5, "searching"),
        (2, now, 47.6548, -122.3021, 4, "delivering"),
    ])
    print("Wrote config + 2 readings")