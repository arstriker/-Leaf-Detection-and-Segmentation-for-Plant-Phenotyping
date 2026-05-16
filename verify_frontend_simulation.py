
import os
import sqlite3
import pandas as pd
import time
from datetime import datetime
from database import PhenotypeDatabase

def verify_frontend_optimization_simulation():
    print("\n--- Simulating Frontend Optimization ---")

    # 1. Setup Mock Database
    db_path = "frontend_test.db"
    if os.path.exists(db_path):
        os.remove(db_path)

    db = PhenotypeDatabase(db_path)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    print("Populating 10,000 records...")
    data = []
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    for i in range(10000):
        data.append((ts, f"Class_{i}", 0.99, 1, "{}"))

    cursor.executemany(
        'INSERT INTO phenotyping_report (timestamp, disease_class, confidence, num_leaves_detected, traits_json) VALUES (?, ?, ?, ?, ?)',
        data
    )
    conn.commit()
    conn.close()

    # 2. Simulate Old Frontend Logic (Fetch All)
    print("\n[Old Logic] Fetching all records...")
    start_old = time.time()
    all_records = db.get_all_reports() # Fetches 10k
    df_old = pd.DataFrame(all_records)
    display_old = df_old.head(5)
    end_old = time.time()
    print(f"Old Logic Time: {end_old - start_old:.6f}s")
    print(f"Memory (rows loaded): {len(all_records)}")

    # 3. Simulate New Frontend Logic (Fetch Limit)
    print("\n[New Logic] Fetching limit=5 records...")
    start_new = time.time()
    limit_records = db.get_all_reports(limit=5) # Fetches 5
    df_new = pd.DataFrame(limit_records)
    # The new logic in app.py still calls head(5) for safety, mimicking that
    display_new = df_new.head(5)
    end_new = time.time()
    print(f"New Logic Time: {end_new - start_new:.6f}s")
    print(f"Memory (rows loaded): {len(limit_records)}")

    # 4. Verification
    if len(limit_records) != 5:
        print("FAIL: New logic did not return 5 records.")
        exit(1)

    print("\nSUCCESS: Simulation confirms frontend will display correct data with reduced load.")

    if os.path.exists(db_path):
        os.remove(db_path)

if __name__ == "__main__":
    verify_frontend_optimization_simulation()
