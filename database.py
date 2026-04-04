import sqlite3
import os
import json
from datetime import datetime


class PhenotypeDatabase:
    def __init__(self, db_path="phenotyping_results.db"):
        self.db_path = db_path
        self.init_db()

    def init_db(self):
        """Initializes the database and creates the necessary tables."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Create table for phenotyping reports
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS phenotyping_report (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            disease_class TEXT NOT NULL,
            confidence REAL NOT NULL,
            num_leaves_detected INTEGER,
            traits_json TEXT NOT NULL
        )
        """)

        conn.commit()
        conn.close()

    def save_report(self, disease_class, confidence, num_leaves, traits_dict):
        """
        Saves a single phenotyping report to the database.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        traits_json = json.dumps(traits_dict)

        cursor.execute(
            """
        INSERT INTO phenotyping_report (timestamp, disease_class, confidence, num_leaves_detected, traits_json)
        VALUES (?, ?, ?, ?, ?)
        """,
            (timestamp, disease_class, confidence, num_leaves, traits_json),
        )

        conn.commit()
        conn.close()

    def get_all_reports(self):
        """
        Retrieves all reports from the database.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM phenotyping_report ORDER BY timestamp DESC")
        rows = cursor.fetchall()

        conn.close()
        return rows

    def get_recent_reports(self, limit=5):
        """
        Retrieves the most recent reports from the database up to the specified limit.
        Used to optimize frontend rendering by avoiding fetching all historical records into memory.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Bolt: ⚡ Performance Improvement
        # Exclude 'traits_json' which can be large, reducing memory parsing and I/O overhead
        cursor.execute(
            "SELECT id, timestamp, disease_class, confidence, num_leaves_detected FROM phenotyping_report ORDER BY timestamp DESC LIMIT ?",
            (limit,),
        )
        rows = cursor.fetchall()

        conn.close()
        return rows


if __name__ == "__main__":
    # Test DB
    print("Testing Database Module...")
    db = PhenotypeDatabase("test_db.sqlite")

    # Mock traits
    mock_traits = {"area_px": 1500, "perimeter_px": 250, "mean_G": 120.5}

    db.save_report("Cassava___Healthy", 0.95, 1, mock_traits)
    reports = db.get_all_reports()

    print(f"Total reports saved: {len(reports)}")
    print("Latest report:", reports[0])

    # Cleanup test db
    if os.path.exists("test_db.sqlite"):
        os.remove("test_db.sqlite")
