import pandas as pd
from database_connection import get_db_engine 
import os
import sys
import sqlite3

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'database', 'dataset.db')
RAW_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'staging', '01_raw')
os.makedirs(RAW_DIR, exist_ok=True)

def get_table_names(db_path):
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [table[0] for table in cursor.fetchall()]
        conn.close()
        return tables
    except Exception as e:
        print(f"Error getting table names: {e}")
        return []

def main():
    print(f"--- [1/3] Running load_data.py ---")
    
    tables = get_table_names(DB_PATH)
    if not tables:
        print("  - ❌ No tables found in database.")
        sys.exit(1)
        
    print(f"  - Found {len(tables)} tables. Exporting to '{os.path.basename(RAW_DIR)}'...")
    
    engine = get_db_engine() 
    
    df = pd.read_sql_table('Refrigerator', engine)
    for table_name in tables:
        df = pd.read_sql_table(table_name, engine)
        output_path = os.path.join(RAW_DIR, f"{table_name}.csv")
        df.to_csv(output_path, index=False)
        print(df.columns)
        print(f"    - Exported '{table_name}' to {os.path.basename(output_path)}")
        
    print(f"  - ✅ All tables exported successfully.")

if __name__ == '__main__':
    main()