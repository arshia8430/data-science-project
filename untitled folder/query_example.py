import sqlite3
import pandas as pd
import os

DB_PATH = os.path.join('database', 'dataset.db')

def get_db_connection(db_path):
    if not os.path.exists(db_path):
        print(f"Error: Database file not found at '{db_path}'")
        return None
    try:
        conn = sqlite3.connect(db_path)
        return conn
    except sqlite3.Error as e:
        print(f"Database connection error: {e}")
        return None

def get_table_info(conn):
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [table[0] for table in cursor.fetchall()]
    
    table_info = {}
    for table_name in tables:
        cursor.execute(f'PRAGMA table_info("{table_name}");')
        columns = [info[1] for info in cursor.fetchall()]
        table_info[table_name] = columns
    return table_info

def find_col_by_keyword(columns, keywords):
    for col in columns:
        for keyword in keywords:
            if keyword in str(col).lower():
                return col
    return None

def run_dynamic_queries(conn, table_info):
    for table, columns in table_info.items():
        print("="*60)
        print(f"Running Dynamic Queries on Table: {table}")
        print("="*60)

        title_col = find_col_by_keyword(columns, ['title', 'نام', 'محصول'])
        price_col = find_col_by_keyword(columns, ['price', 'قیمت'])
        numeric_col_for_sort = find_col_by_keyword(columns, ['stars', 'امتیاز', 'ظرفیت', 'توان', 'وزن', 'capacity', 'power', 'weight'])

        if not title_col:
            title_col = f'"{columns[0]}"' 

        queries_to_run = []
        
        queries_to_run.append({
            "title": f"1. [{table}] Show 5 random items",
            "query": f'SELECT {title_col} FROM "{table}" ORDER BY RANDOM() LIMIT 5;'
        })
        
        queries_to_run.append({
            "title": f"2. [{table}] Count total number of items",
            "query": f'SELECT COUNT(*) as total_items FROM "{table}";'
        })
        
        if price_col:
            queries_to_run.append({
                "title": f"3. [{table}] Find 5 most expensive items",
                "query": f'SELECT {title_col}, "{price_col}" FROM "{table}" WHERE "{price_col}" IS NOT NULL ORDER BY "{price_col}" DESC LIMIT 5;'
            })
            queries_to_run.append({
                "title": f"4. [{table}] Calculate average price",
                "query": f'SELECT AVG("{price_col}") as average_price FROM "{table}";'
            })

        if numeric_col_for_sort:
            queries_to_run.append({
                "title": f"5. [{table}] Find top 5 items based on column '{numeric_col_for_sort}'",
                "query": f'SELECT {title_col}, "{numeric_col_for_sort}" FROM "{table}" WHERE "{numeric_col_for_sort}" IS NOT NULL ORDER BY "{numeric_col_for_sort}" DESC LIMIT 5;'
            })
            
        for item in queries_to_run:
            print(f"\n--- Query: {item['title']} ---")
            try:
                df = pd.read_sql_query(item['query'], conn)
                if df.empty:
                    print("--> No results found.")
                else:
                    print(df.to_string())
            except sqlite3.OperationalError as e:
                print(f"--> Could not execute query. Error: {e}")

def main():
    conn = get_db_connection(DB_PATH)
    if conn:
        try:
            table_info = get_table_info(conn)
            if not table_info:
                print("No tables found in the database.")
                return
                
            print(f"✅ Database connection successful. Found tables: {list(table_info.keys())}\n")
            run_dynamic_queries(conn, table_info)
        finally:
            conn.close()
            print("\n✅ Database connection closed.")

if __name__ == '__main__':
    main()