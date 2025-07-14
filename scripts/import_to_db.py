import pandas as pd
import os
import glob
from sqlalchemy import create_engine
import re

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'database', 'dataset.db')

def persian_to_english_numerals(text: str) -> str:
    if not isinstance(text, str):
        return str(text)
    
    persian_nums = '۰۱۲۳۴۵۶۷۸۹'
    english_nums = '0123456789'
    translation_table = str.maketrans(persian_nums, english_nums)
    return text.translate(translation_table)

def extract_number_from_text(text: str) -> float:
    if not isinstance(text, str):
        return text
    match = re.search(r'\d+\.?\d*', text)
    if match:
        return float(match.group(0))
    return None

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in df.columns:
        df[col] = df[col].apply(persian_to_english_numerals)
        if df[col].dtype == 'object':
            df[col] = df[col].str.replace("سانتی متر", "", regex=False)
            df[col] = df[col].str.replace("سانتی‌متر", "", regex=False)
            df[col] = df[col].str.replace("سانتیمتر", "", regex=False)
            df[col] = df[col].str.replace(",", "", regex=False)
            df[col] = df[col].str.replace("لیتر", "", regex=False)
            df[col] = df[col].str.replace("عدد", "", regex=False)
            df[col] = df[col].str.strip()

    for col in df.columns:
        numeric_series = pd.to_numeric(df[col], errors='coerce')
        numeric_ratio = numeric_series.notna().sum() / len(df)

        if numeric_ratio > 0.7:
            print(f"  - Column '{col}' identified as numeric ({numeric_ratio:.0%}). Forcing conversion.")
            mask = numeric_series.isna() & df[col].notna()
            df.loc[mask, col] = df.loc[mask, col].apply(extract_number_from_text)
            
            df[col] = pd.to_numeric(df[col], errors='coerce')

    if 'title' in df.columns:
        df.dropna(subset=['title'], inplace=True)
        df.drop_duplicates(subset=['title'], keep='first', inplace=True)
    elif 'id' in df.columns:
         df.dropna(subset=['id'], inplace=True)
         df.drop_duplicates(subset=['id'], keep='first', inplace=True)

    return df

def find_excel_files(base_path):
    search_pattern = os.path.join(base_path, '**', '*.xlsx')
    return glob.glob(search_pattern, recursive=True)

def import_data_to_db():
    excel_files = find_excel_files(DATA_DIR)
    if not excel_files:
        print(f"Error: No Excel files found in '{DATA_DIR}'")
        return

    engine = create_engine(f'sqlite:///{DB_PATH}')
    
    print("Starting data import process...")
    
    for file_path in excel_files:
        table_name = os.path.basename(os.path.dirname(file_path))
        print(f"\nProcessing category: {table_name}")
        
        try:
            df = pd.read_excel(file_path, header=1)
            
            print(f"  - Read {len(df)} rows and {len(df.columns)} columns.")
            
            cleaned_df = clean_dataframe(df)
            print(f"  - After cleaning, {len(cleaned_df)} rows remain.")

            if cleaned_df.empty:
                print(f"  - Warning: No data left for '{table_name}' after cleaning. Skipping.")
                continue

            cleaned_df.to_sql(table_name, engine, if_exists='replace', index=False)
            print(f"  - ✅ Successfully created/replaced table '{table_name}' in the database.")

        except Exception as e:
            print(f"  - ❌ Error processing file for '{table_name}': {e}")

    print("\nData import process finished.")

if __name__ == '__main__':
    import_data_to_db()