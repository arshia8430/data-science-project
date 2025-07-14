import pandas as pd
import os
import glob
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import numpy as np

RAW_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'staging', '01_raw')
PREPROCESSED_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'staging', '02_preprocessed')
os.makedirs(PREPROCESSED_DIR, exist_ok=True)

def process_single_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    
    cols_to_drop = ['title', 'image_path', 'product_url', 'url']
    df.drop(columns=cols_to_drop, inplace=True, errors='ignore')

    df.dropna(axis=1, how='all', inplace=True)

    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            if df[col].isnull().sum() > 0:
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
        elif pd.api.types.is_object_dtype(df[col]):
            if df[col].isnull().sum() > 0:
                if not df[col].mode().empty:
                    mode_val = df[col].mode()[0]
                    df[col] = df[col].fillna(mode_val)

    le = LabelEncoder()
    object_cols = df.select_dtypes(include=['object']).columns
    
    if len(object_cols) > 0:
        for col in object_cols:
            df[col] = le.fit_transform(df[col])

    all_numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    
    cols_to_exclude_from_scaling = ['price', 'rating']
    
    cols_to_scale = [col for col in all_numeric_cols if col not in cols_to_exclude_from_scaling]

    if cols_to_scale:
        scaler = MinMaxScaler()
        df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])

    return df

def main():
    print(f"--- [2/3] Running preprocess.py ---")
    
    input_files = glob.glob(os.path.join(RAW_DIR, '*.csv'))
    
    if not input_files:
        print(f"  - WARNING: No raw data files found in '{os.path.basename(RAW_DIR)}'.")
        return

    print(f"  - Found {len(input_files)} files to preprocess...")

    for file_path in input_files:
        file_name = os.path.basename(file_path)
        print(f"    - Preprocessing '{file_name}'...")
        
        try:
            df = pd.read_csv(file_path)
            processed_df = process_single_dataframe(df)
            output_path = os.path.join(PREPROCESSED_DIR, file_name)
            processed_df.to_csv(output_path, index=False)
            print(f"      - Successfully processed. Final shape: {processed_df.shape}")
            
        except Exception as e:
            print(f"    - ERROR: Failed to process '{file_name}'. Details: {e}")
            continue
        
    print(f"  - ✅ Preprocessing complete for all files.")

if __name__ == '__main__':
    main()