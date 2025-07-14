import pandas as pd
import os
import glob
import numpy as np

PREPROCESSED_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'staging', '02_preprocessed')
FINAL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'staging', '03_final')
os.makedirs(FINAL_DIR, exist_ok=True)

def feature_engineer_dataframe(df: pd.DataFrame, category_name: str) -> pd.DataFrame:
    
    df = df.copy()
    original_cols = df.columns.tolist()

    
    if category_name == 'Refrigerator':
        if all(c in df.columns for c in ['height', 'width', 'depth']):
            df['volume_m3'] = (df['height'] * df['width'] * df['depth']) / 1_000_000
            df['form_factor_ratio'] = df['height'] / (df['width'] + 0.01)
        if all(c in df.columns for c in ['fridge_shelves', 'freezer_shelves']):
            df['shelf_to_capacity_ratio'] = (df['fridge_shelves'] + df['freezer_shelves']) / (df['total_capacity'] + 0.01)

    elif category_name == 'Washing_machine':
        if all(c in df.columns for c in ['price', 'capacity']) and df['capacity'].mean() > 0:
            df['price_per_kg'] = df['price'] / (df['capacity'] + 0.01)
        if all(c in df.columns for c in ['water_consumption', 'power_consumption', 'capacity']) and df['capacity'].mean() > 0:
            df['efficiency_score'] = df['capacity'] / (df['water_consumption'] + df['power_consumption'] + 0.01)
        if all(c in df.columns for c in ['height', 'width', 'depth']):
            df['volume_m3'] = (df['height'] * df['width'] * df['depth']) / 1_000_000

    elif category_name == 'Dishwasher':
        if all(c in df.columns for c in ['height', 'width', 'depth', 'capacity']) and df['height'].mean() > 0:
            df['compactness'] = df['capacity'] / ((df['height'] * df['width'] * df['depth']) + 0.01)



    new_features = [col for col in df.columns if col not in original_cols]
    if new_features:
        print(f"      - Created new features: {new_features}")
            
    return df

def main():
    print(f"--- [3/3] Running feature_engineering.py ---")
    
    input_files = glob.glob(os.path.join(PREPROCESSED_DIR, '*.csv'))

    if not input_files:
        print(f"  - WARNING: No preprocessed files found in '{os.path.basename(PREPROCESSED_DIR)}'.")
        return
        
    print(f"  - Found {len(input_files)} files for feature engineering...")

    for file_path in input_files:
        file_name = os.path.basename(file_path)
        category_name = file_name.replace('.csv', '')
        print(f"    - Engineering features for '{file_name}'...")
        
        try:
            df = pd.read_csv(file_path)
            
            final_df = feature_engineer_dataframe(df, category_name)
            
            output_path = os.path.join(FINAL_DIR, file_name)
            final_df.to_csv(output_path, index=False)
            print(f"      - Successfully engineered features. Final shape: {final_df.shape}")
            
        except Exception as e:
            print(f"    - ERROR: Failed to engineer features for '{file_name}'. Details: {e}")
            continue

    print(f"  - ✅ Feature engineering complete for all files.")

if __name__ == '__main__':
    main()