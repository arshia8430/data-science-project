import pandas as pd
import os
import glob
import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import xgboost as xgb

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
FINAL_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'staging', '03_final')
os.makedirs(MODELS_DIR, exist_ok=True)

def train_all_possible_models():
    
    print(f"{'#'*20} Starting Automated Training for All Models {'#'*20}")
    
    final_datasets = glob.glob(os.path.join(FINAL_DATA_DIR, '*.csv'))
    if not final_datasets:
        print("  - ERROR: No final data files found. Please run the data preparation pipeline first.")
        return

    best_models_map = {
        'Gas_stove': {'price': RandomForestRegressor(random_state=42, n_jobs=-1), 'rating_with_price': xgb.XGBRegressor(random_state=42, n_jobs=-1), 'rating_without_price': RandomForestRegressor(random_state=42, n_jobs=-1)},
        'Washing_machine': {'price': xgb.XGBRegressor(random_state=42, n_jobs=-1), 'rating_with_price': xgb.XGBRegressor(random_state=42, n_jobs=-1), 'rating_without_price': xgb.XGBRegressor(random_state=42, n_jobs=-1)},
        'Stirrer': {'price': RandomForestRegressor(random_state=42, n_jobs=-1), 'rating_with_price': RandomForestRegressor(random_state=42, n_jobs=-1), 'rating_without_price': RandomForestRegressor(random_state=42, n_jobs=-1)},
        'Refrigerator': {'price': RandomForestRegressor(random_state=42, n_jobs=-1), 'rating_with_price': RandomForestRegressor(random_state=42, n_jobs=-1), 'rating_without_price': RandomForestRegressor(random_state=42, n_jobs=-1)},
        'Rice_cooker': {'price': xgb.XGBRegressor(random_state=42, n_jobs=-1), 'rating_with_price': xgb.XGBRegressor(random_state=42, n_jobs=-1), 'rating_without_price': RandomForestRegressor(random_state=42, n_jobs=-1)},
        'Meat_grinder': {'price': RandomForestRegressor(random_state=42, n_jobs=-1), 'rating_with_price': LinearRegression(), 'rating_without_price': LinearRegression()},
        'fryer': {'price': RandomForestRegressor(random_state=42, n_jobs=-1), 'rating_with_price': RandomForestRegressor(random_state=42, n_jobs=-1), 'rating_without_price': RandomForestRegressor(random_state=42, n_jobs=-1)},
        'Dishwasher': {'price': xgb.XGBRegressor(random_state=42, n_jobs=-1), 'rating_with_price': RandomForestRegressor(random_state=42, n_jobs=-1), 'rating_without_price': RandomForestRegressor(random_state=42, n_jobs=-1)},
        'Juicer': {'price': RandomForestRegressor(random_state=42, n_jobs=-1), 'rating_with_price': RandomForestRegressor(random_state=42, n_jobs=-1), 'rating_without_price': LinearRegression()}
    }
    
    tasks = ['price', 'rating_with_price', 'rating_without_price']

    for data_path in final_datasets:
        category_name = os.path.basename(data_path).replace('.csv', '')
        print(f"\n{'='*20} Processing Category: {category_name} {'='*20}")
        df = pd.read_csv(data_path)

        for task_name in tasks:
            print(f"  - Task: Training model for '{task_name}'...")
            
            model_to_train = best_models_map.get(category_name, {}).get(task_name)
            if model_to_train is None:
                print(f"    - WARNING: No best model defined for this task. Skipping.")
                continue

            if task_name == 'price':
                if 'price' not in df.columns: continue
                target_col, drop_cols = 'price', ['price', 'rating']
            elif task_name == 'rating_with_price':
                if 'rating' not in df.columns: continue
                target_col, drop_cols = 'rating', ['rating']
            elif task_name == 'rating_without_price':
                if 'rating' not in df.columns: continue
                target_col, drop_cols = 'rating', ['price', 'rating']
            
            y = df[target_col]
            X = df.drop(columns=drop_cols, errors='ignore').select_dtypes(include=np.number)
            
            if X.empty:
                print(f"    - WARNING: No features available for this task. Skipping.")
                continue

            model_to_train.fit(X, y)

            model_filename = f"{category_name}_{task_name}_model.joblib"
            model_path = os.path.join(MODELS_DIR, model_filename)
            joblib.dump(model_to_train, model_path)
            print(f"    - ✅ Model saved: {model_filename}")

    print(f"\n{'#'*20} All Models Trained and Saved Successfully {'#'*20}")

if __name__ == '__main__':
    train_all_possible_models()