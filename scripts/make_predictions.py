import pandas as pd
import os
import sys
import joblib
import argparse
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts'))
from preprocess import process_single_dataframe
from feature_engineering import feature_engineer_dataframe

MODELS_DIR = 'models'

def predict_missing_values(input_df: pd.DataFrame) -> pd.DataFrame:
    
    df = input_df.copy()
    
    for index, row in df.iterrows():
        category = row.get('category')
        if not category or pd.isna(category):
            print(f"Skipping row {index}: 'category' column is missing.")
            continue
            
        print(f"\n--- Processing Row {index} (Category: {category}) ---")
        
        row_df = pd.DataFrame([row])
        
        price_is_missing = pd.isna(row.get('price'))
        rating_is_missing = pd.isna(row.get('rating'))
        
        preprocessed_row = process_single_dataframe(row_df.copy())
        final_features = feature_engineer_dataframe(preprocessed_row.copy(), category)

        if price_is_missing:
            print(f"  - Task: Predict missing 'price'.")
            model_path = os.path.join(MODELS_DIR, f"{category}_price_model.joblib")
            if os.path.exists(model_path):
                model = joblib.load(model_path)
                
                X_predict = final_features.copy()
                if 'rating' in X_predict.columns:
                    X_predict = X_predict.drop(columns=['rating'])
                
                model_features = [feat for feat in model.feature_names_in_ if feat in X_predict.columns]
                X_predict = X_predict[model_features]

                predicted_price = model.predict(X_predict)[0]
                df.loc[index, 'price'] = predicted_price
                print(f"  - ✅ Predicted Price: {predicted_price:,.0f}")
            else:
                print(f"  - WARNING: Price model for '{category}' not found.")

        if rating_is_missing:
            print(f"  - Task: Predict missing 'rating'.")
            
            if 'price' in row and pd.notna(df.loc[index, 'price']):
                task_type = 'rating_with_price'
                print("    - Using model trained WITH price.")
            else:
                task_type = 'rating_without_price'
                print("    - Using model trained WITHOUT price.")
                
            model_path = os.path.join(MODELS_DIR, f"{category}_{task_type}_model.joblib")
            if os.path.exists(model_path):
                model = joblib.load(model_path)
                
                X_predict = final_features.copy()
                
                # If using the no-price model, drop price from features
                if task_type == 'rating_without_price' and 'price' in X_predict.columns:
                    X_predict = X_predict.drop(columns=['price'])
                
                model_features = [feat for feat in model.feature_names_in_ if feat in X_predict.columns]
                X_predict = X_predict[model_features]

                predicted_rating = model.predict(X_predict)[0]
                df.loc[index, 'rating'] = predicted_rating
                print(f"  - ✅ Predicted Rating: {predicted_rating:.2f}")
            else:
                print(f"  - WARNING: Rating model for '{category}' ({task_type}) not found.")
                
    return df

def main():
    parser = argparse.ArgumentParser(description="Intelligent Prediction Engine.")
    parser.add_argument(
        '--input', 
        type=str, 
        required=True, 
        help="Path to the input CSV file containing raw data with a 'category' column."
    )
    
    args = parser.parse_args()
    
    print("====== Starting Intelligent Prediction Pipeline ======")
    
    try:
        input_df = pd.read_csv(args.input)
        
        predicted_df = predict_missing_values(input_df)
        
        output_filename = f"predicted_{os.path.basename(args.input)}"
        predicted_df.to_csv(output_filename, index=False)
        
        print(f"\n====== ✅ Pipeline Finished Successfully. Output saved to '{output_filename}' ======")
        
    except FileNotFoundError:
        print(f"ERROR: Input file not found at '{args.input}'")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == '__main__':
    main()