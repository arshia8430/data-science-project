#!/bin/bash

# --- Script to add Phase 3 structure to the project ---

echo "🚀 Adding Phase 3 directories and files..."

# 1. Create a directory to store trained models
mkdir -p models
echo "  - Created 'models/' directory."

# 2. Create a directory for Jupyter notebooks for experimentation
mkdir -p notebooks
echo "  - Created 'notebooks/' directory."

# 3. Create the new scripts for training and prediction inside the 'scripts' folder
touch scripts/train_model.py
touch scripts/make_predictions.py
echo "  - Created 'scripts/train_model.py' and 'scripts/make_predictions.py'."

# 4. Create the two main pipeline files in the root directory
touch run_training_pipeline.py
touch run_prediction_pipeline.py
echo "  - Created 'run_training_pipeline.py' and 'run_prediction_pipeline.py'."

# 5. Create an example notebook file
touch notebooks/01_model_exploration.ipynb
echo "  - Created example notebook 'notebooks/01_model_exploration.ipynb'."

echo "✅ Phase 3 structure added successfully!"
