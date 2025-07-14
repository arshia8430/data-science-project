import subprocess
import os

def main():
    print("====== Starting Final End-to-End Project Pipeline ======")
    
    # A list of scripts to run in sequence
    pipeline_scripts = [
        "pipeline.py",                   # Runs the entire data prep (load, preprocess, feature_engineer)
        os.path.join("scripts", "train_model.py"),    # Trains all models for all categories
        os.path.join("scripts", "predict.py")         # Runs a showcase of the prediction engine
    ]
    
    for script in pipeline_scripts:
        print(f"\n{'='*20} Executing: {script} {'='*20}")
        try:
            subprocess.run(["python", script], check=True, text=True)
        except subprocess.CalledProcessError as e:
            print(f"!!!!!! SCRIPT FAILED: {script} !!!!!!")
            print(f"Error details: {e}")
            print("====== Pipeline Aborted. ======")
            return
        except FileNotFoundError:
            print(f"!!!!!! SCRIPT NOT FOUND: {script} !!!!!!")
            print("Please ensure all required scripts are in their correct locations.")
            print("====== Pipeline Aborted. ======")
            return

    print("\n====== ✅✅✅ Final End-to-End Pipeline Executed Successfully! ✅✅✅ ======")

if __name__ == "__main__":
    main()