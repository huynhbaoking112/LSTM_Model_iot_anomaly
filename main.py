"""
Main entry point for the NO2 Anomaly Detection project.
This script orchestrates the entire pipeline from data loading to evaluation.
"""
import argparse
import os
import sys

# Ensure the source directory is in the Python path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.kaggle_data_loader import KaggleDataLoader
from src.preprocessor import DataPreprocessor
from src.lstm_model import LSTMAutoencoder
from src.evaluator import ModelEvaluator

def run_load_data():
    """Handles the data loading and initial processing step."""
    print("====== [Step 1/3] Running Data Loading Pipeline ======")
    loader = KaggleDataLoader()
    processed_df = loader.load_and_process()
    if processed_df.empty:
        print("❌ Data loading failed. Halting pipeline.")
        sys.exit(1)
    print("✅ Data loading complete.\n")

def run_train_pipeline():
    """Handles the model training pipeline."""
    print("====== [Step 2/3] Running Training Pipeline ======")
    # 1. Preprocess the data
    preprocessor = DataPreprocessor()
    train_X, val_X, _, _ = preprocessor.run()
    
    if train_X is None:
        print("❌ Halting pipeline due to preprocessing failure.")
        sys.exit(1)

    # 2. Build and train the model
    autoencoder = LSTMAutoencoder()
    autoencoder.build_model()
    autoencoder.train(train_X, val_X)
    
    # 3. Plot training history
    autoencoder.plot_training_history()
    
    # 4. Set anomaly threshold
    autoencoder.load_model() 
    autoencoder.set_anomaly_threshold(val_X)
    print("✅ Training pipeline complete.\n")

def run_evaluation_pipeline():
    """Handles the model evaluation pipeline."""
    print("====== [Step 3/3] Running Evaluation Pipeline ======")
    # 1. Load artifacts
    evaluator = ModelEvaluator()
    if not evaluator.load_artifacts():
        sys.exit(1)
        
    # 2. Get the preprocessed data
    preprocessor = DataPreprocessor()
    _, val_X, test_X, _ = preprocessor.run()
    
    if test_X is None:
        print("❌ Halting pipeline due to preprocessing failure.")
        sys.exit(1)
        
    # 3. Perform evaluation
    results = evaluator.evaluate(test_X, val_X)
    
    # 4. Plot results
    if not results.empty:
        evaluator.plot_results(results)

    print("✅ Evaluation pipeline complete.")

def main():
    """Main function to parse arguments and run selected pipelines."""
    parser = argparse.ArgumentParser(description="NO2 Anomaly Detection Pipeline")
    parser.add_argument(
        'action', 
        nargs='?', 
        default='all', 
        choices=['load', 'train', 'evaluate', 'all'],
        help="Action to perform: 'load' data, 'train' model, 'evaluate' model, or run 'all' steps (default)."
    )
    args = parser.parse_args()

    print(f"🚀 Starting pipeline with action: '{args.action}'")

    if args.action == 'all':
        run_load_data()
        run_train_pipeline()
        run_evaluation_pipeline()
        print("\n🎉======= Full Pipeline Completed Successfully! =======🎉")
        print("Check the 'reports/figures' directory for all output plots.")

    elif args.action == 'load':
        run_load_data()
        
    elif args.action == 'train':
        run_train_pipeline()

    elif args.action == 'evaluate':
        run_evaluation_pipeline()

if __name__ == '__main__':
    main()
