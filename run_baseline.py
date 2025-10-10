"""
Example script for running baseline models only
"""

import sys
import os
import logging
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from data_processing.data_processor import YelpDataProcessor
from feature_engineering.feature_extractor import FeatureEngineer
from models.baseline_models import BaselineModels
from evaluation.evaluator import ModelEvaluator
from utils.config import config


def main():
    """Run baseline models only"""
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info("Running baseline models...")
    
    # Load processed data
    processor = YelpDataProcessor()
    merged_df = processor.create_merged_dataset()
    
    # Feature engineering
    feature_engineer = FeatureEngineer()
    df_with_features = feature_engineer.engineer_features(merged_df)
    
    # Select features
    X, feature_names = feature_engineer.select_features(df_with_features)
    y = df_with_features['is_open']
    
    # Train baseline models
    baseline_models = BaselineModels()
    results = baseline_models.train_all_models(X, y)
    
    # Compare models
    evaluator = ModelEvaluator()
    comparison_df = evaluator.compare_models(results)
    
    print("\nBaseline Model Comparison:")
    print(comparison_df)
    
    # Save results
    evaluator.save_results(results, "results/baseline_results.json")
    
    logger.info("Baseline models completed!")


if __name__ == "__main__":
    main()
