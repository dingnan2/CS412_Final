"""
Example script for running ensemble framework only
"""

import sys
import os
import logging
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from data_processing.data_processor import YelpDataProcessor
from feature_engineering.feature_extractor import FeatureEngineer
from models.ensemble_framework import UserWeightedEnsemble
from evaluation.evaluator import ModelEvaluator
from utils.config import config


def main():
    """Run ensemble framework only"""
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info("Running ensemble framework...")
    
    # Load processed data
    processor = YelpDataProcessor()
    merged_df = processor.create_merged_dataset()
    
    # Feature engineering
    feature_engineer = FeatureEngineer()
    df_with_features = feature_engineer.engineer_features(merged_df)
    
    # Create user-weighted features
    ensemble = UserWeightedEnsemble()
    df_weighted = ensemble.create_user_weights(df_with_features)
    df_aggregated = ensemble.aggregate_user_weighted_features(df_weighted)
    df_categorized = ensemble.create_category_features(df_aggregated)
    
    # Select features
    X, feature_names = feature_engineer.select_features(df_categorized)
    y = df_categorized['is_open']
    categories = df_categorized['category_group']
    
    # Train ensemble
    ensemble.train_ensemble_model(X, y)
    ensemble.train_category_models(X, y, categories)
    
    # Evaluate ensemble
    results = ensemble.evaluate_ensemble(X, y, categories)
    
    print("\nEnsemble Results:")
    print(f"Ensemble AUC: {results['ensemble_auc']:.4f}")
    print(f"Category-aware AUC: {results['category_aware_auc']:.4f}")
    print(f"Combined AUC: {results['combined_auc']:.4f}")
    
    # Save results
    evaluator = ModelEvaluator()
    evaluator.save_results(results, "results/ensemble_results.json")
    
    logger.info("Ensemble framework completed!")


if __name__ == "__main__":
    main()
