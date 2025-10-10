"""
Main execution script for CS 412 Research Project
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
from models.ensemble_framework import UserWeightedEnsemble
from evaluation.evaluator import ModelEvaluator
from utils.config import config


def setup_logging():
    """Setup logging configuration"""
    Path("logs").mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/project.log'),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)


def main():
    """Main execution function"""
    logger = setup_logging()
    logger.info("Starting CS 412 Research Project execution...")
    
    try:
        # Step 1: Data Processing
        logger.info("Step 1: Processing Yelp dataset...")
        processor = YelpDataProcessor()
        
        # Process sample data for testing (set sample_size=None for full dataset)
        sample_size = 10000
        processed_data = processor.process_all_data(sample_size=sample_size)
        
        # Create merged dataset
        merged_df = processor.create_merged_dataset()
        logger.info(f"Merged dataset shape: {merged_df.shape}")
        
        # Step 2: Feature Engineering
        logger.info("Step 2: Feature engineering...")
        feature_engineer = FeatureEngineer()
        
        # Engineer features
        df_with_features = feature_engineer.engineer_features(merged_df)
        
        # Select features for modeling
        X, feature_names = feature_engineer.select_features(df_with_features)
        y = df_with_features['is_open']
        
        logger.info(f"Feature matrix shape: {X.shape}")
        logger.info(f"Target distribution: {y.value_counts().to_dict()}")
        
        # Step 3: Baseline Models
        logger.info("Step 3: Training baseline models...")
        baseline_models = BaselineModels()
        
        # Train all baseline models
        baseline_results = baseline_models.train_all_models(X, y)
        
        # Save baseline models
        baseline_models.save_models()
        
        # Step 4: Ensemble Framework
        logger.info("Step 4: Training ensemble framework...")
        ensemble = UserWeightedEnsemble()
        
        # Create user-weighted features
        df_weighted = ensemble.create_user_weights(df_with_features)
        df_aggregated = ensemble.aggregate_user_weighted_features(df_weighted)
        
        # Create category features
        df_categorized = ensemble.create_category_features(df_aggregated)
        
        # Prepare data for ensemble
        X_ensemble, _ = feature_engineer.select_features(df_categorized)
        y_ensemble = df_categorized['is_open']
        categories = df_categorized['category_group']
        
        # Train ensemble
        ensemble.train_ensemble_model(X_ensemble, y_ensemble)
        ensemble.train_category_models(X_ensemble, y_ensemble, categories)
        
        # Evaluate ensemble
        ensemble_results = ensemble.evaluate_ensemble(X_ensemble, y_ensemble, categories)
        
        # Save ensemble models
        ensemble.save_ensemble()
        
        # Step 5: Evaluation and Comparison
        logger.info("Step 5: Evaluating and comparing models...")
        evaluator = ModelEvaluator()
        
        # Compare all models
        all_results = {
            **baseline_results,
            'ensemble': ensemble_results
        }
        
        # Generate comparison
        comparison_df = evaluator.compare_models(all_results)
        print("\nModel Comparison:")
        print(comparison_df)
        
        # Save results
        evaluator.save_results(all_results)
        evaluator.generate_report(all_results)
        
        logger.info("Project execution completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during execution: {e}")
        raise


if __name__ == "__main__":
    main()
