"""
Quick start example for CS 412 Research Project
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from utils.config import config
from feature_engineering.feature_extractor import FeatureEngineer
from models.baseline_models import BaselineModels


def main():
    """Quick start demonstration"""
    print("CS 412 Research Project - Quick Start")
    print("=" * 40)
    
    # Load configuration
    print("\n1. Loading configuration...")
    print(f"Data path: {config.get_data_path('raw')}")
    print(f"Test size: {config.get_evaluation_params()['test_size']}")
    
    # Create sample data
    print("\n2. Creating sample data...")
    sample_data = {
        'business_id': ['b1', 'b2', 'b3', 'b4', 'b5'],
        'stars': [4.5, 3.0, 5.0, 2.5, 4.0],
        'review_count': [100, 50, 200, 25, 150],
        'is_open': [1, 0, 1, 0, 1],
        'categories': ['Restaurants', 'Shopping', 'Restaurants', 'Services', 'Restaurants']
    }
    
    df = pd.DataFrame(sample_data)
    print(f"Sample data shape: {df.shape}")
    print(df.head())
    
    # Feature engineering
    print("\n3. Feature engineering...")
    feature_engineer = FeatureEngineer()
    df_with_features = feature_engineer.engineer_features(df)
    
    print(f"Original shape: {df.shape}")
    print(f"With features shape: {df_with_features.shape}")
    
    # Select features
    X, feature_names = feature_engineer.select_features(df_with_features)
    y = df_with_features['is_open']
    
    print(f"Feature matrix shape: {X.shape}")
    print(f"Target distribution: {y.value_counts().to_dict()}")
    print(f"Selected features: {feature_names}")
    
    # Train baseline models
    print("\n4. Training baseline models...")
    baseline_models = BaselineModels()
    results = baseline_models.train_all_models(X, y)
    
    # Compare models
    comparison_df = baseline_models.compare_models()
    print("\nModel Comparison:")
    print(comparison_df)
    
    print("\n✅ Quick start completed successfully!")
    print("\nNext steps:")
    print("1. Download Yelp dataset to data/raw/")
    print("2. Run: python main.py")
    print("3. Check results/ directory for outputs")


if __name__ == "__main__":
    main()
