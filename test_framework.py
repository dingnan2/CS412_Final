"""
Test script for data processing with sample data
"""

import sys
import os
import pandas as pd
import numpy as np
import json
import logging
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

def setup_logging():
    """Setup logging"""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    return logging.getLogger(__name__)

def test_data_loading():
    """Test loading Yelp data"""
    logger = setup_logging()
    
    logger.info("Testing Yelp dataset loading...")
    
    # Check if data files exist
    data_dir = Path("data/raw")
    business_file = data_dir / "yelp_academic_dataset_business.json"
    
    if not business_file.exists():
        logger.error(f"Business file not found: {business_file}")
        return False
    
    logger.info(f"Found business file: {business_file}")
    
    # Load sample data
    try:
        data = []
        with open(business_file, 'r', encoding='utf-8') as file:
            for i, line in enumerate(file):
                if i >= 1000:  # Load first 1000 businesses for testing
                    break
                data.append(json.loads(line.strip()))
        
        df = pd.DataFrame(data)
        logger.info(f"Loaded {len(df)} businesses")
        logger.info(f"Columns: {list(df.columns)}")
        logger.info(f"Shape: {df.shape}")
        
        # Check key columns
        if 'is_open' in df.columns:
            open_count = df['is_open'].sum()
            closed_count = len(df) - open_count
            logger.info(f"Open businesses: {open_count}")
            logger.info(f"Closed businesses: {closed_count}")
        
        if 'stars' in df.columns:
            logger.info(f"Rating range: {df['stars'].min()} - {df['stars'].max()}")
            logger.info(f"Average rating: {df['stars'].mean():.2f}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return False

def test_baseline_models():
    """Test baseline models with sample data"""
    logger = setup_logging()
    
    logger.info("Testing baseline models...")
    
    try:
        # Create sample data
        np.random.seed(42)
        n_samples = 1000
        
        # Generate sample features
        X = pd.DataFrame({
            'stars': np.random.uniform(1, 5, n_samples),
            'review_count': np.random.poisson(50, n_samples),
            'feature1': np.random.normal(0, 1, n_samples),
            'feature2': np.random.normal(0, 1, n_samples)
        })
        
        # Generate sample target (business success)
        y = pd.Series(np.random.binomial(1, 0.7, n_samples))  # 70% success rate
        
        logger.info(f"Sample data shape: {X.shape}")
        logger.info(f"Target distribution: {y.value_counts().to_dict()}")
        
        # Test basic model
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        model = LogisticRegression(random_state=42)
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        logger.info(f"Baseline model accuracy: {accuracy:.4f}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error testing baseline models: {e}")
        return False

def main():
    """Main test function"""
    logger = setup_logging()
    
    print("🧪 CS 412 Research Project - Framework Test")
    print("=" * 45)
    
    # Test 1: Data loading
    print("\n1. Testing data loading...")
    if test_data_loading():
        print("✅ Data loading test passed!")
    else:
        print("❌ Data loading test failed!")
        return
    
    # Test 2: Baseline models
    print("\n2. Testing baseline models...")
    if test_baseline_models():
        print("✅ Baseline models test passed!")
    else:
        print("❌ Baseline models test failed!")
        return
    
    print("\n🎉 All tests passed! Your framework is working!")
    print("\nNext steps:")
    print("1. Run full data processing: python run_baseline.py")
    print("2. Start your experiments!")
    print("3. Follow the timeline in MIDTERM_TIMELINE.md")

if __name__ == "__main__":
    main()
