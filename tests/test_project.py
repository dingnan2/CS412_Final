"""
Test suite for CS 412 Research Project
"""

import unittest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from data_processing.data_processor import YelpDataProcessor
from feature_engineering.feature_extractor import FeatureEngineer, SentimentAnalyzer
from models.baseline_models import BaselineModels
from utils.config import config


class TestDataProcessor(unittest.TestCase):
    """Test data processing functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.processor = YelpDataProcessor()
        
    def test_config_loading(self):
        """Test configuration loading"""
        self.assertIsNotNone(config.get('data.raw_path'))
        self.assertIsNotNone(config.get('models.logistic_regression'))
    
    def test_feature_creation(self):
        """Test feature creation methods"""
        # Create sample data
        sample_data = {
            'business_id': ['b1', 'b2', 'b3'],
            'stars': [4.5, 3.0, 5.0],
            'review_count': [100, 50, 200],
            'is_open': [1, 0, 1]
        }
        df = pd.DataFrame(sample_data)
        
        # Test business features
        business_features = self.processor.create_business_features(df)
        self.assertIsInstance(business_features, pd.DataFrame)
        self.assertEqual(len(business_features), 3)


class TestFeatureEngineering(unittest.TestCase):
    """Test feature engineering functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.feature_engineer = FeatureEngineer()
        self.sentiment_analyzer = SentimentAnalyzer()
    
    def test_sentiment_analysis(self):
        """Test sentiment analysis"""
        test_text = "This restaurant is amazing! Great food and service."
        sentiment = self.sentiment_analyzer.analyze_sentiment(test_text)
        
        self.assertIn('polarity', sentiment)
        self.assertIn('subjectivity', sentiment)
        self.assertIn('compound', sentiment)
        self.assertGreater(sentiment['polarity'], 0)  # Should be positive
    
    def test_feature_selection(self):
        """Test feature selection"""
        # Create sample data with features
        sample_data = {
            'business_id': ['b1', 'b2'],
            'stars': [4.5, 3.0],
            'review_count': [100, 50],
            'is_open': [1, 0],
            'feature1': [1, 0],
            'feature2': [0, 1]
        }
        df = pd.DataFrame(sample_data)
        
        X, feature_names = self.feature_engineer.select_features(df)
        
        self.assertIsInstance(X, pd.DataFrame)
        self.assertIsInstance(feature_names, list)
        self.assertNotIn('business_id', feature_names)
        self.assertNotIn('is_open', feature_names)


class TestBaselineModels(unittest.TestCase):
    """Test baseline models functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.baseline_models = BaselineModels()
    
    def test_model_initialization(self):
        """Test model initialization"""
        self.assertIn('logistic_regression', self.baseline_models.models)
        self.assertIn('decision_tree', self.baseline_models.models)
        self.assertIn('random_forest', self.baseline_models.models)
    
    def test_data_preparation(self):
        """Test data preparation"""
        # Create sample data
        X = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [0, 1, 0, 1, 0]
        })
        y = pd.Series([1, 0, 1, 0, 1])
        
        X_train, X_test, y_train, y_test = self.baseline_models.prepare_data(X, y)
        
        self.assertEqual(len(X_train), 4)  # 80% of 5
        self.assertEqual(len(X_test), 1)   # 20% of 5
        self.assertEqual(len(y_train), 4)
        self.assertEqual(len(y_test), 1)


class TestUtils(unittest.TestCase):
    """Test utility functions"""
    
    def test_safe_divide(self):
        """Test safe division function"""
        from utils.utils import safe_divide
        
        numerator = np.array([10, 20, 30])
        denominator = np.array([2, 0, 5])  # One zero division
        
        result = safe_divide(numerator, denominator, default=0.0)
        
        self.assertEqual(result[0], 5.0)   # 10/2
        self.assertEqual(result[1], 0.0)   # Default for division by zero
        self.assertEqual(result[2], 6.0)    # 30/5
    
    def test_calculate_class_weights(self):
        """Test class weight calculation"""
        from utils.utils import calculate_class_weights
        
        y = np.array([0, 0, 0, 1, 1])  # Imbalanced classes
        weights = calculate_class_weights(y)
        
        self.assertIn(0, weights)
        self.assertIn(1, weights)
        self.assertGreater(weights[1], weights[0])  # Minority class should have higher weight


def run_tests():
    """Run all tests"""
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_classes = [
        TestDataProcessor,
        TestFeatureEngineering,
        TestBaselineModels,
        TestUtils
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    if success:
        print("\n✅ All tests passed!")
    else:
        print("\n❌ Some tests failed!")
        exit(1)
