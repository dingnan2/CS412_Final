"""
Unified Train/Test Split Utilities for Temporal Validation

This module provides a single source of truth for data splitting
to ensure consistency across all experiments.

Key Features:
- Temporal holdout split (train on early years, test on late years)
- Random stratified split (for comparison)
- Consistent interface for all modules
- Prevents temporal data leakage

Usage:
    from utils.split_utils import TemporalSplitter
    
    splitter = TemporalSplitter(
        split_type='temporal_holdout',
        test_size=0.2,
        random_state=42
    )
    
    X_train, X_test, y_train, y_test, train_idx, test_idx = splitter.split(
        X, y, metadata=df[['_prediction_year']]
    )
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, List
import logging

logger = logging.getLogger(__name__)


class TemporalSplitter:
    """
    Unified temporal split utility.
    
    Ensures all experiments use the same split strategy to prevent
    inconsistent results across different phases.
    """
    
    def __init__(self, 
                 split_type: str = 'temporal_holdout',
                 test_size: float = 0.2,
                 random_state: int = 42,
                 train_years: Optional[List[int]] = None,
                 test_years: Optional[List[int]] = None):
        """
        Initialize temporal splitter.
        
        Args:
            split_type: 'temporal_holdout' or 'random'
            test_size: Proportion for test set (used if years not specified)
            random_state: Random seed for reproducibility
            train_years: Explicit list of training years (optional)
            test_years: Explicit list of test years (optional)
        """
        self.split_type = split_type
        self.test_size = test_size
        self.random_state = random_state
        self.train_years = train_years
        self.test_years = test_years
        
        # Validate split type
        valid_types = ['temporal_holdout', 'random']
        if split_type not in valid_types:
            raise ValueError(f"split_type must be one of {valid_types}, got '{split_type}'")
        
        logger.info(f"Initialized TemporalSplitter:")
        logger.info(f"  Split type: {split_type}")
        logger.info(f"  Test size: {test_size}")
        logger.info(f"  Random state: {random_state}")
        if train_years:
            logger.info(f"  Train years: {train_years}")
        if test_years:
            logger.info(f"  Test years: {test_years}")
    
    def split(self, 
              X: np.ndarray, 
              y: np.ndarray,
              metadata: Optional[pd.DataFrame] = None) -> Tuple[np.ndarray, np.ndarray, 
                                                                  np.ndarray, np.ndarray,
                                                                  np.ndarray, np.ndarray]:
        """
        Perform train/test split.
        
        Args:
            X: Feature matrix (numpy array or pandas DataFrame)
            y: Target vector (numpy array or pandas Series)
            metadata: DataFrame with temporal metadata (_prediction_year column)
        
        Returns:
            Tuple of (X_train, X_test, y_train, y_test, train_indices, test_indices)
        """
        # Convert to numpy arrays if needed
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
        
        # Perform split based on type
        if self.split_type == 'temporal_holdout':
            return self._temporal_holdout_split(X, y, metadata)
        elif self.split_type == 'random':
            return self._random_split(X, y)
        else:
            raise ValueError(f"Unknown split_type: {self.split_type}")
    
    def _temporal_holdout_split(self, 
                                X: np.ndarray, 
                                y: np.ndarray,
                                metadata: Optional[pd.DataFrame]) -> Tuple:
        """
        Temporal holdout: train on early years, test on late years.
        
        This is the V2 leakage-free split strategy.
        """
        logger.info("="*70)
        logger.info("TEMPORAL HOLDOUT SPLIT (Leakage-Free)")
        logger.info("="*70)
        
        # Check if metadata is available
        if metadata is None or '_prediction_year' not in metadata.columns:
            logger.warning("No temporal metadata found (_prediction_year column missing)")
            logger.warning("Falling back to random split...")
            return self._random_split(X, y)
        
        # Get available years
        years = sorted(metadata['_prediction_year'].unique())
        logger.info(f"Available years: {years}")
        
        # Determine train/test years
        if self.train_years is not None and self.test_years is not None:
            # Use explicitly specified years
            train_years = self.train_years
            test_years = self.test_years
            logger.info("Using explicitly specified train/test years")
        else:
            # Automatically split based on test_size
            n_test_years = max(1, int(len(years) * self.test_size))
            train_years = years[:-n_test_years]
            test_years = years[-n_test_years:]
            logger.info(f"Automatically determined split (test_size={self.test_size})")
        
        logger.info(f"\nTemporal Holdout Configuration:")
        logger.info(f"  Train years: {train_years}")
        logger.info(f"  Test years: {test_years}")
        
        # Create masks
        train_mask = metadata['_prediction_year'].isin(train_years)
        test_mask = metadata['_prediction_year'].isin(test_years)
        
        train_indices = np.where(train_mask)[0]
        test_indices = np.where(test_mask)[0]
        
        # Extract data
        X_train = X[train_indices]
        X_test = X[test_indices]
        y_train = y[train_indices]
        y_test = y[test_indices]
        
        # Summary statistics
        logger.info(f"\n{'='*70}")
        logger.info(f"TEMPORAL HOLDOUT SUMMARY")
        logger.info(f"{'='*70}")
        logger.info(f"Train set: {len(train_indices):,} samples ({len(train_indices)/len(y)*100:.1f}%)")
        logger.info(f"Test set: {len(test_indices):,} samples ({len(test_indices)/len(y)*100:.1f}%)")
        
        # Check class distribution
        train_class_dist = np.bincount(y_train.astype(int))
        test_class_dist = np.bincount(y_test.astype(int))
        
        logger.info(f"\nClass distribution:")
        logger.info(f"  Train - Class 0: {train_class_dist[0]:,} ({train_class_dist[0]/len(y_train)*100:.1f}%), "
                   f"Class 1: {train_class_dist[1]:,} ({train_class_dist[1]/len(y_train)*100:.1f}%)")
        logger.info(f"  Test  - Class 0: {test_class_dist[0]:,} ({test_class_dist[0]/len(y_test)*100:.1f}%), "
                   f"Class 1: {test_class_dist[1]:,} ({test_class_dist[1]/len(y_test)*100:.1f}%)")
        
        # Validate split
        if len(train_indices) == 0 or len(test_indices) == 0:
            raise ValueError("Split resulted in empty train or test set")
        
        logger.info(f"\n{'='*70}\n")
        
        return X_train, X_test, y_train, y_test, train_indices, test_indices
    
    def _random_split(self, X: np.ndarray, y: np.ndarray) -> Tuple:
        """
        Random stratified split.
        
        WARNING: This may introduce temporal leakage if used with temporal data.
        """
        from sklearn.model_selection import train_test_split
        
        logger.info("="*70)
        logger.info("RANDOM STRATIFIED SPLIT")
        logger.info("="*70)
        logger.warning("⚠️  WARNING: Using RANDOM split - may have temporal leakage!")
        logger.warning("⚠️  This should only be used for comparison purposes.")
        
        # Create indices for tracking
        indices = np.arange(len(X))
        
        X_train, X_test, y_train, y_test, train_idx, test_idx = train_test_split(
            X, y, indices,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y
        )
        
        logger.info(f"Train set: {len(train_idx):,} samples ({len(train_idx)/len(y)*100:.1f}%)")
        logger.info(f"Test set: {len(test_idx):,} samples ({len(test_idx)/len(y)*100:.1f}%)")
        logger.info(f"\n{'='*70}\n")
        
        return X_train, X_test, y_train, y_test, train_idx, test_idx


def get_default_splitter(split_type: str = 'temporal_holdout') -> TemporalSplitter:
    """
    Get a splitter with default project settings.
    
    Args:
        split_type: 'temporal_holdout' or 'random'
    
    Returns:
        TemporalSplitter instance with default settings
    """
    return TemporalSplitter(
        split_type=split_type,
        test_size=0.2,
        random_state=42,
        train_years=[2012, 2013, 2014, 2015, 2016, 2017, 2018],
        test_years=[2019, 2020]
    )


if __name__ == "__main__":
    # Test the splitter
    print("Testing TemporalSplitter...")
    
    # Create dummy data
    np.random.seed(42)
    n_samples = 1000
    X = np.random.randn(n_samples, 10)
    y = np.random.randint(0, 2, n_samples)
    
    # Create metadata with years
    years = np.random.choice([2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020], 
                            n_samples)
    metadata = pd.DataFrame({'_prediction_year': years})
    
    # Test temporal split
    splitter = get_default_splitter('temporal_holdout')
    X_train, X_test, y_train, y_test, train_idx, test_idx = splitter.split(X, y, metadata)
    
    print(f"\nTemporal split successful!")
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")
    
    # Test random split
    splitter_random = TemporalSplitter(split_type='random', random_state=42)
    X_train, X_test, y_train, y_test, train_idx, test_idx = splitter_random.split(X, y)
    
    print(f"\nRandom split successful!")
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")

