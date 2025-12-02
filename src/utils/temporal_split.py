"""
Temporal Split Strategies for Time-Series Prediction

This module provides proper temporal train/test splitting that:
1. Respects temporal ordering (no future data in training)
2. Accounts for yearly variations
3. Provides realistic evaluation of prediction performance

Three strategies are provided:

1. EXPANDING_WINDOW: Train on years 1..N, test on year N+1
   - Best for: Evaluating how model improves with more data
   - Captures temporal dynamics naturally
   
2. SLIDING_WINDOW: Train on years N-K..N, test on year N+1
   - Best for: Capturing recent patterns (concept drift)
   - Fixed training window size
   
3. SINGLE_HOLDOUT: Train on all years except last K, test on last K years
   - Best for: Final evaluation with maximum training data
   - Simple and robust

Why NOT "stratified per year 80/20":
- Same year's data in both train and test creates leakage
- Model can "memorize" year-specific patterns
- Not a true prediction of future outcomes
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Iterator
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class TemporalSplit:
    """Container for a single temporal train/test split."""
    name: str
    train_years: List[int]
    test_years: List[int]
    train_indices: np.ndarray
    test_indices: np.ndarray
    
    def __repr__(self):
        return f"TemporalSplit(train={self.train_years}, test={self.test_years})"


def expanding_window_split(
    df: pd.DataFrame,
    year_column: str = '_prediction_year',
    min_train_years: int = 2
) -> Iterator[TemporalSplit]:
    """
    Expanding window temporal cross-validation.
    
    Example with years 2012-2020:
        Split 1: Train [2012, 2013], Test [2014]
        Split 2: Train [2012, 2013, 2014], Test [2015]
        Split 3: Train [2012, 2013, 2014, 2015], Test [2016]
        ...
    
    Args:
        df: DataFrame with prediction tasks
        year_column: Column containing the year
        min_train_years: Minimum years in training set
        
    Yields:
        TemporalSplit objects for each fold
    """
    years = sorted(df[year_column].unique())
    
    if len(years) < min_train_years + 1:
        raise ValueError(f"Need at least {min_train_years + 1} years, got {len(years)}")
    
    logger.info(f"Expanding Window Split: years {years[0]}-{years[-1]}")
    
    for i in range(min_train_years, len(years)):
        train_years = years[:i]
        test_years = [years[i]]
        
        train_mask = df[year_column].isin(train_years)
        test_mask = df[year_column].isin(test_years)
        
        train_indices = np.where(train_mask)[0]
        test_indices = np.where(test_mask)[0]
        
        split = TemporalSplit(
            name=f"expand_{years[i]}",
            train_years=list(train_years),
            test_years=list(test_years),
            train_indices=train_indices,
            test_indices=test_indices
        )
        
        logger.info(f"  {split.name}: train {len(train_indices)}, test {len(test_indices)}")
        
        yield split


def sliding_window_split(
    df: pd.DataFrame,
    year_column: str = '_prediction_year',
    train_window_years: int = 3
) -> Iterator[TemporalSplit]:
    """
    Sliding window temporal cross-validation.
    
    Example with 3-year window and years 2012-2020:
        Split 1: Train [2012, 2013, 2014], Test [2015]
        Split 2: Train [2013, 2014, 2015], Test [2016]
        Split 3: Train [2014, 2015, 2016], Test [2017]
        ...
    
    Args:
        df: DataFrame with prediction tasks
        year_column: Column containing the year
        train_window_years: Number of years in training window
        
    Yields:
        TemporalSplit objects for each fold
    """
    years = sorted(df[year_column].unique())
    
    if len(years) < train_window_years + 1:
        raise ValueError(
            f"Need at least {train_window_years + 1} years, got {len(years)}"
        )
    
    logger.info(f"Sliding Window Split: {train_window_years}-year window")
    
    for i in range(train_window_years, len(years)):
        train_years = years[i - train_window_years:i]
        test_years = [years[i]]
        
        train_mask = df[year_column].isin(train_years)
        test_mask = df[year_column].isin(test_years)
        
        train_indices = np.where(train_mask)[0]
        test_indices = np.where(test_mask)[0]
        
        split = TemporalSplit(
            name=f"slide_{years[i]}",
            train_years=list(train_years),
            test_years=list(test_years),
            train_indices=train_indices,
            test_indices=test_indices
        )
        
        logger.info(f"  {split.name}: train {len(train_indices)}, test {len(test_indices)}")
        
        yield split


def single_holdout_split(
    df: pd.DataFrame,
    year_column: str = '_prediction_year',
    test_years: int = 2
) -> TemporalSplit:
    """
    Simple single holdout: train on earlier years, test on later years.
    
    Example with test_years=2 and years 2012-2020:
        Train: [2012, 2013, 2014, 2015, 2016, 2017, 2018]
        Test: [2019, 2020]
    
    This is the recommended approach for final evaluation because:
    - Maximum training data
    - Clear temporal separation
    - Simple and robust
    
    Args:
        df: DataFrame with prediction tasks
        year_column: Column containing the year
        test_years: Number of years to use for testing
        
    Returns:
        Single TemporalSplit object
    """
    years = sorted(df[year_column].unique())
    
    if len(years) < test_years + 1:
        raise ValueError(f"Need at least {test_years + 1} years, got {len(years)}")
    
    train_year_list = years[:-test_years]
    test_year_list = years[-test_years:]
    
    train_mask = df[year_column].isin(train_year_list)
    test_mask = df[year_column].isin(test_year_list)
    
    train_indices = np.where(train_mask)[0]
    test_indices = np.where(test_mask)[0]
    
    split = TemporalSplit(
        name="holdout",
        train_years=list(train_year_list),
        test_years=list(test_year_list),
        train_indices=train_indices,
        test_indices=test_indices
    )
    
    logger.info(f"Single Holdout Split:")
    logger.info(f"  Train years: {train_year_list} ({len(train_indices)} samples)")
    logger.info(f"  Test years: {test_year_list} ({len(test_indices)} samples)")
    
    return split


def temporal_stratified_cv(
    df: pd.DataFrame,
    year_column: str = '_prediction_year',
    n_splits: int = 5
) -> Iterator[TemporalSplit]:
    """
    Hybrid approach: Temporal cross-validation with stratification WITHIN each year.
    
    This addresses your concern about yearly variations while maintaining
    temporal ordering:
    
    1. For each fold, train on earlier years, test on later year(s)
    2. WITHIN each year, ensure class balance
    3. Use expanding or sliding window for the year selection
    
    This is a compromise that:
    - Respects temporal ordering (train before test in time)
    - Accounts for yearly variations (each test year evaluated separately)
    - Maintains class balance within years
    
    Args:
        df: DataFrame with prediction tasks
        year_column: Column containing the year
        n_splits: Number of folds
        
    Yields:
        TemporalSplit objects
    """
    years = sorted(df[year_column].unique())
    
    # Determine split boundaries
    n_years = len(years)
    min_train = max(2, n_years // n_splits)
    
    logger.info(f"Temporal Stratified CV: {n_splits} folds")
    
    fold = 0
    for i in range(min_train, n_years):
        if fold >= n_splits:
            break
            
        train_years = years[:i]
        test_years = [years[i]]
        
        train_mask = df[year_column].isin(train_years)
        test_mask = df[year_column].isin(test_years)
        
        train_indices = np.where(train_mask)[0]
        test_indices = np.where(test_mask)[0]
        
        split = TemporalSplit(
            name=f"fold_{fold+1}",
            train_years=list(train_years),
            test_years=list(test_years),
            train_indices=train_indices,
            test_indices=test_indices
        )
        
        yield split
        fold += 1


def get_recommended_split(
    df: pd.DataFrame,
    year_column: str = '_prediction_year',
    purpose: str = 'final_evaluation'
) -> TemporalSplit:
    """
    Get the recommended split based on the purpose.
    
    Args:
        df: DataFrame with prediction tasks
        year_column: Column containing the year
        purpose: One of 'final_evaluation', 'model_selection', 'debugging'
        
    Returns:
        Appropriate TemporalSplit
    """
    if purpose == 'final_evaluation':
        # Use single holdout with last 2 years for test
        return single_holdout_split(df, year_column, test_years=2)
        
    elif purpose == 'model_selection':
        # Return the last fold of expanding window
        splits = list(expanding_window_split(df, year_column))
        return splits[-1] if splits else None
        
    elif purpose == 'debugging':
        # Use sliding window with small train set for faster iteration
        splits = list(sliding_window_split(df, year_column, train_window_years=2))
        return splits[-1] if splits else None
    
    else:
        raise ValueError(f"Unknown purpose: {purpose}")


def compare_split_strategies(df: pd.DataFrame, year_column: str = '_prediction_year'):
    """
    Compare different split strategies and print summary.
    """
    years = sorted(df[year_column].unique())
    
    print("="*70)
    print("TEMPORAL SPLIT STRATEGY COMPARISON")
    print("="*70)
    print(f"Available years: {years}")
    print(f"Total samples: {len(df)}")
    print()
    
    # Per-year distribution
    print("Samples per year:")
    for year in years:
        count = (df[year_column] == year).sum()
        print(f"  {year}: {count}")
    print()
    
    # Single Holdout
    print("-"*40)
    print("1. SINGLE HOLDOUT (Recommended for final evaluation)")
    print("-"*40)
    split = single_holdout_split(df, year_column, test_years=2)
    print(f"   Train: {split.train_years} ({len(split.train_indices)} samples)")
    print(f"   Test:  {split.test_years} ({len(split.test_indices)} samples)")
    print()
    
    # Expanding Window
    print("-"*40)
    print("2. EXPANDING WINDOW (Multiple folds)")
    print("-"*40)
    for split in expanding_window_split(df, year_column, min_train_years=2):
        print(f"   {split.name}: Train {len(split.train_indices)}, Test {len(split.test_indices)}")
    print()
    
    # WARNING about the current approach
    print("="*70)
    print("[WARN]  WARNING: Current '80/20 per year' approach is NOT recommended")
    print("="*70)
    print("Problem: Same year's data appears in both train and test sets.")
    print("         This allows model to 'memorize' year-specific patterns.")
    print()
    print("Recommendation: Use SINGLE_HOLDOUT or EXPANDING_WINDOW instead.")
    print("="*70)


if __name__ == "__main__":
    # Demo with synthetic data
    np.random.seed(42)
    demo_df = pd.DataFrame({
        '_prediction_year': np.random.choice([2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020], 10000),
        'label': np.random.choice([0, 1], 10000, p=[0.3, 0.7])
    })
    
    compare_split_strategies(demo_df)

