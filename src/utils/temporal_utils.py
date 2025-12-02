"""
Temporal utility functions for time-aware prediction tasks.

This module provides functions to handle temporal aspects of the prediction pipeline:
- Filter data by cutoff dates
- Compute temporal windows (recent, early, etc.)
- Validate data sufficiency at specific time points
- Create prediction tasks with proper temporal constraints
"""

import pandas as pd
import numpy as np
from datetime import timedelta
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def filter_reviews_by_cutoff(reviews_df: pd.DataFrame, 
                             cutoff_date: pd.Timestamp,
                             date_col: str = 'date') -> pd.DataFrame:
    """
    Filter reviews to only include those before or at the cutoff date.
    
    This is critical for preventing temporal leakage - we should only use
    information available at the prediction time.
    
    Args:
        reviews_df: DataFrame containing reviews
        cutoff_date: Timestamp representing the prediction cutoff
        date_col: Name of the date column (default: 'date')
        
    Returns:
        Filtered DataFrame with only historical reviews
        
    Example:
        >>> reviews = pd.DataFrame({'date': ['2020-01-01', '2021-01-01'], ...})
        >>> cutoff = pd.Timestamp('2020-12-31')
        >>> filtered = filter_reviews_by_cutoff(reviews, cutoff)
        # Only reviews from 2020 and before
    """
    if date_col not in reviews_df.columns:
        raise ValueError(f"Date column '{date_col}' not found in reviews")
    
    # Ensure date column is datetime
    if not pd.api.types.is_datetime64_any_dtype(reviews_df[date_col]):
        reviews_df[date_col] = pd.to_datetime(reviews_df[date_col], errors='coerce')
    
    # Filter by cutoff
    filtered_df = reviews_df[reviews_df[date_col] <= cutoff_date].copy()
    
    logger.debug(f"Filtered reviews: {len(reviews_df)} -> {len(filtered_df)} "
                f"(cutoff: {cutoff_date.date()})")
    
    return filtered_df


def compute_temporal_window(cutoff_date: pd.Timestamp,
                            window_months: int,
                            window_type: str = 'recent') -> Tuple[pd.Timestamp, pd.Timestamp]:
    """
    Compute start and end dates for a temporal window.
    
    Args:
        cutoff_date: Reference date (usually prediction cutoff)
        window_months: Size of window in months
        window_type: Type of window ('recent', 'early', 'prediction')
            - 'recent': [cutoff - window_months, cutoff]
            - 'early': [first_date, first_date + window_months]
            - 'prediction': [cutoff, cutoff + window_months]
            
    Returns:
        Tuple of (start_date, end_date)
        
    Example:
        >>> cutoff = pd.Timestamp('2020-12-31')
        >>> start, end = compute_temporal_window(cutoff, 3, 'recent')
        # Returns (2020-10-01, 2020-12-31) for 3-month recent window
    """
    if window_type == 'recent':
        # Recent window: last N months before cutoff
        end_date = cutoff_date
        start_date = cutoff_date - pd.DateOffset(months=window_months)
        
    elif window_type == 'prediction':
        # Prediction window: N months after cutoff
        start_date = cutoff_date
        end_date = cutoff_date + pd.DateOffset(months=window_months)
        
    elif window_type == 'early':
        # Early window: first N months (need first_date as cutoff_date)
        start_date = cutoff_date
        end_date = cutoff_date + pd.DateOffset(months=window_months)
        
    else:
        raise ValueError(f"Unknown window_type: {window_type}")
    
    return start_date, end_date


def has_sufficient_data(business_id: str,
                       reviews_df: pd.DataFrame,
                       cutoff_date: pd.Timestamp,
                       min_reviews: int = 3,
                       min_days_active: int = 90) -> bool:
    """
    Check if a business has sufficient data at the cutoff date for prediction.
    
    Criteria:
    1. At least min_reviews reviews before cutoff
    2. At least min_days_active days of activity before cutoff
    3. Latest review not too far before cutoff (still active)
    
    Args:
        business_id: ID of the business to check
        reviews_df: DataFrame containing all reviews
        cutoff_date: Prediction cutoff date
        min_reviews: Minimum number of reviews required
        min_days_active: Minimum days of business activity
        
    Returns:
        True if business has sufficient data, False otherwise
    """
    # Get business reviews up to cutoff
    business_reviews = reviews_df[
        (reviews_df['business_id'] == business_id) &
        (reviews_df['date'] <= cutoff_date)
    ]
    
    # Check 1: Minimum review count
    if len(business_reviews) < min_reviews:
        return False
    
    # Check 2: Minimum days active
    first_review = business_reviews['date'].min()
    days_active = (cutoff_date - first_review).days
    if days_active < min_days_active:
        return False
    
    # Check 3: Recent activity (last review within 6 months of cutoff)
    # This ensures business is still active at cutoff time
    last_review = business_reviews['date'].max()
    days_since_last = (cutoff_date - last_review).days
    if days_since_last > 180:  # 6 months
        return False
    
    return True


def extract_year_from_date(date: pd.Timestamp) -> int:
    """
    Extract year from a timestamp.
    
    Args:
        date: Pandas Timestamp
        
    Returns:
        Year as integer
    """
    return date.year


def create_prediction_tasks(business_df: pd.DataFrame,
                           reviews_df: pd.DataFrame,
                           prediction_years: List[int],
                           prediction_window_months: int = 12,
                           min_reviews: int = 3,
                           tasks_per_business: str = 'multiple') -> pd.DataFrame:
    """
    Create prediction tasks for temporal validation.
    
    A prediction task is defined by:
    - business_id: Which business to predict
    - cutoff_date: Date at which features are computed
    - target_date: Date at which label is evaluated
    - prediction_year: Year for stratification
    
    Args:
        business_df: DataFrame with business information
        reviews_df: DataFrame with all reviews
        prediction_years: List of years to create tasks for (e.g., [2012, 2013, ..., 2020])
        prediction_window_months: How many months ahead to predict (default: 12)
        min_reviews: Minimum reviews required at cutoff date
        tasks_per_business: 'multiple' or 'single'
            - 'multiple': Create one task per business per valid year
            - 'single': Create one task per business (random year)
            
    Returns:
        DataFrame with columns:
        - business_id
        - cutoff_date
        - target_date
        - prediction_year
        - is_valid (bool indicating if task has sufficient data)
        
    Example:
        >>> tasks = create_prediction_tasks(
        ...     business_df, reviews_df,
        ...     prediction_years=[2018, 2019, 2020],
        ...     prediction_window_months=12
        ... )
        # Creates tasks like:
        # - Business A, cutoff=2018-12-31, target=2019-12-31
        # - Business A, cutoff=2019-12-31, target=2020-12-31
        # - Business B, cutoff=2018-12-31, target=2019-12-31
        # etc.
    """
    logger.info(f"Creating prediction tasks for years: {prediction_years}")
    logger.info(f"Prediction window: {prediction_window_months} months")
    logger.info(f"Tasks per business: {tasks_per_business}")
    
    tasks = []
    
    # Ensure reviews date column is datetime
    if not pd.api.types.is_datetime64_any_dtype(reviews_df['date']):
        reviews_df['date'] = pd.to_datetime(reviews_df['date'], errors='coerce')
    
    # For each business
    for idx, business in business_df.iterrows():
        business_id = business['business_id']
        
        # Get all reviews for this business
        business_reviews = reviews_df[reviews_df['business_id'] == business_id]
        
        if len(business_reviews) == 0:
            continue
        
        # Determine valid years for this business
        first_review_year = business_reviews['date'].min().year
        last_review_year = business_reviews['date'].max().year
        
        # Valid prediction years: business must have existed before the year
        # and still have data after the prediction target
        valid_years = [
            year for year in prediction_years
            if year >= first_review_year and year < last_review_year
        ]
        
        if len(valid_years) == 0:
            continue
        
        # Decide how many tasks to create
        if tasks_per_business == 'single':
            # Randomly select one year
            selected_years = [np.random.choice(valid_years)]
        else:  # 'multiple'
            selected_years = valid_years
        
        # Create tasks for selected years
        for year in selected_years:
            # Define cutoff and target dates
            cutoff_date = pd.Timestamp(f'{year}-12-31')
            target_date = cutoff_date + pd.DateOffset(months=prediction_window_months)
            
            # Check if business has sufficient data at cutoff
            is_valid = has_sufficient_data(
                business_id, 
                reviews_df, 
                cutoff_date,
                min_reviews=min_reviews
            )
            
            # Create task
            task = {
                'business_id': business_id,
                'cutoff_date': cutoff_date,
                'target_date': target_date,
                'prediction_year': year,
                'is_valid': is_valid,
                # Add business metadata for convenience
                'business_name': business.get('name', ''),
                'business_city': business.get('city', ''),
                'business_state': business.get('state', ''),
            }
            
            tasks.append(task)
    
    # Convert to DataFrame
    tasks_df = pd.DataFrame(tasks)
    
    # Log statistics
    logger.info(f"\nPrediction tasks created:")
    logger.info(f"  Total tasks: {len(tasks_df):,}")
    logger.info(f"  Valid tasks: {tasks_df['is_valid'].sum():,}")
    logger.info(f"  Invalid tasks: {(~tasks_df['is_valid']).sum():,}")
    logger.info(f"  Unique businesses: {tasks_df['business_id'].nunique():,}")
    
    if len(tasks_df) > 0:
        logger.info(f"\nTasks per year:")
        for year in sorted(tasks_df['prediction_year'].unique()):
            year_count = (tasks_df['prediction_year'] == year).sum()
            year_valid = tasks_df[tasks_df['prediction_year'] == year]['is_valid'].sum()
            logger.info(f"  {year}: {year_count:,} total, {year_valid:,} valid")
    
    return tasks_df


def filter_tasks_by_date_range(tasks_df: pd.DataFrame,
                               reviews_df: pd.DataFrame,
                               min_date: Optional[pd.Timestamp] = None,
                               max_date: Optional[pd.Timestamp] = None) -> pd.DataFrame:
    """
    Filter prediction tasks to ensure review data coverage.
    
    Args:
        tasks_df: DataFrame of prediction tasks
        reviews_df: DataFrame of all reviews
        min_date: Minimum date for review data (default: earliest review)
        max_date: Maximum date for review data (default: latest review)
        
    Returns:
        Filtered tasks DataFrame
    """
    if min_date is None:
        min_date = reviews_df['date'].min()
    if max_date is None:
        max_date = reviews_df['date'].max()
    
    logger.info(f"Filtering tasks by date range: {min_date.date()} to {max_date.date()}")
    
    # Filter tasks where cutoff and target are within data range
    valid_tasks = tasks_df[
        (tasks_df['cutoff_date'] >= min_date) &
        (tasks_df['target_date'] <= max_date)
    ].copy()
    
    logger.info(f"Tasks after date filtering: {len(tasks_df)} -> {len(valid_tasks)}")
    
    return valid_tasks