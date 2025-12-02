"""
Label inference utilities for business status prediction.

This module handles the challenging task of inferring business operational status
at specific points in time, given that we only have:
1. Final status (is_open field as of dataset end date)
2. Review activity timestamps

The key challenge: Yelp doesn't provide exact closure dates, so we must infer
when a business closed based on review activity patterns.
"""

import pandas as pd
import numpy as np
from datetime import timedelta
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def estimate_closure_date(business_id: str,
                          reviews_df: pd.DataFrame,
                          final_is_open: int,
                          closure_lag_days: int = 180) -> Optional[pd.Timestamp]:
    """
    Estimate when a business closed based on review activity.
    
    Assumption: A business typically stops receiving reviews ~6 months before
    officially closing (customers notice declining quality/service).
    
    Args:
        business_id: ID of the business
        reviews_df: DataFrame containing all reviews
        final_is_open: Final status (1=open, 0=closed)
        closure_lag_days: Days after last review to assume closure (default: 180)
        
    Returns:
        Estimated closure date (Timestamp) or None if business is open
        
    Logic:
        - If final_is_open == 1: Return None (still open)
        - If final_is_open == 0: 
            - Find last review date
            - Estimate closure = last_review + closure_lag_days
    """
    if final_is_open == 1:
        # Business is currently open
        return None
    
    # Business is closed - estimate when
    business_reviews = reviews_df[reviews_df['business_id'] == business_id]
    
    if len(business_reviews) == 0:
        logger.warning(f"No reviews found for business {business_id}")
        return None
    
    last_review_date = business_reviews['date'].max()
    estimated_closure = last_review_date + timedelta(days=closure_lag_days)
    
    return estimated_closure


def infer_business_status(business_id: str,
                         target_date: pd.Timestamp,
                         business_df: pd.DataFrame,
                         reviews_df: pd.DataFrame,
                         closure_lag_days: int = 180) -> Tuple[int, float]:
    """
    Infer if a business was open or closed at a specific target date.
    
    This is the core function for generating labels in temporal validation.
    
    Args:
        business_id: ID of the business
        target_date: Date at which to evaluate status
        business_df: DataFrame with business information (including is_open)
        reviews_df: DataFrame with all reviews
        closure_lag_days: Days after last review to assume closure
        
    Returns:
        Tuple of (status, confidence)
        - status: 0 (closed) or 1 (open)
        - confidence: 0.0 to 1.0 indicating inference confidence
        
    Logic Flow:
        1. Get final status (is_open field)
        2. If currently open:
            - Was it open at target_date? 
            - Check if target_date > first_review_date
            - High confidence if far from boundaries
        3. If currently closed:
            - Estimate closure date
            - If target_date < estimated_closure: was open (1)
            - If target_date >= estimated_closure: was closed (0)
            - Confidence decreases near estimated closure date
    """
    # Get business information
    business_info = business_df[business_df['business_id'] == business_id]
    
    if len(business_info) == 0:
        logger.warning(f"Business {business_id} not found in business_df")
        return 0, 0.0
    
    business_info = business_info.iloc[0]
    final_is_open = business_info['is_open']
    
    # Get business reviews
    business_reviews = reviews_df[reviews_df['business_id'] == business_id]
    
    if len(business_reviews) == 0:
        logger.warning(f"No reviews for business {business_id}")
        return 0, 0.0
    
    first_review = business_reviews['date'].min()
    last_review = business_reviews['date'].max()
    
    # Case 1: Business is currently OPEN (final_is_open == 1)
    if final_is_open == 1:
        # Check if business had started operations by target_date
        if target_date < first_review:
            # Business didn't exist yet
            return 0, 0.0  # Not applicable / no data
        
        # Business was operating at target_date (assume continuous operation)
        status = 1
        
        # Confidence: High if target_date is well within operational period
        # Lower if target_date is very recent (might have just opened)
        days_since_start = (target_date - first_review).days
        
        if days_since_start < 90:  # Less than 3 months old
            confidence = 0.7  # Medium confidence (new business)
        else:
            confidence = 0.95  # High confidence (established business)
        
        return status, confidence
    
    # Case 2: Business is currently CLOSED (final_is_open == 0)
    else:
        # Estimate when it closed
        estimated_closure = estimate_closure_date(
            business_id, reviews_df, final_is_open, closure_lag_days
        )
        
        if estimated_closure is None:
            # Shouldn't happen, but handle gracefully
            logger.warning(f"Could not estimate closure for {business_id}")
            return 0, 0.5
        
        # Compare target_date with estimated closure
        if target_date < estimated_closure:
            # Business was still open at target_date
            status = 1
            
            # Confidence: Lower if target_date is close to closure
            days_to_closure = (estimated_closure - target_date).days
            
            if days_to_closure < 90:  # Within 3 months of closure
                confidence = 0.6  # Lower confidence (uncertain period)
            elif days_to_closure < 180:  # 3-6 months before closure
                confidence = 0.8  # Medium confidence
            else:  # More than 6 months before closure
                confidence = 0.9  # High confidence
        
        else:
            # Business had already closed by target_date
            status = 0
            
            # Confidence: Lower if target_date is close to closure
            days_after_closure = (target_date - estimated_closure).days
            
            if days_after_closure < 90:  # Within 3 months after closure
                confidence = 0.6  # Lower confidence (uncertain period)
            else:  # Well after closure
                confidence = 0.9  # High confidence
        
        return status, confidence


def calculate_label_confidence(business_id: str,
                              target_date: pd.Timestamp,
                              business_df: pd.DataFrame,
                              reviews_df: pd.DataFrame) -> float:
    """
    Calculate confidence score for a label without inferring the label itself.
    
    This is useful for pre-filtering tasks before generating full labels.
    
    Args:
        business_id: ID of the business
        target_date: Date at which to evaluate
        business_df: DataFrame with business information
        reviews_df: DataFrame with reviews
        
    Returns:
        Confidence score (0.0 to 1.0)
        
    Factors affecting confidence:
        1. Review activity pattern (consistent vs sporadic)
        2. Distance from operational boundaries (opening/closing)
        3. Review volume (more reviews = higher confidence)
    """
    business_reviews = reviews_df[reviews_df['business_id'] == business_id]
    
    if len(business_reviews) == 0:
        return 0.0
    
    # Factor 1: Review volume (more reviews = more confidence)
    num_reviews = len(business_reviews)
    volume_confidence = min(num_reviews / 50.0, 1.0)  # Cap at 50 reviews
    
    # Factor 2: Review consistency (regular reviews = more confidence)
    # Calculate time gaps between consecutive reviews
    sorted_dates = business_reviews['date'].sort_values()
    time_gaps = sorted_dates.diff().dt.days.dropna()
    
    if len(time_gaps) > 0:
        avg_gap = time_gaps.mean()
        gap_std = time_gaps.std()
        
        # Consistent reviews (small std) = high confidence
        if gap_std < 30:  # Reviews every ~1 month consistently
            consistency_confidence = 1.0
        elif gap_std < 90:  # Somewhat regular
            consistency_confidence = 0.8
        else:  # Sporadic
            consistency_confidence = 0.6
    else:
        consistency_confidence = 0.5
    
    # Factor 3: Distance from boundaries
    first_review = business_reviews['date'].min()
    last_review = business_reviews['date'].max()
    
    # Check if target_date is too close to boundaries
    days_from_start = (target_date - first_review).days
    days_from_end = (last_review - target_date).days
    
    min_distance = min(days_from_start, days_from_end)
    
    if min_distance < 90:  # Within 3 months of boundary
        boundary_confidence = 0.6
    elif min_distance < 180:  # 3-6 months from boundary
        boundary_confidence = 0.8
    else:  # Well within operational period
        boundary_confidence = 1.0
    
    # Overall confidence: weighted average
    overall_confidence = (
        0.3 * volume_confidence +
        0.3 * consistency_confidence +
        0.4 * boundary_confidence
    )
    
    return overall_confidence


def batch_infer_labels(tasks_df: pd.DataFrame,
                       business_df: pd.DataFrame,
                       reviews_df: pd.DataFrame,
                       closure_lag_days: int = 180) -> pd.DataFrame:
    """
    Infer labels for all tasks in batch.
    
    Args:
        tasks_df: DataFrame of prediction tasks
        business_df: DataFrame with business information
        reviews_df: DataFrame with reviews
        closure_lag_days: Days after last review to assume closure
        
    Returns:
        tasks_df with added columns:
        - label: 0 or 1 (business status at target_date)
        - label_confidence: 0.0 to 1.0
    """
    logger.info(f"Inferring labels for {len(tasks_df)} tasks...")
    
    labels = []
    confidences = []
    
    for idx, task in tasks_df.iterrows():
        status, confidence = infer_business_status(
            business_id=task['business_id'],
            target_date=task['target_date'],
            business_df=business_df,
            reviews_df=reviews_df,
            closure_lag_days=closure_lag_days
        )
        
        labels.append(status)
        confidences.append(confidence)
        
        if (idx + 1) % 1000 == 0:
            logger.info(f"  Processed {idx + 1}/{len(tasks_df)} tasks")
    
    tasks_df['label'] = labels
    tasks_df['label_confidence'] = confidences
    
    # Log statistics
    logger.info(f"\nLabel inference complete:")
    logger.info(f"  Open (1): {sum(labels)} ({sum(labels)/len(labels)*100:.1f}%)")
    logger.info(f"  Closed (0): {len(labels) - sum(labels)} ({(len(labels)-sum(labels))/len(labels)*100:.1f}%)")
    logger.info(f"  Avg confidence: {np.mean(confidences):.3f}")
    logger.info(f"  Min confidence: {np.min(confidences):.3f}")
    logger.info(f"  Max confidence: {np.max(confidences):.3f}")
    
    return tasks_df