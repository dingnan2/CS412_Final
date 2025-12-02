"""
Label Inference V2 - Leakage-Free Design

The key insight: The original label inference used review activity patterns
(last_review_date) to estimate closure. But features also use review activity
patterns, creating circular dependency (label leakage).

SOLUTION: Two approaches to avoid leakage:

Approach A: "Pure Final Status" (Recommended)
- Use ONLY the final is_open status as the label
- Each business gets ONE prediction task
- Cutoff date determines what features are available
- Label is the FUTURE outcome (is_open as of dataset end date)
- This is a TRUE prediction task: predict future status from historical data

Approach B: "Rating-Based Inference" (Alternative)
- Infer closure from RATING trajectory (declining ratings → likely to close)
- Then EXCLUDE rating-based features from the model
- This maintains temporal richness but requires careful feature management

This module implements Approach A as the primary method.
"""

import pandas as pd
import numpy as np
from datetime import timedelta
from typing import Dict, Tuple, Optional, List
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# Approach A: Pure Final Status (Recommended - No Leakage)
# ============================================================================

def create_prediction_tasks_v2(
    business_df: pd.DataFrame,
    reviews_df: pd.DataFrame,
    cutoff_dates: List[pd.Timestamp],
    min_reviews_before_cutoff: int = 3,
    min_days_active_before_cutoff: int = 180
) -> pd.DataFrame:
    """
    Create prediction tasks using the Pure Final Status approach.
    
    Key Principle:
    - Label = final is_open status (ground truth, no inference needed)
    - Features = computed using ONLY data before cutoff_date
    - Prediction task = "Given historical data up to cutoff, predict final status"
    
    This is a TRUE prediction task with NO label leakage because:
    - The label (is_open) is an independent ground truth
    - Features use only historical data
    - No circular dependency
    
    Args:
        business_df: Business data with 'is_open' column
        reviews_df: Review data with 'date' column
        cutoff_dates: List of dates to use as prediction cutoffs
        min_reviews_before_cutoff: Minimum reviews required before cutoff
        min_days_active_before_cutoff: Minimum days of activity before cutoff
        
    Returns:
        DataFrame with prediction tasks
    """
    logger.info("="*70)
    logger.info("CREATING PREDICTION TASKS (V2 - Leakage-Free)")
    logger.info("="*70)
    logger.info(f"Cutoff dates: {len(cutoff_dates)}")
    logger.info(f"Min reviews before cutoff: {min_reviews_before_cutoff}")
    
    tasks = []
    
    for cutoff_date in cutoff_dates:
        logger.info(f"\nProcessing cutoff: {cutoff_date.strftime('%Y-%m-%d')}")
        
        # Filter reviews before cutoff
        reviews_before = reviews_df[reviews_df['date'] < cutoff_date]
        
        # Get businesses with sufficient activity before cutoff
        business_activity = reviews_before.groupby('business_id').agg({
            'date': ['min', 'max', 'count']
        })
        business_activity.columns = ['first_review', 'last_review', 'review_count']
        business_activity = business_activity.reset_index()
        
        # Filter: minimum reviews
        qualified = business_activity[
            business_activity['review_count'] >= min_reviews_before_cutoff
        ]
        
        # Filter: minimum activity period
        qualified['days_active'] = (
            qualified['last_review'] - qualified['first_review']
        ).dt.days
        qualified = qualified[
            qualified['days_active'] >= min_days_active_before_cutoff
        ]
        
        # Filter: last review should be within reasonable time before cutoff
        # (to ensure business was active near cutoff, not abandoned long ago)
        max_gap_days = 365  # At most 1 year gap before cutoff
        qualified['days_to_cutoff'] = (
            cutoff_date - qualified['last_review']
        ).dt.days
        qualified = qualified[qualified['days_to_cutoff'] <= max_gap_days]
        
        logger.info(f"  Qualified businesses: {len(qualified)}")
        
        # Create tasks
        for _, row in qualified.iterrows():
            business_id = row['business_id']
            
            # Get final status from business_df
            business_info = business_df[business_df['business_id'] == business_id]
            if len(business_info) == 0:
                continue
                
            final_is_open = business_info.iloc[0]['is_open']
            
            task = {
                'business_id': business_id,
                'cutoff_date': cutoff_date,
                'prediction_year': cutoff_date.year,
                'label': int(final_is_open),  # Ground truth - no inference!
                'label_confidence': 1.0,  # Perfect confidence (ground truth)
                'label_source': 'final_status',
                'reviews_before_cutoff': row['review_count'],
                'days_active': row['days_active'],
                'days_to_cutoff': row['days_to_cutoff']
            }
            tasks.append(task)
    
    tasks_df = pd.DataFrame(tasks)
    
    # Log statistics
    logger.info(f"\n{'='*70}")
    logger.info("TASK CREATION SUMMARY")
    logger.info(f"{'='*70}")
    logger.info(f"Total tasks: {len(tasks_df)}")
    logger.info(f"Unique businesses: {tasks_df['business_id'].nunique()}")
    
    if len(tasks_df) > 0:
        label_counts = tasks_df['label'].value_counts()
        logger.info(f"Label distribution:")
        logger.info(f"  Open (1): {label_counts.get(1, 0)} ({label_counts.get(1, 0)/len(tasks_df)*100:.1f}%)")
        logger.info(f"  Closed (0): {label_counts.get(0, 0)} ({label_counts.get(0, 0)/len(tasks_df)*100:.1f}%)")
        
        logger.info(f"\nTasks per year:")
        for year in sorted(tasks_df['prediction_year'].unique()):
            count = (tasks_df['prediction_year'] == year).sum()
            logger.info(f"  {year}: {count}")
    
    return tasks_df


def get_leakage_free_features() -> Dict[str, List[str]]:
    """
    Define which features are safe to use with the Pure Final Status approach.
    
    Since labels are NOT inferred from review patterns, we can safely use
    review-based features. The key is ensuring features use only data
    before the cutoff date (handled in feature engineering).
    
    Returns:
        Dict mapping category names to feature lists
    """
    return {
        'A_Static': [
            'stars', 'review_count', 'category_encoded', 'state_encoded',
            'city_encoded', 'has_multiple_categories', 'category_count', 'price_range'
        ],
        'B_Review_Agg': [
            'total_reviews', 'avg_review_stars', 'std_review_stars',
            'days_since_first_review', 'review_recency_ratio', 'review_frequency',
            'total_useful_votes', 'avg_useful_per_review'
        ],
        'C_Sentiment': [
            'avg_sentiment', 'std_sentiment', 'sentiment_volatility',
            'pct_positive_reviews', 'pct_negative_reviews', 'pct_neutral_reviews',
            'avg_text_length', 'std_text_length', 'sentiment_recent_3m'
        ],
        'D_User_Weighted': [
            'avg_reviewer_credibility', 'std_reviewer_credibility',
            'weighted_avg_rating', 'weighted_sentiment',
            'pct_high_credibility_reviewers', 'weighted_useful_votes',
            'avg_reviewer_tenure', 'avg_reviewer_experience', 'review_diversity'
        ],
        'E_Temporal': [
            'rating_recent_vs_all', 'rating_recent_vs_early',
            'reviews_recent_3m_count', 'engagement_recent_vs_all',
            'sentiment_recent_vs_all', 'review_momentum',
            'lifecycle_stage', 'rating_trend_3m'
        ],
        'F_Location': [
            'category_avg_success_rate', 'state_avg_success_rate',
            'city_avg_success_rate', 'category_competitiveness',
            'location_density'
        ]
    }


# ============================================================================
# Approach B: Rating-Based Inference (Alternative)
# ============================================================================

def estimate_closure_from_rating(
    business_id: str,
    reviews_df: pd.DataFrame,
    final_is_open: int,
    rating_decline_threshold: float = 0.5,
    min_reviews_for_decline: int = 10
) -> Tuple[Optional[pd.Timestamp], float]:
    """
    Alternative: Estimate closure date from rating trajectory, not review activity.
    
    Logic:
    - If final_is_open == 1: Business is open, no closure
    - If final_is_open == 0:
        - Look for significant rating decline
        - Estimate closure as ~6 months after decline becomes consistent
    
    If using this approach, MUST EXCLUDE these features:
    - avg_review_stars, std_review_stars
    - rating_recent_vs_all, rating_recent_vs_early
    - rating_trend_3m
    - weighted_avg_rating
    
    Args:
        business_id: Business ID
        reviews_df: Review data
        final_is_open: Final status
        rating_decline_threshold: Minimum decline to signal closure
        min_reviews_for_decline: Minimum reviews to calculate decline
        
    Returns:
        Tuple of (estimated_closure_date, confidence)
    """
    if final_is_open == 1:
        return None, 1.0
    
    business_reviews = reviews_df[reviews_df['business_id'] == business_id].copy()
    
    if len(business_reviews) < min_reviews_for_decline:
        # Not enough data to estimate
        return None, 0.3
    
    # Sort by date
    business_reviews = business_reviews.sort_values('date')
    
    # Calculate rolling average rating
    business_reviews['rolling_rating'] = (
        business_reviews['stars'].rolling(window=5, min_periods=3).mean()
    )
    
    # Find first sustained decline
    first_half = business_reviews.head(len(business_reviews) // 2)
    second_half = business_reviews.tail(len(business_reviews) // 2)
    
    if len(first_half) == 0 or len(second_half) == 0:
        return None, 0.3
    
    first_half_avg = first_half['stars'].mean()
    second_half_avg = second_half['stars'].mean()
    
    rating_decline = first_half_avg - second_half_avg
    
    if rating_decline >= rating_decline_threshold:
        # Significant decline detected
        # Estimate closure as 6 months after decline started
        decline_start = second_half['date'].iloc[0]
        estimated_closure = decline_start + timedelta(days=180)
        confidence = min(0.5 + rating_decline * 0.3, 0.9)
    else:
        # No clear decline, use last review + lag
        last_review = business_reviews['date'].max()
        estimated_closure = last_review + timedelta(days=180)
        confidence = 0.5
    
    return estimated_closure, confidence


def get_features_for_rating_inference() -> Dict[str, List[str]]:
    """
    Features safe to use when using rating-based label inference.
    
    MUST EXCLUDE all rating-related features since they're used for inference.
    """
    return {
        'A_Static': [
            'review_count', 'category_encoded', 'state_encoded',
            'city_encoded', 'has_multiple_categories', 'category_count', 'price_range'
            # EXCLUDED: 'stars' (uses rating)
        ],
        'B_Review_Agg': [
            'total_reviews', 'days_since_first_review', 'review_recency_ratio', 
            'review_frequency', 'total_useful_votes', 'avg_useful_per_review'
            # EXCLUDED: 'avg_review_stars', 'std_review_stars'
        ],
        'C_Sentiment': [
            'avg_sentiment', 'std_sentiment', 'sentiment_volatility',
            'pct_positive_reviews', 'pct_negative_reviews', 'pct_neutral_reviews',
            'avg_text_length', 'std_text_length', 'sentiment_recent_3m'
        ],
        'D_User_Weighted': [
            'avg_reviewer_credibility', 'std_reviewer_credibility',
            'weighted_sentiment',  # OK - uses sentiment, not rating
            'pct_high_credibility_reviewers', 'weighted_useful_votes',
            'avg_reviewer_tenure', 'avg_reviewer_experience', 'review_diversity'
            # EXCLUDED: 'weighted_avg_rating'
        ],
        'E_Temporal': [
            'reviews_recent_3m_count', 'engagement_recent_vs_all',
            'sentiment_recent_vs_all', 'review_momentum', 'lifecycle_stage'
            # EXCLUDED: 'rating_recent_vs_all', 'rating_recent_vs_early', 'rating_trend_3m'
        ],
        'F_Location': [
            'category_avg_success_rate', 'state_avg_success_rate',
            'city_avg_success_rate', 'category_competitiveness',
            'location_density'
        ]
    }


# ============================================================================
# Validation and Documentation
# ============================================================================

def validate_no_leakage(
    features_df: pd.DataFrame,
    label_source: str = 'final_status'
) -> Dict[str, any]:
    """
    Validate that there's no potential label leakage.
    
    Args:
        features_df: Feature DataFrame
        label_source: How labels were generated
        
    Returns:
        Validation report
    """
    report = {
        'label_source': label_source,
        'is_valid': True,
        'warnings': [],
        'feature_categories': {}
    }
    
    if label_source == 'final_status':
        # Pure Final Status approach - all features are safe
        report['note'] = (
            "Labels use ground truth (is_open), not inferred from features. "
            "All features are safe to use as long as they only use data before cutoff."
        )
        
    elif label_source == 'rating_inference':
        # Rating-based inference - check for excluded features
        excluded = ['stars', 'avg_review_stars', 'std_review_stars',
                   'rating_recent_vs_all', 'rating_recent_vs_early',
                   'rating_trend_3m', 'weighted_avg_rating']
        
        present_excluded = [f for f in excluded if f in features_df.columns]
        
        if present_excluded:
            report['is_valid'] = False
            report['warnings'].append(
                f"LEAKAGE RISK: These features should be excluded when using "
                f"rating-based inference: {present_excluded}"
            )
    
    return report


def print_leakage_prevention_guide():
    """Print a guide on preventing label leakage."""
    guide = """
    ============================================================================
    LABEL LEAKAGE PREVENTION GUIDE
    ============================================================================
    
    ORIGINAL PROBLEM:
    -----------------
    The original label inference used review activity patterns:
        estimated_closure = last_review_date + 180 days
    
    But features also used review activity patterns:
        - reviews_recent_3m_count
        - review_recency_ratio
        - days_since_last_review
    
    This created CIRCULAR DEPENDENCY:
        Features → predict → Labels ← derived from → same patterns as Features
    
    SOLUTION (Approach A - Recommended):
    ------------------------------------
    Use ONLY the final is_open status as label:
    
        Label = business_df['is_open']  # Ground truth, no inference
    
    Why this works:
    1. Labels are independent ground truth (not inferred)
    2. Features use only historical data before cutoff
    3. Prediction task: "Given historical patterns, predict future outcome"
    4. No circular dependency
    
    IMPLEMENTATION:
    ---------------
    Use create_prediction_tasks_v2() which:
    1. Creates tasks with cutoff dates
    2. Uses final is_open as label (no inference)
    3. Validates business has sufficient pre-cutoff activity
    4. Returns tasks ready for feature engineering
    
    ============================================================================
    """
    print(guide)


if __name__ == "__main__":
    print_leakage_prevention_guide()

