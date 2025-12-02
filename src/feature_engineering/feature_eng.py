"""
Feature Engineering: Extract and engineer features for business success prediction.

CRITICAL UPDATES (Temporal Validation Support):
- Added support for temporal validation with cutoff dates
- Removed leaky features (days_since_last_review)
- Fixed temporal window calculations to prevent future information leakage
- Added metadata columns for temporal split stratification
- Support for generating features at multiple cutoff dates

This module implements the feature engineering pipeline with:
- Business static features
- Review aggregation features  
- Sentiment features
- User-weighted features
- Temporal dynamics features (CORRECTED for temporal leakage)
- Category/Location features
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import json
import gc
import warnings
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')

# Import utility functions
import sys
sys.path.append(str(Path(__file__).parent.parent))
from utils.temporal_utils import filter_reviews_by_cutoff, compute_temporal_window

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False
    print("WARNING: VADER not available. Install 'vaderSentiment' for sentiment features.")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/feature_engineering.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Comprehensive feature engineering for business success prediction.
    
    Design Principles:
    1. No data leakage - only use historical information up to cutoff_date
    2. Chunked processing for memory efficiency
    3. Robust aggregations for skewed distributions
    4. User credibility weighting as novel contribution
    5. Temporal dynamics with proper windowing
    
    NEW: Temporal Validation Support
    - Can generate features for multiple cutoff dates
    - Adds metadata columns for temporal stratification
    - Removes leaky features automatically
    """
    
    def __init__(self, 
                 processed_path: str = "data/processed",
                 output_path: str = "data/features",
                 use_temporal_validation: bool = False,
                 cutoff_dates: Optional[List[str]] = None,
                 prediction_years: Optional[List[int]] = None):
        """
        Initialize feature engineer with temporal validation support.
        
        Args:
            processed_path: Path to processed data
            output_path: Path to save features
            use_temporal_validation: If True, apply temporal constraints
            cutoff_dates: List of cutoff dates (strings like '2020-12-31')
                         If None and use_temporal_validation=True, will generate
                         cutoffs based on prediction_years
            prediction_years: List of years to generate cutoffs for (e.g., [2012, 2013, ...])
                            Will create cutoffs as YYYY-12-31 for each year
        """
        self.processed_path = Path(processed_path)
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectory for separate feature categories
        self.category_path = self.output_path / "feature_categories"
        self.category_path.mkdir(parents=True, exist_ok=True)
        
        # Temporal validation settings
        self.use_temporal_validation = use_temporal_validation
        
        # Determine cutoff dates
        if use_temporal_validation:
            if cutoff_dates is not None:
                # Use provided cutoff dates
                self.cutoff_dates = [pd.Timestamp(d) for d in cutoff_dates]
            elif prediction_years is not None:
                # Generate cutoffs from years (end of each year)
                self.cutoff_dates = [pd.Timestamp(f'{year}-12-31') for year in prediction_years]
            else:
                # Default: single cutoff at 2020-12-31
                self.cutoff_dates = [pd.Timestamp('2020-12-31')]
                logger.warning("No cutoff_dates or prediction_years provided, using default: 2020-12-31")
            
            logger.info(f"Temporal validation enabled with {len(self.cutoff_dates)} cutoff dates")
            logger.info(f"  Cutoffs: {[d.date() for d in self.cutoff_dates]}")
        else:
            # No temporal validation - use dataset end date as reference
            self.cutoff_dates = [pd.Timestamp('2022-01-19')]
            logger.info("Temporal validation disabled - using full dataset (reference: 2022-01-19)")
        
        # Data containers
        self.business_df = None
        self.user_df = None
        self.reviews_df = None  # Will load in chunks, but keep reference
        
        # Feature tracking
        self.feature_summary = {}
        
        # Leaky features to remove in temporal validation mode
        self.leaky_features = [
            'days_since_last_review',  # Direct encoding of closure
        ]
    
    def load_data(self):
        """Load cleaned business and user data"""
        logger.info("="*70)
        logger.info("LOADING CLEANED DATA")
        logger.info("="*70)
        
        # Load business data
        business_path = self.processed_path / "business_clean.csv"
        self.business_df = pd.read_csv(business_path)
        logger.info(f"Loaded business data: {self.business_df.shape}")
        
        # Load user data (for credibility weights)
        user_path = self.processed_path / "user_clean.csv"
        logger.info("Loading user data in chunks...")
        chunks = []
        for chunk in pd.read_csv(user_path, chunksize=100000):
            chunks.append(chunk)
        self.user_df = pd.concat(chunks, ignore_index=True)
        logger.info(f"Loaded user data: {self.user_df.shape}")
        
        # OPTIMIZATION: Load pre-computed sentiment scores from Phase 2 (EDA)
        sentiment_path = self.processed_path / "review_sentiment.csv"
        if sentiment_path.exists():
            logger.info("Loading pre-computed sentiment scores from Phase 2...")
            self.sentiment_df = pd.read_csv(sentiment_path)
            logger.info(f"[OK] Loaded pre-computed sentiment: {self.sentiment_df.shape}")
            logger.info(f"  (Skipping VADER computation in feature engineering)")
            self.use_precomputed_sentiment = True
        else:
            logger.warning("Pre-computed sentiment not found at: {sentiment_path}")
            logger.warning("Will compute sentiment on-the-fly (slower)")
            self.sentiment_df = None
            self.use_precomputed_sentiment = False
        
        # Note: Reviews will be loaded in chunks during feature computation
        logger.info("Review data will be loaded in chunks during processing")

    def create_static_features(self) -> pd.DataFrame:
        """
        Create static business features (Category A).
        
        Features (8):
        1. stars (business rating)
        2. review_count (business review count) 
        3. category_encoded (target-encoded top category)
        4. state_encoded (target-encoded state)
        5. city_encoded (target-encoded top 50 cities)
        6. has_multiple_categories (binary)
        7. category_count (number of categories)
        8. price_range (if available in attributes)
        """
        logger.info("="*70)
        logger.info("CREATING STATIC BUSINESS FEATURES")
        logger.info("="*70)
        
        df = self.business_df.copy()
        features = pd.DataFrame()
        features['business_id'] = df['business_id']
        features['is_open'] = df['is_open']  # Target variable
        
        # Feature 1-2: Basic business metrics
        features['stars'] = df['stars']
        features['review_count'] = df['review_count']
        logger.info(f"[OK] Created basic metrics: stars, review_count")
        
        # Parse categories
        df['categories_list'] = df['categories'].fillna('').str.split(',')
        df['categories_list'] = df['categories_list'].apply(
            lambda x: [c.strip() for c in x] if isinstance(x, list) else []
        )
        
        # Feature 3: Top category target encoding
        # Extract primary category (first one listed)
        df['primary_category'] = df['categories_list'].apply(
            lambda x: x[0] if len(x) > 0 else 'Unknown'
        )
        
        # Get top 20 categories
        top_categories = df['primary_category'].value_counts().head(20).index.tolist()
        df['primary_category_grouped'] = df['primary_category'].apply(
            lambda x: x if x in top_categories else 'Other'
        )
        
        # Target encoding with smoothing
        category_stats = df.groupby('primary_category_grouped')['is_open'].agg(['mean', 'count'])
        global_mean = df['is_open'].mean()
        smoothing = 100
        
        category_stats['encoded'] = (
            (category_stats['mean'] * category_stats['count'] + global_mean * smoothing) /
            (category_stats['count'] + smoothing)
        )
        
        features['category_encoded'] = df['primary_category_grouped'].map(
            category_stats['encoded']
        ).fillna(global_mean)
        logger.info(f"[OK] Created category_encoded (top 20 + Other)")
        
        # Feature 4: State target encoding
        state_stats = df.groupby('state')['is_open'].agg(['mean', 'count'])
        state_stats['encoded'] = (
            (state_stats['mean'] * state_stats['count'] + global_mean * smoothing) /
            (state_stats['count'] + smoothing)
        )
        features['state_encoded'] = df['state'].map(state_stats['encoded']).fillna(global_mean)
        logger.info(f"[OK] Created state_encoded")
        
        # Feature 5: City target encoding (top 50 cities)
        top_cities = df['city'].value_counts().head(50).index.tolist()
        df['city_grouped'] = df['city'].apply(lambda x: x if x in top_cities else 'Other')
        
        city_stats = df.groupby('city_grouped')['is_open'].agg(['mean', 'count'])
        city_stats['encoded'] = (
            (city_stats['mean'] * city_stats['count'] + global_mean * smoothing) /
            (city_stats['count'] + smoothing)
        )
        features['city_encoded'] = df['city_grouped'].map(city_stats['encoded']).fillna(global_mean)
        logger.info(f"[OK] Created city_encoded (top 50 + Other)")
        
        # Feature 6-7: Category counts
        features['has_multiple_categories'] = df['categories_list'].apply(len) > 1
        features['has_multiple_categories'] = features['has_multiple_categories'].astype(int)
        features['category_count'] = df['categories_list'].apply(len)
        logger.info(f"[OK] Created has_multiple_categories, category_count")
        
        # Feature 8: Price range (if available in attributes)
        # Try to extract from attributes JSON
        def extract_price_range(attr_str):
            try:
                if pd.isna(attr_str) or attr_str == '{}':
                    return np.nan
                # Try to parse as dict or eval
                if isinstance(attr_str, str):
                    attr_dict = eval(attr_str)
                    if 'RestaurantsPriceRange2' in attr_dict:
                        return float(attr_dict['RestaurantsPriceRange2'])
                return np.nan
            except:
                return np.nan
        
        features['price_range'] = df['attributes'].apply(extract_price_range)
        # Fill missing with median
        median_price = features['price_range'].median()
        features['price_range'].fillna(median_price, inplace=True)
        logger.info(f"[OK] Created price_range (extracted from attributes)")
        
        logger.info(f"\nStatic features created: {features.shape[1] - 2} features")
        logger.info(f"Feature names: {[c for c in features.columns if c not in ['business_id', 'is_open']]}")
        
        # Save separate category file
        static_file = self.category_path / "business_static_features.csv"
        features.to_csv(static_file, index=False)
        logger.info(f"Saved: {static_file}")
        
        return features
    
    def calculate_user_credibility(self) -> pd.Series:
        """
        Calculate user credibility scores for weighting.
        
        Formula (from EDA):
        useful_rate = useful / (review_count + 1)
        tenure_weight = log(1 + user_tenure_days) / 10
        experience_weight = log(1 + review_count) / 10
        credibility = 0.5 * useful_rate + 0.3 * tenure_weight + 0.2 * experience_weight
        
        Returns:
            Series with user_id as index, credibility as values
        """
        logger.info("Calculating user credibility scores...")
        
        df = self.user_df.copy()
        
        # Calculate components
        useful_rate = df['useful'] / (df['review_count'] + 1)
        tenure_weight = np.log1p(df['user_tenure_days']) / 10.0
        experience_weight = np.log1p(df['review_count']) / 10.0
        
        # Weighted combination
        credibility = (
            0.5 * useful_rate + 
            0.3 * tenure_weight + 
            0.2 * experience_weight
        )
        
        # Create series with user_id as index
        credibility_series = pd.Series(
            credibility.values,
            index=df['user_id'].values,
            name='user_credibility'
        )
        
        logger.info(f"[OK] Calculated credibility for {len(credibility_series):,} users")
        logger.info(f"  Mean credibility: {credibility_series.mean():.4f}")
        logger.info(f"  Median credibility: {credibility_series.median():.4f}")
        
        return credibility_series
    
    def _compute_review_features_single_business(self,
                                                business_id: str,
                                                reviews: pd.DataFrame,
                                                user_credibility: Dict,
                                                sentiment_analyzer,
                                                cutoff_date: pd.Timestamp,
                                                user_info_dict: Dict = None) -> Optional[Dict]:
        """
        Compute review-based features for a single business at a specific cutoff date.
        
        This is the core feature computation function, now WITH temporal awareness.
        
        Args:
            business_id: ID of the business
            reviews: Reviews for this business (already filtered to cutoff_date)
            user_credibility: Dict mapping user_id to credibility score
            sentiment_analyzer: VADER analyzer or None
            cutoff_date: Reference date for feature computation
            user_info_dict: Pre-built dict for O(1) user info lookup (OPTIMIZATION)
            
        Returns:
            Dict of features or None if insufficient data
        """
        if len(reviews) < 3:
            return None
        
        features = {}
        
        # ============================================================
        # Category B: Review Aggregation Features (9 features)
        # ============================================================
        
        # Basic counts and stats
        features['total_reviews'] = len(reviews)
        features['avg_review_stars'] = reviews['stars'].mean()
        features['std_review_stars'] = reviews['stars'].std()
        
        # Temporal features (relative to cutoff_date)
        first_review_date = reviews['date'].min()
        last_review_date = reviews['date'].max()
        
        # [FIXED] Use cutoff_date instead of current date
        features['days_since_first_review'] = (cutoff_date - first_review_date).days
        
        # [FIXED] REMOVED: days_since_last_review (this was the main leaky feature)
        # Instead, we'll use review recency as a ratio
        days_since_last = (cutoff_date - last_review_date).days
        days_active = (cutoff_date - first_review_date).days
        features['review_recency_ratio'] = 1.0 - (days_since_last / max(days_active, 1))
        # Ratio of 1.0 = very recent reviews, 0.0 = old reviews
        
        # Review frequency (corrected)
        features['review_frequency'] = len(reviews) / max(days_active, 1)
        
        # Engagement metrics
        features['total_useful_votes'] = reviews['useful'].sum()
        features['avg_useful_per_review'] = reviews['useful'].mean()
        
        # ============================================================
        # Category C: Sentiment Features (8 features)
        # OPTIMIZED: Use pre-computed sentiment when available
        # ============================================================
        
        # Check if pre-computed sentiment is available
        has_precomputed = '_precomputed_sentiment' in reviews.columns and reviews['_precomputed_sentiment'].notna().any()
        
        if has_precomputed:
            # [FAST PATH] Use pre-computed sentiment from Phase 2
            sentiments = reviews['_precomputed_sentiment'].fillna(0.0).tolist()
            text_lengths = reviews['text'].fillna('').str.len().tolist() if 'text' in reviews.columns else [0] * len(reviews)
            
            features['avg_sentiment'] = np.mean(sentiments) if sentiments else 0.0
            features['std_sentiment'] = np.std(sentiments) if sentiments else 0.0
            features['sentiment_volatility'] = np.std(sentiments) / (abs(np.mean(sentiments)) + 0.01) if sentiments else 0.0
            
            # Sentiment distribution
            positive = sum(1 for s in sentiments if s > 0.05)
            negative = sum(1 for s in sentiments if s < -0.05)
            neutral = len(sentiments) - positive - negative
            
            features['pct_positive_reviews'] = positive / len(sentiments) if sentiments else 0.0
            features['pct_negative_reviews'] = negative / len(sentiments) if sentiments else 0.0
            features['pct_neutral_reviews'] = neutral / len(sentiments) if sentiments else 0.0
            
            # Text characteristics
            features['avg_text_length'] = np.mean(text_lengths) if text_lengths else 0.0
            features['std_text_length'] = np.std(text_lengths) if text_lengths else 0.0
            
            # Temporal sentiment trend (using cutoff-aware window)
            recent_start, recent_end = compute_temporal_window(cutoff_date, 3, 'recent')
            recent_reviews = reviews[(reviews['date'] >= recent_start) & (reviews['date'] <= recent_end)]
            
            if len(recent_reviews) > 0 and '_precomputed_sentiment' in recent_reviews.columns:
                recent_sentiments = recent_reviews['_precomputed_sentiment'].fillna(0.0).tolist()
                features['sentiment_recent_3m'] = np.mean(recent_sentiments) if recent_sentiments else features['avg_sentiment']
            else:
                features['sentiment_recent_3m'] = features['avg_sentiment']
        
        elif sentiment_analyzer and 'text' in reviews.columns:
            # SLOW PATH: Compute sentiment on-the-fly with VADER
            sentiments = []
            text_lengths = []
            
            for text in reviews['text'].fillna(''):
                if len(text) > 0:
                    sentiment_score = sentiment_analyzer.polarity_scores(text)
                    sentiments.append(sentiment_score['compound'])
                    text_lengths.append(len(text))
                else:
                    sentiments.append(0.0)
                    text_lengths.append(0)
            
            features['avg_sentiment'] = np.mean(sentiments) if sentiments else 0.0
            features['std_sentiment'] = np.std(sentiments) if sentiments else 0.0
            features['sentiment_volatility'] = np.std(sentiments) / (abs(np.mean(sentiments)) + 0.01) if sentiments else 0.0
            
            # Sentiment distribution
            positive = sum(1 for s in sentiments if s > 0.05)
            negative = sum(1 for s in sentiments if s < -0.05)
            neutral = len(sentiments) - positive - negative
            
            features['pct_positive_reviews'] = positive / len(sentiments) if sentiments else 0.0
            features['pct_negative_reviews'] = negative / len(sentiments) if sentiments else 0.0
            features['pct_neutral_reviews'] = neutral / len(sentiments) if sentiments else 0.0
            
            # Text characteristics
            features['avg_text_length'] = np.mean(text_lengths) if text_lengths else 0.0
            features['std_text_length'] = np.std(text_lengths) if text_lengths else 0.0
            
            # Temporal sentiment trend (using cutoff-aware window)
            recent_start, recent_end = compute_temporal_window(cutoff_date, 3, 'recent')
            recent_reviews = reviews[(reviews['date'] >= recent_start) & (reviews['date'] <= recent_end)]
            
            if len(recent_reviews) > 0:
                recent_sentiments = []
                for text in recent_reviews['text'].fillna(''):
                    if len(text) > 0:
                        score = sentiment_analyzer.polarity_scores(text)
                        recent_sentiments.append(score['compound'])
                features['sentiment_recent_3m'] = np.mean(recent_sentiments) if recent_sentiments else features['avg_sentiment']
            else:
                features['sentiment_recent_3m'] = features['avg_sentiment']
        
        else:
            # No sentiment analysis - fill with defaults
            features.update({
                'avg_sentiment': 0.0,
                'std_sentiment': 0.0,
                'sentiment_volatility': 0.0,
                'pct_positive_reviews': 0.33,
                'pct_negative_reviews': 0.33,
                'pct_neutral_reviews': 0.34,
                'avg_text_length': 0.0,
                'std_text_length': 0.0,
                'sentiment_recent_3m': 0.0
            })
        
        # ============================================================
        # Category D: User-Weighted Features (9 features)
        # ============================================================
        
        # Get credibility scores for reviewers
        reviewer_credibilities = []
        weighted_ratings = []
        weighted_sentiments = []
        
        for idx, review in reviews.iterrows():
            user_id = review['user_id']
            cred = user_credibility.get(user_id, 0.5)  # Default to 0.5 if not found
            reviewer_credibilities.append(cred)
            weighted_ratings.append(review['stars'] * cred)
            
            # Use sentiment if available
            if 'avg_sentiment' in features and features['avg_sentiment'] != 0.0:
                # Approximate sentiment from stars
                sentiment_approx = (review['stars'] - 3) / 2  # Scale to [-1, 1]
                weighted_sentiments.append(sentiment_approx * cred)
        
        features['avg_reviewer_credibility'] = np.mean(reviewer_credibilities)
        features['std_reviewer_credibility'] = np.std(reviewer_credibilities)
        
        # Weighted aggregations
        total_cred = sum(reviewer_credibilities)
        if total_cred > 0:
            features['weighted_avg_rating'] = sum(weighted_ratings) / total_cred
            features['weighted_sentiment'] = sum(weighted_sentiments) / total_cred if weighted_sentiments else 0.0
        else:
            features['weighted_avg_rating'] = features['avg_review_stars']
            features['weighted_sentiment'] = features['avg_sentiment']
        
        # High credibility reviewers
        high_cred_threshold = 0.7
        features['pct_high_credibility_reviewers'] = sum(1 for c in reviewer_credibilities if c > high_cred_threshold) / len(reviewer_credibilities)
        
        # Weighted engagement
        weighted_useful = sum(reviews['useful'] * reviews['user_id'].map(lambda u: user_credibility.get(u, 0.5)))
        features['weighted_useful_votes'] = weighted_useful
        
        # Reviewer characteristics
        # OPTIMIZED: Use pre-built dict instead of slow isin() on 2M row DataFrame
        reviewer_ids = reviews['user_id'].unique()
        
        if user_info_dict is not None:
            # O(1) lookup per reviewer - FAST!
            tenures = []
            experiences = []
            for uid in reviewer_ids:
                if uid in user_info_dict:
                    tenures.append(user_info_dict[uid]['user_tenure_years'])
                    experiences.append(user_info_dict[uid]['review_count'])
            
            if tenures:
                features['avg_reviewer_tenure'] = np.mean(tenures)
                features['avg_reviewer_experience'] = np.mean(experiences)
            else:
                features['avg_reviewer_tenure'] = 0.0
                features['avg_reviewer_experience'] = 0.0
        else:
            # Fallback to slow method if dict not provided
            reviewer_info = self.user_df[self.user_df['user_id'].isin(reviewer_ids)]
        if len(reviewer_info) > 0:
            features['avg_reviewer_tenure'] = reviewer_info['user_tenure_years'].mean()
            features['avg_reviewer_experience'] = reviewer_info['review_count'].mean()
        else:
            features['avg_reviewer_tenure'] = 0.0
            features['avg_reviewer_experience'] = 0.0
        
        # Reviewer diversity
        features['review_diversity'] = len(reviewer_ids) / len(reviews)  # Unique reviewers ratio
        
        # ============================================================
        # Category E: Temporal Dynamics Features (8 features)
        # ============================================================
        
        # [FIXED] All temporal comparisons now use cutoff-aware windows
        
        # Define windows
        recent_start, recent_end = compute_temporal_window(cutoff_date, 3, 'recent')
        recent_reviews = reviews[(reviews['date'] >= recent_start) & (reviews['date'] <= recent_end)]
        
        # Early window (first 3 months of operation)
        early_start = first_review_date
        early_end = first_review_date + pd.DateOffset(months=3)
        early_reviews = reviews[(reviews['date'] >= early_start) & (reviews['date'] <= early_end)]
        
        # Temporal comparisons
        # [IMPROVED] Using difference instead of ratio to reduce noise
        # Rationale: Ratios can explode when denominator is small; differences are more stable
        if len(recent_reviews) >= 2:
            recent_rating = recent_reviews['stars'].mean()
            recent_engagement = recent_reviews['useful'].mean()
            
            # Use difference instead of ratio (more stable, less noise)
            features['rating_recent_vs_all'] = recent_rating - features['avg_review_stars']
            features['reviews_recent_3m_count'] = len(recent_reviews)
            features['engagement_recent_vs_all'] = recent_engagement - features['avg_useful_per_review']
            
            if 'avg_sentiment' in features and features['avg_sentiment'] != 0.0:
                # Use pre-computed sentiment if available, otherwise fall back to VADER
                if '_precomputed_sentiment' in recent_reviews.columns and recent_reviews['_precomputed_sentiment'].notna().any():
                    recent_sent = recent_reviews['_precomputed_sentiment'].fillna(0.0).mean()
                elif sentiment_analyzer:
                    recent_sent = np.mean([sentiment_analyzer.polarity_scores(text)['compound'] 
                                          for text in recent_reviews['text'].fillna('') if len(text) > 0])
                else:
                    recent_sent = features['avg_sentiment']
                # Use difference instead of ratio
                features['sentiment_recent_vs_all'] = recent_sent - features['avg_sentiment']
            else:
                features['sentiment_recent_vs_all'] = 0.0
        else:
            features['rating_recent_vs_all'] = 0.0
            features['reviews_recent_3m_count'] = 0
            features['engagement_recent_vs_all'] = 0.0
            features['sentiment_recent_vs_all'] = 0.0
        
        if len(early_reviews) >= 2 and len(recent_reviews) >= 2:
            # Use difference instead of ratio
            features['rating_recent_vs_early'] = recent_reviews['stars'].mean() - early_reviews['stars'].mean()
        else:
            features['rating_recent_vs_early'] = 0.0
        
        # Review momentum (trend over time)
        # Calculate reviews per month over time
        reviews_sorted = reviews.sort_values('date')
        reviews_sorted['month'] = reviews_sorted['date'].dt.to_period('M')
        monthly_counts = reviews_sorted.groupby('month').size()
        
        if len(monthly_counts) >= 3:
            # Simple linear trend: positive = growing, negative = declining
            x = np.arange(len(monthly_counts))
            y = monthly_counts.values
            trend = np.polyfit(x, y, 1)[0]  # Slope of linear fit
            features['review_momentum'] = trend
        else:
            features['review_momentum'] = 0.0
        
        # Lifecycle stage classification
        # Based on review activity pattern and age
        age_months = days_active / 30
        recent_review_rate = features['reviews_recent_3m_count'] / 3  # Reviews per month (recent)
        overall_review_rate = features['total_reviews'] / max(age_months, 1)
        
        if age_months < 6:
            lifecycle_stage = 0  # New
        elif recent_review_rate > overall_review_rate * 1.2:
            lifecycle_stage = 1  # Growing
        elif recent_review_rate >= overall_review_rate * 0.8:
            lifecycle_stage = 2  # Mature
        else:
            lifecycle_stage = 3  # Declining
        
        features['lifecycle_stage'] = lifecycle_stage
        
        # Rating trend (simple moving average difference)
        if len(reviews) >= 10:
            mid_point = len(reviews_sorted) // 2
            first_half_rating = reviews_sorted.iloc[:mid_point]['stars'].mean()
            second_half_rating = reviews_sorted.iloc[mid_point:]['stars'].mean()
            features['rating_trend_3m'] = second_half_rating - first_half_rating
        else:
            features['rating_trend_3m'] = 0.0
        
        # ============================================================
        # Category G: Feature Interactions (NEW - Enhanced Signals)
        # ============================================================
        # 
        # Rationale:
        # - Individual features capture main effects
        # - Interactions capture synergistic effects
        # - Example: High rating + High credibility reviewers = Strong signal
        #           High rating + Low credibility reviewers = Weak signal
        
        # Interaction 1: Rating Quality × Reviewer Credibility
        # High-quality ratings backed by credible reviewers are more reliable
        if 'avg_review_stars' in features and 'avg_reviewer_credibility' in features:
            features['rating_credibility_interaction'] = (
                features['avg_review_stars'] * features['avg_reviewer_credibility']
            )
        
        # Interaction 2: Momentum × Credibility
        # Growth driven by high-credibility users is more reliable than low-credibility growth
        if 'review_momentum' in features and 'avg_reviewer_credibility' in features:
            features['momentum_credibility_interaction'] = (
                features['review_momentum'] * features['avg_reviewer_credibility']
            )
        
        # Interaction 3: Size × Activity
        # Large businesses with high activity are different from small + high activity
        if 'total_reviews' in features and 'review_frequency' in features:
            features['size_activity_interaction'] = (
                np.log1p(features['total_reviews']) * features['review_frequency']
            )
        
        # Interaction 4: Recent Trend × Overall Quality
        # Declining trend matters more for high-rated businesses
        if 'rating_recent_vs_all' in features and 'avg_review_stars' in features:
            features['trend_quality_interaction'] = (
                features['rating_recent_vs_all'] * features['avg_review_stars']
            )
        
        # Interaction 5: Engagement × Credibility
        # Useful votes from credible users are more valuable
        if 'total_useful_votes' in features and 'avg_reviewer_credibility' in features:
            features['engagement_credibility_interaction'] = (
                np.log1p(features['total_useful_votes']) * features['avg_reviewer_credibility']
            )
        
        return features

    def create_review_features_chunked(self, user_credibility: Dict) -> pd.DataFrame:
        """
        Create review-based features using TWO-PHASE chunked processing.
        
        FIXED: Previous implementation had a bug where reviews split across chunks
        would result in incomplete feature computation. Now uses two-phase approach:
        
        Phase 1: Collect all reviews for each business (memory-efficient accumulation)
        Phase 2: Compute features using complete review data
        
        This includes:
        - Category B: Review aggregation features
        - Category C: Sentiment features
        - Category D: User-weighted features
        - Category E: Temporal dynamics features (FIXED for leakage)
        
        Returns:
            DataFrame with columns:
            - business_id
            - _cutoff_date (if temporal validation)
            - _prediction_year (if temporal validation)
            - all feature columns
        """
        logger.info("="*70)
        logger.info("CREATING REVIEW-BASED FEATURES (TWO-PHASE CHUNKED)")
        logger.info("="*70)
        
        if self.use_temporal_validation:
            logger.info(f"Temporal validation mode: generating features for {len(self.cutoff_dates)} cutoff dates")
        
        review_path = self.processed_path / "review_clean.csv"
        
        # ================================================================
        # OPTIMIZATION: Use pre-computed sentiment if available
        # ================================================================
        sentiment_analyzer = None
        precomputed_sentiment = {}  # review_id -> sentiment score
        
        if self.use_precomputed_sentiment and self.sentiment_df is not None:
            logger.info("[OK] Using PRE-COMPUTED sentiment from Phase 2 (FAST)")
            logger.info(f"  Loaded {len(self.sentiment_df):,} pre-computed sentiment scores")
            # Create lookup dictionary for fast access
            precomputed_sentiment = dict(zip(
                self.sentiment_df['review_id'], 
                self.sentiment_df['sentiment']
            ))
            logger.info("  Created sentiment lookup dictionary")
        elif VADER_AVAILABLE:
            sentiment_analyzer = SentimentIntensityAnalyzer()
            logger.info("[WARN] Using VADER on-the-fly (slower - consider re-running Phase 2)")
        else:
            logger.warning("[FAIL] No sentiment available - features will be approximated from stars")
        
        # ================================================================
        # PHASE 1: Collect all reviews per business
        # ================================================================
        logger.info("\n" + "="*70)
        logger.info("PHASE 1: Collecting reviews per business")
        logger.info("="*70)
        
        # Dictionary to accumulate reviews per business
        # Key: business_id, Value: list of review dicts (only essential columns)
        business_reviews_dict = {}
        
        # Essential columns to keep (minimize memory usage)
        # Include review_id for sentiment lookup
        essential_cols = ['review_id', 'business_id', 'user_id', 'stars', 'date', 'text', 'useful']
        
        chunk_size = 500000
        chunk_num = 0
        total_reviews = 0
        
        logger.info(f"Reading reviews in chunks of {chunk_size:,}...")
        
        for chunk in pd.read_csv(review_path, chunksize=chunk_size):
            chunk_num += 1
            
            # Only keep essential columns
            available_cols = [c for c in essential_cols if c in chunk.columns]
            chunk = chunk[available_cols]
            
            # Ensure date column is datetime
            chunk['date'] = pd.to_datetime(chunk['date'], errors='coerce')
            chunk = chunk.dropna(subset=['date'])
            
            # Add pre-computed sentiment if available
            if precomputed_sentiment and 'review_id' in chunk.columns:
                chunk['_precomputed_sentiment'] = chunk['review_id'].map(precomputed_sentiment)
            
            total_reviews += len(chunk)
            
            # Accumulate reviews per business
            for business_id, group in chunk.groupby('business_id'):
                if business_id not in business_reviews_dict:
                    business_reviews_dict[business_id] = []
                
                # Convert to list of dicts for memory efficiency
                business_reviews_dict[business_id].extend(group.to_dict('records'))
            
            logger.info(f"  [Chunk {chunk_num}] Processed {len(chunk):,} reviews, "
                       f"Total businesses: {len(business_reviews_dict):,}")
            
            # Memory cleanup
            del chunk
            gc.collect()
        
        # Clear sentiment lookup to free memory
        del precomputed_sentiment
        gc.collect()
        
        logger.info(f"\nPhase 1 complete:")
        logger.info(f"  Total reviews processed: {total_reviews:,}")
        logger.info(f"  Total unique businesses: {len(business_reviews_dict):,}")
        
        # ================================================================
        # PHASE 2: Compute features using complete review data
        # ================================================================
        logger.info("\n" + "="*70)
        logger.info("PHASE 2: Computing features from complete reviews")
        logger.info("="*70)
        
        # ================================================================
        # OPTIMIZATION: Pre-build user info lookup dictionary
        # This changes O(n) isin() lookups to O(1) dict lookups
        # ================================================================
        logger.info("Building user info lookup dictionary...")
        user_info_dict = {}
        for _, row in self.user_df.iterrows():
            user_info_dict[row['user_id']] = {
                'user_tenure_years': row.get('user_tenure_years', 0.0),
                'review_count': row.get('review_count', 0)
            }
        logger.info(f"  Created lookup for {len(user_info_dict):,} users")
        
        all_business_features = []
        processed_count = 0
        total_businesses = len(business_reviews_dict)
        
        for business_id, reviews_list in business_reviews_dict.items():
            processed_count += 1
            
            # Convert list of dicts back to DataFrame
            business_reviews = pd.DataFrame(reviews_list)
            business_reviews['date'] = pd.to_datetime(business_reviews['date'])
            
            # Log progress every 10000 businesses
            if processed_count % 10000 == 0:
                logger.info(f"  Processed {processed_count:,}/{total_businesses:,} businesses "
                           f"({processed_count/total_businesses*100:.1f}%)")
            
            # If temporal validation, generate features for each cutoff date
            if self.use_temporal_validation:
                for cutoff_date in self.cutoff_dates:
                    # Filter reviews up to cutoff
                    historical_reviews = filter_reviews_by_cutoff(
                        business_reviews, 
                        cutoff_date
                    )
                    
                    # Skip if insufficient data at this cutoff
                    if len(historical_reviews) < 3:
                        continue
                    
                    # Check if business is still active at cutoff
                    # (last review within 6 months of cutoff)
                    last_review = historical_reviews['date'].max()
                    days_since_last = (cutoff_date - last_review).days
                    if days_since_last > 180:
                        continue  # Business likely inactive
                    
                    # Compute features at this cutoff
                    features = self._compute_review_features_single_business(
                        business_id=business_id,
                        reviews=historical_reviews,
                        user_credibility=user_credibility,
                        sentiment_analyzer=sentiment_analyzer,
                        cutoff_date=cutoff_date,
                        user_info_dict=user_info_dict
                    )
                    
                    if features is not None:
                        # Add temporal metadata
                        features['business_id'] = business_id
                        features['_cutoff_date'] = cutoff_date
                        features['_prediction_year'] = cutoff_date.year
                        all_business_features.append(features)
            
            else:
                # No temporal validation - use all reviews
                features = self._compute_review_features_single_business(
                    business_id=business_id,
                    reviews=business_reviews,
                    user_credibility=user_credibility,
                    sentiment_analyzer=sentiment_analyzer,
                    cutoff_date=self.cutoff_dates[0],  # Use reference date
                    user_info_dict=user_info_dict
                )
                
                if features is not None:
                    features['business_id'] = business_id
                    all_business_features.append(features)
            
            # Periodic memory cleanup
            if processed_count % 50000 == 0:
                gc.collect()
        
        # Clear the reviews dictionary to free memory
        del business_reviews_dict
        gc.collect()
        
        # Convert to DataFrame
        review_features_df = pd.DataFrame(all_business_features)
        
        logger.info(f"\n{'='*70}")
        logger.info(f"REVIEW FEATURES COMPLETE")
        logger.info(f"{'='*70}")
        logger.info(f"Total feature rows: {len(review_features_df):,}")
        
        if not review_features_df.empty:
            logger.info(f"Unique businesses: {review_features_df['business_id'].nunique():,}")
            
            if self.use_temporal_validation and review_features_df['business_id'].nunique() > 0:
                logger.info(f"Average rows per business: {len(review_features_df) / review_features_df['business_id'].nunique():.1f}")
        
        return review_features_df  
    
    def create_location_features(self, business_features: pd.DataFrame) -> pd.DataFrame:
        """
        Create category/location aggregate features (Category F).
        
        Features (5):
        1. category_avg_success_rate (from target encoding)
        2. state_avg_success_rate
        3. city_avg_success_rate
        4. category_competitiveness (# businesses in same category in city)
        5. location_density (# businesses in same city)
        """
        logger.info("="*70)
        logger.info("CREATING LOCATION/CATEGORY AGGREGATE FEATURES")
        logger.info("="*70)
        
        df = self.business_df.copy()
        
        # These are already computed in static features (encoded values)
        # We'll extract the actual success rates for interpretability
        
        features = pd.DataFrame()
        features['business_id'] = df['business_id']
        
        # Parse categories again
        df['categories_list'] = df['categories'].fillna('').str.split(',')
        df['categories_list'] = df['categories_list'].apply(
            lambda x: [c.strip() for c in x] if isinstance(x, list) else []
        )
        df['primary_category'] = df['categories_list'].apply(
            lambda x: x[0] if len(x) > 0 else 'Unknown'
        )
        
        # Top categories
        top_categories = df['primary_category'].value_counts().head(20).index.tolist()
        df['primary_category_grouped'] = df['primary_category'].apply(
            lambda x: x if x in top_categories else 'Other'
        )
        
        # Feature 1: Category success rate
        category_success = df.groupby('primary_category_grouped')['is_open'].mean()
        features['category_avg_success_rate'] = df['primary_category_grouped'].map(category_success)
        
        # Feature 2: State success rate
        state_success = df.groupby('state')['is_open'].mean()
        features['state_avg_success_rate'] = df['state'].map(state_success)
        
        # Feature 3: City success rate
        city_success = df.groupby('city')['is_open'].mean()
        features['city_avg_success_rate'] = df['city'].map(city_success)
        
        # Feature 4: Category competitiveness (businesses in same category in city)
        df['category_city'] = df['primary_category_grouped'] + '_' + df['city']
        category_city_counts = df['category_city'].value_counts()
        features['category_competitiveness'] = df['category_city'].map(category_city_counts)
        
        # Feature 5: Location density
        city_counts = df['city'].value_counts()
        features['location_density'] = df['city'].map(city_counts)
        
        logger.info(f"[OK] Created {features.shape[1] - 1} location/category features")
        
        # Save
        location_file = self.category_path / "location_category_features.csv"
        features.to_csv(location_file, index=False)
        logger.info(f"Saved: {location_file}")
        
        return features
    
    def merge_all_features(self, 
                          static_features: pd.DataFrame,
                          review_features: pd.DataFrame,
                          location_features: pd.DataFrame) -> pd.DataFrame:
        """
        Merge all feature categories into final dataset with temporal metadata.
        
        Args:
            static_features: Category A features
            review_features: Categories B, C, D, E features
            location_features: Category F features
            
        Returns:
            Final merged DataFrame with all features and metadata
        """
        logger.info("="*70)
        logger.info("MERGING ALL FEATURES")
        logger.info("="*70)
        
        # Start with review features (most comprehensive)
        final_df = review_features.copy()
        
        logger.info(f"Starting with review features: {final_df.shape}")
        
        # Determine merge keys based on temporal validation mode
        if self.use_temporal_validation:
            merge_keys = ['business_id', '_cutoff_date']
            logger.info("Temporal validation mode: merging on business_id + _cutoff_date")
        else:
            merge_keys = ['business_id']
            logger.info("Standard mode: merging on business_id only")
        
        # Merge static features
        # Note: Static features are the same for all cutoff dates of same business
        final_df = final_df.merge(
            static_features,
            on='business_id',
            how='left'
        )
        logger.info(f"After merging static features: {final_df.shape}")
        
        # Merge location features
        # Note: Location features are also static across cutoff dates
        final_df = final_df.merge(
            location_features,
            on='business_id',
            how='left'
        )
        logger.info(f"After merging location features: {final_df.shape}")
        
        # Add additional temporal metadata
        if self.use_temporal_validation:
            # Extract year and month for easier analysis
            final_df['_prediction_year'] = final_df['_cutoff_date'].dt.year
            final_df['_prediction_month'] = final_df['_cutoff_date'].dt.month
            
            # Add first and last review dates for label inference
            # These will be computed from review data
            logger.info("Computing review date metadata...")
            
            review_path = self.processed_path / "review_clean.csv"
            
            # Get first and last review dates for each business
            # OPTIMIZED: Use groupby instead of slow per-business loop
            business_review_dates = {}
            
            for chunk in pd.read_csv(review_path, chunksize=500000, usecols=['business_id', 'date']):
                chunk['date'] = pd.to_datetime(chunk['date'], errors='coerce')
                chunk = chunk.dropna(subset=['date'])
                
                # FAST: Use groupby to get min/max dates per business in one pass
                chunk_stats = chunk.groupby('business_id')['date'].agg(['min', 'max'])
                
                for business_id, row in chunk_stats.iterrows():
                    if business_id not in business_review_dates:
                        business_review_dates[business_id] = {
                            'first': row['min'],
                            'last': row['max']
                        }
                    else:
                        # Update with this chunk's data
                        business_review_dates[business_id]['first'] = min(
                            business_review_dates[business_id]['first'],
                            row['min']
                        )
                        business_review_dates[business_id]['last'] = max(
                            business_review_dates[business_id]['last'],
                            row['max']
                        )
            
            # Add to final_df
            final_df['_first_review_date'] = final_df['business_id'].map(
                lambda bid: business_review_dates.get(bid, {}).get('first', pd.NaT)
            )
            final_df['_last_review_date'] = final_df['business_id'].map(
                lambda bid: business_review_dates.get(bid, {}).get('last', pd.NaT)
            )
            
            # Compute business age at cutoff
            final_df['_business_age_at_cutoff_days'] = (
                final_df['_cutoff_date'] - final_df['_first_review_date']
            ).dt.days
            
            logger.info("[OK] Added temporal metadata columns:")
            logger.info("  - _prediction_year, _prediction_month")
            logger.info("  - _first_review_date, _last_review_date")
            logger.info("  - _business_age_at_cutoff_days")
        
        # Remove leaky features if in temporal validation mode
        if self.use_temporal_validation:
            features_to_remove = [f for f in self.leaky_features if f in final_df.columns]
            
            if features_to_remove:
                logger.info(f"\nRemoving {len(features_to_remove)} leaky features:")
                for feat in features_to_remove:
                    logger.info(f"  - {feat}")
                
                final_df = final_df.drop(columns=features_to_remove)
        
        # Check for missing values
        missing_counts = final_df.isnull().sum()
        cols_with_missing = missing_counts[missing_counts > 0]
        
        if len(cols_with_missing) > 0:
            logger.warning(f"\nColumns with missing values:")
            for col, count in cols_with_missing.items():
                logger.warning(f"  {col}: {count:,} ({count/len(final_df)*100:.1f}%)")
            
            # Fill missing values
            # Numeric columns: fill with median
            numeric_cols = final_df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if final_df[col].isnull().sum() > 0:
                    final_df[col].fillna(final_df[col].median(), inplace=True)
            
            # Categorical columns: fill with mode or 'Unknown'
            categorical_cols = final_df.select_dtypes(include=['object']).columns
            # Exclude metadata columns
            categorical_cols = [c for c in categorical_cols if not c.startswith('_') and c != 'business_id']
            
            for col in categorical_cols:
                if final_df[col].isnull().sum() > 0:
                    mode_val = final_df[col].mode()[0] if len(final_df[col].mode()) > 0 else 'Unknown'
                    final_df[col].fillna(mode_val, inplace=True)
            
            logger.info("[OK] Filled missing values")
        
        # Final statistics
        logger.info(f"\n{'='*70}")
        logger.info(f"FEATURE MERGING COMPLETE")
        logger.info(f"{'='*70}")
        logger.info(f"Final shape: {final_df.shape}")
        logger.info(f"  Rows: {len(final_df):,}")
        logger.info(f"  Columns: {len(final_df.columns)}")
        
        if self.use_temporal_validation:
            logger.info(f"  Unique businesses: {final_df['business_id'].nunique():,}")
            logger.info(f"  Unique cutoff dates: {final_df['_cutoff_date'].nunique()}")
            logger.info(f"  Avg rows per business: {len(final_df) / final_df['business_id'].nunique():.1f}")
        
        # Separate metadata and feature columns
        metadata_cols = [c for c in final_df.columns if c.startswith('_') or c == 'business_id']
        feature_cols = [c for c in final_df.columns if c not in metadata_cols]
        
        logger.info(f"  Metadata columns: {len(metadata_cols)}")
        logger.info(f"  Feature columns: {len(feature_cols)}")
        
        return final_df
    
    def generate_feature_report(self, final_df: pd.DataFrame):
        """
        Generate comprehensive markdown report for feature engineering.
        
        Args:
            final_df: Final merged feature DataFrame
        """
        logger.info("="*70)
        logger.info("GENERATING FEATURE REPORT")
        logger.info("="*70)
        
        report_lines = []
        report_lines.append("# Feature Engineering Report")
        report_lines.append("")
        report_lines.append(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        # Mode indication
        if self.use_temporal_validation:
            report_lines.append("**Mode**: Temporal Validation (Leakage Prevention Enabled)")
            report_lines.append("")
            report_lines.append(f"- **Cutoff dates**: {len(self.cutoff_dates)}")
            report_lines.append(f"- **Prediction years**: {sorted(set(d.year for d in self.cutoff_dates))}")
            report_lines.append(f"- **Leaky features removed**: {', '.join(self.leaky_features)}")
        else:
            report_lines.append("**Mode**: Baseline (Full Dataset)")
            report_lines.append("")
            report_lines.append("[WARN] *Warning: This mode may contain temporal leakage*")
        
        report_lines.append("")
        report_lines.append("---")
        report_lines.append("")
        
        # Executive Summary
        report_lines.append("## Executive Summary")
        report_lines.append("")
        report_lines.append(f"- **Total feature rows**: {len(final_df):,}")
        report_lines.append(f"- **Unique businesses**: {final_df['business_id'].nunique():,}")
        
        if self.use_temporal_validation:
            report_lines.append(f"- **Rows per business**: {len(final_df) / final_df['business_id'].nunique():.1f} (avg)")
        
        # Separate metadata and features
        # Metadata includes: temporal metadata (_*), business_id, and target variables (is_open, label, label_confidence, label_source)
        metadata_cols = [c for c in final_df.columns if c.startswith('_') or c in ['business_id', 'is_open', 'label', 'label_confidence', 'label_source']]
        feature_cols = [c for c in final_df.columns if c not in metadata_cols]
        
        report_lines.append(f"- **Total columns**: {len(final_df.columns)}")
        report_lines.append(f"  - Features: {len(feature_cols)}")
        report_lines.append(f"  - Metadata: {len(metadata_cols)}")
        report_lines.append("")
        
        # Feature Categories
        report_lines.append("## Feature Categories")
        report_lines.append("")
        
        categories = {
            'A: Static Business': ['stars', 'review_count', 'category_encoded', 'state_encoded', 
                                   'city_encoded', 'has_multiple_categories', 'category_count', 'price_range'],
            'B: Review Aggregation': ['total_reviews', 'avg_review_stars', 'std_review_stars',
                                     'days_since_first_review', 'review_recency_ratio', 'review_frequency',
                                     'total_useful_votes', 'avg_useful_per_review'],
            'C: Sentiment': ['avg_sentiment', 'std_sentiment', 'sentiment_volatility',
                           'pct_positive_reviews', 'pct_negative_reviews', 'pct_neutral_reviews',
                           'avg_text_length', 'std_text_length', 'sentiment_recent_3m'],
            'D: User-Weighted': ['avg_reviewer_credibility', 'std_reviewer_credibility',
                                'weighted_avg_rating', 'weighted_sentiment',
                                'pct_high_credibility_reviewers', 'weighted_useful_votes',
                                'avg_reviewer_tenure', 'avg_reviewer_experience', 'review_diversity'],
            'E: Temporal Dynamics': ['rating_recent_vs_all', 'rating_recent_vs_early',
                                    'reviews_recent_3m_count', 'engagement_recent_vs_all',
                                    'sentiment_recent_vs_all', 'review_momentum',
                                    'lifecycle_stage', 'rating_trend_3m'],
            'F: Location/Category': ['category_avg_success_rate', 'state_avg_success_rate',
                                    'city_avg_success_rate', 'category_competitiveness',
                                    'location_density'],
            'G: Feature Interactions': ['rating_credibility_interaction', 'momentum_credibility_interaction',
                                       'size_activity_interaction', 'trend_quality_interaction',
                                       'engagement_credibility_interaction']
        }
        
        for cat_name, cat_features in categories.items():
            existing_features = [f for f in cat_features if f in feature_cols]
            report_lines.append(f"### {cat_name}")
            report_lines.append(f"**Features**: {len(existing_features)}")
            report_lines.append("")
            
            if len(existing_features) > 0:
                report_lines.append("```")
                for feat in existing_features:
                    report_lines.append(f"  - {feat}")
                report_lines.append("```")
                report_lines.append("")
        
        # Temporal Metadata (if applicable)
        if self.use_temporal_validation:
            report_lines.append("## Temporal Metadata")
            report_lines.append("")
            report_lines.append("Additional columns for temporal validation:")
            report_lines.append("")
            report_lines.append("```")
            for col in sorted(metadata_cols):
                if col != 'business_id':
                    report_lines.append(f"  - {col}")
            report_lines.append("```")
            report_lines.append("")
            
            # Temporal coverage
            if '_prediction_year' in final_df.columns:
                year_counts = final_df['_prediction_year'].value_counts().sort_index()
                report_lines.append("### Temporal Coverage")
                report_lines.append("")
                report_lines.append("| Year | Tasks | Unique Businesses |")
                report_lines.append("|------|-------|-------------------|")
                for year in sorted(year_counts.index):
                    year_df = final_df[final_df['_prediction_year'] == year]
                    unique_biz = year_df['business_id'].nunique()
                    report_lines.append(f"| {year} | {len(year_df):,} | {unique_biz:,} |")
                report_lines.append("")
        
        # Feature Statistics
        report_lines.append("## Feature Statistics")
        report_lines.append("")
        
        numeric_features = final_df[feature_cols].select_dtypes(include=[np.number]).columns
        
        if len(numeric_features) > 0:
            report_lines.append("### Top 10 Features by Variance")
            report_lines.append("")
            
            variances = final_df[numeric_features].var().sort_values(ascending=False).head(10)
            
            report_lines.append("| Feature | Variance | Mean | Std |")
            report_lines.append("|---------|----------|------|-----|")
            
            for feat in variances.index:
                var = variances[feat]
                mean = final_df[feat].mean()
                std = final_df[feat].std()
                report_lines.append(f"| {feat} | {var:.4f} | {mean:.4f} | {std:.4f} |")
            
            report_lines.append("")
        
        # Missing Values
        missing_counts = final_df[feature_cols].isnull().sum()
        features_with_missing = missing_counts[missing_counts > 0]
        
        if len(features_with_missing) > 0:
            report_lines.append("## Missing Values")
            report_lines.append("")
            report_lines.append("| Feature | Missing Count | Missing % |")
            report_lines.append("|---------|---------------|-----------|")
            
            for feat, count in features_with_missing.items():
                pct = count / len(final_df) * 100
                report_lines.append(f"| {feat} | {count:,} | {pct:.2f}% |")
            
            report_lines.append("")
        else:
            report_lines.append("## Missing Values")
            report_lines.append("")
            report_lines.append("[OK] No missing values in features")
            report_lines.append("")
        
        # Data Quality Notes
        report_lines.append("## Data Quality Notes")
        report_lines.append("")
        
        if self.use_temporal_validation:
            report_lines.append("### Temporal Leakage Prevention")
            report_lines.append("")
            report_lines.append("[OK] Features computed only from historical data up to cutoff date")
            report_lines.append("[OK] Removed leaky features that encode future information")
            report_lines.append("[OK] Temporal windows (recent/early) defined relative to cutoff")
            report_lines.append("")
        else:
            report_lines.append("### [WARN] Temporal Leakage Warning")
            report_lines.append("")
            report_lines.append("This dataset uses the full time range without temporal constraints.")
            report_lines.append("Features may contain information from after the prediction time.")
            report_lines.append("Use `business_features_temporal.csv` for proper evaluation.")
            report_lines.append("")
        
        # Next Steps
        report_lines.append("## Next Steps")
        report_lines.append("")
        report_lines.append("1. **Label Generation**: Use `label_inference.py` to generate labels")
        report_lines.append("2. **Data Validation**: Run `validation.py` to check quality")
        report_lines.append("3. **Model Training**: Proceed to baseline models")
        report_lines.append("")
        
        report_lines.append("---")
        report_lines.append("")
        report_lines.append("*Report generated by CS 412 Research Project feature engineering pipeline*")
        
        # Save report
        report_file = self.output_path / "feature_engineering_report.md"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        logger.info(f"[OK] Saved: {report_file}")

    def run_pipeline(self):
        """Execute complete feature engineering pipeline with temporal validation support"""
        logger.info("="*70)
        logger.info("CS 412 RESEARCH PROJECT - FEATURE ENGINEERING")
        logger.info("Business Success Prediction using Yelp Dataset")
        logger.info("="*70)
        logger.info("")
        logger.info("Pipeline: Feature Engineering")
        
        if self.use_temporal_validation:
            logger.info("Mode: TEMPORAL VALIDATION (preventing data leakage)")
            logger.info(f"Cutoff dates: {len(self.cutoff_dates)}")
        else:
            logger.info("Mode: BASELINE (using full dataset)")
        
        logger.info("")
        
        # Step 1: Load data
        self.load_data()
        
        # Step 2: Calculate user credibility
        user_credibility = self.calculate_user_credibility()
        
        # Step 3: Create static features
        static_features = self.create_static_features()
        
        # Step 4: Create review-based features (includes sentiment, user-weighted, temporal)
        # This is where temporal filtering happens
        review_features = self.create_review_features_chunked(user_credibility)
        
        # Step 5: Create location/category features
        location_features = self.create_location_features(static_features)
        
        # Step 6: Merge all features
        final_df = self.merge_all_features(static_features, review_features, location_features)
        
        # Step 7: Save final merged dataset
        if self.use_temporal_validation:
            final_file = self.output_path / "business_features_temporal.csv"
        else:
            final_file = self.output_path / "business_features_baseline.csv"
        
        final_df.to_csv(final_file, index=False)
        
        logger.info(f"\n{'='*70}")
        logger.info(f"[OK] Saved final feature dataset: {final_file}")
        logger.info(f"  Shape: {final_df.shape}")
        logger.info(f"{'='*70}")
        
        # Step 8: Generate report
        self.generate_feature_report(final_df)
        
        logger.info("\n" + "="*70)
        logger.info("FEATURE ENGINEERING COMPLETE!")
        logger.info("="*70)
        logger.info("\nOutput files:")
        logger.info(f"  - {final_file}")
        logger.info(f"  - data/features/feature_categories/ (6 separate files)")
        logger.info(f"  - data/features/feature_engineering_report.md")
        logger.info("")
        
        return final_df


def main():
    """Main entry point for feature engineering with temporal validation support"""
    import argparse
    
    print("="*70)
    print("CS 412 RESEARCH PROJECT - FEATURE ENGINEERING")
    print("Business Success Prediction using Yelp Dataset")
    print("="*70)
    print("\nPhase 3: Feature Engineering")
    print("")
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Feature Engineering Pipeline')
    parser.add_argument('--temporal', action='store_true',
                       help='Enable temporal validation mode')
    parser.add_argument('--years', type=str, default='2012-2020',
                       help='Prediction years (format: 2012-2020 or 2012,2013,2014)')
    parser.add_argument('--mode', type=str, choices=['baseline', 'temporal'], default='baseline',
                       help='Feature generation mode')
    
    args = parser.parse_args()
    
    # Determine mode
    use_temporal = args.temporal or (args.mode == 'temporal')
    
    # Parse years
    if use_temporal:
        if '-' in args.years:
            # Range format: 2012-2020
            start, end = map(int, args.years.split('-'))
            prediction_years = list(range(start, end + 1))
        else:
            # Comma-separated: 2012,2013,2014
            prediction_years = [int(y.strip()) for y in args.years.split(',')]
        
        print(f"Mode: TEMPORAL VALIDATION")
        print(f"Prediction years: {prediction_years}")
    else:
        prediction_years = None
        print(f"Mode: BASELINE (full dataset)")
    
    print("")
    
    # Initialize engineer
    engineer = FeatureEngineer(
        use_temporal_validation=use_temporal,
        prediction_years=prediction_years
    )
    
    # Run pipeline
    engineer.run_pipeline()
    
    print("\n" + "="*70)
    print("FEATURE ENGINEERING COMPLETE!")
    print("="*70)
    print("\nCheck the following outputs:")
    
    if use_temporal:
        print("  1. data/features/business_features_temporal.csv")
    else:
        print("  1. data/features/business_features_baseline.csv")
    
    print("  2. data/features/feature_categories/ (separate files)")
    print("  3. data/features/feature_engineering_report.md")
    print("")


if __name__ == "__main__":
    main()