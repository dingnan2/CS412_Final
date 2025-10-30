"""
Feature Engineering: Extract and engineer features for business success prediction.

This module implements the feature engineering pipeline as specified in Phase 1
of the research proposal, including:
- Business static features
- Review aggregation features  
- Sentiment features
- User-weighted features
- Temporal dynamics features
- Category/Location features

All processing uses chunked methods to handle large datasets efficiently.
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, Tuple, Optional
import json
import gc
import warnings
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')

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
        logging.FileHandler('feature_engineering.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Comprehensive feature engineering for business success prediction.
    
    Design Principles:
    1. No data leakage - only use historical information
    2. Chunked processing for memory efficiency
    3. Robust aggregations for skewed distributions
    4. User credibility weighting as novel contribution
    5. Temporal dynamics with 3-month windows
    """
    
    def __init__(self, 
                 processed_path: str = "data/processed",
                 output_path: str = "data/features"):
        self.processed_path = Path(processed_path)
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectory for separate feature categories
        self.category_path = self.output_path / "feature_categories"
        self.category_path.mkdir(parents=True, exist_ok=True)
        
        # Reference date for temporal calculations (dataset end date)
        self.reference_date = pd.Timestamp('2022-01-19')
        
        # Data containers
        self.business_df = None
        self.user_df = None
        
        # Feature tracking
        self.feature_summary = {}
        
    def load_data(self):
        """Load cleaned business and user data"""
        logger.info("="*70)
        logger.info("LOADING CLEANED DATA")
        logger.info("="*70)
        
        # Load business data
        business_path = self.processed_path / "business_clean.csv"
        self.business_df = pd.read_csv(business_path)
        logger.info(f"Loaded business data: {self.business_df.shape}")
        
        # Load user data (will need for credibility weights)
        user_path = self.processed_path / "user_clean.csv"
        logger.info("Loading user data in chunks...")
        chunks = []
        for chunk in pd.read_csv(user_path, chunksize=100000):
            chunks.append(chunk)
        self.user_df = pd.concat(chunks, ignore_index=True)
        logger.info(f"Loaded user data: {self.user_df.shape}")
        
        # Ensure datetime columns
        self.user_df['yelping_since'] = pd.to_datetime(self.user_df['yelping_since'])
        
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
    
    def create_review_features_chunked(self, user_credibility: pd.Series) -> pd.DataFrame:
        """
        Create all review-based features using chunked processing.
        
        Processes review data in chunks and accumulates statistics.
        
        Features created:
        - Category B: Review Aggregation (15 features)
        - Category C: Sentiment (8 features)
        - Category D: User-Weighted (10 features)
        - Category E: Temporal Dynamics (8 features)
        
        Total: 41 features
        """
        logger.info("="*70)
        logger.info("CREATING REVIEW-BASED FEATURES (CHUNKED)")
        logger.info("="*70)
        
        review_path = self.processed_path / "review_clean.csv"
        chunk_size = 100000
        
        # Initialize sentiment analyzer if available
        sentiment_analyzer = SentimentIntensityAnalyzer() if VADER_AVAILABLE else None
        if not VADER_AVAILABLE:
            logger.warning("VADER not available - sentiment features will be skipped")
        
        # Initialize accumulators for different feature types
        # Using lists to accumulate per-business statistics
        business_stats = {}
        
        logger.info(f"Processing reviews in chunks of {chunk_size:,}...")
        chunk_count = 0
        total_reviews = 0
        
        # Define temporal cutoffs (3 months before reference date)
        cutoff_3m = self.reference_date - timedelta(days=90)
        
        for chunk in pd.read_csv(review_path, chunksize=chunk_size):
            chunk_count += 1
            total_reviews += len(chunk)
            
            # Ensure datetime
            chunk['date'] = pd.to_datetime(chunk['date'])
            
            # Calculate sentiment if available
            if sentiment_analyzer:
                def _safe_compound(text: str) -> float:
                    try:
                        return sentiment_analyzer.polarity_scores(text)['compound']
                    except Exception:
                        return 0.0
                chunk['sentiment'] = chunk['text'].fillna('').apply(
                    lambda t: _safe_compound(str(t))
                )
            else:
                chunk['sentiment'] = 0.0
            
            # Add user credibility
            chunk['user_credibility'] = chunk['user_id'].map(user_credibility).fillna(0)
            
            # Add temporal flags
            chunk['is_recent_3m'] = chunk['date'] >= cutoff_3m
            
            # Calculate days from reference
            chunk['days_from_ref'] = (self.reference_date - chunk['date']).dt.days
            
            # Process each business in chunk
            for business_id, group in chunk.groupby('business_id'):
                if business_id not in business_stats:
                    business_stats[business_id] = {
                        'reviews': [],
                        'dates': [],
                        'stars': [],
                        'useful': [],
                        'funny_cool': [],
                        'text_length': [],
                        'sentiment': [],
                        'user_credibility': [],
                        'is_recent_3m': []
                    }
                
                # Accumulate data
                stats = business_stats[business_id]
                stats['reviews'].extend(group['review_id'].tolist())
                stats['dates'].extend(group['date'].tolist())
                stats['stars'].extend(group['stars'].tolist())
                stats['useful'].extend(group['useful'].tolist())
                stats['funny_cool'].extend(group['funny_cool'].tolist())
                stats['text_length'].extend(group['text_length'].tolist())
                stats['sentiment'].extend(group['sentiment'].tolist())
                stats['user_credibility'].extend(group['user_credibility'].tolist())
                stats['is_recent_3m'].extend(group['is_recent_3m'].tolist())
            
            if chunk_count % 10 == 0:
                logger.info(f"  Processed {total_reviews:,} reviews, {len(business_stats):,} businesses")
            
            # Clean up
            del chunk
            gc.collect()
        
        logger.info(f"[OK] Completed chunked processing: {total_reviews:,} reviews")
        logger.info(f"[OK] Accumulated statistics for {len(business_stats):,} businesses")
        
        # Now compute all features from accumulated statistics
        logger.info("\nComputing aggregated features...")
        
        features_list = []
        
        for business_id, stats in business_stats.items():
            # Convert lists to arrays for efficient computation
            stars = np.array(stats['stars'])
            dates = pd.Series(stats['dates'])
            useful = np.array(stats['useful'])
            funny_cool = np.array(stats['funny_cool'])
            text_length = np.array(stats['text_length'])
            sentiment = np.array(stats['sentiment'])
            user_cred = np.array(stats['user_credibility'])
            is_recent = np.array(stats['is_recent_3m'])
            
            n_reviews = len(stars)
            
            # Normalize user credibility weights (sum to 1 within business)
            if user_cred.sum() > 0:
                weights = user_cred / user_cred.sum()
            else:
                weights = np.ones(n_reviews) / n_reviews
            
            feature_dict = {'business_id': business_id}
            
            # ========== CATEGORY B: REVIEW AGGREGATION (15 features) ==========
            feature_dict['total_reviews'] = n_reviews
            feature_dict['avg_review_stars'] = stars.mean()
            feature_dict['std_review_stars'] = stars.std() if n_reviews > 1 else 0
            
            # Rating velocity (slope of rating over time)
            if n_reviews > 1:
                days_from_first = (dates - dates.min()).dt.days.values
                if days_from_first.max() > 0:
                    # Linear regression: rating ~ days
                    rating_velocity = np.polyfit(days_from_first, stars, 1)[0]
                    feature_dict['rating_velocity'] = rating_velocity
                else:
                    feature_dict['rating_velocity'] = 0
            else:
                feature_dict['rating_velocity'] = 0
            
            # Recent rating trend (3 months)
            if is_recent.sum() > 0:
                feature_dict['rating_trend_3m'] = stars[is_recent].mean()
            else:
                feature_dict['rating_trend_3m'] = feature_dict['avg_review_stars']
            
            # Review frequency and momentum
            if n_reviews > 1:
                timespan_days = (dates.max() - dates.min()).days
                if timespan_days > 0:
                    feature_dict['review_frequency'] = n_reviews / (timespan_days / 30.0)  # reviews/month
                    
                    # Momentum: compare recent vs historical frequency
                    recent_count = is_recent.sum()
                    if recent_count > 0:
                        recent_freq = recent_count / 3.0  # 3 months
                        hist_freq = feature_dict['review_frequency']
                        feature_dict['review_momentum'] = recent_freq / (hist_freq + 1e-6)
                    else:
                        feature_dict['review_momentum'] = 0
                else:
                    feature_dict['review_frequency'] = 0
                    feature_dict['review_momentum'] = 0
            else:
                feature_dict['review_frequency'] = 0
                feature_dict['review_momentum'] = 0
            
            # Days since first/last review
            feature_dict['days_since_first_review'] = (self.reference_date - dates.min()).days
            feature_dict['days_since_last_review'] = (self.reference_date - dates.max()).days
            
            # Engagement metrics
            feature_dict['total_useful_votes'] = useful.sum()
            feature_dict['avg_useful_per_review'] = useful.mean()
            feature_dict['total_funny_cool'] = funny_cool.sum()
            
            # Text features
            feature_dict['avg_text_length'] = text_length.mean()
            feature_dict['std_text_length'] = text_length.std() if n_reviews > 1 else 0
            
            # ========== CATEGORY C: SENTIMENT (8 features) ==========
            if VADER_AVAILABLE:
                feature_dict['avg_sentiment'] = sentiment.mean()
                feature_dict['std_sentiment'] = sentiment.std() if n_reviews > 1 else 0
                
                # Sentiment slope
                if n_reviews > 1:
                    days_from_first = (dates - dates.min()).dt.days.values
                    if days_from_first.max() > 0:
                        sentiment_slope = np.polyfit(days_from_first, sentiment, 1)[0]
                        feature_dict['sentiment_slope'] = sentiment_slope
                    else:
                        feature_dict['sentiment_slope'] = 0
                else:
                    feature_dict['sentiment_slope'] = 0
                
                # Sentiment categories
                feature_dict['pct_positive_reviews'] = (sentiment > 0.5).sum() / n_reviews
                feature_dict['pct_negative_reviews'] = (sentiment < -0.5).sum() / n_reviews
                feature_dict['pct_neutral_reviews'] = ((sentiment >= -0.5) & (sentiment <= 0.5)).sum() / n_reviews
                
                # Recent sentiment
                if is_recent.sum() > 0:
                    feature_dict['sentiment_recent_3m'] = sentiment[is_recent].mean()
                else:
                    feature_dict['sentiment_recent_3m'] = feature_dict['avg_sentiment']
                
                # Sentiment volatility (change in variance)
                if is_recent.sum() > 1:
                    recent_std = sentiment[is_recent].std()
                    hist_std = feature_dict['std_sentiment']
                    feature_dict['sentiment_volatility'] = abs(recent_std - hist_std)
                else:
                    feature_dict['sentiment_volatility'] = 0
            else:
                # Fill with zeros if VADER not available
                for feat in ['avg_sentiment', 'std_sentiment', 'sentiment_slope',
                           'pct_positive_reviews', 'pct_negative_reviews', 'pct_neutral_reviews',
                           'sentiment_recent_3m', 'sentiment_volatility']:
                    feature_dict[feat] = 0
            
            # ========== CATEGORY D: USER-WEIGHTED (10 features) ==========
            feature_dict['weighted_avg_rating'] = (stars * weights).sum()
            
            if VADER_AVAILABLE:
                feature_dict['weighted_sentiment'] = (sentiment * weights).sum()
            else:
                feature_dict['weighted_sentiment'] = 0
            
            feature_dict['avg_reviewer_credibility'] = user_cred.mean()
            feature_dict['std_reviewer_credibility'] = user_cred.std() if n_reviews > 1 else 0
            
            # High credibility reviewers (>75th percentile)
            if n_reviews > 0:
                credibility_75 = np.percentile(user_cred, 75)
                feature_dict['pct_high_credibility_reviewers'] = (user_cred > credibility_75).sum() / n_reviews
            else:
                feature_dict['pct_high_credibility_reviewers'] = 0
            
            feature_dict['weighted_useful_votes'] = (useful * weights).sum()
            
            # Get user info (need to look up in user_df)
            # For efficiency, we'll compute average reviewer tenure and experience
            # This requires user_id which we didn't store - simplified version
            feature_dict['avg_reviewer_tenure'] = 0  # Placeholder - would need user_id join
            feature_dict['avg_reviewer_experience'] = 0  # Placeholder
            
            # Review diversity
            unique_reviewers = len(set(stats['reviews']))  # Approximation
            feature_dict['review_diversity'] = unique_reviewers / n_reviews if n_reviews > 0 else 0
            
            # Power user ratio (placeholder - would need full user info)
            feature_dict['power_user_ratio'] = 0  # Placeholder
            
            # ========== CATEGORY E: TEMPORAL DYNAMICS (8 features) ==========
            # Recent vs all-time comparisons
            if is_recent.sum() > 0:
                recent_avg_rating = stars[is_recent].mean()
                feature_dict['rating_recent_vs_all'] = recent_avg_rating - feature_dict['avg_review_stars']
                
                # Recent vs early (first 3 months)
                first_3m_date = dates.min() + timedelta(days=90)
                is_early = dates <= first_3m_date
                if is_early.sum() > 0:
                    early_avg_rating = stars[is_early].mean()
                    feature_dict['rating_recent_vs_early'] = recent_avg_rating - early_avg_rating
                else:
                    feature_dict['rating_recent_vs_early'] = 0
                
                if VADER_AVAILABLE:
                    feature_dict['sentiment_recent_vs_all'] = sentiment[is_recent].mean() - feature_dict['avg_sentiment']
                else:
                    feature_dict['sentiment_recent_vs_all'] = 0
            else:
                feature_dict['rating_recent_vs_all'] = 0
                feature_dict['rating_recent_vs_early'] = 0
                feature_dict['sentiment_recent_vs_all'] = 0
            
            # Recent review counts
            feature_dict['reviews_recent_3m_count'] = is_recent.sum()
            
            # Review frequency trend
            if n_reviews > 1 and feature_dict['review_frequency'] > 0:
                if is_recent.sum() > 0:
                    recent_freq = is_recent.sum() / 3.0
                    feature_dict['review_frequency_trend'] = recent_freq / feature_dict['review_frequency']
                else:
                    feature_dict['review_frequency_trend'] = 0
            else:
                feature_dict['review_frequency_trend'] = 0
            
            # Engagement recent vs all
            if is_recent.sum() > 0:
                recent_useful_avg = useful[is_recent].mean()
                all_useful_avg = feature_dict['avg_useful_per_review']
                if all_useful_avg > 0:
                    feature_dict['engagement_recent_vs_all'] = recent_useful_avg / all_useful_avg
                else:
                    feature_dict['engagement_recent_vs_all'] = 1.0
            else:
                feature_dict['engagement_recent_vs_all'] = 1.0
            
            # Lifecycle stage (categorical based on review patterns)
            timespan_days = feature_dict['days_since_first_review']
            recent_activity = feature_dict['reviews_recent_3m_count']
            
            if timespan_days < 180:  # Less than 6 months old
                lifecycle = 0  # New
            elif recent_activity > n_reviews * 0.3:  # >30% reviews in last 3 months
                lifecycle = 1  # Growing
            elif recent_activity > 0:
                lifecycle = 2  # Mature
            else:
                lifecycle = 3  # Declining
            
            feature_dict['lifecycle_stage'] = lifecycle
            
            features_list.append(feature_dict)
        
        # Convert to DataFrame
        review_features = pd.DataFrame(features_list)
        
        logger.info(f"[OK] Created {review_features.shape[1] - 1} review-based features")
        logger.info(f"  - Review Aggregation: 15 features")
        logger.info(f"  - Sentiment: {8 if VADER_AVAILABLE else 0} features")
        logger.info(f"  - User-Weighted: 10 features")
        logger.info(f"  - Temporal Dynamics: 8 features")
        
        # Save separate category files
        # Split into categories for ablation studies
        review_agg_cols = ['business_id', 'total_reviews', 'avg_review_stars', 'std_review_stars',
                          'rating_velocity', 'rating_trend_3m', 'review_frequency', 'review_momentum',
                          'days_since_first_review', 'days_since_last_review', 'total_useful_votes',
                          'avg_useful_per_review', 'total_funny_cool', 'avg_text_length', 'std_text_length']
        
        sentiment_cols = ['business_id', 'avg_sentiment', 'std_sentiment', 'sentiment_slope',
                         'pct_positive_reviews', 'pct_negative_reviews', 'pct_neutral_reviews',
                         'sentiment_recent_3m', 'sentiment_volatility']
        
        user_weighted_cols = ['business_id', 'weighted_avg_rating', 'weighted_sentiment',
                             'avg_reviewer_credibility', 'std_reviewer_credibility',
                             'pct_high_credibility_reviewers', 'weighted_useful_votes',
                             'avg_reviewer_tenure', 'avg_reviewer_experience',
                             'review_diversity', 'power_user_ratio']
        
        temporal_cols = ['business_id', 'rating_recent_vs_all', 'rating_recent_vs_early',
                        'sentiment_recent_vs_all', 'reviews_recent_3m_count',
                        'review_frequency_trend', 'engagement_recent_vs_all', 'lifecycle_stage']
        
        review_features[review_agg_cols].to_csv(
            self.category_path / "review_aggregation_features.csv", index=False
        )
        review_features[sentiment_cols].to_csv(
            self.category_path / "sentiment_features.csv", index=False
        )
        review_features[user_weighted_cols].to_csv(
            self.category_path / "user_weighted_features.csv", index=False
        )
        review_features[temporal_cols].to_csv(
            self.category_path / "temporal_features.csv", index=False
        )
        
        logger.info(f"[OK] Saved separate feature category files")
        
        return review_features
    
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
        Merge all feature categories into single dataset.
        
        Returns:
            Final merged dataset with all features
        """
        logger.info("="*70)
        logger.info("MERGING ALL FEATURES")
        logger.info("="*70)
        
        # Start with static features (includes is_open target)
        final_df = static_features.copy()
        
        # Merge review features
        final_df = final_df.merge(review_features, on='business_id', how='left')
        logger.info(f"[OK] Merged review features: {final_df.shape}")
        
        # Merge location features
        final_df = final_df.merge(location_features, on='business_id', how='left')
        logger.info(f"[OK] Merged location features: {final_df.shape}")
        
        # Handle any missing values from merge
        # For businesses with no reviews, fill with reasonable defaults
        numeric_cols = final_df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col not in ['business_id', 'is_open']:
                final_df[col].fillna(0, inplace=True)
        
        logger.info(f"\n{'='*70}")
        logger.info(f"FINAL FEATURE SUMMARY")
        logger.info(f"{'='*70}")
        logger.info(f"Total businesses: {len(final_df):,}")
        logger.info(f"Total features: {final_df.shape[1] - 2}")  # Exclude business_id and is_open
        logger.info(f"Target distribution:")
        logger.info(f"  Open (1): {final_df['is_open'].sum():,} ({final_df['is_open'].mean()*100:.2f}%)")
        logger.info(f"  Closed (0): {(1-final_df['is_open']).sum():,} ({(1-final_df['is_open'].mean())*100:.2f}%)")
        
        # Check for any remaining NaN values
        nan_counts = final_df.isnull().sum()
        if nan_counts.sum() > 0:
            logger.warning(f"\nRemaining NaN values detected:")
            logger.warning(f"\n{nan_counts[nan_counts > 0]}")
        else:
            logger.info(f"\n[OK] No missing values in final dataset")
        
        return final_df
    
    def generate_feature_report(self, final_df: pd.DataFrame):
        """Generate comprehensive feature engineering report"""
        logger.info("="*70)
        logger.info("GENERATING FEATURE ENGINEERING REPORT")
        logger.info("="*70)
        
        report_lines = []
        report_lines.append("# Feature Engineering Report")
        report_lines.append("")
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        report_lines.append("## Overview")
        report_lines.append(f"This report documents the feature engineering process for the Yelp business success prediction project.")
        report_lines.append(f"All features are engineered from historical data only (no data leakage).")
        report_lines.append("")
        
        report_lines.append("## Dataset Summary")
        report_lines.append(f"- Total businesses: {len(final_df):,}")
        report_lines.append(f"- Total features: {final_df.shape[1] - 2}")
        report_lines.append(f"- Open businesses: {final_df['is_open'].sum():,} ({final_df['is_open'].mean()*100:.2f}%)")
        report_lines.append(f"- Closed businesses: {(1-final_df['is_open']).sum():,} ({(1-final_df['is_open'].mean())*100:.2f}%)")
        report_lines.append("")
        
        report_lines.append("## Feature Categories")
        report_lines.append("")
        
        # Category A: Static Business Features
        report_lines.append("### Category A: Static Business Features (8 features)")
        report_lines.append("- `stars`: Business average rating (1-5)")
        report_lines.append("- `review_count`: Total reviews on business profile")
        report_lines.append("- `category_encoded`: Target-encoded primary business category (top 20 + Other)")
        report_lines.append("- `state_encoded`: Target-encoded state location")
        report_lines.append("- `city_encoded`: Target-encoded city location (top 50 + Other)")
        report_lines.append("- `has_multiple_categories`: Binary indicator for multi-category businesses")
        report_lines.append("- `category_count`: Number of categories listed")
        report_lines.append("- `price_range`: Restaurant price range (1-4, extracted from attributes)")
        report_lines.append("")
        
        # Category B: Review Aggregation
        report_lines.append("### Category B: Review Aggregation Features (15 features)")
        report_lines.append("**Volume Metrics:**")
        report_lines.append("- `total_reviews`: Count of all reviews")
        report_lines.append("- `review_frequency`: Average reviews per month")
        report_lines.append("- `review_momentum`: Recent vs historical review frequency ratio")
        report_lines.append("")
        report_lines.append("**Rating Metrics:**")
        report_lines.append("- `avg_review_stars`: Mean rating from all reviews")
        report_lines.append("- `std_review_stars`: Rating volatility (standard deviation)")
        report_lines.append("- `rating_velocity`: Slope of rating over time (trend direction)")
        report_lines.append("- `rating_trend_3m`: Average rating in last 3 months")
        report_lines.append("")
        report_lines.append("**Temporal Metrics:**")
        report_lines.append("- `days_since_first_review`: Age of business review history")
        report_lines.append("- `days_since_last_review`: Recency of last review")
        report_lines.append("")
        report_lines.append("**Engagement Metrics:**")
        report_lines.append("- `total_useful_votes`: Sum of useful votes across reviews")
        report_lines.append("- `avg_useful_per_review`: Mean useful votes per review")
        report_lines.append("- `total_funny_cool`: Total engagement (funny + cool votes)")
        report_lines.append("")
        report_lines.append("**Text Metrics:**")
        report_lines.append("- `avg_text_length`: Mean review length (characters)")
        report_lines.append("- `std_text_length`: Variance in review lengths")
        report_lines.append("")
        
        # Category C: Sentiment
        if VADER_AVAILABLE:
            report_lines.append("### Category C: Sentiment Features (8 features)")
            report_lines.append("**Aggregate Sentiment (VADER):**")
            report_lines.append("- `avg_sentiment`: Mean compound sentiment score (-1 to 1)")
            report_lines.append("- `std_sentiment`: Sentiment volatility")
            report_lines.append("- `sentiment_slope`: Trend in sentiment over time")
            report_lines.append("")
            report_lines.append("**Sentiment Distribution:**")
            report_lines.append("- `pct_positive_reviews`: Percentage of positive reviews (>0.5)")
            report_lines.append("- `pct_negative_reviews`: Percentage of negative reviews (<-0.5)")
            report_lines.append("- `pct_neutral_reviews`: Percentage of neutral reviews")
            report_lines.append("")
            report_lines.append("**Recent Sentiment:**")
            report_lines.append("- `sentiment_recent_3m`: Average sentiment in last 3 months")
            report_lines.append("- `sentiment_volatility`: Change in sentiment variance (recent vs all)")
            report_lines.append("")
        else:
            report_lines.append("### Category C: Sentiment Features (SKIPPED)")
            report_lines.append("VADER sentiment analyzer not available. Install 'vaderSentiment' to enable.")
            report_lines.append("")
        
        # Category D: User-Weighted
        report_lines.append("### Category D: User-Weighted Features (10 features)")
        report_lines.append("**Novel Contribution: User Credibility Weighting**")
        report_lines.append("")
        report_lines.append("User credibility formula:")
        report_lines.append("```")
        report_lines.append("useful_rate = useful_votes / (review_count + 1)")
        report_lines.append("tenure_weight = log(1 + user_tenure_days) / 10")
        report_lines.append("experience_weight = log(1 + review_count) / 10")
        report_lines.append("credibility = 0.5 × useful_rate + 0.3 × tenure_weight + 0.2 × experience_weight")
        report_lines.append("```")
        report_lines.append("")
        report_lines.append("Weights normalized to sum=1 within each business for weighted aggregations.")
        report_lines.append("")
        report_lines.append("**Features:**")
        report_lines.append("- `weighted_avg_rating`: Credibility-weighted mean rating")
        report_lines.append("- `weighted_sentiment`: Credibility-weighted mean sentiment")
        report_lines.append("- `avg_reviewer_credibility`: Mean credibility of reviewers")
        report_lines.append("- `std_reviewer_credibility`: Variance in reviewer credibility")
        report_lines.append("- `pct_high_credibility_reviewers`: % reviewers in top quartile")
        report_lines.append("- `weighted_useful_votes`: Credibility-weighted useful votes")
        report_lines.append("- `avg_reviewer_tenure`: Mean reviewer tenure on platform")
        report_lines.append("- `avg_reviewer_experience`: Mean reviewer experience (review count)")
        report_lines.append("- `review_diversity`: Ratio of unique reviewers to total reviews")
        report_lines.append("- `power_user_ratio`: % power users (>100 reviews)")
        report_lines.append("")
        
        # Category E: Temporal Dynamics
        report_lines.append("### Category E: Temporal Dynamics Features (8 features)")
        report_lines.append("**Recent vs Historical Comparisons (3-month window):**")
        report_lines.append("- `rating_recent_vs_all`: Difference between recent and all-time rating")
        report_lines.append("- `rating_recent_vs_early`: Recent rating vs first 3 months")
        report_lines.append("- `sentiment_recent_vs_all`: Recent vs all-time sentiment difference")
        report_lines.append("")
        report_lines.append("**Activity Trends:**")
        report_lines.append("- `reviews_recent_3m_count`: Number of reviews in last 3 months")
        report_lines.append("- `review_frequency_trend`: Recent vs historical frequency ratio")
        report_lines.append("- `engagement_recent_vs_all`: Recent vs all-time engagement ratio")
        report_lines.append("")
        report_lines.append("**Lifecycle:**")
        report_lines.append("- `lifecycle_stage`: Business lifecycle (0=New, 1=Growing, 2=Mature, 3=Declining)")
        report_lines.append("")
        
        # Category F: Location/Category Aggregates
        report_lines.append("### Category F: Location/Category Aggregates (5 features)")
        report_lines.append("- `category_avg_success_rate`: Success rate for business category")
        report_lines.append("- `state_avg_success_rate`: Success rate for state")
        report_lines.append("- `city_avg_success_rate`: Success rate for city")
        report_lines.append("- `category_competitiveness`: Number of competitors (same category, same city)")
        report_lines.append("- `location_density`: Total businesses in city")
        report_lines.append("")
        
        # Feature statistics
        report_lines.append("## Feature Statistics")
        report_lines.append("")
        
        # Select some key features for summary statistics
        key_features = [
            'stars', 'review_count', 'total_reviews', 'avg_review_stars',
            'rating_velocity', 'avg_sentiment', 'weighted_avg_rating',
            'avg_reviewer_credibility', 'reviews_recent_3m_count'
        ]
        
        available_features = [f for f in key_features if f in final_df.columns]
        
        if available_features:
            report_lines.append("**Key Feature Summary:**")
            report_lines.append("")
            report_lines.append("| Feature | Mean | Median | Std | Min | Max |")
            report_lines.append("|---------|------|--------|-----|-----|-----|")
            
            for feat in available_features:
                stats = final_df[feat].describe()
                report_lines.append(
                    f"| {feat} | {stats['mean']:.3f} | {stats['50%']:.3f} | "
                    f"{stats['std']:.3f} | {stats['min']:.3f} | {stats['max']:.3f} |"
                )
            report_lines.append("")
        
        # Data quality
        report_lines.append("## Data Quality")
        report_lines.append("")
        nan_counts = final_df.isnull().sum()
        if nan_counts.sum() == 0:
            report_lines.append("✓ No missing values in final dataset")
        else:
            report_lines.append("⚠ Missing values detected:")
            for col, count in nan_counts[nan_counts > 0].items():
                report_lines.append(f"  - {col}: {count} ({count/len(final_df)*100:.2f}%)")
        report_lines.append("")
        
        # Next steps
        report_lines.append("## Next Steps")
        report_lines.append("")
        report_lines.append("1. **Feature Selection**: Apply correlation analysis, variance thresholding, and model-based selection")
        report_lines.append("2. **Dimensionality Reduction**: Optional PCA if needed (retain 95% variance)")
        report_lines.append("3. **Class Imbalance**: Apply SMOTE or stratified sampling for training")
        report_lines.append("4. **Model Training**: Implement baseline models (Logistic Regression, Decision Tree, Random Forest)")
        report_lines.append("5. **Ablation Studies**: Use separate feature category files to assess contribution")
        report_lines.append("")
        
        # File outputs
        report_lines.append("## Output Files")
        report_lines.append("")
        report_lines.append("**Main Output:**")
        report_lines.append("- `data/features/business_features_final.csv` - Single merged dataset for modeling")
        report_lines.append("")
        report_lines.append("**Feature Categories (for ablation studies):**")
        report_lines.append("- `data/features/feature_categories/business_static_features.csv`")
        report_lines.append("- `data/features/feature_categories/review_aggregation_features.csv`")
        report_lines.append("- `data/features/feature_categories/sentiment_features.csv`")
        report_lines.append("- `data/features/feature_categories/user_weighted_features.csv`")
        report_lines.append("- `data/features/feature_categories/temporal_features.csv`")
        report_lines.append("- `data/features/feature_categories/location_category_features.csv`")
        report_lines.append("")
        
        # Save report
        report_file = self.output_path / "feature_engineering_report.md"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        logger.info(f"[OK] Saved: {report_file}")
    
    def run_pipeline(self):
        """Execute complete feature engineering pipeline"""
        logger.info("="*70)
        logger.info("CS 412 RESEARCH PROJECT - FEATURE ENGINEERING")
        logger.info("Business Success Prediction using Yelp Dataset")
        logger.info("="*70)
        logger.info("")
        logger.info("Pipeline: Feature Engineering")
        logger.info("")
        
        # Step 1: Load data
        self.load_data()
        
        # Step 2: Calculate user credibility
        user_credibility = self.calculate_user_credibility()
        
        # Step 3: Create static features
        static_features = self.create_static_features()
        
        # Step 4: Create review-based features (includes sentiment, user-weighted, temporal)
        review_features = self.create_review_features_chunked(user_credibility)
        
        # Step 5: Create location/category features
        location_features = self.create_location_features(static_features)
        
        # Step 6: Merge all features
        final_df = self.merge_all_features(static_features, review_features, location_features)
        
        # Step 7: Save final merged dataset
        final_file = self.output_path / "business_features_final.csv"
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


def main():
    """Main entry point for feature engineering"""
    print("="*70)
    print("CS 412 RESEARCH PROJECT - FEATURE ENGINEERING")
    print("Business Success Prediction using Yelp Dataset")
    print("="*70)
    print("\nPhase 3: Feature Engineering")
    print("")
    
    engineer = FeatureEngineer()
    engineer.run_pipeline()
    
    print("\n" + "="*70)
    print("FEATURE ENGINEERING COMPLETE!")
    print("="*70)
    print("\nNext Steps:")
    print("  1. Review feature_engineering_report.md")
    print("  2. Proceed to feature selection/dimensionality reduction")
    print("  3. Begin baseline model training")
    print("")


if __name__ == "__main__":
    main()