"""
Data processing pipeline for Yelp dataset
"""

import pandas as pd
import numpy as np
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta

from ..utils.config import config
from ..utils.utils import load_json_data, save_json_data, convert_to_dataframe


class YelpDataProcessor:
    """Main class for processing Yelp dataset"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.raw_path = config.get_data_path("raw")
        self.processed_path = config.get_data_path("processed")
        
        # Create directories if they don't exist
        Path(self.processed_path).mkdir(parents=True, exist_ok=True)
        
    def load_business_data(self) -> pd.DataFrame:
        """Load and process business data"""
        self.logger.info("Loading business data...")
        
        file_path = Path(self.raw_path) / config.get("yelp_files.business")
        data = load_json_data(str(file_path))
        df = convert_to_dataframe(data)
        
        # Convert relevant columns
        df['is_open'] = df['is_open'].astype(int)
        df['stars'] = pd.to_numeric(df['stars'], errors='coerce')
        df['review_count'] = pd.to_numeric(df['review_count'], errors='coerce')
        
        # Parse attributes
        df['attributes'] = df['attributes'].apply(
            lambda x: json.loads(x) if isinstance(x, str) else x
        )
        
        self.logger.info(f"Loaded {len(df)} businesses")
        return df
    
    def load_review_data(self, sample_size: Optional[int] = None) -> pd.DataFrame:
        """Load and process review data"""
        self.logger.info("Loading review data...")
        
        file_path = Path(self.raw_path) / config.get("yelp_files.review")
        
        if sample_size:
            # Load sample for testing
            data = []
            with open(file_path, 'r', encoding='utf-8') as file:
                for i, line in enumerate(file):
                    if i >= sample_size:
                        break
                    data.append(json.loads(line.strip()))
        else:
            data = load_json_data(str(file_path))
        
        df = convert_to_dataframe(data)
        
        # Convert date column
        df['date'] = pd.to_datetime(df['date'])
        df['stars'] = pd.to_numeric(df['stars'], errors='coerce')
        
        self.logger.info(f"Loaded {len(df)} reviews")
        return df
    
    def load_user_data(self) -> pd.DataFrame:
        """Load and process user data"""
        self.logger.info("Loading user data...")
        
        file_path = Path(self.raw_path) / config.get("yelp_files.user")
        data = load_json_data(str(file_path))
        df = convert_to_dataframe(data)
        
        # Convert numeric columns
        numeric_cols = ['review_count', 'useful', 'funny', 'cool', 'fans']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Convert date columns
        df['yelping_since'] = pd.to_datetime(df['yelping_since'])
        
        self.logger.info(f"Loaded {len(df)} users")
        return df
    
    def load_checkin_data(self) -> pd.DataFrame:
        """Load and process check-in data"""
        self.logger.info("Loading check-in data...")
        
        file_path = Path(self.raw_path) / config.get("yelp_files.checkin")
        data = load_json_data(str(file_path))
        df = convert_to_dataframe(data)
        
        # Parse date column
        df['date'] = df['date'].apply(lambda x: x.split(',') if isinstance(x, str) else [])
        
        self.logger.info(f"Loaded {len(df)} check-ins")
        return df
    
    def load_tip_data(self) -> pd.DataFrame:
        """Load and process tip data"""
        self.logger.info("Loading tip data...")
        
        file_path = Path(self.raw_path) / config.get("yelp_files.tip")
        data = load_json_data(str(file_path))
        df = convert_to_dataframe(data)
        
        # Convert date column
        df['date'] = pd.to_datetime(df['date'])
        
        self.logger.info(f"Loaded {len(df)} tips")
        return df
    
    def create_business_features(self, business_df: pd.DataFrame) -> pd.DataFrame:
        """Create business-level features"""
        df = business_df.copy()
        
        # Extract categories
        df['categories_list'] = df['categories'].str.split(', ')
        
        # Extract attributes
        if 'attributes' in df.columns:
            # Common attributes to extract
            attr_cols = ['BusinessParking', 'WiFi', 'RestaurantsPriceRange2', 
                        'RestaurantsDelivery', 'RestaurantsTakeOut']
            
            for attr in attr_cols:
                df[f'attr_{attr}'] = df['attributes'].apply(
                    lambda x: x.get(attr) if isinstance(x, dict) else None
                )
        
        # Location features
        df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
        df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
        
        return df
    
    def create_temporal_features(self, review_df: pd.DataFrame) -> pd.DataFrame:
        """Create temporal features from review data"""
        df = review_df.copy()
        
        # Sort by business and date
        df = df.sort_values(['business_id', 'date'])
        
        # Calculate rating trends
        df['rating_change'] = df.groupby('business_id')['stars'].diff()
        df['rating_velocity'] = df.groupby('business_id')['rating_change'].rolling(
            window=30, min_periods=1
        ).mean().reset_index(0, drop=True)
        
        # Review frequency features
        df['days_since_last_review'] = df.groupby('business_id')['date'].diff().dt.days
        
        # Rolling averages
        windows = config.get_feature_params('temporal')['windows']
        for window in windows:
            df[f'avg_rating_{window}d'] = df.groupby('business_id')['stars'].rolling(
                window=window, min_periods=1
            ).mean().reset_index(0, drop=True)
            
            df[f'review_count_{window}d'] = df.groupby('business_id')['stars'].rolling(
                window=window, min_periods=1
            ).count().reset_index(0, drop=True)
        
        return df
    
    def create_user_weighted_features(self, review_df: pd.DataFrame, 
                                    user_df: pd.DataFrame) -> pd.DataFrame:
        """Create user-weighted features"""
        df = review_df.copy()
        
        # Merge with user data
        df = df.merge(user_df[['user_id', 'review_count', 'useful', 'funny', 'cool', 
                              'yelping_since']], on='user_id', how='left')
        
        # Calculate user weights
        min_reviews = config.get_feature_params('user_weighting')['min_reviews']
        usefulness_threshold = config.get_feature_params('user_weighting')['usefulness_threshold']
        
        # User credibility score
        df['user_credibility'] = (
            np.log1p(df['review_count']) * 
            (df['useful'] / (df['review_count'] + 1)) * 
            (df['yelping_since'].dt.year - 2004)  # Years on platform
        )
        
        # Weighted ratings
        df['weighted_stars'] = df['stars'] * df['user_credibility']
        
        # Calculate weighted averages by business
        business_weights = df.groupby('business_id').agg({
            'weighted_stars': 'sum',
            'user_credibility': 'sum',
            'stars': ['mean', 'std', 'count']
        }).reset_index()
        
        business_weights.columns = ['business_id', 'weighted_stars_sum', 
                                  'user_credibility_sum', 'avg_stars', 
                                  'stars_std', 'review_count']
        
        business_weights['weighted_avg_stars'] = (
            business_weights['weighted_stars_sum'] / 
            business_weights['user_credibility_sum']
        )
        
        return business_weights
    
    def process_all_data(self, sample_size: Optional[int] = None) -> Dict[str, pd.DataFrame]:
        """Process all Yelp data files"""
        self.logger.info("Starting data processing pipeline...")
        
        # Load all data
        business_df = self.load_business_data()
        review_df = self.load_review_data(sample_size)
        user_df = self.load_user_data()
        checkin_df = self.load_checkin_data()
        tip_df = self.load_tip_data()
        
        # Create features
        business_features = self.create_business_features(business_df)
        temporal_features = self.create_temporal_features(review_df)
        user_weighted_features = self.create_user_weighted_features(review_df, user_df)
        
        # Save processed data
        processed_data = {
            'business': business_features,
            'reviews': temporal_features,
            'users': user_df,
            'checkins': checkin_df,
            'tips': tip_df,
            'user_weighted': user_weighted_features
        }
        
        # Save to processed directory
        for name, df in processed_data.items():
            output_path = Path(self.processed_path) / f"{name}_processed.csv"
            df.to_csv(output_path, index=False)
            self.logger.info(f"Saved {name} data to {output_path}")
        
        self.logger.info("Data processing completed!")
        return processed_data
    
    def create_merged_dataset(self) -> pd.DataFrame:
        """Create merged dataset for modeling"""
        self.logger.info("Creating merged dataset...")
        
        # Load processed data
        business_df = pd.read_csv(Path(self.processed_path) / "business_processed.csv")
        user_weighted_df = pd.read_csv(Path(self.processed_path) / "user_weighted_processed.csv")
        
        # Merge datasets
        merged_df = business_df.merge(user_weighted_df, on='business_id', how='left')
        
        # Fill missing values
        merged_df = merged_df.fillna(0)
        
        # Save merged dataset
        output_path = Path(self.processed_path) / "merged_dataset.csv"
        merged_df.to_csv(output_path, index=False)
        
        self.logger.info(f"Merged dataset saved to {output_path}")
        return merged_df
