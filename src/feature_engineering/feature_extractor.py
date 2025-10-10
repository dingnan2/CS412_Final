"""
Feature engineering modules for CS 412 Research Project
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Any, Optional, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
import re

from ..utils.config import config


class SentimentAnalyzer:
    """Sentiment analysis for review text"""
    
    def __init__(self, method: str = "textblob"):
        self.method = method
        self.logger = logging.getLogger(__name__)
        
        if method == "textblob":
            from textblob import TextBlob
            self.analyzer = TextBlob
        elif method == "vader":
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            self.analyzer = SentimentIntensityAnalyzer()
        else:
            raise ValueError(f"Unsupported sentiment method: {method}")
    
    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        """Analyze sentiment of text"""
        if pd.isna(text) or text == "":
            return {"polarity": 0.0, "subjectivity": 0.0, "compound": 0.0}
        
        try:
            if self.method == "textblob":
                blob = self.analyzer(text)
                return {
                    "polarity": blob.sentiment.polarity,
                    "subjectivity": blob.sentiment.subjectivity,
                    "compound": blob.sentiment.polarity  # For compatibility
                }
            elif self.method == "vader":
                scores = self.analyzer.polarity_scores(text)
                return {
                    "polarity": scores["compound"],
                    "subjectivity": 0.0,  # VADER doesn't provide subjectivity
                    "compound": scores["compound"]
                }
        except Exception as e:
            self.logger.warning(f"Error analyzing sentiment: {e}")
            return {"polarity": 0.0, "subjectivity": 0.0, "compound": 0.0}
    
    def extract_sentiment_features(self, df: pd.DataFrame, 
                                 text_col: str = "text") -> pd.DataFrame:
        """Extract sentiment features from text column"""
        df = df.copy()
        
        # Analyze sentiment for each text
        sentiment_scores = df[text_col].apply(self.analyze_sentiment)
        
        # Extract individual scores
        df['sentiment_polarity'] = sentiment_scores.apply(lambda x: x['polarity'])
        df['sentiment_subjectivity'] = sentiment_scores.apply(lambda x: x['subjectivity'])
        df['sentiment_compound'] = sentiment_scores.apply(lambda x: x['compound'])
        
        # Categorize sentiment
        df['sentiment_category'] = pd.cut(
            df['sentiment_compound'],
            bins=[-1, -0.1, 0.1, 1],
            labels=['negative', 'neutral', 'positive']
        )
        
        return df


class TemporalFeatureExtractor:
    """Extract temporal features from time-series data"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def extract_temporal_features(self, df: pd.DataFrame, 
                                date_col: str = "date",
                                group_col: str = "business_id") -> pd.DataFrame:
        """Extract temporal features"""
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col])
        
        # Sort by group and date
        df = df.sort_values([group_col, date_col])
        
        # Time-based features
        df['year'] = df[date_col].dt.year
        df['month'] = df[date_col].dt.month
        df['day_of_week'] = df[date_col].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        # Rolling features
        windows = config.get_feature_params('temporal')['windows']
        
        for window in windows:
            # Rolling averages
            df[f'avg_rating_{window}d'] = df.groupby(group_col)['stars'].rolling(
                window=window, min_periods=1
            ).mean().reset_index(0, drop=True)
            
            df[f'review_count_{window}d'] = df.groupby(group_col)['stars'].rolling(
                window=window, min_periods=1
            ).count().reset_index(0, drop=True)
            
            # Rolling sentiment
            if 'sentiment_compound' in df.columns:
                df[f'avg_sentiment_{window}d'] = df.groupby(group_col)['sentiment_compound'].rolling(
                    window=window, min_periods=1
                ).mean().reset_index(0, drop=True)
        
        # Trend features
        df['rating_trend'] = df.groupby(group_col)['stars'].diff()
        df['rating_velocity'] = df.groupby(group_col)['rating_trend'].rolling(
            window=30, min_periods=1
        ).mean().reset_index(0, drop=True)
        
        # Time gaps
        df['days_since_last_review'] = df.groupby(group_col)[date_col].diff().dt.days
        
        return df


class BusinessAttributeExtractor:
    """Extract features from business attributes"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def extract_category_features(self, df: pd.DataFrame, 
                                category_col: str = "categories") -> pd.DataFrame:
        """Extract features from business categories"""
        df = df.copy()
        
        # Split categories
        df['categories_list'] = df[category_col].str.split(', ')
        
        # Most common categories
        all_categories = []
        for categories in df['categories_list'].dropna():
            all_categories.extend(categories)
        
        category_counts = pd.Series(all_categories).value_counts()
        top_categories = category_counts.head(20).index.tolist()
        
        # Create binary features for top categories
        for category in top_categories:
            df[f'category_{category}'] = df['categories_list'].apply(
                lambda x: 1 if category in x else 0
            )
        
        # Category diversity
        df['category_count'] = df['categories_list'].apply(
            lambda x: len(x) if isinstance(x, list) else 0
        )
        
        return df
    
    def extract_location_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract location-based features"""
        df = df.copy()
        
        # Convert coordinates
        df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
        df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
        
        # Distance from city center (assuming approximate center)
        if 'latitude' in df.columns and 'longitude' in df.columns:
            # This is a simplified distance calculation
            df['distance_from_center'] = np.sqrt(
                (df['latitude'] - df['latitude'].mean())**2 + 
                (df['longitude'] - df['longitude'].mean())**2
            )
        
        # State and city features
        if 'state' in df.columns:
            df['state_encoded'] = LabelEncoder().fit_transform(df['state'])
        
        if 'city' in df.columns:
            df['city_encoded'] = LabelEncoder().fit_transform(df['city'])
        
        return df
    
    def extract_attribute_features(self, df: pd.DataFrame, 
                                 attr_col: str = "attributes") -> pd.DataFrame:
        """Extract features from business attributes"""
        df = df.copy()
        
        if attr_col not in df.columns:
            return df
        
        # Common attributes to extract
        attr_mapping = {
            'BusinessParking': 'parking',
            'WiFi': 'wifi',
            'RestaurantsPriceRange2': 'price_range',
            'RestaurantsDelivery': 'delivery',
            'RestaurantsTakeOut': 'takeout',
            'RestaurantsReservations': 'reservations',
            'RestaurantsAttire': 'attire',
            'RestaurantsGoodForGroups': 'good_for_groups',
            'RestaurantsTableService': 'table_service',
            'OutdoorSeating': 'outdoor_seating',
            'Alcohol': 'alcohol',
            'NoiseLevel': 'noise_level',
            'Ambience': 'ambience',
            'HasTV': 'has_tv',
            'Caters': 'caters',
            'WheelchairAccessible': 'wheelchair_accessible'
        }
        
        for attr_key, feature_name in attr_mapping.items():
            df[f'attr_{feature_name}'] = df[attr_col].apply(
                lambda x: self._extract_attribute_value(x, attr_key)
            )
        
        return df
    
    def _extract_attribute_value(self, attributes: Any, attr_key: str) -> Any:
        """Extract specific attribute value"""
        if not isinstance(attributes, dict):
            return None
        
        return attributes.get(attr_key)


class TextFeatureExtractor:
    """Extract features from text data"""
    
    def __init__(self, max_features: int = 1000):
        self.max_features = max_features
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words='english',
            ngram_range=(1, 2)
        )
        self.logger = logging.getLogger(__name__)
    
    def extract_text_features(self, df: pd.DataFrame, 
                            text_col: str = "text") -> pd.DataFrame:
        """Extract TF-IDF features from text"""
        df = df.copy()
        
        # Clean text
        df[f'{text_col}_clean'] = df[text_col].apply(self._clean_text)
        
        # TF-IDF features
        try:
            tfidf_matrix = self.vectorizer.fit_transform(df[f'{text_col}_clean'])
            tfidf_df = pd.DataFrame(
                tfidf_matrix.toarray(),
                columns=[f'tfidf_{i}' for i in range(tfidf_matrix.shape[1])]
            )
            
            # Concatenate with original dataframe
            df = pd.concat([df.reset_index(drop=True), tfidf_df], axis=1)
            
        except Exception as e:
            self.logger.warning(f"Error extracting TF-IDF features: {e}")
        
        # Text length features
        df['text_length'] = df[text_col].str.len()
        df['word_count'] = df[text_col].str.split().str.len()
        df['avg_word_length'] = df['text_length'] / (df['word_count'] + 1)
        
        return df
    
    def _clean_text(self, text: str) -> str:
        """Clean text for processing"""
        if pd.isna(text):
            return ""
        
        # Convert to lowercase
        text = str(text).lower()
        
        # Remove special characters
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text


class FeatureEngineer:
    """Main feature engineering class"""
    
    def __init__(self):
        self.sentiment_analyzer = SentimentAnalyzer()
        self.temporal_extractor = TemporalFeatureExtractor()
        self.business_extractor = BusinessAttributeExtractor()
        self.text_extractor = TextFeatureExtractor()
        self.logger = logging.getLogger(__name__)
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all feature engineering steps"""
        self.logger.info("Starting feature engineering...")
        
        # Apply different feature extractors
        if 'text' in df.columns:
            df = self.text_extractor.extract_text_features(df)
            df = self.sentiment_analyzer.extract_sentiment_features(df)
        
        if 'date' in df.columns:
            df = self.temporal_extractor.extract_temporal_features(df)
        
        if 'categories' in df.columns:
            df = self.business_extractor.extract_category_features(df)
        
        df = self.business_extractor.extract_location_features(df)
        
        if 'attributes' in df.columns:
            df = self.business_extractor.extract_attribute_features(df)
        
        self.logger.info(f"Feature engineering completed. Shape: {df.shape}")
        return df
    
    def select_features(self, df: pd.DataFrame, 
                      target_col: str = "is_open") -> Tuple[pd.DataFrame, List[str]]:
        """Select relevant features for modeling"""
        # Remove non-feature columns
        exclude_cols = [
            'business_id', 'user_id', 'review_id', 'name', 'address', 
            'city', 'state', 'postal_code', 'categories', 'attributes',
            'text', 'date', 'yelping_since', target_col
        ]
        
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # Remove columns with too many missing values
        missing_threshold = 0.5
        feature_cols = [col for col in feature_cols 
                       if df[col].isnull().sum() / len(df) < missing_threshold]
        
        # Select features
        X = df[feature_cols].fillna(0)
        
        self.logger.info(f"Selected {len(feature_cols)} features")
        return X, feature_cols
