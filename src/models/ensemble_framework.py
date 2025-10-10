"""
User-weighted Ensemble Framework for CS 412 Research Project
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Any, Optional, Tuple
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import joblib
from pathlib import Path

from ..utils.config import config
from ..utils.utils import calculate_class_weights


class UserWeightedEnsemble:
    """User-weighted ensemble framework with multi-level classification"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.category_models = {}
        self.ensemble_model = None
        self.scalers = {}
        self.category_encoders = {}
        self.results = {}
        
        # Initialize base models
        self._initialize_base_models()
    
    def _initialize_base_models(self):
        """Initialize base models for ensemble"""
        # Get model parameters from config
        rf_params = config.get_model_params('random_forest')
        xgb_params = config.get_model_params('xgboost')
        nn_params = config.get_model_params('neural_network')
        
        # Create base models
        self.base_models = {
            'random_forest': RandomForestClassifier(**rf_params),
            'xgboost': xgb.XGBClassifier(**xgb_params),
            'neural_network': MLPClassifier(**nn_params)
        }
        
        self.logger.info("Initialized base models for ensemble")
    
    def create_user_weights(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create user-weighted features"""
        df = df.copy()
        
        # User credibility calculation
        min_reviews = config.get_feature_params('user_weighting')['min_reviews']
        usefulness_threshold = config.get_feature_params('user_weighting')['usefulness_threshold']
        
        # Calculate user weights
        df['user_weight'] = (
            np.log1p(df['review_count']) * 
            (df['useful'] / (df['review_count'] + 1)) * 
            (df['yelping_since'].dt.year - 2004)  # Years on platform
        )
        
        # Filter users by minimum criteria
        df['user_qualified'] = (
            (df['review_count'] >= min_reviews) &
            (df['useful'] / (df['review_count'] + 1) >= usefulness_threshold)
        )
        
        # Weighted ratings
        df['weighted_stars'] = df['stars'] * df['user_weight']
        
        return df
    
    def aggregate_user_weighted_features(self, df: pd.DataFrame, 
                                      group_col: str = 'business_id') -> pd.DataFrame:
        """Aggregate user-weighted features by business"""
        # Group by business and calculate weighted aggregates
        aggregated = df.groupby(group_col).agg({
            'weighted_stars': ['sum', 'mean', 'std'],
            'user_weight': ['sum', 'mean', 'std'],
            'stars': ['mean', 'std', 'count'],
            'sentiment_compound': ['mean', 'std'],
            'text_length': ['mean', 'std'],
            'user_qualified': 'sum'
        }).reset_index()
        
        # Flatten column names
        aggregated.columns = [
            f"{col[0]}_{col[1]}" if col[1] else col[0] 
            for col in aggregated.columns
        ]
        
        # Calculate weighted average rating
        aggregated['weighted_avg_rating'] = (
            aggregated['weighted_stars_sum'] / aggregated['user_weight_sum']
        )
        
        # Calculate user engagement ratio
        aggregated['qualified_user_ratio'] = (
            aggregated['user_qualified_sum'] / aggregated['stars_count']
        )
        
        return aggregated
    
    def create_category_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create business category features for multi-level classification"""
        df = df.copy()
        
        # Extract main categories
        df['main_category'] = df['categories'].str.split(',').str[0].str.strip()
        
        # Get top categories
        top_categories = df['main_category'].value_counts().head(10).index.tolist()
        
        # Create category groups
        category_groups = {
            'Restaurants': ['Restaurants', 'Food', 'Fast Food', 'Coffee & Tea', 'Bakeries'],
            'Shopping': ['Shopping', 'Fashion', 'Home & Garden', 'Electronics'],
            'Services': ['Beauty & Spas', 'Health & Medical', 'Automotive', 'Professional Services'],
            'Entertainment': ['Nightlife', 'Arts & Entertainment', 'Sports & Recreation'],
            'Other': [cat for cat in top_categories if cat not in 
                     ['Restaurants', 'Shopping', 'Services', 'Entertainment']]
        }
        
        # Map categories to groups
        df['category_group'] = df['main_category'].map(
            {cat: group for group, cats in category_groups.items() 
             for cat in cats}
        ).fillna('Other')
        
        return df
    
    def train_category_models(self, X: pd.DataFrame, y: pd.Series, 
                            categories: pd.Series) -> Dict[str, Any]:
        """Train category-specific models"""
        self.logger.info("Training category-specific models...")
        
        category_results = {}
        unique_categories = categories.unique()
        
        for category in unique_categories:
            if pd.isna(category):
                continue
                
            self.logger.info(f"Training model for category: {category}")
            
            # Filter data for this category
            mask = categories == category
            X_cat = X[mask]
            y_cat = y[mask]
            
            if len(X_cat) < 10:  # Skip categories with too few samples
                self.logger.warning(f"Skipping category {category} - too few samples")
                continue
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_cat, y_cat, test_size=0.2, random_state=42, stratify=y_cat
            )
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train ensemble for this category
            category_ensemble = VotingClassifier(
                estimators=[
                    ('rf', RandomForestClassifier(n_estimators=50, random_state=42)),
                    ('xgb', xgb.XGBClassifier(n_estimators=50, random_state=42)),
                    ('nn', MLPClassifier(hidden_layer_sizes=(50,), random_state=42))
                ],
                voting='soft'
            )
            
            category_ensemble.fit(X_train_scaled, y_train)
            
            # Evaluate
            y_pred_proba = category_ensemble.predict_proba(X_test_scaled)[:, 1]
            roc_auc = roc_auc_score(y_test, y_pred_proba)
            
            # Store results
            category_results[category] = {
                'model': category_ensemble,
                'scaler': scaler,
                'roc_auc': roc_auc,
                'n_samples': len(X_cat)
            }
            
            self.logger.info(f"Category {category}: ROC-AUC = {roc_auc:.4f}, Samples = {len(X_cat)}")
        
        self.category_models = category_results
        return category_results
    
    def train_ensemble_model(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Train the main ensemble model"""
        self.logger.info("Training main ensemble model...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Create ensemble
        self.ensemble_model = VotingClassifier(
            estimators=[
                ('rf', self.base_models['random_forest']),
                ('xgb', self.base_models['xgboost']),
                ('nn', self.base_models['neural_network'])
            ],
            voting='soft'
        )
        
        # Train ensemble
        self.ensemble_model.fit(X_train_scaled, y_train)
        self.scalers['ensemble'] = scaler
        
        # Evaluate
        y_pred_proba = self.ensemble_model.predict_proba(X_test_scaled)[:, 1]
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        result = {
            'roc_auc': roc_auc,
            'n_samples': len(X_train)
        }
        
        self.logger.info(f"Ensemble model trained. ROC-AUC = {roc_auc:.4f}")
        return result
    
    def predict_with_category_awareness(self, X: pd.DataFrame, 
                                     categories: pd.Series) -> np.ndarray:
        """Make predictions using category-aware models"""
        predictions = np.zeros(len(X))
        prediction_weights = np.zeros(len(X))
        
        for category, model_info in self.category_models.items():
            mask = categories == category
            
            if mask.sum() == 0:
                continue
            
            # Scale features
            X_cat = X[mask]
            X_cat_scaled = model_info['scaler'].transform(X_cat)
            
            # Make predictions
            cat_predictions = model_info['model'].predict_proba(X_cat_scaled)[:, 1]
            
            # Weight by model performance
            weight = model_info['roc_auc']
            predictions[mask] += cat_predictions * weight
            prediction_weights[mask] += weight
        
        # Normalize by weights
        predictions = np.divide(predictions, prediction_weights, 
                              out=np.zeros_like(predictions), 
                              where=prediction_weights!=0)
        
        return predictions
    
    def evaluate_ensemble(self, X: pd.DataFrame, y: pd.Series, 
                         categories: pd.Series) -> Dict[str, Any]:
        """Evaluate the ensemble framework"""
        self.logger.info("Evaluating ensemble framework...")
        
        # Main ensemble predictions
        X_scaled = self.scalers['ensemble'].transform(X)
        ensemble_pred = self.ensemble_model.predict_proba(X_scaled)[:, 1]
        ensemble_auc = roc_auc_score(y, ensemble_pred)
        
        # Category-aware predictions
        category_pred = self.predict_with_category_awareness(X, categories)
        category_auc = roc_auc_score(y, category_pred)
        
        # Combined predictions (weighted average)
        combined_pred = 0.7 * ensemble_pred + 0.3 * category_pred
        combined_auc = roc_auc_score(y, combined_pred)
        
        results = {
            'ensemble_auc': ensemble_auc,
            'category_aware_auc': category_auc,
            'combined_auc': combined_auc,
            'ensemble_predictions': ensemble_pred,
            'category_predictions': category_pred,
            'combined_predictions': combined_pred
        }
        
        self.logger.info(f"Evaluation completed:")
        self.logger.info(f"  Ensemble AUC: {ensemble_auc:.4f}")
        self.logger.info(f"  Category-aware AUC: {category_auc:.4f}")
        self.logger.info(f"  Combined AUC: {combined_auc:.4f}")
        
        return results
    
    def save_ensemble(self, output_dir: str = "results/models"):
        """Save ensemble models"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save main ensemble
        if self.ensemble_model:
            ensemble_path = output_path / "ensemble_model.joblib"
            joblib.dump(self.ensemble_model, ensemble_path)
            self.logger.info(f"Saved ensemble model to {ensemble_path}")
        
        # Save category models
        for category, model_info in self.category_models.items():
            model_path = output_path / f"category_{category}_model.joblib"
            joblib.dump(model_info['model'], model_path)
            
            scaler_path = output_path / f"category_{category}_scaler.joblib"
            joblib.dump(model_info['scaler'], scaler_path)
            
            self.logger.info(f"Saved category {category} model and scaler")
        
        # Save ensemble scaler
        if 'ensemble' in self.scalers:
            scaler_path = output_path / "ensemble_scaler.joblib"
            joblib.dump(self.scalers['ensemble'], scaler_path)
            self.logger.info(f"Saved ensemble scaler to {scaler_path}")
    
    def load_ensemble(self, model_dir: str = "results/models"):
        """Load ensemble models"""
        model_path = Path(model_dir)
        
        # Load main ensemble
        ensemble_file = model_path / "ensemble_model.joblib"
        if ensemble_file.exists():
            self.ensemble_model = joblib.load(ensemble_file)
            self.logger.info(f"Loaded ensemble model from {ensemble_file}")
        
        # Load ensemble scaler
        scaler_file = model_path / "ensemble_scaler.joblib"
        if scaler_file.exists():
            self.scalers['ensemble'] = joblib.load(scaler_file)
            self.logger.info(f"Loaded ensemble scaler from {scaler_file}")
        
        # Load category models (simplified - would need to know categories)
        # This is a placeholder for the full implementation
        self.logger.info("Category models loading not implemented in this version")
