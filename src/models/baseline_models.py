"""
Baseline models for CS 412 Research Project
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Any, Optional, Tuple
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path

from ..utils.config import config
from ..utils.utils import calculate_class_weights


class BaselineModels:
    """Implementation of baseline models for business success prediction"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.models = {}
        self.scalers = {}
        self.results = {}
        
        # Initialize models with configuration parameters
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize baseline models with configured parameters"""
        # Logistic Regression
        lr_params = config.get_model_params('logistic_regression')
        self.models['logistic_regression'] = LogisticRegression(**lr_params)
        
        # Decision Tree
        dt_params = config.get_model_params('decision_tree')
        self.models['decision_tree'] = DecisionTreeClassifier(**dt_params)
        
        # Random Forest
        rf_params = config.get_model_params('random_forest')
        self.models['random_forest'] = RandomForestClassifier(**rf_params)
        
        self.logger.info("Initialized baseline models")
    
    def prepare_data(self, X: pd.DataFrame, y: pd.Series, 
                    test_size: float = 0.2) -> Tuple:
        """Prepare data for training and testing"""
        # Get evaluation parameters
        eval_params = config.get_evaluation_params()
        test_size = eval_params.get('test_size', test_size)
        random_state = eval_params.get('random_state', 42)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, 
            stratify=y
        )
        
        self.logger.info(f"Data split: Train {X_train.shape}, Test {X_test.shape}")
        return X_train, X_test, y_train, y_test
    
    def train_model(self, model_name: str, X_train: pd.DataFrame, 
                   y_train: pd.Series, scale_features: bool = True) -> Dict[str, Any]:
        """Train a specific model"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        self.logger.info(f"Training {model_name}...")
        
        # Scale features if needed
        if scale_features and model_name in ['logistic_regression']:
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            self.scalers[model_name] = scaler
        else:
            X_train_scaled = X_train
        
        # Handle class imbalance
        if model_name in ['logistic_regression', 'decision_tree', 'random_forest']:
            class_weights = calculate_class_weights(y_train.values)
            if model_name == 'logistic_regression':
                self.models[model_name].class_weight = class_weights
            else:
                self.models[model_name].class_weight = 'balanced'
        
        # Train model
        self.models[model_name].fit(X_train_scaled, y_train)
        
        # Cross-validation
        cv_scores = self._cross_validate(model_name, X_train_scaled, y_train)
        
        result = {
            'model_name': model_name,
            'cv_scores': cv_scores,
            'cv_mean': np.mean(cv_scores),
            'cv_std': np.std(cv_scores)
        }
        
        self.logger.info(f"{model_name} trained. CV Score: {result['cv_mean']:.4f} ± {result['cv_std']:.4f}")
        return result
    
    def _cross_validate(self, model_name: str, X: np.ndarray, 
                       y: pd.Series) -> List[float]:
        """Perform cross-validation"""
        eval_params = config.get_evaluation_params()
        cv_folds = eval_params.get('cv_folds', 5)
        random_state = eval_params.get('random_state', 42)
        
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        scores = cross_val_score(
            self.models[model_name], X, y, 
            cv=cv, scoring='roc_auc'
        )
        
        return scores.tolist()
    
    def evaluate_model(self, model_name: str, X_test: pd.DataFrame, 
                      y_test: pd.Series) -> Dict[str, Any]:
        """Evaluate a trained model"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        # Scale test data if needed
        if model_name in self.scalers:
            X_test_scaled = self.scalers[model_name].transform(X_test)
        else:
            X_test_scaled = X_test
        
        # Make predictions
        y_pred = self.models[model_name].predict(X_test_scaled)
        y_pred_proba = self.models[model_name].predict_proba(X_test_scaled)[:, 1]
        
        # Calculate metrics
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        # Classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        result = {
            'model_name': model_name,
            'roc_auc': roc_auc,
            'accuracy': report['accuracy'],
            'precision': report['macro avg']['precision'],
            'recall': report['macro avg']['recall'],
            'f1_score': report['macro avg']['f1-score'],
            'confusion_matrix': cm.tolist(),
            'classification_report': report
        }
        
        self.logger.info(f"{model_name} evaluation completed. ROC-AUC: {roc_auc:.4f}")
        return result
    
    def train_all_models(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Train all baseline models"""
        self.logger.info("Training all baseline models...")
        
        # Prepare data
        X_train, X_test, y_train, y_test = self.prepare_data(X, y)
        
        # Train and evaluate each model
        results = {}
        for model_name in self.models.keys():
            # Train model
            train_result = self.train_model(model_name, X_train, y_train)
            
            # Evaluate model
            eval_result = self.evaluate_model(model_name, X_test, y_test)
            
            # Combine results
            results[model_name] = {
                'training': train_result,
                'evaluation': eval_result
            }
        
        self.results = results
        self.logger.info("All baseline models trained and evaluated")
        return results
    
    def save_models(self, output_dir: str = "results/models"):
        """Save trained models and scalers"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save models
        for model_name, model in self.models.items():
            model_path = output_path / f"{model_name}.joblib"
            joblib.dump(model, model_path)
            self.logger.info(f"Saved {model_name} to {model_path}")
        
        # Save scalers
        for scaler_name, scaler in self.scalers.items():
            scaler_path = output_path / f"{scaler_name}_scaler.joblib"
            joblib.dump(scaler, scaler_path)
            self.logger.info(f"Saved {scaler_name} scaler to {scaler_path}")
    
    def load_models(self, model_dir: str = "results/models"):
        """Load trained models and scalers"""
        model_path = Path(model_dir)
        
        # Load models
        for model_name in self.models.keys():
            model_file = model_path / f"{model_name}.joblib"
            if model_file.exists():
                self.models[model_name] = joblib.load(model_file)
                self.logger.info(f"Loaded {model_name} from {model_file}")
        
        # Load scalers
        for scaler_name in ['logistic_regression']:  # Only LR uses scaler
            scaler_file = model_path / f"{scaler_name}_scaler.joblib"
            if scaler_file.exists():
                self.scalers[scaler_name] = joblib.load(scaler_file)
                self.logger.info(f"Loaded {scaler_name} scaler from {scaler_file}")
    
    def get_feature_importance(self, model_name: str, 
                              feature_names: List[str]) -> pd.DataFrame:
        """Get feature importance for tree-based models"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        if model_name not in ['decision_tree', 'random_forest']:
            self.logger.warning(f"Feature importance not available for {model_name}")
            return pd.DataFrame()
        
        # Get feature importance
        importance = self.models[model_name].feature_importances_
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def compare_models(self) -> pd.DataFrame:
        """Compare performance of all models"""
        if not self.results:
            self.logger.warning("No results available. Train models first.")
            return pd.DataFrame()
        
        comparison_data = []
        for model_name, result in self.results.items():
            eval_result = result['evaluation']
            comparison_data.append({
                'Model': model_name,
                'ROC-AUC': eval_result['roc_auc'],
                'Accuracy': eval_result['accuracy'],
                'Precision': eval_result['precision'],
                'Recall': eval_result['recall'],
                'F1-Score': eval_result['f1_score']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('ROC-AUC', ascending=False)
        
        return comparison_df
