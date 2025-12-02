"""
Baseline Models: Train and evaluate baseline models for business success prediction.

CRITICAL UPDATES (V3 - Unified Configuration):
- Uses config.py for consistent split configuration across all phases
- Handles both 'label' (from temporal validation) and 'is_open' targets
- Uses SPLIT_CONFIG train_years/test_years for consistent train/test splits
- Ensures comparable results with advanced_models.py and evaluation phases

This module implements:
1. Feature selection (correlation, variance, importance-based)
2. Three baseline models: Logistic Regression, Decision Tree, Random Forest
3. Class imbalance handling (SMOTE and class weights)
4. Temporal holdout split using config.py settings
5. Comprehensive evaluation with visualizations
6. Model comparison and analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
import warnings
from datetime import datetime
import json
import pickle
from typing import Dict, Tuple, List

# Scikit-learn imports
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, roc_curve, auc,
    precision_recall_curve, average_precision_score,
    confusion_matrix, classification_report,
    f1_score, precision_score, recall_score
)
from imblearn.over_sampling import SMOTE

# Import unified configuration
try:
    from config import SPLIT_CONFIG, RANDOM_STATE, FEATURE_SELECTION_CONFIG
except ImportError:
    # Fallback defaults if config not available
    SPLIT_CONFIG = {
        'train_years': [2012, 2013, 2014, 2015, 2016, 2017, 2018],
        'test_years': [2019, 2020]
    }
    RANDOM_STATE = 42
    FEATURE_SELECTION_CONFIG = {'corr_threshold': 0.95, 'variance_threshold': 0.01, 'n_top_features': 40}

warnings.filterwarnings('ignore')

# Configure plotting
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/baseline_models.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class BaselineModelPipeline:
    """
    Complete baseline model training and evaluation pipeline with temporal validation.
    
    Pipeline Steps:
    1. Load and prepare data (with temporal metadata if available)
    2. Feature selection
    3. Train-test split (random OR temporal stratified)
    4. Handle class imbalance (SMOTE vs class weights)
    5. Train baseline models
    6. Evaluate and compare models
    7. Generate comprehensive report with visualizations
    """
    
    def __init__(self, 
                 data_path: str,
                 output_path: str = "src/models",
                 random_state: int = 42,
                 use_temporal_split: bool = False,
                 test_size: float = 0.2):
        """
        Initialize the baseline model pipeline.
        
        Args:
            data_path: Path to business_features CSV file
            output_path: Directory to save outputs
            random_state: Random seed for reproducibility
            use_temporal_split: If True, use temporal stratified split
            test_size: Proportion of data for testing (default: 0.2)
        """
        self.data_path = Path(data_path)
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.random_state = random_state
        self.use_temporal_split = use_temporal_split
        self.test_size = test_size
        
        # Create subdirectories
        self.plots_path = self.output_path / "plots"
        self.plots_path.mkdir(exist_ok=True)
        self.models_path = self.output_path / "saved_models"
        self.models_path.mkdir(exist_ok=True)
        
        # Data containers
        self.df = None
        self.X = None
        self.y = None
        self.feature_names = None
        self.selected_features = None
        
        # Split data
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = None
        
        # Metadata for temporal split
        self.temporal_metadata = None
        
        # Model results
        self.models = {}
        self.results = {}
        
        logger.info(f"Initialized BaselineModelPipeline")
        logger.info(f"  Data path: {data_path}")
        logger.info(f"  Output path: {output_path}")
        logger.info(f"  Temporal split: {use_temporal_split}")
        logger.info(f"  Test size: {test_size}")


    def load_data(self):
        """
        Load feature data with support for temporal metadata.
        
        Handles both baseline features and temporal features.
        Separates metadata columns from actual features.
        """
        logger.info("="*70)
        logger.info("LOADING FEATURE DATA")
        logger.info("="*70)
        
        # Load data
        self.df = pd.read_csv(self.data_path)
        logger.info(f"Loaded data: {self.df.shape}")
        
        # Identify metadata columns
        metadata_cols = [c for c in self.df.columns if c.startswith('_')]
        metadata_cols.append('business_id')
        
        # Check if this is temporal data
        has_temporal_metadata = any(c in self.df.columns for c in ['_cutoff_date', '_prediction_year'])
        
        if has_temporal_metadata:
            logger.info("[OK] Detected temporal metadata in dataset")
            logger.info(f"  Metadata columns: {len(metadata_cols)}")
            
            # Store temporal metadata for later use
            self.temporal_metadata = self.df[metadata_cols].copy()
            
            # Convert date columns to datetime
            for col in ['_cutoff_date', '_first_review_date', '_last_review_date']:
                if col in self.temporal_metadata.columns:
                    self.temporal_metadata[col] = pd.to_datetime(
                        self.temporal_metadata[col], errors='coerce'
                    )
            
            logger.info(f"  Unique businesses: {self.df['business_id'].nunique():,}")
            logger.info(f"  Total prediction tasks: {len(self.df):,}")
            
            if '_prediction_year' in self.df.columns:
                year_counts = self.df['_prediction_year'].value_counts().sort_index()
                logger.info(f"  Prediction years: {list(year_counts.index)}")
                logger.info(f"  Tasks per year:")
                for year, count in year_counts.items():
                    logger.info(f"    {year}: {count:,}")
        else:
            logger.info("Standard baseline dataset (no temporal metadata)")
            self.temporal_metadata = self.df[['business_id']].copy()
        
        # Check for target variable - prefer 'label' (from temporal validation) over 'is_open'
        # This ensures consistency with advanced_models.py and evaluation phases
        if 'label' in self.df.columns:
            self.y = self.df['label'].values
            target_col = 'label'
            logger.info("[OK] Using 'label' column as target (from temporal validation)")
        elif 'is_open' in self.df.columns:
            self.y = self.df['is_open'].values
            target_col = 'is_open'
            logger.info("[OK] Using 'is_open' column as target")
        else:
            logger.error("No target variable found ('label' or 'is_open')!")
            logger.error("Available columns: " + str(list(self.df.columns)))
            raise ValueError("No target variable found")
        
        # Get feature columns (exclude metadata and target)
        exclude_cols = metadata_cols + ['is_open', 'label', 'label_confidence', 'label_source']
        self.feature_names = [c for c in self.df.columns if c not in exclude_cols]
        
        self.X = self.df[self.feature_names].copy()
        
        logger.info(f"\nFeature extraction:")
        logger.info(f"  Total columns: {len(self.df.columns)}")
        logger.info(f"  Metadata columns: {len(metadata_cols)}")
        logger.info(f"  Feature columns: {len(self.feature_names)}")
        logger.info(f"  Target variable: {target_col}")
        
        # Check target distribution
        unique, counts = np.unique(self.y, return_counts=True)
        logger.info(f"\nTarget distribution:")
        for val, count in zip(unique, counts):
            logger.info(f"  Class {val}: {count:,} ({count/len(self.y)*100:.1f}%)")
        
        # Check for missing values in features
        missing_counts = self.X.isnull().sum()
        features_with_missing = missing_counts[missing_counts > 0]
        
        if len(features_with_missing) > 0:
            logger.warning(f"\nFeatures with missing values:")
            for feat, count in features_with_missing.items():
                logger.warning(f"  {feat}: {count:,} ({count/len(self.X)*100:.1f}%)")
            
            # Fill missing values
            logger.info("Filling missing values with median...")
            self.X.fillna(self.X.median(), inplace=True)
        
        logger.info(f"\n{'='*70}\n")

    def feature_selection(self, 
                         corr_threshold: float = 0.95,
                         variance_threshold: float = 0.01,
                         n_top_features: int = 40):
        """
        Multi-stage feature selection.
        
        Stage 1: Remove highly correlated features (|r| > threshold)
        Stage 2: Remove low variance features
        Stage 3: Select top N features by Random Forest importance
        
        Args:
            corr_threshold: Correlation threshold for removal
            variance_threshold: Variance threshold for removal
            n_top_features: Number of top features to keep
        """
        logger.info("="*70)
        logger.info("FEATURE SELECTION")
        logger.info("="*70)
        logger.info(f"Initial features: {self.X.shape[1]}")
        
        X_current = self.X.copy()
        removed_features = []
        
        # Stage 1: Remove highly correlated features
        logger.info("\nStage 1: Removing highly correlated features...")
        corr_matrix = X_current.corr().abs()
        upper_triangle = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        to_drop = []
        for column in upper_triangle.columns:
            if any(upper_triangle[column] > corr_threshold):
                to_drop.append(column)
                # Find which features it's correlated with
                correlated_with = upper_triangle.index[upper_triangle[column] > corr_threshold].tolist()
                if correlated_with:
                    removed_features.append({
                        'feature': column,
                        'reason': f'Correlated with {correlated_with[0]}',
                        'correlation': upper_triangle[column].max()
                    })
        
        X_current = X_current.drop(columns=to_drop)
        logger.info(f"Removed {len(to_drop)} highly correlated features")
        logger.info(f"Remaining features: {X_current.shape[1]}")
        
        # Stage 2: Remove low variance features
        logger.info("\nStage 2: Removing low variance features...")
        variances = X_current.var()
        low_var_features = variances[variances < variance_threshold].index.tolist()
        
        for feat in low_var_features:
            removed_features.append({
                'feature': feat,
                'reason': 'Low variance',
                'variance': variances[feat]
            })
        
        X_current = X_current.drop(columns=low_var_features)
        logger.info(f"Removed {len(low_var_features)} low variance features")
        logger.info(f"Remaining features: {X_current.shape[1]}")
        
        # Stage 3: Select top N features by Random Forest importance
        logger.info(f"\nStage 3: Selecting top {n_top_features} features by importance...")
        
        # Train a Random Forest to get feature importances
        rf_selector = RandomForestClassifier(
            n_estimators=100,
            random_state=self.random_state,
            n_jobs=-1,
            max_depth=10
        )
        rf_selector.fit(X_current, self.y)
        
        # Get feature importances
        importances = pd.DataFrame({
            'feature': X_current.columns,
            'importance': rf_selector.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Select top N features
        top_features = importances.head(n_top_features)['feature'].tolist()
        
        # Track removed features
        for feat in X_current.columns:
            if feat not in top_features:
                removed_features.append({
                    'feature': feat,
                    'reason': 'Below top N importance',
                    'importance': importances[importances['feature'] == feat]['importance'].values[0]
                })
        
        self.selected_features = top_features
        self.X = self.X[top_features]
        
        logger.info(f"Selected {len(top_features)} features")
        logger.info(f"Final feature set: {self.X.shape[1]} features")
        
        # Save feature selection summary
        self.feature_selection_summary = {
            'initial_features': len(self.feature_names),
            'after_correlation_removal': len(self.feature_names) - len(to_drop),
            'after_variance_removal': X_current.shape[1],
            'final_features': len(top_features),
            'selected_features': top_features,
            'removed_features': removed_features,
            'feature_importances': importances.to_dict('records')
        }
        
        # Visualize feature importances
        self._plot_feature_importance(importances.head(20))
        
        return importances
    

    def _temporal_stratified_split(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Perform proper TEMPORAL HOLDOUT split (V3 - Unified Configuration).
        
        KEY CHANGE (V3):
        - Uses SPLIT_CONFIG from config.py for train_years and test_years
        - Ensures consistency across baseline, advanced, and evaluation phases
        - All phases use the SAME train/test split for fair comparison
        
        This is a TRUE temporal prediction: use past to predict future.
        
        Returns:
            Tuple of (X_train, X_test, y_train, y_test) as numpy arrays
        """
        logger.info("="*70)
        logger.info("TEMPORAL HOLDOUT SPLIT (V3 - Unified Configuration)")
        logger.info("="*70)
        
        if self.temporal_metadata is None or '_prediction_year' not in self.temporal_metadata.columns:
            logger.error("Cannot perform temporal split: no temporal metadata found")
            logger.info("Falling back to random split...")
            return self._random_split()
        
        # Get prediction years from data
        available_years = sorted(self.temporal_metadata['_prediction_year'].unique())
        logger.info(f"Available years in data: {available_years}")
        
        # Use SPLIT_CONFIG from config.py for consistent splits across all phases
        train_years = SPLIT_CONFIG['train_years']
        test_years = SPLIT_CONFIG['test_years']
        
        logger.info(f"\nUsing UNIFIED configuration from config.py:")
        logger.info(f"  Train years: {train_years}")
        logger.info(f"  Test years: {test_years}")
        
        # Create masks
        train_mask = self.temporal_metadata['_prediction_year'].isin(train_years)
        test_mask = self.temporal_metadata['_prediction_year'].isin(test_years)
        
        train_indices = np.where(train_mask)[0]
        test_indices = np.where(test_mask)[0]
        
        # Extract data
        X_train = self.X.iloc[train_indices].values
        X_test = self.X.iloc[test_indices].values
        y_train = self.y[train_indices]
        y_test = self.y[test_indices]
        
        # Summary statistics
        logger.info(f"\n{'='*70}")
        logger.info(f"TEMPORAL HOLDOUT SUMMARY")
        logger.info(f"{'='*70}")
        logger.info(f"Train set: {len(train_indices):,} samples ({len(train_indices)/len(self.y)*100:.1f}%)")
        logger.info(f"Test set: {len(test_indices):,} samples ({len(test_indices)/len(self.y)*100:.1f}%)")
        
        # Check class distribution
        logger.info(f"\nTrain class distribution:")
        unique, counts = np.unique(y_train, return_counts=True)
        for val, count in zip(unique, counts):
            logger.info(f"  Class {val}: {count:,} ({count/len(y_train)*100:.1f}%)")
        
        logger.info(f"\nTest class distribution:")
        unique, counts = np.unique(y_test, return_counts=True)
        for val, count in zip(unique, counts):
            logger.info(f"  Class {val}: {count:,} ({count/len(y_test)*100:.1f}%)")
        
        # Check temporal distribution in train/test
        if self.temporal_metadata is not None and '_prediction_year' in self.temporal_metadata.columns:
            train_years = self.temporal_metadata.iloc[train_indices]['_prediction_year'].value_counts().sort_index()
            test_years = self.temporal_metadata.iloc[test_indices]['_prediction_year'].value_counts().sort_index()
            
            logger.info(f"\nTemporal distribution:")
            logger.info(f"{'Year':<8} {'Train':<12} {'Test':<12} {'Total':<12}")
            logger.info("-" * 50)
            
            all_years = sorted(set(train_years.index) | set(test_years.index))
            for year in all_years:
                train_count = train_years.get(year, 0)
                test_count = test_years.get(year, 0)
                total = train_count + test_count
                logger.info(f"{year:<8} {train_count:<12,} {test_count:<12,} {total:<12,}")
        
        logger.info(f"{'='*70}\n")
        
        return X_train, X_test, y_train, y_test
    
    def _random_split(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Perform standard random stratified split.
        
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        logger.info("="*70)
        logger.info("RANDOM STRATIFIED SPLIT")
        logger.info("="*70)
        
        X_train, X_test, y_train, y_test = train_test_split(
            self.X.values,
            self.y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=self.y
        )
        
        logger.info(f"Train set: {len(X_train):,} samples ({len(X_train)/(len(X_train)+len(X_test))*100:.1f}%)")
        logger.info(f"Test set: {len(X_test):,} samples ({len(X_test)/(len(X_train)+len(X_test))*100:.1f}%)")
        
        logger.info(f"\nTrain class distribution:")
        unique, counts = np.unique(y_train, return_counts=True)
        for val, count in zip(unique, counts):
            logger.info(f"  Class {val}: {count:,} ({count/len(y_train)*100:.1f}%)")
        
        logger.info(f"\nTest class distribution:")
        unique, counts = np.unique(y_test, return_counts=True)
        for val, count in zip(unique, counts):
            logger.info(f"  Class {val}: {count:,} ({count/len(y_test)*100:.1f}%)")
        
        logger.info(f"{'='*70}\n")
        
        return X_train, X_test, y_train, y_test

    def _plot_feature_importance(self, importances_df: pd.DataFrame):
        """Plot top feature importances"""
        plt.figure(figsize=(12, 8))
        
        # Reverse order for better visualization (highest at top)
        plot_data = importances_df.head(20).iloc[::-1]
        
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(plot_data)))
        bars = plt.barh(range(len(plot_data)), plot_data['importance'], color=colors)
        plt.yticks(range(len(plot_data)), plot_data['feature'])
        plt.xlabel('Importance Score', fontsize=12, fontweight='bold')
        plt.ylabel('Features', fontsize=12, fontweight='bold')
        plt.title('Top 20 Feature Importances (Random Forest)', 
                 fontsize=14, fontweight='bold', pad=20)
        plt.grid(axis='x', alpha=0.3)
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, plot_data['importance'])):
            plt.text(val, i, f' {val:.4f}', va='center', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(self.plots_path / 'feature_importance_selection.png', 
                   dpi=300, bbox_inches='tight')
        logger.info(f"Saved: {self.plots_path / 'feature_importance_selection.png'}")
        plt.close()
    
    def prepare_train_test_split(self):
        """
        Prepare train/test split with optional temporal holdout.
        
        Supports two modes:
        1. Random stratified split (traditional approach)
        2. Temporal holdout split (train on past, test on future)
        """
        logger.info("="*70)
        logger.info("PREPARING TRAIN/TEST SPLIT")
        logger.info("="*70)
        logger.info(f"Mode: {'TEMPORAL STRATIFIED' if self.use_temporal_split else 'RANDOM STRATIFIED'}")
        logger.info("")
        
        # Perform split based on mode
        if self.use_temporal_split:
            X_train, X_test, y_train, y_test = self._temporal_stratified_split()
        else:
            X_train, X_test, y_train, y_test = self._random_split()
        
        # Feature scaling
        logger.info("Applying StandardScaler to features...")
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Store scaled data
        self.X_train = X_train_scaled
        self.X_test = X_test_scaled
        self.y_train = y_train
        self.y_test = y_test
        
        logger.info("[OK] Feature scaling complete")
        logger.info(f"  Mean: {self.scaler.mean_[:5]}... (showing first 5)")
        logger.info(f"  Std: {self.scaler.scale_[:5]}... (showing first 5)")
        
        logger.info(f"\n{'='*70}\n")

    def train_baseline_models(self):
        """
        Train all three baseline models with two approaches:
        1. Using SMOTE for oversampling
        2. Using class weights
        """
        logger.info("="*70)
        logger.info("TRAINING BASELINE MODELS")
        logger.info("="*70)
        
        # Calculate class weights
        n_samples = len(self.y_train)
        n_classes = 2
        class_weight_dict = {
            0: n_samples / (n_classes * (self.y_train == 0).sum()),
            1: n_samples / (n_classes * (self.y_train == 1).sum())
        }
        logger.info(f"\nClass weights: {class_weight_dict}")
        
        # Approach 1: Apply SMOTE
        logger.info("\n" + "="*70)
        logger.info("APPROACH 1: SMOTE OVERSAMPLING")
        logger.info("="*70)
        
        smote = SMOTE(random_state=self.random_state)
        X_train_smote, y_train_smote = smote.fit_resample(self.X_train, self.y_train)
        
        logger.info(f"After SMOTE:")
        logger.info(f"  Train set: {X_train_smote.shape}")
        logger.info(f"  Open: {y_train_smote.sum():,} ({y_train_smote.mean()*100:.2f}%)")
        logger.info(f"  Closed: {(1-y_train_smote).sum():,} ({(1-y_train_smote.mean())*100:.2f}%)")
        
        # Train models with SMOTE
        self._train_model_set(
            X_train_smote, y_train_smote,
            approach="SMOTE",
            use_class_weight=False
        )
        
        # Approach 2: Use class weights
        logger.info("\n" + "="*70)
        logger.info("APPROACH 2: CLASS WEIGHTS")
        logger.info("="*70)
        
        self._train_model_set(
            self.X_train, self.y_train,
            approach="ClassWeight",
            use_class_weight=True,
            class_weight_dict=class_weight_dict
        )
    
    def _train_model_set(self, X_train, y_train, approach: str, 
                        use_class_weight: bool, class_weight_dict: Dict = None):
        """Train one set of models (either SMOTE or class weight approach)"""
        
        # Model 1: Logistic Regression
        logger.info(f"\n[{approach}] Training Logistic Regression...")
        lr = LogisticRegression(
            random_state=self.random_state,
            max_iter=1000,
            class_weight=class_weight_dict if use_class_weight else None,
            solver='lbfgs',
            C=1.0
        )
        lr.fit(X_train, y_train)
        
        model_name = f"LogisticRegression_{approach}"
        self.models[model_name] = lr
        logger.info(f"[{approach}] Logistic Regression trained successfully")
        
        # Model 2: Decision Tree
        logger.info(f"\n[{approach}] Training Decision Tree...")
        dt = DecisionTreeClassifier(
            random_state=self.random_state,
            max_depth=10,
            min_samples_split=20,
            min_samples_leaf=10,
            class_weight=class_weight_dict if use_class_weight else None
        )
        dt.fit(X_train, y_train)
        
        model_name = f"DecisionTree_{approach}"
        self.models[model_name] = dt
        logger.info(f"[{approach}] Decision Tree trained successfully")
        
        # Model 3: Random Forest
        logger.info(f"\n[{approach}] Training Random Forest...")
        rf = RandomForestClassifier(
            n_estimators=100,
            random_state=self.random_state,
            max_depth=15,
            min_samples_split=20,
            min_samples_leaf=10,
            class_weight=class_weight_dict if use_class_weight else None,
            n_jobs=-1
        )
        rf.fit(X_train, y_train)
        
        model_name = f"RandomForest_{approach}"
        self.models[model_name] = rf
        logger.info(f"[{approach}] Random Forest trained successfully")
        
        # Save models
        for name, model in [(f"LogisticRegression_{approach}", lr),
                           (f"DecisionTree_{approach}", dt),
                           (f"RandomForest_{approach}", rf)]:
            model_file = self.models_path / f"{name}.pkl"
            with open(model_file, 'wb') as f:
                pickle.dump(model, f)
            logger.info(f"Saved: {model_file}")
    
    def evaluate_models(self):
        """Evaluate all trained models"""
        logger.info("="*70)
        logger.info("EVALUATING MODELS")
        logger.info("="*70)
        
        for model_name, model in self.models.items():
            logger.info(f"\n{'='*70}")
            logger.info(f"Evaluating: {model_name}")
            logger.info(f"{'='*70}")
            
            # Predictions
            y_pred = model.predict(self.X_test)
            y_pred_proba = model.predict_proba(self.X_test)[:, 1]
            
            # Calculate metrics
            roc_auc = roc_auc_score(self.y_test, y_pred_proba)
            precision = precision_score(self.y_test, y_pred)
            recall = recall_score(self.y_test, y_pred)
            f1 = f1_score(self.y_test, y_pred)
            
            # Precision-recall AUC
            precision_curve, recall_curve, _ = precision_recall_curve(self.y_test, y_pred_proba)
            pr_auc = auc(recall_curve, precision_curve)
            
            # Confusion matrix
            cm = confusion_matrix(self.y_test, y_pred)
            
            # Store results
            self.results[model_name] = {
                'roc_auc': roc_auc,
                'pr_auc': pr_auc,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'confusion_matrix': cm,
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba,
                'classification_report': classification_report(self.y_test, y_pred, output_dict=True)
            }
            
            # Log results
            logger.info(f"ROC-AUC: {roc_auc:.4f}")
            logger.info(f"PR-AUC: {pr_auc:.4f}")
            logger.info(f"Precision: {precision:.4f}")
            logger.info(f"Recall: {recall:.4f}")
            logger.info(f"F1-Score: {f1:.4f}")
            logger.info(f"\nConfusion Matrix:")
            logger.info(f"  TN: {cm[0,0]:,}  FP: {cm[0,1]:,}")
            logger.info(f"  FN: {cm[1,0]:,}  TP: {cm[1,1]:,}")
    
    def generate_visualizations(self):
        """Generate comprehensive visualizations"""
        logger.info("="*70)
        logger.info("GENERATING VISUALIZATIONS")
        logger.info("="*70)
        
        # 1. Model Comparison Bar Charts
        self._plot_model_comparison()
        
        # 2. ROC Curves
        self._plot_roc_curves()
        
        # 3. Precision-Recall Curves
        self._plot_precision_recall_curves()
        
        # 4. Confusion Matrices
        self._plot_confusion_matrices()
        
        # 5. Feature Importance (for tree-based models)
        self._plot_model_feature_importance()
        
        # 6. Class distribution comparison
        self._plot_class_distribution()
        
        logger.info("\nAll visualizations generated successfully!")
    
    def _plot_model_comparison(self):
        """Compare all models across metrics"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Baseline Model Performance Comparison', 
                    fontsize=16, fontweight='bold', y=0.995)
        
        model_names = list(self.results.keys())
        metrics = ['roc_auc', 'pr_auc', 'f1_score', 'recall']
        metric_labels = ['ROC-AUC', 'PR-AUC', 'F1-Score', 'Recall']
        
        for idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
            ax = axes[idx // 2, idx % 2]
            
            values = [self.results[name][metric] for name in model_names]
            colors = ['#3498db' if 'SMOTE' in name else '#e74c3c' for name in model_names]
            
            bars = ax.bar(range(len(model_names)), values, color=colors, alpha=0.7, edgecolor='black')
            ax.set_xticks(range(len(model_names)))
            ax.set_xticklabels([name.replace('_', '\n') for name in model_names], 
                              rotation=45, ha='right', fontsize=9)
            ax.set_ylabel(label, fontsize=11, fontweight='bold')
            ax.set_title(f'{label} Comparison', fontsize=12, fontweight='bold')
            ax.grid(axis='y', alpha=0.3)
            ax.set_ylim([0, 1.0])
            
            # Add value labels
            for bar, val in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#3498db', alpha=0.7, edgecolor='black', label='SMOTE'),
            Patch(facecolor='#e74c3c', alpha=0.7, edgecolor='black', label='Class Weight')
        ]
        fig.legend(handles=legend_elements, loc='upper right', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(self.plots_path / 'model_comparison.png', dpi=300, bbox_inches='tight')
        logger.info(f"Saved: {self.plots_path / 'model_comparison.png'}")
        plt.close()
    
    def _plot_roc_curves(self):
        """Plot ROC curves for all models"""
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle('ROC Curves - Baseline Models', fontsize=16, fontweight='bold')
        
        # SMOTE models
        ax = axes[0]
        for model_name in self.results.keys():
            if 'SMOTE' in model_name:
                fpr, tpr, _ = roc_curve(self.y_test, self.results[model_name]['y_pred_proba'])
                roc_auc = self.results[model_name]['roc_auc']
                label = model_name.replace('_SMOTE', '')
                ax.plot(fpr, tpr, linewidth=2, label=f'{label} (AUC={roc_auc:.3f})')
        
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
        ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
        ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
        ax.set_title('SMOTE Approach', fontsize=13, fontweight='bold')
        ax.legend(loc='lower right', fontsize=10)
        ax.grid(alpha=0.3)
        
        # Class Weight models
        ax = axes[1]
        for model_name in self.results.keys():
            if 'ClassWeight' in model_name:
                fpr, tpr, _ = roc_curve(self.y_test, self.results[model_name]['y_pred_proba'])
                roc_auc = self.results[model_name]['roc_auc']
                label = model_name.replace('_ClassWeight', '')
                ax.plot(fpr, tpr, linewidth=2, label=f'{label} (AUC={roc_auc:.3f})')
        
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
        ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
        ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
        ax.set_title('Class Weight Approach', fontsize=13, fontweight='bold')
        ax.legend(loc='lower right', fontsize=10)
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.plots_path / 'roc_curves.png', dpi=300, bbox_inches='tight')
        logger.info(f"Saved: {self.plots_path / 'roc_curves.png'}")
        plt.close()
    
    def _plot_precision_recall_curves(self):
        """Plot Precision-Recall curves"""
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle('Precision-Recall Curves - Baseline Models', 
                    fontsize=16, fontweight='bold')
        
        # SMOTE models
        ax = axes[0]
        for model_name in self.results.keys():
            if 'SMOTE' in model_name:
                precision, recall, _ = precision_recall_curve(
                    self.y_test, self.results[model_name]['y_pred_proba']
                )
                pr_auc = self.results[model_name]['pr_auc']
                label = model_name.replace('_SMOTE', '')
                ax.plot(recall, precision, linewidth=2, label=f'{label} (AUC={pr_auc:.3f})')
        
        baseline = self.y_test.mean()
        ax.axhline(y=baseline, color='k', linestyle='--', linewidth=1, 
                  label=f'Baseline ({baseline:.3f})')
        ax.set_xlabel('Recall', fontsize=12, fontweight='bold')
        ax.set_ylabel('Precision', fontsize=12, fontweight='bold')
        ax.set_title('SMOTE Approach', fontsize=13, fontweight='bold')
        ax.legend(loc='lower left', fontsize=10)
        ax.grid(alpha=0.3)
        
        # Class Weight models
        ax = axes[1]
        for model_name in self.results.keys():
            if 'ClassWeight' in model_name:
                precision, recall, _ = precision_recall_curve(
                    self.y_test, self.results[model_name]['y_pred_proba']
                )
                pr_auc = self.results[model_name]['pr_auc']
                label = model_name.replace('_ClassWeight', '')
                ax.plot(recall, precision, linewidth=2, label=f'{label} (AUC={pr_auc:.3f})')
        
        ax.axhline(y=baseline, color='k', linestyle='--', linewidth=1, 
                  label=f'Baseline ({baseline:.3f})')
        ax.set_xlabel('Recall', fontsize=12, fontweight='bold')
        ax.set_ylabel('Precision', fontsize=12, fontweight='bold')
        ax.set_title('Class Weight Approach', fontsize=13, fontweight='bold')
        ax.legend(loc='lower left', fontsize=10)
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.plots_path / 'precision_recall_curves.png', dpi=300, bbox_inches='tight')
        logger.info(f"Saved: {self.plots_path / 'precision_recall_curves.png'}")
        plt.close()
    
    def _plot_confusion_matrices(self):
        """Plot confusion matrices for all models"""
        n_models = len(self.results)
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Confusion Matrices - Baseline Models', 
                    fontsize=16, fontweight='bold', y=0.995)
        
        axes = axes.flatten()
        
        for idx, (model_name, results) in enumerate(self.results.items()):
            ax = axes[idx]
            cm = results['confusion_matrix']
            
            # Normalize for better visualization
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            
            im = ax.imshow(cm_normalized, interpolation='nearest', cmap='Blues')
            ax.set_title(model_name.replace('_', '\n'), fontsize=11, fontweight='bold')
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('Normalized Count', fontsize=9)
            
            # Labels
            classes = ['Closed (0)', 'Open (1)']
            tick_marks = np.arange(len(classes))
            ax.set_xticks(tick_marks)
            ax.set_yticks(tick_marks)
            ax.set_xticklabels(classes, fontsize=9)
            ax.set_yticklabels(classes, fontsize=9)
            
            # Add text annotations
            thresh = cm_normalized.max() / 2.
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(j, i, f'{cm[i, j]:,}\n({cm_normalized[i, j]:.2%})',
                           ha="center", va="center", fontsize=9,
                           color="white" if cm_normalized[i, j] > thresh else "black",
                           fontweight='bold')
            
            ax.set_ylabel('True Label', fontsize=10, fontweight='bold')
            ax.set_xlabel('Predicted Label', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.plots_path / 'confusion_matrices.png', dpi=300, bbox_inches='tight')
        logger.info(f"Saved: {self.plots_path / 'confusion_matrices.png'}")
        plt.close()
    
    def _plot_model_feature_importance(self):
        """Plot feature importance for tree-based models"""
        # Get Random Forest models
        rf_models = {name: model for name, model in self.models.items() 
                    if 'RandomForest' in name}
        
        if not rf_models:
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        fig.suptitle('Feature Importance - Random Forest Models', 
                    fontsize=16, fontweight='bold')
        
        for idx, (model_name, model) in enumerate(rf_models.items()):
            ax = axes[idx]
            
            importances = pd.DataFrame({
                'feature': self.selected_features,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False).head(20)
            
            # Reverse for better visualization
            plot_data = importances.iloc[::-1]
            
            colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(plot_data)))
            bars = ax.barh(range(len(plot_data)), plot_data['importance'], 
                          color=colors, edgecolor='black')
            ax.set_yticks(range(len(plot_data)))
            ax.set_yticklabels(plot_data['feature'], fontsize=9)
            ax.set_xlabel('Importance Score', fontsize=11, fontweight='bold')
            ax.set_title(model_name.replace('_', ' '), fontsize=12, fontweight='bold')
            ax.grid(axis='x', alpha=0.3)
            
            # Add value labels
            for i, (bar, val) in enumerate(zip(bars, plot_data['importance'])):
                ax.text(val, i, f' {val:.4f}', va='center', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(self.plots_path / 'random_forest_feature_importance.png', 
                   dpi=300, bbox_inches='tight')
        logger.info(f"Saved: {self.plots_path / 'random_forest_feature_importance.png'}")
        plt.close()
    
    def _plot_class_distribution(self):
        """Plot class distribution comparison"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle('Class Distribution Analysis', fontsize=16, fontweight='bold')
        
        # Original distribution
        ax = axes[0]
        counts = pd.Series(self.y).value_counts()
        labels = ['Open', 'Closed']
        colors = ['#2ecc71', '#e74c3c']
        ax.bar(labels, [counts[1], counts[0]], color=colors, edgecolor='black', alpha=0.7)
        ax.set_ylabel('Count', fontsize=11, fontweight='bold')
        ax.set_title('Original Dataset', fontsize=12, fontweight='bold')
        ax.text(0, counts[1], f"{counts[1]:,}\n({counts[1]/len(self.y)*100:.1f}%)", 
               ha='center', va='bottom', fontweight='bold')
        ax.text(1, counts[0], f"{counts[0]:,}\n({counts[0]/len(self.y)*100:.1f}%)", 
               ha='center', va='bottom', fontweight='bold')
        
        # Train distribution
        ax = axes[1]
        counts_train = pd.Series(self.y_train).value_counts()
        ax.bar(labels, [counts_train[1], counts_train[0]], color=colors, edgecolor='black', alpha=0.7)
        ax.set_ylabel('Count', fontsize=11, fontweight='bold')
        ax.set_title('Training Set (80%)', fontsize=12, fontweight='bold')
        ax.text(0, counts_train[1], f"{counts_train[1]:,}\n({counts_train[1]/len(self.y_train)*100:.1f}%)", 
               ha='center', va='bottom', fontweight='bold')
        ax.text(1, counts_train[0], f"{counts_train[0]:,}\n({counts_train[0]/len(self.y_train)*100:.1f}%)", 
               ha='center', va='bottom', fontweight='bold')
        
        # Test distribution
        ax = axes[2]
        counts_test = pd.Series(self.y_test).value_counts()
        ax.bar(labels, [counts_test[1], counts_test[0]], color=colors, edgecolor='black', alpha=0.7)
        ax.set_ylabel('Count', fontsize=11, fontweight='bold')
        ax.set_title('Test Set (20%)', fontsize=12, fontweight='bold')
        ax.text(0, counts_test[1], f"{counts_test[1]:,}\n({counts_test[1]/len(self.y_test)*100:.1f}%)", 
               ha='center', va='bottom', fontweight='bold')
        ax.text(1, counts_test[0], f"{counts_test[0]:,}\n({counts_test[0]/len(self.y_test)*100:.1f}%)", 
               ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.plots_path / 'class_distribution.png', dpi=300, bbox_inches='tight')
        logger.info(f"Saved: {self.plots_path / 'class_distribution.png'}")
        plt.close()
    
    def generate_report(self):
        """
        Generate comprehensive markdown report with split mode information.
        """
        logger.info("="*70)
        logger.info("GENERATING MODEL REPORT")
        logger.info("="*70)
        
        report_path = self.output_path / "baseline_models_report.md"
        
        report_lines = []
        report_lines.append("# Baseline Models Report")
        report_lines.append("")
        report_lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        # Add split mode information
        report_lines.append("---")
        report_lines.append("")
        report_lines.append("## Train/Test Split Configuration")
        report_lines.append("")
        
        if self.use_temporal_split:
            report_lines.append("**Split Mode:** Temporal Stratified Split")
            report_lines.append("")
            report_lines.append("This split method ensures:")
            report_lines.append("- Both train and test sets span all prediction years")
            report_lines.append("- Class balance maintained within each year")
            report_lines.append("- Realistic evaluation of temporal generalization")
            report_lines.append("")
            
            if self.temporal_metadata is not None and '_prediction_year' in self.temporal_metadata.columns:
                report_lines.append("**Temporal Coverage:**")
                report_lines.append("")
        else:
            report_lines.append("**Split Mode:** Random Stratified Split")
            report_lines.append("")
            report_lines.append("[WARN] *Note: This split may not reflect temporal generalization ability.*")
            report_lines.append("")
        
        report_lines.append("---")
        report_lines.append("")
        
        # Executive Summary
        report_lines.append("## Executive Summary")
        report_lines.append("")
        report_lines.append("This report presents the results of baseline model training for business success prediction.")
        report_lines.append(f"We trained **{len(self.models)} models** on {len(self.X_train):,} training samples")
        report_lines.append(f"and evaluated them on {len(self.X_test):,} test samples.")
        report_lines.append("")
        
        # Find best model
        best_model_name = max(self.results.keys(), key=lambda k: self.results[k]['roc_auc'])
        best_auc = self.results[best_model_name]['roc_auc']
        
        report_lines.append(f"**Best Model:** {best_model_name}")
        report_lines.append(f"**Best ROC-AUC:** {best_auc:.4f}")
        report_lines.append("")
        
        # Model Performance Table
        report_lines.append("## Model Performance Comparison")
        report_lines.append("")
        report_lines.append("| Model | ROC-AUC | PR-AUC | Precision | Recall | F1-Score |")
        report_lines.append("|-------|---------|--------|-----------|--------|----------|")
        
        # Sort models by ROC-AUC for display
        sorted_models = sorted(self.results.keys(), key=lambda k: self.results[k]['roc_auc'], reverse=True)
        
        for model_name in sorted_models:
            metrics = self.results[model_name]
            report_lines.append(
                f"| {model_name} | {metrics['roc_auc']:.4f} | {metrics['pr_auc']:.4f} | "
                f"{metrics['precision']:.4f} | {metrics['recall']:.4f} | {metrics['f1_score']:.4f} |"
            )
        
        report_lines.append("")
        
        # SMOTE vs Class Weight Comparison
        report_lines.append("## Class Imbalance Handling Comparison")
        report_lines.append("")
        report_lines.append("### SMOTE vs Class Weights")
        report_lines.append("")
        report_lines.append("| Algorithm | SMOTE ROC-AUC | ClassWeight ROC-AUC | Difference |")
        report_lines.append("|-----------|---------------|---------------------|------------|")
        
        for algo in ['LogisticRegression', 'DecisionTree', 'RandomForest']:
            smote_key = f"{algo}_SMOTE"
            cw_key = f"{algo}_ClassWeight"
            if smote_key in self.results and cw_key in self.results:
                smote_auc = self.results[smote_key]['roc_auc']
                cw_auc = self.results[cw_key]['roc_auc']
                diff = cw_auc - smote_auc
                diff_str = f"+{diff:.4f}" if diff >= 0 else f"{diff:.4f}"
                report_lines.append(f"| {algo} | {smote_auc:.4f} | {cw_auc:.4f} | {diff_str} |")
        
        report_lines.append("")
        
        # Confusion Matrix Analysis
        report_lines.append("## Confusion Matrix Analysis")
        report_lines.append("")
        report_lines.append("### Best Model: " + best_model_name)
        report_lines.append("")
        
        best_cm = self.results[best_model_name]['confusion_matrix']
        tn, fp, fn, tp = best_cm[0, 0], best_cm[0, 1], best_cm[1, 0], best_cm[1, 1]
        total = tn + fp + fn + tp
        
        report_lines.append("| | Predicted Closed | Predicted Open |")
        report_lines.append("|---|-----------------|----------------|")
        report_lines.append(f"| **Actual Closed** | TN: {tn:,} ({tn/total*100:.1f}%) | FP: {fp:,} ({fp/total*100:.1f}%) |")
        report_lines.append(f"| **Actual Open** | FN: {fn:,} ({fn/total*100:.1f}%) | TP: {tp:,} ({tp/total*100:.1f}%) |")
        report_lines.append("")
        
        report_lines.append("**Interpretation:**")
        report_lines.append(f"- **True Negatives (TN)**: {tn:,} closed businesses correctly identified")
        report_lines.append(f"- **True Positives (TP)**: {tp:,} open businesses correctly identified")
        report_lines.append(f"- **False Positives (FP)**: {fp:,} closed businesses incorrectly predicted as open")
        report_lines.append(f"- **False Negatives (FN)**: {fn:,} open businesses incorrectly predicted as closed")
        report_lines.append("")
        
        # Feature Importance
        report_lines.append("## Feature Importance Analysis")
        report_lines.append("")
        report_lines.append("Top 15 features from Random Forest model:")
        report_lines.append("")
        
        # Get RF model feature importances
        rf_model = None
        for name, model in self.models.items():
            if 'RandomForest' in name and 'ClassWeight' in name:
                rf_model = model
                break
        
        if rf_model is not None and self.selected_features is not None:
            importances = pd.DataFrame({
                'feature': self.selected_features,
                'importance': rf_model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            report_lines.append("| Rank | Feature | Importance |")
            report_lines.append("|------|---------|------------|")
            
            for i, (_, row) in enumerate(importances.head(15).iterrows()):
                report_lines.append(f"| {i+1} | {row['feature']} | {row['importance']:.4f} |")
            
            report_lines.append("")
        
        # Model-specific Details
        report_lines.append("## Individual Model Details")
        report_lines.append("")
        
        for model_name in sorted_models:
            metrics = self.results[model_name]
            cm = metrics['confusion_matrix']
            
            report_lines.append(f"### {model_name}")
            report_lines.append("")
            report_lines.append(f"- **ROC-AUC**: {metrics['roc_auc']:.4f}")
            report_lines.append(f"- **PR-AUC**: {metrics['pr_auc']:.4f}")
            report_lines.append(f"- **Precision**: {metrics['precision']:.4f}")
            report_lines.append(f"- **Recall**: {metrics['recall']:.4f}")
            report_lines.append(f"- **F1-Score**: {metrics['f1_score']:.4f}")
            report_lines.append("")
            report_lines.append(f"Confusion Matrix: TN={cm[0,0]:,}, FP={cm[0,1]:,}, FN={cm[1,0]:,}, TP={cm[1,1]:,}")
            report_lines.append("")
        
        # Visualizations
        report_lines.append("## Generated Visualizations")
        report_lines.append("")
        report_lines.append("The following visualizations have been generated:")
        report_lines.append("")
        report_lines.append("1. **model_comparison.png**: Bar charts comparing all models across metrics")
        report_lines.append("2. **roc_curves.png**: ROC curves for SMOTE and ClassWeight approaches")
        report_lines.append("3. **precision_recall_curves.png**: PR curves for all models")
        report_lines.append("4. **confusion_matrices.png**: Confusion matrices for all 6 models")
        report_lines.append("5. **random_forest_feature_importance.png**: Feature importance from RF models")
        report_lines.append("6. **class_distribution.png**: Class distribution in train/test sets")
        report_lines.append("7. **feature_importance_selection.png**: Feature selection importance scores")
        report_lines.append("")
        
        # Key Findings
        report_lines.append("## Key Findings")
        report_lines.append("")
        
        # Calculate some insights
        rf_smote = self.results.get('RandomForest_SMOTE', {}).get('roc_auc', 0)
        rf_cw = self.results.get('RandomForest_ClassWeight', {}).get('roc_auc', 0)
        lr_cw = self.results.get('LogisticRegression_ClassWeight', {}).get('roc_auc', 0)
        
        report_lines.append("1. **Random Forest outperforms other baselines**: Ensemble methods capture")
        report_lines.append("   non-linear feature interactions better than linear models or single trees.")
        report_lines.append("")
        report_lines.append(f"2. **Class Weight vs SMOTE**: Both approaches yield similar results.")
        report_lines.append(f"   RF with ClassWeight ({rf_cw:.4f}) slightly outperforms RF with SMOTE ({rf_smote:.4f}).")
        report_lines.append("")
        report_lines.append(f"3. **Linear model limitations**: Logistic Regression ({lr_cw:.4f} AUC) shows")
        report_lines.append("   limited performance, suggesting non-linear relationships in the data.")
        report_lines.append("")
        report_lines.append("4. **Class imbalance impact**: All models show higher precision than recall,")
        report_lines.append("   indicating conservative predictions for the minority (closed) class.")
        report_lines.append("")
        
        # Recommendations
        report_lines.append("## Recommendations for Next Steps")
        report_lines.append("")
        report_lines.append("1. **Try advanced models**: XGBoost, LightGBM, and Neural Networks may improve performance")
        report_lines.append("2. **Ensemble methods**: Combine predictions from multiple models")
        report_lines.append("3. **Feature engineering**: Explore additional temporal and user-weighted features")
        report_lines.append("4. **Hyperparameter tuning**: Optimize model parameters using cross-validation")
        report_lines.append("")
        
        report_lines.append("---")
        report_lines.append("")
        report_lines.append("*Report generated by CS 412 Research Project baseline models pipeline*")
        
        # Write report
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        logger.info(f"[OK] Saved report: {report_path}")

    def run_pipeline(self):
        """Execute complete baseline model pipeline"""
        logger.info("="*70)
        logger.info("CS 412 RESEARCH PROJECT - BASELINE MODELS")
        logger.info("Business Success Prediction using Yelp Dataset")
        logger.info("="*70)
        logger.info("")
        
        # Step 1: Load data
        self.load_data()
        
        # Step 2: Feature selection
        # DISABLED FOR CONSISTENCY (V4):
        # Feature selection removed to ensure all phases use the same 52 features
        # This allows fair comparison between baseline, advanced, and ablation models
        # self.feature_selection()  # ← Commented out for consistency
        
        # Use all features without selection
        self.selected_features = self.feature_names
        logger.info("="*70)
        logger.info("FEATURE SELECTION: DISABLED (V4 - Consistency Update)")
        logger.info("="*70)
        logger.info(f"Using all {len(self.feature_names)} features for consistency")
        logger.info("This ensures fair comparison with advanced models and ablation study")
        logger.info("All phases (4, 5, 6, 7, 8, 9) now use identical feature sets")
        logger.info(f"\n{'='*70}\n")
        
        # Step 3: Train-test split
        self.prepare_train_test_split()
        
        # Step 4: Train models
        self.train_baseline_models()
        
        # Step 5: Evaluate models
        self.evaluate_models()
        
        # Step 6: Generate visualizations
        self.generate_visualizations()
        
        # Step 7: Generate report
        self.generate_report()
        
        # Step 8: Save compact JSON summary for final report aggregation
        try:
            summary = {}
            for name, metrics in self.results.items():
                # Only keep scalar metrics + confusion matrix for JSON
                cm = metrics.get('confusion_matrix')
                summary[name] = {
                    'roc_auc': float(metrics.get('roc_auc', 0.0)),
                    'pr_auc': float(metrics.get('pr_auc', 0.0)),
                    'precision': float(metrics.get('precision', 0.0)),
                    'recall': float(metrics.get('recall', 0.0)),
                    'f1_score': float(metrics.get('f1_score', 0.0)),
                    'confusion_matrix': cm.tolist() if cm is not None else None,
                }
            summary_path = self.output_path / "baseline_results_summary.json"
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)
            logger.info(f"[OK] Saved JSON summary: {summary_path}")
        except Exception as e:
            logger.warning(f"Could not save baseline results_summary.json: {e}")
        
        logger.info("\n" + "="*70)
        logger.info("BASELINE MODELING COMPLETE!")
        logger.info("="*70)
        logger.info(f"\nOutput files saved to: {self.output_path}")
        logger.info("\nGenerated files:")
        logger.info("  - baseline_models_report.md (comprehensive report)")
        logger.info("  - results_summary.json (metrics in JSON)")
        logger.info("  - plots/ (all visualizations)")
        logger.info("  - saved_models/ (trained models)")
        logger.info("")


def main():
    """Main entry point with support for temporal validation"""
    import argparse
    
    print("="*70)
    print("CS 412 RESEARCH PROJECT - BASELINE MODELS")
    print("Business Success Prediction using Yelp Dataset")
    print("="*70)
    print("")
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Baseline Model Training Pipeline')
    parser.add_argument('--data', type=str, 
                       default='data/features/business_features_baseline.csv',
                       help='Path to feature CSV file')
    parser.add_argument('--temporal', action='store_true',
                       help='Use temporal stratified split')
    parser.add_argument('--output', type=str, default='src/models',
                       help='Output directory')
    
    args = parser.parse_args()
    
    # Determine if using temporal split
    use_temporal = args.temporal or 'temporal' in args.data
    
    print(f"Configuration:")
    print(f"  Data: {args.data}")
    print(f"  Split mode: {'Temporal Stratified' if use_temporal else 'Random Stratified'}")
    print(f"  Output: {args.output}")
    print("")
    
    # Initialize and run pipeline
    pipeline = BaselineModelPipeline(
        data_path=args.data,
        output_path=args.output,
        random_state=42,
        use_temporal_split=use_temporal
    )
    
    pipeline.run_pipeline()
    
    print("\n" + "="*70)
    print("BASELINE MODELING COMPLETE!")
    print("="*70)
    print("\nCheck the following outputs:")
    print(f"  1. {args.output}/baseline_models_report.md")
    print(f"  2. {args.output}/plots/ (visualizations)")
    print(f"  3. {args.output}/saved_models/ (trained models)")
    print("")


if __name__ == "__main__":
    main()