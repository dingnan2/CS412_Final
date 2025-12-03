"""
Temporal Validation Pipeline (V3 - Unified Configuration)

This module implements proper temporal validation WITHOUT label leakage:

KEY CHANGE FROM V1:
- V1 (Old): Inferred labels from review activity patterns, then used review-based 
  features to predict those labels. This created CIRCULAR DEPENDENCY.
- V2 (New): Uses final is_open status as GROUND TRUTH label. No inference needed.
  This is a TRUE prediction task: predict future business status from historical data.

CRITICAL (V3):
- Uses SPLIT_CONFIG from config.py for consistent train/test years
- Generates labeled data file used by ALL subsequent phases (5-9)
- Ensures all modeling phases use the SAME train/test split

Temporal Split Strategy:
- Train on earlier years (SPLIT_CONFIG['train_years'])
- Test on later years (SPLIT_CONFIG['test_years'])
- TRUE temporal prediction: use past to predict future
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import sys

# Import utilities
sys.path.append(str(Path(__file__).parent.parent))
from utils.validation import validate_feature_quality

# Import unified configuration
try:
    from config import SPLIT_CONFIG, RANDOM_STATE, DATA_PATHS
except ImportError:
    # Fallback defaults if config not available
    SPLIT_CONFIG = {
        'train_years': [2012, 2013, 2014, 2015, 2016, 2017, 2018],
        'test_years': [2019, 2020]
    }
    RANDOM_STATE = 42
    DATA_PATHS = {'features_labeled': 'data/features/business_features_temporal_labeled_12m.csv'}

# Scikit-learn imports
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

# Configure plotting
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class TemporalValidator:
    """
    Leakage-Free Temporal Validation Framework.
    
    Key Design Principles:
    1. Labels use GROUND TRUTH (is_open), not inferred from patterns
    2. Temporal split uses HOLDOUT (train on past, test on future)
    3. Features only use data BEFORE the cutoff date
    
    This ensures NO circular dependency between features and labels.
    """
    
    def __init__(self,
                 features_path: str = "data/features/business_features_temporal.csv",
                 business_path: str = "data/processed/business_clean.csv",
                 review_path: str = "data/processed/review_clean.csv",
                 output_path: str = "src/models/temporal_validation",
                 random_state: int = 42,
                 test_years: int = 2):
        """
        Initialize temporal validator.
        
        Args:
            features_path: Path to temporal features CSV
            business_path: Path to business data CSV
            review_path: Path to review data CSV (for activity validation)
            output_path: Directory to save outputs
            random_state: Random seed
            test_years: Number of years to use for testing (default: 2)
        """
        self.features_path = Path(features_path)
        self.business_path = Path(business_path)
        self.review_path = Path(review_path)
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.random_state = random_state
        self.test_years = test_years
        
        # Create subdirectories
        self.plots_path = self.output_path / "plots"
        self.plots_path.mkdir(exist_ok=True)
        
        # Data containers
        self.features_df = None
        self.business_df = None
        self.tasks_df = None  # Features + labels + metadata
        
        # Results
        self.results = {}
        self.split_info = {}
        
    
    def load_data(self):
        """Load all required data."""
        
        # Load features
        self.features_df = pd.read_csv(self.features_path)
        
        # Check for required columns
        required_cols = ['business_id', '_cutoff_date', '_prediction_year']
        missing = [c for c in required_cols if c not in self.features_df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        # Convert dates
        self.features_df['_cutoff_date'] = pd.to_datetime(self.features_df['_cutoff_date'])
        
        # Load business data (for ground truth labels)
        self.business_df = pd.read_csv(self.business_path)
        
        # Check is_open column
        if 'is_open' not in self.business_df.columns:
            raise ValueError("Business data must have 'is_open' column for ground truth labels")
        
    
    def generate_labels_leakage_free(self):
        """
        Generate labels using GROUND TRUTH (is_open status).
        
        This is the KEY FIX for label leakage:
        - OLD: Inferred labels from review patterns (LEAKAGE!)
        - NEW: Use final is_open status directly (NO LEAKAGE!)
        
        The prediction task becomes:
        "Given historical data up to cutoff_date, predict if business 
        will still be open at the dataset end date."
        """
        
        
        # Check if features_df already has is_open column (and remove it to avoid conflict)
        if 'is_open' in self.features_df.columns:
            self.features_df = self.features_df.drop(columns=['is_open'])
        
        # Merge features with business ground truth
        business_labels = self.business_df[['business_id', 'is_open']].copy()
        
        # Merge
        self.tasks_df = self.features_df.merge(
            business_labels,
            on='business_id',
            how='left'
        )
        
        # Verify is_open column exists after merge
        if 'is_open' not in self.tasks_df.columns:
            # Check for renamed columns
            is_open_cols = [c for c in self.tasks_df.columns if 'is_open' in c]
            if is_open_cols:
                self.tasks_df['is_open'] = self.tasks_df[is_open_cols[-1]]
            else:
                raise ValueError("Could not find is_open column after merge")
        
        # Rename for consistency
        self.tasks_df['label'] = self.tasks_df['is_open']
        self.tasks_df['label_confidence'] = 1.0  # Ground truth = 100% confidence
        self.tasks_df['label_source'] = 'ground_truth'
        
        # Drop rows with missing labels
        initial_count = len(self.tasks_df)
        self.tasks_df = self.tasks_df.dropna(subset=['label'])
        
        
        # Label distribution (handle both int and float labels)
        self.tasks_df['label'] = self.tasks_df['label'].astype(int)
        label_counts = self.tasks_df['label'].value_counts()
        n_open = label_counts.get(1, 0)
        n_closed = label_counts.get(0, 0)
        total = len(self.tasks_df)
        
    
    def validate_and_filter(self):
        """Validate feature quality and filter invalid rows."""
        
        initial_count = len(self.tasks_df)
        
        # Get feature columns
        feature_cols = [c for c in self.tasks_df.columns 
                       if not c.startswith('_') and c not in 
                       ['business_id', 'label', 'label_confidence', 'label_source', 'is_open']]
        
        
        # Validate feature quality
        features_only = self.tasks_df[feature_cols]
        
        validated_features, validation_report = validate_feature_quality(
            features_only,
            max_missing_rate=0.1,
            check_inf=True
        )
        
        # Update tasks_df with validated features
        metadata_cols = [c for c in self.tasks_df.columns if c not in feature_cols]
        self.tasks_df = pd.concat([
            self.tasks_df[metadata_cols].reset_index(drop=True),
            validated_features.reset_index(drop=True)
        ], axis=1)
        
        
    
    def create_temporal_holdout_split(self):
        """
        Create proper temporal holdout split (V3 - Unified Configuration).
        
        CRITICAL (V3):
        - Uses SPLIT_CONFIG from config.py for train/test years
        - Ensures consistency with ALL subsequent phases (5-9)
        
        This is a TRUE temporal prediction task: train on past, test on future.
        """
        
        available_years = sorted(self.tasks_df['_prediction_year'].unique())
        
        # Use SPLIT_CONFIG from config.py for consistent splits across ALL phases
        train_years = SPLIT_CONFIG['train_years']
        test_years_list = SPLIT_CONFIG['test_years']
        
        
        # Create masks
        train_mask = self.tasks_df['_prediction_year'].isin(train_years)
        test_mask = self.tasks_df['_prediction_year'].isin(test_years_list)
        
        train_indices = np.where(train_mask)[0]
        test_indices = np.where(test_mask)[0]
        
        
        # Store split info
        self.split_info = {
            'train_years': list(train_years),
            'test_years': list(test_years_list),
            'train_size': len(train_indices),
            'test_size': len(test_indices),
            'split_type': 'temporal_holdout'
        }
        
        # Check class distribution
        y_train = self.tasks_df.iloc[train_indices]['label'].values
        y_test = self.tasks_df.iloc[test_indices]['label'].values
        
        
        
        return train_indices, test_indices
    
    def train_and_evaluate(self):
        """
        Train and evaluate models with proper temporal holdout split.
        """
        
        # Prepare features and target
        feature_cols = [c for c in self.tasks_df.columns 
                       if not c.startswith('_') and c not in 
                       ['business_id', 'label', 'label_confidence', 'label_source', 'is_open']]
        
        X = self.tasks_df[feature_cols].values
        y = self.tasks_df['label'].values
        
        
        # Create temporal holdout split
        train_indices, test_indices = self.create_temporal_holdout_split()
        
        X_train = X[train_indices]
        X_test = X[test_indices]
        y_train = y[train_indices]
        y_test = y[test_indices]
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Define models
        models = {
            'LogisticRegression': LogisticRegression(
                max_iter=1000,
                class_weight='balanced',
                random_state=self.random_state
            ),
            'DecisionTree': DecisionTreeClassifier(
                max_depth=10,
                min_samples_split=20,
                class_weight='balanced',
                random_state=self.random_state
            ),
            'RandomForest': RandomForestClassifier(
                n_estimators=100,
                max_depth=15,
                min_samples_split=20,
                class_weight='balanced',
                random_state=self.random_state,
                n_jobs=-1
            )
        }
        
        
        for name, model in models.items():
            
            # Train
            model.fit(X_train_scaled, y_train)
            
            # Predict
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            
            # Evaluate
            results = {
                'model_name': name,
                'roc_auc': roc_auc_score(y_test, y_pred_proba),
                'precision': precision_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred),
                'f1': f1_score(y_test, y_pred),
                'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
                'split_info': self.split_info
            }
            
            self.results[name] = results
            
        
    
    def generate_visualizations(self):
        """Generate comparison visualizations."""
        
        if not self.results:
            return
        
        # Model comparison bar chart
        fig, ax = plt.subplots(figsize=(10, 6))
        
        models = list(self.results.keys())
        metrics = ['roc_auc', 'precision', 'recall', 'f1']
        
        x = np.arange(len(models))
        width = 0.2
        
        for i, metric in enumerate(metrics):
            values = [self.results[m][metric] for m in models]
            ax.bar(x + i*width, values, width, label=metric.upper())
        
        ax.set_xlabel('Model')
        ax.set_ylabel('Score')
        ax.set_title('Model Performance (Temporal Holdout - Leakage Free)')
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim([0, 1.05])
        
        # Add split info to plot
        split_text = f"Train: {self.split_info.get('train_years', [])}\nTest: {self.split_info.get('test_years', [])}"
        ax.text(0.02, 0.98, split_text, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(self.plots_path / 'model_comparison_temporal.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_results_json(self):
        """Save results to JSON file."""
        # Save JSON results
        results_path = self.output_path / "temporal_validation_results.json"
        with open(results_path, 'w', encoding='utf-8') as f:
            # Convert numpy types to Python types for JSON
            # Helper function to convert numpy types
            def convert_to_native(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {k: convert_to_native(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_to_native(i) for i in obj]
                else:
                    return obj
            
            json_results = {}
            for name, metrics in self.results.items():
                json_results[name] = {
                    'roc_auc': float(metrics['roc_auc']),
                    'precision': float(metrics['precision']),
                    'recall': float(metrics['recall']),
                    'f1': float(metrics['f1']),
                    'confusion_matrix': convert_to_native(metrics['confusion_matrix']),
                    'split_info': convert_to_native(self.split_info)
                }
            json.dump(json_results, f, indent=2)
    
    def temporal_cross_validation(self, model, model_name: str, n_splits: int = 5):
        """
        Temporal CV: Train on past, test on future (multiple windows).
        
        Example with 5 splits on 2012-2020 data:
        - Fold 1: Train 2012-2016, Test 2017
        - Fold 2: Train 2012-2017, Test 2018  
        - Fold 3: Train 2012-2018, Test 2019
        - Fold 4: Train 2012-2019, Test 2020
        
        Args:
            model: Sklearn-compatible model
            model_name: Name for logging
            n_splits: Number of folds
            
        Returns:
            List of fold results with metrics
        """
        from sklearn.base import clone
        
        
        years = sorted(self.tasks_df['_prediction_year'].unique())
        
        if len(years) < n_splits + 1:
            n_splits = len(years) - 1
        
        cv_results = []
        
        for i in range(n_splits):
            # Progressive training window
            train_years = years[:-(n_splits-i)]
            test_year = years[-(n_splits-i)]
            
            
            # Create masks
            train_mask = self.tasks_df['_prediction_year'].isin(train_years)
            test_mask = self.tasks_df['_prediction_year'] == test_year
            
            X_train_fold = self.X[train_mask]
            y_train_fold = self.y[train_mask]
            X_test_fold = self.X[test_mask]
            y_test_fold = self.y[test_mask]
            
            
            # Clone model to avoid fitting the same instance
            model_fold = clone(model)
            
            # Train and evaluate
            model_fold.fit(X_train_fold, y_train_fold)
            y_pred_proba = model_fold.predict_proba(X_test_fold)[:, 1]
            y_pred = (y_pred_proba > 0.5).astype(int)
            
            # Compute metrics
            from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
            auc = roc_auc_score(y_test_fold, y_pred_proba)
            precision = precision_score(y_test_fold, y_pred)
            recall = recall_score(y_test_fold, y_pred)
            f1 = f1_score(y_test_fold, y_pred)
            
            cv_results.append({
                'fold': i+1,
                'train_years': list(train_years),
                'test_year': int(test_year),
                'train_size': len(X_train_fold),
                'test_size': len(X_test_fold),
                'roc_auc': float(auc),
                'precision': float(precision),
                'recall': float(recall),
                'f1': float(f1)
            })
            
        # Summary statistics
        aucs = [r['roc_auc'] for r in cv_results]
        precisions = [r['precision'] for r in cv_results]
        recalls = [r['recall'] for r in cv_results]
        f1s = [r['f1'] for r in cv_results]
        
        
        # Save results
        cv_file = self.output_path / f'temporal_cv_{model_name}.json'
        with open(cv_file, 'w') as f:
            json.dump({
                'model': model_name,
                'n_splits': n_splits,
                'fold_results': cv_results,
                'summary': {
                    'mean_auc': float(np.mean(aucs)),
                    'std_auc': float(np.std(aucs)),
                    'mean_precision': float(np.mean(precisions)),
                    'std_precision': float(np.std(precisions)),
                    'mean_recall': float(np.mean(recalls)),
                    'std_recall': float(np.std(recalls)),
                    'mean_f1': float(np.mean(f1s)),
                    'std_f1': float(np.std(f1s))
                }
            }, f, indent=2)
        
        
        return cv_results
    
    def run_pipeline(self, prediction_window_months: int = 12):
        """
        Execute complete temporal validation pipeline.
        
        Args:
            prediction_window_months: Not used in V2 (kept for API compatibility)
        """
        
        # Step 1: Load data
        self.load_data()
        
        # Step 2: Generate labels (LEAKAGE-FREE)
        self.generate_labels_leakage_free()
        
        # Step 3: Validate and filter
        self.validate_and_filter()
        
        # Step 3.5: Save labeled temporal features for downstream models
        
        labeled_path = self.features_path.with_name(
            f"{self.features_path.stem}_labeled_{prediction_window_months}m.csv"
        )
        self.tasks_df.to_csv(labeled_path, index=False)
        
        
        # Step 4: Train and evaluate
        self.train_and_evaluate()
        
        # Step 5: Temporal Cross-Validation (NEW - for robust evaluation)
        
        from sklearn.ensemble import RandomForestClassifier
        rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            class_weight='balanced',
            random_state=42
        )
        self.temporal_cross_validation(rf_model, 'RandomForest_CV', n_splits=4)
   
        # Step 6: Generate visualizations
        self.generate_visualizations()
        
        # Step 7: Save results JSON
        self.save_results_json()
        


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Temporal Validation Pipeline (V2 - Leakage-Free)')
    parser.add_argument('--features', type=str,
                       default='data/features/business_features_temporal.csv',
                       help='Path to temporal features')
    parser.add_argument('--test-years', type=int, default=2,
                       help='Number of years to use for testing (default: 2)')
    
    args = parser.parse_args()
    
    print("="*70)
    print("CS 412 RESEARCH PROJECT - TEMPORAL VALIDATION (V2)")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Features: {args.features}")
    print(f"  Test years: {args.test_years}")
    print("")
    
    validator = TemporalValidator(
        features_path=args.features,
        test_years=args.test_years
    )
    
    validator.run_pipeline()
    
    print("\n" + "="*70)
    print("COMPLETE!")
    print("="*70)


if __name__ == "__main__":
    main()
