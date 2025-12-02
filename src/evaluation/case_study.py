"""
Case Study Analysis (V3 - Unified Configuration)

This module provides in-depth analysis of specific prediction cases:
1. Select interesting cases (correct/incorrect predictions)
2. Analyze feature contributions for each case
3. Compare similar businesses with different outcomes
4. Optional: SHAP values for model interpretability
5. Generate detailed case study report with insights

CRITICAL (V3):
- Uses SPLIT_CONFIG from config.py for consistent train/test splits
- Uses the SAME split as baseline_models.py and advanced_models.py
- All results are directly comparable across phases

Types of cases analyzed:
- True Positives: Correctly predicted to stay open
- True Negatives: Correctly predicted to close
- False Positives: Predicted to stay open but closed
- False Negatives: Predicted to close but stayed open
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Import unified configuration
try:
    from config import SPLIT_CONFIG, RANDOM_STATE, CASE_STUDY_CONFIG
except ImportError:
    # Fallback defaults if config not available
    SPLIT_CONFIG = {
        'train_years': [2012, 2013, 2014, 2015, 2016, 2017, 2018],
        'test_years': [2019, 2020]
    }
    RANDOM_STATE = 42
    CASE_STUDY_CONFIG = {'n_cases_per_type': 5, 'use_shap': False}

# Try to import SHAP for interpretability
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("WARNING: SHAP not available. Install 'shap' for interpretability analysis.")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/case_study.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configure plotting
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class CaseStudyAnalyzer:
    """
    Comprehensive case study analysis for business success prediction.
    
    Analyzes specific prediction cases to understand:
    - Why certain predictions succeed or fail
    - Which features drive specific predictions
    - Patterns in misclassified cases
    - Insights for model improvement
    """
    
    def __init__(self,
                 data_path: str = "data/features/business_features_temporal.csv",
                 business_path: str = "data/processed/business_clean.csv",
                 model_path: Optional[str] = None,
                 output_path: str = "src/evaluation/case_study",
                 random_state: int = 42,
                 use_shap: bool = True):
        """
        Initialize case study analyzer.
        
        Args:
            data_path: Path to features CSV
            business_path: Path to business data CSV (for location info)
            model_path: Path to saved model (if None, train new model)
            output_path: Directory to save outputs
            random_state: Random seed
            use_shap: If True, use SHAP for interpretability
        """
        self.data_path = Path(data_path)
        self.business_path = Path(business_path)
        self.model_path = Path(model_path) if model_path else None
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.random_state = random_state
        self.use_shap = use_shap and SHAP_AVAILABLE
        
        # Create subdirectories
        self.plots_path = self.output_path / "plots"
        self.plots_path.mkdir(exist_ok=True)
        self.cases_path = self.output_path / "cases"
        self.cases_path.mkdir(exist_ok=True)
        
        # Data
        self.df = None
        self.business_df = None  # For location info
        self.model = None
        self.scaler = None
        self.feature_names = None
        
        # Predictions
        self.X_test = None
        self.y_test = None
        self.y_pred = None
        self.y_pred_proba = None
        self.test_indices = None
        
        # Case studies
        self.selected_cases = {}
        
        logger.info(f"Initialized CaseStudyAnalyzer")
        logger.info(f"  SHAP available: {self.use_shap}")
    
    def load_and_prepare_data(self):
        """Load data and train/load model."""
        logger.info("="*70)
        logger.info("LOADING DATA AND MODEL")
        logger.info("="*70)
        
        # Load features data
        self.df = pd.read_csv(self.data_path)
        logger.info(f"Loaded features: {self.df.shape}")
        
        # Load business data for location info
        if self.business_path.exists():
            self.business_df = pd.read_csv(self.business_path)
            logger.info(f"Loaded business data: {self.business_df.shape}")
            # Keep only necessary columns for location lookup
            location_cols = ['business_id', 'name', 'city', 'state', 'categories']
            available_cols = [c for c in location_cols if c in self.business_df.columns]
            self.business_df = self.business_df[available_cols]
        else:
            logger.warning(f"Business data not found at {self.business_path}, location info will be unavailable")
            self.business_df = None
        
        # Identify features
        metadata_cols = [c for c in self.df.columns if c.startswith('_')]
        metadata_cols.extend(['business_id', 'label', 'label_confidence', 'label_source', 'is_open'])
        
        self.feature_names = [c for c in self.df.columns if c not in metadata_cols]
        
        X = self.df[self.feature_names].values
        
        if 'label' in self.df.columns:
            y = self.df['label'].values
        elif 'is_open' in self.df.columns:
            y = self.df['is_open'].values
        else:
            raise ValueError("No target variable found")
        
        logger.info(f"Features: {len(self.feature_names)}")
        
        # Temporal HOLDOUT split (V3 - Unified Configuration)
        if '_prediction_year' in self.df.columns:
            logger.info("Using TEMPORAL HOLDOUT split (V3 - Unified Configuration)...")
            logger.info("Using SPLIT_CONFIG from config.py for consistency with baseline/advanced models")
            
            # Use SPLIT_CONFIG from config.py - SAME as baseline_models.py and advanced_models.py
            train_years = SPLIT_CONFIG['train_years']
            test_years = SPLIT_CONFIG['test_years']
            
            logger.info(f"Train years (from config): {train_years}")
            logger.info(f"Test years (from config): {test_years}")
            
            train_mask = self.df['_prediction_year'].isin(train_years)
            test_mask = self.df['_prediction_year'].isin(test_years)
            
            train_indices = np.where(train_mask)[0]
            test_indices = np.where(test_mask)[0]
            
            self.test_indices = np.array(test_indices)
            
            X_train = X[train_indices]
            X_test = X[test_indices]
            y_train = y[train_indices]
            y_test = y[test_indices]
        else:
            X_train, X_test, y_train, y_test, train_idx, test_idx = train_test_split(
                X, y, np.arange(len(X)),
                test_size=0.2,
                random_state=self.random_state,
                stratify=y
            )
            self.test_indices = test_idx
        
        # Scale
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        self.X_test = X_test_scaled
        self.y_test = y_test
        
        # Train or load model
        if self.model_path and self.model_path.exists():
            logger.info(f"Loading model from {self.model_path}...")
            import pickle
            with open(self.model_path, 'rb') as f:
                self.model = pickle.load(f)
        else:
            logger.info("Training new RandomForest model...")
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=15,
                min_samples_split=20,
                class_weight='balanced',
                random_state=self.random_state,
                n_jobs=-1
            )
            self.model.fit(X_train_scaled, y_train)
        
        # Make predictions
        self.y_pred = self.model.predict(self.X_test)
        self.y_pred_proba = self.model.predict_proba(self.X_test)[:, 1]
        
        # Performance
        from sklearn.metrics import roc_auc_score, f1_score
        auc = roc_auc_score(self.y_test, self.y_pred_proba)
        f1 = f1_score(self.y_test, self.y_pred)
        
        logger.info(f"\nModel Performance:")
        logger.info(f"  ROC-AUC: {auc:.4f}")
        logger.info(f"  F1: {f1:.4f}")
        
        # Initialize SHAP if enabled
        self.shap_values = None
        if self.use_shap:
            logger.info("\nInitializing SHAP explainer...")
            try:
                # Use TreeExplainer for RandomForest (faster)
                explainer = shap.TreeExplainer(self.model)
                # Compute SHAP values for test set (for class 1 = open)
                shap_values = explainer.shap_values(self.X_test)
                
                # For binary classification, shap_values is a list [class0, class1]
                if isinstance(shap_values, list):
                    self.shap_values = shap_values[1]  # Class 1 (open) SHAP values
                else:
                    self.shap_values = shap_values
                    
                logger.info(f"  SHAP values computed for {len(self.shap_values)} test samples")
            except Exception as e:
                logger.warning(f"SHAP computation failed: {e}")
                logger.warning("Falling back to deviation-based analysis")
                self.shap_values = None
        
        logger.info(f"\n{'='*70}\n")
    
    def select_interesting_cases(self, n_per_type: int = 5):
        """
        Select interesting cases for detailed analysis.
        
        Args:
            n_per_type: Number of cases to select per type (TP, TN, FP, FN)
        """
        logger.info("="*70)
        logger.info("SELECTING INTERESTING CASES")
        logger.info("="*70)
        
        # Classify predictions
        tp_mask = (self.y_test == 1) & (self.y_pred == 1)  # True Positive
        tn_mask = (self.y_test == 0) & (self.y_pred == 0)  # True Negative
        fp_mask = (self.y_test == 0) & (self.y_pred == 1)  # False Positive
        fn_mask = (self.y_test == 1) & (self.y_pred == 0)  # False Negative
        
        logger.info(f"True Positives: {tp_mask.sum():,}")
        logger.info(f"True Negatives: {tn_mask.sum():,}")
        logger.info(f"False Positives: {fp_mask.sum():,}")
        logger.info(f"False Negatives: {fn_mask.sum():,}")
        
        # Select cases with highest/lowest confidence for each type
        def select_cases(mask, case_type, n):
            indices = np.where(mask)[0]
            
            if len(indices) == 0:
                logger.warning(f"No {case_type} cases found")
                return []
            
            # Get probabilities for these cases
            probs = self.y_pred_proba[indices]
            
            # Select high confidence cases
            if len(indices) <= n:
                selected = indices
            else:
                # For TP/TN: select highest confidence
                # For FP/FN: select cases with medium confidence (interesting errors)
                if case_type in ['TP', 'TN']:
                    top_indices = np.argsort(probs)[-n:]
                else:
                    # For errors, select cases close to decision boundary (0.5)
                    distances = np.abs(probs - 0.5)
                    top_indices = np.argsort(distances)[:n]
                
                selected = indices[top_indices]
            
            return selected
        
        # Select cases
        self.selected_cases['TP'] = select_cases(tp_mask, 'TP', n_per_type)
        self.selected_cases['TN'] = select_cases(tn_mask, 'TN', n_per_type)
        self.selected_cases['FP'] = select_cases(fp_mask, 'FP', n_per_type)
        self.selected_cases['FN'] = select_cases(fn_mask, 'FN', n_per_type)
        
        logger.info(f"\nSelected cases:")
        for case_type, indices in self.selected_cases.items():
            logger.info(f"  {case_type}: {len(indices)}")
        
        logger.info(f"\n{'='*70}\n")
    
    def analyze_case(self, test_idx, case_type: str) -> Dict:
        """
        Analyze a single case in detail.
        
        Args:
            test_idx: Index in test set (can be numpy int or Python int)
            case_type: Type of case (TP, TN, FP, FN)
            
        Returns:
            Dict with case analysis
        """
        # Ensure test_idx is a Python int for consistent indexing
        test_idx = int(test_idx)
        
        # Get original data index
        original_idx = int(self.test_indices[test_idx])
        
        # Get business info from features
        business_info = self.df.iloc[original_idx]
        business_id = str(business_info.get('business_id', 'Unknown'))
        
        # Get location info from business_df
        city = 'Unknown'
        state = 'Unknown'
        business_name = 'Unknown'
        categories = 'Unknown'
        
        if self.business_df is not None and business_id != 'Unknown':
            business_lookup = self.business_df[self.business_df['business_id'] == business_id]
            if len(business_lookup) > 0:
                bus_row = business_lookup.iloc[0]
                city = str(bus_row.get('city', 'Unknown'))
                state = str(bus_row.get('state', 'Unknown'))
                business_name = str(bus_row.get('name', 'Unknown'))
                categories = str(bus_row.get('categories', 'Unknown'))
        
        # Get features
        features = self.X_test[test_idx]
        
        # Get prediction
        true_label = self.y_test[test_idx]
        pred_label = self.y_pred[test_idx]
        pred_proba = self.y_pred_proba[test_idx]
        
        # Feature importance for this case
        # Convert NumPy types to Python native types for JSON serialization
        feature_values_raw = dict(zip(self.feature_names, 
                                     self.df.iloc[original_idx][self.feature_names].values))
        feature_values = {}
        for k, v in feature_values_raw.items():
            # Convert NumPy types to Python native types
            if isinstance(v, (np.integer, np.int64, np.int32)):
                feature_values[k] = int(v)
            elif isinstance(v, (np.floating, np.float64, np.float32)):
                feature_values[k] = float(v)
            elif pd.isna(v):
                feature_values[k] = None
            else:
                feature_values[k] = v
        
        # Get top contributing features - use SHAP if available, otherwise deviation analysis
        top_features = {}
        
        if self.use_shap and hasattr(self, 'shap_values') and self.shap_values is not None:
            # Use SHAP values for case-specific importance
            try:
                case_shap = self.shap_values[test_idx, :]  # Explicit 2D indexing
            except (IndexError, TypeError):
                case_shap = self.shap_values[test_idx]  # Fallback to 1D
            top_feature_indices = np.argsort(np.abs(case_shap))[-10:][::-1]
            
            for idx in top_feature_indices:
                feat_name = self.feature_names[idx]
                value = feature_values[feat_name]
                shap_value = float(case_shap[idx])
                top_features[feat_name] = {
                    'value': value,
                    'shap_value': shap_value,
                    'importance': abs(shap_value),
                    'direction': 'positive' if shap_value > 0 else 'negative'
                }
        else:
            # Fallback: deviation-based case-specific analysis
            # Compare this case to the mean of same prediction type
            feature_importance = self.model.feature_importances_
            
            # Get average feature values for this case type's correct predictions
            same_type_mask = None
            if case_type == 'TP':
                same_type_mask = (self.y_test == 1) & (self.y_pred == 1)
            elif case_type == 'TN':
                same_type_mask = (self.y_test == 0) & (self.y_pred == 0)
            elif case_type == 'FP':
                # Compare to TN (correctly predicted closed) to see why this was mispredicted
                same_type_mask = (self.y_test == 0) & (self.y_pred == 0)
            else:  # FN
                # Compare to TP (correctly predicted open) to see why this was mispredicted
                same_type_mask = (self.y_test == 1) & (self.y_pred == 1)
            
            if same_type_mask is not None and same_type_mask.sum() > 0:
                reference_mean = self.X_test[same_type_mask].mean(axis=0)
                case_features = self.X_test[test_idx, :]  # Ensure proper 2D indexing
                deviations = case_features - reference_mean
                
                # Weight deviations by global feature importance
                weighted_deviations = np.abs(deviations) * feature_importance
                top_feature_indices = np.argsort(weighted_deviations)[-10:][::-1]
                
                for idx in top_feature_indices:
                    feat_name = self.feature_names[idx]
                    value = feature_values[feat_name]
                    deviation = float(deviations[idx])
                    top_features[feat_name] = {
                        'value': value,
                        'importance': float(feature_importance[idx]),
                        'deviation_from_reference': deviation,
                        'direction': 'higher' if deviation > 0 else 'lower'
                    }
            else:
                # Fallback to global importance
                top_feature_indices = np.argsort(feature_importance)[-10:][::-1]
                for idx in top_feature_indices:
                    feat_name = self.feature_names[idx]
                    value = feature_values[feat_name]
                    importance = float(feature_importance[idx])
                    top_features[feat_name] = {
                        'value': value,
                        'importance': importance
                    }
        
        # Convert prediction_year if it's a NumPy type
        prediction_year = business_info.get('_prediction_year', 'Unknown')
        if isinstance(prediction_year, (np.integer, np.int64, np.int32)):
            prediction_year = int(prediction_year)
        
        analysis = {
            'case_type': case_type,
            'business_id': business_id,
            'business_name': business_name,
            'city': city,
            'state': state,
            'categories': categories,
            'prediction_year': prediction_year,
            'true_label': int(true_label),
            'predicted_label': int(pred_label),
            'prediction_probability': float(pred_proba),
            'top_features': top_features,
            'feature_values': feature_values
        }
        
        return analysis
    
    def generate_case_reports(self):
        """Generate detailed reports for selected cases."""
        logger.info("="*70)
        logger.info("GENERATING CASE REPORTS")
        logger.info("="*70)
        
        all_analyses = {}
        
        for case_type, indices in self.selected_cases.items():
            logger.info(f"\nAnalyzing {case_type} cases...")
            
            case_analyses = []
            
            for test_idx in indices:
                analysis = self.analyze_case(test_idx, case_type)
                case_analyses.append(analysis)
            
            all_analyses[case_type] = case_analyses
            
            # Save individual case reports
            case_file = self.cases_path / f"{case_type}_cases.json"
            with open(case_file, 'w') as f:
                json.dump(case_analyses, f, indent=2)
            
            logger.info(f"  Saved {len(case_analyses)} cases to {case_file}")
        
        logger.info(f"\n{'='*70}\n")
        
        return all_analyses
    
    def generate_visualizations(self, all_analyses: Dict):
        """Generate case study visualizations."""
        logger.info("="*70)
        logger.info("GENERATING VISUALIZATIONS")
        logger.info("="*70)
        
        # Feature importance comparison across case types
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        case_types = ['TP', 'TN', 'FP', 'FN']
        
        for idx, case_type in enumerate(case_types):
            if case_type not in all_analyses or len(all_analyses[case_type]) == 0:
                continue
            
            ax = axes[idx]
            
            # Aggregate feature importance across cases of this type
            feature_importance_sum = {}
            
            for case in all_analyses[case_type]:
                for feat, info in case['top_features'].items():
                    if feat not in feature_importance_sum:
                        feature_importance_sum[feat] = 0
                    feature_importance_sum[feat] += info['importance']
            
            # Get top 10 features
            sorted_features = sorted(feature_importance_sum.items(), 
                                   key=lambda x: x[1], 
                                   reverse=True)[:10]
            
            features = [f[0] for f in sorted_features]
            importance = [f[1] for f in sorted_features]
            
            ax.barh(features, importance, color='steelblue')
            ax.set_xlabel('Cumulative Importance', fontsize=10)
            ax.set_title(f'{case_type} - Top Features', fontsize=11, fontweight='bold')
            ax.invert_yaxis()
            ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.plots_path / 'case_feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("[OK] Saved: case_feature_importance.png")
        
        logger.info(f"\n{'='*70}\n")
    
    def analyze_error_feature_distribution(self):
        """
        Quantitative analysis: Which features differ significantly in error cases?
        
        Uses Welch's t-test to identify features that are significantly different
        in FP/FN cases compared to overall distribution.
        
        Returns:
            Dict with significant features for each error type
        """
        from scipy import stats
        
        logger.info("\n" + "="*70)
        logger.info("QUANTITATIVE ERROR ANALYSIS")
        logger.info("="*70)
        
        # Get error masks
        fp_mask = (self.y_test == 0) & (self.y_pred == 1)  # False Positive
        fn_mask = (self.y_test == 1) & (self.y_pred == 0)  # False Negative
        
        error_types = {
            'False Positive': fp_mask,
            'False Negative': fn_mask
        }
        
        all_results = {}
        
        for error_type, mask in error_types.items():
            n_errors = mask.sum()
            if n_errors == 0:
                logger.info(f"\n{error_type}: No cases found")
                continue
                
            logger.info(f"\n{error_type} Analysis:")
            logger.info(f"  Count: {n_errors}")
            
            error_features = self.X_test[mask]
            overall_features = self.X_test
            
            significant_features = []
            
            # Iterate through feature indices
            for col_idx, col_name in enumerate(self.feature_names):
                # Extract column data
                error_col = error_features[:, col_idx]
                overall_col = overall_features[:, col_idx]
                
                # Skip columns with no variance
                if overall_col.std() == 0:
                    continue
                
                try:
                    # Welch's t-test (doesn't assume equal variance)
                    # Remove NaN values
                    error_col_clean = error_col[~np.isnan(error_col)]
                    overall_col_clean = overall_col[~np.isnan(overall_col)]
                    
                    if len(error_col_clean) == 0 or len(overall_col_clean) == 0:
                        continue
                    
                    t_stat, p_value = stats.ttest_ind(
                        error_col_clean,
                        overall_col_clean,
                        equal_var=False
                    )
                    
                    if p_value < 0.05 and not np.isnan(p_value):
                        error_mean = error_col.mean() if len(error_col_clean) > 0 else 0
                        overall_mean = overall_col.mean()
                        overall_std = overall_col.std()
                        
                        # Cohen's d (effect size)
                        effect_size = (error_mean - overall_mean) / (overall_std + 1e-10)
                        
                        significant_features.append({
                            'feature': col_name,
                            'error_mean': float(error_mean),
                            'overall_mean': float(overall_mean),
                            'difference': float(error_mean - overall_mean),
                            'effect_size': float(effect_size),
                            'p_value': float(p_value),
                            't_statistic': float(t_stat)
                        })
                except Exception as e:
                    # Skip features that cause errors
                    continue
            
            # Sort by absolute effect size
            significant_features.sort(key=lambda x: abs(x['effect_size']), reverse=True)
            
            # Log top 10
            logger.info(f"\n  Top 10 discriminative features:")
            for i, feat in enumerate(significant_features[:10], 1):
                logger.info(f"    {i}. {feat['feature']}")
                logger.info(f"       Error mean: {feat['error_mean']:.4f}, "
                           f"Overall mean: {feat['overall_mean']:.4f}")
                logger.info(f"       Effect size (Cohen's d): {feat['effect_size']:.4f}, "
                           f"p-value: {feat['p_value']:.4e}")
            
            all_results[error_type] = significant_features
        
        # Save results
        results_file = self.output_path / 'error_feature_analysis.json'
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        logger.info(f"\n[OK] Saved: {results_file}")
        
        return all_results
    
    def generate_report(self, all_analyses: Dict):
        """Generate comprehensive case study report."""
        logger.info("="*70)
        logger.info("GENERATING REPORT")
        logger.info("="*70)
        
        report_path = self.output_path / "case_study_report.md"
        
        report_lines = []
        report_lines.append("# Case Study Report")
        report_lines.append("")
        report_lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        report_lines.append("---")
        report_lines.append("")
        
        report_lines.append("## Executive Summary")
        report_lines.append("")
        report_lines.append("This report presents detailed analysis of selected prediction cases.")
        report_lines.append("Cases are selected from four categories:")
        report_lines.append("")
        report_lines.append("- **True Positives (TP)**: Correctly predicted to stay open")
        report_lines.append("- **True Negatives (TN)**: Correctly predicted to close")
        report_lines.append("- **False Positives (FP)**: Predicted to stay open but closed")
        report_lines.append("- **False Negatives (FN)**: Predicted to close but stayed open")
        report_lines.append("")
        
        # Analyze each case type
        for case_type in ['TP', 'TN', 'FP', 'FN']:
            if case_type not in all_analyses or len(all_analyses[case_type]) == 0:
                continue
            
            report_lines.append(f"## {case_type} Cases")
            report_lines.append("")
            
            # Description
            descriptions = {
                'TP': "Businesses correctly predicted to remain open.",
                'TN': "Businesses correctly predicted to close.",
                'FP': "Businesses predicted to stay open but actually closed (model errors).",
                'FN': "Businesses predicted to close but actually stayed open (model errors)."
            }
            report_lines.append(descriptions[case_type])
            report_lines.append("")
            
            # Analyze patterns
            cases = all_analyses[case_type]
            
            # Get common features
            all_top_features = {}
            for case in cases:
                for feat in case['top_features'].keys():
                    all_top_features[feat] = all_top_features.get(feat, 0) + 1
            
            common_features = sorted(all_top_features.items(), 
                                   key=lambda x: x[1], 
                                   reverse=True)[:5]
            
            report_lines.append("**Common Important Features:**")
            report_lines.append("")
            for feat, count in common_features:
                report_lines.append(f"- `{feat}` (in {count}/{len(cases)} cases)")
            report_lines.append("")
            
            # Sample case
            if len(cases) > 0:
                sample = cases[0]
                report_lines.append("**Sample Case:**")
                report_lines.append("")
                report_lines.append(f"- **Business ID**: {sample['business_id']}")
                report_lines.append(f"- **Business Name**: {sample.get('business_name', 'Unknown')}")
                report_lines.append(f"- **Location**: {sample['city']}, {sample['state']}")
                report_lines.append(f"- **Categories**: {sample.get('categories', 'Unknown')}")
                report_lines.append(f"- **Prediction Year**: {sample['prediction_year']}")
                report_lines.append(f"- **Prediction Probability**: {sample['prediction_probability']:.3f}")
                report_lines.append("")
                
                report_lines.append("Top Contributing Features:")
                report_lines.append("")
                for i, (feat, info) in enumerate(list(sample['top_features'].items())[:5], 1):
                    value = info.get('value', 0)
                    if isinstance(value, (int, float)):
                        value_str = f"{value:.3f}"
                    else:
                        value_str = str(value)
                    
                    # Add direction info if available (from SHAP or deviation analysis)
                    direction = info.get('direction', '')
                    if direction:
                        report_lines.append(f"{i}. **{feat}**: {value_str} ({direction})")
                    else:
                        report_lines.append(f"{i}. **{feat}**: {value_str}")
                report_lines.append("")
                
            # Add case-specific pattern analysis for errors (FP and FN)
            if case_type in ['FP', 'FN'] and len(cases) > 0:
                report_lines.append(f"**Why the model made this error ({case_type}):**")
                report_lines.append("")
                
                # Analyze feature patterns that distinguish this error type
                avg_deviation = {}
                for case in cases:
                    for feat, info in case['top_features'].items():
                        if 'deviation_from_reference' in info:
                            if feat not in avg_deviation:
                                avg_deviation[feat] = []
                            avg_deviation[feat].append(info['deviation_from_reference'])
                
                if avg_deviation:
                    report_lines.append("Key distinguishing features (vs correctly predicted cases):")
                    report_lines.append("")
                    try:
                        sorted_features = sorted(avg_deviation.items(), 
                                                key=lambda x: abs(float(np.mean(x[1]))), 
                                                reverse=True)[:3]
                        for feat, devs in sorted_features:
                            avg_dev = float(np.mean(devs))
                            direction = "higher" if avg_dev > 0 else "lower"
                            report_lines.append(f"- `{feat}`: {direction} than expected ({avg_dev:+.2f}σ)")
                    except Exception as e:
                        report_lines.append(f"- Error computing deviations: {e}")
                    report_lines.append("")
        
        report_lines.append("## Key Insights")
        report_lines.append("")
        
        # FP insights
        if 'FP' in all_analyses and len(all_analyses['FP']) > 0:
            report_lines.append("### False Positives (Prediction Errors)")
            report_lines.append("")
            report_lines.append("Common patterns in businesses predicted to stay open but closed:")
            report_lines.append("")
            report_lines.append("- May have had stable historical performance")
            report_lines.append("- Recent decline not captured by current features")
            report_lines.append("- External factors (location changes, competition) not modeled")
            report_lines.append("")
        
        # FN insights
        if 'FN' in all_analyses and len(all_analyses['FN']) > 0:
            report_lines.append("### False Negatives (Missed Survivors)")
            report_lines.append("")
            report_lines.append("Common patterns in businesses predicted to close but stayed open:")
            report_lines.append("")
            report_lines.append("- May have had temporary difficulties")
            report_lines.append("- Recovery signals not captured")
            report_lines.append("- Strong intangible factors (loyal customer base, unique offerings)")
            report_lines.append("")
        
        report_lines.append("## Recommendations")
        report_lines.append("")
        report_lines.append("Based on case study analysis:")
        report_lines.append("")
        report_lines.append("1. **Feature Engineering**: Add features capturing recent trends more accurately")
        report_lines.append("2. **External Data**: Consider incorporating location-based economic indicators")
        report_lines.append("3. **Temporal Dynamics**: Better model recovery patterns after temporary decline")
        report_lines.append("4. **Ensemble Methods**: Combine models with different strengths to reduce errors")
        report_lines.append("")
        
        report_lines.append("---")
        report_lines.append("")
        report_lines.append("*Report generated by CS 412 Research Project case study pipeline*")
        
        # Write report
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        logger.info(f"[OK] Saved report: {report_path}")
        logger.info(f"\n{'='*70}\n")
    
    def run_pipeline(self, n_cases_per_type: int = 5):
        """
        Execute complete case study pipeline.
        
        Args:
            n_cases_per_type: Number of cases to analyze per type
        """
        logger.info("="*70)
        logger.info("CS 412 RESEARCH PROJECT - CASE STUDY")
        logger.info("="*70)
        logger.info("")
        
        # Step 1: Load data and model
        self.load_and_prepare_data()
        
        # Step 2: Select interesting cases
        self.select_interesting_cases(n_per_type=n_cases_per_type)
        
        # Step 3: Generate case reports
        all_analyses = self.generate_case_reports()
        
        # Step 4: Generate visualizations
        self.generate_visualizations(all_analyses)
        
        # Step 5: Quantitative error analysis (NEW)
        self.analyze_error_feature_distribution()
        
        # Step 6: Generate report
        self.generate_report(all_analyses)
        
        logger.info("\n" + "="*70)
        logger.info("CASE STUDY COMPLETE!")
        logger.info("="*70)
        logger.info(f"\nOutputs saved to: {self.output_path}")
        logger.info("")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Case Study Pipeline')
    parser.add_argument('--data', type=str,
                       default='data/features/business_features_temporal_labeled_12m.csv',
                       help='Path to features CSV (recommend: labeled temporal features)')
    parser.add_argument('--model', type=str, default=None,
                       help='Path to saved model (optional)')
    parser.add_argument('--n-cases', type=int, default=5,
                       help='Number of cases per type to analyze')
    parser.add_argument('--use-shap', action='store_true',
                       help='Use SHAP for interpretability')
    
    args = parser.parse_args()
    
    print("="*70)
    print("CS 412 RESEARCH PROJECT - CASE STUDY")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Data: {args.data}")
    print(f"  Model: {args.model or 'Train new model'}")
    print(f"  Cases per type: {args.n_cases}")
    print(f"  Use SHAP: {args.use_shap}")
    print("")
    
    analyzer = CaseStudyAnalyzer(
        data_path=args.data,
        model_path=args.model,
        use_shap=args.use_shap
    )
    
    analyzer.run_pipeline(n_cases_per_type=args.n_cases)
    
    print("\n" + "="*70)
    print("COMPLETE!")
    print("="*70)


if __name__ == "__main__":
    main()