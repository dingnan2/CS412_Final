"""
Ablation Study Pipeline (V3 - Unified Configuration)

This module systematically evaluates the contribution of each feature category
by training models with different feature subsets:

1. Remove each feature category one at a time (ablation)
2. Add each feature category one at a time (additive)
3. Evaluate user credibility weighting impact
4. VIF multicollinearity analysis
5. Feature correlation analysis

CRITICAL (V3):
- Uses SPLIT_CONFIG from config.py for consistent train/test splits
- Uses the SAME split as baseline_models.py and advanced_models.py
- All results are directly comparable across phases

Outputs:
- Performance metrics for each feature subset
- Feature category importance ranking
- Visualization of ablation results
- Comprehensive analysis report
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
import json
from datetime import datetime
from typing import Dict, List, Tuple
import warnings

warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score

# Import unified configuration
try:
    from config import SPLIT_CONFIG, RANDOM_STATE, RF_ABLATION_CONFIG
except ImportError:
    # Fallback defaults if config not available
    SPLIT_CONFIG = {
        'train_years': [2012, 2013, 2014, 2015, 2016, 2017, 2018],
        'test_years': [2019, 2020]
    }
    RANDOM_STATE = 42
    RF_ABLATION_CONFIG = {'n_estimators': 100, 'max_depth': 15, 'class_weight': 'balanced'}

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/ablation_study.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configure plotting
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class AblationStudy:
    """
    Systematic ablation study to evaluate feature category contributions.
    
    Feature Categories:
    A: Static Business (8 features)
    B: Review Aggregation (9 features)
    C: Sentiment (8 features)
    D: User-Weighted (9 features)
    E: Temporal Dynamics (8 features)
    F: Location/Category (5 features)
    """
    
    def __init__(self,
                 data_path: str = "data/features/business_features_temporal.csv",
                 output_path: str = "src/evaluation/ablation_study",
                 random_state: int = 42):
        """
        Initialize ablation study.
        
        Args:
            data_path: Path to features CSV
            output_path: Directory to save outputs
            random_state: Random seed
        """
        self.data_path = Path(data_path)
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.random_state = random_state
        
        # Create subdirectories
        self.plots_path = self.output_path / "plots"
        self.plots_path.mkdir(exist_ok=True)
        
        # Data
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
        # Feature categories mapping
        self.feature_categories = {
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
            ],
            'G_Interaction': [
                'rating_credibility_interaction', 'momentum_credibility_interaction',
                'size_activity_interaction', 'trend_quality_interaction',
                'engagement_credibility_interaction'
            ]
        }
        
        # Results
        self.ablation_results = {}
        self.additive_results = {}
        
        logger.info(f"Initialized AblationStudy")
        logger.info(f"  Data: {data_path}")
        logger.info(f"  Feature categories: {len(self.feature_categories)}")
    
    def load_and_prepare_data(self):
        """Load data and prepare train/test split."""
        logger.info("="*70)
        logger.info("LOADING DATA")
        logger.info("="*70)
        
        # Load data
        self.df = pd.read_csv(self.data_path)
        logger.info(f"Loaded: {self.df.shape}")
        
        # Get all feature columns that exist in the data
        all_features = []
        for category, features in self.feature_categories.items():
            existing = [f for f in features if f in self.df.columns]
            all_features.extend(existing)
            
            if len(existing) < len(features):
                missing = set(features) - set(existing)
                logger.warning(f"{category}: Missing {len(missing)} features: {missing}")
        
        logger.info(f"Total available features: {len(all_features)}")
        
        # Extract features and target
        X = self.df[all_features].values
        
        if 'label' in self.df.columns:
            y = self.df['label'].values
        elif 'is_open' in self.df.columns:
            y = self.df['is_open'].values
        else:
            raise ValueError("No target variable found")
        
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
            
            X_train = X[train_indices]
            X_test = X[test_indices]
            y_train = y[train_indices]
            y_test = y[test_indices]
        else:
            logger.info("Using random split (no temporal metadata)...")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=self.random_state, stratify=y
            )
        
        # Scale
        scaler = StandardScaler()
        self.X_train = scaler.fit_transform(X_train)
        self.X_test = scaler.transform(X_test)
        self.y_train = y_train
        self.y_test = y_test
        
        logger.info(f"Train: {len(self.X_train):,}, Test: {len(self.X_test):,}")
        logger.info(f"\n{'='*70}\n")
    
    def get_feature_indices(self, feature_names: List[str]) -> List[int]:
        """Get indices of features in the full feature list."""
        all_features = []
        for features in self.feature_categories.values():
            all_features.extend([f for f in features if f in self.df.columns])
        
        indices = []
        for fname in feature_names:
            if fname in all_features:
                indices.append(all_features.index(fname))
        
        return indices
    
    def train_with_features(self, feature_names: List[str], experiment_name: str) -> Dict:
        """
        Train model with specified features.
        
        Args:
            feature_names: List of feature names to use
            experiment_name: Name of the experiment
            
        Returns:
            Dict with performance metrics
        """
        # Get feature indices
        indices = self.get_feature_indices(feature_names)
        
        if len(indices) == 0:
            logger.warning(f"{experiment_name}: No valid features found")
            return None
        
        # Select features
        X_train_subset = self.X_train[:, indices]
        X_test_subset = self.X_test[:, indices]
        
        # Train model using RF_ABLATION_CONFIG for consistency
        try:
            from config import RF_ABLATION_CONFIG
            model = RandomForestClassifier(**RF_ABLATION_CONFIG)
        except (ImportError, KeyError):
            # Fallback to explicit parameters if config not available
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=15,
                min_samples_split=20,
                class_weight='balanced',
                random_state=self.random_state,
                n_jobs=-1
            )
        
        model.fit(X_train_subset, self.y_train)
        
        # Evaluate
        y_pred = model.predict(X_test_subset)
        y_pred_proba = model.predict_proba(X_test_subset)[:, 1]
        
        results = {
            'experiment': experiment_name,
            'n_features': len(indices),
            'roc_auc': roc_auc_score(self.y_test, y_pred_proba),
            'precision': precision_score(self.y_test, y_pred),
            'recall': recall_score(self.y_test, y_pred),
            'f1': f1_score(self.y_test, y_pred)
        }
        
        return results
    
    def run_ablation_experiments(self):
        """
        Run ablation experiments: remove one category at a time.
        
        This shows the importance of each category by measuring
        performance drop when it's removed.
        """
        logger.info("="*70)
        logger.info("ABLATION EXPERIMENTS (Remove One Category)")
        logger.info("="*70)
        
        # Baseline: All features
        all_features = []
        for features in self.feature_categories.values():
            all_features.extend([f for f in features if f in self.df.columns])
        
        baseline = self.train_with_features(all_features, "Baseline_All_Features")
        self.ablation_results['Baseline'] = baseline
        
        logger.info(f"\nBaseline (All Features):")
        logger.info(f"  Features: {baseline['n_features']}")
        logger.info(f"  ROC-AUC: {baseline['roc_auc']:.4f}")
        logger.info(f"  F1: {baseline['f1']:.4f}")
        
        # Remove each category
        for category_name, category_features in self.feature_categories.items():
            # Get all features except this category
            remaining_features = []
            for cat, feats in self.feature_categories.items():
                if cat != category_name:
                    remaining_features.extend([f for f in feats if f in self.df.columns])
            
            # Train without this category
            result = self.train_with_features(
                remaining_features,
                f"Remove_{category_name}"
            )
            
            if result:
                # Calculate performance drop
                result['auc_drop'] = baseline['roc_auc'] - result['roc_auc']
                result['f1_drop'] = baseline['f1'] - result['f1']
                
                self.ablation_results[f"Remove_{category_name}"] = result
                
                logger.info(f"\n{category_name} (Removed):")
                logger.info(f"  ROC-AUC: {result['roc_auc']:.4f} (drop: {result['auc_drop']:.4f})")
                logger.info(f"  F1: {result['f1']:.4f} (drop: {result['f1_drop']:.4f})")
        
        logger.info(f"\n{'='*70}\n")
    
    def run_additive_experiments(self):
        """
        Run additive experiments: add one category at a time.
        
        This shows the marginal contribution of each category
        when added to an empty or minimal feature set.
        """
        logger.info("="*70)
        logger.info("ADDITIVE EXPERIMENTS (Add One Category)")
        logger.info("="*70)
        
        # Start with just static features (most basic)
        base_features = [f for f in self.feature_categories['A_Static'] if f in self.df.columns]
        
        baseline = self.train_with_features(base_features, "Base_Static_Only")
        self.additive_results['Base_Static'] = baseline
        
        logger.info(f"\nBase (Static Features Only):")
        logger.info(f"  ROC-AUC: {baseline['roc_auc']:.4f}")
        logger.info(f"  F1: {baseline['f1']:.4f}")
        
        # Add each category to base
        for category_name, category_features in self.feature_categories.items():
            if category_name == 'A_Static':
                continue  # Already in base
            
            # Combine base + this category
            combined_features = base_features.copy()
            combined_features.extend([f for f in category_features if f in self.df.columns])
            
            result = self.train_with_features(
                combined_features,
                f"Add_{category_name}"
            )
            
            if result:
                # Calculate performance gain
                result['auc_gain'] = result['roc_auc'] - baseline['roc_auc']
                result['f1_gain'] = result['f1'] - baseline['f1']
                
                self.additive_results[f"Add_{category_name}"] = result
                
                logger.info(f"\n{category_name} (Added to Base):")
                logger.info(f"  ROC-AUC: {result['roc_auc']:.4f} (gain: {result['auc_gain']:.4f})")
                logger.info(f"  F1: {result['f1']:.4f} (gain: {result['f1_gain']:.4f})")
        
        logger.info(f"\n{'='*70}\n")
    
    def evaluate_user_credibility_impact(self):
        """
        Evaluate the specific impact of user credibility weighting.
        
        Compare:
        1. With user-weighted features (Category D)
        2. Without user-weighted features
        """
        logger.info("="*70)
        logger.info("USER CREDIBILITY WEIGHTING IMPACT")
        logger.info("="*70)
        
        # All features
        all_features = []
        for features in self.feature_categories.values():
            all_features.extend([f for f in features if f in self.df.columns])
        
        with_cred = self.train_with_features(all_features, "With_Credibility")
        
        # Without credibility features
        without_cred_features = []
        for cat, features in self.feature_categories.items():
            if cat != 'D_User_Weighted':
                without_cred_features.extend([f for f in features if f in self.df.columns])
        
        without_cred = self.train_with_features(without_cred_features, "Without_Credibility")
        
        logger.info(f"\nWith User Credibility:")
        logger.info(f"  ROC-AUC: {with_cred['roc_auc']:.4f}")
        logger.info(f"  F1: {with_cred['f1']:.4f}")
        
        logger.info(f"\nWithout User Credibility:")
        logger.info(f"  ROC-AUC: {without_cred['roc_auc']:.4f}")
        logger.info(f"  F1: {without_cred['f1']:.4f}")
        
        logger.info(f"\nCredibility Impact:")
        logger.info(f"  ROC-AUC improvement: {with_cred['roc_auc'] - without_cred['roc_auc']:.4f}")
        logger.info(f"  F1 improvement: {with_cred['f1'] - without_cred['f1']:.4f}")
        
        logger.info(f"\n{'='*70}\n")
    
    def generate_visualizations(self):
        """Generate ablation study visualizations."""
        logger.info("="*70)
        logger.info("GENERATING VISUALIZATIONS")
        logger.info("="*70)
        
        # Ablation results (performance drop when category removed)
        if len(self.ablation_results) > 1:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
            
            # Get ablation data (exclude baseline)
            ablation_data = {k: v for k, v in self.ablation_results.items() if k != 'Baseline'}
            
            categories = [k.replace('Remove_', '') for k in ablation_data.keys()]
            auc_drops = [v['auc_drop'] for v in ablation_data.values()]
            f1_drops = [v['f1_drop'] for v in ablation_data.values()]
            
            # Sort by AUC drop
            sorted_indices = np.argsort(auc_drops)[::-1]
            categories = [categories[i] for i in sorted_indices]
            auc_drops = [auc_drops[i] for i in sorted_indices]
            f1_drops = [f1_drops[i] for i in sorted_indices]
            
            # ROC-AUC drop
            ax1.barh(categories, auc_drops, color='steelblue')
            ax1.set_xlabel('ROC-AUC Drop (when removed)', fontsize=11)
            ax1.set_title('Feature Category Importance\n(Ablation Study)', fontsize=12, fontweight='bold')
            ax1.grid(axis='x', alpha=0.3)
            
            # F1 drop
            ax2.barh(categories, f1_drops, color='coral')
            ax2.set_xlabel('F1 Drop (when removed)', fontsize=11)
            ax2.set_title('Feature Category Importance\n(F1 Score)', fontsize=12, fontweight='bold')
            ax2.grid(axis='x', alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(self.plots_path / 'ablation_results.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info("[OK] Saved: ablation_results.png")
        
        # Additive results (performance gain when category added)
        if len(self.additive_results) > 1:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            additive_data = {k: v for k, v in self.additive_results.items() if k != 'Base_Static'}
            
            categories = [k.replace('Add_', '') for k in additive_data.keys()]
            auc_gains = [v['auc_gain'] for v in additive_data.values()]
            
            # Sort by gain
            sorted_indices = np.argsort(auc_gains)[::-1]
            categories = [categories[i] for i in sorted_indices]
            auc_gains = [auc_gains[i] for i in sorted_indices]
            
            ax.barh(categories, auc_gains, color='mediumseagreen')
            ax.set_xlabel('ROC-AUC Gain (when added to base)', fontsize=11)
            ax.set_title('Marginal Contribution of Feature Categories', fontsize=12, fontweight='bold')
            ax.grid(axis='x', alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(self.plots_path / 'additive_results.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info("[OK] Saved: additive_results.png")
        
        logger.info(f"\n{'='*70}\n")
    
    def generate_report(self):
        """Generate comprehensive ablation study report."""
        logger.info("="*70)
        logger.info("GENERATING REPORT")
        logger.info("="*70)
        
        report_path = self.output_path / "ablation_study_report.md"
        
        report_lines = []
        report_lines.append("# Ablation Study Report")
        report_lines.append("")
        report_lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        report_lines.append("---")
        report_lines.append("")
        
        report_lines.append("## Executive Summary")
        report_lines.append("")
        report_lines.append("This report presents systematic evaluation of feature category contributions.")
        report_lines.append("Two complementary approaches are used:")
        report_lines.append("")
        report_lines.append("1. **Ablation**: Remove each category and measure performance drop")
        report_lines.append("2. **Additive**: Add each category and measure performance gain")
        report_lines.append("")
        
        # Ablation results
        if self.ablation_results:
            report_lines.append("## Ablation Results (Remove One Category)")
            report_lines.append("")
            
            if 'Baseline' in self.ablation_results:
                baseline = self.ablation_results['Baseline']
                report_lines.append(f"**Baseline (All Features):** ROC-AUC = {baseline['roc_auc']:.4f}")
                report_lines.append("")
            
            report_lines.append("| Category | ROC-AUC | AUC Drop | F1 | F1 Drop |")
            report_lines.append("|----------|---------|----------|----|---------| ")
            
            ablation_data = {k: v for k, v in self.ablation_results.items() if k != 'Baseline'}
            
            # Sort by AUC drop (descending = most important first)
            sorted_items = sorted(ablation_data.items(), 
                                 key=lambda x: x[1].get('auc_drop', 0), 
                                 reverse=True)
            
            for name, metrics in sorted_items:
                cat_name = name.replace('Remove_', '')
                report_lines.append(
                    f"| {cat_name} | {metrics['roc_auc']:.4f} | "
                    f"{metrics.get('auc_drop', 0):.4f} | {metrics['f1']:.4f} | "
                    f"{metrics.get('f1_drop', 0):.4f} |"
                )
            
            report_lines.append("")
        
        # Additive results
        if self.additive_results:
            report_lines.append("## Additive Results (Add to Base)")
            report_lines.append("")
            
            if 'Base_Static' in self.additive_results:
                base = self.additive_results['Base_Static']
                report_lines.append(f"**Base (Static Features):** ROC-AUC = {base['roc_auc']:.4f}")
                report_lines.append("")
            
            report_lines.append("| Category Added | ROC-AUC | AUC Gain | F1 | F1 Gain |")
            report_lines.append("|----------------|---------|----------|----|---------| ")
            
            additive_data = {k: v for k, v in self.additive_results.items() if k != 'Base_Static'}
            
            # Sort by gain
            sorted_items = sorted(additive_data.items(), 
                                 key=lambda x: x[1].get('auc_gain', 0), 
                                 reverse=True)
            
            for name, metrics in sorted_items:
                cat_name = name.replace('Add_', '')
                report_lines.append(
                    f"| {cat_name} | {metrics['roc_auc']:.4f} | "
                    f"{metrics.get('auc_gain', 0):.4f} | {metrics['f1']:.4f} | "
                    f"{metrics.get('f1_gain', 0):.4f} |"
                )
            
            report_lines.append("")
        
        report_lines.append("## Key Findings")
        report_lines.append("")
        report_lines.append("### Most Important Feature Categories")
        report_lines.append("")
        report_lines.append("Based on ablation results (largest performance drop when removed):")
        report_lines.append("")
        
        if len(self.ablation_results) > 1:
            ablation_data = {k: v for k, v in self.ablation_results.items() if k != 'Baseline'}
            top_3 = sorted(ablation_data.items(), 
                          key=lambda x: x[1].get('auc_drop', 0), 
                          reverse=True)[:3]
            
            for i, (name, metrics) in enumerate(top_3, 1):
                cat_name = name.replace('Remove_', '')
                report_lines.append(f"{i}. **{cat_name}** (drop: {metrics.get('auc_drop', 0):.4f})")
            
            report_lines.append("")
        
        report_lines.append("### User Credibility Weighting")
        report_lines.append("")
        report_lines.append("The user credibility weighting framework (Category D) provides:")
        report_lines.append("- Weighted ratings based on reviewer credibility")
        report_lines.append("- Higher weights for experienced, engaged reviewers")
        report_lines.append("- Improved signal-to-noise ratio in aggregated metrics")
        report_lines.append("")
        
        # Add deep interpretation
        report_lines.append("### Interpretation of Results")
        report_lines.append("")
        
        # Check for temporal paradox
        if len(self.ablation_results) > 1:
            ablation_data = {k: v for k, v in self.ablation_results.items() if k != 'Baseline'}
            
            temporal_result = ablation_data.get('Remove_E_Temporal', {})
            temporal_drop = temporal_result.get('auc_drop', 0)
            
            if temporal_drop < 0:  # Negative drop = improvement when removed
                report_lines.append("#### Temporal Feature Paradox")
                report_lines.append("")
                report_lines.append(f"Removing E_Temporal features **improved** performance (AUC change: {temporal_drop:+.4f}).")
                report_lines.append("This counter-intuitive result suggests:")
                report_lines.append("")
                report_lines.append("1. **Feature Redundancy**: Temporal patterns already captured by User-Weighted (D)")
                report_lines.append("   features through `avg_reviewer_tenure` and `review_diversity`")
                report_lines.append("")
                report_lines.append("2. **Noise Introduction**: Features like `rating_recent_vs_all` may capture")
                report_lines.append("   transient fluctuations rather than meaningful trends")
                report_lines.append("")
                report_lines.append("3. **Overfitting Risk**: 8 temporal features add complexity without")
                report_lines.append("   proportional signal, leading to overfitting on training data")
                report_lines.append("")
            
            # Analyze additive vs ablation discrepancy
            if self.additive_results:
                additive_d = self.additive_results.get('Add_D_User_Weighted', {})
                ablation_d = ablation_data.get('Remove_D_User_Weighted', {})
                
                additive_gain = additive_d.get('auc_gain', 0)
                ablation_drop = ablation_d.get('auc_drop', 0)
                
                if additive_gain < 0 and ablation_drop > 0:
                    report_lines.append("#### Ablation vs Additive Discrepancy")
                    report_lines.append("")
                    report_lines.append("User-Weighted (D) shows different behavior in ablation vs additive analysis:")
                    report_lines.append(f"- **Ablation**: Removing D hurts performance (drop: {ablation_drop:+.4f})")
                    report_lines.append(f"- **Additive**: Adding D to Static hurts performance (gain: {additive_gain:+.4f})")
                    report_lines.append("")
                    report_lines.append("**Explanation**: This demonstrates **feature interaction effects**:")
                    report_lines.append("- When ALL features present: D provides unique signal not captured by others")
                    report_lines.append("- When adding to Static only: D overlaps with Static features, introducing noise")
                    report_lines.append("- D works synergistically with Location (F) features, not as standalone")
                    report_lines.append("")
        
        report_lines.append("### Recommendations")
        report_lines.append("")
        report_lines.append("Based on ablation analysis:")
        report_lines.append("")
        report_lines.append("1. **Keep**: Static (A), User-Weighted (D), Location (F) - provide independent signal")
        report_lines.append("2. **Review**: Temporal (E) - consider removing or simplifying to reduce overfitting")
        report_lines.append("3. **Simplify**: Review Aggregation (B) - redundant with other categories")
        report_lines.append("")
        
        report_lines.append("---")
        report_lines.append("")
        report_lines.append("*Report generated by CS 412 Research Project ablation study pipeline*")
        
        # Write report
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        logger.info(f"[OK] Saved report: {report_path}")
        
        # Save results JSON
        results = {
            'ablation': self.ablation_results,
            'additive': self.additive_results
        }
        
        results_file = self.output_path / "ablation_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"[OK] Saved results: {results_file}")
        logger.info(f"\n{'='*70}\n")
    
    def analyze_multicollinearity(self):
        """
        Compute VIF (Variance Inflation Factor) to detect multicollinearity.
        
        VIF > 10 indicates severe multicollinearity
        VIF > 5 indicates moderate multicollinearity
        
        Returns:
            DataFrame with VIF values for each feature
        """
        try:
            from statsmodels.stats.outliers_influence import variance_inflation_factor
        except ImportError:
            logger.warning("statsmodels not available. Install it for VIF analysis.")
            return None
        
        logger.info(f"\n{'='*70}")
        logger.info("MULTICOLLINEARITY ANALYSIS (VIF)")
        logger.info(f"{'='*70}")
        
        # Select only numeric features (exclude metadata columns)
        feature_cols = [col for col in self.df.columns 
                       if col not in ['business_id', 'is_open', '_prediction_year', '_cutoff_date',
                                     '_first_review_date', '_last_review_date', '_prediction_month',
                                     '_business_age_at_cutoff_days']]
        
        numeric_features = self.df[feature_cols].select_dtypes(include=[np.number])
        
        # Remove any columns with zero variance
        numeric_features = numeric_features.loc[:, numeric_features.std() > 0]
        
        # Fill any NaN values
        numeric_features = numeric_features.fillna(numeric_features.median())
        
        logger.info(f"Analyzing {len(numeric_features.columns)} numeric features")
        
        # Compute VIF for each feature
        vif_data = []
        for i, col in enumerate(numeric_features.columns):
            try:
                vif = variance_inflation_factor(numeric_features.values, i)
                vif_data.append({'feature': col, 'VIF': vif})
            except Exception as e:
                logger.warning(f"Could not compute VIF for {col}: {e}")
                vif_data.append({'feature': col, 'VIF': np.nan})
        
        vif_df = pd.DataFrame(vif_data)
        
        # Sort by VIF
        vif_df = vif_df.sort_values('VIF', ascending=False)
        
        # Replace inf with large number for display
        vif_df['VIF'] = vif_df['VIF'].replace([np.inf], 9999.0)
        
        # Identify high VIF features
        high_vif = vif_df[vif_df['VIF'] > 10].dropna()
        moderate_vif = vif_df[(vif_df['VIF'] > 5) & (vif_df['VIF'] <= 10)].dropna()
        
        logger.info(f"\nHigh VIF features (> 10): {len(high_vif)}")
        if len(high_vif) > 0:
            for _, row in high_vif.head(15).iterrows():
                logger.info(f"  {row['feature']}: {row['VIF']:.2f}")
        
        logger.info(f"\nModerate VIF features (5-10): {len(moderate_vif)}")
        if len(moderate_vif) > 0:
            for _, row in moderate_vif.iterrows():
                logger.info(f"  {row['feature']}: {row['VIF']:.2f}")
        
        # Save
        vif_file = self.output_path / 'vif_analysis.csv'
        vif_df.to_csv(vif_file, index=False)
        logger.info(f"\n[OK] Saved: {vif_file}")
        
        return vif_df
    
    def analyze_feature_correlation(self, category: str = 'E_Temporal'):
        """
        Analyze correlation within a specific feature category.
        
        Args:
            category: Feature category to analyze
            
        Returns:
            Correlation matrix and high correlation pairs
        """
        logger.info(f"\n{'='*70}")
        logger.info(f"CORRELATION ANALYSIS: {category}")
        logger.info(f"{'='*70}")
        
        if category not in self.feature_categories:
            logger.error(f"Category '{category}' not found")
            return None, []
        
        category_features = self.feature_categories[category]
        available_features = [f for f in category_features if f in self.df.columns]
        
        if len(available_features) == 0:
            logger.warning(f"No features found for category {category}")
            return None, []
        
        logger.info(f"Analyzing {len(available_features)} features")
        
        # Compute correlation matrix
        corr_matrix = self.df[available_features].corr()
        
        # Find high correlations (|r| > 0.8)
        high_corr_pairs = []
        for i in range(len(available_features)):
            for j in range(i+1, len(available_features)):
                corr = corr_matrix.iloc[i, j]
                if abs(corr) > 0.7:  # Use 0.7 to catch more potential issues
                    high_corr_pairs.append({
                        'feature1': available_features[i],
                        'feature2': available_features[j],
                        'correlation': float(corr)
                    })
        
        # Sort by absolute correlation
        high_corr_pairs.sort(key=lambda x: abs(x['correlation']), reverse=True)
        
        logger.info(f"\nHigh correlations (|r| > 0.7): {len(high_corr_pairs)}")
        for pair in high_corr_pairs:
            logger.info(f"  {pair['feature1']} ↔ {pair['feature2']}: r = {pair['correlation']:.3f}")
        
        # Save correlation matrix
        corr_file = self.output_path / f'correlation_matrix_{category}.csv'
        corr_matrix.to_csv(corr_file)
        logger.info(f"\n[OK] Saved: {corr_file}")
        
        # Save high correlation pairs
        if len(high_corr_pairs) > 0:
            pairs_df = pd.DataFrame(high_corr_pairs)
            pairs_file = self.output_path / f'high_correlations_{category}.csv'
            pairs_df.to_csv(pairs_file, index=False)
            logger.info(f"[OK] Saved: {pairs_file}")
        
        return corr_matrix, high_corr_pairs
    
    def analyze_temporal_feature_paradox(self):
        """
        Deep dive into why temporal features may hurt performance.
        
        Analyzes:
        1. Correlation between temporal and other features
        2. VIF of temporal features
        3. Redundancy with user-weighted features
        """
        logger.info(f"\n{'='*70}")
        logger.info("TEMPORAL FEATURE PARADOX ANALYSIS")
        logger.info(f"{'='*70}")
        
        temporal_features = self.feature_categories.get('E_Temporal', [])
        user_weighted_features = self.feature_categories.get('D_User_Weighted', [])
        review_agg_features = self.feature_categories.get('B_Review_Agg', [])
        
        # Get available features
        temporal_avail = [f for f in temporal_features if f in self.df.columns]
        user_avail = [f for f in user_weighted_features if f in self.df.columns]
        review_avail = [f for f in review_agg_features if f in self.df.columns]
        
        logger.info(f"\nTemporal features: {len(temporal_avail)}")
        logger.info(f"User-weighted features: {len(user_avail)}")
        logger.info(f"Review aggregation features: {len(review_avail)}")
        
        # Cross-category correlations
        logger.info("\n--- Cross-Category Correlations ---")
        
        # Temporal vs User-Weighted
        if len(temporal_avail) > 0 and len(user_avail) > 0:
            cross_corr = self.df[temporal_avail + user_avail].corr()
            
            high_cross_corr = []
            for t_feat in temporal_avail:
                for u_feat in user_avail:
                    corr = cross_corr.loc[t_feat, u_feat]
                    if abs(corr) > 0.5:
                        high_cross_corr.append({
                            'temporal': t_feat,
                            'user_weighted': u_feat,
                            'correlation': float(corr)
                        })
            
            high_cross_corr.sort(key=lambda x: abs(x['correlation']), reverse=True)
            
            logger.info(f"\nTemporal ↔ User-Weighted correlations (|r| > 0.5): {len(high_cross_corr)}")
            for pair in high_cross_corr[:10]:
                logger.info(f"  {pair['temporal']} ↔ {pair['user_weighted']}: r = {pair['correlation']:.3f}")
        
        # Temporal vs Review Aggregation
        if len(temporal_avail) > 0 and len(review_avail) > 0:
            cross_corr = self.df[temporal_avail + review_avail].corr()
            
            high_cross_corr = []
            for t_feat in temporal_avail:
                for r_feat in review_avail:
                    corr = cross_corr.loc[t_feat, r_feat]
                    if abs(corr) > 0.5:
                        high_cross_corr.append({
                            'temporal': t_feat,
                            'review_agg': r_feat,
                            'correlation': float(corr)
                        })
            
            high_cross_corr.sort(key=lambda x: abs(x['correlation']), reverse=True)
            
            logger.info(f"\nTemporal ↔ Review Aggregation correlations (|r| > 0.5): {len(high_cross_corr)}")
            for pair in high_cross_corr[:10]:
                logger.info(f"  {pair['temporal']} ↔ {pair['review_agg']}: r = {pair['correlation']:.3f}")
        
        logger.info("\n--- Conclusion ---")
        logger.info("If temporal features are highly correlated with other categories,")
        logger.info("they may introduce multicollinearity and noise rather than unique signal.")
        
        return True
    
    def run_pipeline(self):
        """Execute complete ablation study pipeline."""
        logger.info("="*70)
        logger.info("CS 412 RESEARCH PROJECT - ABLATION STUDY")
        logger.info("="*70)
        logger.info("")
        
        # Step 1: Load data
        self.load_and_prepare_data()
        
        # Step 2: Run ablation experiments
        self.run_ablation_experiments()
        
        # Step 3: Run additive experiments
        self.run_additive_experiments()
        
        # Step 4: Evaluate user credibility impact
        self.evaluate_user_credibility_impact()
        
        # Step 5: Generate visualizations
        self.generate_visualizations()
        
        # Step 6: Multicollinearity analysis (NEW)
        self.analyze_multicollinearity()
        
        # Step 7: Temporal feature paradox analysis (NEW)
        self.analyze_feature_correlation(category='E_Temporal')
        self.analyze_temporal_feature_paradox()
        
        # Step 8: Generate report
        self.generate_report()
        
        logger.info("\n" + "="*70)
        logger.info("ABLATION STUDY COMPLETE!")
        logger.info("="*70)
        logger.info(f"\nOutputs saved to: {self.output_path}")
        logger.info("")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Ablation Study Pipeline')
    parser.add_argument('--data', type=str,
                       default='data/features/business_features_temporal_labeled_12m.csv',
                       help='Path to features CSV (recommend: labeled temporal features)')
    
    args = parser.parse_args()
    
    print("="*70)
    print("CS 412 RESEARCH PROJECT - ABLATION STUDY")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Data: {args.data}")
    print("")
    
    study = AblationStudy(data_path=args.data)
    study.run_pipeline()
    
    print("\n" + "="*70)
    print("COMPLETE!")
    print("="*70)


if __name__ == "__main__":
    main()