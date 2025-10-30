"""
Baseline Models: Train and evaluate baseline models for business success prediction.

This module implements:
1. Feature selection (correlation, variance, importance-based)
2. Three baseline models: Logistic Regression, Decision Tree, Random Forest
3. Class imbalance handling (SMOTE and class weights)
4. Comprehensive evaluation with visualizations
5. Model comparison and analysis

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

warnings.filterwarnings('ignore')

# Configure plotting
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('baseline_models.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class BaselineModelPipeline:
    """
    Complete baseline model training and evaluation pipeline.
    
    Pipeline Steps:
    1. Load and prepare data
    2. Feature selection
    3. Train-test split with stratification
    4. Handle class imbalance (SMOTE vs class weights)
    5. Train baseline models
    6. Evaluate and compare models
    7. Generate comprehensive report with visualizations
    """
    
    def __init__(self, 
                 data_path: str,
                 output_path: str = "src/models",
                 random_state: int = 42):
        """
        Initialize the baseline model pipeline.
        
        Args:
            data_path: Path to business_features_final.csv
            output_path: Directory to save outputs
            random_state: Random seed for reproducibility
        """
        self.data_path = Path(data_path)
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.random_state = random_state
        
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
        
        # Train-test splits
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
        # Scaled data
        self.scaler = None
        self.X_train_scaled = None
        self.X_test_scaled = None
        
        # Models
        self.models = {}
        self.results = {}
        
        # Feature selection tracking
        self.feature_selection_summary = {}
        
    def load_data(self):
        """Load feature-engineered dataset"""
        logger.info("="*70)
        logger.info("LOADING DATA")
        logger.info("="*70)
        
        self.df = pd.read_csv(self.data_path)
        logger.info(f"Loaded dataset: {self.df.shape}")
        logger.info(f"Columns: {list(self.df.columns)[:10]}...")
        
        # Separate features and target
        self.y = self.df['is_open']
        self.X = self.df.drop(columns=['business_id', 'is_open'])
        self.feature_names = list(self.X.columns)
        
        logger.info(f"Features: {self.X.shape[1]}")
        logger.info(f"Target distribution:")
        logger.info(f"  Open (1): {self.y.sum():,} ({self.y.mean()*100:.2f}%)")
        logger.info(f"  Closed (0): {(1-self.y).sum():,} ({(1-self.y.mean())*100:.2f}%)")
        
        # Check for missing values
        missing = self.X.isnull().sum()
        if missing.sum() > 0:
            logger.warning(f"Missing values detected:")
            logger.warning(missing[missing > 0])
            # Fill missing with median
            self.X = self.X.fillna(self.X.median())
            logger.info("Filled missing values with median")
        
        # Check for infinite values
        inf_mask = np.isinf(self.X.values)
        if inf_mask.any():
            logger.warning(f"Infinite values detected: {inf_mask.sum()}")
            self.X = self.X.replace([np.inf, -np.inf], np.nan)
            self.X = self.X.fillna(self.X.median())
            logger.info("Replaced infinite values with median")
        
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
    
    def prepare_train_test_split(self, test_size: float = 0.2):
        """
        Create stratified train-test split.
        
        Args:
            test_size: Proportion of test set
        """
        logger.info("="*70)
        logger.info("TRAIN-TEST SPLIT")
        logger.info("="*70)
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y,
            test_size=test_size,
            random_state=self.random_state,
            stratify=self.y
        )
        
        logger.info(f"Train set: {self.X_train.shape}")
        logger.info(f"Test set: {self.X_test.shape}")
        logger.info(f"\nTrain target distribution:")
        logger.info(f"  Open: {self.y_train.sum():,} ({self.y_train.mean()*100:.2f}%)")
        logger.info(f"  Closed: {(1-self.y_train).sum():,} ({(1-self.y_train.mean())*100:.2f}%)")
        logger.info(f"\nTest target distribution:")
        logger.info(f"  Open: {self.y_test.sum():,} ({self.y_test.mean()*100:.2f}%)")
        logger.info(f"  Closed: {(1-self.y_test).sum():,} ({(1-self.y_test.mean())*100:.2f}%)")
        
        # Scale features
        logger.info("\nScaling features (StandardScaler)...")
        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        # Save scaler
        with open(self.models_path / 'scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)
        logger.info(f"Saved: {self.models_path / 'scaler.pkl'}")
    
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
        X_train_smote, y_train_smote = smote.fit_resample(self.X_train_scaled, self.y_train)
        
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
            self.X_train_scaled, self.y_train,
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
            y_pred = model.predict(self.X_test_scaled)
            y_pred_proba = model.predict_proba(self.X_test_scaled)[:, 1]
            
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
        counts = self.y.value_counts()
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
        counts_train = self.y_train.value_counts()
        ax.bar(labels, [counts_train[1], counts_train[0]], color=colors, edgecolor='black', alpha=0.7)
        ax.set_ylabel('Count', fontsize=11, fontweight='bold')
        ax.set_title('Training Set (80%)', fontsize=12, fontweight='bold')
        ax.text(0, counts_train[1], f"{counts_train[1]:,}\n({counts_train[1]/len(self.y_train)*100:.1f}%)", 
               ha='center', va='bottom', fontweight='bold')
        ax.text(1, counts_train[0], f"{counts_train[0]:,}\n({counts_train[0]/len(self.y_train)*100:.1f}%)", 
               ha='center', va='bottom', fontweight='bold')
        
        # Test distribution
        ax = axes[2]
        counts_test = self.y_test.value_counts()
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
        """Generate comprehensive markdown report"""
        logger.info("="*70)
        logger.info("GENERATING REPORT")
        logger.info("="*70)
        
        report_lines = []
        report_lines.append("# Baseline Models Report")
        report_lines.append("")
        report_lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        report_lines.append("---")
        report_lines.append("")
        
        # Executive Summary
        report_lines.append("## Executive Summary")
        report_lines.append("")
        report_lines.append("This report presents the results of baseline model training for business success prediction.")
        report_lines.append(f"We trained **6 models** (3 algorithms × 2 imbalance handling approaches) on {len(self.y_train):,} training samples")
        report_lines.append(f"and evaluated them on {len(self.y_test):,} test samples.")
        report_lines.append("")
        
        # Find best model
        best_model_name = max(self.results.keys(), key=lambda x: self.results[x]['roc_auc'])
        best_roc_auc = self.results[best_model_name]['roc_auc']
        
        report_lines.append(f"**Best Model:** {best_model_name}")
        report_lines.append(f"**Best ROC-AUC:** {best_roc_auc:.4f}")
        report_lines.append("")
        report_lines.append("---")
        report_lines.append("")
        
        # Dataset Overview
        report_lines.append("## 1. Dataset Overview")
        report_lines.append("")
        report_lines.append(f"- **Total Samples:** {len(self.df):,}")
        report_lines.append(f"- **Features (Initial):** {len(self.feature_names)}")
        report_lines.append(f"- **Features (Selected):** {len(self.selected_features)}")
        report_lines.append(f"- **Target Variable:** `is_open` (1=Open, 0=Closed)")
        report_lines.append("")
        report_lines.append("**Class Distribution:**")
        report_lines.append(f"- Open Businesses: {self.y.sum():,} ({self.y.mean()*100:.2f}%)")
        report_lines.append(f"- Closed Businesses: {(1-self.y).sum():,} ({(1-self.y.mean())*100:.2f}%)")
        report_lines.append("")
        report_lines.append("![Class Distribution](./plots/class_distribution.png)")
        report_lines.append("")
        report_lines.append("---")
        report_lines.append("")
        
        # Feature Selection
        report_lines.append("## 2. Feature Selection")
        report_lines.append("")
        report_lines.append("**Multi-Stage Feature Selection Process:**")
        report_lines.append("")
        report_lines.append(f"1. **Initial Features:** {self.feature_selection_summary['initial_features']}")
        report_lines.append(f"2. **After Correlation Removal (|r| > 0.95):** {self.feature_selection_summary['after_correlation_removal']}")
        report_lines.append(f"3. **After Variance Removal (var < 0.01):** {self.feature_selection_summary['after_variance_removal']}")
        report_lines.append(f"4. **Final Selected Features:** {self.feature_selection_summary['final_features']}")
        report_lines.append("")
        report_lines.append("**Top 10 Selected Features:**")
        report_lines.append("")
        
        top_10_features = self.feature_selection_summary['feature_importances'][:10]
        report_lines.append("| Rank | Feature | Importance |")
        report_lines.append("|------|---------|------------|")
        for i, feat in enumerate(top_10_features, 1):
            report_lines.append(f"| {i} | {feat['feature']} | {feat['importance']:.4f} |")
        report_lines.append("")
        report_lines.append("![Feature Importance](./plots/feature_importance_selection.png)")
        report_lines.append("")
        report_lines.append("---")
        report_lines.append("")
        
        # Model Training
        report_lines.append("## 3. Model Training")
        report_lines.append("")
        report_lines.append("**Baseline Models:**")
        report_lines.append("")
        report_lines.append("1. **Logistic Regression**")
        report_lines.append("   - Linear model for binary classification")
        report_lines.append("   - Hyperparameters: C=1.0, solver='lbfgs', max_iter=1000")
        report_lines.append("")
        report_lines.append("2. **Decision Tree**")
        report_lines.append("   - Non-linear model with interpretable rules")
        report_lines.append("   - Hyperparameters: max_depth=10, min_samples_split=20, min_samples_leaf=10")
        report_lines.append("")
        report_lines.append("3. **Random Forest**")
        report_lines.append("   - Ensemble of decision trees")
        report_lines.append("   - Hyperparameters: n_estimators=100, max_depth=15, min_samples_split=20")
        report_lines.append("")
        report_lines.append("**Class Imbalance Handling:**")
        report_lines.append("")
        report_lines.append("- **Approach 1: SMOTE** - Synthetic Minority Over-sampling")
        report_lines.append("- **Approach 2: Class Weights** - Weighted loss function")
        report_lines.append("")
        report_lines.append("---")
        report_lines.append("")
        
        # Results
        report_lines.append("## 4. Model Performance")
        report_lines.append("")
        report_lines.append("### 4.1 Overall Comparison")
        report_lines.append("")
        report_lines.append("| Model | ROC-AUC | PR-AUC | Precision | Recall | F1-Score |")
        report_lines.append("|-------|---------|--------|-----------|--------|----------|")
        
        for model_name in sorted(self.results.keys()):
            results = self.results[model_name]
            report_lines.append(
                f"| {model_name.replace('_', ' ')} | "
                f"{results['roc_auc']:.4f} | "
                f"{results['pr_auc']:.4f} | "
                f"{results['precision']:.4f} | "
                f"{results['recall']:.4f} | "
                f"{results['f1_score']:.4f} |"
            )
        report_lines.append("")
        report_lines.append("![Model Comparison](./plots/model_comparison.png)")
        report_lines.append("")
        
        # ROC and PR curves
        report_lines.append("### 4.2 ROC Curves")
        report_lines.append("")
        report_lines.append("ROC (Receiver Operating Characteristic) curves show the trade-off between")
        report_lines.append("True Positive Rate and False Positive Rate at various classification thresholds.")
        report_lines.append("")
        report_lines.append("![ROC Curves](./plots/roc_curves.png)")
        report_lines.append("")
        
        report_lines.append("### 4.3 Precision-Recall Curves")
        report_lines.append("")
        report_lines.append("Precision-Recall curves are particularly useful for imbalanced datasets,")
        report_lines.append("showing the trade-off between precision and recall.")
        report_lines.append("")
        report_lines.append("![Precision-Recall Curves](./plots/precision_recall_curves.png)")
        report_lines.append("")
        
        # Confusion matrices
        report_lines.append("### 4.4 Confusion Matrices")
        report_lines.append("")
        report_lines.append("Detailed breakdown of predictions vs actual labels:")
        report_lines.append("")
        report_lines.append("![Confusion Matrices](./plots/confusion_matrices.png)")
        report_lines.append("")
        
        # Feature importance
        report_lines.append("### 4.5 Feature Importance (Random Forest)")
        report_lines.append("")
        report_lines.append("Top features identified by Random Forest models:")
        report_lines.append("")
        report_lines.append("![Random Forest Feature Importance](./plots/random_forest_feature_importance.png)")
        report_lines.append("")
        report_lines.append("---")
        report_lines.append("")
        
        # Detailed Analysis
        report_lines.append("## 5. Detailed Analysis")
        report_lines.append("")
        
        # Best model analysis
        best_results = self.results[best_model_name]
        report_lines.append(f"### 5.1 Best Model: {best_model_name}")
        report_lines.append("")
        report_lines.append("**Performance Metrics:**")
        report_lines.append("")
        report_lines.append(f"- **ROC-AUC:** {best_results['roc_auc']:.4f}")
        report_lines.append(f"- **PR-AUC:** {best_results['pr_auc']:.4f}")
        report_lines.append(f"- **Precision:** {best_results['precision']:.4f} (% of predicted closures that are actually closed)")
        report_lines.append(f"- **Recall:** {best_results['recall']:.4f} (% of actual closures that are detected)")
        report_lines.append(f"- **F1-Score:** {best_results['f1_score']:.4f} (harmonic mean of precision and recall)")
        report_lines.append("")
        
        cm = best_results['confusion_matrix']
        tn, fp, fn, tp = cm.ravel()
        report_lines.append("**Confusion Matrix Breakdown:**")
        report_lines.append("")
        report_lines.append(f"- **True Negatives (TN):** {tn:,} - Correctly predicted as closed")
        report_lines.append(f"- **False Positives (FP):** {fp:,} - Incorrectly predicted as closed")
        report_lines.append(f"- **False Negatives (FN):** {fn:,} - Incorrectly predicted as open")
        report_lines.append(f"- **True Positives (TP):** {tp:,} - Correctly predicted as open")
        report_lines.append("")
        
        # SMOTE vs Class Weight comparison
        report_lines.append("### 5.2 SMOTE vs Class Weight Comparison")
        report_lines.append("")
        report_lines.append("**SMOTE Approach:**")
        smote_avg_roc = np.mean([self.results[name]['roc_auc'] for name in self.results if 'SMOTE' in name])
        report_lines.append(f"- Average ROC-AUC: {smote_avg_roc:.4f}")
        report_lines.append("- Pros: Balanced training data, better minority class learning")
        report_lines.append("- Cons: Synthetic samples may not represent true distribution")
        report_lines.append("")
        
        report_lines.append("**Class Weight Approach:**")
        cw_avg_roc = np.mean([self.results[name]['roc_auc'] for name in self.results if 'ClassWeight' in name])
        report_lines.append(f"- Average ROC-AUC: {cw_avg_roc:.4f}")
        report_lines.append("- Pros: Uses only real data, faster training")
        report_lines.append("- Cons: May still bias towards majority class")
        report_lines.append("")
        
        # Model comparison
        report_lines.append("### 5.3 Model Algorithm Comparison")
        report_lines.append("")
        
        for algo in ['LogisticRegression', 'DecisionTree', 'RandomForest']:
            algo_models = [name for name in self.results if algo in name]
            avg_roc = np.mean([self.results[name]['roc_auc'] for name in algo_models])
            report_lines.append(f"**{algo}:**")
            report_lines.append(f"- Average ROC-AUC: {avg_roc:.4f}")
            
            if algo == 'LogisticRegression':
                report_lines.append("- Simple, interpretable, fast training")
                report_lines.append("- Best for: Understanding linear relationships")
            elif algo == 'DecisionTree':
                report_lines.append("- Rule-based, highly interpretable")
                report_lines.append("- Best for: Understanding feature interactions")
            else:  # RandomForest
                report_lines.append("- Ensemble method, robust, handles non-linearity")
                report_lines.append("- Best for: Overall predictive performance")
            report_lines.append("")
        
        report_lines.append("---")
        report_lines.append("")
        
        # Key Findings
        report_lines.append("## 6. Key Findings")
        report_lines.append("")
        report_lines.append("### ✅ Strengths")
        report_lines.append("")
        report_lines.append(f"1. **Strong Discriminative Power:** Best ROC-AUC of {best_roc_auc:.4f} indicates good separation")
        report_lines.append("2. **Feature Engineering Success:** Selected features show high predictive value")
        report_lines.append("3. **Consistent Performance:** Models perform reasonably across different approaches")
        report_lines.append("")
        
        report_lines.append("### ⚠️ Challenges")
        report_lines.append("")
        report_lines.append("1. **Class Imbalance:** 80/20 split requires careful handling")
        report_lines.append("2. **Minority Class Recall:** Detecting business closures remains challenging")
        report_lines.append("3. **Model Complexity:** Trade-off between performance and interpretability")
        report_lines.append("")
        
        report_lines.append("### 💡 Insights")
        report_lines.append("")
        
        # Get top 5 features from best Random Forest model
        rf_smote_name = 'RandomForest_SMOTE'
        if rf_smote_name in self.models:
            rf_model = self.models[rf_smote_name]
            top_5_feat = pd.DataFrame({
                'feature': self.selected_features,
                'importance': rf_model.feature_importances_
            }).sort_values('importance', ascending=False).head(5)
            
            report_lines.append("**Top 5 Predictive Features:**")
            for i, row in top_5_feat.iterrows():
                report_lines.append(f"- `{row['feature']}`: {row['importance']:.4f}")
        report_lines.append("")
        
        report_lines.append("---")
        report_lines.append("")
        
        # Recommendations
        report_lines.append("## 7. Recommendations & Next Steps")
        report_lines.append("")
        report_lines.append("### 📈 Model Improvement")
        report_lines.append("")
        report_lines.append("1. **Hyperparameter Tuning:**")
        report_lines.append("   - Use GridSearchCV or RandomizedSearchCV")
        report_lines.append("   - Focus on regularization parameters (C, max_depth, n_estimators)")
        report_lines.append("")
        report_lines.append("2. **Advanced Algorithms:**")
        report_lines.append("   - XGBoost or LightGBM for gradient boosting")
        report_lines.append("   - Neural Networks for complex patterns")
        report_lines.append("   - Ensemble stacking methods")
        report_lines.append("")
        report_lines.append("3. **Feature Engineering:**")
        report_lines.append("   - Interaction features between top predictors")
        report_lines.append("   - Polynomial features for non-linear relationships")
        report_lines.append("   - Domain-specific feature engineering")
        report_lines.append("")
        
        report_lines.append("### 🎯 Business Application")
        report_lines.append("")
        report_lines.append("1. **Threshold Selection:**")
        report_lines.append("   - Choose optimal probability threshold based on business costs")
        report_lines.append("   - Consider cost of false positives vs false negatives")
        report_lines.append("")
        report_lines.append("2. **Model Deployment:**")
        report_lines.append("   - Implement prediction API for real-time scoring")
        report_lines.append("   - Set up monitoring for model performance drift")
        report_lines.append("   - Create alert system for high-risk businesses")
        report_lines.append("")
        report_lines.append("3. **Interpretability:**")
        report_lines.append("   - Generate SHAP values for individual predictions")
        report_lines.append("   - Create decision rules from tree models")
        report_lines.append("   - Provide actionable insights to stakeholders")
        report_lines.append("")
        
        report_lines.append("---")
        report_lines.append("")
        
        # Technical Details
        report_lines.append("## 8. Technical Details")
        report_lines.append("")
        report_lines.append("### Model Files")
        report_lines.append("")
        report_lines.append("**Saved Models:**")
        for name in self.models.keys():
            report_lines.append(f"- `saved_models/{name}.pkl`")
        report_lines.append(f"- `saved_models/scaler.pkl`")
        report_lines.append("")
        
        report_lines.append("### Selected Features")
        report_lines.append("")
        report_lines.append(f"**Total:** {len(self.selected_features)} features")
        report_lines.append("")
        report_lines.append("<details>")
        report_lines.append("<summary>Click to expand full feature list</summary>")
        report_lines.append("")
        for i, feat in enumerate(self.selected_features, 1):
            report_lines.append(f"{i}. {feat}")
        report_lines.append("")
        report_lines.append("</details>")
        report_lines.append("")
        
        report_lines.append("### Reproducibility")
        report_lines.append("")
        report_lines.append(f"- **Random State:** {self.random_state}")
        report_lines.append(f"- **Train/Test Split:** 80/20 stratified")
        report_lines.append(f"- **Scaling:** StandardScaler (mean=0, std=1)")
        report_lines.append("")
        
        report_lines.append("---")
        report_lines.append("")
        
        # Conclusion
        report_lines.append("## 9. Conclusion")
        report_lines.append("")
        report_lines.append(f"This baseline model analysis establishes a strong foundation with a best ROC-AUC of **{best_roc_auc:.4f}**.")
        report_lines.append(f"The **{best_model_name}** model demonstrates the most promising performance and serves as")
        report_lines.append("the benchmark for future model iterations.")
        report_lines.append("")
        report_lines.append("Key takeaways:")
        report_lines.append(f"- Feature engineering produced {len(self.selected_features)} highly predictive features")
        report_lines.append("- Both SMOTE and class weighting approaches show merit")
        report_lines.append("- Random Forest models generally outperform simpler approaches")
        report_lines.append("- Further improvements possible through hyperparameter tuning and advanced algorithms")
        report_lines.append("")
        report_lines.append("**Next Steps:** Proceed to advanced model development and deployment preparation.")
        report_lines.append("")
        
        # Save report
        report_file = self.output_path / "baseline_models_report.md"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        logger.info(f"Saved: {report_file}")
        
        # Save results summary as JSON
        results_summary = {}
        for model_name, results in self.results.items():
            results_summary[model_name] = {
                'roc_auc': float(results['roc_auc']),
                'pr_auc': float(results['pr_auc']),
                'precision': float(results['precision']),
                'recall': float(results['recall']),
                'f1_score': float(results['f1_score']),
                'confusion_matrix': results['confusion_matrix'].tolist(),
                'classification_report': results['classification_report']
            }
        
        with open(self.output_path / 'results_summary.json', 'w') as f:
            json.dump(results_summary, f, indent=2)
        
        logger.info(f"Saved: {self.output_path / 'results_summary.json'}")
    
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
        self.feature_selection()
        
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
    """Main entry point"""
    print("="*70)
    print("CS 412 RESEARCH PROJECT - BASELINE MODELS")
    print("Business Success Prediction using Yelp Dataset")
    print("="*70)
    print("")
    
    # Path to features (update this path as needed)
    data_path = "data/features/business_features_final.csv"
    output_path = "src/models"
    
    print(f"Data path: {data_path}")
    print(f"Output path: {output_path}")
    print("")
    
    # Initialize and run pipeline
    pipeline = BaselineModelPipeline(
        data_path=data_path,
        output_path=output_path,
        random_state=42
    )
    
    pipeline.run_pipeline()
    
    print("\n" + "="*70)
    print("BASELINE MODELING COMPLETE!")
    print("="*70)
    print("\nCheck the following outputs:")
    print("  1. src/models/baseline_models_report.md")
    print("  2. src/models/plots/ (visualizations)")
    print("  3. src/models/saved_models/ (trained models)")
    print("")


if __name__ == "__main__":
    main()