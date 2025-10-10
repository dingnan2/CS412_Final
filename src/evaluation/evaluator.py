"""
Evaluation metrics and validation framework for CS 412 Research Project
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Any, Optional, Tuple
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve, confusion_matrix,
    classification_report, mean_squared_error, mean_absolute_error, r2_score
)
from sklearn.model_selection import cross_val_score, StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

from ..utils.config import config


class ModelEvaluator:
    """Comprehensive model evaluation framework"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.results = {}
        
    def evaluate_classification(self, y_true: np.ndarray, y_pred: np.ndarray, 
                              y_pred_proba: Optional[np.ndarray] = None,
                              model_name: str = "model") -> Dict[str, Any]:
        """Evaluate classification model performance"""
        
        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
        f1 = f1_score(y_true, y_pred, average='weighted')
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # ROC-AUC if probabilities available
        roc_auc = None
        if y_pred_proba is not None:
            roc_auc = roc_auc_score(y_true, y_pred_proba)
        
        # Classification report
        report = classification_report(y_true, y_pred, output_dict=True)
        
        results = {
            'model_name': model_name,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'confusion_matrix': cm.tolist(),
            'classification_report': report
        }
        
        self.logger.info(f"{model_name} - Accuracy: {accuracy:.4f}, F1: {f1:.4f}, ROC-AUC: {roc_auc:.4f if roc_auc else 'N/A'}")
        return results
    
    def evaluate_regression(self, y_true: np.ndarray, y_pred: np.ndarray,
                          model_name: str = "model") -> Dict[str, Any]:
        """Evaluate regression model performance"""
        
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        
        results = {
            'model_name': model_name,
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'r2_score': r2
        }
        
        self.logger.info(f"{model_name} - RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
        return results
    
    def cross_validate_model(self, model, X: np.ndarray, y: np.ndarray,
                           cv_folds: int = 5, scoring: str = 'roc_auc') -> Dict[str, Any]:
        """Perform cross-validation"""
        
        eval_params = config.get_evaluation_params()
        cv_folds = eval_params.get('cv_folds', cv_folds)
        random_state = eval_params.get('random_state', 42)
        
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
        
        results = {
            'cv_scores': scores.tolist(),
            'cv_mean': np.mean(scores),
            'cv_std': np.std(scores),
            'cv_min': np.min(scores),
            'cv_max': np.max(scores)
        }
        
        self.logger.info(f"Cross-validation {scoring}: {results['cv_mean']:.4f} ± {results['cv_std']:.4f}")
        return results
    
    def plot_roc_curve(self, y_true: np.ndarray, y_pred_proba: np.ndarray,
                      model_name: str = "model", save_path: Optional[str] = None):
        """Plot ROC curve"""
        
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = roc_auc_score(y_true, y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {model_name}')
        plt.legend(loc="lower right")
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"ROC curve saved to {save_path}")
        
        plt.show()
    
    def plot_precision_recall_curve(self, y_true: np.ndarray, y_pred_proba: np.ndarray,
                                  model_name: str = "model", save_path: Optional[str] = None):
        """Plot precision-recall curve"""
        
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='darkorange', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve - {model_name}')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Precision-recall curve saved to {save_path}")
        
        plt.show()
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray,
                            model_name: str = "model", save_path: Optional[str] = None):
        """Plot confusion matrix"""
        
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Confusion matrix saved to {save_path}")
        
        plt.show()
    
    def plot_feature_importance(self, feature_names: List[str], 
                              importance_scores: np.ndarray,
                              model_name: str = "model", 
                              top_n: int = 20,
                              save_path: Optional[str] = None):
        """Plot feature importance"""
        
        # Get top N features
        top_indices = np.argsort(importance_scores)[-top_n:]
        top_features = [feature_names[i] for i in top_indices]
        top_scores = importance_scores[top_indices]
        
        plt.figure(figsize=(10, 8))
        plt.barh(range(len(top_features)), top_scores)
        plt.yticks(range(len(top_features)), top_features)
        plt.xlabel('Importance Score')
        plt.title(f'Top {top_n} Feature Importance - {model_name}')
        plt.gca().invert_yaxis()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Feature importance plot saved to {save_path}")
        
        plt.show()
    
    def compare_models(self, results: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
        """Compare multiple models"""
        
        comparison_data = []
        for model_name, result in results.items():
            comparison_data.append({
                'Model': model_name,
                'Accuracy': result.get('accuracy', 0),
                'Precision': result.get('precision', 0),
                'Recall': result.get('recall', 0),
                'F1-Score': result.get('f1_score', 0),
                'ROC-AUC': result.get('roc_auc', 0)
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('ROC-AUC', ascending=False)
        
        return comparison_df
    
    def plot_model_comparison(self, results: Dict[str, Dict[str, Any]], 
                            metric: str = 'roc_auc',
                            save_path: Optional[str] = None):
        """Plot model comparison"""
        
        models = list(results.keys())
        scores = [results[model].get(metric, 0) for model in models]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(models, scores)
        plt.title(f'Model Comparison - {metric.upper()}')
        plt.ylabel(metric.upper())
        plt.xticks(rotation=45)
        
        # Add value labels on bars
        for bar, score in zip(bars, scores):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{score:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Model comparison plot saved to {save_path}")
        
        plt.show()
    
    def save_results(self, results: Dict[str, Any], 
                    output_path: str = "results/evaluation_results.json"):
        """Save evaluation results to JSON"""
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        self.logger.info(f"Results saved to {output_file}")
    
    def load_results(self, input_path: str = "results/evaluation_results.json") -> Dict[str, Any]:
        """Load evaluation results from JSON"""
        
        input_file = Path(input_path)
        if not input_file.exists():
            self.logger.warning(f"Results file not found: {input_file}")
            return {}
        
        with open(input_file, 'r') as f:
            results = json.load(f)
        
        self.logger.info(f"Results loaded from {input_file}")
        return results
    
    def generate_report(self, results: Dict[str, Any], 
                       output_path: str = "results/evaluation_report.txt"):
        """Generate text report of evaluation results"""
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            f.write("CS 412 Research Project - Model Evaluation Report\n")
            f.write("=" * 50 + "\n\n")
            
            for model_name, result in results.items():
                f.write(f"Model: {model_name}\n")
                f.write("-" * 20 + "\n")
                f.write(f"Accuracy: {result.get('accuracy', 0):.4f}\n")
                f.write(f"Precision: {result.get('precision', 0):.4f}\n")
                f.write(f"Recall: {result.get('recall', 0):.4f}\n")
                f.write(f"F1-Score: {result.get('f1_score', 0):.4f}\n")
                f.write(f"ROC-AUC: {result.get('roc_auc', 0):.4f}\n")
                f.write("\n")
        
        self.logger.info(f"Report saved to {output_file}")


class ValidationFramework:
    """Framework for model validation and testing"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.evaluator = ModelEvaluator()
    
    def temporal_validation(self, X: pd.DataFrame, y: pd.Series, 
                          model, date_col: str = 'date',
                          test_months: int = 6) -> Dict[str, Any]:
        """Perform temporal validation"""
        
        # Sort by date
        df = pd.concat([X, y], axis=1)
        df = df.sort_values(date_col)
        
        # Split by time
        cutoff_date = df[date_col].max() - pd.DateOffset(months=test_months)
        
        train_mask = df[date_col] <= cutoff_date
        test_mask = df[date_col] > cutoff_date
        
        X_train = df[train_mask].drop(columns=[y.name])
        y_train = df[train_mask][y.name]
        X_test = df[test_mask].drop(columns=[y.name])
        y_test = df[test_mask][y.name]
        
        # Train and evaluate
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        results = self.evaluator.evaluate_classification(
            y_test, y_pred, y_pred_proba, "temporal_validation"
        )
        
        return results
    
    def stratified_validation(self, X: pd.DataFrame, y: pd.Series, 
                            model, stratify_col: str = 'category') -> Dict[str, Any]:
        """Perform stratified validation by category"""
        
        # Get unique categories
        categories = X[stratify_col].unique()
        category_results = {}
        
        for category in categories:
            mask = X[stratify_col] == category
            X_cat = X[mask]
            y_cat = y[mask]
            
            if len(X_cat) < 10:  # Skip categories with too few samples
                continue
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_cat, y_cat, test_size=0.2, random_state=42, stratify=y_cat
            )
            
            # Train and evaluate
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
            
            category_results[category] = self.evaluator.evaluate_classification(
                y_test, y_pred, y_pred_proba, f"category_{category}"
            )
        
        return category_results
