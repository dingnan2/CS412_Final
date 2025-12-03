"""
Statistical Significance Testing for Model Comparisons

This module provides statistical tests to determine if performance differences
between models are statistically significant, not just random noise.

Key Methods:
- Bootstrap Confidence Intervals for AUC differences
- Permutation tests for feature importance
- McNemar's test for classifier comparison
"""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from sklearn.utils import resample
from scipy import stats
from pathlib import Path
from typing import Dict, List, Tuple
import json



class StatisticalTester:
    """
    Perform statistical tests to validate model performance differences.
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize statistical tester.
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        np.random.seed(random_state)
    
    def bootstrap_confidence_interval(self,
                                     y_true: np.ndarray,
                                     pred1: np.ndarray,
                                     pred2: np.ndarray,
                                     metric: str = 'roc_auc',
                                     n_iterations: int = 1000,
                                     confidence_level: float = 0.95) -> Dict:
        """
        Bootstrap confidence interval for performance difference between two models.
        
        This answers: "Is Model 2 significantly better than Model 1?"
        
        Args:
            y_true: True labels
            pred1: Predictions from model 1 (baseline)
            pred2: Predictions from model 2 (improved)
            metric: 'roc_auc', 'precision', 'recall', 'f1'
            n_iterations: Number of bootstrap iterations
            confidence_level: Confidence level (e.g., 0.95 for 95% CI)
            
        Returns:
            Dict with mean_diff, ci_low, ci_high, p_value, is_significant
        """
        
        # Determine metric function
        if metric == 'roc_auc':
            metric_fn = roc_auc_score
        elif metric == 'precision':
            metric_fn = lambda yt, yp: precision_score(yt, (yp > 0.5).astype(int))
        elif metric == 'recall':
            metric_fn = lambda yt, yp: recall_score(yt, (yp > 0.5).astype(int))
        elif metric == 'f1':
            metric_fn = lambda yt, yp: f1_score(yt, (yp > 0.5).astype(int))
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        # Bootstrap sampling
        diffs = []
        model1_scores = []
        model2_scores = []
        
        for i in range(n_iterations):
            # Resample with replacement
            indices = resample(
                range(len(y_true)),
                replace=True,
                random_state=self.random_state + i
            )
            
            y_boot = y_true[indices]
            pred1_boot = pred1[indices]
            pred2_boot = pred2[indices]
            
            # Compute metrics
            score1 = metric_fn(y_boot, pred1_boot)
            score2 = metric_fn(y_boot, pred2_boot)
            
            model1_scores.append(score1)
            model2_scores.append(score2)
            diffs.append(score2 - score1)
        
        diffs = np.array(diffs)
        
        # Compute statistics
        mean_diff = np.mean(diffs)
        std_diff = np.std(diffs)
        
        # Confidence interval
        alpha = 1 - confidence_level
        ci_low = np.percentile(diffs, alpha/2 * 100)
        ci_high = np.percentile(diffs, (1 - alpha/2) * 100)
        
        # P-value (one-sided test: Model 2 > Model 1)
        p_value = np.mean(diffs <= 0)

        is_significant = (ci_low > 0)  
        

        return {
            'mean_diff': mean_diff,
            'std_diff': std_diff,
            'ci_low': ci_low,
            'ci_high': ci_high,
            'p_value': p_value,
            'is_significant': is_significant,
            'model1_mean': np.mean(model1_scores),
            'model2_mean': np.mean(model2_scores),
            'model1_std': np.std(model1_scores),
            'model2_std': np.std(model2_scores)
        }
    
    def compare_multiple_models(self,
                               y_true: np.ndarray,
                               predictions: Dict[str, np.ndarray],
                               baseline_model: str,
                               metric: str = 'roc_auc',
                               n_iterations: int = 1000) -> pd.DataFrame:
        """
        Compare multiple models against a baseline with bootstrap CI.
        
        Args:
            y_true: True labels
            predictions: Dict of {model_name: predictions}
            baseline_model: Name of baseline model
            metric: Metric to use
            n_iterations: Bootstrap iterations
            
        Returns:
            DataFrame with comparison results
        """
        
        if baseline_model not in predictions:
            raise ValueError(f"Baseline model '{baseline_model}' not found")
        
        baseline_pred = predictions[baseline_model]
        results = []
        
        for model_name, model_pred in predictions.items():
            if model_name == baseline_model:
                # Baseline reference
                results.append({
                    'model': model_name,
                    'mean_diff': 0.0,
                    'ci_low': 0.0,
                    'ci_high': 0.0,
                    'p_value': 1.0,
                    'is_significant': False,
                    'conclusion': 'Baseline'
                })
            else:
                # Compare to baseline
                comparison = self.bootstrap_confidence_interval(
                y_true=y_true,
                pred1=baseline_pred,
                pred2=model_pred,
                metric=metric,
                    n_iterations=n_iterations,
                    confidence_level=0.95
            )
                
                # Determine conclusion
                if comparison['is_significant']:
                    if comparison['mean_diff'] > 0:
                        conclusion = f"[OK] Better (+{comparison['mean_diff']:.4f})"
                    else:
                        conclusion = f"[FAIL] Worse ({comparison['mean_diff']:.4f})"
                else:
                    conclusion = "[WARN] No significant difference"
            
            results.append({
                'model': model_name,
                    'mean_diff': comparison['mean_diff'],
                    'ci_low': comparison['ci_low'],
                    'ci_high': comparison['ci_high'],
                    'p_value': comparison['p_value'],
                    'is_significant': comparison['is_significant'],
                    'conclusion': conclusion
            })
        
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('mean_diff', ascending=False)
        
        
        return results_df
    
    def mcnemar_test(self,
                     y_true: np.ndarray,
                     pred1: np.ndarray,
                    pred2: np.ndarray,
                    threshold: float = 0.5) -> Dict:
        """
        McNemar's test for comparing two binary classifiers.
        
        Tests whether two classifiers disagree in systematic ways.
        
        Args:
            y_true: True labels
            pred1: Predictions from model 1 (probabilities)
            pred2: Predictions from model 2 (probabilities)
            threshold: Classification threshold
            
        Returns:
            Dict with test statistic and p-value
        """
        
        # Convert probabilities to binary predictions
        y_pred1 = (pred1 > threshold).astype(int)
        y_pred2 = (pred2 > threshold).astype(int)
        
        # contingency table
        correct1 = (y_pred1 == y_true)
        correct2 = (y_pred2 == y_true)
        
        n01 = np.sum(correct1 & ~correct2)
        n10 = np.sum(~correct1 & correct2)
        
        
        # McNemar's test statistic (with continuity correction)
        if (n01 + n10) == 0:
            return {'statistic': 0.0, 'p_value': 1.0, 'is_significant': False}
        
        statistic = ((abs(n01 - n10) - 1) ** 2) / (n01 + n10)
        p_value = stats.chi2.sf(statistic, df=1)
        
        is_significant = (p_value < 0.05)
        

        return {
            'statistic': statistic,
            'p_value': p_value,
            'is_significant': is_significant,
            'n01': n01,
            'n10': n10
        }


def main():
    """
    Example usage: Compare RandomForest vs Ensemble
    """
    import sys
    sys.path.append(str(Path(__file__).parent.parent))
    
    
    # Load predictions (example paths)
    # You'll need to save predictions from baseline_models.py and advanced_models.py
    
    # Example:
    # predictions = np.load('predictions.npz')
    # y_true = predictions['y_true']
    # rf_pred = predictions['rf_pred']
    # ensemble_pred = predictions['ensemble_pred']
    
    # tester = StatisticalTester()
        
    # Test 1: Bootstrap CI
    # result = tester.bootstrap_confidence_interval(
    #     y_true=y_true,
    #     pred1=rf_pred,
    #     pred2=ensemble_pred,
    #     metric='roc_auc',
    #     n_iterations=1000
    # )
        
    # Test 2: Multiple model comparison
    # predictions_dict = {
    #     'RandomForest': rf_pred,
    #     'XGBoost': xgb_pred,
    #     'LightGBM': lgbm_pred,
    #     'Ensemble': ensemble_pred
    # }
        
    # results_df = tester.compare_multiple_models(
    #     y_true=y_true,
    #     predictions=predictions_dict,
    #     baseline_model='RandomForest',
    #     metric='roc_auc'
    # )
    


if __name__ == "__main__":
    main()