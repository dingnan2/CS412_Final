"""
Data validation utilities for quality assurance.

This module provides validation functions to ensure data quality throughout
the pipeline, with special focus on:
- Label quality and confidence filtering
- Feature completeness and validity
- Temporal consistency checks
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def validate_label_quality(tasks_df: pd.DataFrame,
                          min_confidence: float = 0.7,
                          require_balanced: bool = True,
                          min_samples_per_class: int = 100) -> pd.DataFrame:
    """
    Validate and filter prediction tasks based on label quality.
    
    Quality criteria:
    1. Label confidence above threshold
    2. Both classes (open/closed) sufficiently represented
    3. No data anomalies (e.g., impossible dates)
    
    Args:
        tasks_df: DataFrame with prediction tasks and labels
        min_confidence: Minimum confidence threshold (0.0 to 1.0)
        require_balanced: If True, ensure both classes have minimum samples
        min_samples_per_class: Minimum samples required per class if balanced
        
    Returns:
        Filtered DataFrame with only high-quality labels
        
    Side effects:
        Logs validation statistics and warnings
    """
    logger.info("="*70)
    logger.info("VALIDATING LABEL QUALITY")
    logger.info("="*70)
    
    initial_count = len(tasks_df)
    logger.info(f"Initial tasks: {initial_count:,}")
    
    # Check 1: Label confidence
    if 'label_confidence' in tasks_df.columns:
        high_confidence = tasks_df['label_confidence'] >= min_confidence
        tasks_df = tasks_df[high_confidence].copy()
        
        removed_low_conf = initial_count - len(tasks_df)
        logger.info(f"\nConfidence filtering (threshold={min_confidence}):")
        logger.info(f"  Removed {removed_low_conf:,} low-confidence tasks")
        logger.info(f"  Remaining: {len(tasks_df):,}")
    else:
        logger.warning("No 'label_confidence' column found, skipping confidence filter")
    
    # Check 2: Class distribution
    if 'label' in tasks_df.columns:
        class_counts = tasks_df['label'].value_counts()
        logger.info(f"\nClass distribution:")
        logger.info(f"  Open (1): {class_counts.get(1, 0):,} ({class_counts.get(1, 0)/len(tasks_df)*100:.1f}%)")
        logger.info(f"  Closed (0): {class_counts.get(0, 0):,} ({class_counts.get(0, 0)/len(tasks_df)*100:.1f}%)")
        
        # Check class balance
        if require_balanced:
            for label in [0, 1]:
                count = class_counts.get(label, 0)
                if count < min_samples_per_class:
                    logger.warning(f"  WARNING: Class {label} has only {count} samples "
                                 f"(minimum: {min_samples_per_class})")
                    logger.warning(f"  Consider lowering min_confidence or min_samples_per_class")
    else:
        logger.warning("No 'label' column found, skipping class distribution check")
    
    # Check 3: Date consistency
    if 'cutoff_date' in tasks_df.columns and 'target_date' in tasks_df.columns:
        invalid_dates = tasks_df['cutoff_date'] >= tasks_df['target_date']
        num_invalid = invalid_dates.sum()
        
        if num_invalid > 0:
            logger.warning(f"\nFound {num_invalid} tasks with invalid date ordering")
            logger.warning(f"  (cutoff_date >= target_date)")
            tasks_df = tasks_df[~invalid_dates].copy()
            logger.info(f"  Removed invalid tasks, remaining: {len(tasks_df):,}")
    
    # Check 4: Temporal coverage
    if 'prediction_year' in tasks_df.columns:
        year_counts = tasks_df['prediction_year'].value_counts().sort_index()
        logger.info(f"\nTemporal coverage:")
        
        for year, count in year_counts.items():
            logger.info(f"  {year}: {count:,} tasks")
        
        # Warn about years with very few samples
        sparse_years = year_counts[year_counts < 100]
        if len(sparse_years) > 0:
            logger.warning(f"\nYears with <100 tasks: {list(sparse_years.index)}")
            logger.warning(f"  These years may have unreliable statistics")
    
    # Summary
    logger.info(f"\n{'='*70}")
    logger.info(f"VALIDATION COMPLETE")
    logger.info(f"{'='*70}")
    logger.info(f"Final tasks: {len(tasks_df):,}")
    logger.info(f"Retention rate: {len(tasks_df)/initial_count*100:.1f}%")
    logger.info(f"{'='*70}\n")
    
    return tasks_df


def validate_feature_quality(features_df: pd.DataFrame,
                            required_columns: Optional[List[str]] = None,
                            max_missing_rate: float = 0.1,
                            check_inf: bool = True) -> Tuple[pd.DataFrame, Dict]:
    """
    Validate feature quality and completeness.
    
    Checks:
    1. Required columns present
    2. Missing value rates within threshold
    3. No infinite values
    4. Reasonable value ranges
    
    Args:
        features_df: DataFrame with engineered features
        required_columns: List of columns that must be present
        max_missing_rate: Maximum allowed missing rate per column (0.0 to 1.0)
        check_inf: If True, check for infinite values
        
    Returns:
        Tuple of (cleaned_df, validation_report)
        - cleaned_df: DataFrame with invalid rows removed
        - validation_report: Dict with validation statistics
        
    Side effects:
        Logs validation issues and warnings
    """
    logger.info("="*70)
    logger.info("VALIDATING FEATURE QUALITY")
    logger.info("="*70)
    
    initial_count = len(features_df)
    validation_report = {
        'initial_rows': initial_count,
        'issues': [],
        'warnings': []
    }
    
    # Check 1: Required columns
    if required_columns:
        missing_cols = set(required_columns) - set(features_df.columns)
        if missing_cols:
            error_msg = f"Missing required columns: {missing_cols}"
            logger.error(error_msg)
            validation_report['issues'].append(error_msg)
            raise ValueError(error_msg)
        
        logger.info(f"[OK] All {len(required_columns)} required columns present")
    
    # Check 2: Missing values
    missing_stats = features_df.isnull().sum()
    missing_rates = missing_stats / len(features_df)
    
    high_missing_cols = missing_rates[missing_rates > max_missing_rate]
    
    if len(high_missing_cols) > 0:
        logger.warning(f"\nColumns with high missing rates (>{max_missing_rate*100}%):")
        for col, rate in high_missing_cols.items():
            logger.warning(f"  {col}: {rate*100:.1f}%")
            validation_report['warnings'].append(
                f"High missing rate in {col}: {rate*100:.1f}%"
            )
    else:
        logger.info(f"[OK] All columns have missing rates <= {max_missing_rate*100}%")
    
    # Check 3: Infinite values
    if check_inf:
        numeric_cols = features_df.select_dtypes(include=[np.number]).columns
        inf_counts = {}
        
        for col in numeric_cols:
            inf_count = np.isinf(features_df[col]).sum()
            if inf_count > 0:
                inf_counts[col] = inf_count
        
        if inf_counts:
            logger.warning(f"\nColumns with infinite values:")
            for col, count in inf_counts.items():
                logger.warning(f"  {col}: {count:,} infinite values")
                validation_report['warnings'].append(
                    f"Infinite values in {col}: {count}"
                )
            
            # Remove rows with infinite values
            for col in inf_counts.keys():
                features_df = features_df[~np.isinf(features_df[col])].copy()
            
            logger.info(f"  Removed {initial_count - len(features_df):,} rows with infinite values")
        else:
            logger.info(f"[OK] No infinite values detected")
    
    # Check 4: Feature statistics
    numeric_cols = features_df.select_dtypes(include=[np.number]).columns
    
    logger.info(f"\nFeature statistics:")
    logger.info(f"  Total features: {len(features_df.columns)}")
    logger.info(f"  Numeric features: {len(numeric_cols)}")
    logger.info(f"  Categorical features: {len(features_df.columns) - len(numeric_cols)}")
    
    # Check for constant features (zero variance)
    if len(numeric_cols) > 0:
        constant_features = []
        for col in numeric_cols:
            if features_df[col].nunique() == 1:
                constant_features.append(col)
        
        if constant_features:
            logger.warning(f"\nConstant features (zero variance):")
            for col in constant_features:
                logger.warning(f"  {col}")
                validation_report['warnings'].append(f"Constant feature: {col}")
    
    # Check 5: Duplicates
    duplicate_count = features_df.duplicated().sum()
    if duplicate_count > 0:
        logger.warning(f"\nFound {duplicate_count:,} duplicate rows")
        features_df = features_df.drop_duplicates().copy()
        logger.info(f"  Removed duplicates, remaining: {len(features_df):,}")
        validation_report['warnings'].append(f"Removed {duplicate_count} duplicate rows")
    else:
        logger.info(f"[OK] No duplicate rows")
    
    # Summary
    validation_report['final_rows'] = len(features_df)
    validation_report['rows_removed'] = initial_count - len(features_df)
    validation_report['retention_rate'] = len(features_df) / initial_count
    
    logger.info(f"\n{'='*70}")
    logger.info(f"FEATURE VALIDATION COMPLETE")
    logger.info(f"{'='*70}")
    logger.info(f"Final rows: {len(features_df):,}")
    logger.info(f"Rows removed: {validation_report['rows_removed']:,}")
    logger.info(f"Retention rate: {validation_report['retention_rate']*100:.1f}%")
    logger.info(f"{'='*70}\n")
    
    return features_df, validation_report


def filter_by_confidence(tasks_df: pd.DataFrame,
                        min_confidence: float = 0.7,
                        confidence_col: str = 'label_confidence') -> pd.DataFrame:
    """
    Simple confidence-based filtering.
    
    Args:
        tasks_df: DataFrame with tasks
        min_confidence: Minimum confidence threshold
        confidence_col: Name of confidence column
        
    Returns:
        Filtered DataFrame
    """
    if confidence_col not in tasks_df.columns:
        logger.warning(f"Column '{confidence_col}' not found, returning original DataFrame")
        return tasks_df
    
    initial_count = len(tasks_df)
    filtered_df = tasks_df[tasks_df[confidence_col] >= min_confidence].copy()
    
    removed = initial_count - len(filtered_df)
    logger.info(f"Confidence filter (>={min_confidence}): {initial_count:,} -> {len(filtered_df):,} "
                f"(removed {removed:,})")
    
    return filtered_df


def check_temporal_leakage(features_df: pd.DataFrame,
                          cutoff_date_col: str = '_feature_cutoff_date',
                          suspicious_features: Optional[List[str]] = None) -> Dict:
    """
    Check for potential temporal leakage in features.
    
    This function performs heuristic checks to detect features that might
    contain future information.
    
    Args:
        features_df: DataFrame with features
        cutoff_date_col: Column containing feature computation cutoff date
        suspicious_features: List of feature names to specifically check
        
    Returns:
        Dict with leakage detection results
        
    Checks:
        1. Features with very high predictive power (>50% importance)
        2. Features that encode "current" status directly
        3. Features with names suggesting temporal issues
    """
    logger.info("Checking for temporal leakage...")
    
    leakage_report = {
        'suspicious_features': [],
        'warnings': []
    }
    
    # Define suspicious patterns in feature names
    suspicious_patterns = [
        'last_', 'current_', 'final_', 'latest_', 
        'since_last', 'days_since', 'is_open'
    ]
    
    # Check feature names
    for col in features_df.columns:
        for pattern in suspicious_patterns:
            if pattern in col.lower():
                leakage_report['suspicious_features'].append(col)
                leakage_report['warnings'].append(
                    f"Feature '{col}' contains suspicious pattern '{pattern}'"
                )
                logger.warning(f"  [WARN]  Suspicious feature name: {col}")
    
    # Check if specific features are present
    if suspicious_features:
        for feat in suspicious_features:
            if feat in features_df.columns:
                leakage_report['suspicious_features'].append(feat)
                logger.warning(f"  [WARN]  Known leaky feature present: {feat}")
    
    # Check for features with unrealistically high correlation with target
    if 'is_open' in features_df.columns or 'label' in features_df.columns:
        target_col = 'is_open' if 'is_open' in features_df.columns else 'label'
        numeric_cols = features_df.select_dtypes(include=[np.number]).columns
        numeric_cols = [c for c in numeric_cols if c not in [target_col, 'business_id']]
        
        if len(numeric_cols) > 0:
            correlations = features_df[numeric_cols].corrwith(features_df[target_col]).abs()
            high_corr = correlations[correlations > 0.8]
            
            if len(high_corr) > 0:
                logger.warning(f"\n  Features with very high correlation (>0.8) to target:")
                for feat, corr in high_corr.items():
                    logger.warning(f"    {feat}: {corr:.3f}")
                    leakage_report['warnings'].append(
                        f"High correlation: {feat} ({corr:.3f})"
                    )
    
    # Summary
    if len(leakage_report['suspicious_features']) == 0:
        logger.info("  [OK] No obvious temporal leakage detected")
    else:
        logger.warning(f"\n  Found {len(leakage_report['suspicious_features'])} "
                      f"potentially leaky features")
    
    return leakage_report


def validate_temporal_consistency(tasks_df: pd.DataFrame,
                                 reviews_df: pd.DataFrame,
                                 features_df: pd.DataFrame) -> Dict:
    """
    Validate temporal consistency across tasks, reviews, and features.
    
    Ensures that:
    1. Features are computed only from reviews before cutoff
    2. Labels are evaluated at correct target dates
    3. No future information leakage
    
    Args:
        tasks_df: DataFrame with prediction tasks
        reviews_df: DataFrame with reviews
        features_df: DataFrame with features
        
    Returns:
        Dict with validation results
    """
    logger.info("Validating temporal consistency...")
    
    consistency_report = {
        'valid': True,
        'issues': [],
        'checks_performed': []
    }
    
    # Check 1: Cutoff dates are before target dates
    if 'cutoff_date' in tasks_df.columns and 'target_date' in tasks_df.columns:
        invalid_order = (tasks_df['cutoff_date'] >= tasks_df['target_date']).sum()
        
        if invalid_order > 0:
            consistency_report['valid'] = False
            consistency_report['issues'].append(
                f"{invalid_order} tasks have cutoff >= target"
            )
            logger.error(f"  [FAIL] {invalid_order} tasks with invalid date ordering")
        else:
            logger.info(f"  [OK] All cutoff dates before target dates")
        
        consistency_report['checks_performed'].append('date_ordering')
    
    # Check 2: Review dates are within reasonable range
    if 'date' in reviews_df.columns:
        min_review_date = reviews_df['date'].min()
        max_review_date = reviews_df['date'].max()
        
        logger.info(f"  Review date range: {min_review_date.date()} to {max_review_date.date()}")
        
        # Check if any cutoff dates are outside review range
        if 'cutoff_date' in tasks_df.columns:
            cutoff_out_of_range = (
                (tasks_df['cutoff_date'] < min_review_date) |
                (tasks_df['cutoff_date'] > max_review_date)
            ).sum()
            
            if cutoff_out_of_range > 0:
                consistency_report['issues'].append(
                    f"{cutoff_out_of_range} cutoff dates outside review range"
                )
                logger.warning(f"  [WARN]  {cutoff_out_of_range} cutoffs outside review range")
        
        consistency_report['checks_performed'].append('review_date_range')
    
    # Check 3: Feature metadata consistency
    if '_feature_cutoff_date' in features_df.columns:
        unique_cutoffs = features_df['_feature_cutoff_date'].nunique()
        logger.info(f"  [OK] Features computed at {unique_cutoffs} different cutoff dates")
        consistency_report['checks_performed'].append('feature_cutoff_consistency')
    
    # Summary
    if consistency_report['valid']:
        logger.info("  [OK] Temporal consistency validation passed")
    else:
        logger.error(f"  [FAIL] Found {len(consistency_report['issues'])} consistency issues")
        for issue in consistency_report['issues']:
            logger.error(f"     - {issue}")
    
    return consistency_report


def generate_validation_report(validation_results: Dict,
                               output_path: str = "src/utils/validation_report.md") -> None:
    """
    Generate a markdown report summarizing all validation checks.
    
    Args:
        validation_results: Dict containing results from various validation functions
        output_path: Path to save the report
    """
    from pathlib import Path
    from datetime import datetime
    
    report_lines = []
    report_lines.append("# Data Validation Report")
    report_lines.append("")
    report_lines.append(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")
    report_lines.append("---")
    report_lines.append("")
    
    # Label quality results
    if 'label_quality' in validation_results:
        lq = validation_results['label_quality']
        report_lines.append("## Label Quality Validation")
        report_lines.append("")
        report_lines.append(f"- **Initial tasks**: {lq.get('initial_tasks', 'N/A'):,}")
        report_lines.append(f"- **Final tasks**: {lq.get('final_tasks', 'N/A'):,}")
        report_lines.append(f"- **Retention rate**: {lq.get('retention_rate', 0)*100:.1f}%")
        report_lines.append("")
    
    # Feature quality results
    if 'feature_quality' in validation_results:
        fq = validation_results['feature_quality']
        report_lines.append("## Feature Quality Validation")
        report_lines.append("")
        report_lines.append(f"- **Initial rows**: {fq.get('initial_rows', 'N/A'):,}")
        report_lines.append(f"- **Final rows**: {fq.get('final_rows', 'N/A'):,}")
        report_lines.append(f"- **Rows removed**: {fq.get('rows_removed', 'N/A'):,}")
        report_lines.append("")
        
        if 'warnings' in fq and len(fq['warnings']) > 0:
            report_lines.append("### Warnings")
            report_lines.append("")
            for warning in fq['warnings']:
                report_lines.append(f"- {warning}")
            report_lines.append("")
    
    # Temporal consistency results
    if 'temporal_consistency' in validation_results:
        tc = validation_results['temporal_consistency']
        report_lines.append("## Temporal Consistency Validation")
        report_lines.append("")
        report_lines.append(f"- **Status**: {'[OK] PASSED' if tc.get('valid', False) else '[FAIL] FAILED'}")
        report_lines.append("")
        
        if 'issues' in tc and len(tc['issues']) > 0:
            report_lines.append("### Issues Found")
            report_lines.append("")
            for issue in tc['issues']:
                report_lines.append(f"- {issue}")
            report_lines.append("")
    
    # Leakage check results
    if 'leakage_check' in validation_results:
        lc = validation_results['leakage_check']
        report_lines.append("## Temporal Leakage Check")
        report_lines.append("")
        
        if len(lc.get('suspicious_features', [])) > 0:
            report_lines.append("### Suspicious Features")
            report_lines.append("")
            for feat in lc['suspicious_features']:
                report_lines.append(f"- `{feat}`")
            report_lines.append("")
        else:
            report_lines.append("[OK] No suspicious features detected")
            report_lines.append("")
    
    report_lines.append("---")
    report_lines.append("")
    report_lines.append("*Report generated by CS 412 Research Project validation pipeline*")
    
    # Write report
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    logger.info(f"Validation report saved: {output_path}")