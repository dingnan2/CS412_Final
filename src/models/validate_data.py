"""
Data Validation Script: Check if business_features_final.csv is ready for modeling

This script validates:
1. File exists and is readable
2. Required columns are present
3. Data types are correct
4. No critical issues with data quality
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys


def validate_data(data_path: str):
    """
    Validate the feature-engineered dataset.
    
    Args:
        data_path: Path to business_features_final.csv
    
    Returns:
        bool: True if validation passes, False otherwise
    """
    print("="*70)
    print("DATA VALIDATION - business_features_final.csv")
    print("="*70)
    print()
    
    # Check 1: File exists
    print("[OK] CHECK 1: File Existence")
    file_path = Path(data_path)
    if not file_path.exists():
        print(f"[FAIL] ERROR: File not found at {data_path}")
        return False
    print(f"  [OK] File found: {data_path}")
    print(f"  [OK] File size: {file_path.stat().st_size / (1024*1024):.2f} MB")
    print()
    
    # Check 2: Load data
    print("[OK] CHECK 2: Data Loading")
    try:
        df = pd.read_csv(data_path)
        print(f"  [OK] Successfully loaded dataset")
        print(f"  [OK] Shape: {df.shape}")
        print(f"  [OK] Rows: {len(df):,}")
        print(f"  [OK] Columns: {len(df.columns)}")
    except Exception as e:
        print(f"  [FAIL] ERROR loading file: {e}")
        return False
    print()
    
    # Check 3: Required columns
    print("[OK] CHECK 3: Required Columns")
    required_columns = ['business_id', 'is_open']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        print(f"  [FAIL] ERROR: Missing required columns: {missing_columns}")
        return False
    
    print(f"  [OK] business_id column present")
    print(f"  [OK] is_open (target) column present")
    print(f"  [OK] Feature columns: {len(df.columns) - 2}")
    print()
    
    # Check 4: Target variable
    print("[OK] CHECK 4: Target Variable (is_open)")
    if 'is_open' not in df.columns:
        print(f"  [FAIL] ERROR: 'is_open' column not found")
        return False
    
    unique_values = df['is_open'].unique()
    print(f"  [OK] Unique values: {sorted(unique_values)}")
    
    if not set(unique_values).issubset({0, 1}):
        print(f"  [FAIL] ERROR: is_open should only contain 0 and 1")
        return False
    
    open_count = (df['is_open'] == 1).sum()
    closed_count = (df['is_open'] == 0).sum()
    print(f"  [OK] Open (1): {open_count:,} ({open_count/len(df)*100:.2f}%)")
    print(f"  [OK] Closed (0): {closed_count:,} ({closed_count/len(df)*100:.2f}%)")
    print()
    
    # Check 5: Data types
    print("[OK] CHECK 5: Data Types")
    non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    non_numeric_cols = [col for col in non_numeric_cols if col not in ['business_id']]
    
    if non_numeric_cols:
        print(f"  [WARN] WARNING: Non-numeric columns found (excluding business_id):")
        for col in non_numeric_cols[:5]:  # Show first 5
            print(f"    - {col}: {df[col].dtype}")
        if len(non_numeric_cols) > 5:
            print(f"    ... and {len(non_numeric_cols) - 5} more")
        print(f"  [WARN] These will need to be encoded or removed")
    else:
        print(f"  [OK] All feature columns are numeric")
    print()
    
    # Check 6: Missing values
    print("[OK] CHECK 6: Missing Values")
    missing_counts = df.isnull().sum()
    missing_cols = missing_counts[missing_counts > 0]
    
    if len(missing_cols) == 0:
        print(f"  [OK] No missing values detected")
    else:
        print(f"  [WARN] WARNING: Missing values detected in {len(missing_cols)} columns")
        print(f"  Top 5 columns with missing values:")
        for col, count in missing_cols.head(5).items():
            pct = count / len(df) * 100
            print(f"    - {col}: {count:,} ({pct:.2f}%)")
        print(f"  Note: These will be filled with median during modeling")
    print()
    
    # Check 7: Infinite values
    print("[OK] CHECK 7: Infinite Values")
    numeric_df = df.select_dtypes(include=[np.number])
    inf_mask = np.isinf(numeric_df.values)
    inf_count = inf_mask.sum()
    
    if inf_count == 0:
        print(f"  [OK] No infinite values detected")
    else:
        print(f"  [WARN] WARNING: {inf_count} infinite values detected")
        print(f"  Note: These will be replaced with median during modeling")
    print()
    
    # Check 8: Feature variance
    print("[OK] CHECK 8: Feature Variance")
    feature_cols = [col for col in df.columns if col not in ['business_id', 'is_open']]
    variances = df[feature_cols].var()
    zero_var_cols = variances[variances == 0].index.tolist()
    low_var_cols = variances[(variances > 0) & (variances < 0.01)].index.tolist()
    
    if zero_var_cols:
        print(f"  [WARN] WARNING: {len(zero_var_cols)} columns with zero variance")
        print(f"    These columns will be removed during feature selection")
    else:
        print(f"  [OK] No zero-variance columns")
    
    if low_var_cols:
        print(f"  [WARN] INFO: {len(low_var_cols)} columns with low variance (< 0.01)")
        print(f"    These may be removed during feature selection")
    print()
    
    # Check 9: Duplicates
    print("[OK] CHECK 9: Duplicate Rows")
    n_duplicates = df.duplicated(subset='business_id').sum()
    
    if n_duplicates > 0:
        print(f"  [WARN] WARNING: {n_duplicates} duplicate business_id values")
        print(f"    Recommendation: Remove duplicates before modeling")
    else:
        print(f"  [OK] No duplicate business_id values")
    print()
    
    # Check 10: Summary statistics
    print("[OK] CHECK 10: Summary Statistics")
    print(f"  Sample of first 5 rows:")
    print(df.head().to_string())
    print()
    print(f"  Feature summary (excluding business_id, is_open):")
    feature_summary = df[feature_cols].describe()
    print(feature_summary.to_string())
    print()
    
    # Final verdict
    print("="*70)
    print("VALIDATION SUMMARY")
    print("="*70)
    print()
    
    critical_issues = []
    warnings = []
    
    # Collect issues
    if missing_columns:
        critical_issues.append(f"Missing required columns: {missing_columns}")
    
    if not set(unique_values).issubset({0, 1}):
        critical_issues.append("Target variable contains invalid values")
    
    if len(missing_cols) > 0:
        warnings.append(f"{len(missing_cols)} columns with missing values")
    
    if inf_count > 0:
        warnings.append(f"{inf_count} infinite values detected")
    
    if zero_var_cols:
        warnings.append(f"{len(zero_var_cols)} zero-variance columns")
    
    if n_duplicates > 0:
        warnings.append(f"{n_duplicates} duplicate rows")
    
    # Report
    if critical_issues:
        print("[FAIL] VALIDATION FAILED")
        print()
        print("Critical Issues:")
        for issue in critical_issues:
            print(f"  - {issue}")
        print()
        print("Please fix these issues before running baseline models.")
        return False
    
    print("[OK] VALIDATION PASSED")
    print()
    
    if warnings:
        print("Warnings (will be handled during modeling):")
        for warning in warnings:
            print(f"  - {warning}")
        print()
    
    print("Dataset is ready for baseline modeling!")
    print()
    print("Next steps:")
    print("  1. Run: python baseline_models.py")
    print("  2. Check output in: src/models/")
    print()
    
    return True


def main():
    """Main entry point"""
    print()
    
    # Default path
    data_path = r"C:\Users\Dingnan\Desktop\CS412_Final Project\data\features\business_features_final.csv"
    
    # Allow command line override
    if len(sys.argv) > 1:
        data_path = sys.argv[1]
    
    print(f"Validating: {data_path}")
    print()
    
    success = validate_data(data_path)
    
    if success:
        print("="*70)
        print("[OK] ALL CHECKS PASSED - Ready to train baseline models!")
        print("="*70)
        sys.exit(0)
    else:
        print("="*70)
        print("[FAIL] VALIDATION FAILED - Please fix issues before proceeding")
        print("="*70)
        sys.exit(1)


if __name__ == "__main__":
    main()