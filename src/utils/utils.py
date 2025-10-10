"""
Utility functions for CS 412 Research Project
"""

import pandas as pd
import numpy as np
import json
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/project.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def load_json_data(file_path: str) -> List[Dict[str, Any]]:
    """Load JSON data from file"""
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                data.append(json.loads(line.strip()))
    except FileNotFoundError:
        logging.error(f"File not found: {file_path}")
        raise
    except json.JSONDecodeError as e:
        logging.error(f"JSON decode error in {file_path}: {e}")
        raise
    
    return data


def save_json_data(data: List[Dict[str, Any]], file_path: str):
    """Save data to JSON file"""
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(file_path, 'w', encoding='utf-8') as file:
        for item in data:
            json.dump(item, file, ensure_ascii=False)
            file.write('\n')


def convert_to_dataframe(data: List[Dict[str, Any]]) -> pd.DataFrame:
    """Convert list of dictionaries to pandas DataFrame"""
    return pd.DataFrame(data)


def calculate_class_weights(y: np.ndarray) -> Dict[int, float]:
    """Calculate class weights for imbalanced datasets"""
    from sklearn.utils.class_weight import compute_class_weight
    
    classes = np.unique(y)
    weights = compute_class_weight('balanced', classes=classes, y=y)
    return dict(zip(classes, weights))


def create_time_windows(df: pd.DataFrame, 
                       time_col: str, 
                       windows: List[int]) -> pd.DataFrame:
    """Create time-based features for different windows"""
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col])
    
    for window in windows:
        # Create rolling features for each window
        df[f'count_{window}d'] = df.groupby('business_id')[time_col].rolling(
            f'{window}d', on=time_col
        ).count().reset_index(0, drop=True)
        
        df[f'avg_rating_{window}d'] = df.groupby('business_id')['stars'].rolling(
            f'{window}d', on=time_col
        ).mean().reset_index(0, drop=True)
    
    return df


def safe_divide(numerator: np.ndarray, 
                denominator: np.ndarray, 
                default: float = 0.0) -> np.ndarray:
    """Safely divide arrays, handling division by zero"""
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.divide(numerator, denominator)
        result[np.isnan(result) | np.isinf(result)] = default
    return result


def extract_categorical_features(df: pd.DataFrame, 
                               categorical_cols: List[str]) -> pd.DataFrame:
    """Extract categorical features using one-hot encoding"""
    df_encoded = df.copy()
    
    for col in categorical_cols:
        if col in df.columns:
            # Handle list-type categories (like business categories)
            if df[col].dtype == 'object' and df[col].str.contains(',').any():
                # Split comma-separated categories
                categories = df[col].str.split(',').explode().str.strip()
                dummies = pd.get_dummies(categories, prefix=col)
                df_encoded = pd.concat([df_encoded, dummies], axis=1)
            else:
                # Regular one-hot encoding
                dummies = pd.get_dummies(df[col], prefix=col)
                df_encoded = pd.concat([df_encoded, dummies], axis=1)
    
    return df_encoded


def validate_dataframe(df: pd.DataFrame, 
                      required_cols: List[str]) -> bool:
    """Validate DataFrame has required columns"""
    missing_cols = set(required_cols) - set(df.columns)
    if missing_cols:
        logging.error(f"Missing required columns: {missing_cols}")
        return False
    return True


def print_data_summary(df: pd.DataFrame, name: str = "Dataset"):
    """Print summary statistics of DataFrame"""
    print(f"\n{name} Summary:")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    if len(df) > 0:
        print(f"Date range: {df.index.min()} to {df.index.max()}")
        print(f"Missing values: {df.isnull().sum().sum()}")
    
    print("\nFirst few rows:")
    print(df.head())
