"""
Global Configuration for CS 412 Research Project

This file contains all hyperparameters and settings used across the project.
Centralizing configuration ensures consistency and reproducibility.

Usage:
    from config import SPLIT_CONFIG, DATA_PATHS, RF_CONFIG, XGBOOST_CONFIG
    
    splitter = TemporalSplitter(**SPLIT_CONFIG)
    model = RandomForestClassifier(**RF_CONFIG)
"""

# ============================================================================
# DATA PATHS - UNIFIED DATA SOURCES FOR ALL PHASES
# ============================================================================

DATA_PATHS = {
    # Raw data
    'raw_business': 'data/raw/yelp_academic_dataset_business.json',
    'raw_review': 'data/raw/yelp_academic_dataset_review.json',
    'raw_user': 'data/raw/yelp_academic_dataset_user.json',
    
    # Processed data
    'business_clean': 'data/processed/business_clean.csv',
    'review_clean': 'data/processed/review_clean.csv',
    'user_clean': 'data/processed/user_clean.csv',
    'review_sentiment': 'data/processed/review_sentiment.csv',
    
    # Feature data - ALL PHASES USE THIS UNIFIED FILE
    'features_temporal': 'data/features/business_features_temporal.csv',
    'features_labeled': 'data/features/business_features_temporal_labeled_12m.csv',
    
    # CRITICAL: All modeling phases (5, 6, 7, 8, 9) must use the same data file
    # Use features_labeled after Phase 4 generates labels
    'model_data': 'data/features/business_features_temporal_labeled_12m.csv',
}

# ============================================================================
# DATASET STATISTICS - For consistent reporting
# ============================================================================

DATASET_STATS = {
    'total_businesses': 150346,
    'total_reviews': 1372781,
    'total_users': 1987897,
    'businesses_after_cleaning': 140858,
    'open_ratio': 0.7901,  # 79.01% open
    'closed_ratio': 0.2099,  # 20.99% closed
    'date_range': '2005-2022',
    'feature_categories': {
        'A_Static': 8,
        'B_Review_Agg': 8,
        'C_Sentiment': 9,
        'D_User_Weighted': 9,
        'E_Temporal': 8,
        'F_Location': 5,
        'G_Interaction': 5,
    },
    'total_features': 52,  # Sum of all categories
}

# ============================================================================
# SPLIT CONFIGURATION - UNIFIED FOR ALL PHASES
# ============================================================================

SPLIT_CONFIG = {
    'split_type': 'temporal_holdout',  # 'temporal_holdout' or 'random'
    'test_size': 0.2,
    'random_state': 42,
    'train_years': [2012, 2013, 2014, 2015, 2016, 2017, 2018],
    'test_years': [2019, 2020]
}

# ============================================================================
# GLOBAL SETTINGS
# ============================================================================

RANDOM_STATE = 42
N_JOBS = -1  # Use all available cores

# ============================================================================
# FEATURE SELECTION CONFIGURATION
# ============================================================================

FEATURE_SELECTION_CONFIG = {
    'corr_threshold': 0.95,      # Remove features with correlation > 0.95
    'variance_threshold': 0.01,   # Remove features with variance < 0.01
    'n_top_features': 40          # Keep top 40 features by importance
}

# ============================================================================
# BASELINE MODEL CONFIGURATIONS
# ============================================================================

# Logistic Regression
LOGISTIC_REGRESSION_CONFIG = {
    'C': 1.0,
    'solver': 'lbfgs',
    'max_iter': 1000,
    'random_state': RANDOM_STATE,
    'n_jobs': N_JOBS
}

# Decision Tree
DECISION_TREE_CONFIG = {
    'max_depth': 10,
    'min_samples_split': 20,
    'min_samples_leaf': 10,
    'random_state': RANDOM_STATE
}

# Random Forest (Baseline)
RF_BASELINE_CONFIG = {
    'n_estimators': 100,
    'max_depth': 15,
    'min_samples_split': 20,
    'min_samples_leaf': 10,
    'class_weight': 'balanced',
    'random_state': RANDOM_STATE,
    'n_jobs': N_JOBS
}

# ============================================================================
# ADVANCED MODEL CONFIGURATIONS (OPTIMIZED)
# ============================================================================

# XGBoost (Optimized)
XGBOOST_CONFIG = {
    'max_depth': 10,              # Increased from 6
    'learning_rate': 0.05,        # Decreased from 0.1 for better convergence
    'n_estimators': 200,          # Increased from 100
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 3,        # Added regularization
    'gamma': 0.1,                 # Added pruning
    'reg_alpha': 0.1,             # L1 regularization
    'reg_lambda': 1.0,            # L2 regularization
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'random_state': RANDOM_STATE,
    'n_jobs': N_JOBS,
    'verbosity': 0
}

# LightGBM (Optimized)
LIGHTGBM_CONFIG = {
    'num_leaves': 31,
    'learning_rate': 0.05,        # Decreased for better convergence
    'n_estimators': 200,          # Increased from 100
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_samples': 20,      # Regularization
    'reg_alpha': 0.1,             # L1 regularization
    'reg_lambda': 0.1,            # L2 regularization
    'objective': 'binary',
    'metric': 'auc',
    'random_state': RANDOM_STATE,
    'n_jobs': N_JOBS,
    'verbosity': -1
}

# Neural Network (Optimized)
NEURAL_NETWORK_CONFIG = {
    'hidden_layer_sizes': (128, 64, 32),  # Increased from (100, 50)
    'activation': 'relu',
    'solver': 'adam',
    'alpha': 0.0001,              # L2 regularization
    'batch_size': 'auto',
    'learning_rate': 'adaptive',
    'learning_rate_init': 0.001,
    'max_iter': 500,              # Increased from 200
    'early_stopping': True,
    'validation_fraction': 0.1,
    'n_iter_no_change': 20,
    'random_state': RANDOM_STATE,
    'verbose': False
}

# ============================================================================
# ENSEMBLE CONFIGURATIONS
# ============================================================================

# Voting Classifier
VOTING_CONFIG = {
    'voting': 'soft',  # Use probability estimates
    'n_jobs': N_JOBS
}

# Stacking Classifier
STACKING_CONFIG = {
    'final_estimator': None,  # Will use LogisticRegression with default params
    'cv': 5,
    'n_jobs': N_JOBS
}

# ============================================================================
# HYPERPARAMETER TUNING GRIDS
# ============================================================================

# XGBoost Grid Search
XGBOOST_PARAM_GRID = {
    'max_depth': [8, 10, 12],
    'learning_rate': [0.03, 0.05, 0.07],
    'n_estimators': [150, 200, 250],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9],
    'min_child_weight': [2, 3, 4],
    'gamma': [0, 0.1, 0.2]
}

# Random Forest Grid Search
RF_PARAM_GRID = {
    'n_estimators': [100, 150, 200],
    'max_depth': [10, 15, 20, None],
    'min_samples_split': [10, 20, 30],
    'min_samples_leaf': [5, 10, 15]
}

# ============================================================================
# CLASS IMBALANCE HANDLING
# ============================================================================

SMOTE_CONFIG = {
    'sampling_strategy': 'auto',  # Balance to 50/50
    'random_state': RANDOM_STATE,
    'k_neighbors': 5,
    'n_jobs': N_JOBS
}

# ============================================================================
# EVALUATION METRICS
# ============================================================================

PRIMARY_METRIC = 'roc_auc'
SECONDARY_METRICS = ['precision', 'recall', 'f1_score']

# ============================================================================
# ABLATION STUDY CONFIGURATION
# ============================================================================

# Random Forest for ablation (consistent with baseline)
RF_ABLATION_CONFIG = {
    'n_estimators': 100,
    'max_depth': 15,
    'min_samples_split': 20,
    'class_weight': 'balanced',
    'random_state': RANDOM_STATE,
    'n_jobs': N_JOBS
}

# ============================================================================
# PARAMETER STUDY CONFIGURATION
# ============================================================================

# Parameter ranges for sensitivity analysis
PARAM_STUDY_RANGES = {
    'max_depth': [3, 5, 7, 10, 15, 20, 25, 30, None],
    'n_estimators': [10, 25, 50, 75, 100, 150, 200, 300],
    'learning_rate': [0.001, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5],
    'min_samples_split': [2, 5, 10, 20, 50, 100, 200],
    'C': [0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 100.0]
}

# ============================================================================
# CASE STUDY CONFIGURATION
# ============================================================================

CASE_STUDY_CONFIG = {
    'n_cases_per_type': 5,  # Number of TP/TN/FP/FN cases to analyze
    'use_shap': False,       # SHAP is too slow for large datasets
    'random_state': RANDOM_STATE
}

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(levelname)s - %(message)s',
    'datefmt': '%Y-%m-%d %H:%M:%S'
}

# ============================================================================
# PLOTTING CONFIGURATION
# ============================================================================

PLOT_CONFIG = {
    'style': 'seaborn-v0_8-darkgrid',
    'palette': 'husl',
    'figure_size': (12, 8),
    'dpi': 300,
    'font_size': 12
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_model_config(model_name: str) -> dict:
    """
    Get configuration for a specific model.
    
    Args:
        model_name: Name of the model
        
    Returns:
        Dictionary of model hyperparameters
    """
    configs = {
        'logistic_regression': LOGISTIC_REGRESSION_CONFIG,
        'decision_tree': DECISION_TREE_CONFIG,
        'random_forest': RF_BASELINE_CONFIG,
        'xgboost': XGBOOST_CONFIG,
        'lightgbm': LIGHTGBM_CONFIG,
        'neural_network': NEURAL_NETWORK_CONFIG
    }
    
    if model_name.lower() not in configs:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(configs.keys())}")
    
    return configs[model_name.lower()].copy()


def get_model_data_path() -> str:
    """
    Get the unified model data path.
    
    All modeling phases (5, 6, 7, 8, 9) should use this path for consistency.
    
    Returns:
        Path to the labeled temporal features CSV
    """
    return DATA_PATHS['model_data']


def get_split_config() -> dict:
    """
    Get the unified split configuration.
    
    All phases should use this configuration to ensure consistent train/test splits.
    
    Returns:
        Dictionary with split configuration
    """
    return SPLIT_CONFIG.copy()


def print_config_summary():
    """Print a summary of current configuration."""
    print("="*70)
    print("CS 412 PROJECT CONFIGURATION SUMMARY")
    print("="*70)
    
    print(f"\nData Paths:")
    print(f"  Model data: {DATA_PATHS['model_data']}")
    print(f"  Features temporal: {DATA_PATHS['features_temporal']}")
    
    print(f"\nDataset Statistics:")
    print(f"  Total businesses: {DATASET_STATS['total_businesses']:,}")
    print(f"  After cleaning: {DATASET_STATS['businesses_after_cleaning']:,}")
    print(f"  Open ratio: {DATASET_STATS['open_ratio']*100:.2f}%")
    print(f"  Closed ratio: {DATASET_STATS['closed_ratio']*100:.2f}%")
    print(f"  Total features: {DATASET_STATS['total_features']}")
    
    print(f"\nSplit Configuration:")
    print(f"  Type: {SPLIT_CONFIG['split_type']}")
    print(f"  Train years: {SPLIT_CONFIG['train_years']}")
    print(f"  Test years: {SPLIT_CONFIG['test_years']}")
    print(f"  Random state: {RANDOM_STATE}")
    
    print(f"\nModel Configurations:")
    print(f"  Random Forest: n_estimators={RF_BASELINE_CONFIG['n_estimators']}, "
          f"max_depth={RF_BASELINE_CONFIG['max_depth']}")
    print(f"  XGBoost: n_estimators={XGBOOST_CONFIG['n_estimators']}, "
          f"max_depth={XGBOOST_CONFIG['max_depth']}, "
          f"learning_rate={XGBOOST_CONFIG['learning_rate']}")
    print(f"  LightGBM: n_estimators={LIGHTGBM_CONFIG['n_estimators']}, "
          f"num_leaves={LIGHTGBM_CONFIG['num_leaves']}")
    print(f"  Neural Network: layers={NEURAL_NETWORK_CONFIG['hidden_layer_sizes']}, "
          f"max_iter={NEURAL_NETWORK_CONFIG['max_iter']}")
    
    print(f"\nFeature Selection:")
    print(f"  Correlation threshold: {FEATURE_SELECTION_CONFIG['corr_threshold']}")
    print(f"  Top features: {FEATURE_SELECTION_CONFIG['n_top_features']}")
    
    print("="*70)


if __name__ == "__main__":
    # Print configuration summary
    print_config_summary()
    
    # Test get_model_config
    print("\nTesting get_model_config()...")
    xgb_config = get_model_config('xgboost')
    print(f"XGBoost config: {xgb_config}")
    
    # Test get_model_data_path
    print(f"\nModel data path: {get_model_data_path()}")

