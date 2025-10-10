"""
Quick baseline experiment for midterm report
"""

import sys
import os
import pandas as pd
import numpy as np
import json
import logging
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def setup_logging():
    """Setup logging"""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    return logging.getLogger(__name__)

def load_sample_data(n_samples=5000):
    """Load sample data from Yelp dataset"""
    logger = setup_logging()
    
    logger.info(f"Loading {n_samples} sample businesses...")
    
    data_dir = Path("data/raw")
    business_file = data_dir / "yelp_academic_dataset_business.json"
    
    data = []
    with open(business_file, 'r', encoding='utf-8') as file:
        for i, line in enumerate(file):
            if i >= n_samples:
                break
            data.append(json.loads(line.strip()))
    
    df = pd.DataFrame(data)
    
    # Clean and prepare data
    df = df.dropna(subset=['stars', 'review_count', 'is_open'])
    df['is_open'] = df['is_open'].astype(int)
    
    logger.info(f"Loaded {len(df)} businesses")
    logger.info(f"Open: {df['is_open'].sum()}, Closed: {(df['is_open']==0).sum()}")
    
    return df

def create_features(df):
    """Create basic features for modeling"""
    logger = setup_logging()
    
    logger.info("Creating features...")
    
    # Basic features
    features = pd.DataFrame({
        'stars': df['stars'],
        'review_count': df['review_count'],
        'stars_squared': df['stars'] ** 2,
        'review_count_log': np.log1p(df['review_count']),
        'stars_review_interaction': df['stars'] * df['review_count']
    })
    
    # Category features (simplified)
    if 'categories' in df.columns:
        # Extract main category
        df['main_category'] = df['categories'].str.split(',').str[0].str.strip()
        
        # Top categories
        top_categories = df['main_category'].value_counts().head(10).index
        
        for category in top_categories:
            features[f'category_{category}'] = (df['main_category'] == category).astype(int)
    
    # Location features (simplified)
    if 'latitude' in df.columns and 'longitude' in df.columns:
        features['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
        features['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
    
    # Fill missing values
    features = features.fillna(0)
    
    logger.info(f"Created {features.shape[1]} features")
    
    return features

def run_baseline_experiment():
    """Run baseline experiment"""
    logger = setup_logging()
    
    print("🚀 CS 412 Research Project - Baseline Experiment")
    print("=" * 50)
    
    # Load data
    df = load_sample_data(5000)
    
    # Create features
    X = create_features(df)
    y = df['is_open']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    logger.info(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
    
    # Define models
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=10),
        'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100)
    }
    
    # Train and evaluate models
    results = {}
    
    for name, model in models.items():
        logger.info(f"Training {name}...")
        
        # Train
        model.fit(X_train, y_train)
        
        # Predict
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else None
        
        results[name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc
        }
        
        roc_auc_str = f"{roc_auc:.4f}" if roc_auc is not None else "N/A"
        logger.info(f"{name} - Accuracy: {accuracy:.4f}, F1: {f1:.4f}, ROC-AUC: {roc_auc_str}")
    
    # Create results summary
    results_df = pd.DataFrame(results).T
    results_df = results_df.sort_values('roc_auc', ascending=False)
    
    print("\n📊 BASELINE RESULTS SUMMARY:")
    print("=" * 30)
    print(results_df.round(4))
    
    # Save results
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    results_df.to_csv(results_dir / "baseline_results.csv")
    logger.info(f"Results saved to {results_dir / 'baseline_results.csv'}")
    
    # Feature importance (for tree-based models)
    if 'Random Forest' in models:
        rf_model = models['Random Forest']
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': rf_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\n🔍 TOP 10 FEATURE IMPORTANCE (Random Forest):")
        print("=" * 45)
        print(feature_importance.head(10).round(4))
        
        feature_importance.to_csv(results_dir / "feature_importance.csv", index=False)
        logger.info(f"Feature importance saved to {results_dir / 'feature_importance.csv'}")
    
    print("\n✅ Baseline experiment completed!")
    print("\nThese results can be used in your midterm report!")
    
    return results_df

if __name__ == "__main__":
    results = run_baseline_experiment()
