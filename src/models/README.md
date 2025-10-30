# Baseline Models - Setup and Execution Guide

## 📋 Overview

This module trains and evaluates three baseline models for business success prediction:
1. **Logistic Regression** - Linear benchmark
2. **Decision Tree** - Non-linear with interpretable rules
3. **Random Forest** - Ensemble method

Each model is trained with two class imbalance handling approaches:
- **SMOTE** (Synthetic Minority Over-sampling Technique)
- **Class Weights** (Weighted loss function)

**Total: 6 models trained and evaluated**

---

## 🔧 Prerequisites

### Required Python Packages

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn
```

Or install from requirements file:

```bash
pip install -r requirements.txt
```

### Required Data

Ensure you have the feature-engineered dataset at:
```
C:\Users\Dingnan\Desktop\CS412_Final Project\data\features\business_features_final.csv
```

---

## 🚀 Quick Start

### Option 1: Run directly with Python

```bash
python baseline_models.py
```

### Option 2: Run from command line with custom path

```python
from baseline_models import BaselineModelPipeline

pipeline = BaselineModelPipeline(
    data_path="path/to/business_features_final.csv",
    output_path="src/models",
    random_state=42
)

pipeline.run_pipeline()
```

---

## 📊 Pipeline Stages

### Stage 1: Data Loading
- Loads `business_features_final.csv`
- Separates features (X) and target (y)
- Checks for missing/infinite values
- **Input:** 150,346 samples × 72 features
- **Output:** Clean dataset ready for modeling

### Stage 2: Feature Selection
Three-stage selection process:
1. **Correlation filtering:** Remove features with |r| > 0.95
2. **Variance filtering:** Remove features with var < 0.01
3. **Importance ranking:** Select top 40 features by Random Forest importance

**Output:** 40 most predictive features

### Stage 3: Train-Test Split
- **Split ratio:** 80% train, 20% test
- **Method:** Stratified split (maintains 80/20 class ratio)
- **Scaling:** StandardScaler (mean=0, std=1)

**Output:** 
- Train: 120,277 samples
- Test: 30,069 samples

### Stage 4: Model Training

**Approach A: SMOTE**
- Apply SMOTE to balance training data
- Train 3 models on balanced dataset

**Approach B: Class Weights**
- Use class weights in loss function
- Train 3 models on original dataset

**Hyperparameters:**
- Logistic Regression: C=1.0, max_iter=1000
- Decision Tree: max_depth=10, min_samples_split=20
- Random Forest: n_estimators=100, max_depth=15

### Stage 5: Model Evaluation

**Metrics:**
- ROC-AUC (primary metric)
- Precision-Recall AUC
- Precision, Recall, F1-Score
- Confusion Matrix

### Stage 6: Visualization Generation

**Generated plots:**
1. `model_comparison.png` - Performance comparison across all metrics
2. `roc_curves.png` - ROC curves for all models
3. `precision_recall_curves.png` - PR curves for all models
4. `confusion_matrices.png` - 6 confusion matrices
5. `feature_importance_selection.png` - Top 20 features by importance
6. `random_forest_feature_importance.png` - RF feature importance
7. `class_distribution.png` - Class distribution analysis

### Stage 7: Report Generation

**Output files:**
- `baseline_models_report.md` - Comprehensive markdown report
- `results_summary.json` - Metrics in JSON format
- `baseline_models.log` - Execution log

---

## 📁 Output Structure

```
src/models/
│
├── baseline_models_report.md        [Main Report - Comprehensive analysis]
├── results_summary.json             [Metrics Summary - JSON format]
├── baseline_models.log              [Execution Log]
│
├── plots/                           [Visualizations]
│   ├── model_comparison.png
│   ├── roc_curves.png
│   ├── precision_recall_curves.png
│   ├── confusion_matrices.png
│   ├── feature_importance_selection.png
│   ├── random_forest_feature_importance.png
│   └── class_distribution.png
│
└── saved_models/                    [Trained Models]
    ├── LogisticRegression_SMOTE.pkl
    ├── LogisticRegression_ClassWeight.pkl
    ├── DecisionTree_SMOTE.pkl
    ├── DecisionTree_ClassWeight.pkl
    ├── RandomForest_SMOTE.pkl
    ├── RandomForest_ClassWeight.pkl
    └── scaler.pkl
```

---

## ⏱️ Expected Runtime

- **Full pipeline:** ~15-20 minutes
- **Breakdown:**
  - Data loading: ~1 min
  - Feature selection: ~3 min
  - Model training: ~10 min
  - Evaluation & visualization: ~5 min

**Note:** Runtime varies based on CPU and RAM

---

## 🔍 Key Outputs to Review

### 1. Main Report (`baseline_models_report.md`)
- Executive summary with best model
- Feature selection analysis
- Detailed performance metrics
- Model comparisons
- Key findings and recommendations

### 2. Model Comparison Plot
Quick visual comparison of all 6 models across 4 metrics

### 3. ROC Curves
Shows discriminative power of each model

### 4. Confusion Matrices
Detailed breakdown of predictions

### 5. Feature Importance
Top predictive features identified

---

## 🎯 Understanding the Results

### ROC-AUC Score Interpretation:
- **0.90-1.00:** Excellent
- **0.80-0.90:** Good
- **0.70-0.80:** Fair
- **0.60-0.70:** Poor
- **0.50-0.60:** Fail (random guessing)

### Class Labels:
- **1 (Open):** Business is currently open (~80% of data)
- **0 (Closed):** Business is closed (~20% of data)

### Confusion Matrix Terms:
- **TN (True Negative):** Correctly predicted closed
- **FP (False Positive):** Predicted closed but actually open
- **FN (False Negative):** Predicted open but actually closed
- **TP (True Positive):** Correctly predicted open

---

## 🐛 Troubleshooting

### Issue: File not found error
**Solution:** Update `data_path` in `baseline_models.py` line 1142

### Issue: Out of memory error
**Solution:** 
- Close other applications
- Reduce `n_estimators` in Random Forest
- Use smaller sample for testing

### Issue: SMOTE import error
**Solution:** 
```bash
pip install imbalanced-learn
```

### Issue: Plots not displaying
**Solution:** Check `src/models/plots/` directory for saved images

---

## 📈 Next Steps After Baseline

1. **Hyperparameter Tuning:**
   - Use GridSearchCV on best performing model
   - Optimize threshold for business objectives

2. **Advanced Models:**
   - Try XGBoost/LightGBM
   - Implement neural networks
   - Create ensemble stacking

3. **Model Interpretation:**
   - Generate SHAP values
   - Analyze prediction patterns
   - Identify key decision factors

4. **Deployment Preparation:**
   - Create prediction API
   - Set up monitoring
   - Prepare documentation

---

## 📞 Support

For questions or issues:
1. Check the log file: `baseline_models.log`
2. Review error messages in console output
3. Verify all prerequisites are installed
4. Ensure data path is correct

---

## ✅ Checklist Before Running

- [ ] Python 3.7+ installed
- [ ] All required packages installed
- [ ] `business_features_final.csv` exists and is accessible
- [ ] At least 4GB RAM available
- [ ] ~2GB disk space for outputs
