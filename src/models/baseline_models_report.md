# Baseline Models Report


This report presents the results of baseline model training for business success prediction.
We trained **6 models** on 76,622 training samples
and evaluated them on 27,405 test samples.

**Best Model:** RandomForest_ClassWeight
**Best ROC-AUC:** 0.8347

## Model Performance Comparison

| Model | ROC-AUC | PR-AUC | Precision | Recall | F1-Score |
|-------|---------|--------|-----------|--------|----------|
| RandomForest_ClassWeight | 0.8347 | 0.9861 | 0.9591 | 0.9238 | 0.9411 |
| RandomForest_SMOTE | 0.8253 | 0.9853 | 0.9628 | 0.8961 | 0.9283 |
| DecisionTree_ClassWeight | 0.6993 | 0.9684 | 0.9654 | 0.6428 | 0.7717 |
| DecisionTree_SMOTE | 0.6925 | 0.9685 | 0.9602 | 0.7468 | 0.8402 |
| LogisticRegression_SMOTE | 0.6845 | 0.9653 | 0.9535 | 0.8690 | 0.9093 |
| LogisticRegression_ClassWeight | 0.6840 | 0.9657 | 0.9540 | 0.8699 | 0.9100 |

## Class Imbalance Handling Comparison

### SMOTE vs Class Weights

| Algorithm | SMOTE ROC-AUC | ClassWeight ROC-AUC | Difference |
|-----------|---------------|---------------------|------------|
| LogisticRegression | 0.6845 | 0.6840 | -0.0005 |
| DecisionTree | 0.6925 | 0.6993 | +0.0068 |
| RandomForest | 0.8253 | 0.8347 | +0.0094 |

## Confusion Matrix Analysis

### Best Model: RandomForest_ClassWeight

| | Predicted Closed | Predicted Open |
|---|-----------------|----------------|
| **Actual Closed** | TN: 684 (2.5%) | FP: 1,012 (3.7%) |
| **Actual Open** | FN: 1,959 (7.1%) | TP: 23,750 (86.7%) |

**Interpretation:**
- **True Negatives (TN)**: 684 closed businesses correctly identified
- **True Positives (TP)**: 23,750 open businesses correctly identified
- **False Positives (FP)**: 1,012 closed businesses incorrectly predicted as open
- **False Negatives (FN)**: 1,959 open businesses incorrectly predicted as closed

## Feature Importance Analysis

Top 15 features from Random Forest model:

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | review_count | 0.1098 |
| 2 | avg_reviewer_tenure | 0.0611 |
| 3 | review_frequency | 0.0584 |
| 4 | size_activity_interaction | 0.0524 |
| 5 | city_avg_success_rate | 0.0308 |
| 6 | days_since_first_review | 0.0303 |
| 7 | avg_useful_per_review | 0.0259 |
| 8 | category_encoded | 0.0257 |
| 9 | weighted_avg_rating | 0.0237 |
| 10 | category_avg_success_rate | 0.0237 |
| 11 | std_review_stars | 0.0233 |
| 12 | avg_reviewer_experience | 0.0233 |
| 13 | weighted_sentiment | 0.0232 |
| 14 | avg_review_stars | 0.0221 |
| 15 | avg_text_length | 0.0221 |

## Individual Model Details

### RandomForest_ClassWeight

- **ROC-AUC**: 0.8347
- **PR-AUC**: 0.9861
- **Precision**: 0.9591
- **Recall**: 0.9238
- **F1-Score**: 0.9411

Confusion Matrix: TN=684, FP=1,012, FN=1,959, TP=23,750

### RandomForest_SMOTE

- **ROC-AUC**: 0.8253
- **PR-AUC**: 0.9853
- **Precision**: 0.9628
- **Recall**: 0.8961
- **F1-Score**: 0.9283

Confusion Matrix: TN=807, FP=889, FN=2,672, TP=23,037

### DecisionTree_ClassWeight

- **ROC-AUC**: 0.6993
- **PR-AUC**: 0.9684
- **Precision**: 0.9654
- **Recall**: 0.6428
- **F1-Score**: 0.7717

Confusion Matrix: TN=1,103, FP=593, FN=9,183, TP=16,526

### DecisionTree_SMOTE

- **ROC-AUC**: 0.6925
- **PR-AUC**: 0.9685
- **Precision**: 0.9602
- **Recall**: 0.7468
- **F1-Score**: 0.8402

Confusion Matrix: TN=901, FP=795, FN=6,509, TP=19,200

### LogisticRegression_SMOTE

- **ROC-AUC**: 0.6845
- **PR-AUC**: 0.9653
- **Precision**: 0.9535
- **Recall**: 0.8690
- **F1-Score**: 0.9093

Confusion Matrix: TN=607, FP=1,089, FN=3,369, TP=22,340

### LogisticRegression_ClassWeight

- **ROC-AUC**: 0.6840
- **PR-AUC**: 0.9657
- **Precision**: 0.9540
- **Recall**: 0.8699
- **F1-Score**: 0.9100

Confusion Matrix: TN=617, FP=1,079, FN=3,345, TP=22,364

## Generated Visualizations

The following visualizations have been generated:

1. **model_comparison.png**: Bar charts comparing all models across metrics
2. **roc_curves.png**: ROC curves for SMOTE and ClassWeight approaches
3. **precision_recall_curves.png**: PR curves for all models
4. **confusion_matrices.png**: Confusion matrices for all 6 models
5. **random_forest_feature_importance.png**: Feature importance from RF models
6. **class_distribution.png**: Class distribution in train/test sets
7. **feature_importance_selection.png**: Feature selection importance scores

## Key Findings

1. **Random Forest outperforms other baselines**: Ensemble methods capture
   non-linear feature interactions better than linear models or single trees.

2. **Class Weight vs SMOTE**: Both approaches yield similar results.
   RF with ClassWeight (0.8347) slightly outperforms RF with SMOTE (0.8253).

3. **Linear model limitations**: Logistic Regression (0.6840 AUC) shows
   limited performance, suggesting non-linear relationships in the data.

4. **Class imbalance impact**: All models show higher precision than recall,
   indicating conservative predictions for the minority (closed) class.


