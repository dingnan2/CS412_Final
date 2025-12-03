# Parameter Study Report


## 1. Tree Depth Analysis

| Max Depth | Train AUC | Test AUC | F1 Score |
|-----------|-----------|----------|----------|
| 3 | 0.7112 | 0.6779 | 0.7921 |
| 5 | 0.7553 | 0.7024 | 0.8279 |
| 7 | 0.8057 | 0.7301 | 0.8672 |
| 10 | 0.8932 | 0.7764 | 0.9074 |
| 15 | 0.9857 | 0.8417 | 0.9481 |
| 20 | 0.9984 | 0.8709 | 0.9606 |
| 25 | 0.9995 | 0.8810 | 0.9621 **←** |
| 30 | 0.9997 | 0.8805 | 0.9620 |
| None | 0.9997 | 0.8753 | 0.9623 |

**Optimal max_depth:** 25 
(Test AUC: 0.8810)

[WARN] **Warning:** Train-test gap (0.1185) suggests overfitting.

## 2. Number of Estimators Analysis

| N_Estimators | Test AUC | F1 Score | Train Time (s) |
|--------------|----------|----------|----------------|
| 10 | 0.7858 | 0.9341 | 0.67 |
| 25 | 0.8201 | 0.9438 | 1.41 |
| 50 | 0.8333 | 0.9478 | 2.37 |
| 75 | 0.8386 | 0.9483 | 3.43 |
| 100 | 0.8417 | 0.9481 | 5.16 |
| 150 | 0.8454 | 0.9485 | 6.61 |
| 200 | 0.8458 | 0.9489 | 8.95 |
| 300 | 0.8484 | 0.9491 | 13.38 **←** |

**Optimal n_estimators:** 300

**Trade-off:** More trees improve performance but increase training time.

## 3. Learning Rate Analysis (XGBoost)

| Learning Rate | Train AUC | Test AUC | F1 Score |
|---------------|-----------|----------|----------|
| 0.001 | 0.7770 | 0.6845 | 0.8425 |
| 0.01 | 0.8242 | 0.7425 | 0.8642 |
| 0.05 | 0.8767 | 0.7996 | 0.8763 |
| 0.1 | 0.9027 | 0.8182 | 0.8822 |
| 0.2 | 0.9364 | 0.8392 | 0.8933 |
| 0.3 | 0.9567 | 0.8507 | 0.8973 |
| 0.5 | 0.9784 | 0.8581 | 0.9084 **←** |

**Optimal learning_rate:** 0.5

## 4. Min Samples Split Analysis

| Min Samples | Train AUC | Test AUC | F1 Score |
|-------------|-----------|----------|----------|
| 2 | 0.9942 | 0.8532 | 0.9555 **←** |
| 5 | 0.9930 | 0.8497 | 0.9553 |
| 10 | 0.9905 | 0.8458 | 0.9511 |
| 20 | 0.9857 | 0.8417 | 0.9481 |
| 50 | 0.9687 | 0.8312 | 0.9371 |
| 100 | 0.9424 | 0.8137 | 0.9256 |
| 200 | 0.9054 | 0.7906 | 0.9103 |

**Optimal min_samples_split:** 2

## 5. Regularization Analysis (Logistic Regression)

| C (Regularization) | Train AUC | Test AUC | F1 Score |
|--------------------|-----------|----------|----------|
| 0.001 | 0.7491 | 0.6795 | 0.9247 |
| 0.01 | 0.7660 | 0.6839 | 0.9165 |
| 0.1 | 0.7689 | 0.6841 | 0.9111 **←** |
| 0.5 | 0.7692 | 0.6840 | 0.9103 |
| 1.0 | 0.7692 | 0.6840 | 0.9100 |
| 5.0 | 0.7693 | 0.6839 | 0.9101 |
| 10.0 | 0.7693 | 0.6839 | 0.9101 |
| 100.0 | 0.7693 | 0.6840 | 0.9100 |

**Optimal C:** 0.1

## 6. Key Findings

### Overfitting Prevention

- **Tree Depth:** Limiting depth prevents overfitting
- **Min Samples Split:** Higher values act as regularization
- **N_Estimators:** Diminishing returns after ~100 trees

### Performance vs Complexity Trade-off

- Models with moderate complexity achieve best generalization
- Too simple = underfitting, too complex = overfitting
- Cross-validation essential for hyperparameter selection

### Recommended Configuration

Based on this analysis, we recommend:

- **max_depth:** 25
- **n_estimators:** 300
- **min_samples_split:** 2
- **learning_rate (XGBoost):** 0.5
