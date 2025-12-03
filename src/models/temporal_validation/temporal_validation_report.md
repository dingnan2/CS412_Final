# Temporal Validation Report (V2 - Leakage-Free)

**Generated:** 2025-12-02 02:47:49

---

## Methodology Changes (V2)

### Label Generation
- **V1 (Old)**: Inferred labels from review activity patterns
- **V2 (New)**: Uses ground truth `is_open` status directly
- **Benefit**: Eliminates circular dependency between features and labels

### Temporal Split
- **V1 (Old)**: 80/20 split within each year
- **V2 (New)**: Temporal holdout (train on past, test on future)
- **Benefit**: True temporal prediction, no data leakage

---

## Split Configuration

- **Split Type**: temporal_holdout
- **Train Years**: [2012, 2013, 2014, 2015, 2016, 2017, 2018]
- **Test Years**: [2019, 2020]
- **Train Samples**: 76,622
- **Test Samples**: 27,405

## Summary

- **Total prediction tasks**: 104,027
- **Unique businesses**: 26,250
- **Prediction years**: [2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020]

## Model Performance

| Model | ROC-AUC | Precision | Recall | F1 |
|-------|---------|-----------|--------|-----|
| LogisticRegression | 0.6840 | 0.9540 | 0.8699 | 0.9100 |
| DecisionTree | 0.6940 | 0.9656 | 0.6370 | 0.7676 |
| RandomForest | 0.8417 | 0.9576 | 0.9388 | 0.9481 |

**Best Model**: RandomForest (ROC-AUC: 0.8417)

## Expected Performance Range

With leakage-free temporal validation, realistic performance is:
- **ROC-AUC**: 0.65 - 0.80
- **F1-Score**: 0.70 - 0.85

If performance exceeds these ranges significantly (e.g., > 0.90),
there may still be issues with the evaluation methodology.
