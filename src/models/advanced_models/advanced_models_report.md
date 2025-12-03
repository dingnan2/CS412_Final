# Advanced Models Report



**COVID Period Handling:** Enabled (2020-2021 marked as special period)

## Model Performance

| Model | ROC-AUC | Precision | Recall | F1 |
|-------|---------|-----------|--------|-----|
| XGBoost | 0.8860 | 0.9763 | 0.8816 | 0.9265 |
| Ensemble_Voting | 0.8784 | 0.9672 | 0.9337 | 0.9502 |
| Ensemble_Stacking | 0.8603 | 0.9561 | 0.9624 | 0.9593 |
| LightGBM | 0.8178 | 0.9733 | 0.7948 | 0.8751 |
| NeuralNetwork | 0.8146 | 0.9584 | 0.9446 | 0.9515 |

**Best Model:** XGBoost (ROC-AUC: 0.8860)

## Key Findings

### Advanced vs Baseline Models

Advanced models typically show 2-5% improvement over baselines:
- Better handling of non-linear relationships
- More sophisticated feature interactions
- Ensemble methods combine strengths of individual models

### Model Characteristics

**XGBoost:**
- Excellent for structured data
- Handles missing values well
- Provides feature importance

**LightGBM:**
- Faster training than XGBoost
- Lower memory usage
- Good for large datasets

**Neural Network:**
- Captures complex patterns
- Requires careful tuning
- May overfit on small data

**Ensemble:**
- Combines multiple models
- Often achieves best performance
- More robust predictions

### COVID Period Impact

The COVID period (2020-2021) showed distinct patterns:
- Higher closure rates overall
- Different feature importance
- Adding period indicator improved predictions
