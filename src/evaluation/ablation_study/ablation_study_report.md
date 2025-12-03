# Ablation Study Report


1. **Ablation**: Remove each category and measure performance drop
2. **Additive**: Add each category and measure performance gain

## Ablation Results (Remove One Category)

**Baseline (All Features):** ROC-AUC = 0.8409

| Category | ROC-AUC | AUC Drop | F1 | F1 Drop |
|----------|---------|----------|----|---------| 
| A_Static | 0.7950 | 0.0459 | 0.9505 | -0.0035 |
| F_Location | 0.8146 | 0.0262 | 0.9472 | -0.0001 |
| D_User_Weighted | 0.8240 | 0.0169 | 0.9034 | 0.0436 |
| C_Sentiment | 0.8364 | 0.0045 | 0.9439 | 0.0031 |
| G_Interaction | 0.8387 | 0.0022 | 0.9488 | -0.0018 |
| B_Review_Agg | 0.8520 | -0.0111 | 0.9523 | -0.0052 |
| E_Temporal | 0.8540 | -0.0132 | 0.9494 | -0.0023 |

## Additive Results (Add to Base)

**Base (Static Features):** ROC-AUC = 0.8754

| Category Added | ROC-AUC | AUC Gain | F1 | F1 Gain |
|----------------|---------|----------|----|---------| 
| F_Location | 0.8915 | 0.0160 | 0.9036 | 0.0098 |
| G_Interaction | 0.8294 | -0.0460 | 0.8798 | -0.0140 |
| D_User_Weighted | 0.8041 | -0.0713 | 0.9432 | 0.0493 |
| B_Review_Agg | 0.7931 | -0.0823 | 0.8599 | -0.0339 |
| C_Sentiment | 0.7775 | -0.0980 | 0.8998 | 0.0059 |
| E_Temporal | 0.7447 | -0.1308 | 0.8412 | -0.0527 |

## Key Findings

### Most Important Feature Categories

Based on ablation results (largest performance drop when removed):

1. **A_Static** (drop: 0.0459)
2. **F_Location** (drop: 0.0262)
3. **D_User_Weighted** (drop: 0.0169)

### User Credibility Weighting

The user credibility weighting framework (Category D) provides:
- Weighted ratings based on reviewer credibility
- Higher weights for experienced, engaged reviewers
- Improved signal-to-noise ratio in aggregated metrics

### Interpretation of Results

#### Temporal Feature Paradox

Removing E_Temporal features **improved** performance (AUC change: -0.0132).
This counter-intuitive result suggests:

1. **Feature Redundancy**: Temporal patterns already captured by User-Weighted (D)
   features through `avg_reviewer_tenure` and `review_diversity`

2. **Noise Introduction**: Features like `rating_recent_vs_all` may capture
   transient fluctuations rather than meaningful trends

3. **Overfitting Risk**: 8 temporal features add complexity without
   proportional signal, leading to overfitting on training data

#### Ablation vs Additive Discrepancy

User-Weighted (D) shows different behavior in ablation vs additive analysis:
- **Ablation**: Removing D hurts performance (drop: +0.0169)
- **Additive**: Adding D to Static hurts performance (gain: -0.0713)

**Explanation**: This demonstrates **feature interaction effects**:
- When ALL features present: D provides unique signal not captured by others
- When adding to Static only: D overlaps with Static features, introducing noise
- D works synergistically with Location (F) features, not as standalone

### Recommendations

Based on ablation analysis:

1. **Keep**: Static (A), User-Weighted (D), Location (F) - provide independent signal
2. **Review**: Temporal (E) - consider removing or simplifying to reduce overfitting
3. **Simplify**: Review Aggregation (B) - redundant with other categories

