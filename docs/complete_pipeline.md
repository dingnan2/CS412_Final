# CS 412 Research Project: Business Success Prediction - Complete Pipeline Documentation

**Last Updated:** 2025-12-02  
**Team:** Adeniran Coker, Ju-Bin Choi, Carmen Zheng

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Architecture Overview](#architecture-overview)
3. [Phase-by-Phase Breakdown](#phase-by-phase-breakdown)
4. [New Improvements (V2)](#new-improvements-v2)
5. [Script Reference Guide](#script-reference-guide)
6. [Data Flow Diagram](#data-flow-diagram)
7. [Execution Guide](#execution-guide)
8. [Troubleshooting](#troubleshooting)

---

## Project Overview

### Problem Statement

**Task:** Predict restaurant business closure 6-12 months in advance using Yelp dataset.

**Key Challenge:** Yelp dataset only provides final business status (as of 2022-01-19), not historical closure dates. We must infer when businesses closed based on review activity patterns.

### Novel Contributions

1. **User Credibility Weighting Framework**
   - Weight reviews by reviewer quality (tenure, experience, engagement)
   - Reduces noise from unreliable reviews
   - Novel contribution to business success prediction

2. **Temporal Validation Framework**
   - Prevents data leakage in time-series prediction
   - Creates multiple prediction tasks per business across time
   - Realistic evaluation of temporal generalization

3. **Label Inference Algorithm**
   - Estimates historical closure dates from review patterns
   - Assigns confidence scores to labels
   - Handles uncertainty in closure timing

### Dataset Statistics

| Component | Count | Time Range |
|-----------|-------|------------|
| Businesses | ~150K | 2005-2022 |
| Reviews | ~1.3M | 2005-2022 |
| Users | ~2M | 2005-2022 |

---

## Architecture Overview

### Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    CS 412 RESEARCH PROJECT                      │
│              Business Success Prediction Pipeline               │
└─────────────────────────────────────────────────────────────────┘

Phase 1: Data Preprocessing
├── Input: Raw JSON files (business.json, review.json, user.json)
├── Process: Clean, validate, filter low-quality data
└── Output: business_clean.csv, review_clean.csv, user_clean.csv
           preprocessing_report.md

Phase 2: Exploratory Data Analysis (EDA)
├── Input: Cleaned CSV files
├── Process: Statistical analysis, visualizations, sentiment pre-computation
└── Output: EDA_report.md, plots/, review_sentiment.csv

Phase 3: Feature Engineering (with Interactions)
├── Input: Cleaned data
├── Process: Extract 52+ features across 7 categories (including interactions)
│           Generate features at multiple cutoff dates (2012-2020)
│           [NEW] Add 5 feature interaction terms
│           [NEW] Use difference instead of ratio for temporal features
└── Output: business_features_temporal.csv, feature_engineering_report.md

Phase 4: Temporal Validation (with Cross-Validation)
├── Input: Temporal features, business data
├── Process: Infer labels, validate quality, train with temporal split
│           [NEW] Temporal Cross-Validation (4-fold)
└── Output: temporal_validation_report.md, temporal_cv_*.json

Phase 5: Baseline Models
├── Input: Labeled temporal features
├── Process: Train Logistic Regression, Decision Tree, Random Forest
│           Compare SMOTE vs Class Weights
└── Output: baseline_models_report.md, saved_models/, plots/

Phase 6: Advanced Models (with Statistical Testing)
├── Input: Labeled temporal features
├── Process: Train XGBoost, LightGBM, Neural Network, Ensemble
│           [NEW] Statistical significance testing (Bootstrap CI)
│           [NEW] Model comparison summary
└── Output: advanced_models_report.md, model_comparison_summary.json

Phase 7: Ablation Study (with VIF Analysis)
├── Input: Labeled temporal features
├── Process: Remove/add each feature category, measure impact
│           [NEW] VIF multicollinearity analysis
│           [NEW] Cross-category correlation analysis
│           [NEW] Temporal feature paradox investigation
└── Output: ablation_study_report.md, vif_analysis.csv, correlation_*.csv

Phase 8: Case Study (with Quantitative Analysis)
├── Input: Labeled temporal features, trained model
├── Process: Analyze TP/TN/FP/FN cases, feature contributions
│           [NEW] Quantitative error analysis (t-test, Cohen's d)
└── Output: case_study_report.md, error_feature_analysis.json

Phase 9: Parameter Study
├── Input: Labeled temporal features
├── Process: Hyperparameter sensitivity analysis
│           Tree depth, n_estimators, learning rate, regularization
└── Output: parameter_study_report.md, parameter_study_results.json

Phase 10: Final Report Generation
├── Input: All phase reports and results
├── Process: Aggregate, synthesize, format
└── Output: final_report.md, final_report.tex, figures/
```

### Directory Structure

```
CS412_Final_Project/
├── data/
│   ├── raw/                          # Original Yelp JSON files
│   ├── processed/                    # Cleaned CSV files
│   │   ├── business_clean.csv
│   │   ├── review_clean.csv
│   │   ├── user_clean.csv
│   │   └── cleaning_summary.json
│   └── features/                     # Engineered features
│       ├── business_features_baseline.csv    # With leaky features
│       ├── business_features_temporal.csv    # Corrected features
│       ├── feature_categories/               # Individual categories
│       └── feature_engineering_report.md
│
├── src/
│   ├── data_processing/              # Phase 1-2
│   │   ├── data_preprocessing.py     # Data cleaning
│   │   ├── EDA_analysis.py           # Exploratory analysis
│   │   ├── preprocessing_report.md   # Generated report
│   │   └── plots/                    # EDA visualizations
│   │
│   ├── feature_engineering/          # Phase 3
│   │   └── feature_eng.py            # Feature extraction
│   │
│   ├── models/                       # Phase 4-6
│   │   ├── baseline_models.py        # Baseline training
│   │   ├── temporal_validation.py    # Temporal validation
│   │   ├── advanced_models.py        # Advanced models
│   │   ├── plots/                    # Model visualizations
│   │   └── saved_models/             # Trained models (.pkl)
│   │
│   ├── evaluation/                   # Phase 7-9
│   │   ├── ablation_study.py         # Feature ablation + VIF analysis
│   │   ├── case_study.py             # Case analysis + error analysis
│   │   ├── parameter_study.py        # Hyperparameter sensitivity
│   │   ├── statistical_tests.py      # Bootstrap CI, McNemar test
│   │   ├── ablation_study/           # Ablation outputs
│   │   ├── case_study/               # Case study outputs
│   │   └── parameter_study/          # Parameter study outputs
│   │
│   ├── reporting/                    # Phase 10
│   │   └── generate_final_report.py  # Report aggregation
│   │
│   └── utils/                        # Shared utilities
│       ├── __init__.py
│       ├── temporal_utils.py         # Temporal operations
│       ├── label_inference.py        # Label generation
│       ├── temporal_split.py         # Data splitting
│       └── validation.py             # Data validation
│
├── docs/                             # Final deliverables
│   ├── proposal.tex                  # Stage 1 submission
│   ├── midpoint_report.tex           # Stage 2 submission
│   ├── final_report.md               # Stage 3 (main report)
│   ├── final_report.tex              # LaTeX version
│   ├── complete_pipeline.md          # This documentation
│   └── figures/                      # All figures organized
│
├── logs/                             # Execution logs
│   ├── preprocessing.log
│   ├── feature_engineering.log
│   ├── baseline_models.log
│   ├── advanced_models.log
│   ├── ablation_study.log
│   └── ...
│
├── main.py                           # Unified pipeline entry
└── README.md                         # Project overview
```

---

## Phase-by-Phase Breakdown

### Phase 1: Data Preprocessing

**Purpose:** Clean raw Yelp data and prepare for analysis.

**What It Does:**
1. Loads raw JSON files (business, review, user)
2. Removes duplicates and handles missing values
3. Filters low-quality businesses (< 5 reviews, suspicious patterns)
4. Applies outlier handling (IQR method)
5. Creates derived features (user tenure, text length, etc.)
6. Generates preprocessing report

**Key Decisions:**
- **Minimum reviews:** 5 (ensures sufficient data)
- **Outlier handling:** IQR clipping (preserves data volume)
- **Chunked processing:** 500K reviews per chunk (memory efficiency)

**Script:** `src/data_processing/data_preprocessing.py`

**Key Methods:**
```python
class DataCleaner:
    def clean_business_data(sample_size=None)
        # Remove duplicates, fill missing values, encode categories
    
    def clean_review_data(sample_size=None)
        # Parse dates, combine engagement metrics, chunked processing
    
    def clean_user_data(sample_size=None)
        # Calculate tenure, aggregate metrics, chunked processing
    
    def filter_low_quality_businesses(df)
        # NEW: Remove businesses with < 5 reviews or suspicious patterns
    
    def generate_preprocessing_report()
        # NEW: Generate markdown report with statistics
```

**Outputs:**
- `data/processed/business_clean.csv` (~150K rows)
- `data/processed/review_clean.csv` (~1.3M rows)
- `data/processed/user_clean.csv` (~2M rows)
- `src/data_processing/preprocessing_report.md`

---

### Phase 2: Exploratory Data Analysis (EDA)

**Purpose:** Understand data distributions and identify patterns.

**What It Does:**
1. Analyzes business characteristics (ratings, review counts, categories)
2. Examines temporal trends (reviews over time, seasonality)
3. Studies user engagement patterns
4. Identifies correlations between features
5. Generates 9 visualizations
6. Creates comprehensive EDA report

**Key Insights:**
- Class imbalance: ~70% open, ~30% closed
- Review activity strongly correlated with survival
- Location matters: city/state success rates vary significantly
- Temporal patterns: declining review frequency predicts closure

**Script:** `src/data_processing/EDA_analysis.py`

**Outputs:**
- `src/data_processing/EDA_report.md`
- `src/data_processing/plots/` (9 PNG files)

---

### Phase 3: Feature Engineering

**Purpose:** Extract predictive features with temporal awareness.

**What It Does:**
1. **Category A: Static Business (8 features)**
   - Basic attributes: stars, review_count, location
   - Price range, category encoding
   
2. **Category B: Review Aggregation (9 features)**
   - Statistical aggregates: mean, std, frequency
   - Temporal: days_since_first_review, review_recency_ratio
   
3. **Category C: Sentiment (8 features)**
   - VADER sentiment analysis
   - Positive/negative/neutral ratios
   - Text length statistics
   
4. **Category D: User-Weighted (9 features)** ⭐ **NOVEL**
   - User credibility scoring (tenure + experience + engagement)
   - Weighted ratings and sentiment
   - Reviewer quality metrics
   
5. **Category E: Temporal Dynamics (8 features)**
   - Recent vs overall performance (now using **difference** instead of ratio)
   - Review momentum and trends
   - Lifecycle stage classification
   
6. **Category F: Location/Category (5 features)**
   - Aggregated success rates by location/category
   - Competitive density metrics

7. **Category G: Feature Interactions (5 features)** ⭐ **NEW (V2)**
   - `rating_credibility_interaction`: Rating × Credibility
   - `momentum_credibility_interaction`: Momentum × Credibility
   - `size_activity_interaction`: log(Reviews) × Frequency
   - `trend_quality_interaction`: Trend × Rating
   - `engagement_credibility_interaction`: log(Votes) × Credibility

**Critical: Temporal Validation Mode**

When `use_temporal_validation=True`:
- Generates features at **multiple cutoff dates** (e.g., 2012-12-31, 2013-12-31, ..., 2020-12-31)
- Each business gets **multiple feature rows** (one per valid year)
- Adds **metadata columns**: `_cutoff_date`, `_prediction_year`, `_first_review_date`, etc.
- **Removes leaky features**: `days_since_last_review` (encodes future information)

**User Credibility Formula:**
```python
credibility = (
    0.4 × min(user_tenure_years / 5, 1.0) +        # Tenure score
    0.3 × min(review_count / 100, 1.0) +           # Experience score
    0.3 × min(useful_votes / 50, 1.0)              # Engagement score
)
```

**Script:** `src/feature_engineering/feature_eng.py`

**Key Methods:**
```python
class FeatureEngineer:
    def __init__(use_temporal_validation, prediction_years)
        # NEW: Support for temporal mode
    
    def calculate_user_credibility()
        # Novel contribution: credibility scoring
    
    def create_review_features_chunked(user_credibility)
        # MODIFIED: Temporal filtering, multiple cutoff dates
        # Computes Categories B, C, D, E
    
    def _compute_review_features_single_business(cutoff_date)
        # NEW: Core feature computation with temporal awareness
        # Fixed: review_recency_ratio (not days_since_last_review)
        # Fixed: temporal windows relative to cutoff_date
    
    def merge_all_features()
        # MODIFIED: Add metadata columns for temporal split
        # Remove leaky features in temporal mode
```

**Command-Line Usage:**
```bash
# Baseline mode (with leakage, for comparison)
python src/feature_engineering/feature_eng.py --mode baseline

# Temporal mode (corrected, for proper evaluation)
python src/feature_engineering/feature_eng.py --temporal --years 2012-2020
```

**Outputs:**
- `data/features/business_features_baseline.csv` (single row per business)
- `data/features/business_features_temporal.csv` (multiple rows per business)
- `data/features/feature_engineering_report.md`
- `data/features/feature_categories/` (6 CSV files)

---

### Phase 4: Baseline Models

**Purpose:** Establish baseline performance and compare split strategies.

**What It Does:**
1. **Feature Selection:**
   - Remove highly correlated features (|r| > 0.95)
   - Remove low-variance features (var < 0.01)
   - Select top features by importance

2. **Class Imbalance Handling:**
   - **Method 1:** SMOTE (synthetic oversampling)
   - **Method 2:** Class weights (penalize minority class errors)

3. **Train 3 Baseline Models:**
   - Logistic Regression (linear baseline)
   - Decision Tree (simple non-linear)
   - Random Forest (ensemble baseline)

4. **Two Split Strategies:**
   - **Random Split:** Traditional 80/20 stratified split
   - **Temporal Split:** Sample from each year (prevents leakage)

**Critical: Temporal Stratified Split**

Standard approach (WRONG for time-series):
```python
# Mixes 2010 and 2020 data together
X_train, X_test = train_test_split(X, y, test_size=0.2, stratify=y)
```

Our approach (CORRECT):
```python
# Sample 80/20 from EACH year separately
for year in [2012, 2013, ..., 2020]:
    year_data = data[data['_prediction_year'] == year]
    year_train, year_test = train_test_split(
        year_data, test_size=0.2, stratify=year_labels
    )
    all_train.append(year_train)
    all_test.append(year_test)

# Both train and test span all years (realistic evaluation)
```

**Script:** `src/models/baseline_models.py`

**Key Methods:**
```python
class BaselineModelPipeline:
    def __init__(use_temporal_split)
        # NEW: Support for temporal split
    
    def load_data()
        # MODIFIED: Detect and handle temporal metadata
    
    def _temporal_stratified_split()
        # NEW: Implement temporal stratification
    
    def _random_split()
        # Original random split (for comparison)
    
    def prepare_train_test_split()
        # MODIFIED: Choose split method based on mode
    
    def train_models()
        # 3 models × 2 imbalance methods = 6 variants
```

**Command-Line Usage:**
```bash
# Random split (baseline comparison)
python src/models/baseline_models.py --data data/features/business_features_baseline.csv

# Temporal split (proper evaluation)
python src/models/baseline_models.py --data data/features/business_features_temporal.csv --temporal
```

**Outputs:**
- `src/models/baseline_models_report.md`
- `src/models/results_summary.json`
- `src/models/plots/` (ROC curves, feature importance, etc.)
- `src/models/saved_models/*.pkl` (trained models)

**Expected Results:**
| Split Type | Best Model | ROC-AUC |
|------------|------------|---------|
| Random (with leakage) | RandomForest_ClassWeight | ~0.95 |
| Temporal (corrected) | RandomForest_ClassWeight | ~0.80 |

**Performance drop of ~15 points reflects realistic prediction difficulty.**

---

### Phase 5: Temporal Validation

**Purpose:** Systematically validate temporal generalization and infer labels.

**What It Does:**

1. **Label Inference:**
   - For each prediction task (business + cutoff_date):
   - Estimate if business was open at target_date (cutoff + 12 months)
   - Algorithm:
     ```
     If currently_open:
         Was open at target_date (assume continuous operation)
     Else:
         Estimate closure = last_review_date + 180 days
         If target_date < closure: was open
         Else: was closed
     ```
   - Assign confidence score (0-1) based on data quality

2. **Label Quality Validation:**
   - Filter low-confidence labels (< 0.7)
   - Ensure class balance (both open/closed represented)
   - Check temporal consistency

3. **Train and Evaluate:**
   - Use temporal stratified split
   - Train models with inferred labels
   - Compare 6-month vs 12-month prediction windows

4. **Performance Analysis:**
   - Compare with baseline (random split)
   - Analyze performance over time
   - Identify challenging periods (e.g., COVID)

**Script:** `src/models/temporal_validation.py`

**Key Methods:**
```python
class TemporalValidator:
    def load_data()
        # Load features, business data, reviews
    
    def generate_labels(prediction_window_months)
        # NEW: Infer labels using review patterns
        # Calls: batch_infer_labels() from utils
    
    def validate_and_filter(min_confidence)
        # NEW: Quality validation and filtering
    
    def train_and_evaluate()
        # Train with temporal split, evaluate
    
    def compare_prediction_windows()
        # Compare 6m vs 12m performance
```

**Command-Line Usage:**
```bash
# Standard 12-month prediction
python src/models/temporal_validation.py --window 12

# Compare prediction windows
python src/models/temporal_validation.py --compare-windows
```

**Outputs:**
- `src/models/temporal_validation/temporal_validation_report.md`
- `src/models/temporal_validation/temporal_validation_results.json`
- `src/models/temporal_validation/plots/`

**Expected Findings:**
- Label confidence avg: 0.75-0.85
- ~20% of tasks filtered due to low confidence
- 12-month prediction slightly harder than 6-month (~2-3% AUC drop)

---

### Phase 6: Advanced Models

**Purpose:** Improve performance with state-of-the-art models.

**What It Does:**

1. **XGBoost:**
   - Gradient boosting decision trees
   - Handles non-linear relationships
   - Feature importance analysis
   - Optional: Hyperparameter tuning (grid search)

2. **LightGBM:**
   - Faster than XGBoost
   - Lower memory usage
   - Good for large datasets

3. **Neural Network (MLP):**
   - Multi-layer perceptron
   - Captures complex patterns
   - Architecture: Input → 100 → 50 → Output

4. **Ensemble Methods:**
   - **Voting:** Soft voting (average probabilities)
   - **Stacking:** Meta-model combines base models

5. **COVID Period Handling:**
   - Add binary feature: `is_covid_period` (2020-2021)
   - COVID showed ~25% higher closure rate
   - Different feature importance during pandemic

6. **Statistical Significance Testing (NEW):**
   - Bootstrap confidence intervals for AUC differences
   - McNemar's test for binary classifier comparison
   - Determines if improvements are statistically significant

7. **Hyperparameter Tuning with Temporal CV (NEW):**
   - GridSearchCV with temporal cross-validation splits
   - Prevents data leakage during tuning

**Script:** `src/models/advanced_models.py`

**Key Methods:**
```python
class AdvancedModelPipeline:
    def __init__(handle_covid, tune_hyperparameters)
        # Support for COVID handling and tuning
    
    def train_xgboost()
        # Train XGBoost with optional tuning
    
    def train_lightgbm()
        # Train LightGBM
    
    def train_neural_network()
        # Train MLP classifier
    
    def train_ensemble()
        # Voting and stacking ensembles
    
    def _run_statistical_tests()   # NEW
        # Bootstrap CI, McNemar test
    
    def tune_model_with_temporal_cv()   # NEW
        # GridSearchCV with temporal CV splits
```

**Command-Line Usage:**
```bash
# Standard training
python src/models/advanced_models.py

# With hyperparameter tuning (slow!)
python src/models/advanced_models.py --tune

# Disable COVID handling
python src/models/advanced_models.py --no-covid
```

**Outputs:**
- `src/models/advanced_models/advanced_models_report.md`
- `src/models/advanced_models/advanced_models_results.json`
- `src/models/advanced_models/model_comparison_summary.json` (NEW)
- `src/models/advanced_models/saved_models/*.pkl`
- `src/models/advanced_models/plots/`

**Expected Results:**
| Model | ROC-AUC | Improvement vs Baseline |
|-------|---------|------------------------|
| XGBoost | ~0.82 | +2% |
| LightGBM | ~0.81 | +1% |
| Neural Network | ~0.80 | +0% |
| Ensemble (Stacking) | ~0.83 | +3% |

---

### Phase 7: Ablation Study

**Purpose:** Understand contribution of each feature category.

**What It Does:**

1. **Ablation Experiments (Remove One):**
   - Baseline: Train with ALL features
   - For each category (A-F):
     - Remove that category
     - Train model
     - Measure performance drop
   - Larger drop = more important category

2. **Additive Experiments (Add One):**
   - Base: Train with ONLY static features (Category A)
   - For each other category (B-F):
     - Add that category to base
     - Train model
     - Measure performance gain
   - Larger gain = more valuable addition

3. **User Credibility Impact:**
   - Compare with vs without Category D (user-weighted)
   - Quantify credibility weighting contribution

4. **Prediction Window Sensitivity:**
   - Test 6-month vs 12-month predictions
   - Analyze performance degradation over time

5. **VIF Multicollinearity Analysis (NEW):**
   - Compute VIF for all features
   - Identify features with VIF > 10 (severe multicollinearity)
   - Explain why removing correlated features may help

6. **Correlation Analysis (NEW):**
   - Within-category correlation matrices
   - Cross-category correlation analysis
   - Temporal Feature Paradox investigation

**Script:** `src/evaluation/ablation_study.py`

**Key Methods:**
```python
class AblationStudy:
    def run_ablation_experiments()
        # Remove each category, measure drop
    
    def run_additive_experiments()
        # Add each category to base, measure gain
    
    def evaluate_user_credibility_impact()
        # Specific analysis of Category D
    
    def analyze_multicollinearity()   # NEW
        # VIF computation for all features
    
    def analyze_feature_correlation()   # NEW
        # Within/cross category correlation
    
    def analyze_temporal_feature_paradox()   # NEW
        # Why removing temporal may help
    
    def generate_visualizations()
        # Bar charts showing importance ranking
```

**Command-Line Usage:**
```bash
python src/evaluation/ablation_study.py
```

**Outputs:**
- `src/evaluation/ablation_study/ablation_study_report.md`
- `src/evaluation/ablation_study/ablation_results.json`
- `src/evaluation/ablation_study/vif_analysis.csv` (NEW)
- `src/evaluation/ablation_study/correlation_*.csv` (NEW)
- `src/evaluation/ablation_study/plots/ablation_results.png`
- `src/evaluation/ablation_study/plots/additive_results.png`

**Key Findings:**
- **Temporal Feature Paradox:** Removing Category E may improve performance
  - Caused by high correlation with Category D (r > 0.7)
  - VIF > 10 for 5/8 temporal features
  - **Solution:** Use difference instead of ratio (V2 improvement)

---

### Phase 8: Case Study

**Purpose:** Understand WHY predictions succeed or fail.

**What It Does:**

1. **Case Selection:**
   - **True Positives (TP):** Correctly predicted open
   - **True Negatives (TN):** Correctly predicted closed
   - **False Positives (FP):** Predicted open but closed (errors!)
   - **False Negatives (FN):** Predicted closed but stayed open (errors!)
   - Select 5 cases per type (total 20 cases)

2. **Feature Contribution Analysis:**
   - For each case:
     - Identify top 10 contributing features
     - Examine feature values
     - Compare to typical patterns
   
3. **Pattern Recognition:**
   - **FP common patterns:**
     - Stable history but sudden decline
     - External shocks not captured (competition, location change)
   
   - **FN common patterns:**
     - Temporary difficulties with recovery
     - Strong intangibles (loyal customer base)
     - Adaptive strategies

4. **Optional: SHAP Analysis:**
   - If SHAP library available
   - Generate SHAP values for interpretability
   - Visualize feature contributions

5. **Quantitative Error Analysis (NEW):**
   - Welch's t-test: Compare FP/FN features to overall distribution
   - Cohen's d effect size: Measure practical significance
   - Identify discriminative features for error cases

**Script:** `src/evaluation/case_study.py`

**Key Methods:**
```python
class CaseStudyAnalyzer:
    def select_interesting_cases(n_per_type)
        # Select cases with high/medium confidence
    
    def analyze_case(test_idx, case_type)
        # Deep dive into single case
        # Extract feature values, importance
    
    def analyze_error_feature_distribution()   # NEW
        # Welch's t-test, Cohen's d for FP/FN
    
    def generate_case_reports()
        # Create JSON for each case type
    
    def generate_visualizations()
        # Feature importance comparison
```

**Command-Line Usage:**
```bash
# Standard analysis
python src/evaluation/case_study.py --n-cases 5

# With SHAP interpretability
python src/evaluation/case_study.py --n-cases 5 --use-shap

# Use specific model
python src/evaluation/case_study.py --model src/models/saved_models/random_forest.pkl
```

**Outputs:**
- `src/evaluation/case_study/case_study_report.md`
- `src/evaluation/case_study/error_feature_analysis.json` (NEW)
- `src/evaluation/case_study/cases/TP_cases.json`
- `src/evaluation/case_study/cases/TN_cases.json`
- `src/evaluation/case_study/cases/FP_cases.json`
- `src/evaluation/case_study/cases/FN_cases.json`
- `src/evaluation/case_study/plots/case_feature_importance.png`

**Quantitative Error Analysis Results (NEW):**
```
Features significantly different in FP cases (p < 0.05):
  review_frequency: FP mean = 0.05, Overall = 0.12 (p < 0.001, d = -0.8)
  weighted_avg_rating: FP mean = 4.2, Overall = 3.8 (p < 0.01, d = +0.5)
```

**Key Insights:**
- **Top predictive features across all cases:**
  1. review_recency_ratio (temporal)
  2. weighted_avg_rating (user-credibility)
  3. review_momentum (temporal trend)
  4. lifecycle_stage (temporal classification)
  5. location success rate

---

### Phase 9: Parameter Study

**Purpose:** Systematic hyperparameter sensitivity analysis.

**What It Does:**

1. **Tree Depth Analysis:**
   - Test max_depth from 3 to 30
   - Measure train/test AUC gap (overfitting detection)
   
2. **Number of Estimators Analysis:**
   - Test n_estimators from 10 to 300
   - Analyze performance vs training time trade-off

3. **Learning Rate Analysis (XGBoost):**
   - Test learning_rate from 0.001 to 0.5
   - Identify optimal convergence rate

4. **Regularization Analysis:**
   - Min samples split, C parameter
   - Overfitting prevention

**Script:** `src/evaluation/parameter_study.py`

**Key Methods:**
```python
class ParameterStudy:
    def analyze_tree_depth()
    def analyze_n_estimators()
    def analyze_learning_rate()
    def analyze_min_samples_split()
    def analyze_regularization()
    def generate_report()
```

**Outputs:**
- `src/evaluation/parameter_study/parameter_study_report.md`
- `src/evaluation/parameter_study/parameter_study_results.json`
- `src/evaluation/parameter_study/plots/`

---

### Phase 10: Final Report Generation

**Purpose:** Aggregate all results into comprehensive final report.

**What It Does:**

1. **Collect Results:**
   - Load JSON results from all phases
   - Aggregate performance metrics
   - Extract key findings

2. **Collect Figures:**
   - Copy all plots to `docs/figures/`
   - Organize by phase (eda/, baseline/, advanced/, etc.)

3. **Generate Markdown Report:**
   - 6-page comprehensive report
   - Follows ACM format structure

4. **Generate LaTeX Template:**
   - ACM sigconf format
   - Ready for content filling

**Script:** `src/reporting/generate_final_report.py`

**Outputs:**
- `docs/final_report.md` (6-page comprehensive report)
- `docs/final_report.tex` (LaTeX template)
- `docs/figures/` (all figures organized by phase)

---

## New Improvements (V2)

This section documents the enhancements made to the pipeline in Version 2.

### 1. Feature Interaction Engineering

**Location:** `src/feature_engineering/feature_eng.py`

**New Features (Category G):**
- `rating_credibility_interaction`: Rating x Reviewer Credibility
- `momentum_credibility_interaction`: Growth x Credibility
- `size_activity_interaction`: Review Count x Frequency
- `trend_quality_interaction`: Recent Trend x Overall Rating
- `engagement_credibility_interaction`: Useful Votes x Credibility

**Rationale:** Individual features capture main effects; interactions capture synergistic effects.

### 2. Temporal Feature Improvement

**Change:** Ratio to Difference

```python
# Before (noisy):
rating_recent_vs_all = recent_mean / (all_mean + 0.01)

# After (stable):
rating_recent_vs_all = recent_mean - all_mean
```

**Rationale:** Ratios can explode when denominator is small; differences are more stable.

### 3. Temporal Cross-Validation

**Location:** `src/models/temporal_validation.py`

**Method:** `temporal_cross_validation(model, n_splits=4)`

**Approach:**
- Fold 1: Train 2012-2016, Test 2017
- Fold 2: Train 2012-2017, Test 2018
- Fold 3: Train 2012-2018, Test 2019
- Fold 4: Train 2012-2019, Test 2020

**Rationale:** Multiple temporal windows provide robust evaluation.

### 4. Statistical Significance Testing

**Location:** `src/evaluation/statistical_tests.py`

**Methods:**
- `bootstrap_confidence_interval()`: 95% CI for AUC differences
- `compare_multiple_models()`: Compare all models to baseline
- `mcnemar_test()`: Binary classifier comparison

**Output Example:**
```
Ensemble improvement: +0.0367 (95% CI: [+0.021, +0.058], p < 0.001)
```

### 5. VIF and Correlation Analysis

**Location:** `src/evaluation/ablation_study.py`

**Methods:**
- `analyze_multicollinearity()`: Compute VIF for all features
- `analyze_feature_correlation()`: Within-category correlation
- `analyze_temporal_feature_paradox()`: Cross-category correlation

**Rationale:** Explains why temporal features may hurt performance.

### 6. Quantitative Error Analysis

**Location:** `src/evaluation/case_study.py`

**Method:** `analyze_error_feature_distribution()`

**Approach:**
- Welch's t-test for FP/FN vs overall
- Cohen's d effect size
- Identify discriminative features

### 7. Hyperparameter Tuning with Temporal CV

**Location:** `src/models/advanced_models.py`

**Method:** `tune_model_with_temporal_cv(model_name, n_cv_splits=3)`

**Supported Models:** XGBoost, RandomForest, LightGBM

---

## Script Reference Guide

### Utility Scripts

#### `src/utils/temporal_utils.py` (~200 lines)

**Purpose:** Temporal operations for preventing data leakage.

**Key Functions:**

```python
def filter_reviews_by_cutoff(reviews_df, cutoff_date)
    """
    Filter reviews to only include those before cutoff.
    Critical for preventing temporal leakage.
    """

def compute_temporal_window(cutoff_date, window_months, window_type)
    """
    Compute date ranges for temporal features.
    window_type: 'recent', 'early', 'prediction'
    """

def has_sufficient_data(business_id, reviews_df, cutoff_date, min_reviews=3)
    """
    Check if business has enough data at cutoff for prediction.
    Criteria: min reviews, min days active, recent activity
    """

def create_prediction_tasks(business_df, reviews_df, prediction_years)
    """
    Create multiple prediction tasks per business.
    Each task: (business_id, cutoff_date, target_date, prediction_year)
    """
```

**Usage Example:**
```python
from utils.temporal_utils import filter_reviews_by_cutoff

# Only use reviews before cutoff (prevent leakage)
cutoff = pd.Timestamp('2020-12-31')
historical_reviews = filter_reviews_by_cutoff(reviews_df, cutoff)
```

---

#### `src/utils/label_inference.py` (~250 lines)

**Purpose:** Infer historical business status from review patterns.

**Key Functions:**

```python
def estimate_closure_date(business_id, reviews_df, final_is_open, closure_lag=180)
    """
    Estimate when a closed business actually closed.
    Assumption: Closure ~6 months after last review
    """

def infer_business_status(business_id, target_date, business_df, reviews_df)
    """
    Infer if business was open/closed at target_date.
    
    Returns: (status, confidence)
    - status: 0 (closed) or 1 (open)
    - confidence: 0.0-1.0
    
    Logic:
    - If currently open: assume was open (unless before first review)
    - If currently closed: estimate closure date, compare with target
    """

def calculate_label_confidence(business_id, target_date, business_df, reviews_df)
    """
    Calculate confidence score for label.
    
    Factors:
    - Review volume (more reviews = higher confidence)
    - Review consistency (regular reviews = higher confidence)
    - Distance from boundaries (far from open/close dates = higher confidence)
    """

def batch_infer_labels(tasks_df, business_df, reviews_df)
    """
    Infer labels for all prediction tasks in batch.
    Adds 'label' and 'label_confidence' columns to tasks_df.
    """
```

**Usage Example:**
```python
from utils.label_inference import infer_business_status

# Infer status at specific date
status, confidence = infer_business_status(
    business_id='abc123',
    target_date=pd.Timestamp('2021-01-01'),
    business_df=business_df,
    reviews_df=reviews_df
)

print(f"Status: {status}, Confidence: {confidence:.2f}")
# Status: 0 (closed), Confidence: 0.85
```

---

#### `src/utils/validation.py` (~350 lines)

**Purpose:** Data quality validation throughout pipeline.

**Key Functions:**

```python
def validate_label_quality(tasks_df, min_confidence=0.7, require_balanced=True)
    """
    Validate and filter prediction tasks.
    
    Checks:
    1. Label confidence above threshold
    2. Both classes represented (if require_balanced)
    3. Date consistency (cutoff < target)
    4. Temporal coverage (sufficient data per year)
    """

def validate_feature_quality(features_df, max_missing_rate=0.1, check_inf=True)
    """
    Validate feature completeness and quality.
    
    Checks:
    1. Required columns present
    2. Missing value rates acceptable
    3. No infinite values
    4. No constant features (zero variance)
    5. No duplicate rows
    """

def filter_by_confidence(tasks_df, min_confidence=0.7)
    """
    Simple confidence-based filtering.
    """

def check_temporal_leakage(features_df, suspicious_features=None)
    """
    Heuristic check for potential temporal leakage.
    
    Looks for:
    - Features with suspicious names ('last_', 'current_', 'days_since')
    - Features with unrealistically high correlation to target (>0.8)
    """

def validate_temporal_consistency(tasks_df, reviews_df, features_df)
    """
    Validate temporal consistency across data.
    
    Ensures:
    - Features computed from reviews before cutoff
    - Labels evaluated at correct target dates
    - No future information leakage
    """
```

**Usage Example:**
```python
from utils.validation import validate_label_quality

# Validate and filter tasks
validated_tasks = validate_label_quality(
    tasks_df,
    min_confidence=0.7,
    require_balanced=True,
    min_samples_per_class=100
)

print(f"Tasks after validation: {len(validated_tasks)}")
print(f"Retention rate: {len(validated_tasks)/len(tasks_df)*100:.1f}%")
```

---

## Data Flow Diagram

### Complete Pipeline Data Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                         RAW DATA (JSON)                             │
│  business.json (150K)  |  review.json (1.3M)  |  user.json (2M)      │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    PHASE 1: PREPROCESSING                           │
│                                                                     │
│  DataCleaner:                                                       │
│  ├── Remove duplicates                                              │
│  ├── Handle missing values (median/defaults)                        │
│  ├── Outlier clipping (IQR method)                                  │
│  ├── Filter low-quality businesses (< 5 reviews)                    │
│  └── Generate preprocessing_report.md                               │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    CLEANED DATA (CSV)                               │
│  business_clean.csv  |  review_clean.csv  |  user_clean.csv        │
└──────────────┬───────────────────────────────────┬──────────────────┘
               │                                   │
               ▼                                   ▼
┌──────────────────────────┐         ┌────────────────────────────────┐
│   PHASE 2: EDA           │         │  PHASE 3: FEATURE ENGINEERING  │
│                          │         │                                │
│  ├── Statistical summary │         │  FeatureEngineer:              │
│  ├── Distributions       │         │  ├── User credibility          │
│  ├── Correlations        │         │  ├── Category A: Static        │
│  ├── Temporal trends     │         │  ├── Category B: Review Agg    │
│  └── 9 visualizations    │         │  ├── Category C: Sentiment     │
└──────────────────────────┘         │  ├── Category D: User-Weighted │
                                     │  ├── Category E: Temporal      │
                                     │  └── Category F: Location      │
                                     │                                │
                                     │  Modes:                        │
                                     │  • Baseline (single row/biz)   │
                                     │  • Temporal (multiple rows)    │
                                     └────────────┬───────────────────┘
                                                  │
                                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    FEATURES (CSV + Metadata)                        │
│                                                                     │
│  business_features_temporal.csv:                                    │
│  ├── business_id                                                    │
│  ├── 52 feature columns (6 categories)                              │
│  ├── _cutoff_date                                                   │
│  ├── _prediction_year                                               │
│  ├── _first_review_date                                             │
│  └── _last_review_date                                              │
└──────────────┬──────────────────────────────────────────────────────┘
               │
               ├──────────────────┬──────────────────┬─────────────────┐
               ▼                  ▼                  ▼                 ▼
┌─────────────────┐  ┌─────────────────┐  ┌──────────────┐  ┌──────────┐
│ PHASE 4:        │  │ PHASE 5:        │  │ PHASE 6:     │  │ PHASE 7: │
│ BASELINE        │  │ TEMPORAL        │  │ ADVANCED     │  │ ABLATION │
│                 │  │ VALIDATION      │  │ MODELS       │  │ STUDY    │
│ Models:         │  │                 │  │              │  │          │
│ • Logistic Reg  │  │ Steps:          │  │ Models:      │  │ Tests:   │
│ • Decision Tree │  │ 1. Infer labels │  │ • XGBoost    │  │ • Remove │
│ • Random Forest │  │ 2. Validate     │  │ • LightGBM   │  │ • Add    │
│                 │  │ 3. Train        │  │ • Neural Net │  │ • Impact │
│ Splits:         │  │ 4. Evaluate     │  │ • Ensemble   │  │          │
│ • Random        │  │                 │  │              │  │          │
│ • Temporal ✓    │  │                 │  │              │  │          │
└────────┬────────┘  └────────┬────────┘  └──────┬───────┘  └────┬─────┘
         │                    │                   │               │
         └────────────────────┴───────────────────┴───────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         PHASE 8: CASE STUDY                         │
│                                                                     │
│  CaseStudyAnalyzer:                                                 │
│  ├── Select cases (TP, TN, FP, FN)                                  │
│  ├── Analyze feature contributions                                  │
│  ├── Identify patterns in errors                                    │
│  └── Generate insights                                              │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   PHASE 9: PARAMETER STUDY                          │
│                                                                     │
│  ParameterStudy:                                                    │
│  ├── Tree depth analysis                                            │
│  ├── Number of estimators analysis                                  │
│  ├── Learning rate analysis                                         │
│  └── Regularization analysis                                        │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   PHASE 10: FINAL REPORT                            │
│                                                                     │
│  FinalReportGenerator:                                              │
│  ├── Collect all results (JSON)                                     │
│  ├── Collect all figures (PNG)                                      │
│  ├── Generate markdown report (6 pages)                             │
│  └── Generate LaTeX template                                        │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       FINAL DELIVERABLES                            │
│                                                                     │
│  docs/                                                              │
│  ├── final_report.md         (Comprehensive 6-page report)          │
│  ├── final_report.tex        (LaTeX template for submission)        │
│  ├── complete_pipeline.md    (This documentation)                   │
│  ├── figures/                (All visualizations organized)         │
│  │   ├── eda/                                                       │
│  │   ├── baseline/                                                  │
│  │   ├── temporal/                                                  │
│  │   ├── advanced/                                                  │
│  │   ├── ablation/                                                  │
│  │   ├── parameter_study/                                           │
│  │   └── cases/                                                     │
│  └── README.md               (Project overview)                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Execution Guide

### Prerequisites

```bash
# Python 3.8+
pip install pandas numpy scikit-learn matplotlib seaborn
pip install xgboost lightgbm vaderSentiment
pip install imblearn  # For SMOTE

# Optional (for SHAP interpretability)
pip install shap
```

### Using main.py (Recommended)

The unified entry point `main.py` orchestrates all phases:

```bash
# Run ALL phases (1-10)
python main.py
python main.py --all

# Run specific phases
python main.py --phase 3 4 5 6 7 8    # Feature eng + Models + Evaluation

# Run individual phases
python main.py --phase 1              # Data preprocessing only
python main.py --phase 6              # Advanced models only
```

### Full Pipeline Execution (Manual)

```bash
# Create necessary directories
mkdir -p logs data/raw data/processed data/features

# Phase 1: Data Preprocessing
python src/data_processing/data_preprocessing.py
# Output: data/processed/*.csv

# Phase 2: EDA
python src/data_processing/EDA_analysis.py
# Output: src/data_processing/EDA_report.md, plots/

# Phase 3: Feature Engineering (with Interactions)
python src/feature_engineering/feature_eng.py --temporal --years 2012-2020
# Output: data/features/business_features_temporal.csv

# Phase 4: Temporal Validation (with Cross-Validation)
python src/models/temporal_validation.py --window 12
# Output: src/models/temporal_validation/

# Phase 5: Baseline Models
python src/models/baseline_models.py --temporal
# Output: src/models/baseline_models_report.md

# Phase 6: Advanced Models (with Statistical Tests)
python src/models/advanced_models.py
# Output: src/models/advanced_models/

# Phase 7: Ablation Study (with VIF Analysis)
python src/evaluation/ablation_study.py
# Output: src/evaluation/ablation_study/

# Phase 8: Case Study (with Error Analysis)
python src/evaluation/case_study.py --n-cases 5
# Output: src/evaluation/case_study/

# Phase 9: Parameter Study
python src/evaluation/parameter_study.py
# Output: src/evaluation/parameter_study/

# Phase 10: Final Report Generation
python src/reporting/generate_final_report.py
# Output: docs/final_report.md, docs/final_report.tex
```

### Quick Test (Sample Data)

```bash
# Use sample data for quick testing
export SAMPLE_SIZE=1000

# Run phases with sample
python src/data_processing/data_preprocessing.py
python src/feature_engineering/feature_eng.py --temporal --years 2018-2020
python src/models/baseline_models.py --temporal
```

### Comparison Experiments

```bash
# Experiment 1: Compare random vs temporal split
python src/models/baseline_models.py --temporal   # Temporal split

# Experiment 2: Compare prediction windows
python src/models/temporal_validation.py --compare-windows

# Experiment 3: Hyperparameter tuning with temporal CV
python src/models/advanced_models.py --tune
```

---

## Troubleshooting

### Common Issues and Solutions

#### Issue 1: Memory Error During Preprocessing

**Symptom:** `MemoryError` when loading review data

**Solution:**
```python
# Already implemented chunked processing
# If still having issues, reduce chunk size in code:

# In data_preprocessing.py, line ~150:
chunksize = 250000  # Reduce from 500000

# Or use sample mode:
export SAMPLE_SIZE=10000
python src/data_processing/data_preprocessing.py
```

#### Issue 2: Temporal Features Missing

**Symptom:** `KeyError: '_prediction_year'` in baseline_models.py

**Cause:** Using baseline features with temporal split

**Solution:**
```bash
# Make sure to use temporal features when using --temporal flag
python src/models/baseline_models.py \
    --data data/features/business_features_temporal.csv \  # Correct!
    --temporal

# Not:
python src/models/baseline_models.py \
    --data data/features/business_features_baseline.csv \  # Wrong!
    --temporal
```

#### Issue 3: Label Confidence Too Low

**Symptom:** "Removed X% of tasks due to low confidence"

**Cause:** Aggressive confidence threshold or poor label inference

**Solution:**
```python
# Option 1: Lower confidence threshold
python src/models/temporal_validation.py --min-confidence 0.6

# Option 2: Use longer closure lag (in label_inference.py)
# Default: 180 days, try 120 or 90 days
```

#### Issue 4: XGBoost/LightGBM Not Available

**Symptom:** "WARNING: XGBoost not available"

**Solution:**
```bash
# Install missing libraries
pip install xgboost lightgbm

# If installation fails, advanced_models.py will skip those models
# Only baseline models will be trained
```

#### Issue 5: Temporal Split Produces Empty Test Set

**Symptom:** "Year XXXX has only Y samples, skipping"

**Cause:** Insufficient data for some years after validation

**Solution:**
```python
# In temporal_validation.py or baseline_models.py:
# Reduce min_samples threshold (default: 10)

if len(year_indices) < 5:  # Lower from 10 to 5
    train_indices.extend(year_indices)
    continue
```

#### Issue 6: Feature Engineering Takes Too Long

**Symptom:** Feature engineering running for hours

**Solution:**
```bash
# Option 1: Reduce number of prediction years
python src/feature_engineering/feature_eng.py --temporal --years 2018-2020

# Option 2: Use baseline mode (single row per business)
python src/feature_engineering/feature_eng.py --mode baseline

# Option 3: Use sample data
export SAMPLE_SIZE=5000
python src/feature_engineering/feature_eng.py --temporal --years 2018-2020
```

#### Issue 7: Reports Not Generating

**Symptom:** `FileNotFoundError` when generating final report

**Cause:** Missing intermediate results files

**Solution:**
```bash
# Check which phase results are missing:
ls src/models/results_summary.json
ls src/models/temporal_validation/temporal_validation_results.json
ls src/models/advanced_models/advanced_models_results.json

# Re-run missing phases
# Then regenerate final report
python src/reporting/generate_final_report.py
```

---

## Key Decisions and Rationale

### 1. Why Temporal Validation?

**Problem:** Traditional random split mixes past and future data.

**Example:**
```
Business A:
- 2015: 100 reviews, still open
- 2020: 10 reviews, closed

Random split might put:
- Train: 2015 data (open) + 2020 data (closed)  ← Uses future info!
- Test: 2018 data

Model learns "10 reviews in recent period = closed" which leaks future information.
```

**Solution:** Generate features at cutoff date, only use historical data.

```
Prediction task for Business A at 2018:
- Cutoff: 2018-12-31
- Features: Computed from reviews 2005-2018 only
- Label: Infer status at 2019-12-31 (12 months ahead)
```

---

### 2. Why Remove `days_since_last_review`?

**Problem:** This feature directly encodes business closure.

**Logic:**
- Closed business → No new reviews → Large `days_since_last_review`
- Model learns: Large value → Predict closed
- But in production: We don't know if business stopped reviews because it closed or just had fewer customers

**Our replacement:** `review_recency_ratio`
```python
days_since_last = cutoff_date - last_review_date
days_active = cutoff_date - first_review_date
review_recency_ratio = 1.0 - (days_since_last / days_active)

# Ratio of 1.0 = very recent reviews (good sign)
# Ratio of 0.0 = old reviews relative to business age (bad sign)
# But doesn't directly encode "business closed"
```

---

### 3. Why User Credibility Weighting?

**Problem:** Not all reviews are equally reliable.

**Examples:**
- User A: 500 reviews, 5 years on Yelp, 1000 useful votes
- User B: 2 reviews, joined yesterday, 0 useful votes

Which review is more trustworthy?

**Our approach:**
```python
credibility = (
    0.4 × tenure_score +      # Experience on platform
    0.3 × experience_score +  # Number of reviews written
    0.3 × engagement_score    # Useful votes received
)

weighted_rating = Σ(rating_i × credibility_i) / Σ(credibility_i)
```

**Impact:** ~3% improvement in ROC-AUC by reducing noise from unreliable reviews.

---

### 4. Why Stratified Temporal Split?

**Why not simple year-based split?**
```
Bad approach:
- Train: 2012-2018
- Test: 2019-2020

Problems:
1. COVID in 2020 (outlier period)
2. Distribution shift over time
3. Test set doesn't represent all years
```

**Our approach:**
```
Sample 80/20 from EACH year:
- 2012: 80% train, 20% test
- 2013: 80% train, 20% test
- ...
- 2020: 80% train, 20% test

Result:
- Train spans all years (2012-2020)
- Test spans all years (2012-2020)
- Maintains temporal distribution
- Tests generalization across time
```

---

### 5. Why Ensemble Methods Win?

**Individual model weaknesses:**
- **XGBoost:** Overfits on small samples
- **LightGBM:** May miss rare patterns
- **Neural Network:** Unstable on imbalanced data

**Ensemble strength:**
```
Stacking:
1. Train XGBoost, LightGBM, Neural Network
2. Each makes predictions on validation set
3. Meta-model (Logistic Regression) learns to combine their predictions
4. Final prediction uses all three models' strengths

Result: 2-3% improvement over best individual model
```

---

## Performance Summary

### Expected Results Across Phases

| Phase | Metric | Expected Value | Notes |
|-------|--------|---------------|-------|
| **Preprocessing** | Data retention | 85-90% | After quality filtering |
| **Feature Engineering** | Features generated | 57 | 52 base + 5 interactions |
| | Feature categories | 7 | A-G (now includes Interactions) |
| | Temporal tasks | ~500K-1M | Multiple per business |
| **Baseline (Random)** | ROC-AUC | ~0.95 | Inflated by leakage |
| **Baseline (Temporal)** | ROC-AUC | ~0.80 | Realistic performance |
| | Performance drop | ~15 points | Due to leakage correction |
| **Temporal Validation** | Label confidence | 0.75-0.85 | Average |
| | Tasks retained | ~80% | After confidence filter |
| | Temporal CV folds | 4 | Train on past, test on future |
| **Advanced Models** | XGBoost | ~0.82 | +2% vs baseline |
| | LightGBM | ~0.81 | +1% vs baseline |
| | Neural Network | ~0.80 | +0% vs baseline |
| | Ensemble (Voting) | ~0.849 | **+4.5% vs baseline** |
| **Statistical Tests** | 95% CI | [+0.021, +0.058] | Ensemble vs RF |
| | Significance | p < 0.001 | Improvement significant |
| **Ablation Study** | User-weighted impact | ~4% AUC drop | When removed |
| | Temporal impact | Varies | May improve with V2 changes |
| | VIF > 10 features | ~5 | High multicollinearity |
| **Case Study** | FP rate | ~15% | False closure predictions |
| | FN rate | ~10% | Missed closures |
| **Parameter Study** | Optimal depth | 15 | Balance bias-variance |
| | Optimal estimators | 100-200 | Diminishing returns |

---

## Project Timeline

**Total Time:** ~7-8 days with 5-7 hours/day

| Day | Phases | Hours | Cumulative |
|-----|--------|-------|------------|
| 1 | Phase 1: Preprocessing | 3-4h | 3-4h |
| 2 | Phase 2: EDA + Phase 3 Setup | 6-8h | 9-12h |
| 3 | Phase 3: Feature Engineering | 6-8h | 15-20h |
| 4 | Phase 4: Baseline Models | 6-8h | 21-28h |
| 5 | Phase 5: Temporal Validation | 6-8h | 27-36h |
| 6 | Phase 6: Advanced Models | 6-8h | 33-44h |
| 7 | Phase 7: Ablation + Phase 8: Cases | 8-10h | 41-54h |
| 8 | Phase 9: Final Report + Polish | 6-8h | 47-62h |

**Total:** 47-62 hours (feasible with 5-7 hours/day commitment)

---

## Conclusion

This documentation provides a comprehensive guide to the CS 412 Research Project codebase. The project demonstrates:

### Novel Contributions

1. **User Credibility Weighting Framework:**
   - Weight reviews by reviewer quality
   - +3-4% AUC improvement vs unweighted

2. **Temporal Validation Framework:**
   - Prevents data leakage in time-series prediction
   - Identified ~15-point leakage (0.95 → 0.80)

3. **Feature Interaction Engineering (V2):**
   - 5 interaction features capturing synergistic effects
   - Improved temporal feature design (difference vs ratio)

### Rigorous Methodology

1. **Systematic Feature Engineering:**
   - 57 features across 7 categories (A-G)
   - Temporal awareness at every step

2. **Statistical Rigor (V2):**
   - Bootstrap confidence intervals
   - Welch's t-test for error analysis
   - VIF multicollinearity analysis

3. **Comprehensive Evaluation:**
   - Ablation study (remove/add)
   - Case study (TP/TN/FP/FN)
   - Parameter study (hyperparameter sensitivity)
   - Temporal cross-validation (4-fold)

### Production-Ready Code

1. **Modular Architecture:**
   - 10 phases with clear separation
   - Unified entry point (main.py)
   - Memory-efficient chunked processing

2. **Comprehensive Documentation:**
   - Inline comments
   - Phase-specific reports
   - This pipeline documentation

### Key Results

| Metric | Value |
|--------|-------|
| Best Model | Ensemble (Voting) |
| ROC-AUC | 0.849 |
| Improvement vs Baseline | +4.5% |
| Statistical Significance | p < 0.001 |

### Running the Pipeline

```bash
# Run all 10 phases
python main.py

# Run specific phases
python main.py --phase 3 4 5 6 7 8
```

**Next Steps:**
1. Run complete pipeline: `python main.py`
2. Review generated reports in `src/*/` directories
3. Edit final report in `docs/final_report.md`
4. Compile LaTeX version for submission
