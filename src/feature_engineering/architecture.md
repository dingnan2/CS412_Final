# Feature Engineering Pipeline Architecture

---

## Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                    FEATURE ENGINEERING PIPELINE                      │
│                                                                      │
│  Input: Cleaned Data (business, review, user)                       │
│  Output: 72 Features + Target Variable                              │
│  Method: Chunked Processing with User Credibility Weighting         │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Data Flow Diagram

```
┌──────────────────┐
│  INPUT DATA      │
│  (data/processed)│
└────────┬─────────┘
         │
    ┌────┴────────────────────────────────────┐
    │                                          │
    ▼                                          ▼
┌───────────────┐                    ┌─────────────────┐
│ business_clean│                    │ user_clean.csv  │
│    .csv       │                    │                 │
│ (150K rows)   │                    │ (2M users)      │
└───────┬───────┘                    └────────┬────────┘
        │                                     │
        │                            ┌────────▼────────┐
        │                            │ USER CREDIBILITY│
        │                            │   CALCULATION   │
        │                            │                 │
        │                            │ Formula:        │
        │                            │ 0.5×useful_rate │
        │                            │ +0.3×tenure     │
        │                            │ +0.2×experience │
        │                            └────────┬────────┘
        │                                     │
        │                                     │
        ▼                                     │
┌──────────────────┐                         │
│ STATIC FEATURES  │                         │
│   (Category A)   │                         │
│                  │                         │
│ • stars          │                         │
│ • review_count   │                         │
│ • category_enc   │                         │
│ • state_enc      │                         │
│ • city_enc       │                         │
│ • price_range    │                         │
│ • category_count │                         │
│                  │                         │
│ 8 features       │                         │
└──────────────────┘                         │
         │                                   │
         │                                   │
    ┌────┴────────────────────┐              │
    │                         │              │
    │                         │              │
    ▼                         │              │
┌──────────────┐              │              │
│review_clean  │              │              │
│   .csv       │◄─────────────┼──────────────┘
│ (7M reviews) │              │
└──────┬───────┘              │
       │                      │
       │ CHUNKED PROCESSING   │
       │ (100K rows/chunk)    │
       │                      │
       ▼                      │
┌──────────────────────────┐  │
│  REVIEW AGGREGATION      │  │
│    (Category B)          │  │
│                          │  │
│ Volume:                  │  │
│ • total_reviews          │  │
│ • review_frequency       │  │
│ • review_momentum        │  │
│                          │  │
│ Rating:                  │  │
│ • avg_review_stars       │  │
│ • std_review_stars       │  │
│ • rating_velocity        │  │
│ • rating_trend_3m        │  │
│                          │  │
│ Temporal:                │  │
│ • days_since_first       │  │
│ • days_since_last        │  │
│                          │  │
│ Engagement:              │  │
│ • total_useful_votes     │  │
│ • avg_useful_per_review  │  │
│ • total_funny_cool       │  │
│                          │  │
│ Text:                    │  │
│ • avg_text_length        │  │
│ • std_text_length        │  │
│                          │  │
│ 15 features              │  │
└──────────────────────────┘  │
       │                      │
       ▼                      │
┌──────────────────────────┐  │
│  SENTIMENT FEATURES      │  │
│    (Category C)          │  │
│                          │  │
│ VADER Analysis:          │  │
│ • avg_sentiment          │  │
│ • std_sentiment          │  │
│ • sentiment_slope        │  │
│                          │  │
│ Distribution:            │  │
│ • pct_positive_reviews   │  │
│ • pct_negative_reviews   │  │
│ • pct_neutral_reviews    │  │
│                          │  │
│ Recent:                  │  │
│ • sentiment_recent_3m    │  │
│ • sentiment_volatility   │  │
│                          │  │
│ 8 features               │  │
└──────────────────────────┘  │
       │                      │
       ▼                      │
┌──────────────────────────┐  │
│  USER-WEIGHTED FEATURES  │  │
│    (Category D)          │  │
│                          │  │
│ Weighted Aggregations:   │  │
│ • weighted_avg_rating    │  │
│ • weighted_sentiment     │  │
│                          │  │
│ Reviewer Quality:        │  │
│ • avg_reviewer_cred      │  │
│ • std_reviewer_cred      │  │
│ • pct_high_cred_rev      │  │
│ • weighted_useful_votes  │  │
│                          │  │
│ Diversity:               │  │
│ • review_diversity       │  │
│ • power_user_ratio       │  │
│ • avg_reviewer_tenure    │  │
│ • avg_reviewer_exp       │  │
│                          │  │
│ 10 features              │  │
└──────────────────────────┘  │
       │                      │
       ▼                      │
┌──────────────────────────┐  │
│  TEMPORAL DYNAMICS       │  │
│    (Category E)          │  │
│                          │  │
│ Recent vs Historical:    │  │
│ • rating_recent_vs_all   │  │
│ • rating_recent_vs_early │  │
│ • sentiment_recent_vs_all│  │
│                          │  │
│ Activity:                │  │
│ • reviews_recent_3m_cnt  │  │
│ • review_freq_trend      │  │
│ • engagement_recent_vs   │  │
│                          │  │
│ Lifecycle:               │  │
│ • lifecycle_stage        │  │
│   (0=New, 1=Growing,     │  │
│    2=Mature, 3=Declining)│  │
│                          │  │
│ 8 features               │  │
└──────────────────────────┘  │
       │                      │
       │                      │
       └──────────────────────┤
                              │
       ┌──────────────────────┘
       │
       ▼
┌──────────────────────────┐
│ LOCATION/CATEGORY        │
│ AGGREGATES               │
│    (Category F)          │
│                          │
│ Success Rates:           │
│ • category_avg_success   │
│ • state_avg_success      │
│ • city_avg_success       │
│                          │
│ Competition:             │
│ • category_competitive   │
│ • location_density       │
│                          │
│ 5 features               │
└──────────┬───────────────┘
           │
           ▼
    ┌──────────────┐
    │   MERGE ALL  │
    │   FEATURES   │
    └──────┬───────┘
           │
           ▼
    ┌──────────────────────────┐
    │  FINAL FEATURE DATASET   │
    │                          │
    │  150,346 businesses      │
    │  72 features + target    │
    │                          │
    │  business_features_final │
    │        .csv              │
    └──────────┬───────────────┘
               │
               │
       ┌───────┴───────┐
       │               │
       ▼               ▼
┌──────────────┐  ┌──────────────┐
│  MAIN OUTPUT │  │  CATEGORY    │
│              │  │  FILES       │
│ Single merged│  │              │
│ CSV for      │  │ 6 separate   │
│ modeling     │  │ files for    │
│              │  │ ablation     │
└──────────────┘  └──────────────┘
```

---

## Feature Categories Summary

```
┌────────────────────────────────────────────────────────────────┐
│                    FEATURE BREAKDOWN (72 Total)                 │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Category A: Static Business          [8 features]    11.1%    │
│  ├─ Basic metrics                     [2]                      │
│  ├─ Encoded categories                [3]                      │
│  └─ Metadata                          [3]                      │
│                                                                 │
│  Category B: Review Aggregation       [15 features]   20.8%    │
│  ├─ Volume metrics                    [3]                      │
│  ├─ Rating metrics                    [4]                      │
│  ├─ Temporal metrics                  [2]                      │
│  ├─ Engagement metrics                [3]                      │
│  └─ Text metrics                      [2]                      │
│                                                                 │
│  Category C: Sentiment                [8 features]    11.1%    │
│  ├─ Aggregate sentiment               [3]                      │
│  ├─ Distribution                      [3]                      │
│  └─ Recent trends                     [2]                      │
│                                                                 │
│  Category D: User-Weighted            [10 features]   13.9%    │
│  ├─ Weighted aggregations             [2]                      │
│  ├─ Reviewer quality                  [4]                      │
│  └─ Diversity metrics                 [4]                      │
│                                                                 │
│  Category E: Temporal Dynamics        [8 features]    11.1%    │
│  ├─ Comparisons (recent vs hist)      [3]                      │
│  ├─ Activity trends                   [3]                      │
│  └─ Lifecycle stage                   [1]                      │
│                                                                 │
│  Category F: Location/Category        [5 features]     6.9%    │
│  ├─ Success rates                     [3]                      │
│  └─ Competition metrics               [2]                      │
│                                                                 │
│  Target Variable: is_open             [1]             1.4%    │
│                                                                 │
└────────────────────────────────────────────────────────────────┘
```

---

## Processing Flow by Stage

```
STAGE 1: INITIALIZATION
═══════════════════════
┌─────────────────────┐
│ Load business data  │ ──► 150,346 rows
│ Load user data      │ ──► 1,987,897 rows
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│ Calculate user      │
│ credibility scores  │ ──► 1,987,897 credibility values
└─────────────────────┘


STAGE 2: STATIC FEATURES
═════════════════════════
┌─────────────────────┐
│ Process business    │
│ metadata            │
│                     │
│ • Parse categories  │
│ • Target encoding   │
│ • Extract price     │
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│ Output: 8 features  │ ──► business_static_features.csv
└─────────────────────┘


STAGE 3: REVIEW FEATURES (CHUNKED)
═══════════════════════════════════
┌─────────────────────────────────┐
│ FOR EACH CHUNK (100K reviews):  │
│                                 │
│  1. Load chunk                  │
│  2. Calculate sentiment (VADER) │
│  3. Map user credibility        │
│  4. Add temporal flags          │
│  5. Group by business           │
│  6. Accumulate statistics       │
│                                 │
│ Repeat ~70 times for 7M reviews │
└─────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│ COMPUTE AGGREGATIONS:           │
│                                 │
│  Per business (150K):           │
│   • Review aggregations  [15]   │
│   • Sentiment features   [8]    │
│   • User-weighted        [10]   │
│   • Temporal dynamics    [8]    │
│                                 │
│  With user credibility weights  │
│  normalized per business        │
└─────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│ Output: 41 features             │
│                                 │
│ ──► review_aggregation_features │
│ ──► sentiment_features          │
│ ──► user_weighted_features      │
│ ──► temporal_features           │
└─────────────────────────────────┘


STAGE 4: LOCATION FEATURES
═══════════════════════════
┌─────────────────────┐
│ Calculate success   │
│ rates by category,  │
│ state, city         │
│                     │
│ Calculate           │
│ competition &       │
│ density metrics     │
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│ Output: 5 features  │ ──► location_category_features.csv
└─────────────────────┘


STAGE 5: MERGE & FINALIZE
══════════════════════════
┌─────────────────────────────────┐
│ Merge all feature categories:   │
│                                 │
│  Static        [8]              │
│  Review Agg    [15]             │
│  Sentiment     [8]              │
│  User-Weighted [10]             │
│  Temporal      [8]              │
│  Location      [5]              │
│                                 │
│  Total: 72 features             │
│  + business_id, is_open         │
└─────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│ Quality checks:                 │
│ • No missing values             │
│ • No infinite values            │
│ • Reasonable feature ranges     │
│ • Target distribution ~80/20    │
└─────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│ Save outputs:                   │
│                                 │
│ ✓ business_features_final.csv   │
│ ✓ 6 category CSV files          │
│ ✓ feature_engineering_report.md │
│ ✓ feature_engineering.log       │
└─────────────────────────────────┘
```

---

## Memory Management Strategy

```
┌─────────────────────────────────────────────────────────────┐
│                   MEMORY OPTIMIZATION                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  PROBLEM: 7M reviews × 15 columns = ~5GB in memory          │
│                                                              │
│  SOLUTION: Chunked Processing with Accumulation             │
│                                                              │
│  ┌────────────────────────────────────────────────┐         │
│  │ Chunk 1 (100K reviews)                         │         │
│  │ ─────────────────────                          │         │
│  │ Process → Aggregate → Accumulate → Free        │ ~100MB  │
│  └────────────────────────────────────────────────┘         │
│                      │                                       │
│                      ▼                                       │
│  ┌────────────────────────────────────────────────┐         │
│  │ Chunk 2 (100K reviews)                         │         │
│  │ ─────────────────────                          │         │
│  │ Process → Aggregate → Accumulate → Free        │ ~100MB  │
│  └────────────────────────────────────────────────┘         │
│                      │                                       │
│                     ...                                      │
│                      │                                       │
│                      ▼                                       │
│  ┌────────────────────────────────────────────────┐         │
│  │ Accumulated Statistics (150K businesses)       │         │
│  │ ───────────────────────────────────────────    │         │
│  │ Store ONLY aggregations, not raw data          │ ~100MB  │
│  │                                                 │         │
│  │ business_stats = {                             │         │
│  │   'business_123': {                            │         │
│  │     'stars': [4, 5, 3, ...],                   │         │
│  │     'sentiment': [0.8, 0.6, ...],              │         │
│  │     'weights': [0.3, 0.4, ...]                 │         │
│  │   }                                            │         │
│  │ }                                              │         │
│  └────────────────────────────────────────────────┘         │
│                      │                                       │
│                      ▼                                       │
│  ┌────────────────────────────────────────────────┐         │
│  │ Final Computation (after all chunks)           │         │
│  │ ───────────────────────────────────────        │         │
│  │ Calculate means, stds, trends from             │         │
│  │ accumulated statistics                         │ ~100MB  │
│  └────────────────────────────────────────────────┘         │
│                                                              │
│  RESULT: Peak memory ~500MB instead of ~5GB                 │
│          10x memory reduction!                              │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## User Credibility Weighting (Novel Contribution)

```
┌──────────────────────────────────────────────────────────────┐
│              USER CREDIBILITY CALCULATION                     │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  INPUT: User profile data                                    │
│  ────────────────────────                                    │
│   • useful votes received                                    │
│   • review count                                             │
│   • tenure on platform (days)                                │
│                                                               │
│  FORMULA:                                                    │
│  ─────────                                                   │
│                                                               │
│   useful_rate = useful / (review_count + 1)                  │
│                                                               │
│   tenure_weight = log(1 + user_tenure_days) / 10            │
│                                                               │
│   experience_weight = log(1 + review_count) / 10            │
│                                                               │
│   credibility = 0.5 × useful_rate                            │
│               + 0.3 × tenure_weight                          │
│               + 0.2 × experience_weight                      │
│                                                               │
│  NORMALIZATION (per business):                              │
│  ──────────────────────────                                 │
│                                                               │
│   For business B with reviewers [r1, r2, ..., rn]:          │
│                                                               │
│   weights[i] = credibility[ri] / Σ credibility[rj]          │
│                                                               │
│   Σ weights = 1.0  (probability distribution)                │
│                                                               │
│  WEIGHTED AGGREGATION:                                       │
│  ───────────────────                                         │
│                                                               │
│   weighted_rating = Σ (rating[i] × weights[i])              │
│                                                               │
│   weighted_sentiment = Σ (sentiment[i] × weights[i])        │
│                                                               │
│  BENEFIT:                                                    │
│  ────────                                                    │
│   • Power users (high credibility) have more influence       │
│   • New/unreliable users have less influence                 │
│   • Normalized weights ensure comparability across          │
│     businesses with different reviewer profiles              │
│                                                               │
└──────────────────────────────────────────────────────────────┘
```

---

## Temporal Windows (3-Month Analysis)

```
┌──────────────────────────────────────────────────────────────┐
│                  TEMPORAL DYNAMICS                            │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  TIMELINE (looking back from 2022-01-19):                    │
│  ──────────────────────────────────────                     │
│                                                               │
│  ├───────────────────────────────┬───────────┤               │
│  │                               │           │               │
│  │     HISTORICAL                │  RECENT   │               │
│  │     (all-time)                │  (3 mo.)  │               │
│  │                               │           │               │
│  2005-02-16              2021-10-19    2022-01-19           │
│  (first review)          (3m cutoff)  (reference)           │
│                                                               │
│  COMPUTED FEATURES:                                          │
│  ──────────────────                                          │
│                                                               │
│  1. Recent vs All-Time                                       │
│     rating_recent_vs_all = mean(recent) - mean(all)         │
│                                                               │
│  2. Recent vs Early (first 3 months)                         │
│     rating_recent_vs_early = mean(recent) - mean(early)     │
│                                                               │
│  3. Activity Momentum                                        │
│     review_frequency_trend = freq(recent) / freq(hist)      │
│                                                               │
│  4. Engagement Shift                                         │
│     engagement_recent_vs_all = useful(recent) / useful(all) │
│                                                               │
│  5. Lifecycle Classification                                 │
│     ┌──────────────────────────────────────┐                │
│     │ IF timespan < 180 days:              │                │
│     │    lifecycle = 0 (New)               │                │
│     │ ELIF recent_count > 30% total:       │                │
│     │    lifecycle = 1 (Growing)           │                │
│     │ ELIF recent_count > 0:               │                │
│     │    lifecycle = 2 (Mature)            │                │
│     │ ELSE:                                 │                │
│     │    lifecycle = 3 (Declining)         │                │
│     └──────────────────────────────────────┘                │
│                                                               │
│  INSIGHT:                                                    │
│  ────────                                                    │
│   Captures trajectory and momentum - critical for           │
│   predicting business survival/closure                       │
│                                                               │
└──────────────────────────────────────────────────────────────┘
```

---

## Output File Structure

```
data/features/
│
├── business_features_final.csv          [Main Output - 80MB]
│   │
│   ├─ business_id                       [Unique identifier]
│   ├─ is_open                           [Target: 0=closed, 1=open]
│   │
│   ├─ [Category A: Static - 8 features]
│   ├─ [Category B: Review Agg - 15 features]
│   ├─ [Category C: Sentiment - 8 features]
│   ├─ [Category D: User-Weighted - 10 features]
│   ├─ [Category E: Temporal - 8 features]
│   └─ [Category F: Location - 5 features]
│
├── feature_engineering_report.md        [Documentation - 10KB]
│   │
│   ├─ Overview
│   ├─ Dataset Summary
│   ├─ Feature Categories (detailed)
│   ├─ Feature Statistics
│   ├─ Data Quality Report
│   └─ Next Steps
│
└── feature_categories/                  [Ablation Studies]
    │
    ├── business_static_features.csv            [10MB]
    │   └─ business_id + 8 static features
    │
    ├── review_aggregation_features.csv         [30MB]
    │   └─ business_id + 15 review features
    │
    ├── sentiment_features.csv                  [20MB]
    │   └─ business_id + 8 sentiment features
    │
    ├── user_weighted_features.csv              [25MB]
    │   └─ business_id + 10 weighted features
    │
    ├── temporal_features.csv                   [20MB]
    │   └─ business_id + 8 temporal features
    │
    └── location_category_features.csv          [10MB]
        └─ business_id + 5 location features
```

---

## Performance Metrics

```
┌──────────────────────────────────────────────────────────────┐
│                  EXECUTION PERFORMANCE                        │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  Expected Runtime: 20-30 minutes (full dataset)              │
│                                                               │
│  Stage Breakdown:                                            │
│  ────────────────                                            │
│   Stage 1: Load data               ~2 min   (  7%)           │
│   Stage 2: Static features         ~1 min   (  3%)           │
│   Stage 3: Review features        ~25 min   ( 83%)  ←MAIN    │
│   Stage 4: Location features       ~1 min   (  3%)           │
│   Stage 5: Merge & save            ~1 min   (  3%)           │
│                                    ──────   ──────           │
│                           TOTAL:   ~30 min  (100%)           │
│                                                               │
│  Memory Usage:                                               │
│  ─────────────                                               │
│   Peak:    ~500MB                                            │
│   Average: ~300MB                                            │
│   Minimum: ~150MB                                            │
│                                                               │
│  Disk I/O:                                                   │
│  ─────────                                                   │
│   Read:  ~6GB  (review_clean.csv chunked reading)           │
│   Write: ~200MB (feature outputs)                            │
│                                                               │
│  CPU Usage:                                                  │
│  ──────────                                                  │
│   Single-core (no parallelization)                          │
│   Expected: 80-100% during processing                        │
│                                                               │
│  Bottlenecks:                                                │
│  ────────────                                                │
│   1. Review chunked reading (I/O bound)                      │
│   2. VADER sentiment computation (CPU bound)                 │
│   3. GroupBy aggregations (CPU bound)                        │
│                                                               │
└──────────────────────────────────────────────────────────────┘
```

---

## Data Quality Assurance

```
┌──────────────────────────────────────────────────────────────┐
│                  QUALITY CHECKS                               │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  PRE-PROCESSING CHECKS:                                      │
│  ───────────────────────                                     │
│   ✓ Input files exist                                        │
│   ✓ No corrupted rows                                        │
│   ✓ Required columns present                                 │
│   ✓ Data types correct                                       │
│                                                               │
│  DURING PROCESSING:                                          │
│  ──────────────────                                          │
│   ✓ User credibility scores valid (0-1 range expected)      │
│   ✓ Sentiment scores valid (-1 to 1)                        │
│   ✓ No division by zero                                     │
│   ✓ Weights sum to 1 per business                           │
│                                                               │
│  POST-PROCESSING VALIDATION:                                 │
│  ──────────────────────────                                 │
│   ✓ Shape = (150346, 74)                                    │
│   ✓ Zero missing values                                      │
│   ✓ Zero infinite values                                     │
│   ✓ Feature ranges reasonable:                              │
│      • stars: [1, 5]                                         │
│      • review_count: [0, max]                                │
│      • sentiment: [-1, 1]                                    │
│      • credibility: [0, ~10]                                 │
│   ✓ Target distribution: ~80/20                              │
│                                                               │
└──────────────────────────────────────────────────────────────┘
```

---

## Next Phase Preview

```
┌──────────────────────────────────────────────────────────────┐
│           WHAT HAPPENS NEXT (Stage 4)                         │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  INPUT: business_features_final.csv (72 features)            │
│                                                               │
│  STEP 1: Feature Selection                                   │
│  ──────────────────────                                      │
│   Remove highly correlated features (|r| > 0.95)             │
│   Remove low variance features (var < 0.01)                  │
│   Rank by feature importance (Random Forest)                 │
│   Target: Reduce to 40-50 most informative features          │
│                                                               │
│  STEP 2: Class Imbalance Handling                           │
│  ─────────────────────────────                              │
│   Stratified train/test split (80/20)                        │
│   Apply SMOTE to training set                                │
│   Use class weights in models                                │
│                                                               │
│  STEP 3: Baseline Models                                     │
│  ───────────────────                                         │
│   Logistic Regression  → ROC-AUC, F1-score                   │
│   Decision Tree        → Interpretability                    │
│   Random Forest        → Feature importance                  │
│                                                               │
│  STEP 4: Ablation Studies                                    │
│  ─────────────────────                                       │
│   Use feature_categories/ files                              │
│   Train with/without each category                           │
│   Assess contribution to performance                          │
│                                                               │
│  OUTPUT: Baseline performance benchmarks                     │
│          Feature importance rankings                         │
│          Optimal feature subset                              │
│                                                               │
└──────────────────────────────────────────────────────────────┘
```

