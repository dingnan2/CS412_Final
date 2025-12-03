# CS 412 Research Project: Business Success Prediction

## Team Members
- Adeniran Coker
- Ju-Bin Choi
- Carmen Zheng

## Project Overview

This project predicts business success (open/closed status) 6-12 months in advance
using the Yelp dataset. We introduce novel methods for:

1. **User Credibility Weighting**: Weight reviews by reviewer credibility
2. **Temporal Validation**: Prevent data leakage in time-series prediction
3. **Label Inference**: Infer historical closure dates from review patterns

## Repository Structure

```
CS412_Final_Project/
├── data/
│   ├── processed/          # Cleaned data
│   └── features/           # Engineered features
├── src/
│   ├── data_processing/    # Phase 1-2: Data cleaning, EDA
│   ├── feature_engineering/# Phase 3: Feature engineering
│   ├── models/             # Phase 4-6: Model training
│   ├── evaluation/         # Phase 7-8: Ablation, case studies
│   ├── reporting/          # Phase 9: Final report generation
│   └── utils/              # Utility functions
├── docs/
│   ├── final_report.md     # Final report (markdown)
│   ├── final_report.tex    # Final report (LaTeX)
│   └── figures/            # All figures
└── README.md
```

## Quick Start

### Prerequisites
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
pip install xgboost lightgbm vaderSentiment
```


### If You Have Not Downloaded the Yelp Dataset Yet

Before running any preprocessing or modeling scripts, ensure that you have downloaded the **Yelp Open Dataset** and placed it under:

```
data/raw/
    yelp_academic_dataset_business.json
    yelp_academic_dataset_review.json
    yelp_academic_dataset_user.json
```

You can download the dataset from the official source:
[Yelp Dataset](https://www.kaggle.com/datasets/yelp-dataset/yelp-dataset/data)

If the files are missing, the pipeline will not run successfully.
After downloading, simply place the JSON files into `data/raw/` and proceed with the following pipelines.



### Run Complete Pipeline

**Recommended: Use the unified entry point `main.py`**

```bash
# Run all phases (1-10) in order
python main.py

# Or explicitly specify --all
python main.py --all

# Run specific phases only (e.g., phases 3, 4, and 6)
python main.py --phase 3 4 6
```

**Pipeline Phases:**
1. Data preprocessing (clean raw Yelp JSON to CSV)
2. Exploratory data analysis (EDA plots + report)
3. Feature engineering (temporal features, user credibility, 2012–2020)
4. Temporal validation (label inference + temporal split, 12-month window)
5. Baseline models (Logistic Regression, Decision Tree, Random Forest)
6. Advanced models (XGBoost, LightGBM, Neural Network, Ensembles)
7. Ablation study (feature category analysis)
8. Case study (TP/TN/FP/FN error analysis)
9. Parameter study (hyperparameter sensitivity analysis)
10. Final report generation (aggregate results to `docs/`)

**Note:** All modeling phases (5-9) use the same data file and split configuration from `config.py` to ensure consistent and comparable results.

## Key Results

- **Best Model**: XGBoost with ROC-AUC = 0.886
- **Baseline Performance**: Random Forest (ClassWeight) with ROC-AUC = 0.835
- **Improvement**: +6.1% relative improvement over baseline
- **Temporal Leakage Impact**: 15-point drop after correction (0.95 → 0.84)
- **User Credibility Impact**: +0.017 AUC (2.0% relative) validated through ablation study
- **Most Important Features**: Review count, user credibility metrics (avg_reviewer_tenure), review frequency

## Contact

For questions or issues, contact:
- Adeniran Coker: ac171@illinois.edu
- Ju-Bin Choi: jubinc2@illinois.edu
- Carmen Zheng: dingnan2@illinois.edu