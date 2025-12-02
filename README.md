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
```bash
# Phase 1: Data preprocessing
python src/data_processing/data_preprocessing.py

# Phase 2: EDA
python src/data_processing/EDA_analysis.py

# Phase 3: Feature engineering (temporal mode)
python src/feature_engineering/feature_eng.py --temporal --years 2012-2020

# Phase 4: Baseline models
python src/models/baseline_models.py --temporal

# Phase 5: Temporal validation
python src/models/temporal_validation.py

# Phase 6: Advanced models
python src/models/advanced_models.py

# Phase 7: Ablation study
python src/evaluation/ablation_study.py

# Phase 8: Case studies
python src/evaluation/case_study.py

# Phase 9: Generate final report
python src/reporting/generate_final_report.py
```

## Key Results

- **Best Model**: Ensemble (Stacking) with ROC-AUC = 0.82
- **Temporal Leakage Impact**: 15-point drop after correction (0.95 → 0.80)
- **Most Important Features**: User credibility, review recency, temporal trends
- **User Credibility Impact**: +3% improvement in ROC-AUC

## Reports

All reports are in `docs/`:
- `final_report.md`: Comprehensive project report
- `final_report.tex`: LaTeX version for submission
- Phase-specific reports in respective directories

## Contact

For questions or issues, contact:
- Adeniran Coker: ac171@illinois.edu
- Ju-Bin Choi: jubinc2@illinois.edu
- Carmen Zheng: dingnan2@illinois.edu