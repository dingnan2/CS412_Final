# CS 412 Research Project: Business Success Prediction using Yelp Dataset

## Project Overview

This project predicts restaurant business success through a temporal-aware framework that jointly predicts business survival and future rating trajectories. The framework implements a **User-weighted Ensemble Framework** with multi-level classification to handle the complexity of business success prediction.

## Team Members

- **Adeniran Coker** (ac171) - Ph.D Civil Engineering
- **Ju-Bin Choi** (jubinc2) - Master of Computer Science  
- **Carmen Zheng** (dingnan2) - Master of Computer Science

## Dataset

- **Source**: Yelp Dataset (150K+ businesses, 8M+ reviews, 2M+ users)
- **Size**: ~9.5 GB total (reviews: 5.3 GB, users: 3.3 GB, business: 118 MB, check-ins: 287 MB, tips: 181 MB)
- **Format**: JSON files

## Project Structure

```
├── data/                    # Raw and processed data
│   ├── raw/                # Original Yelp dataset files
│   ├── processed/          # Cleaned and feature-engineered data
│   └── samples/            # Sample data for testing
├── src/                    # Source code
│   ├── data_processing/    # Data loading and preprocessing
│   ├── feature_engineering/ # Feature extraction and engineering
│   ├── models/             # Machine learning models
│   ├── evaluation/         # Evaluation metrics and validation
│   └── utils/              # Utility functions
├── notebooks/              # Jupyter notebooks for analysis
├── config/                 # Configuration files
├── results/                # Model outputs and results
├── docs/                   # Documentation
└── tests/                  # Unit tests
```

## Methodology

### Phase 1: Feature Engineering
- Sentiment analysis from reviews
- Rating velocity and temporal trends
- Review volume metrics
- Business attribute encoding
- User engagement patterns

### Phase 2: Novel Framework Development
- **User-weighted Aggregation**: Weight users by platform tenure and usefulness votes
- **Multi-level Classification**: Category-specific models
- **Ensemble Architecture**: Combine Random Forest, XGBoost, and Neural Networks

### Phase 3: Model Training and Validation
- Stratified sampling (80/20 split)
- K-fold cross-validation
- Hyperparameter optimization

### Phase 4: Evaluation and Analysis
- ROC-AUC, precision/recall, F1-scores
- Multi-task analysis
- Feature importance analysis

## Installation

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Download Yelp dataset to `data/raw/`
4. Run preprocessing: `python src/data_processing/preprocess.py`

## Usage

```bash
# Run baseline models
python src/models/baseline_models.py

# Train ensemble framework
python src/models/ensemble_framework.py

# Evaluate models
python src/evaluation/evaluate_models.py
```

## Milestones

- **Stage 1**: Data Preparation and EDA (Sept 16 - Sept 30)
- **Stage 2**: Baseline Implementation (Oct 1 - Oct 15)  
- **Stage 3**: Advanced Framework Development (Oct 16 - Oct 29)
- **Stage 4**: Model Evaluation and Analysis (Oct 30 - Nov 15)
- **Stage 5**: Final Report and Documentation (Nov 16 - Dec 2)

## Challenges & Mitigation

1. **Data Imbalance**: Use stratified sampling and SMOTE
2. **High-Dimensional Features**: Apply dimensionality reduction
3. **Computational Complexity**: Optimize with efficient implementations

## License

This project is for academic research purposes only.
