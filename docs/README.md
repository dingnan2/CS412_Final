"""
CS 412 Research Project Documentation

## Project Overview

This project implements a user-weighted ensemble framework for predicting business success using the Yelp dataset. The framework combines multiple machine learning models with user credibility weighting and category-aware classification.

## Project Structure

```
├── data/                    # Data storage
│   ├── raw/                # Original Yelp dataset
│   ├── processed/          # Processed data files
│   └── samples/            # Sample data for testing
├── src/                    # Source code
│   ├── data_processing/    # Data loading and preprocessing
│   ├── feature_engineering/ # Feature extraction
│   ├── models/             # ML models and ensemble
│   ├── evaluation/         # Evaluation metrics
│   └── utils/              # Utility functions
├── notebooks/              # Jupyter notebooks
├── config/                 # Configuration files
├── results/                # Model outputs and results
├── docs/                   # Documentation
└── tests/                  # Unit tests
```

## Key Components

### 1. Data Processing (`src/data_processing/`)
- **YelpDataProcessor**: Main class for loading and processing Yelp dataset
- Handles JSON parsing, data cleaning, and feature creation
- Creates temporal features, user-weighted aggregations

### 2. Feature Engineering (`src/feature_engineering/`)
- **SentimentAnalyzer**: Text sentiment analysis using TextBlob/VADER
- **TemporalFeatureExtractor**: Time-based feature extraction
- **BusinessAttributeExtractor**: Business metadata feature extraction
- **TextFeatureExtractor**: TF-IDF and text-based features

### 3. Models (`src/models/`)
- **BaselineModels**: Logistic Regression, Decision Tree, Random Forest
- **UserWeightedEnsemble**: Advanced ensemble with user weighting and category awareness

### 4. Evaluation (`src/evaluation/`)
- **ModelEvaluator**: Comprehensive evaluation metrics
- **ValidationFramework**: Cross-validation and temporal validation

## Usage

### Quick Start
```bash
# Install dependencies
pip install -r requirements.txt

# Run full pipeline
python main.py

# Run baseline models only
python run_baseline.py

# Run ensemble framework only
python run_ensemble.py
```

### Data Processing
```python
from src.data_processing.data_processor import YelpDataProcessor

processor = YelpDataProcessor()
processed_data = processor.process_all_data(sample_size=10000)
merged_df = processor.create_merged_dataset()
```

### Feature Engineering
```python
from src.feature_engineering.feature_extractor import FeatureEngineer

feature_engineer = FeatureEngineer()
df_with_features = feature_engineer.engineer_features(df)
X, feature_names = feature_engineer.select_features(df_with_features)
```

### Model Training
```python
from src.models.baseline_models import BaselineModels

baseline_models = BaselineModels()
results = baseline_models.train_all_models(X, y)
```

## Configuration

Edit `config/config.yaml` to modify:
- Model parameters
- Feature engineering settings
- Evaluation metrics
- Data paths

## Results

Results are saved in the `results/` directory:
- `models/`: Trained model files
- `plots/`: Visualization outputs
- `reports/`: Evaluation reports
- `evaluation_results.json`: Detailed metrics

## Methodology

### Phase 1: Feature Engineering
- Sentiment analysis from reviews
- Rating velocity and temporal trends
- Review volume metrics
- Business attribute encoding
- User engagement patterns

### Phase 2: User-weighted Ensemble Framework
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

## Challenges and Mitigation

1. **Data Imbalance**: Use stratified sampling and SMOTE
2. **High-Dimensional Features**: Apply dimensionality reduction
3. **Computational Complexity**: Optimize with efficient implementations

## Team Members

- **Adeniran Coker** (ac171) - Ph.D Civil Engineering
- **Ju-Bin Choi** (jubinc2) - Master of Computer Science  
- **Carmen Zheng** (dingnan2) - Master of Computer Science

## License

This project is for academic research purposes only.
