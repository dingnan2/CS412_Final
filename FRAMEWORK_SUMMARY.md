# CS 412 Research Project Framework - Complete Setup

## 🎯 Project Overview

This framework implements a **User-weighted Ensemble Framework** for predicting business success using the Yelp dataset. The system combines multiple machine learning models with user credibility weighting and category-aware classification.

## 📁 Project Structure Created

```
CS412_Final Project/
├── 📄 README.md                    # Main project documentation
├── 📄 requirements.txt             # Python dependencies
├── 📄 main.py                      # Main execution script
├── 📄 run_baseline.py              # Baseline models only
├── 📄 run_ensemble.py             # Ensemble framework only
├── 📁 data/                       # Data storage
│   ├── 📁 raw/                    # Original Yelp dataset
│   ├── 📁 processed/              # Processed data files
│   └── 📁 samples/                # Sample data for testing
├── 📁 src/                        # Source code
│   ├── 📁 data_processing/        # Data loading and preprocessing
│   │   ├── 📄 data_processor.py   # Main data processing class
│   │   └── 📄 preprocess.py       # Data processing script
│   ├── 📁 feature_engineering/    # Feature extraction
│   │   └── 📄 feature_extractor.py # Feature engineering modules
│   ├── 📁 models/                 # ML models and ensemble
│   │   ├── 📄 baseline_models.py  # Baseline models (LR, DT, RF)
│   │   └── 📄 ensemble_framework.py # User-weighted ensemble
│   ├── 📁 evaluation/             # Evaluation metrics
│   │   └── 📄 evaluator.py        # Model evaluation framework
│   └── 📁 utils/                  # Utility functions
│       ├── 📄 config.py          # Configuration management
│       └── 📄 utils.py            # General utilities
├── 📁 notebooks/                  # Jupyter notebooks
│   └── 📄 quick_start.py          # Quick start example
├── 📁 config/                     # Configuration files
│   └── 📄 config.yaml             # Main configuration
├── 📁 results/                    # Model outputs and results
├── 📁 docs/                       # Documentation
│   ├── 📄 README.md               # Detailed documentation
│   └── 📄 DATA_SETUP.md           # Data setup instructions
└── 📁 tests/                      # Unit tests
    └── 📄 test_project.py         # Test suite
```

## 🚀 Key Features Implemented

### 1. **Data Processing Pipeline**
- ✅ JSON data loading and parsing
- ✅ Temporal feature extraction
- ✅ User-weighted aggregations
- ✅ Business attribute processing
- ✅ Data validation and cleaning

### 2. **Feature Engineering**
- ✅ Sentiment analysis (TextBlob/VADER)
- ✅ Temporal trend features
- ✅ Business category encoding
- ✅ Text-based features (TF-IDF)
- ✅ Location-based features

### 3. **Baseline Models**
- ✅ Logistic Regression
- ✅ Decision Tree
- ✅ Random Forest
- ✅ Cross-validation
- ✅ Class imbalance handling

### 4. **User-weighted Ensemble Framework**
- ✅ User credibility scoring
- ✅ Category-aware classification
- ✅ Multi-model ensemble (RF, XGBoost, Neural Networks)
- ✅ Weighted voting system

### 5. **Evaluation Framework**
- ✅ Comprehensive metrics (ROC-AUC, Precision, Recall, F1)
- ✅ Cross-validation
- ✅ Temporal validation
- ✅ Model comparison
- ✅ Visualization tools

### 6. **Configuration Management**
- ✅ YAML-based configuration
- ✅ Model parameter management
- ✅ Feature engineering settings
- ✅ Evaluation parameters

## 🛠️ Usage Instructions

### Quick Start
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download Yelp dataset to data/raw/
# (See docs/DATA_SETUP.md for details)

# 3. Run full pipeline
python main.py

# 4. Or run individual components
python run_baseline.py      # Baseline models only
python run_ensemble.py     # Ensemble framework only
```

### Configuration
Edit `config/config.yaml` to modify:
- Model parameters
- Feature engineering settings
- Evaluation metrics
- Data paths

## 📊 Expected Outputs

The framework will generate:
- **Models**: Trained model files in `results/models/`
- **Plots**: Visualization outputs in `results/plots/`
- **Reports**: Evaluation reports in `results/reports/`
- **Metrics**: Detailed JSON results in `results/evaluation_results.json`

## 🎯 Methodology Alignment

The framework implements the exact methodology described in your research proposal:

### Phase 1: Feature Engineering ✅
- Sentiment features from reviews
- Rating velocity and temporal trends
- Review volume metrics
- Business attribute features
- User engagement patterns

### Phase 2: Novel Framework Development ✅
- **User-weighted Aggregation**: Users weighted by platform tenure and usefulness votes
- **Multi-level Classification**: Category-specific survival prediction models
- **Ensemble Architecture**: Combines Random Forest, XGBoost, and Neural Networks

### Phase 3: Model Training and Validation ✅
- Stratified sampling (80/20 split)
- K-fold cross-validation
- Hyperparameter optimization

### Phase 4: Evaluation and Analysis ✅
- ROC-AUC, precision/recall, F1-scores
- Multi-task analysis
- Feature importance analysis

## 🔧 Technical Implementation

- **Language**: Python 3.8+
- **ML Libraries**: scikit-learn, XGBoost, TensorFlow
- **Data Processing**: pandas, numpy
- **Text Processing**: NLTK, TextBlob, spaCy
- **Visualization**: matplotlib, seaborn, plotly
- **Configuration**: PyYAML

## 📈 Next Steps

1. **Download Yelp Dataset**: Place JSON files in `data/raw/`
2. **Run Experiments**: Execute the full pipeline
3. **Analyze Results**: Review outputs in `results/` directory
4. **Customize**: Modify configuration for different experiments
5. **Extend**: Add new features or models as needed

## 🎉 Ready to Use!

Your CS 412 research project framework is now complete and ready for experimentation. The system provides a solid foundation for implementing your user-weighted ensemble approach to business success prediction using the Yelp dataset.

**Happy coding and good luck with your research! 🚀**
