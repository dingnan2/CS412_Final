# CS 412 Research Project - Data Setup Instructions

## Dataset Download

1. **Download Yelp Dataset**:
   - Visit: https://www.yelp.com/dataset
   - Download the Yelp Academic Dataset
   - Extract the files to `data/raw/` directory

2. **Required Files**:
   - `yelp_academic_dataset_business.json`
   - `yelp_academic_dataset_review.json`
   - `yelp_academic_dataset_user.json`
   - `yelp_academic_dataset_checkin.json`
   - `yelp_academic_dataset_tip.json`

## Directory Structure After Setup

```
data/
├── raw/
│   ├── yelp_academic_dataset_business.json
│   ├── yelp_academic_dataset_review.json
│   ├── yelp_academic_dataset_user.json
│   ├── yelp_academic_dataset_checkin.json
│   └── yelp_academic_dataset_tip.json
├── processed/
│   ├── business_processed.csv
│   ├── reviews_processed.csv
│   ├── users_processed.csv
│   ├── checkins_processed.csv
│   ├── tips_processed.csv
│   ├── user_weighted_processed.csv
│   └── merged_dataset.csv
└── samples/
    └── (sample data for testing)
```

## Quick Start

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Download and Place Data**:
   - Download Yelp dataset
   - Place JSON files in `data/raw/` directory

3. **Run Data Processing**:
   ```bash
   python src/data_processing/preprocess.py
   ```

4. **Run Full Pipeline**:
   ```bash
   python main.py
   ```

## Sample Data for Testing

If you want to test the framework without the full dataset:

1. Create sample data files in `data/samples/`
2. Modify the data processor to use sample data
3. Run with `sample_size=1000` parameter

## Troubleshooting

### Common Issues:

1. **Memory Issues**: Use `sample_size` parameter to limit data size
2. **File Not Found**: Ensure JSON files are in correct directory
3. **Permission Errors**: Check file permissions and directory access

### Performance Tips:

1. **Start Small**: Use sample data for initial testing
2. **Incremental Processing**: Process data in chunks
3. **Memory Management**: Monitor memory usage during processing

## Data Format

The Yelp dataset contains JSON files with the following structure:

- **Business**: Business metadata, categories, attributes
- **Review**: User reviews with ratings and text
- **User**: User profiles and statistics
- **Check-in**: Business check-in data
- **Tip**: User tips and recommendations

Each file contains one JSON object per line (JSONL format).
