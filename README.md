# Mood Estimation Project

This project analyzes smartphone usage patterns to predict user mood using machine learning techniques. It processes raw smartphone usage data to extract behavioral patterns and uses XGBoost models to predict user mood.

## Project Structure

```
mood-estimation/
├── data/                          # Data files (not in repo)
│   ├── dataset_mood_smartphone.csv    # Raw dataset
│   ├── dataset_mood_smartphone_cleaned.csv  # Cleaned dataset
│   ├── mood_prediction_features.csv   # Full feature set
│   └── mood_prediction_simple_features.csv  # Simple feature set
│
├── data_analysis/                 # Data cleaning and analysis
│   ├── data_cleaning.py              # Data cleaning pipeline
│   ├── analyze_cleaned_data.py       # Analysis of cleaned data
│   ├── mood_variable_analysis.py     # Analysis of mood variables
│   ├── temporal_patterns.py          # Temporal pattern analysis
│   └── time_analysis.py              # Time-based analysis
│
├── feature_engineering/           # Feature creation and analysis
│   ├── feature_engineering.py        # Complex feature engineering
│   ├── feature_analysis.py           # Feature analysis
│   └── simple_feature_engineering.py  # Basic feature creation
│
└── modeling/                      # Model implementation and evaluation
    ├── xgboost_full_features.py      # Complex feature model
    ├── xgboost_simple_features.py    # Simple feature model
    └── error_analysis.py             # Model error analysis
```

## Features

### Simple Features (31 total)
- Basic temporal patterns
- Recent history (24h, 48h, 72h, 168h windows)
- User baseline features

### Complex Features (119 total)
- Advanced temporal patterns
- Circumplex emotion model features
- Fine-grained app usage patterns
- Communication patterns
- Activity patterns
- Rolling statistics over multiple time windows

## Model Performance

### Simple Features Model
- Training R²: 0.576, RMSE: 0.452
- Validation R²: 0.149, RMSE: 0.555
- Test R²: 0.034, RMSE: 0.636

### Complex Features Model
- Training R²: 0.991, RMSE: 0.095
- Validation R²: 0.996, RMSE: 0.037
- Test R²: 0.992, RMSE: 0.068

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Unix/macOS
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Clean and analyze the data:
```bash
python data_analysis/data_cleaning.py
python data_analysis/analyze_cleaned_data.py
```

2. Generate features:
```bash
python feature_engineering/feature_engineering.py  # Complex features
python feature_engineering/simple_feature_engineering.py  # Simple features
```

3. Train and evaluate models:
```bash
python modeling/xgboost_simple_features.py
python modeling/xgboost_full_features.py
```

## Note
The data files and generated outputs (plots, models) are not included in the repository to save space. Running the scripts will generate these files locally.
