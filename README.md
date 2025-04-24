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

\begin{table}[h]
\centering
\scriptsize  % smaller font to make the table more compact
\setlength{\tabcolsep}{5pt}  % slightly narrower columns
\renewcommand{\arraystretch}{1.15}  % slightly tighter row spacing
\caption{Performance of the XGBoost Model}
\label{tab:tuned_model_performance}
\begin{tabular}{llccc}
\toprule
\textbf{Split} & \textbf{Class} & \textbf{Precision} & \textbf{Recall} & \textbf{F1-score} \\
\midrule
Train & High & 0.83 & 0.97 & 0.89 \\
      & Low  & 0.82 & 0.67 & 0.74 \\
\textbf{Train Accuracy} & & \multicolumn{3}{r}{0.8263} \\
\midrule
Val & High  & 0.81 & 0.92 & 0.86 \\
    & Low   & 0.73 & 0.51 & 0.60 \\
\textbf{Val Accuracy} & & \multicolumn{3}{r}{0.7981} \\
\midrule
Test & High & 0.91 & 0.97 & 0.94 \\
     & Low  & 0.89 & 0.72 & 0.80 \\
\textbf{Test Accuracy} & & \multicolumn{3}{r}{0.8957} \\
\bottomrule
\end{tabular}
\parbox{0.9\linewidth}{
\footnotesize
\textit{Note:} Scores are reported per class, with macro-averaged F1-scores and overall accuracy shown for each split. The tuned model performs strongly across splits and classes.}
\end{table}

\vspace{-2.5em}
\begin{table}[h]
\centering
\scriptsize
\setlength{\tabcolsep}{5pt}
\renewcommand{\arraystretch}{1.15}
\caption{Performance of the LSTM Model}
\label{tab:tuned_rnn_performance}
\begin{tabular}{llccc}
\toprule
\textbf{Split} & \textbf{Class} & \textbf{Precision} & \textbf{Recall} & \textbf{F1-score} \\
\midrule
Train & High & 0.82 & 0.95 & 0.88 \\
      & Low  & 0.82 & 0.94 & 0.88 \\
\textbf{Train Accuracy} & & \multicolumn{3}{r}{0.8777} \\
\midrule
Val & High  & 0.88 & 0.96 & 0.92 \\
    & Low   & 0.88 & 0.98 & 0.92 \\
\textbf{Val Accuracy} & & \multicolumn{3}{r}{0.9236} \\
\midrule
Test & High & 0.80 & 0.95 & 0.87 \\
     & Low  & 0.80 & 0.95 & 0.87 \\
\textbf{Test Accuracy} & & \multicolumn{3}{r}{0.8651} \\
\bottomrule
\end{tabular}

\vspace{0.5em}
\parbox{0.9\linewidth}{
\footnotesize
\textit{Note:} Scores are reported per class, with macro-averaged F1-scores and overall accuracy shown for each split.
}
\end{table}


## Setup

1. Create a conda virtual environment:
```bash
conda create -n myenv
conda activate myenv
```

2. Install dependencies:
```bash
conda install pip
conda install --file requirements.txt
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
