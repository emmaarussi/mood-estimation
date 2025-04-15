import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import sys
sys.path.append('..')
from feature_engineering.feature_engineering import (
    create_temporal_features,
    create_lag_features,
    create_activity_features,
    create_communication_features,
    create_app_usage_features,
    create_circumplex_features,
    pivot_long_to_wide
)

def smape(y_true, y_pred):
    """Symmetric Mean Absolute Percentage Error"""
    return 100 * np.mean(np.abs(y_pred - y_true) / ((np.abs(y_true) + np.abs(y_pred)) / 2))

def wmape(y_true, y_pred):
    """Weighted Mean Absolute Percentage Error"""
    return np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true)) * 100

def prepare_features(df, target_col='mood', window_sizes=[24, 48, 72, 168]):
    """Prepare all features for modeling"""
    print("Creating features...")
    
    # Convert to wide format if needed
    if 'variable' in df.columns:
        df = pivot_long_to_wide(df)
    
    # Sort by time first
    df = df.sort_values(['id', 'time'])
    
    # Create basic temporal features (no data leakage)
    features = create_temporal_features(df)
    
    # Create lagged features manually to prevent leakage
    for lag in [8, 16, 24, 48, 72, 168]:
        # Simple lag
        features[f'mood_lag_{lag}h'] = features.groupby('id')[target_col].shift(lag)
        
        # Rolling stats on past data only
        features[f'mood_rolling_mean_{lag}h'] = features.groupby('id')[target_col].transform(
            lambda x: x.shift(1).rolling(window=lag, min_periods=1).mean()
        )
        features[f'mood_rolling_std_{lag}h'] = features.groupby('id')[target_col].transform(
            lambda x: x.shift(1).rolling(window=lag, min_periods=1).std()
        )
        features[f'mood_rolling_min_{lag}h'] = features.groupby('id')[target_col].transform(
            lambda x: x.shift(1).rolling(window=lag, min_periods=1).min()
        )
        features[f'mood_rolling_max_{lag}h'] = features.groupby('id')[target_col].transform(
            lambda x: x.shift(1).rolling(window=lag, min_periods=1).max()
        )
    
    # Activity features with proper lag
    for col in ['activity', 'screen']:
        if col in features.columns:
            for window in window_sizes:
                features[f'{col}_past_{window}h'] = features.groupby('id')[col].transform(
                    lambda x: x.shift(1).rolling(window=window, min_periods=1).mean()
                )
    
    # Communication features with proper lag
    for col in ['call', 'sms']:
        if col in features.columns:
            for window in window_sizes:
                features[f'{col}_past_{window}h'] = features.groupby('id')[col].transform(
                    lambda x: x.shift(1).rolling(window=window, min_periods=1).sum()
                )
    
    # Emotion features with proper lag
    for col in ['circumplex_arousal', 'circumplex_valence']:
        if col in features.columns:
            for window in window_sizes:
                features[f'{col}_past_{window}h'] = features.groupby('id')[col].transform(
                    lambda x: x.shift(1).rolling(window=window, min_periods=1).mean()
                )
    
    # User-specific features using only past data
    features['user_avg_mood'] = features.groupby('id')[target_col].transform(
        lambda x: x.expanding().mean().shift(1)
    )
    features['mood_vs_average'] = features[target_col] - features['user_avg_mood']
    
    # Drop rows with missing target
    features = features.dropna(subset=[target_col])
    
    return features

def evaluate_predictions(y_true, y_pred, title="Model Performance"):
    """Calculate and display multiple performance metrics"""
    metrics = {
        'R²': r2_score(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'SMAPE': smape(y_true, y_pred),
        'WMAPE': wmape(y_true, y_pred)
    }
    
    print(f"\n{title}:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    return metrics

def plot_results(y_true, y_pred, dates, title="Predictions vs Actual"):
    """Create visualization of predictions"""
    plt.figure(figsize=(15, 10))
    
    # Scatter plot
    plt.subplot(2, 1, 1)
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    plt.xlabel("Actual Mood")
    plt.ylabel("Predicted Mood")
    plt.title(f"{title} - Scatter Plot")
    
    # Time series plot
    plt.subplot(2, 1, 2)
    plt.plot(dates, y_true, label='Actual', alpha=0.7)
    plt.plot(dates, y_pred, label='Predicted', alpha=0.7)
    plt.xlabel("Date")
    plt.ylabel("Mood")
    plt.title(f"{title} - Time Series")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'data_analysis/plots/modeling/{title.lower().replace(" ", "_")}.png')
    plt.close()

def main():
    # Load data
    print("Loading data...")
    df = pd.read_csv('data/dataset_mood_smartphone_cleaned.csv')
    df['time'] = pd.to_datetime(df['time'], format='mixed')
    
    # Prepare features
    features = prepare_features(df)
    
    # Prepare train/val/test splits
    train_end = pd.to_datetime('2014-05-08')
    val_end = pd.to_datetime('2014-05-23')
    
    # Add buffer days between splits to ensure independence
    buffer_days = 7  # One week buffer
    
    # Split features with buffer
    train_mask = features['time'] <= (train_end - pd.Timedelta(days=buffer_days))
    val_mask = (features['time'] > (train_end + pd.Timedelta(days=buffer_days))) & \
               (features['time'] <= (val_end - pd.Timedelta(days=buffer_days)))
    test_mask = features['time'] > (val_end + pd.Timedelta(days=buffer_days))
    
    # Drop buffer periods
    features = features[train_mask | val_mask | test_mask].copy()
    
    # Handle categorical variables
    features = pd.get_dummies(features, columns=['time_of_day'])
    
    # Get feature columns (exclude id, time, and target)
    feature_cols = features.columns.difference(['id', 'time', 'mood'])
    
    # Fill missing values
    for col in feature_cols:
        if features[col].dtype in [np.float64, np.int64]:
            features[col] = features[col].fillna(features[col].mean())
        else:
            features[col] = features[col].fillna(0)
    
    # Split data
    X_train = features[train_mask][feature_cols]
    y_train = features[train_mask]['mood']
    X_val = features[val_mask][feature_cols]
    y_val = features[val_mask]['mood']
    X_test = features[test_mask][feature_cols]
    y_test = features[test_mask]['mood']
    
    # Train model with better parameters
    print("Training model...")
    model = xgb.XGBRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        min_child_weight=2,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=1,
        objective='reg:squarederror',
        random_state=42
    )
    
    # Train with early stopping
    eval_set = [(X_val, y_val)]
    model.fit(
        X_train, y_train,
        eval_set=eval_set,
        verbose=False
    )
    
    # Make predictions
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)
    
    # Evaluate performance
    train_metrics = evaluate_predictions(y_train, y_train_pred, "Training Performance")
    val_metrics = evaluate_predictions(y_val, y_val_pred, "Validation Performance")
    test_metrics = evaluate_predictions(y_test, y_test_pred, "Test Performance")
    
    # Create visualizations
    plot_results(y_train, y_train_pred, features[train_mask]['time'], "Training Results")
    plot_results(y_val, y_val_pred, features[val_mask]['time'], "Validation Results")
    plot_results(y_test, y_test_pred, features[test_mask]['time'], "Test Results")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    })
    feature_importance = feature_importance.sort_values('importance', ascending=False)
    
    plt.figure(figsize=(12, 8))
    sns.barplot(data=feature_importance.head(20), x='importance', y='feature')
    plt.title("Top 20 Feature Importance")
    plt.tight_layout()
    plt.savefig('data_analysis/plots/modeling/feature_importance_full.png')
    plt.close()
    
    print("\nTop 20 Features:")
    print(feature_importance.head(20))
    
    # Save model
    model.save_model('models/xgboost_full.model')
    print("\nModel saved to 'xgboost_full.model'")

if __name__ == "__main__":
    main()
