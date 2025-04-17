import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

def smape(y_true, y_pred):
    """Symmetric Mean Absolute Percentage Error"""
    return 100 * np.mean(np.abs(y_pred - y_true) / ((np.abs(y_true) + np.abs(y_pred)) / 2))

def wmape(y_true, y_pred):
    """Weighted Mean Absolute Percentage Error"""
    return np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true)) * 100



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

def main(input_file=None):
    import os
    import sys
    
    # Handle input file path
    if input_file is None:
        if len(sys.argv) > 1:
            input_file = sys.argv[1]
        else:
            input_file = 'data/mood_prediction_features.csv'
    
    if not os.path.exists(input_file):
        print(f"Error: Input file not found: {input_file}")
        sys.exit(1)
    
    # Create necessary directories
    os.makedirs('data_analysis/plots/modeling', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # Load data
    print("Loading data...")
    features = pd.read_csv(input_file)
    features['time'] = pd.to_datetime(features['time'], format='mixed')
    
    # Filter for rows where we have mood recordings
    print("\nInitial shape:", features.shape)
    features = features.dropna(subset=['mood'])
    print("Shape after filtering for mood recordings:", features.shape)
    
    # Check data
    print("\nMood recording statistics:")
    print(features['mood'].describe())
    
    # Check feature NaN percentages
    print("\nFeature NaN percentages:")
    nan_percentages = features.isna().mean() * 100
    print(nan_percentages.sort_values(ascending=False).head())
    
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
    print("\nModel saved to 'models/xgboost_full.model'")

if __name__ == "__main__":
    main()
