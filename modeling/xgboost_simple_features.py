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

def prepare_rolling_window_data(df, window_size=7):
    """Prepare data with rolling window features"""
    # Sort by user and time
    df = df.sort_values(['id', 'time'])
    
    # Create daily aggregates
    daily = df.groupby(['id', df['time'].dt.date]).agg({
        'mood': 'mean',
        'recent_activity': 'mean',
        'daily_screen_time': 'sum',
        'communication_time': 'sum',
        'circumplex_arousal': 'mean',
        'circumplex_valence': 'mean',
        'emotion_intensity': 'mean',
        'hour': lambda x: len(x),  # number of measurements
    }).reset_index()
    daily.columns = ['id', 'date'] + list(daily.columns[2:])
    daily['date'] = pd.to_datetime(daily['date'])
    
    # Create features from rolling windows
    features = []
    targets = []
    dates = []
    user_ids = []
    
    for user in daily['id'].unique():
        user_data = daily[daily['id'] == user].copy()
        
        for i in range(len(user_data) - window_size):
            window = user_data.iloc[i:i+window_size]
            target_day = user_data.iloc[i+window_size]
            
            # Skip if gap is more than 1 day
            if (target_day['date'] - window['date'].iloc[-1]).days > 1:
                continue
                
            # Create features
            feature_dict = {
                'user_id': user,
                'date': target_day['date'],
                'measurements': window['hour'].mean(),
                'mood_mean': window['mood'].mean(),
                'mood_std': window['mood'].std(),
                'mood_trend': window['mood'].iloc[-1] - window['mood'].iloc[0],
                'activity_mean': window['recent_activity'].mean(),
                'screen_time_mean': window['daily_screen_time'].mean(),
                'communication_mean': window['communication_time'].mean(),
                'arousal_mean': window['circumplex_arousal'].mean(),
                'valence_mean': window['circumplex_valence'].mean(),
                'emotion_mean': window['emotion_intensity'].mean(),
                'day_of_week': target_day['date'].dayofweek,
                'is_weekend': target_day['date'].dayofweek >= 5
            }
            
            features.append(feature_dict)
            targets.append(target_day['mood'])
            dates.append(target_day['date'])
            user_ids.append(user)
    
    X = pd.DataFrame(features)
    y = np.array(targets)
    
    return X, y, dates, user_ids

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
    dates = pd.to_datetime(dates)
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
            input_file = 'data/mood_prediction_simple_features.csv'
    
    if not os.path.exists(input_file):
        print(f"Error: Input file not found: {input_file}")
        sys.exit(1)
    
    # Create necessary directories
    os.makedirs('data_analysis/plots/modeling', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # Load data
    print("Loading data...")
    df = pd.read_csv(input_file)
    df['time'] = pd.to_datetime(df['time'])
    
    # Print initial stats
    print(f"\nInitial shape: {df.shape}")
    print("\nMood recording statistics:")
    print(df['mood'].describe())
    print(f"\nNaN in mood: {df['mood'].isna().sum()}")
    
    # Prepare train/val/test splits
    train_end = pd.to_datetime('2014-05-08')
    val_end = pd.to_datetime('2014-05-23')
    
    # Prepare features
    print("Preparing features...")
    X, y, dates, user_ids = prepare_rolling_window_data(df)
    dates = pd.to_datetime(dates)
    
    # Split data
    train_mask = dates <= train_end
    val_mask = (dates > train_end) & (dates <= val_end)
    test_mask = dates > val_end
    
    # Recompute masks after filtering
    train_mask = dates <= train_end
    val_mask = (dates > train_end) & (dates <= val_end)
    test_mask = dates > val_end
    
    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    
    # Train model
    print("Training model...")
    model = xgb.XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        objective='reg:squarederror'
    )
    
    eval_set = [(X_val.drop(['user_id', 'date'], axis=1), y_val)]
    model.fit(
        X_train.drop(['user_id', 'date'], axis=1),
        y_train,
        eval_set=eval_set,
        verbose=False
    )
    
    # Make predictions
    y_train_pred = model.predict(X_train.drop(['user_id', 'date'], axis=1))
    y_val_pred = model.predict(X_val.drop(['user_id', 'date'], axis=1))
    y_test_pred = model.predict(X_test.drop(['user_id', 'date'], axis=1))
    
    # Evaluate performance
    train_metrics = evaluate_predictions(y_train, y_train_pred, "Training Performance")
    val_metrics = evaluate_predictions(y_val, y_val_pred, "Validation Performance")
    test_metrics = evaluate_predictions(y_test, y_test_pred, "Test Performance")
    
    # Create visualizations
    plot_results(y_train, y_train_pred, dates[train_mask], "Training Results")
    plot_results(y_val, y_val_pred, dates[val_mask], "Validation Results")
    plot_results(y_test, y_test_pred, dates[test_mask], "Test Results")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X_train.drop(['user_id', 'date'], axis=1).columns,
        'importance': model.feature_importances_
    })
    feature_importance = feature_importance.sort_values('importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=feature_importance, x='importance', y='feature')
    plt.title("Feature Importance")
    plt.tight_layout()
    plt.savefig('data_analysis/plots/modeling/feature_importance_simple.png')
    plt.close()
    
    print("\nFeature Importance:")
    print(feature_importance)
    
    # Save model
    model.save_model('models/xgboost_simple.model')
    print("\nModel saved to 'models/xgboost_simple.model'")

if __name__ == "__main__":
    main()
