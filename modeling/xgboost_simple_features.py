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

def prepare_rolling_window_data(df, window_size=7):
    """Prepare data with rolling window features to predict next day's mean mood using past week of data"""
    # Convert time to datetime
    df['time'] = pd.to_datetime(df['time'])
    
    # Sort by user and time
    df = df.sort_values(['id', 'time'])
    
    # Only keep rows where mood is not NaN
    df = df[df['mood'].notna()]
    
    # Add date column for grouping
    df['date'] = df['time'].dt.date
    
    # Create features for each user
    features = []
    targets = []
    dates = []
    user_ids = []
    
    for user in df['id'].unique():
        user_data = df[df['id'] == user].copy()
        
        # Calculate daily statistics
        daily_stats = user_data.groupby('date').agg({
            'mood': ['mean', 'std', 'count'],
            'activity_mean_24h': 'mean',
            'screen_time_24h': 'sum',
            'communication_24h': 'sum',
            'emotion_intensity_24h': 'mean'
        }).reset_index()
        
        # Rename columns
        daily_stats.columns = ['date', 'mood_mean', 'mood_std', 'n_measurements', 
                             'activity_mean', 'screen_time', 'communication', 'emotion_intensity']
        daily_stats = daily_stats.sort_values('date')
        
        # For each day, predict next day's mean mood
        for i in range(window_size, len(daily_stats) - 1):  # Need window_size days of history
            current_date = daily_stats.iloc[i]['date']
            next_date = daily_stats.iloc[i + 1]['date']
            
            # Skip if gap is more than 1 day
            if (next_date - current_date).days > 1:
                continue
            
            # Get the window of past 7 days
            window_data = daily_stats.iloc[i-window_size+1:i+1]
            
            # Skip if we don't have enough days in the window
            if len(window_data) < window_size:
                continue
            
            # Calculate weekly features
            weekly_features = {
                'id': user,
                'time': pd.to_datetime(current_date),
                
                # Time features
                'day_of_week': pd.to_datetime(current_date).dayofweek,
                'month': pd.to_datetime(current_date).month,
                
                # Weekly mood features
                'mood_mean_7d': window_data['mood_mean'].mean(),
                'mood_std_7d': window_data['mood_mean'].std(),
                'mood_trend_7d': (window_data['mood_mean'].iloc[-1] - window_data['mood_mean'].iloc[0]) / window_size,
                'mood_yesterday': window_data['mood_mean'].iloc[-1],
                
                # Weekly activity features
                'activity_mean_7d': window_data['activity_mean'].mean(),
                'activity_std_7d': window_data['activity_mean'].std(),
                'activity_trend_7d': (window_data['activity_mean'].iloc[-1] - window_data['activity_mean'].iloc[0]) / window_size,
                
                # Weekly screen time
                'screen_time_7d': window_data['screen_time'].mean(),
                'screen_time_std_7d': window_data['screen_time'].std(),
                
                # Weekly communication
                'communication_7d': window_data['communication'].mean(),
                'communication_std_7d': window_data['communication'].std(),
                
                # Weekly emotion
                'emotion_intensity_7d': window_data['emotion_intensity'].mean(),
                'emotion_intensity_std_7d': window_data['emotion_intensity'].std(),
                
                # Data quality
                'measurements_7d': window_data['n_measurements'].sum(),
                'days_with_data': (window_data['n_measurements'] > 0).sum()
            }

            
            features.append(weekly_features)
            targets.append(daily_stats.iloc[i + 1]['mood_mean'])  # Next day's mean mood
            dates.append(pd.to_datetime(current_date))
            user_ids.append(user)
    
    X = pd.DataFrame(features)
    y = np.array(targets)
    
    return X, y, dates, user_ids

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
    
    # Prepare features with rolling window
    print("Preparing features...")
    X, y, dates, user_ids = prepare_rolling_window_data(df)
    dates = pd.to_datetime(dates)
    
    # Split data
    train_mask = dates <= train_end
    val_mask = (dates > train_end) & (dates <= val_end)
    test_mask = dates > val_end
    
    # Drop id and time columns before training
    feature_cols = [col for col in X.columns if col not in ['id', 'time']]
    
    X_train = X[train_mask][feature_cols]
    y_train = y[train_mask]
    X_val = X[val_mask][feature_cols]
    y_val = y[val_mask]
    X_test = X[test_mask][feature_cols]
    y_test = y[test_mask]
    
    # Train model
    print("Training model...")
    model = xgb.XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        objective='reg:squarederror'
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
    plot_results(y_train, y_train_pred, dates[train_mask], "Training Results")
    plot_results(y_val, y_val_pred, dates[val_mask], "Validation Results")
    plot_results(y_test, y_test_pred, dates[test_mask], "Test Results")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_
    })
    feature_importance = feature_importance.sort_values('importance', ascending=False)
    
    plt.figure(figsize=(12, 8))
    sns.barplot(data=feature_importance.head(20), x='importance', y='feature')
    plt.title("Top 20 Feature Importance")
    plt.tight_layout()
    plt.savefig('data_analysis/plots/modeling/feature_importance_simple.png')
    plt.close()
    
    print("\nTop 20 Features:")
    print(feature_importance.head(20))
    
    # Save model
    model.save_model('models/xgboost_simple.model')
    print("\nModel saved to 'models/xgboost_simple.model'")

if __name__ == "__main__":
    main()



