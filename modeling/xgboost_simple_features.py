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
    """Prepare data with rolling window features to predict next day's mean mood"""
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
        
        # Calculate daily mean moods
        daily_moods = user_data.groupby('date')['mood'].mean().reset_index()
        daily_moods = daily_moods.sort_values('date')
        
        # For each day's last recording, predict next day's mean mood
        for i in range(len(daily_moods) - 1):  # -1 to ensure we have next day's data
            current_date = daily_moods.iloc[i]['date']
            next_date = daily_moods.iloc[i+1]['date']
            
            # Skip if gap is more than 1 day
            if (next_date - current_date).days > 1:
                continue
            
            # Get the last recording of the current day
            current_day_data = user_data[user_data['date'] == current_date]
            if len(current_day_data) == 0:
                continue
                
            current_row = current_day_data.iloc[-1]  # Use last recording of the day
            next_day_mean_mood = daily_moods.iloc[i+1]['mood']  # Use mean mood of next day


            # Extract time features
            current_time = current_row['time']
            feature_dict = {
                'id': user,
                'time': current_time,
                # Time features
                'hour': current_time.hour,
                'day_of_week': current_time.dayofweek,
                'month': current_time.month,
                'time_of_day': current_time.hour // 6,  # 0-3 representing 6-hour blocks
                
                # Mood features
                'prev_mood': current_row['prev_mood'] if not pd.isna(current_row['prev_mood']) else 0,
                'mood_std_24h': current_row['mood_std_24h'] if not pd.isna(current_row['mood_std_24h']) else 0,
                'mood_trend': current_row['mood_trend'] if not pd.isna(current_row['mood_trend']) else 0,
                'mood_vs_baseline': current_row['mood_vs_baseline'] if not pd.isna(current_row['mood_vs_baseline']) else 0,
                
                # Activity features
                'activity_mean_24h': current_row['activity_mean_24h'] if not pd.isna(current_row['activity_mean_24h']) else 0,
                'activity_std_24h': current_row['activity_std_24h'] if not pd.isna(current_row['activity_std_24h']) else 0,
                'screen_time_24h': current_row['screen_time_24h'] if not pd.isna(current_row['screen_time_24h']) else 0,
                
                # Communication features
                'communication_24h': current_row['communication_24h'] if not pd.isna(current_row['communication_24h']) else 0,
                
                # Emotion features
                'emotion_intensity_24h': current_row['emotion_intensity_24h'] if not pd.isna(current_row['emotion_intensity_24h']) else 0,
                
                # Data quality
                'measurements_24h': current_row['measurements_24h'] if not pd.isna(current_row['measurements_24h']) else 0,
                
                # User baseline
                'user_avg_mood': current_row['user_avg_mood'] if not pd.isna(current_row['user_avg_mood']) else 0
            }
            
            features.append(feature_dict)
            targets.append(next_day_mean_mood)  # Use next day's mean mood as target
            dates.append(current_row['time'])
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



