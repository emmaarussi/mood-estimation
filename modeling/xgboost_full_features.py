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
            
            # Create feature dictionary using pre-computed features
            feature_dict = {
                'user_id': user,
                'date': current_row['time'],
                # Temporal features
                'hour': current_row['hour'],
                'day_of_week': current_row['day_of_week'],
                'month': current_row['month'],
                'time_of_day': current_row['time_of_day'],
                'hour_sin': current_row['hour_sin'],
                'hour_cos': current_row['hour_cos'],
                'day_sin': current_row['day_sin'],
                'day_cos': current_row['day_cos'],
                
                # Mood features - 24h, 72h, 168h windows
                'mood_lag_24h': current_row['mood_lag_24h'],
                'mood_lag_72h': current_row['mood_lag_72h'],
                'mood_lag_168h': current_row['mood_lag_168h'],
                'mood_rolling_mean_24h': current_row['mood_rolling_mean_24h'],
                'mood_rolling_mean_72h': current_row['mood_rolling_mean_72h'],
                'mood_rolling_mean_168h': current_row['mood_rolling_mean_168h'],
                'mood_rolling_std_24h': current_row['mood_rolling_std_24h'],
                'mood_rolling_std_72h': current_row['mood_rolling_std_72h'],
                'mood_rolling_std_168h': current_row['mood_rolling_std_168h'],
                
                # Activity features - 24h, 72h, 168h windows
                'activity_intensity_24h': current_row['activity_intensity_24h'],
                'activity_intensity_72h': current_row['activity_intensity_72h'],
                'activity_intensity_168h': current_row['activity_intensity_168h'],
                'activity_variability_24h': current_row['activity_variability_24h'],
                'activity_variability_72h': current_row['activity_variability_72h'],
                'activity_variability_168h': current_row['activity_variability_168h'],
                'screen_time_24h': current_row['screen_time_24h'],
                'screen_time_72h': current_row['screen_time_72h'],
                'screen_time_168h': current_row['screen_time_168h'],
                
                # Communication features - 24h, 72h, 168h windows
                'call_frequency_24h': current_row['call_frequency_24h'],
                'call_frequency_72h': current_row['call_frequency_72h'],
                'call_frequency_168h': current_row['call_frequency_168h'],
                'sms_frequency_24h': current_row['sms_frequency_24h'],
                'sms_frequency_72h': current_row['sms_frequency_72h'],
                'sms_frequency_168h': current_row['sms_frequency_168h'],
                'total_communication_24h': current_row['total_communication_24h'],
                'total_communication_72h': current_row['total_communication_72h'],
                'total_communication_168h': current_row['total_communication_168h'],
                
                # App usage features - 24h, 72h, 168h windows
                'productive_ratio_24h': current_row['productive_ratio_24h'],
                'productive_ratio_72h': current_row['productive_ratio_72h'],
                'productive_ratio_168h': current_row['productive_ratio_168h'],
                'app_diversity_24h': current_row['app_diversity_24h'],
                'app_diversity_72h': current_row['app_diversity_72h'],
                'app_diversity_168h': current_row['app_diversity_168h'],
                
                # Circumplex features - 24h, 72h, 168h windows
                'arousal_std_24h': current_row['arousal_std_24h'],
                'arousal_std_72h': current_row['arousal_std_72h'],
                'arousal_std_168h': current_row['arousal_std_168h'],
                'valence_std_24h': current_row['valence_std_24h'],
                'valence_std_72h': current_row['valence_std_72h'],
                'valence_std_168h': current_row['valence_std_168h'],
                'affect_intensity_24h': current_row['affect_intensity_24h'],
                'affect_intensity_72h': current_row['affect_intensity_72h'],
                'affect_intensity_168h': current_row['affect_intensity_168h'],
                'affect_angle_24h': current_row['affect_angle_24h'],
                'affect_angle_72h': current_row['affect_angle_72h'],
                'affect_angle_168h': current_row['affect_angle_168h'],
                
                # Gap features
                'avg_gap_hours': current_row['avg_gap_hours'],
                'gap_std_hours': current_row['gap_std_hours'],
                'max_gap_hours': current_row['max_gap_hours'],
                'gap_category_normal': 1 if current_row['gap_category'] == 'normal' else 0,
                'gap_category_12_24h': 1 if current_row['gap_category'] == '12-24h' else 0,
                'gap_category_24_48h': 1 if current_row['gap_category'] == '24-48h' else 0,
                'gap_category_48h_plus': 1 if current_row['gap_category'] == '48h+' else 0,
                
                # Other features
                'time_since_last_mood': current_row['time_since_last_mood']
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
            input_file = 'data/mood_prediction_features.csv'
    
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
    
    X_train = X[train_mask].drop(['user_id', 'date'], axis=1)
    y_train = y[train_mask]
    X_val = X[val_mask].drop(['user_id', 'date'], axis=1)
    y_val = y[val_mask]
    X_test = X[test_mask].drop(['user_id', 'date'], axis=1)
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
    plt.savefig('data_analysis/plots/modeling/feature_importance_full.png')
    plt.close()
    
    print("\nTop 20 Features:")
    print(feature_importance.head(20))
    
    # Save model
    model.save_model('models/xgboost_full.model')
    print("\nModel saved to 'models/xgboost_full.model'")

if __name__ == "__main__":
    main()
