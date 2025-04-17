import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import optuna
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
        'MSE': mean_squared_error(y_true, y_pred),
        'MAE': np.mean(np.abs(y_true - y_pred)),
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

def objective(trial, X_train, X_val, y_train, y_val):
    """Optuna objective function for hyperparameter optimization"""
    param = {
        'max_depth': trial.suggest_int('max_depth', 2, 7),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'n_estimators': trial.suggest_int('n_estimators', 50, 400),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'objective': 'reg:squarederror',
        'random_state': 42
    }
    
    model = xgb.XGBRegressor(**param)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    
    y_pred = model.predict(X_val)
    mse = mean_squared_error(y_val, y_pred)
    return mse

def main():
    # Load data
    print("Loading data...")
    features = pd.read_csv('data/mood_prediction_simple_features.csv')
    features['time'] = pd.to_datetime(features['time'])
    
    print("\nInitial shape:", features.shape)
    
    # Check mood distribution
    print("\nMood recording statistics:")
    print(features['mood'].describe())
    print("\nNaN in mood:", features['mood'].isna().sum())
    
    # Prepare features
    print("Preparing features...")
    X, y, dates, user_ids = prepare_rolling_window_data(features)
    
    # Split data
    train_end = pd.to_datetime('2014-05-08')
    val_end = pd.to_datetime('2014-05-23')
    
    dates = pd.to_datetime(dates)
    train_mask = dates <= train_end
    val_mask = (dates > train_end) & (dates <= val_end)
    test_mask = dates > val_end
    
    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    
    # Drop non-feature columns for training
    feature_cols = X_train.columns.difference(['user_id', 'date'])
    X_train_feat = X_train[feature_cols]
    X_val_feat = X_val[feature_cols]
    X_test_feat = X_test[feature_cols]
    
    # Create and run study
    print("\nPerforming Bayesian Optimization for hyperparameter tuning...")
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, X_train_feat, X_val_feat, y_train, y_val),
                  n_trials=50, show_progress_bar=True)
    
    # Print best parameters
    print("\nBest parameters found:")
    for key, value in study.best_params.items():
        print(f"{key}: {value}")
    print(f"Best MSE: {study.best_value:.4f}")
    
    # Train final model with best parameters
    best_params = study.best_params
    best_params['objective'] = 'reg:squarederror'
    best_params['random_state'] = 42
    
    best_model = xgb.XGBRegressor(**best_params)
    best_model.fit(
        X_train_feat,
        y_train,
        verbose=False
    )
    
    # Make predictions
    y_train_pred = best_model.predict(X_train_feat)
    y_val_pred = best_model.predict(X_val_feat)
    y_test_pred = best_model.predict(X_test_feat)
    
    # Evaluate predictions
    train_metrics = evaluate_predictions(y_train, y_train_pred, "Training Performance")
    val_metrics = evaluate_predictions(y_val, y_val_pred, "Validation Performance")
    test_metrics = evaluate_predictions(y_test, y_test_pred, "Test Performance")
    
    # Plot results
    plot_results(y_train, y_train_pred, dates[train_mask], "Training Results")
    plot_results(y_val, y_val_pred, dates[val_mask], "Validation Results")
    plot_results(y_test, y_test_pred, dates[test_mask], "Test Results")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': best_model.feature_importances_
    })
    feature_importance = feature_importance.sort_values('importance', ascending=False)
    
    print("\nFeature Importance:")
    print(feature_importance)
    
    # Save model
    best_model.save_model('models/xgboost_simple_bayesian.model')
    print("\nModel saved to 'models/xgboost_simple_bayesian.model'")

if __name__ == "__main__":
    main()
