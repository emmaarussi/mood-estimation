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
    features = pd.read_csv('data/mood_prediction_features.csv')
    features['time'] = pd.to_datetime(features['time'])
    
    print("\nInitial shape:", features.shape)
    
    # Check mood distribution
    print("\nMood recording statistics:")
    print(features['mood'].describe())
    
    # Check for NaN values
    nan_percentages = features.isna().mean() * 100
    print("\nFeature NaN percentages:")
    print(nan_percentages[nan_percentages > 0])
    
    # Handle NaN values
    features = features.fillna(0)  # Replace NaN with 0 for now
    
    # Split data
    train_end = pd.to_datetime('2014-05-08')
    val_end = pd.to_datetime('2014-05-23')
    
    # Add buffer days between splits to ensure independence
    buffer_days = 7  # One week buffer
    
    # Split features with buffer
    train_mask = features['time'] <= (train_end - pd.Timedelta(days=buffer_days))
    val_mask = (features['time'] > (train_end + pd.Timedelta(days=buffer_days))) & \
               (features['time'] <= (val_end - pd.Timedelta(days=buffer_days)))
    test_mask = features['time'] > (val_end + pd.Timedelta(days=buffer_days))
    
    # Calculate user statistics only from training data
    user_stats = features[train_mask].groupby('id')['mood'].agg(['mean', 'std']).rename(
        columns={'mean': 'user_avg_mood', 'std': 'user_std_mood'})
    
    # Merge user statistics back
    features = features.merge(user_stats, on='id', how='left')
    
    # Fill remaining NaN values with global statistics
    features = features.fillna(features[train_mask][['mood']].mean())
    
    # Handle categorical variables
    features = pd.get_dummies(features, columns=['time_of_day'])
    
    # Get feature columns (exclude id, time, and target)
    feature_cols = features.columns.difference(['id', 'time', 'mood'])
    
    # Split data
    X_train = features[train_mask][feature_cols]
    y_train = features[train_mask]['mood']
    X_val = features[val_mask][feature_cols]
    y_val = features[val_mask]['mood']
    X_test = features[test_mask][feature_cols]
    y_test = features[test_mask]['mood']
    
    # Create and run study
    print("\nPerforming Bayesian Optimization for hyperparameter tuning...")
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, X_train, X_val, y_train, y_val),
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
    best_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    
    # Make predictions
    y_train_pred = best_model.predict(X_train)
    y_val_pred = best_model.predict(X_val)
    y_test_pred = best_model.predict(X_test)
    
    # Evaluate predictions
    train_metrics = evaluate_predictions(y_train, y_train_pred, "Training Performance")
    val_metrics = evaluate_predictions(y_val, y_val_pred, "Validation Performance")
    test_metrics = evaluate_predictions(y_test, y_test_pred, "Test Performance")
    
    # Plot results
    plot_results(y_train, y_train_pred, features[train_mask]['time'], "Training Results")
    plot_results(y_val, y_val_pred, features[val_mask]['time'], "Validation Results")
    plot_results(y_test, y_test_pred, features[test_mask]['time'], "Test Results")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': best_model.feature_importances_
    })
    feature_importance = feature_importance.sort_values('importance', ascending=False)
    
    print("\nTop 20 Features:")
    print(feature_importance.head(20))
    
    # Save model
    best_model.save_model('models/xgboost_complex_bayesian.model')
    print("\nModel saved to 'models/xgboost_complex_bayesian.model'")

if __name__ == "__main__":
    main()
