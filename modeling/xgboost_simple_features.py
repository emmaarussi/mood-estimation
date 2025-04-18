import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from modeling.metrics import evaluate_predictions
from feature_engineering.simple_feature_engineering import prepare_rolling_window_data

def main(input_file=None):

    # Handle input file path
    #if input_file is None:
    #    if len(sys.argv) > 1:
    #        input_file = sys.argv[1]
    #    else:
    #        input_file = 'data/mood_prediction_simple_features.csv'
    
    input_file = 'data/mood_prediction_simple_features.csv'
    
    #if not os.path.exists(input_file):
    #    print(f"Error: Input file not found: {input_file}")
    #    sys.exit(1)
    
    # Create necessary directories
    #os.makedirs('data_analysis/plots/modeling', exist_ok=True)
    #os.makedirs('models', exist_ok=True)
    
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
    X, y, dates, user_ids, encoder = prepare_rolling_window_data(df, window_size=5)
    dates = pd.to_datetime(dates)
    
    full_data = pd.concat([X, pd.Series(y, name='target_outcome')],axis=1)
    
    train_list = []
    val_list = []
    test_list = []
    
    # Split data per user
    for user, group in full_data.groupby('user_id'):
        sorted_group = group.sort_values('date')
        n = len(sorted_group)
        
        if n < 10:
            continue
        
        train_end = int(n * 0.8)
        val_end = int(n * 0.9)

        train = sorted_group.iloc[:train_end]
        val = sorted_group.iloc[train_end:val_end]
        test = sorted_group.iloc[val_end:]

        train_list.append(train)
        val_list.append(val)
        test_list.append(test)

    train_data = pd.concat(train_list).reset_index(drop=True)
    val_data = pd.concat(val_list).reset_index(drop=True)
    test_data = pd.concat(test_list).reset_index(drop=True)
    
    X_train, y_train = train_data.drop('target_outcome', axis=1), train_data['target_outcome']
    X_val, y_val = val_data.drop('target_outcome', axis=1), val_data['target_outcome']
    X_test, y_test = test_data.drop('target_outcome', axis=1), test_data['target_outcome']
    
    # Train model
    print("Training model...")
    # Baseline Model
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
    
    # Save model
    model.save_model('models/baseline_xgboost_simple.model')
    print("\nModel saved to 'models/baseline_xgboost_simple.model'")
    
    predict_evaluate_models(model, X_train, y_train, X_val, y_val, X_test, y_test, model_name="Baseline Model")
    # Tuning Model
    print("\n--- Starting Hyperparameter Tuning (GridSearchCV) ---")
    param_grid = {
        'learning_rate': [0.05, 0.1],       # Common effective rates
        'n_estimators':   [100, 300, 500],  # Keep a range, interacts with learning_rate
        'max_depth':      [3, 5, 7],        # Key depths to explore
        'min_child_weight':[1, 3],          # Default and a slightly higher value
        'subsample':      [0.8, 1.0],       # Common subsampling values
        'colsample_bytree':[0.8, 1.0],       # Common feature sampling values
        # 'gamma':          [0.0, 0.2],     # Can often be omitted for speed initially
        # 'reg_alpha':      [0.0, 0.1],     # Can often be omitted for speed initially
        'reg_lambda':     [1.0],            # Often start with default L2
    }

    xgb_base = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)

    # Define the GroupKFold cross-validator
    n_splits_cv = 3  # Or more, depending on data size
    time_split = TimeSeriesSplit(n_splits=n_splits_cv)

    # Sort X_train and y_train by date to ensure correct temporal order
    train_sorted = X_train.sort_values('date').reset_index(drop=True)
    y_train_sorted = y_train.loc[train_sorted.index]

    grid_search = GridSearchCV(
        estimator=xgb_base,
        param_grid=param_grid,
        scoring='neg_root_mean_squared_error',
        cv=time_split,  # Use TimeSeriesSplit here
        verbose=0,
        n_jobs=-1
    )

    print(f"Performing {n_splits_cv}-Fold TimeSeriesSplit Cross-Validation for Tuning...")
    grid_search.fit(
        train_sorted.drop(['user_id', 'date'], axis=1),  # Features only
        y_train_sorted                                   # Target
    )

    print(f"\nBest parameters found: {grid_search.best_params_}")
    print(f"Best CV RMSE score: {-grid_search.best_score_:.4f}")

    # --- 3. Evaluate Tuned Model ---
    print("\n--- Evaluating Tuned Model ---")
    best_model = grid_search.best_estimator_ # This is already refitted on the whole train set
    
    predict_evaluate_models(best_model, X_train, y_train, X_val, y_val, X_test, y_test, model_name="Tuned Model")
    
    # Save tuned model
    best_model.save_model('models/xgboost_simple_tuned.model')
    print("\nTuned model saved to 'models/xgboost_simple_tuned.model'")
    

def predict_evaluate_models(model, X_train, y_train, X_val, y_val, X_test, y_test, model_name="Model"):
    """Makes predictions, evaluates, plots results, and shows feature importance for a given model."""

    print(f"\n--- Evaluating {model_name} ---")

    # Make predictions
    # Prepare feature sets by dropping non-feature columns
    X_train_features = X_train.drop(['user_id', 'date'], axis=1)
    X_val_features = X_val.drop(['user_id', 'date'], axis=1)
    X_test_features = X_test.drop(['user_id', 'date'], axis=1)

    y_train_pred = model.predict(X_train_features)
    y_val_pred = model.predict(X_val_features)
    y_test_pred = model.predict(X_test_features)

    # Evaluate performance using the passed-in true values and the predictions
    train_metrics = evaluate_predictions(y_train, y_train_pred, f"{model_name} Training Performance")
    val_metrics = evaluate_predictions(y_val, y_val_pred, f"{model_name} Validation Performance")
    test_metrics = evaluate_predictions(y_test, y_test_pred, f"{model_name} Test Performance")

    # Create visualizations using data and predictions
    plot_results_scatter(y_train, y_train_pred, X_train, f"{model_name} Training Results scatter")
    plot_results_scatter(y_val, y_val_pred, X_val, f"{model_name} Validation Results scatter")
    plot_results_scatter(y_test, y_test_pred, X_test, f"{model_name} Test Results scatter")

    plot_results_ts(y_train, y_train_pred, X_train, f"{model_name} Training Results TS")
    plot_results_ts(y_val, y_val_pred, X_val, f"{model_name} Validation Results TS")
    plot_results_ts(y_test, y_test_pred, X_test, f"{model_name} Test Results TS")

    # Feature importance using the passed-in model and training features
    print(f"\n--- Feature Importance ({model_name}) ---")
    
    # Check if the model has feature_importances_ attribute (like XGBoost)
    if hasattr(model, 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'feature': X_train_features.columns, # Use columns from the feature set
            'importance': model.feature_importances_
        })
        feature_importance = feature_importance.sort_values('importance', ascending=False)

        plt.figure(figsize=(10, 6))
        sns.barplot(data=feature_importance, x='importance', y='feature')
        plt.title(f"Feature Importance ({model_name})")
        plt.tight_layout()
        # Make filenames unique for different models
        filename_fi = f'data_analysis/plots/modeling/feature_importance_{model_name.lower().replace(" ", "_")}.png'
        plt.savefig(filename_fi)
        plt.close()

        print(feature_importance)
    else:
        print(f"Model type {type(model).__name__} does not support feature importances.")

    # Optionally return metrics if needed elsewhere
    return train_metrics, val_metrics, test_metrics
    

def plot_results_scatter(y_true, y_pred, dates, title="Scatter Predictions vs Actual"):
    """Create visualization of predictions"""
    plt.figure(figsize=(16, 8))
    
    # Scatter plot
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    plt.xlabel("Actual Mood")
    plt.ylabel("Predicted Mood")
    plt.title(f"{title} - Scatter Plot")
    #plt.show()
    plt.close()
    filename = f'data_analysis/plots/modeling/{title.lower().replace(" ", "_")}.png'
    plt.savefig(filename)
    
    print("Figure saved to:", filename)
    
    return

def plot_results_ts(y_true, y_pred, X, title="Time Series Predictions vs Actual"):
    # Time series plot
    df_plot = pd.DataFrame({
        'date': pd.to_datetime(X['date']),
        'y_true': y_true,
        'y_pred': y_pred
    })
    grouped = df_plot.groupby('date').mean().reset_index()

    plt.figure(figsize=(14, 6))
    plt.plot(grouped['date'], grouped['y_true'], label='Mean Actual Mood', marker='o')
    plt.plot(grouped['date'], grouped['y_pred'], label='Mean Predicted Mood', marker='x')
    plt.xlabel("Date")
    plt.ylabel("Mood")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    filename = f'data_analysis/plots/modeling/{title.lower().replace(" ", "_")}.png'
    plt.savefig(filename)
    #plt.show()
    plt.close()
    plt.close()
    
    print("Figure saved to:", filename)


if __name__ == "__main__":
    main()
