import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, accuracy_score, precision_score, recall_score, f1_score, RocCurveDisplay

def smape(y_true, y_pred):
    """Symmetric Mean Absolute Percentage Error"""
    return 100 * np.mean(np.abs(y_pred - y_true) / ((np.abs(y_true) + np.abs(y_pred)) / 2))


def wmape(y_true, y_pred):
    """Weighted Mean Absolute Percentage Error"""
    return np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true)) * 100


def evaluate_predictions(y_true, y_pred, title="Model Performance"):
    """
    Calculate and display appropriate performance metrics.
    Automatically detects regression vs classification based on y_true values.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Determine if classification
    unique_vals = np.unique(y_true)
    is_binary = set(unique_vals).issubset({0, 1})

    print(f"\n{title}:")
    results = {}

    if is_binary:
        results['Accuracy']  = accuracy_score(y_true, y_pred)
        results['Precision'] = precision_score(y_true, y_pred, zero_division=0)
        results['Recall']    = recall_score(y_true, y_pred, zero_division=0)
        results['F1']        = f1_score(y_true, y_pred, zero_division=0)
    else:
        results['R2']    = r2_score(y_true, y_pred)
        results['MAE']   = mean_absolute_error(y_true, y_pred)
        results['MSE']   = mean_squared_error(y_true, y_pred)
        results['RMSE']  = np.sqrt(mean_squared_error(y_true, y_pred))
        results['SMAPE'] = smape(y_true, y_pred)
        results['WMAPE'] = wmape(y_true, y_pred)

    for metric, value in results.items():
        print(f"{metric}: {value:.4f}")
    return results


def plot_results_scatter(y_true, y_pred, X, title="Scatter Predictions vs Actual"):
    """Create scatter plot of predictions vs actual values."""
    plt.figure(figsize=(16, 8))
    plt.scatter(y_true, y_pred, alpha=0.5)
    mn, mx = min(np.min(y_true), np.min(y_pred)), max(np.max(y_true), np.max(y_pred))
    plt.plot([mn, mx], [mn, mx], 'r--')
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(title)
    filename = f"data_analysis/plots/modeling/{title.lower().replace(' ', '_')}.png"
    plt.savefig(filename)
    plt.close()
    print("Scatter plot saved to:", filename)


def plot_results_ts(y_true, y_pred, X, title="Time Series Predictions vs Actual"):
    """Create time-series plot of mean predictions vs actuals by date."""
    df_plot = pd.DataFrame({
        'date': pd.to_datetime(X['date']),
        'y_true': y_true,
        'y_pred': y_pred
    })
    grouped = df_plot.groupby('date').mean().reset_index()

    plt.figure(figsize=(14, 6))
    plt.plot(grouped['date'], grouped['y_true'], label='Mean Actual', marker='o')
    plt.plot(grouped['date'], grouped['y_pred'], label='Mean Predicted', marker='x')
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.title(title)
    plt.legend()
    filename = f"data_analysis/plots/modeling/{title.lower().replace(' ', '_')}.png"
    plt.savefig(filename)
    plt.close()
    print("Time series plot saved to:", filename)


def plot_roc_curve(model, X_test, y_test, model_name="Model"):
    """Plot and save ROC curve for a classifier."""
    if hasattr(model, "predict_proba"):
        disp = RocCurveDisplay.from_estimator(
            model, X_test, y_test, name=f"ROC Curve ({model_name})"
        )
        filename = f"data_analysis/plots/modeling/roc_curve_{model_name.lower().replace(' ', '_')}.png"
        disp.figure_.savefig(filename)
        plt.close(disp.figure_)
        print("ROC curve saved to:", filename)
    else:
        print(f"ROC curve not available for model type {type(model).__name__}.")


def evaluate_model_pipeline(model, X_train, y_train, X_val, y_val, X_test, y_test,model_name="Model"):
    """Runs full evaluation pipeline: metrics, plots, ROC, and feature importance."""
    print(f"\n=== Evaluating {model_name} ===")
    # Prepare feature sets
    X_train_feat = X_train.drop(['id', 'date'], axis=1)
    X_val_feat   = X_val.drop(['id', 'date'], axis=1)
    X_test_feat  = X_test.drop(['id', 'date'], axis=1)

    # Predictions
    y_train_pred = model.predict(X_train_feat)
    y_val_pred   = model.predict(X_val_feat)
    y_test_pred  = model.predict(X_test_feat)

    # Metrics
    evaluate_predictions(y_train, y_train_pred, f"{model_name} Training Performance")
    evaluate_predictions(y_val,   y_val_pred,   f"{model_name} Validation Performance")
    evaluate_predictions(y_test,  y_test_pred,  f"{model_name} Test Performance")

    # Scatter & TS plots
    plot_results_scatter(y_train, y_train_pred, X_train, f"{model_name} Training Scatter")
    plot_results_scatter(y_val,   y_val_pred,   X_val,   f"{model_name} Validation Scatter")
    plot_results_scatter(y_test,  y_test_pred,  X_test,  f"{model_name} Test Scatter")

    plot_results_ts(y_train, y_train_pred, X_train, f"{model_name} Training TS")
    plot_results_ts(y_val,   y_val_pred,   X_val,   f"{model_name} Validation TS")
    plot_results_ts(y_test,  y_test_pred,  X_test,  f"{model_name} Test TS")

    # ROC for classification
    plot_roc_curve(model, X_test_feat, y_test, model_name)

    # Feature importance
    print(f"\n--- Feature Importance ({model_name}) ---")
    if hasattr(model, 'feature_importances_'):
        fi = pd.DataFrame({
            'feature': X_train_feat.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        plt.figure(figsize=(10, 6))
        sns.barplot(data=fi, x='importance', y='feature')
        plt.title(f"Feature Importance ({model_name})")
        plt.tight_layout()
        fname = f"data_analysis/plots/modeling/feature_importance_{model_name.lower().replace(' ', '_')}.png"
        plt.savefig(fname)
        plt.close()
        print("Feature importance saved to:", fname)
        print(fi)
    else:
        print(f"Model type {type(model).__name__} has no feature_importances_.")