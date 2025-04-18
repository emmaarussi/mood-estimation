import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

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
        'MSE': mean_absolute_error(y_true, y_pred),
        'MAE': mean_absolute_error(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'SMAPE': smape(y_true, y_pred),
        'WMAPE': wmape(y_true, y_pred)
    }
    
    print(f"\n{title}:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    return metrics