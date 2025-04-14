import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load predictions
df = pd.read_csv('../data/mood_prediction_simple_features.csv')
df['time'] = pd.to_datetime(df['time'])

# Create a simple example
actual = np.array([7.0, 6.5, 8.0, 7.2, 6.8])
pred_mean = np.full_like(actual, actual.mean())  # Always predict mean
pred_good = actual * 0.9 + actual.mean() * 0.1  # Good predictions
pred_bad = actual * 0.1 + actual.mean() * 0.9   # Bad predictions (closer to mean)

def calc_metrics(y_true, y_pred):
    wmape = np.mean(np.abs(y_true - y_pred) / np.abs(y_true)) * 100
    r2 = 1 - np.sum((y_true - y_pred)**2) / np.sum((y_true - y_true.mean())**2)
    return wmape, r2

# Calculate metrics
wmape_mean, r2_mean = calc_metrics(actual, pred_mean)
wmape_good, r2_good = calc_metrics(actual, pred_good)
wmape_bad, r2_bad = calc_metrics(actual, pred_bad)

# Create visualization
plt.figure(figsize=(15, 5))

# Plot 1: Example predictions
plt.subplot(1, 2, 1)
x = range(len(actual))
plt.plot(x, actual, 'o-', label='Actual', color='black')
plt.plot(x, pred_mean, '--', label=f'Mean (WMAPE={wmape_mean:.1f}%, R²={r2_mean:.2f})', color='red')
plt.plot(x, pred_good, '--', label=f'Good (WMAPE={wmape_good:.1f}%, R²={r2_good:.2f})', color='green')
plt.plot(x, pred_bad, '--', label=f'Bad (WMAPE={wmape_bad:.1f}%, R²={r2_bad:.2f})', color='orange')
plt.title('Example Predictions vs Actual')
plt.ylabel('Mood')
plt.legend()

# Plot 2: Error Distribution
plt.subplot(1, 2, 2)
errors = {
    'Mean Prediction': pred_mean - actual,
    'Good Prediction': pred_good - actual,
    'Bad Prediction': pred_bad - actual
}
plt.boxplot(errors.values(), labels=errors.keys())
plt.title('Error Distribution')
plt.ylabel('Error (Predicted - Actual)')

plt.tight_layout()
plt.savefig('reports/error_analysis.png')
print("Analysis saved to reports/error_analysis.png")

# Print detailed metrics
print("\nDetailed Metrics:")
print(f"Mean Prediction: WMAPE = {wmape_mean:.1f}%, R² = {r2_mean:.2f}")
print(f"Good Prediction: WMAPE = {wmape_good:.1f}%, R² = {r2_good:.2f}")
print(f"Bad Prediction: WMAPE = {wmape_bad:.1f}%, R² = {r2_bad:.2f}")
