import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

class MoodEstimator:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        
    def prepare_data(self, features, mood_scores):
        """
        Prepare and scale the input data.
        
        Args:
            features (pd.DataFrame): Input features like sleep, activity, etc.
            mood_scores (np.array): Target mood scores
        
        Returns:
            tuple: Scaled training and testing sets
        """
        X_train, X_test, y_train, y_test = train_test_split(
            features, mood_scores, test_size=0.2, random_state=42
        )
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def train(self, X_train, y_train):
        """Train the mood estimation model."""
        self.model.fit(X_train, y_train)
    
    def predict(self, X):
        """Make mood predictions for given features."""
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def plot_predictions(self, y_true, y_pred, title="Mood Predictions"):
        """Plot actual vs predicted mood scores."""
        plt.figure(figsize=(10, 6))
        plt.scatter(y_true, y_pred, alpha=0.5)
        plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--')
        plt.xlabel("Actual Mood Scores")
        plt.ylabel("Predicted Mood Scores")
        plt.title(title)
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    # Example usage
    # Generate synthetic data for demonstration
    np.random.seed(42)
    n_samples = 100
    
    # Example features: sleep_hours, physical_activity, social_interactions
    features = pd.DataFrame({
        'sleep_hours': np.random.normal(7, 1, n_samples),
        'physical_activity': np.random.normal(60, 15, n_samples),
        'social_interactions': np.random.normal(5, 2, n_samples)
    })
    
    # Generate synthetic mood scores (0-10 scale)
    mood_scores = (0.3 * features['sleep_hours'] + 
                  0.4 * features['physical_activity']/60 +
                  0.3 * features['social_interactions'])
    mood_scores = (mood_scores - mood_scores.min()) / (mood_scores.max() - mood_scores.min()) * 10
    
    # Initialize and train model
    estimator = MoodEstimator()
    X_train, X_test, y_train, y_test = estimator.prepare_data(features, mood_scores)
    estimator.train(X_train, y_train)
    
    # Make predictions
    y_pred = estimator.predict(X_test)
    
    # Plot results
    estimator.plot_predictions(y_test, y_pred, "Mood Estimation Results")
