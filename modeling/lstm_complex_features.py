import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Attention
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
import optuna
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns

def create_sequences(X, y, sequence_length):
    """Create sequences for LSTM input"""
    Xs, ys = [], []
    for i in range(len(X) - sequence_length):
        Xs.append(X[i:(i + sequence_length)])
        ys.append(y[i + sequence_length])
    return np.array(Xs), np.array(ys)

def smape(y_true, y_pred):
    """Symmetric Mean Absolute Percentage Error"""
    return 100 * np.mean(np.abs(y_pred - y_true) / ((np.abs(y_true) + np.abs(y_pred)) / 2))

def wmape(y_true, y_pred):
    """Weighted Mean Absolute Percentage Error"""
    return np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true)) * 100

def evaluate_predictions(y_true, y_pred, title="Model Performance"):
    """Calculate and display multiple performance metrics"""
    metrics = {
        'R²': tf.keras.metrics.CoeffientOfDetermination()(y_true, y_pred).numpy(),
        'MSE': tf.keras.metrics.mean_squared_error(y_true, y_pred).numpy(),
        'MAE': tf.keras.metrics.mean_absolute_error(y_true, y_pred).numpy(),
        'RMSE': np.sqrt(tf.keras.metrics.mean_squared_error(y_true, y_pred).numpy()),
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

def create_model(trial, input_shape):
    """Create LSTM model with Optuna-suggested hyperparameters"""
    model = Sequential()
    
    # LSTM layers
    n_layers = trial.suggest_int('n_layers', 1, 3)
    for i in range(n_layers):
        units = trial.suggest_int(f'lstm_units_l{i}', 32, 256)
        return_sequences = i < n_layers - 1
        if i == 0:
            model.add(LSTM(units, return_sequences=return_sequences, input_shape=input_shape))
        else:
            model.add(LSTM(units, return_sequences=return_sequences))
        
        # Add dropout after each LSTM layer
        dropout_rate = trial.suggest_float(f'dropout_l{i}', 0.1, 0.5)
        model.add(Dropout(dropout_rate))
    
    # Dense layers
    n_dense = trial.suggest_int('n_dense', 1, 2)
    for i in range(n_dense):
        units = trial.suggest_int(f'dense_units_l{i}', 16, 128)
        model.add(Dense(units, activation='relu'))
        dropout_rate = trial.suggest_float(f'dense_dropout_l{i}', 0.1, 0.5)
        model.add(Dropout(dropout_rate))
    
    # Output layer
    model.add(Dense(1))
    
    # Compile model
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    model.compile(optimizer=Adam(learning_rate=learning_rate),
                 loss='mse',
                 metrics=['mae'])
    
    return model

def objective(trial, X_train_seq, y_train_seq, X_val_seq, y_val_seq):
    """Optuna objective function for hyperparameter optimization"""
    model = create_model(trial, X_train_seq.shape[1:])
    
    # Training parameters
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
    epochs = 50
    
    # Early stopping
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )
    
    # Train model
    history = model.fit(
        X_train_seq, y_train_seq,
        validation_data=(X_val_seq, y_val_seq),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping],
        verbose=0
    )
    
    return history.history['val_loss'][-1]

def main():
    # Load data
    print("Loading data...")
    features = pd.read_csv('data/mood_prediction_features.csv')
    features['time'] = pd.to_datetime(features['time'])
    
    print("\nInitial shape:", features.shape)
    
    # Check mood distribution
    print("\nMood recording statistics:")
    print(features['mood'].describe())
    
    # Split data
    train_end = pd.to_datetime('2014-05-08')
    val_end = pd.to_datetime('2014-05-23')
    
    # Add buffer days between splits
    buffer_days = 7  # One week buffer
    
    # Split features with buffer
    train_mask = features['time'] <= (train_end - pd.Timedelta(days=buffer_days))
    val_mask = (features['time'] > (train_end + pd.Timedelta(days=buffer_days))) & \
               (features['time'] <= (val_end - pd.Timedelta(days=buffer_days)))
    test_mask = features['time'] > (val_end + pd.Timedelta(days=buffer_days))
    
    # Calculate user statistics only from training data
    user_stats = features[train_mask].groupby('id')['mood'].agg(['mean', 'std']).reset_index()
    user_stats.columns = ['id', 'user_avg_mood', 'user_std_mood']
    features = features.merge(user_stats, on='id', how='left')
    
    # Handle missing values
    features = features.fillna(features[train_mask][['user_avg_mood', 'user_std_mood']].mean())
    
    # Handle categorical variables
    features = pd.get_dummies(features, columns=['time_of_day'])
    
    # Get feature columns
    feature_cols = features.columns.difference(['id', 'time', 'mood'])
    
    # Scale features
    scaler = StandardScaler()
    features_scaled = features.copy()
    features_scaled[feature_cols] = scaler.fit_transform(features[feature_cols])
    
    # Create sequences
    sequence_length = 24  # 24-hour window
    
    # Prepare sequences for each split
    X_train = features_scaled[train_mask][feature_cols].values
    y_train = features_scaled[train_mask]['mood'].values
    X_train_seq, y_train_seq = create_sequences(X_train, y_train, sequence_length)
    
    X_val = features_scaled[val_mask][feature_cols].values
    y_val = features_scaled[val_mask]['mood'].values
    X_val_seq, y_val_seq = create_sequences(X_val, y_val, sequence_length)
    
    X_test = features_scaled[test_mask][feature_cols].values
    y_test = features_scaled[test_mask]['mood'].values
    X_test_seq, y_test_seq = create_sequences(X_test, y_test, sequence_length)
    
    # Perform hyperparameter optimization
    print("\nPerforming Bayesian Optimization for hyperparameter tuning...")
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, X_train_seq, y_train_seq, X_val_seq, y_val_seq),
                  n_trials=50)
    
    # Print best parameters
    print("\nBest parameters found:")
    for key, value in study.best_params.items():
        print(f"{key}: {value}")
    print(f"Best validation loss: {study.best_value:.4f}")
    
    # Train final model with best parameters
    best_model = create_model(study.best_trial, X_train_seq.shape[1:])
    
    # Train with early stopping
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )
    
    history = best_model.fit(
        X_train_seq, y_train_seq,
        validation_data=(X_val_seq, y_val_seq),
        epochs=50,
        batch_size=study.best_params['batch_size'],
        callbacks=[early_stopping],
        verbose=1
    )
    
    # Make predictions
    y_train_pred = best_model.predict(X_train_seq)
    y_val_pred = best_model.predict(X_val_seq)
    y_test_pred = best_model.predict(X_test_seq)
    
    # Evaluate predictions
    train_metrics = evaluate_predictions(y_train_seq, y_train_pred, "Training Performance")
    val_metrics = evaluate_predictions(y_val_seq, y_val_pred, "Validation Performance")
    test_metrics = evaluate_predictions(y_test_seq, y_test_pred, "Test Performance")
    
    # Plot results
    train_dates = features[train_mask]['time'].iloc[sequence_length:].reset_index(drop=True)
    val_dates = features[val_mask]['time'].iloc[sequence_length:].reset_index(drop=True)
    test_dates = features[test_mask]['time'].iloc[sequence_length:].reset_index(drop=True)
    
    plot_results(y_train_seq, y_train_pred, train_dates, "Training Results")
    plot_results(y_val_seq, y_val_pred, val_dates, "Validation Results")
    plot_results(y_test_seq, y_test_pred, test_dates, "Test Results")
    
    # Plot training history
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('data_analysis/plots/modeling/lstm_training_history.png')
    plt.close()
    
    # Save model
    best_model.save('models/lstm_complex.h5')
    print("\nModel saved to 'models/lstm_complex.h5'")

if __name__ == "__main__":
    main()
