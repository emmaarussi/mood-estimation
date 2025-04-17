import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import optuna
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

class MoodDataset(Dataset):
    def __init__(self, sequences, targets):
        self.sequences = torch.FloatTensor(sequences)
        self.targets = torch.FloatTensor(targets)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

def create_sequences(data, sequence_length):
    """Create sequences for LSTM input"""
    sequences = []
    targets = []
    
    # Group by user
    for user_id, user_data in data.groupby('id'):
        user_data = user_data.sort_values('time')
        
        # Get mood and circumplex values
        mood_data = user_data[user_data['variable'] == 'mood'][['time', 'value']].copy()
        arousal_data = user_data[user_data['variable'] == 'circumplex.arousal'][['time', 'value']].copy()
        valence_data = user_data[user_data['variable'] == 'circumplex.valence'][['time', 'value']].copy()
        
        # Merge all features on time
        features_data = mood_data.merge(arousal_data, on='time', how='inner', suffixes=('_mood', '_arousal'))
        features_data = features_data.merge(valence_data, on='time', how='inner', suffixes=('', '_valence'))
        features_data = features_data.sort_values('time')
        
        # Extract features
        features = features_data[['value_mood', 'value_arousal', 'value']].values  # value is valence
        
        # Create sequences for this user
        for i in range(len(features) - sequence_length):
            sequences.append(features[i:i + sequence_length])
            targets.append(features[i + sequence_length, 0])  # mood is target
            
    return np.array(sequences), np.array(targets)

def smape(y_true, y_pred):
    """Symmetric Mean Absolute Percentage Error"""
    return 100 * np.mean(np.abs(y_pred - y_true) / ((np.abs(y_true) + np.abs(y_pred)) / 2))

def wmape(y_true, y_pred):
    """Weighted Mean Absolute Percentage Error"""
    return np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true)) * 100

def evaluate_predictions(y_true, y_pred, title="Model Performance"):
    """Calculate and display multiple performance metrics"""
    mse = np.mean((y_true - y_pred) ** 2)
    mae = np.mean(np.abs(y_true - y_pred))
    
    metrics = {
        'R²': 1 - np.sum((y_true - y_pred)**2) / np.sum((y_true - np.mean(y_true))**2),
        'MSE': mse,
        'MAE': mae,
        'RMSE': np.sqrt(mse),
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

def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=50, patience=5):
    """Train the model with early stopping"""
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y.unsqueeze(1))
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y.unsqueeze(1))
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        print(f'Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}')
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered")
                break
    
    model.load_state_dict(best_model_state)
    return model, train_losses, val_losses

def objective(trial, train_loader, val_loader, input_size, device):
    """Optuna objective function for hyperparameter optimization"""
    # Hyperparameters to optimize
    hidden_size = trial.suggest_int('hidden_size', 32, 256)
    num_layers = trial.suggest_int('num_layers', 1, 3)
    dropout = trial.suggest_float('dropout', 0.1, 0.5)
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    
    # Create model
    model = LSTM(input_size, hidden_size, num_layers, dropout).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Train model
    model, _, _ = train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=20)
    
    # Evaluate on validation set
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y.unsqueeze(1))
            val_loss += loss.item()
    
    return val_loss / len(val_loader)

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    print("Loading data...")
    data = pd.read_csv('data/dataset_mood_smartphone_cleaned.csv')
    data['time'] = pd.to_datetime(data['time'])
    
    print("\nInitial shape:", data.shape)
    
    # Filter for required variables
    required_vars = ['mood', 'circumplex.arousal', 'circumplex.valence']
    data = data[data['variable'].isin(required_vars)]
    
    # Get mood statistics
    mood_stats = data[data['variable'] == 'mood']['value'].describe()
    print("\nMood recording statistics:")
    print(mood_stats)
    
    # Split data
    train_end = pd.to_datetime('2014-05-08')
    val_end = pd.to_datetime('2014-05-23')
    
    # Split by time
    train_data = data[data['time'] <= train_end]
    val_data = data[(data['time'] > train_end) & (data['time'] <= val_end)]
    test_data = data[data['time'] > val_end]
    
    # Create sequences for each split
    sequence_length = 24  # 24-hour window
    X_train_seq, y_train = create_sequences(train_data, sequence_length)
    X_val_seq, y_val = create_sequences(val_data, sequence_length)
    X_test_seq, y_test = create_sequences(test_data, sequence_length)
    
    print("\nSequence shapes:")
    print(f"Training: {X_train_seq.shape}")
    print(f"Validation: {X_val_seq.shape}")
    print(f"Test: {X_test_seq.shape}")
    
    # Create data loaders
    train_dataset = TensorDataset(torch.FloatTensor(X_train_seq), torch.FloatTensor(y_train))
    val_dataset = TensorDataset(torch.FloatTensor(X_val_seq), torch.FloatTensor(y_val))
    test_dataset = TensorDataset(torch.FloatTensor(X_test_seq), torch.FloatTensor(y_test))
    
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # Perform hyperparameter optimization
    print("\nPerforming Bayesian Optimization for hyperparameter tuning...")
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, train_loader, val_loader, 
                                         input_size=X_train_seq.shape[2], device=device),
                  n_trials=50)
    
    # Print best parameters
    print("\nBest parameters found:")
    for key, value in study.best_params.items():
        print(f"{key}: {value}")
    print(f"Best validation loss: {study.best_value:.4f}")
    
    # Train final model with best parameters
    best_model = LSTM(input_size=X_train_seq.shape[2],
                     hidden_size=study.best_params['hidden_size'],
                     num_layers=study.best_params['num_layers'],
                     dropout=study.best_params['dropout']).to(device)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(best_model.parameters(), lr=study.best_params['learning_rate'])
    
    # Train model
    best_model, train_losses, val_losses = train_model(best_model, train_loader, val_loader, 
                                                      criterion, optimizer, device)
    
    # Make predictions
    best_model.eval()
    predictions = {
        'train': [],
        'val': [],
        'test': []
    }
    targets = {
        'train': y_train,
        'val': y_val,
        'test': y_test
    }
    
    with torch.no_grad():
        for loader_name, loader in [('train', train_loader), ('val', val_loader), ('test', test_loader)]:
            for batch_X, _ in loader:
                batch_X = batch_X.to(device)
                outputs = best_model(batch_X)
                predictions[loader_name].extend(outputs.cpu().numpy())
    
    # Convert predictions to numpy arrays
    for split in predictions:
        predictions[split] = np.array(predictions[split]).reshape(-1)
    
    # Inverse transform predictions for evaluation
    scaler = StandardScaler()
    features_to_scale = ['value']
    data[features_to_scale] = scaler.fit_transform(data[features_to_scale])
    
    for split in ['train', 'val', 'test']:
        predictions[split] = scaler.inverse_transform(
            np.column_stack([predictions[split], 
                           np.zeros_like(predictions[split]), 
                           np.zeros_like(predictions[split])])
        )[:, 0]
        
        targets[split] = scaler.inverse_transform(
            np.column_stack([targets[split], 
                           np.zeros_like(targets[split]), 
                           np.zeros_like(targets[split])])
        )[:, 0]
    
    # Evaluate predictions
    train_metrics = evaluate_predictions(targets['train'], predictions['train'], "Training Performance")
    val_metrics = evaluate_predictions(targets['val'], predictions['val'], "Validation Performance")
    test_metrics = evaluate_predictions(targets['test'], predictions['test'], "Test Performance")
    
    # Plot results
    train_dates = train_data['time'].iloc[sequence_length:].reset_index(drop=True)
    val_dates = val_data['time'].iloc[sequence_length:].reset_index(drop=True)
    test_dates = test_data['time'].iloc[sequence_length:].reset_index(drop=True)
    
    plot_results(targets['train'], predictions['train'], train_dates, "Training Results")
    plot_results(targets['val'], predictions['val'], val_dates, "Validation Results")
    plot_results(targets['test'], predictions['test'], test_dates, "Test Results")
    
    # Plot training history
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('data_analysis/plots/modeling/lstm_training_history.png')
    plt.close()
    
    # Save model
    torch.save(best_model.state_dict(), 'models/lstm_raw.pt')
    print("\nModel saved to 'models/lstm_raw.pt'")

if __name__ == "__main__":
    main()
