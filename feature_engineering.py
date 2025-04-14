import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def create_temporal_features(df):
    """Create time-based features"""
    features = df.copy()
    
    # Basic time features
    features['hour'] = features['time'].dt.hour
    features['day_of_week'] = features['time'].dt.dayofweek
    features['is_weekend'] = features['day_of_week'].isin([5, 6]).astype(int)
    features['month'] = features['time'].dt.month
    
    # Time of day categories
    features['time_of_day'] = pd.cut(
        features['hour'],
        bins=[-1, 6, 12, 18, 24],
        labels=['night', 'morning', 'afternoon', 'evening']
    )
    
    # Cyclical encoding of time features
    features['hour_sin'] = np.sin(2 * np.pi * features['hour']/24)
    features['hour_cos'] = np.cos(2 * np.pi * features['hour']/24)
    features['day_sin'] = np.sin(2 * np.pi * features['day_of_week']/7)
    features['day_cos'] = np.cos(2 * np.pi * features['day_of_week']/7)
    
    return features

def create_lag_features(df, variable, lags=[8, 16, 24, 48, 72, 168]):
    """Create lagged features for a given variable"""
    features = df.copy()
    
    # Sort by user and time
    features = features.sort_values(['id', 'time'])
    
    # Create lag features
    for lag in lags:
        lag_name = f'{variable}_lag_{lag}h'
        features[lag_name] = features.groupby('id')[variable].shift(lag)
        
        # Add rolling statistics
        if lag > 1:
            # Rolling mean
            features[f'{variable}_rolling_mean_{lag}h'] = features.groupby('id')[variable].transform(
                lambda x: x.rolling(window=lag, min_periods=1).mean()
            )
            # Rolling std
            features[f'{variable}_rolling_std_{lag}h'] = features.groupby('id')[variable].transform(
                lambda x: x.rolling(window=lag, min_periods=1).std()
            )
            # Rolling min/max
            features[f'{variable}_rolling_min_{lag}h'] = features.groupby('id')[variable].transform(
                lambda x: x.rolling(window=lag, min_periods=1).min()
            )
            features[f'{variable}_rolling_max_{lag}h'] = features.groupby('id')[variable].transform(
                lambda x: x.rolling(window=lag, min_periods=1).max()
            )
    
    return features

def create_activity_features(df, window_sizes=[24, 48, 72, 168]):
    """Create activity-based features"""
    features = df.copy()
    
    # Calculate activity levels for different time windows
    for window in window_sizes:
        # Activity intensity (if column exists)
        if 'activity' in features.columns:
            features[f'activity_intensity_{window}h'] = features.groupby('id')['activity'].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean()
            )
            
            # Activity variability
            features[f'activity_variability_{window}h'] = features.groupby('id')['activity'].transform(
                lambda x: x.rolling(window=window, min_periods=1).std()
            )
        else:
            features[f'activity_intensity_{window}h'] = 0
            features[f'activity_variability_{window}h'] = 0
        
        # Screen time (if column exists)
        if 'screen' in features.columns:
            features[f'screen_time_{window}h'] = features.groupby('id')['screen'].transform(
                lambda x: x.rolling(window=window, min_periods=1).sum()
            )
        else:
            features[f'screen_time_{window}h'] = 0
    
    return features

def create_communication_features(df, window_sizes=[24, 48, 72, 168]):
    """Create communication-based features"""
    features = df.copy()
    
    for window in window_sizes:
        # Call frequency (if column exists)
        if 'call' in features.columns:
            features[f'call_frequency_{window}h'] = features.groupby('id')['call'].transform(
                lambda x: x.rolling(window=window, min_periods=1).sum()
            )
        else:
            features[f'call_frequency_{window}h'] = 0
        
        # SMS frequency (if column exists)
        if 'sms' in features.columns:
            features[f'sms_frequency_{window}h'] = features.groupby('id')['sms'].transform(
                lambda x: x.rolling(window=window, min_periods=1).sum()
            )
        else:
            features[f'sms_frequency_{window}h'] = 0
        
        # Total communication events
        features[f'total_communication_{window}h'] = (
            features[f'call_frequency_{window}h'] + features[f'sms_frequency_{window}h']
        )
    
    return features

def create_app_usage_features(df, window_sizes=[24, 48, 72, 168]):
    """Create app usage-based features"""
    features = df.copy()
    
    # Define app categories
    productive_apps = ['appCat_office', 'appCat_utilities', 'appCat_finance']
    entertainment_apps = ['appCat_entertainment', 'appCat_game', 'appCat_social']
    communication_apps = ['appCat_communication']
    
    # Get all app category columns
    app_columns = [col for col in features.columns if col.startswith('appCat_')]
    
    for window in window_sizes:
        # Calculate usage time for each category
        for app_cat in app_columns:
            features[f'{app_cat}_time_{window}h'] = features.groupby('id')[app_cat].transform(
                lambda x: x.rolling(window=window, min_periods=1).sum()
            )
        
        # Calculate productive vs entertainment ratio
        productive_time = sum(features[f'{app}_time_{window}h'] for app in productive_apps if app in features.columns)
        entertainment_time = sum(features[f'{app}_time_{window}h'] for app in entertainment_apps if app in features.columns)
        features[f'productive_ratio_{window}h'] = productive_time / (entertainment_time + 1)  # Add 1 to avoid division by zero
        
        # Calculate app diversity (number of different app categories used)
        time_columns = [col for col in features.columns if col.startswith('appCat_') and col.endswith(f'_time_{window}h')]
        features[f'app_diversity_{window}h'] = (features[time_columns] > 0).sum(axis=1)
    
    return features

def create_circumplex_features(df, window_sizes=[24, 48, 72, 168]):
    """Create features based on circumplex model of affect"""
    features = df.copy()
    
    for window in window_sizes:
        # Arousal features (if column exists)
        if 'circumplex_arousal' in features.columns:
            features[f'arousal_mean_{window}h'] = features.groupby('id')['circumplex_arousal'].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean()
            )
            features[f'arousal_std_{window}h'] = features.groupby('id')['circumplex_arousal'].transform(
                lambda x: x.rolling(window=window, min_periods=1).std()
            )
        else:
            features[f'arousal_mean_{window}h'] = 0
            features[f'arousal_std_{window}h'] = 0
        
        # Valence features (if column exists)
        if 'circumplex_valence' in features.columns:
            features[f'valence_mean_{window}h'] = features.groupby('id')['circumplex_valence'].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean()
            )
            features[f'valence_std_{window}h'] = features.groupby('id')['circumplex_valence'].transform(
                lambda x: x.rolling(window=window, min_periods=1).std()
            )
        else:
            features[f'valence_mean_{window}h'] = 0
            features[f'valence_std_{window}h'] = 0
        
        # Combined features
        features[f'affect_intensity_{window}h'] = np.sqrt(
            features[f'arousal_mean_{window}h']**2 + features[f'valence_mean_{window}h']**2
        )
        features[f'affect_angle_{window}h'] = np.arctan2(
            features[f'arousal_mean_{window}h'], 
            features[f'valence_mean_{window}h']
        )
    
    return features

def pivot_long_to_wide(df):
    """Convert data from long to wide format, handling duplicates by taking the mean"""
    # First, sort by time to ensure correct ordering
    df = df.sort_values(['id', 'time'])
    
    # Handle duplicates by taking the mean for each combination of id, time, and variable
    df = df.groupby(['id', 'time', 'variable'])['value'].mean().reset_index()
    
    # Pivot the data
    wide_df = df.pivot(index=['id', 'time'], columns='variable', values='value').reset_index()
    
    # Rename columns to avoid dots in column names
    wide_df.columns = [col.replace('.', '_') for col in wide_df.columns]
    
    return wide_df

def prepare_features_for_modeling(df):
    """Prepare all features for mood prediction modeling"""
    print("Creating features for mood prediction...")
    
    # Convert to wide format
    features = pivot_long_to_wide(df)
    
    # Apply feature creation functions
    features = create_temporal_features(features)
    features = create_lag_features(features, 'mood')
    features = create_activity_features(features)
    features = create_communication_features(features)
    features = create_app_usage_features(features)
    features = create_circumplex_features(features)
    
    # Add user-specific features
    features['user_avg_mood'] = features.groupby('id')['mood'].transform('mean')
    features['user_std_mood'] = features.groupby('id')['mood'].transform('std')
    
    # Calculate time since last mood report
    features['time_since_last_mood'] = features.groupby('id')['time'].diff().dt.total_seconds() / 3600
    
    print("Feature creation complete!")
    return features

def main():
    # Load the cleaned data
    df = pd.read_csv('data/cleaned_dataset.csv')
    df['time'] = pd.to_datetime(df['time'], format='mixed')
    
    # Create features
    features = prepare_features_for_modeling(df)
    
    # Save features
    features.to_csv('data/mood_prediction_features.csv', index=False)
    print("Features saved to 'data/mood_prediction_features.csv'")
    
    # Print feature summary
    print("\nFeature Summary:")
    print(f"Total number of features: {len(features.columns)}")
    print("\nFeature categories:")
    print("1. Temporal features")
    print("2. Lag features (previous mood states)")
    print("3. Activity-based features")
    print("4. Communication features")
    print("5. App usage features")
    print("6. Circumplex-based features")

if __name__ == "__main__":
    main()
