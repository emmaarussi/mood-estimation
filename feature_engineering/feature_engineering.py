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

def create_lag_features(df, variable, lags=[24, 72, 168]):
    """Create lagged features for a given variable"""
    features = df.copy()
    
    # Sort by user and time
    features = features.sort_values(['id', 'time'])
    
    # Create lag features
    for lag in lags:
        # Convert lag hours to number of records (assuming hourly data)
        lag_periods = lag
        
        # Create basic lag feature
        lag_name = f'{variable}_lag_{lag}h'
        features[lag_name] = features.groupby('id')[variable].shift(1)  # Shift by 1 to avoid using current value
        
        # Add rolling statistics using only past data
        if lag > 1:
            # Use shift(1) to exclude current value, then calculate rolling stats
            # This ensures we only use data that would have been available at prediction time
            features[f'{variable}_rolling_mean_{lag}h'] = features.groupby('id')[variable].transform(
                lambda x: x.shift(1).rolling(window=lag_periods, min_periods=1, closed='left').mean()
            )
            features[f'{variable}_rolling_std_{lag}h'] = features.groupby('id')[variable].transform(
                lambda x: x.shift(1).rolling(window=lag_periods, min_periods=1, closed='left').std()
            )
            features[f'{variable}_rolling_min_{lag}h'] = features.groupby('id')[variable].transform(
                lambda x: x.shift(1).rolling(window=lag_periods, min_periods=1, closed='left').min()
            )
            features[f'{variable}_rolling_max_{lag}h'] = features.groupby('id')[variable].transform(
                lambda x: x.shift(1).rolling(window=lag_periods, min_periods=1, closed='left').max()
            )
    
    return features

def create_activity_features(df, window_sizes=[24, 72, 168]):
    """Create activity-based features"""
    features = df.copy()
    
    # Calculate activity levels for different time windows
    for window in window_sizes:
        # Activity intensity (if column exists)
        if 'activity' in features.columns:
            features[f'activity_intensity_{window}h'] = features.groupby('id')['activity'].transform(
                lambda x: x.shift(1).rolling(window=window, min_periods=1, closed='left').mean()
            )
            
            # Activity variability
            features[f'activity_variability_{window}h'] = features.groupby('id')['activity'].transform(
                lambda x: x.shift(1).rolling(window=window, min_periods=1, closed='left').std()
            )
        else:
            features[f'activity_intensity_{window}h'] = 0
            features[f'activity_variability_{window}h'] = 0
        
        # Screen time (if column exists)
        if 'screen' in features.columns:
            features[f'screen_time_{window}h'] = features.groupby('id')['screen'].transform(
                lambda x: x.shift(1).rolling(window=window, min_periods=1, closed='left').sum()
            )
        else:
            features[f'screen_time_{window}h'] = 0
    
    return features

def create_communication_features(df, window_sizes=[24, 72, 168]):
    """Create communication-based features"""
    features = df.copy()
    
    for window in window_sizes:
        # Call frequency (if column exists)
        if 'call' in features.columns:
            features[f'call_frequency_{window}h'] = features.groupby('id')['call'].transform(
                lambda x: x.shift(1).rolling(window=window, min_periods=1, closed='left').sum()
            )
        else:
            features[f'call_frequency_{window}h'] = 0
        
        # SMS frequency (if column exists)
        if 'sms' in features.columns:
            features[f'sms_frequency_{window}h'] = features.groupby('id')['sms'].transform(
                lambda x: x.shift(1).rolling(window=window, min_periods=1, closed='left').sum()
            )
        else:
            features[f'sms_frequency_{window}h'] = 0
        
        # Total communication events
        features[f'total_communication_{window}h'] = (
            features[f'call_frequency_{window}h'] + features[f'sms_frequency_{window}h']
        )
    
    return features

def create_app_usage_features(df, window_sizes=[24, 72, 168]):
    """Create app usage-based features"""
    features = df.copy()
    
    # Define app categories
    productive_apps = ['appCat.office', 'appCat.utilities', 'appCat.finance']
    entertainment_apps = ['appCat.entertainment', 'appCat.game', 'appCat.social']
    communication_apps = ['appCat.communication']
    
    # Get all app category columns
    app_columns = [col for col in features.columns if col.startswith('appCat.')]
    
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

def create_circumplex_features(df, window_sizes=[24, 72, 168]):
    """Create features based on circumplex model of affect"""
    features = df.copy()
    
    for window in window_sizes:
        # Arousal features (if column exists)
        if 'circumplex.arousal' in features.columns:
            features[f'arousal_mean_{window}h'] = features.groupby('id')['circumplex.arousal'].transform(
                lambda x: x.shift(1).rolling(window=window, min_periods=1, closed='left').mean()
            )
            features[f'arousal_std_{window}h'] = features.groupby('id')['circumplex.arousal'].transform(
                lambda x: x.shift(1).rolling(window=window, min_periods=1, closed='left').std()
            )
        else:
            features[f'arousal_mean_{window}h'] = 0
            features[f'arousal_std_{window}h'] = 0
        
        # Valence features (if column exists)
        if 'circumplex.valence' in features.columns:
            features[f'valence_mean_{window}h'] = features.groupby('id')['circumplex.valence'].transform(
                lambda x: x.shift(1).rolling(window=window, min_periods=1, closed='left').mean()
            )
            features[f'valence_std_{window}h'] = features.groupby('id')['circumplex.valence'].transform(
                lambda x: x.shift(1).rolling(window=window, min_periods=1, closed='left').std()
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
    
    # Sort by time to ensure correct temporal order
    features = features.sort_values(['id', 'time'])
    
    # Apply feature creation functions in order of dependency
    # 1. First, create basic temporal features (these don't depend on any other features)
    features = create_temporal_features(features)
    
    # 2. Create lag features for mood (these should only use past mood values)
    features = create_lag_features(features, 'mood')
    
    # 3. Create activity and behavioral features
    features = create_activity_features(features)
    features = create_communication_features(features)
    features = create_app_usage_features(features)
    features = create_circumplex_features(features)
    
    
    # Calculate time since last mood report using only past data
    features['time_since_last_mood'] = features.groupby('id')['time'].transform(
        lambda x: x.diff().dt.total_seconds() / 3600
    ).shift(1)  # Shift to only use past data
    
    # Convert categorical columns to numeric before filling NaN
    if 'time_of_day' in features.columns:
        features['time_of_day'] = features['time_of_day'].astype('category').cat.codes
    
    # Fill NaN values that might have been created by shifting
    features = features.fillna(0)
    
    print("Feature creation complete!")
    return features

def main(input_file=None, output_file=None):
    import sys
    
    # Handle input/output paths
    if input_file is None:
        if len(sys.argv) > 1:
            input_file = sys.argv[1]
        else:
            input_file = 'data/dataset_mood_smartphone_cleaned.csv'
    
    if output_file is None:
        if len(sys.argv) > 2:
            output_file = sys.argv[2]
        else:
            output_file = 'data/mood_prediction_features.csv'
    
    # Ensure input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file not found: {input_file}")
        sys.exit(1)
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Load the cleaned data
    print(f"Loading data from {input_file}...")
    df = pd.read_csv(input_file)
    df['time'] = pd.to_datetime(df['time'], format='mixed')
    
    # Create features
    features = prepare_features_for_modeling(df)
    
    # Save features
    print(f"Saving features to {output_file}...")
    features.to_csv(output_file, index=False)
    print(f"Features saved to '{output_file}'")
    
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
