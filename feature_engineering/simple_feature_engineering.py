import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def create_basic_features(df):
    """Create a simple set of features for initial analysis"""
    print("Creating basic features...")
    
    # First convert to wide format - handle duplicates by taking mean
    features = df.groupby(['id', 'time', 'variable'])['value'].mean().reset_index()
    features = features.pivot(index=['id', 'time'], columns='variable', values='value').reset_index()
    features.columns = [col.replace('.', '_') for col in features.columns]
    
    # Sort by user and time
    features = features.sort_values(['id', 'time'])
    
    # Filter for timestamps where we have mood recordings
    print(f"Initial shape: {features.shape}")
    features = features.dropna(subset=['mood'])
    print(f"Shape after filtering for mood recordings: {features.shape}")
    
    # 1. Time Features (no leakage - only current timestamp)
    features['hour'] = pd.to_datetime(features['time']).dt.hour
    features['day_of_week'] = pd.to_datetime(features['time']).dt.dayofweek
    
    # 2. Recent History (24h window, excluding current)
    features['prev_mood'] = features.groupby('id')['mood'].shift(1)
    features['mood_std_24h'] = features.groupby('id')['mood'].transform(
        lambda x: x.shift(1).rolling(window=24, min_periods=1, closed='left').std()
    )
    features['mood_trend'] = features.groupby('id')['prev_mood'].transform(
        lambda x: x.diff(periods=24)  # 24-hour trend using only past data
    )
    
    # 3. Activity Level (24h window, excluding current)
    if 'activity' in features.columns:
        features['activity_mean_24h'] = features.groupby('id')['activity'].transform(
            lambda x: x.shift(1).rolling(window=24, min_periods=1, closed='left').mean()
        )
        features['activity_std_24h'] = features.groupby('id')['activity'].transform(
            lambda x: x.shift(1).rolling(window=24, min_periods=1, closed='left').std()
        )
    else:
        features['activity_mean_24h'] = 0
        features['activity_std_24h'] = 0
        
    # 4. Screen Time (24h window, excluding current)
    if 'screen' in features.columns:
        features['screen_time_24h'] = features.groupby('id')['screen'].transform(
            lambda x: x.shift(1).rolling(window=24, min_periods=1, closed='left').sum()
        )
    else:
        features['screen_time_24h'] = 0
    
    # 5. Communication (24h window, excluding current)
    comm_cols = [col for col in features.columns if 'communication' in col.lower()]
    if comm_cols:
        features['communication_24h'] = features.groupby('id')[comm_cols].transform(
            lambda x: x.shift(1).rolling(window=24, min_periods=1, closed='left').sum()
        ).sum(axis=1)
    else:
        features['communication_24h'] = 0
    
    # 6. Circumplex (24h window, excluding current)
    if 'circumplex_arousal' in features.columns and 'circumplex_valence' in features.columns:
        # Calculate emotion metrics using past 24h
        arousal_24h = features.groupby('id')['circumplex_arousal'].transform(
            lambda x: x.shift(1).rolling(window=24, min_periods=1, closed='left').mean()
        )
        valence_24h = features.groupby('id')['circumplex_valence'].transform(
            lambda x: x.shift(1).rolling(window=24, min_periods=1, closed='left').mean()
        )
        features['emotion_intensity_24h'] = np.sqrt(arousal_24h**2 + valence_24h**2)
    else:
        features['emotion_intensity_24h'] = 0
    
    # 7. User baseline (expanding window of past data only)
    features['user_avg_mood'] = features.groupby('id')['mood'].transform(
        lambda x: x.shift(1).expanding().mean()
    )
    features['mood_vs_baseline'] = features['prev_mood'] - features['user_avg_mood']
    
    # 8. Data quality
    features['measurements_24h'] = features.groupby('id')['mood'].transform(
        lambda x: x.shift(1).rolling(window=24, min_periods=1, closed='left').count()
    )
    
    print("Basic feature creation complete!")
    return features
    print("Basic feature creation complete!")
    return features

if __name__ == "__main__":
    import sys
    import os

    # Get input and output paths
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    else:
        input_file = 'data/dataset_mood_smartphone_cleaned.csv'

    if len(sys.argv) > 2:
        output_file = sys.argv[2]
    else:
        output_file = 'data/mood_prediction_simple_features.csv'

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
    
    # Create basic features
    features = create_basic_features(df)
    
    # Save features
    print(f"Saving features to {output_file}...")
    features.to_csv(output_file, index=False)
    print("\nFeature Summary:")
    print(f"Total features: {len(features.columns)}")
    print("\nBasic features created:")
    print("1. Time: hour, is_weekend")
    print("2. History: prev_mood, mood_change")
    print("3. Activity: recent_activity")
    print("4. Phone: daily_screen_time")
    print("5. Social: communication_time")
    print("6. Emotion: emotion_intensity")
    print("7. Personal: user_avg_mood, mood_vs_average")
