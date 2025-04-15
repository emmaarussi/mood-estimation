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
    
    # 1. Time Features
    features['hour'] = pd.to_datetime(features['time']).dt.hour
    features['is_weekend'] = pd.to_datetime(features['time']).dt.dayofweek.isin([5, 6]).astype(int)
    
    # 2. Recent History (last 24h)
    features = features.sort_values(['id', 'time'])
    features['prev_mood'] = features.groupby('id')['mood'].shift(1)
    features['mood_change'] = features['mood'] - features['prev_mood']
    
    # 3. Activity Level
    if 'activity' in features.columns:
        features['recent_activity'] = features.groupby('id')['activity'].transform(
            lambda x: x.rolling(window=24, min_periods=1).mean()
        )
    else:
        features['recent_activity'] = 0
        
    # 4. Screen Time
    if 'screen' in features.columns:
        features['daily_screen_time'] = features.groupby('id')['screen'].transform(
            lambda x: x.rolling(window=24, min_periods=1).sum()
        )
    else:
        features['daily_screen_time'] = 0
    
    # 5. Communication
    # Combine all communication apps
    comm_cols = [col for col in features.columns if 'communication' in col.lower()]
    features['communication_time'] = features[comm_cols].sum(axis=1) if comm_cols else 0
    
    # 6. Basic Circumplex
    if 'circumplex_arousal' in features.columns and 'circumplex_valence' in features.columns:
        features['emotion_intensity'] = np.sqrt(
            features['circumplex_arousal']**2 + features['circumplex_valence']**2
        )
    else:
        features['emotion_intensity'] = 0
    
    # 7. User baseline
    features['user_avg_mood'] = features.groupby('id')['mood'].transform('mean')
    features['mood_vs_average'] = features['mood'] - features['user_avg_mood']
    
    print("Basic feature creation complete!")
    return features

if __name__ == "__main__":
    # Load the cleaned data
    df = pd.read_csv('data/dataset_mood_smartphone_cleaned.csv')
    df['time'] = pd.to_datetime(df['time'], format='mixed')
    
    # Create basic features
    features = create_basic_features(df)
    
    # Save features
    features.to_csv('data/mood_prediction_simple_features.csv', index=False)
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
