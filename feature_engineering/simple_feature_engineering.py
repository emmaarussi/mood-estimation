import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import LabelEncoder

def create_basic_features(df):
    """Create a simple set of features for initial analysis"""
    print("Creating basic features...")
    
    # First convert to wide format - handle duplicates by taking mean
    features = df.groupby(['id', 'time', 'variable'])['value'].mean().reset_index()
    features = features.pivot(index=['id', 'time'], columns='variable', values='value').reset_index()
    features.columns = [col.replace('.', '_') for col in features.columns]
    
    # Filter for timestamps where we have mood recordings
    print(f"Initial shape: {features.shape}")
    features = features.dropna(subset=['mood'])
    print(f"Shape after filtering for mood recordings: {features.shape}")
    
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

def prepare_rolling_window_data(df, window_size=7, categorical=False):
    """Prepare data with rolling window features"""
    # Sort by user and time
    df = df.sort_values(['id', 'time'])
    
    # Create daily aggregates
    daily = df.groupby(['id', df['time'].dt.date]).agg({
        'mood': 'mean',
        'recent_activity': 'mean',
        'daily_screen_time': 'sum',
        'communication_time': 'sum',
        'circumplex_arousal': 'mean',
        'circumplex_valence': 'mean',
        'emotion_intensity': 'mean',
        'hour': lambda x: len(x),  # number of measurements
    }).reset_index()
    daily.columns = ['id', 'date'] + list(daily.columns[2:])
    daily['date'] = pd.to_datetime(daily['date'])
    
    # Create features from rolling windows
    features = []
    targets = []
    dates = []
    user_ids = []
    
    for user in daily['id'].unique():
        user_data = daily[daily['id'] == user].copy()
        
        for i in range(len(user_data) - window_size):
            window = user_data.iloc[i:i+window_size]
            target_day = user_data.iloc[i+window_size]
            
            # Skip if gap is more than 1 day
            if (target_day['date'] - window['date'].iloc[-1]).days > 1:
                continue
                
            # Create features
            feature_dict = {
                'user_id': user,
                'date': target_day['date'],
                'measurements': window['hour'].mean(),
                'mood_mean': window['mood'].mean(),
                'mood_std': window['mood'].std(),
                'mood_trend': window['mood'].iloc[-1] - window['mood'].iloc[0],
                'mood_lag':window['mood'].iloc[-1],
                'activity_roll_mean': window['recent_activity'].mean(),
                'screen_time_mean': window['daily_screen_time'].mean(),
                'communication_mean': window['communication_time'].mean(),
                'arousal_mean': window['circumplex_arousal'].mean(),
                'arousal_lag': window['circumplex_arousal'].iloc[-1],
                'valence_mean': window['circumplex_valence'].mean(),
                'valence_lag':window['circumplex_valence'].iloc[-1],
                'emotion_mean': window['emotion_intensity'].mean(),
                'emotion_lag':window['emotion_intensity'].iloc[-1],
                'day_of_week': target_day['date'].dayofweek,
                'is_weekend': target_day['date'].dayofweek >= 5
            }
            
            features.append(feature_dict)
            targets.append(target_day['mood'])
            dates.append(target_day['date'])
            user_ids.append(user)
    
    X = pd.DataFrame(features)
    y = np.array(targets)
    
    X = pd.DataFrame(features)
    y = np.array(targets)
    encoder = None

    if categorical:
        labels = ['Low', 'High']
        bins = [0.5, 6.5, 10.5]  # Low: 0.5-6.5, High: 6.5-10.5
        y_categorical = pd.cut(y, bins=bins, labels=labels, right=True, include_lowest=True)
        mask = ~pd.isna(y_categorical)
        X = X[mask].reset_index(drop=True)
        y_categorical = y_categorical[mask]
        dates = np.array(dates)[mask]
        user_ids = np.array(user_ids)[mask]
        encoder = LabelEncoder()
        y = encoder.fit_transform(y_categorical.astype(str))
        
        print(f"\nBinning mood scores into classes:")
        print(f"  Bins: {bins}")
        print(f"  Labels: {labels}")
        print("  Number of samples in each bin:")
        print(y_categorical.value_counts())

    return X, y, dates, user_ids, encoder

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
