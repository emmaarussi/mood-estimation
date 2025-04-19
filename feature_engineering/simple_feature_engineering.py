import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from pathlib import Path

window_length = 4

def feature_engineering_pipeline(window_length=3):
    # Main function!
    
    output_path = Path('data')
    input_file = 'data/dataset_mood_smartphone_cleaned.csv'
    
    # Input
    df = pd.read_csv(input_file)
    
    # Generate DataFrames 
    daily_df = get_daily_data(df)
    basic_feature_df = create_basic_features(daily_df)
    rolling_window_df = add_rolling_window_features(basic_feature_df, window_length=window_length)
    
    daily_df_base = output_path / 'daily_data'
    basic_features_base = output_path / 'basic_features'
    rolling_features_base = output_path / f'rolling_features_{window_length}d'
    
    # Save daily_df
    daily_df.to_csv(f"{daily_df_base}.csv", index=False)
    daily_df.to_parquet(f"{daily_df_base}.parquet", index=False)
    print(f"  Saved daily data to {daily_df_base}.csv/.parquet")

    # Save basic_feature_df
    basic_feature_df.to_csv(f"{basic_features_base}.csv", index=False)
    basic_feature_df.to_parquet(f"{basic_features_base}.parquet", index=False)
    print(f"  Saved basic features to {basic_features_base}.csv/.parquet")

    # Save rolling_window_df
    rolling_window_df.to_csv(f"{rolling_features_base}.csv", index=False)
    rolling_window_df.to_parquet(f"{rolling_features_base}.parquet", index=False)
    print(f"  Saved rolling window features to {rolling_features_base}.csv/.parquet")
    
def get_daily_data(df):
    
    # Add date columns
    df['time'] = pd.to_datetime(df['time'], format='mixed')
    df['date'] = df['time'].dt.date
    
    # First convert to wide format - handle duplicates by taking mean
    df = df.groupby(['id', 'time', 'variable'])['value'].mean().reset_index()
    df = df.pivot(index=['id', 'time'], columns='variable', values='value').reset_index()
    
    # Rename Columns
    df.columns = [col.replace('.', '_') for col in df.columns]
    
    df['date'] = df['time'].dt.date
    
    # Create daily aggregates
    daily = df.groupby(['id', 'date']).agg({
        'activity': 'mean',
        'appCat_builtin': 'sum',
        'appCat_communication': 'sum',
        'appCat_entertainment': 'sum',
        'appCat_finance': 'sum',
        'appCat_game': 'sum',
        'appCat_office': 'sum',
        'appCat_other': 'sum',
        'appCat_social': 'sum',
        'appCat_travel': 'sum',
        'appCat_unknown': 'sum',
        'appCat_utilities': 'sum',
        'appCat_weather': 'sum',
        'call': 'sum',
        'circumplex_arousal': 'mean',
        'circumplex_valence': 'mean',
        'mood': 'mean',
        'screen':'sum',
        'sms':'sum'
    }).reset_index()
    
    # Fill missing activity with zero
    daily.fillna({'activity':0})
    
    # NOTE: all missing values between circumplex and mood are common
    # Only one observation has missing mood, but non-missing circumplex
    # This is not an issue because we delete this entry anyway
    # daily_df[(~daily_df['circumplex_arousal'].isna() & daily_df['mood'].isna())]
    
    daily = daily.sort_values(by=['id', 'date'])
    
    daily['date'] = pd.to_datetime(daily['date'])

    # Get the mood from the next row within each group
    daily['next_day_mood'] = daily.groupby('id')['mood'].shift(-1)

    # Get the date from the next row within each group
    daily['next_day_date'] = daily.groupby('id')['date'].shift(-1)

    # Calculate the difference in days between the current date and the next available date
    daily['date_diff'] = (daily['next_day_date'] - daily['date']).dt.days

    # Assign target_mood only if the next available date is exactly one day later
    daily['target_mood'] = np.where(
        daily['date_diff'] == 1,  # Condition: Is the next row exactly 1 day later?
        daily['next_day_mood'],   # Value if True: Use the mood from the next row
        pd.NA                       # Value if False: Assign NA (or np.nan)
    )

    # Drop temporary columns and rows where target_mood is NA
    daily = daily.drop(columns=['next_day_mood', 'next_day_date', 'date_diff'])
    
    return daily

def create_basic_features(daily_df: pd.DataFrame):
    """Create a simple set of features for initial analysis"""
    print("Creating basic features...")
    
    df = daily_df.copy()
    
    # Create aggregated app usage features
    social_columns = ['appCat_communication', 'appCat_social', 'appCat_builtin']
    entertainment_leisure_columns = ['appCat_entertainment', 'appCat_game', 'appCat_travel', 'appCat_weather']
    productivity_work_columns =  ['appCat_office', 'appCat_finance', 'appCat_utilities']
    miscellaneous_columns = ['appCat_other', 'appCat_unknown']
    
    df['social_communication'] = df[social_columns].sum(axis=1)
    df['entertainment_leisure'] = df[entertainment_leisure_columns].sum(axis=1)
    df['productivity_work'] = df[productivity_work_columns].sum(axis=1)
    df['miscellaneous'] = df[miscellaneous_columns].sum(axis=1)
    
    df['day_of_week'] = df['date'].dt.weekday
    df['is_weekend'] = df['day_of_week'] >= 5
    
    # Emotion intensity feature
    df['emotion_intensity'] = np.sqrt(df['circumplex_arousal']**2 + df['circumplex_valence']**2)
    
    # Add mood missing feature
    df["mood_missing"] = df["mood"].isna().astype(int)
    
    features_to_drop = (social_columns + entertainment_leisure_columns 
                        + productivity_work_columns + miscellaneous_columns)
    
    df = df.drop(features_to_drop, axis = 1)
    
    return df
    
def add_rolling_window_features(df: pd.DataFrame,window_length=3):
    # window_length in days
    
    # Create Rolling window
    def calendar_rolling_features(group, columns, window=f"{window_length}D"):
        group['date'] = pd.to_datetime(group['date'])
        group = group.sort_values('date')
        group = group.set_index('date')
        for col in columns:
            group[f'{col}_rolling_{window_length}d'] = group[col].rolling(window).mean()
        return group.reset_index()

    # Define features for rolling window
    calendar_rolling_features_list = [
        'activity', 'call', 'sms', 'screen', 'mood',
        'social_communication', 'entertainment_leisure', 'productivity_work',
        'miscellaneous', 'circumplex_arousal', 'circumplex_valence', 'emotion_intensity'
    ]

    # Apply the function per user
    rolled_df = df.groupby('id', group_keys=False).apply(
        calendar_rolling_features,
        columns=calendar_rolling_features_list
    )
    
    # Impute Missing Rolling Features
     # Impute Missing Rolling Features
    rolling_cols = [f'{col}_rolling_{window_length}d' for col in calendar_rolling_features_list]
    # Get original column names corresponding to rolling columns
    original_cols_map = {f'{col}_rolling_{window_length}d': col for col in calendar_rolling_features_list}

    for col in rolling_cols:
        if col in rolled_df.columns:
            original_col = original_cols_map[col] # Find the original feature name

            # 1. Fill initial NaNs using the user's mean of the *available rolling values*
            rolled_df[col] = rolled_df.groupby('id')[col].transform(lambda x: x.fillna(x.mean()))

            # 2. Fill remaining NaNs (users with NO rolling values) using the user's mean of the *original feature*
            # Check if there are still NaNs after the first transform
            if rolled_df[col].isnull().any():
                 # Calculate the user-specific mean of the ORIGINAL column
                 user_mean_original = rolled_df.groupby('id')[original_col].transform('mean')
                 # Fill NaNs in the ROLLING column with the user's mean from the ORIGINAL column
                 rolled_df[col] = rolled_df[col].fillna(user_mean_original)

            # 3. Final Fallback (Optional but recommended): Fill any remaining NaNs (e.g., user has no data for original feature either)
            #    You could use 0 or the global mean of the *original* feature. Using 0 is often simple.
            if rolled_df[col].isnull().any():
                 rolled_df[col] = rolled_df[col].fillna(0)
    
    # Drop old columns
    rolled_df = rolled_df.drop([x for x in calendar_rolling_features_list if x != 'mood'], axis=1)
    
    rolled_df = rolled_df.dropna(subset=['target_mood'])
    
    print("Basic feature creation complete!")
    
    return rolled_df

def prepare_rolling_window_data(daily, window_size=7, categorical=False):
    """Prepare data with rolling window features"""
    
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
    feature_engineering_pipeline(window_length=window_length)