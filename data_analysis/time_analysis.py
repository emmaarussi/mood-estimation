import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import timedelta

# Create output directory
TIME_ANALYSIS_DIR = 'outputs/time_analysis'
os.makedirs(TIME_ANALYSIS_DIR, exist_ok=True)

def analyze_time_spans(df):
    """Analyze the time span and sampling intervals for each participant"""
    print("\nAnalyzing time spans and sampling intervals...")
    
    # Get time spans for each user and variable
    time_spans = df.groupby(['id', 'variable']).agg({
        'time': ['min', 'max', 'count']
    }).reset_index()
    
    time_spans.columns = ['id', 'variable', 'start_time', 'end_time', 'record_count']
    time_spans['duration_days'] = (time_spans['end_time'] - time_spans['start_time']).dt.total_seconds() / (24 * 3600)
    
    # Print overall study duration
    print("\nOverall study duration:")
    print(f"Start: {df['time'].min()}")
    print(f"End: {df['time'].max()}")
    print(f"Total days: {(df['time'].max() - df['time'].min()).days}")
    
    # Print per-user statistics
    print("\nPer-user participation duration:")
    user_duration = time_spans.groupby('id').agg({
        'start_time': 'min',
        'end_time': 'max',
        'duration_days': 'max'
    }).sort_values('duration_days', ascending=False)
    print(user_duration)
    
    # Analyze sampling intervals for key variables
    key_vars = ['mood', 'circumplex.arousal', 'circumplex.valence', 'activity']
    intervals = {}
    
    for var in key_vars:
        var_data = df[df['variable'] == var].copy()
        var_data = var_data.sort_values('time')
        
        # Calculate time differences between consecutive records for each user
        intervals[var] = var_data.groupby('id').agg({
            'time': lambda x: x.diff().median()
        }).reset_index()
        
        print(f"\nTypical sampling interval for {var}:")
        print(intervals[var]['time'].describe())
    
    # Plot participation timeline
    plt.figure(figsize=(15, 8))
    for idx, user in enumerate(df['id'].unique()):
        user_data = df[df['id'] == user]
        plt.scatter(user_data['time'], [idx] * len(user_data), 
                   alpha=0.1, s=1, label=user if idx == 0 else "")
    
    plt.yticks(range(len(df['id'].unique())), df['id'].unique())
    plt.xlabel('Time')
    plt.ylabel('User ID')
    plt.title('Participation Timeline by User')
    plt.tight_layout()
    plt.savefig(os.path.join(TIME_ANALYSIS_DIR, 'participation_timeline.png'))
    plt.close()
    
    # Plot distribution of sampling intervals for mood
    mood_data = df[df['variable'] == 'mood'].copy().sort_values('time')
    mood_intervals = mood_data.groupby('id')['time'].diff().dt.total_seconds() / 3600  # Convert to hours
    
    plt.figure(figsize=(12, 6))
    sns.histplot(mood_intervals[mood_intervals < 48], bins=50)  # Filter out intervals > 48 hours for better visualization
    plt.title('Distribution of Mood Sampling Intervals')
    plt.xlabel('Hours between mood reports')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(os.path.join(TIME_ANALYSIS_DIR, 'mood_sampling_intervals.png'))
    plt.close()
    
    return user_duration, intervals

def main():
    # Load data
    print("Loading data...")
    df = pd.read_csv('data/dataset_mood_smartphone.csv')
    df['time'] = pd.to_datetime(df['time'])
    
    # Run analysis
    user_duration, intervals = analyze_time_spans(df)

if __name__ == "__main__":
    main()
