import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import os

# Create output directory for feature analysis
FEATURE_ANALYSIS_DIR = 'outputs/feature_analysis'
os.makedirs(FEATURE_ANALYSIS_DIR, exist_ok=True)

def load_and_prepare_data(file_path):
    """Load and prepare the dataset with basic temporal features"""
    print("Loading and preparing data...")
    df = pd.read_csv(file_path)
    df['time'] = pd.to_datetime(df['time'])
    
    # Add basic temporal features
    df['hour'] = df['time'].dt.hour
    df['day_of_week'] = df['time'].dt.dayofweek
    df['date'] = df['time'].dt.date
    return df

def analyze_circumplex_relationship(df):
    """Analyze and visualize relationship between mood, arousal, and valence"""
    print("\nAnalyzing circumplex measures...")
    
    # Get mood data points with corresponding arousal and valence
    mood_data = df[df['variable'] == 'mood'][['time', 'id', 'value']].rename(columns={'value': 'mood'})
    arousal_data = df[df['variable'] == 'circumplex.arousal'][['time', 'id', 'value']].rename(columns={'value': 'arousal'})
    valence_data = df[df['variable'] == 'circumplex.valence'][['time', 'id', 'value']].rename(columns={'value': 'valence'})
    
    # Merge data
    merged = pd.merge_asof(
        pd.merge_asof(mood_data.sort_values('time'), 
                     arousal_data.sort_values('time'), 
                     by='id', on='time', 
                     tolerance=pd.Timedelta('1H')),
        valence_data.sort_values('time'),
        by='id', on='time',
        tolerance=pd.Timedelta('1H')
    )
    
    # Create scatter plots
    plt.figure(figsize=(15, 5))
    
    plt.subplot(131)
    sns.scatterplot(data=merged, x='arousal', y='mood', alpha=0.5)
    plt.title('Mood vs. Arousal')
    
    plt.subplot(132)
    sns.scatterplot(data=merged, x='valence', y='mood', alpha=0.5)
    plt.title('Mood vs. Valence')
    
    plt.subplot(133)
    sns.scatterplot(data=merged, x='valence', y='arousal', hue='mood', alpha=0.5)
    plt.title('Arousal vs. Valence (colored by mood)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(FEATURE_ANALYSIS_DIR, 'circumplex_relationships.png'))
    plt.close()
    
    return merged

def analyze_temporal_patterns(df):
    """Analyze and visualize temporal patterns in mood and activity"""
    print("\nAnalyzing temporal patterns...")
    
    # Prepare mood data by hour and day
    mood_data = df[df['variable'] == 'mood'].copy()
    
    # Hourly patterns
    plt.figure(figsize=(15, 5))
    
    plt.subplot(121)
    sns.boxplot(data=mood_data, x='hour', y='value')
    plt.title('Mood Distribution by Hour')
    plt.xlabel('Hour of Day')
    plt.ylabel('Mood')
    
    plt.subplot(122)
    sns.boxplot(data=mood_data, x='day_of_week', y='value')
    plt.title('Mood Distribution by Day of Week')
    plt.xlabel('Day of Week (0=Monday)')
    plt.ylabel('Mood')
    
    plt.tight_layout()
    plt.savefig(os.path.join(FEATURE_ANALYSIS_DIR, 'temporal_patterns.png'))
    plt.close()

def analyze_app_usage(df):
    """Analyze and visualize app usage patterns"""
    print("\nAnalyzing app usage patterns...")
    
    # Calculate daily app usage by category
    app_data = df[df['variable'].str.startswith('appCat')].copy()
    daily_app_usage = app_data.groupby(['date', 'variable'])['value'].sum().reset_index()
    
    # Split apps into productive and entertainment
    productive_apps = ['appCat.office', 'appCat.utilities', 'appCat.finance']
    entertainment_apps = ['appCat.entertainment', 'appCat.game', 'appCat.social']
    
    # Calculate daily ratios
    daily_productive = daily_app_usage[daily_app_usage['variable'].isin(productive_apps)].groupby('date')['value'].sum()
    daily_entertainment = daily_app_usage[daily_app_usage['variable'].isin(entertainment_apps)].groupby('date')['value'].sum()
    daily_ratio = (daily_productive / daily_entertainment).reset_index()
    
    # Plot distribution of productive/entertainment ratio
    plt.figure(figsize=(10, 5))
    sns.histplot(data=daily_ratio, x='value', bins=30)
    plt.title('Distribution of Productive vs Entertainment App Usage Ratio')
    plt.xlabel('Ratio (Productive/Entertainment time)')
    plt.ylabel('Count')
    plt.savefig(os.path.join(FEATURE_ANALYSIS_DIR, 'app_usage_ratio.png'))
    plt.close()
    
    return daily_ratio

def analyze_communication_patterns(df):
    """Analyze and visualize communication patterns"""
    print("\nAnalyzing communication patterns...")
    
    # Get communication data
    calls = df[df['variable'] == 'call'].copy()
    sms = df[df['variable'] == 'sms'].copy()
    
    # Calculate hourly event counts
    calls['hour'] = calls['time'].dt.hour
    sms['hour'] = sms['time'].dt.hour
    
    hourly_calls = calls.groupby('hour')['value'].sum()
    hourly_sms = sms.groupby('hour')['value'].sum()
    
    # Plot hourly patterns
    plt.figure(figsize=(12, 5))
    plt.plot(hourly_calls.index, hourly_calls.values, label='Calls')
    plt.plot(hourly_sms.index, hourly_sms.values, label='SMS')
    plt.title('Communication Events by Hour')
    plt.xlabel('Hour of Day')
    plt.ylabel('Number of Events')
    plt.legend()
    plt.savefig(os.path.join(FEATURE_ANALYSIS_DIR, 'communication_patterns.png'))
    plt.close()

def main():
    # Load data
    df = load_and_prepare_data('data/dataset_mood_smartphone.csv')
    
    # Run analyses
    circumplex_data = analyze_circumplex_relationship(df)
    analyze_temporal_patterns(df)
    app_ratio_data = analyze_app_usage(df)
    analyze_communication_patterns(df)
    
    print("\nAnalysis complete! Check the outputs/feature_analysis directory for visualizations.")

if __name__ == "__main__":
    main()
