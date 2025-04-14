import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import calendar

def load_and_prepare_data(file_path):
    """Load and prepare the dataset for temporal analysis"""
    df = pd.read_csv(file_path)
    df['time'] = pd.to_datetime(df['time'])
    df['hour'] = df['time'].dt.hour
    df['day_of_week'] = df['time'].dt.dayofweek
    return df

def plot_daily_patterns(df, output_path):
    """Create a 2x2 plot of daily patterns for mood, arousal, valence, and activity"""
    plt.figure(figsize=(15, 12))
    variables = ['mood', 'circumplex.arousal', 'circumplex.valence', 'activity']
    titles = ['Mood', 'Arousal', 'Valence', 'Activity']
    
    for idx, (var, title) in enumerate(zip(variables, titles)):
        plt.subplot(2, 2, idx + 1)
        
        # Calculate mean and confidence intervals
        stats = df.groupby('hour')[var].agg(['mean', 'std', 'count'])
        stats['ci'] = 1.96 * stats['std'] / np.sqrt(stats['count'])
        
        # Plot mean with confidence intervals
        plt.plot(stats.index, stats['mean'], 'b-', label='Mean')
        plt.fill_between(stats.index, 
                        stats['mean'] - stats['ci'],
                        stats['mean'] + stats['ci'],
                        alpha=0.2)
        
        plt.title(f'{title} by Hour of Day')
        plt.xlabel('Hour')
        plt.ylabel(title)
        plt.grid(True, alpha=0.3)
        plt.xticks(range(0, 24, 3))
    
    plt.tight_layout()
    plt.savefig(f'{output_path}/daily_patterns.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_weekly_patterns(df, output_path):
    """Create a 2x2 plot of weekly patterns for mood, arousal, valence, and activity"""
    plt.figure(figsize=(15, 12))
    variables = ['mood', 'circumplex.arousal', 'circumplex.valence', 'activity']
    titles = ['Mood', 'Arousal', 'Valence', 'Activity']
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
    for idx, (var, title) in enumerate(zip(variables, titles)):
        plt.subplot(2, 2, idx + 1)
        
        # Calculate mean and confidence intervals
        stats = df.groupby('day_of_week')[var].agg(['mean', 'std', 'count'])
        stats['ci'] = 1.96 * stats['std'] / np.sqrt(stats['count'])
        
        # Plot mean with confidence intervals
        plt.plot(stats.index, stats['mean'], 'g-', label='Mean')
        plt.fill_between(stats.index, 
                        stats['mean'] - stats['ci'],
                        stats['mean'] + stats['ci'],
                        alpha=0.2)
        
        plt.title(f'{title} by Day of Week')
        plt.xlabel('Day')
        plt.ylabel(title)
        plt.grid(True, alpha=0.3)
        plt.xticks(range(7), days, rotation=45)
    
    plt.tight_layout()
    plt.savefig(f'{output_path}/weekly_patterns.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # Set style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # Load data
    df = load_and_prepare_data('data/dataset_mood_smartphone.csv')
    
    # Create output directory if it doesn't exist
    output_path = 'outputs/temporal_patterns'
    os.makedirs(output_path, exist_ok=True)
    
    # Generate plots
    plot_daily_patterns(df, output_path)
    plot_weekly_patterns(df, output_path)
    
    print("Temporal pattern plots have been generated in the outputs/temporal_patterns directory.")

if __name__ == "__main__":
    import os
    main()
