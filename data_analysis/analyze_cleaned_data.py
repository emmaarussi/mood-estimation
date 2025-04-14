import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
from datetime import datetime
import calendar

# Constants
DATA_DIR = 'data'
OUTPUT_DIR = 'data_analysis/plots/cleaned'
CORRELATION_DIR = os.path.join(OUTPUT_DIR, 'correlations')
TEMPORAL_DIR = os.path.join(OUTPUT_DIR, 'temporal')
APP_DIR = os.path.join(OUTPUT_DIR, 'app_usage')

def analyze_mood_patterns(df):
    """Analyze patterns in mood, arousal, and valence after cleaning.
    
    Args:
        df (pd.DataFrame): Cleaned dataframe
    """
    plt.figure(figsize=(15, 5))
    
    # Plot mood distribution
    plt.subplot(131)
    sns.histplot(data=df[df['variable'] == 'mood'], x='value', bins=20)
    plt.title('Distribution of Mood\n(After Cleaning)')
    plt.xlabel('Mood Value')
    
    # Plot arousal distribution
    plt.subplot(132)
    sns.histplot(data=df[df['variable'] == 'circumplex.arousal'], x='value', bins=20)
    plt.title('Distribution of Arousal\n(After Cleaning)')
    plt.xlabel('Arousal Value')
    
    # Plot valence distribution
    plt.subplot(133)
    sns.histplot(data=df[df['variable'] == 'circumplex.valence'], x='value', bins=20)
    plt.title('Distribution of Valence\n(After Cleaning)')
    plt.xlabel('Valence Value')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'mood_distributions.png'))
    plt.close()

def analyze_temporal_patterns(df):
    """Analyze temporal patterns in the cleaned data.
    
    Args:
        df (pd.DataFrame): Cleaned dataframe
    """
    # Add hour and day columns
    df['hour'] = df['time'].dt.hour
    df['day'] = df['time'].dt.day_name()
    
    # Plot hourly patterns for mood variables
    mood_vars = ['mood', 'circumplex.arousal', 'circumplex.valence']
    plt.figure(figsize=(15, 5))
    
    for i, var in enumerate(mood_vars, 1):
        plt.subplot(1, 3, i)
        var_data = df[df['variable'] == var].copy()
        hourly_mean = var_data.groupby('hour')['value'].mean()
        hourly_std = var_data.groupby('hour')['value'].std()
        
        plt.plot(hourly_mean.index, hourly_mean.values, 'b-', label='Mean')
        plt.fill_between(hourly_mean.index, 
                        hourly_mean.values - hourly_std.values,
                        hourly_mean.values + hourly_std.values,
                        alpha=0.2)
        plt.title(f'Hourly {var.capitalize()} Pattern')
        plt.xlabel('Hour of Day')
        plt.ylabel('Value')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(TEMPORAL_DIR, 'hourly_patterns.png'))
    plt.close()
    
    # Plot daily patterns
    plt.figure(figsize=(15, 5))
    days_order = list(calendar.day_name)
    
    for i, var in enumerate(mood_vars, 1):
        plt.subplot(1, 3, i)
        var_data = df[df['variable'] == var].copy()
        daily_mean = var_data.groupby('day')['value'].mean()
        daily_std = var_data.groupby('day')['value'].std()
        
        # Reorder days
        daily_mean = daily_mean.reindex(days_order)
        daily_std = daily_std.reindex(days_order)
        
        plt.bar(range(len(days_order)), daily_mean.values)
        plt.errorbar(range(len(days_order)), daily_mean.values, 
                    yerr=daily_std.values, fmt='none', color='black', capsize=5)
        plt.title(f'Daily {var.capitalize()} Pattern')
        plt.xticks(range(len(days_order)), days_order, rotation=45)
        plt.ylabel('Value')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(TEMPORAL_DIR, 'daily_patterns.png'))
    plt.close()

def analyze_app_usage_patterns(df):
    """Analyze app usage patterns in the cleaned data.
    
    Args:
        df (pd.DataFrame): Cleaned dataframe
    """
    # Get app categories
    app_cats = [col for col in df['variable'].unique() if col.startswith('appCat')]
    
    # Calculate daily usage for each category
    daily_usage = {}
    for cat in app_cats:
        cat_data = df[df['variable'] == cat].copy()
        cat_data['date'] = cat_data['time'].dt.date
        daily_usage[cat] = cat_data.groupby('date')['value'].sum().mean()
    
    # Plot average daily usage
    plt.figure(figsize=(12, 6))
    usage_series = pd.Series(daily_usage)
    usage_series.sort_values(ascending=True).plot(kind='barh')
    plt.title('Average Daily App Usage by Category\n(After Cleaning)')
    plt.xlabel('Average Daily Usage (seconds)')
    plt.tight_layout()
    plt.savefig(os.path.join(APP_DIR, 'daily_app_usage.png'))
    plt.close()
    
    # Calculate and plot usage correlations
    usage_corr = pd.DataFrame()
    for cat in app_cats:
        cat_data = df[df['variable'] == cat].copy()
        cat_data['date'] = cat_data['time'].dt.date
        usage_corr[cat.replace('appCat.', '')] = cat_data.groupby('date')['value'].sum()
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(usage_corr.corr(), annot=True, cmap='RdBu_r', center=0)
    plt.title('App Usage Category Correlations\n(After Cleaning)')
    plt.tight_layout()
    plt.savefig(os.path.join(APP_DIR, 'app_correlations.png'))
    plt.close()

def analyze_mood_correlations(df):
    """Analyze correlations between mood and other variables in cleaned data.
    
    Args:
        df (pd.DataFrame): Cleaned dataframe
    """
    # Prepare data for correlation analysis
    mood_data = df[df['variable'] == 'mood'].copy()
    mood_data['date_hour'] = mood_data['time'].dt.floor('h')
    hourly_mood = mood_data.groupby('date_hour')['value'].mean().fillna(0)
    
    correlations = []
    for var in df['variable'].unique():
        if var != 'mood':
            var_data = df[df['variable'] == var].copy()
            var_data['date_hour'] = var_data['time'].dt.floor('h')
            hourly_var = var_data.groupby('date_hour')['value'].mean().fillna(0)
            
            # Calculate correlation
            common_hours = hourly_mood.index.intersection(hourly_var.index)
            if len(common_hours) > 0:
                corr, p_value = stats.pearsonr(
                    hourly_mood[common_hours],
                    hourly_var[common_hours]
                )
                correlations.append({
                    'variable': var,
                    'correlation': corr,
                    'p_value': p_value,
                    'n_samples': len(common_hours)
                })
    
    # Create correlation plot
    corr_df = pd.DataFrame(correlations)
    corr_df['significant'] = corr_df['p_value'] < 0.05
    corr_df = corr_df.sort_values('correlation', key=abs, ascending=False)
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(len(corr_df)), corr_df['correlation'])
    
    # Color bars by significance
    for i, (_, row) in enumerate(corr_df.iterrows()):
        bars[i].set_color('darkblue' if row['significant'] else 'lightblue')
    
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.title('Correlations with Mood\n(After Cleaning)')
    plt.xticks(range(len(corr_df)), corr_df['variable'], rotation=45, ha='right')
    plt.ylabel('Correlation Coefficient')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(CORRELATION_DIR, 'mood_correlations.png'))
    plt.close()
    
    return corr_df

def main():
    """Main analysis function."""
    # Create output directories
    for dir_path in [OUTPUT_DIR, CORRELATION_DIR, TEMPORAL_DIR, APP_DIR]:
        os.makedirs(dir_path, exist_ok=True)
    
    # Load cleaned data
    print("\n=== Loading Cleaned Dataset ===")
    df = pd.read_csv(os.path.join(DATA_DIR, 'dataset_mood_smartphone_cleaned.csv'))
    df['time'] = pd.to_datetime(df['time'], format='mixed')
    
    # Analyze mood patterns
    print("\n=== Analyzing Mood Patterns ===")
    analyze_mood_patterns(df)
    
    # Analyze temporal patterns
    print("\n=== Analyzing Temporal Patterns ===")
    analyze_temporal_patterns(df)
    
    # Analyze app usage patterns
    print("\n=== Analyzing App Usage Patterns ===")
    analyze_app_usage_patterns(df)
    
    # Analyze correlations
    print("\n=== Analyzing Correlations ===")
    corr_df = analyze_mood_correlations(df)
    
    # Print significant correlations
    sig_corr = corr_df[corr_df['significant']].copy()
    print("\nSignificant correlations with mood (p < 0.05):")
    print(sig_corr[['variable', 'correlation', 'p_value', 'n_samples']].to_string())

if __name__ == "__main__":
    main()
