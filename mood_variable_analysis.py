import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from datetime import datetime
import calendar
from sklearn.impute import SimpleImputer
import warnings
import os
warnings.filterwarnings('ignore')

# App category definitions
APP_CATEGORIES = {
    'appCat.builtin': 'System built-in applications (core system functions)',
    'appCat.communication': 'Communication apps (email, messaging, phone)',
    'appCat.entertainment': 'Entertainment apps (media, streaming, content consumption)',
    'appCat.finance': 'Financial applications (banking, budgeting, payments)',
    'appCat.game': 'Gaming applications (mobile games)',
    'appCat.office': 'Productivity apps (documents, spreadsheets, notes)',
    'appCat.other': 'Uncategorized applications',
    'appCat.social': 'Social media and networking apps',
    'appCat.travel': 'Travel-related applications (maps, transport, booking)',
    'appCat.unknown': 'Applications with unknown categories',
    'appCat.utilities': 'Utility applications (calculator, calendar, etc.)',
    'appCat.weather': 'Weather information applications'
}

# App category groupings for analysis
APP_GROUPS = {
    'Productivity': ['appCat.office', 'appCat.utilities', 'appCat.finance'],
    'Entertainment': ['appCat.entertainment', 'appCat.game', 'appCat.social'],
    'Information': ['appCat.weather', 'appCat.travel'],
    'System': ['appCat.builtin'],
    'Communication': ['appCat.communication'],
    'Other': ['appCat.other', 'appCat.unknown']
}

# Define output directories
DATA_DIR = 'data'
OUTPUT_DIR = 'outputs'
DATA_QUALITY_DIR = os.path.join(OUTPUT_DIR, 'data_quality')
TEMPORAL_PATTERNS_DIR = os.path.join(OUTPUT_DIR, 'temporal_patterns')
CORRELATIONS_DIR = os.path.join(OUTPUT_DIR, 'correlations')

# Create directories 
for directory in [DATA_DIR, OUTPUT_DIR, DATA_QUALITY_DIR, TEMPORAL_PATTERNS_DIR, CORRELATIONS_DIR]:
    os.makedirs(directory, exist_ok=True)

def load_and_prepare_data(file_path):
    """Load and prepare the dataset for analysis"""
    print("Loading data...")
    df = pd.read_csv(file_path)
    df['time'] = pd.to_datetime(df['time'])
    return df

def analyze_data_quality(df):
    """Analyze data quality including missing values, outliers, and value ranges"""
    print("\nAnalyzing data quality...")
    
    # Analyze user distribution
    user_counts = df.groupby('variable')['id'].nunique()
    print("\nNumber of unique users per variable:")
    print(user_counts)
    
    # Analyze user ID distribution
    unique_ids = sorted(df['id'].unique())
    print("\nUnique user IDs in the dataset:")
    print(unique_ids)
    print(f"\nTotal number of unique users: {len(unique_ids)}")
    print(f"ID range: {min(unique_ids)} to {max(unique_ids)}")
    
    # Check for missing IDs in the sequence
    if unique_ids[0].startswith('AS14.'):
        missing_ids = [f'AS14.{str(i).zfill(2)}' for i in range(1, 34) 
                      if f'AS14.{str(i).zfill(2)}' not in unique_ids]
        if missing_ids:
            print(f"\nMissing IDs in sequence: {missing_ids}")
    
    # Analyze time spans and sampling intervals
    print("\nAnalyzing temporal coverage...")
    
    # Overall study duration
    study_start = df['time'].min()
    study_end = df['time'].max()
    study_duration = (study_end - study_start).days
    print(f"\nStudy Duration:")
    print(f"Start: {study_start}")
    print(f"End: {study_end}")
    print(f"Total days: {study_duration}")
    
    # Per-user participation duration
    user_spans = df.groupby('id').agg({
        'time': ['min', 'max', 'count']
    })
    user_spans.columns = ['start_time', 'end_time', 'record_count']
    user_spans['duration_days'] = (user_spans['end_time'] - user_spans['start_time']).dt.days
    
    print("\nParticipation duration by user (days):")
    print(user_spans['duration_days'].describe())
    
    # Analyze sampling intervals for key variables
    key_vars = ['mood', 'circumplex.arousal', 'circumplex.valence', 'activity']
    print("\nTypical sampling intervals (hours):")
    
    for var in key_vars:
        var_data = df[df['variable'] == var].copy()
        var_data = var_data.sort_values('time')
        
        # Calculate intervals for each user
        intervals = var_data.groupby('id')['time'].diff().dt.total_seconds() / 3600  # Convert to hours
        
        print(f"\n{var}:")
        print(intervals.describe())
    
    # Plot participation timeline
    plt.figure(figsize=(15, 8))
    for idx, user in enumerate(unique_ids):
        user_data = df[df['id'] == user]
        plt.scatter(user_data['time'], [idx] * len(user_data), 
                   alpha=0.1, s=1, label=user if idx == 0 else "")
    
    plt.yticks(range(len(unique_ids)), unique_ids)
    plt.xlabel('Time')
    plt.ylabel('User ID')
    plt.title('Participation Timeline by User')
    plt.tight_layout()
    plt.savefig(os.path.join(DATA_QUALITY_DIR, 'participation_timeline.png'))
    plt.close()
    
    # Plot mood sampling intervals
    mood_data = df[df['variable'] == 'mood'].copy().sort_values('time')
    mood_intervals = mood_data.groupby('id')['time'].diff().dt.total_seconds() / 3600
    
    plt.figure(figsize=(12, 6))
    sns.histplot(mood_intervals[mood_intervals < 48], bins=50)
    plt.title('Distribution of Mood Sampling Intervals')
    plt.xlabel('Hours between mood reports')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(os.path.join(DATA_QUALITY_DIR, 'mood_sampling_intervals.png'))
    plt.close()
    
    # Create a summary DataFrame
    summary = pd.DataFrame()
    variables = df['variable'].unique()
    
    for var in variables:
        var_data = df[df['variable'] == var]
        values = var_data['value']
        
        stats_dict = {
            'total_records': len(var_data),
            'unique_users': var_data['id'].nunique(),
            'avg_records_per_user': len(var_data) / var_data['id'].nunique(),
            'missing': values.isnull().sum(),
            'missing_pct': (values.isnull().sum() / len(values)) * 100
        }
        
        # Add specific stats based on variable type
        if var in ['call', 'sms']:  # Binary variables
            stats_dict.update({
                'active_events': values.sum(),  # Number of 1s (calls/SMS)
                'active_pct': (values == 1).mean() * 100,  # Percentage of active events
                'users_with_activity': var_data[var_data['value'] == 1]['id'].nunique(),
                'avg_events_per_user': values.sum() / var_data['id'].nunique()
            })
        elif var.startswith('appCat') or var == 'screen':  # Time duration variables
            stats_dict.update({
                'mean_duration': values.mean(),
                'median_duration': values.median(),
                'std_duration': values.std(),
                'max_duration': values.max(),
                'total_duration': values.sum(),
                'avg_daily_duration': values.sum() / var_data['id'].nunique() / len(var_data['time'].dt.date.unique())
            })
        else:  # Scale variables (mood, arousal, valence, activity)
            stats_dict.update({
                'mean': values.mean(),
                'median': values.median(),
                'std': values.std(),
                'min': values.min(),
                'max': values.max(),
                'zeros_pct': (values == 0).mean() * 100
            })
        
        summary = pd.concat([summary, pd.DataFrame([stats_dict], index=[var])])
    
    # Save detailed summary to CSV
    summary.to_csv(os.path.join(DATA_QUALITY_DIR, 'data_quality_summary.csv'))
    
    # Plot missing data patterns by user
    plt.figure(figsize=(15, 8))
    pivot_data = df.pivot_table(
        values='value',
        index='id',
        columns='variable',
        aggfunc='count'
    ).isnull()
    
    sns.heatmap(pivot_data, 
                cmap='YlOrRd',
                cbar_kws={'label': 'Missing'},
                xticklabels=True)
    plt.title('Missing Data Patterns by User')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(DATA_QUALITY_DIR, 'missing_data_patterns_by_user.png'))
    plt.close()
    
    # Plot binary variable patterns (calls and SMS)
    for var in ['call', 'sms']:
        var_data = df[df['variable'] == var].copy()
        var_data['date'] = var_data['time'].dt.date
        var_data['hour'] = var_data['time'].dt.hour
        
        # Create hourly activity heatmap by user
        pivot_data = var_data.pivot_table(
            values='value',
            index='id',
            columns='hour',
            aggfunc='sum'
        ).fillna(0)
        
        plt.figure(figsize=(15, 8))
        sns.heatmap(pivot_data,
                    cmap='YlOrRd',
                    cbar_kws={'label': f'{var.upper()} Count'})
        plt.title(f'{var.upper()} Activity Pattern by User and Hour')
        plt.xlabel('Hour of Day')
        plt.ylabel('User ID')
        plt.tight_layout()
        plt.savefig(os.path.join(DATA_QUALITY_DIR, f'{var}_patterns_by_user.png'))
        plt.close()
    
    return summary

# Import data cleaning functions
from data_cleaning import remove_outliers, handle_missing_data

def plot_daily_patterns_2x2(df):
    """Create a 2x2 plot of daily patterns for mood, arousal, valence, and activity"""
    plt.figure(figsize=(15, 12))
    variables = ['mood', 'circumplex.arousal', 'circumplex.valence', 'activity']
    titles = ['Mood', 'Arousal', 'Valence', 'Activity']
    
    for idx, (var, title) in enumerate(zip(variables, titles)):
        plt.subplot(2, 2, idx + 1)
        
        # Get data for this variable
        var_data = df[df['variable'] == var].copy()
        var_data['hour'] = var_data['time'].dt.hour
        
        # Calculate mean and confidence intervals
        stats = var_data.groupby('hour')['value'].agg(['mean', 'std', 'count'])
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
    plt.savefig(os.path.join(TEMPORAL_PATTERNS_DIR, 'daily_patterns_2x2.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_weekly_patterns_2x2(df):
    """Create a 2x2 plot of weekly patterns for mood, arousal, valence, and activity"""
    plt.figure(figsize=(15, 12))
    variables = ['mood', 'circumplex.arousal', 'circumplex.valence', 'activity']
    titles = ['Mood', 'Arousal', 'Valence', 'Activity']
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
    for idx, (var, title) in enumerate(zip(variables, titles)):
        plt.subplot(2, 2, idx + 1)
        
        # Get data for this variable
        var_data = df[df['variable'] == var].copy()
        var_data['day_of_week'] = var_data['time'].dt.dayofweek
        
        # Calculate mean and confidence intervals
        stats = var_data.groupby('day_of_week')['value'].agg(['mean', 'std', 'count'])
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
    plt.savefig(os.path.join(TEMPORAL_PATTERNS_DIR, 'weekly_patterns_2x2.png'), dpi=300, bbox_inches='tight')
    plt.close()

def analyze_temporal_patterns(df):
    """Analyze temporal patterns in the data, accounting for user differences"""
    print("\nAnalyzing temporal patterns...")
    
    # Create 2x2 plots for daily and weekly patterns
    plot_daily_patterns_2x2(df)
    plot_weekly_patterns_2x2(df)
    print("Created daily and weekly pattern plots in outputs/temporal_patterns/")
    
    # Add time-based features
    df['hour'] = df['time'].dt.hour
    df['day_of_week'] = df['time'].dt.dayofweek
    df['weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    df['date'] = df['time'].dt.date
    
    # Separate analysis for different types of variables
    scale_vars = ['mood', 'circumplex.arousal', 'circumplex.valence', 'activity']
    binary_vars = ['call', 'sms']
    time_vars = [var for var in df['variable'].unique() if var.startswith('appCat') or var == 'screen']
    
    # 1. Analyze scale variables
    for var in scale_vars:
        var_data = df[df['variable'] == var].copy()
        
        # Time of day analysis with user variation
        plt.figure(figsize=(15, 6))
        sns.boxplot(data=var_data, x='hour', y='value', hue='id')
        plt.title(f'{var} by Hour of Day (Per User)')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
        plt.tight_layout()
        plt.savefig(os.path.join(TEMPORAL_PATTERNS_DIR, f'{var}_by_hour_per_user.png'))
        plt.close()
        
        # Aggregate view without user separation
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=var_data, x='hour', y='value')
        plt.title(f'{var} by Hour of Day (All Users)')
        plt.savefig(os.path.join(TEMPORAL_PATTERNS_DIR, f'{var}_by_hour.png'))
        plt.close()
        
        # Weekly patterns
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=var_data, x='day_of_week', y='value')
        plt.xticks(range(7), calendar.day_abbr)
        plt.title(f'{var} by Day of Week')
        plt.savefig(os.path.join(TEMPORAL_PATTERNS_DIR, f'{var}_by_day.png'))
        plt.close()
    
    # 2. Analyze binary variables (calls and SMS)
    for var in binary_vars:
        var_data = df[df['variable'] == var].copy()
        
        # Hourly activity patterns
        hourly_counts = var_data.groupby(['id', 'hour'])['value'].sum().reset_index()
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=hourly_counts, x='hour', y='value')
        plt.title(f'{var.upper()} Frequency by Hour')
        plt.ylabel('Number of Events')
        plt.savefig(os.path.join(TEMPORAL_PATTERNS_DIR, f'{var}_frequency_by_hour.png'))
        plt.close()
        
        # Daily activity patterns
        daily_counts = var_data.groupby(['id', 'date'])['value'].sum().reset_index()
        plt.figure(figsize=(12, 6))
        sns.histplot(data=daily_counts, x='value', bins=20)
        plt.title(f'Distribution of Daily {var.upper()} Events')
        plt.xlabel('Number of Events per Day')
        plt.savefig(os.path.join(TEMPORAL_PATTERNS_DIR, f'{var}_daily_distribution.png'))
        plt.close()
    
    # 3. Analyze time duration variables
    for var in time_vars:
        var_data = df[df['variable'] == var].copy()
        
        # Average duration by hour
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=var_data, x='hour', y='value')
        plt.title(f'{var} Usage Duration by Hour')
        plt.ylabel('Duration (seconds)')
        plt.savefig(os.path.join(TEMPORAL_PATTERNS_DIR, f'{var}_duration_by_hour.png'))
        plt.close()
        
        # Daily usage patterns
        daily_usage = var_data.groupby(['id', 'date'])['value'].sum().reset_index()
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=daily_usage, y='value')
        plt.title(f'Distribution of Daily {var} Usage')
        plt.ylabel('Total Daily Duration (seconds)')
        plt.savefig(os.path.join(TEMPORAL_PATTERNS_DIR, f'{var}_daily_usage.png'))
        plt.close()
    
    return df

def analyze_app_categories(df):
    """Analyze app category usage patterns and documentation.
    
    This function provides detailed analysis of app category usage, including:
    1. Usage statistics per category
    2. User engagement patterns
    3. Temporal distribution of usage
    4. Category correlations
    
    Args:
        df (pd.DataFrame): The dataset containing app usage information
        
    Returns:
        pd.DataFrame: Summary statistics for each app category
    """
    print("\nAnalyzing app categories...")
    
    # Filter for app categories only
    app_data = df[df['variable'].str.startswith('appCat')].copy()
    
    # Basic statistics per category
    stats = []
    for category, description in APP_CATEGORIES.items():
        cat_data = app_data[app_data['variable'] == category]
        active_users = cat_data['id'].nunique()
        total_duration = cat_data['value'].sum()
        avg_duration_per_user = total_duration / active_users if active_users > 0 else 0
        
        stats.append({
            'category': category,
            'description': description,
            'total_duration_hours': total_duration / 3600,  # Convert seconds to hours
            'active_users': active_users,
            'avg_duration_per_user_hours': avg_duration_per_user / 3600,
            'records': len(cat_data)
        })
    
    stats_df = pd.DataFrame(stats)
    stats_df = stats_df.sort_values('total_duration_hours', ascending=False)
    
    # Save statistics to CSV
    stats_df.to_csv(os.path.join(APP_ANALYSIS_DIR, 'app_category_statistics.csv'), index=False)
    
    # Plot total usage by category
    plt.figure(figsize=(12, 6))
    sns.barplot(data=stats_df, x='category', y='total_duration_hours')
    plt.xticks(rotation=45, ha='right')
    plt.title('Total Usage Duration by App Category')
    plt.xlabel('App Category')
    plt.ylabel('Total Duration (hours)')
    plt.tight_layout()
    plt.savefig(os.path.join(APP_ANALYSIS_DIR, 'app_category_usage.png'))
    plt.close()
    
    # Plot user engagement heatmap
    daily_usage = app_data.pivot_table(
        index='id',
        columns='variable',
        values='value',
        aggfunc='sum'
    ).fillna(0) / 3600  # Convert to hours
    
    plt.figure(figsize=(15, 8))
    sns.heatmap(daily_usage, cmap='YlOrRd')
    plt.title('App Usage Patterns by User')
    plt.xlabel('App Category')
    plt.ylabel('User ID')
    plt.tight_layout()
    plt.savefig(os.path.join(APP_ANALYSIS_DIR, 'user_app_usage_patterns.png'))
    plt.close()
    
    # Calculate and plot category correlations
    correlations = daily_usage.corr()
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlations, annot=True, cmap='coolwarm', center=0, fmt='.2f')
    plt.title('App Category Usage Correlations')
    plt.tight_layout()
    plt.savefig(os.path.join(APP_ANALYSIS_DIR, 'app_category_correlations.png'))
    plt.close()
    
    return stats_df

def analyze_mood_correlations(df):
    """Analyze correlations between mood and other variables using time-windowed aggregation"""
    print("\nAnalyzing correlations with mood...")
    
    # Get unique variables excluding mood
    variables = [var for var in df['variable'].unique() if var != 'mood']
    
    # Function to aggregate data in time windows
    def aggregate_time_window(group):
        return group['value'].mean()
    
    # Get mood data
    mood_data = df[df['variable'] == 'mood'].copy()
    mood_data['time_window'] = mood_data['time'].dt.floor('H')  # Hourly windows
    mood_agg = mood_data.groupby('time_window').apply(aggregate_time_window).to_frame('mood')
    
    # Calculate correlations for each variable with mood
    correlations = []
    for var in variables:
        # Get variable data
        var_data = df[df['variable'] == var].copy()
        var_data['time_window'] = var_data['time'].dt.floor('H')
        var_agg = var_data.groupby('time_window').apply(aggregate_time_window).to_frame(var)
        
        # Merge and calculate correlation
        merged = pd.merge(mood_agg, var_agg, left_index=True, right_index=True, how='inner')
        if len(merged) > 0:  # Only calculate if we have matching data
            corr = merged['mood'].corr(merged[var])
            sample_size = len(merged)
            # Calculate p-value
            t_stat = corr * np.sqrt((sample_size-2)/(1-corr**2))
            p_value = 2 * (1 - stats.t.cdf(abs(t_stat), sample_size-2))
            
            correlations.append({
                'variable': var,
                'correlation': corr,
                'sample_size': sample_size,
                'p_value': p_value
            })
    
    # Create correlation dataframe and sort
    corr_df = pd.DataFrame(correlations)
    corr_df['significant'] = corr_df['p_value'] < 0.05
    corr_df = corr_df.sort_values('correlation', ascending=False)
    
    # Plot correlations
    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(len(corr_df)), corr_df['correlation'])
    
    # Color bars based on significance
    for i, (significant, bar) in enumerate(zip(corr_df['significant'], bars)):
        bar.set_color('darkblue' if significant else 'lightgray')
    
    plt.xticks(range(len(corr_df)), corr_df['variable'], rotation=45, ha='right')
    plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    plt.title('Correlations with Mood (Dark Blue = Statistically Significant)')
    plt.ylabel('Correlation Coefficient')
    plt.tight_layout()
    plt.savefig(os.path.join(CORRELATIONS_DIR, 'mood_correlations.png'))
    plt.close()
    
    # Create detailed correlation plots for top significant correlations
    significant_vars = corr_df[corr_df['significant']]['variable'].head(5)
    for var in significant_vars:
        var_data = df[df['variable'] == var].copy()
        var_data['time_window'] = var_data['time'].dt.floor('H')
        var_agg = var_data.groupby('time_window').apply(aggregate_time_window).to_frame(var)
        
        merged = pd.merge(mood_agg, var_agg, left_index=True, right_index=True, how='inner')
        
        plt.figure(figsize=(8, 6))
        plt.scatter(merged[var], merged['mood'], alpha=0.5)
        plt.xlabel(var)
        plt.ylabel('Mood')
        plt.title(f'Mood vs {var}')
        
        # Add trend line (with error handling)
        try:
            # Remove any infinite or NaN values
            mask = np.isfinite(merged[var]) & np.isfinite(merged['mood'])
            if mask.any():
                z = np.polyfit(merged[var][mask], merged['mood'][mask], 1)
                p = np.poly1d(z)
                x_range = np.linspace(merged[var][mask].min(), merged[var][mask].max(), 100)
                plt.plot(x_range, p(x_range), "r--", alpha=0.8)
        except Exception as e:
            print(f"Warning: Could not add trend line for {var}: {str(e)}")
        
        plt.tight_layout()
        plt.savefig(os.path.join(CORRELATIONS_DIR, f'mood_vs_{var.replace(".", "_")}.png'))
        plt.close()
    
    print("\nCorrelations with mood (sorted by absolute correlation):")
    corr_df['abs_corr'] = abs(corr_df['correlation'])
    print(corr_df.sort_values('abs_corr', ascending=False)[["variable", "correlation", "p_value", "significant", "sample_size"]])
    
    return corr_df

def analyze_daily_patterns(df):
    """Analyze daily patterns of variables"""
    print("\nAnalyzing daily patterns...")
    
    # Add hour
    df['hour'] = df['time'].dt.hour
    
    # Sample data for faster plotting (max 1000 points per variable)
    df_sampled = df.groupby('variable').apply(
        lambda x: x.sample(n=min(1000, len(x)), random_state=42)
    ).reset_index(drop=True)
    
    # Create subplots for each variable
    variables = df_sampled['variable'].unique()
    fig, axes = plt.subplots(len(variables), 1, figsize=(10, 3*len(variables)))
    
    if len(variables) == 1:
        axes = [axes]
    
    for i, var in enumerate(variables):
        var_data = df_sampled[df_sampled['variable'] == var]
        
        # Calculate hourly means and confidence intervals
        hourly_stats = var_data.groupby('hour')['value'].agg(['mean', 'std', 'count']).reset_index()
        hourly_stats['ci'] = 1.96 * hourly_stats['std'] / np.sqrt(hourly_stats['count'])
        
        # Plot
        axes[i].plot(hourly_stats['hour'], hourly_stats['mean'], '-o')
        axes[i].fill_between(
            hourly_stats['hour'],
            hourly_stats['mean'] - hourly_stats['ci'],
            hourly_stats['mean'] + hourly_stats['ci'],
            alpha=0.2
        )
        axes[i].set_title(f'Daily Pattern of {var}')
        axes[i].set_xlabel('Hour of Day')
        axes[i].set_ylabel('Value')
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('daily_patterns.png')
    plt.close()

def analyze_weekly_patterns(df):
    """Analyze weekly patterns of variables"""
    print("\nAnalyzing weekly patterns...")
    
    # Add day of week
    df['day_of_week'] = df['time'].dt.dayofweek
    
    # Sample data for faster plotting
    df_sampled = df.groupby('variable').apply(
        lambda x: x.sample(n=min(1000, len(x)), random_state=42)
    ).reset_index(drop=True)
    
    # Create subplots
    variables = df_sampled['variable'].unique()
    fig, axes = plt.subplots(len(variables), 1, figsize=(10, 3*len(variables)))
    
    if len(variables) == 1:
        axes = [axes]
    
    for i, var in enumerate(variables):
        var_data = df_sampled[df_sampled['variable'] == var]
        
        # Calculate daily means and confidence intervals
        daily_stats = var_data.groupby('day_of_week')['value'].agg(['mean', 'std', 'count']).reset_index()
        daily_stats['ci'] = 1.96 * daily_stats['std'] / np.sqrt(daily_stats['count'])
        
        # Plot
        axes[i].plot(daily_stats['day_of_week'], daily_stats['mean'], '-o')
        axes[i].fill_between(
            daily_stats['day_of_week'],
            daily_stats['mean'] - daily_stats['ci'],
            daily_stats['mean'] + daily_stats['ci'],
            alpha=0.2
        )
        axes[i].set_title(f'Weekly Pattern of {var}')
        axes[i].set_xlabel('Day of Week')
        axes[i].set_ylabel('Value')
        axes[i].set_xticks(range(7))
        axes[i].set_xticklabels([calendar.day_abbr[x] for x in range(7)])
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('weekly_patterns.png')
    plt.close()

def main():
    # File path
    file_path = os.path.join(DATA_DIR, 'dataset_mood_smartphone.csv')
    
    # 1. Load data
    print("\n=== Step 1: Loading and Preparing Data ===")
    df = load_and_prepare_data(file_path)
    
    # 2. Analyze data quality
    print("\n=== Step 2: Analyzing Data Quality ===")
    quality_summary = analyze_data_quality(df)
    print("\nData Quality Summary:")
    print(quality_summary)
    
    # 3. Remove outliers
    print("\n=== Step 3: Removing Outliers ===")
    df_clean = remove_outliers(df)
    
    # 4. Handle missing data
    print("\n=== Step 4: Handling Missing Data ===")
    df_clean = handle_missing_data(df_clean)
    
    # Save cleaned dataset
    os.makedirs('data', exist_ok=True)
    df_clean.to_csv('data/cleaned_dataset.csv', index=False)
    print("\nSaved cleaned dataset to 'data/cleaned_dataset.csv'")
    
    # 5. Analyze temporal patterns
    print("\n=== Step 5: Analyzing Temporal Patterns ===")
    df_clean = analyze_temporal_patterns(df_clean)
    
    # 6. Analyze correlations with mood
    print("\n=== Step 6: Analyzing Variable Correlations ===")
    correlations = analyze_mood_correlations(df_clean)
    
    # 7. Analyze daily and weekly patterns
    print("\n=== Step 7: Analyzing Daily and Weekly Patterns ===")
    analyze_daily_patterns(df_clean)
    analyze_weekly_patterns(df_clean)
    
    print("\nAnalysis complete! The following files have been created:")
    print("\nData Quality Outputs (in outputs/data_quality/):")
    print("1. data_quality_summary.csv - Summary of data quality metrics")
    print("2. missing_data_patterns.png - Visualization of missing data patterns")
    
    print("\nTemporal Pattern Outputs (in outputs/temporal_patterns/):")
    print("3. [variable]_by_hour.png - Hourly patterns for key variables")
    print("4. [variable]_by_day.png - Daily patterns for key variables")
    print("5. daily_patterns.png - Daily patterns of each variable")
    print("6. weekly_patterns.png - Weekly patterns of each variable")
    
    print("\nCorrelation Outputs (in outputs/correlations/):")
    print("7. mood_correlations.png - Correlations between variables and mood")
    print("8. mood_vs_[variable].png - Detailed correlation plots for significant variables")
    
    # Print top correlations with mood
    print("\nTop 5 variables most correlated with mood:")
    print(correlations.head())

if __name__ == "__main__":
    main()
