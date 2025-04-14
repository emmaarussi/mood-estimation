import pandas as pd
import numpy as np
from datetime import timedelta
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def remove_outliers(df):
    """Remove extreme and incorrect values using statistical and domain-based approaches.
    
    Approach:
    1. Statistical outliers: Use IQR method with variable thresholds
    2. Domain-based validation: Apply known valid ranges for each variable
    
    
    Args:
        df (pd.DataFrame): Input dataframe with columns ['id', 'time', 'variable', 'value']
    
    Returns:
        pd.DataFrame: Clean dataset with outliers removed
    """
    logger.info("Starting outlier removal process...")
    df_clean = df.copy()
    total_outliers = 0
    
    # Define valid ranges for each variable type
    valid_ranges = {
        'mood': (1, 10),  # 1-10 scale
        'circumplex.arousal': (-2, 2),  # -2 to 2 scale
        'circumplex.valence': (-2, 2),  # -2 to 2 scale
        'activity': (0, 1),  # 0-1 scale
        'screen': (0, 7200),  # Max 2 hours per event
        'call': (0, 1),  # Binary
        'sms': (0, 1)   # Binary
    }
    
    # Process each variable
    for var in df_clean['variable'].unique():
        mask = df_clean['variable'] == var
        values = df_clean.loc[mask, 'value']
        var_outliers = 0
        
        if var in valid_ranges:
            # Apply domain-based validation
            min_val, max_val = valid_ranges[var]
            invalid_mask = (values < min_val) | (values > max_val)
            n_invalid = invalid_mask.sum()
            if n_invalid > 0:
                logger.info(f"Removed {n_invalid} invalid values from {var} (outside [{min_val}, {max_val}])")
                df_clean = df_clean.loc[~(mask & invalid_mask)]
                var_outliers += n_invalid
        
        if var not in ['call', 'sms']:  # Skip binary variables
            # Apply IQR method with variable thresholds
            Q1 = values.quantile(0.25)
            Q3 = values.quantile(0.75)
            IQR = Q3 - Q1
            
            # Use different thresholds based on variable type
            if var.startswith('appCat') or var == 'screen':
                threshold = 5  # More permissive for usage times
            else:
                threshold = 3  # Standard for psychological measures
            
            lower = Q1 - threshold * IQR
            upper = Q3 + threshold * IQR
            
            outlier_mask = (values < lower) | (values > upper)
            n_outliers = outlier_mask.sum()
            if n_outliers > 0:
                logger.info(f"Removed {n_outliers} statistical outliers from {var}")
                df_clean = df_clean.loc[~(mask & outlier_mask)]
                var_outliers += n_outliers
        
        total_outliers += var_outliers
    
    logger.info(f"Total outliers removed: {total_outliers} ({total_outliers/len(df)*100:.2f}% of data)")
    return df_clean

def analyze_mood_data_patterns(df):
    """Analyze patterns in mood data recording, focusing on temporal gaps and recording frequency.
    
    Args:
        df (pd.DataFrame): Input dataframe with columns ['id', 'time', 'variable', 'value']
    
    Returns:
        dict: Analysis results including recording patterns and gaps
    """
    logger.info("Analyzing mood data patterns...")
    
    # Filter for mood data only
    mood_data = df[df['variable'] == 'mood'].copy()
    mood_data = mood_data.sort_values(['id', 'time'])
    
    analysis = {}
    
    # Basic statistics
    n_users = mood_data['id'].nunique()
    total_records = len(mood_data)
    time_span = mood_data['time'].max() - mood_data['time'].min()
    avg_daily_records = total_records / time_span.days
    
    analysis['basic_stats'] = {
        'total_users': n_users,
        'total_records': total_records,
        'time_span_days': time_span.days,
        'avg_records_per_day': avg_daily_records
    }
    
    # Analyze recording patterns per user
    user_patterns = []
    total_time_covered = pd.Timedelta(0)
    total_time_gaps = pd.Timedelta(0)
    gap_distribution = {
        '12-24h': 0,
        '24-48h': 0,
        '48h+': 0
    }
    
    for user_id in mood_data['id'].unique():
        user_data = mood_data[mood_data['id'] == user_id]
        time_diffs = user_data['time'].diff()
        
        # Calculate user's time coverage
        user_span = user_data['time'].max() - user_data['time'].min()
        total_time_covered += user_span
        
        # Analyze gaps in detail
        gaps = time_diffs[time_diffs > pd.Timedelta(hours=12)]
        total_gap_time = gaps.sum() if not gaps.empty else pd.Timedelta(0)
        total_time_gaps += total_gap_time
        
        # Categorize gaps by duration
        for gap in gaps:
            if gap <= pd.Timedelta(hours=24):
                gap_distribution['12-24h'] += 1
            elif gap <= pd.Timedelta(hours=48):
                gap_distribution['24-48h'] += 1
            else:
                gap_distribution['48h+'] += 1
        
        # Calculate active hours (when users typically record)
        active_hours = user_data['time'].dt.hour.value_counts()
        morning_records = active_hours[6:12].sum() if not active_hours.empty else 0
        afternoon_records = active_hours[12:18].sum() if not active_hours.empty else 0
        evening_records = active_hours[18:24].sum() if not active_hours.empty else 0
        night_records = active_hours[0:6].sum() if not active_hours.empty else 0
        
        user_patterns.append({
            'user_id': user_id,
            'n_records': len(user_data),
            'avg_records_per_day': len(user_data) / time_span.days,
            'time_coverage': user_span.total_seconds() / (24 * 3600),  # in days
            'total_gap_time': total_gap_time.total_seconds() / (24 * 3600),  # in days
            'recording_distribution': {
                'morning': morning_records,
                'afternoon': afternoon_records,
                'evening': evening_records,
                'night': night_records
            },
            'n_gaps': len(gaps),
            'avg_gap_hours': gaps.dt.total_seconds().mean() / 3600 if len(gaps) > 0 else 0,
            'max_gap_hours': gaps.dt.total_seconds().max() / 3600 if len(gaps) > 0 else 0
        })
    
    analysis['user_patterns'] = user_patterns
    
    # Calculate coverage statistics
    total_possible_time = n_users * time_span
    coverage_percentage = (total_time_covered - total_time_gaps) / total_possible_time * 100
    
    # Summary statistics across users
    records_per_day = [p['avg_records_per_day'] for p in user_patterns]
    gaps_per_user = [p['n_gaps'] for p in user_patterns]
    gap_lengths = [p['avg_gap_hours'] for p in user_patterns if p['avg_gap_hours'] > 0]
    
    # Analyze recording patterns
    all_recording_dist = pd.DataFrame([p['recording_distribution'] for p in user_patterns])
    peak_recording_time = all_recording_dist.mean().idxmax()
    
    analysis['summary'] = {
        'records_per_day': {
            'mean': np.mean(records_per_day),
            'std': np.std(records_per_day),
            'min': np.min(records_per_day),
            'max': np.max(records_per_day)
        },
        'gaps': {
            'total_gaps': sum(gaps_per_user),
            'distribution': gap_distribution,
            'avg_gaps_per_user': np.mean(gaps_per_user),
            'avg_gap_length_hours': np.mean(gap_lengths) if gap_lengths else 0
        },
        'coverage': {
            'percentage': coverage_percentage,
            'total_gap_days': total_time_gaps.total_seconds() / (24 * 3600)
        },
        'recording_patterns': {
            'peak_time': peak_recording_time,
            'distribution': all_recording_dist.mean().to_dict()
        }
    }
    
    # Log key findings
    logger.info(f"Found {total_records:,} mood records from {n_users} users over {time_span.days} days")
    logger.info(f"Data coverage: {coverage_percentage:.1f}% of total possible recording time")
    logger.info(f"Users record {analysis['summary']['records_per_day']['mean']:.1f} ± {analysis['summary']['records_per_day']['std']:.1f} times per day")
    logger.info(f"Gap distribution: {gap_distribution['12-24h']} gaps 12-24h, {gap_distribution['24-48h']} gaps 24-48h, {gap_distribution['48h+']} gaps >48h")
    logger.info(f"Peak recording time: {peak_recording_time}")
    
    return analysis

def handle_gaps(df):
    """Process temporal gaps in the dataset and add gap-related features
    
    Args:
        df (pd.DataFrame): Input dataframe
    
    Returns:
        pd.DataFrame: DataFrame with gap features added
    """
    logger.info("Processing temporal gaps...")
    
    # Create a copy to avoid modifying the original dataframe
    df = df.copy()
    
    # Sort by user and time
    df = df.sort_values(['id', 'time'])
    
    # Calculate time since last record for each user
    df['time_since_last'] = df.groupby(['id', 'variable'])['time'].diff()
    
    # Convert to hours
    df['hours_since_last'] = df['time_since_last'].dt.total_seconds() / 3600
    
    # Categorize gaps
    df['gap_category'] = pd.cut(df['hours_since_last'],
                               bins=[-float('inf'), 12, 24, 48, float('inf')],
                               labels=['normal', '12-24h', '24-48h', '48h+'])
    
    # Calculate running statistics of gaps per user
    gap_stats = df[df['variable'] == 'mood'].groupby('id').agg({
        'hours_since_last': ['mean', 'std', 'max']
    }).reset_index()
    
    gap_stats.columns = ['id', 'avg_gap_hours', 'gap_std_hours', 'max_gap_hours']
    
    # Merge gap statistics back to the main dataframe
    df = df.merge(gap_stats, on='id', how='left')
    
    # Log gap statistics
    gap_counts = df[df['variable'] == 'mood']['gap_category'].value_counts()
    logger.info("Gap distribution:")
    for category, count in gap_counts.items():
        if category != 'normal':
            logger.info(f"  {category}: {count} gaps")
    
    return df

def clean_dataset(input_file, output_file=None):
    """Clean the dataset by removing outliers and analyzing temporal patterns
    
    Args:
        input_file (str): Path to input CSV file
        output_file (str, optional): Path to save cleaned CSV file. If None, will use 'cleaned_' prefix
    
    Returns:
        tuple: (cleaned_dataset, analysis_results)
            - cleaned_dataset (pd.DataFrame): Cleaned dataset with gap features
            - analysis_results (dict): Analysis results including mood patterns and gaps
    """
    logger.info(f"Loading dataset from {input_file}")
    df = pd.read_csv(input_file)
    df['time'] = pd.to_datetime(df['time'])
    
    # Analyze mood data patterns
    mood_analysis = analyze_mood_data_patterns(df)
    
    # Remove outliers
    df_clean = remove_outliers(df)
    
    # Process gaps and add gap-related features
    df_clean = handle_gaps(df_clean)
    
    # Fill missing values for circumplex variables with user means
    for var in ['circumplex.arousal', 'circumplex.valence']:
        missing = df_clean[df_clean['variable'] == var]['value'].isna().sum()
        if missing > 0:
            user_means = df_clean[df_clean['variable'] == var].groupby('id')['value'].transform('mean')
            df_clean.loc[(df_clean['variable'] == var) & (df_clean['value'].isna()), 'value'] = user_means
            logger.info(f"Filled {missing} missing values with user means for {var}")
    
    # Save cleaned dataset
    if output_file is None:
        output_file = input_file.replace('.csv', '_cleaned.csv')
    df_clean.to_csv(output_file, index=False)
    logger.info(f"Saved cleaned dataset to {output_file}")
    
    # Combine analysis results
    analysis_results = {
        'mood_patterns': mood_analysis,
        'gap_features': {
            'columns_added': ['time_since_last', 'hours_since_last', 'gap_category',
                            'avg_gap_hours', 'gap_std_hours', 'max_gap_hours']
        }
    }
    
    return df_clean, analysis_results

def analyze_dataset_structure(df):
    """Analyze the structure and characteristics of the dataset to determine ML model suitability.
    
    This function examines:
    1. Data balance/imbalance
    2. Temporal characteristics
    3. Feature distributions
    4. Missing patterns
    5. Suitable and unsuitable ML models
    
    Args:
        df (pd.DataFrame): Input dataframe with columns ['id', 'time', 'variable', 'value']
    
    Returns:
        dict: Analysis results and recommendations
    """
    logger.info("Analyzing dataset structure and ML model suitability...")
    analysis = {}
    
    # 1. Basic dataset statistics
    n_users = df['id'].nunique()
    n_variables = df['variable'].nunique()
    time_span = df['time'].max() - df['time'].min()
    analysis['basic_stats'] = {
        'n_users': n_users,
        'n_variables': n_variables,
        'time_span_days': time_span.days,
        'total_records': len(df)
    }
    
    # 2. Check data balance
    user_counts = df.groupby('id').size()
    variable_counts = df.groupby(['id', 'variable']).size().unstack(fill_value=0)
    
    analysis['balance'] = {
        'records_per_user': {
            'min': user_counts.min(),
            'max': user_counts.max(),
            'mean': user_counts.mean(),
            'std': user_counts.std()
        },
        'coefficient_of_variation': user_counts.std() / user_counts.mean(),
        'is_balanced': (user_counts.std() / user_counts.mean()) < 0.5
    }
    
    # 3. Temporal characteristics
    time_diffs = df.groupby(['id', 'variable'])['time'].diff()
    sampling_stats = {
        'median_sampling_interval': time_diffs.median(),
        'sampling_regularity': time_diffs.std() / time_diffs.mean(),
        'has_regular_sampling': (time_diffs.std() / time_diffs.mean() < 0.5)
    }
    analysis['temporal'] = sampling_stats
    
    # 4. Feature analysis
    feature_stats = {}
    for var in df['variable'].unique():
        var_data = df[df['variable'] == var]['value']
        feature_stats[var] = {
            'mean': var_data.mean(),
            'std': var_data.std(),
            'missing_pct': (var_data.isnull().sum() / len(var_data)) * 100,
            'unique_values': var_data.nunique()
        }
    analysis['features'] = feature_stats
    

    
    logger.info("Dataset structure analysis complete")
    return analysis

def save_analysis_report(analysis, base_filename='analysis_report'):
    """Save the analysis report to both text and markdown files.
    
    Args:
        analysis (dict): Analysis results from analyze_dataset_structure
        base_filename (str): Base name for the output files (without extension)
    """
    # Save as text file
    txt_content = ["=== Dataset Structure Analysis Report ==="]
    
    # Basic Statistics
    txt_content.extend([
        "\nBasic Statistics:",
        f"- Number of users: {analysis['basic_stats']['n_users']}",
        f"- Number of variables: {analysis['basic_stats']['n_variables']}",
        f"- Time span: {analysis['basic_stats']['time_span_days']} days",
        f"- Total records: {analysis['basic_stats']['total_records']:,}"
    ])
    
    # Data Balance
    balance = analysis['balance']['records_per_user']
    txt_content.extend([
        "\nData Balance:",
        f"- Records per user: {balance['mean']:.1f} ± {balance['std']:.1f}",
        f"- Min records: {balance['min']}, Max records: {balance['max']}",
        f"- Data is {'balanced' if analysis['balance']['is_balanced'] else 'imbalanced'}"
    ])
    
    # Temporal Characteristics
    txt_content.extend([
        "\nTemporal Characteristics:",
        f"- Median sampling interval: {analysis['temporal']['median_sampling_interval']}",
        f"- Sampling is {'regular' if analysis['temporal']['has_regular_sampling'] else 'irregular'}"
    ])
    
    # Feature Statistics
    txt_content.append("\nFeature Statistics:")
    for var, stats in analysis['features'].items():
        txt_content.extend([
            f"\n{var}:",
            f"  - Mean: {stats['mean']:.2f} ± {stats['std']:.2f}",
            f"  - Missing: {stats['missing_pct']:.1f}%",
            f"  - Unique values: {stats['unique_values']}"
        ])
    
    # ML Models
    txt_content.append("\nRecommended ML Models:")
    for model in analysis['ml_recommendations']['suitable_models']:
        txt_content.append(f"✓ {model}")
    
    txt_content.append("\nNot Recommended ML Models:")
    for model in analysis['ml_recommendations']['unsuitable_models']:
        txt_content.append(f"✗ {model}")
    
    if analysis['ml_recommendations']['notes']:
        txt_content.append("\nImportant Notes:")
        for note in analysis['ml_recommendations']['notes']:
            txt_content.append(f"! {note}")
    
    # Save text file
    txt_file = f"{base_filename}.txt"
    with open(txt_file, 'w') as f:
        f.write('\n'.join(txt_content))
    logger.info(f"Saved text report to {txt_file}")
    
    # Create markdown content with better formatting
    md_content = ["# Dataset Structure Analysis Report", ""]
    
    # Basic Statistics
    md_content.extend([
        "## Basic Statistics",
        f"* **Number of users:** {analysis['basic_stats']['n_users']}",
        f"* **Number of variables:** {analysis['basic_stats']['n_variables']}",
        f"* **Time span:** {analysis['basic_stats']['time_span_days']} days",
        f"* **Total records:** {analysis['basic_stats']['total_records']:,}",
        ""
    ])
    
    # Data Balance
    md_content.extend([
        "## Data Balance",
        f"* **Records per user:** {balance['mean']:.1f} ± {balance['std']:.1f}",
        f"* **Min records:** {balance['min']}",
        f"* **Max records:** {balance['max']}",
        f"* **Status:** {'✅ Balanced' if analysis['balance']['is_balanced'] else '⚠️ Imbalanced'}",
        ""
    ])
    
    # Temporal Characteristics
    md_content.extend([
        "## Temporal Characteristics",
        f"* **Median sampling interval:** {analysis['temporal']['median_sampling_interval']}",
        f"* **Sampling status:** {'✅ Regular' if analysis['temporal']['has_regular_sampling'] else '⚠️ Irregular'}",
        ""
    ])
    
    # Feature Statistics
    md_content.extend(["## Feature Statistics", ""])
    for var, stats in analysis['features'].items():
        md_content.extend([
            f"### {var}",
            f"* **Mean:** {stats['mean']:.2f} ± {stats['std']:.2f}",
            f"* **Missing:** {stats['missing_pct']:.1f}%",
            f"* **Unique values:** {stats['unique_values']}",
            ""
        ])
    

    
    # Save markdown file
    md_file = f"{base_filename}.md"
    with open(md_file, 'w') as f:
        f.write('\n'.join(md_content))
    logger.info(f"Saved markdown report to {md_file}")

def print_analysis_report(analysis, base_filename='report/dataset_analysis'):
    """Print a formatted report of the dataset analysis to console.
    
    Args:
        analysis (dict): Analysis results from analyze_dataset_structure
        base_filename (str): Base name for the report files (without extension)
    """
    try:
        with open(f'{base_filename}.txt', 'r') as f:
            print(f.read())
    except FileNotFoundError:
        logger.error(f"Report file not found at {base_filename}.txt")
        # Fall back to direct printing
        save_analysis_report(analysis, base_filename)
        with open(f'{base_filename}.txt', 'r') as f:
            print(f.read())

if __name__ == "__main__":
    # Example usage
    input_file = "data/dataset_mood_smartphone.csv"
    
    # Clean the dataset and analyze mood patterns
    df_clean, mood_analysis = clean_dataset(input_file)
    
    # Analyze the cleaned dataset structure
    analysis = analyze_dataset_structure(df_clean)
    
    # Add mood analysis to the report
    analysis['mood_patterns'] = mood_analysis
    
    # Save and print the analysis report
    save_analysis_report(analysis, 'report/dataset_analysis')
    print_analysis_report(analysis)
