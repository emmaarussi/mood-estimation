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
    3. Time-based validation: Check for physiologically impossible changes
    
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
    
    # Check for physiologically impossible changes
    for var in ['mood', 'circumplex.arousal', 'circumplex.valence']:
        mask = df_clean['variable'] == var
        var_data = df_clean[mask].copy()
        var_data = var_data.sort_values(['id', 'time'])
        
        # Calculate rate of change per minute
        var_data['value_diff'] = var_data.groupby('id')['value'].diff()
        var_data['time_diff'] = var_data.groupby('id')['time'].diff().dt.total_seconds() / 60
        var_data['rate'] = var_data['value_diff'].abs() / var_data['time_diff']
        
        # Remove changes that are too rapid (more than full scale per 5 minutes)
        if var == 'mood':
            max_rate = 9 / 5  # Full mood scale (9 points) per 5 minutes
        else:
            max_rate = 4 / 5  # Full circumplex scale (4 points) per 5 minutes
        
        rapid_mask = var_data['rate'] > max_rate
        n_rapid = rapid_mask.sum()
        if n_rapid > 0:
            logger.info(f"Removed {n_rapid} physiologically improbable changes from {var}")
            df_clean = df_clean.loc[~mask | ~rapid_mask.reindex(mask.index, fill_value=False)]
            total_outliers += n_rapid
    
    logger.info(f"Total outliers removed: {total_outliers} ({total_outliers/len(df)*100:.2f}% of data)")
    return df_clean

def handle_missing_data(df):
    """Handle missing data using appropriate imputation techniques
    
    Args:
        df (pd.DataFrame): Input dataframe with columns ['id', 'time', 'variable', 'value']
    
    Returns:
        pd.DataFrame: Clean dataset with missing values imputed
    """
    logger.info("Starting missing data imputation...")
    
    # Different imputation strategies for different variable types
    time_vars = [var for var in df['variable'].unique() if var.startswith('appCat') or var == 'screen']
    binary_vars = ['call', 'sms']
    scale_vars = ['mood', 'circumplex.arousal', 'circumplex.valence', 'activity']
    
    df_clean = df.copy()
    total_imputed = 0
    
    # Impute time-based variables with 0 (assuming missing means no usage)
    for var in time_vars:
        mask = (df_clean['variable'] == var) & (df_clean['value'].isnull())
        n_imputed = mask.sum()
        if n_imputed > 0:
            df_clean.loc[mask, 'value'] = 0
            logger.info(f"Imputed {n_imputed} missing values with 0 for {var}")
            total_imputed += n_imputed
    
    # Impute binary variables with 0
    for var in binary_vars:
        mask = (df_clean['variable'] == var) & (df_clean['value'].isnull())
        n_imputed = mask.sum()
        if n_imputed > 0:
            df_clean.loc[mask, 'value'] = 0
            logger.info(f"Imputed {n_imputed} missing values with 0 for {var}")
            total_imputed += n_imputed
    
    # Advanced imputation for scale variables
    for var in scale_vars:
        var_imputed = 0
        # Process each user separately
        for user_id in df_clean[df_clean['variable'] == var]['id'].unique():
            # Get user's data for this variable
            mask = (df_clean['variable'] == var) & (df_clean['id'] == user_id)
            user_data = df_clean[mask].copy()
            user_data = user_data.sort_values('time')
            
            n_missing = user_data['value'].isnull().sum()
            if n_missing > 0:
                # Calculate time differences
                time_diff = user_data['time'].diff()
                long_gaps = time_diff > pd.Timedelta(hours=6)
                
                # For short gaps: Use linear interpolation
                short_gaps = user_data['value'].isnull() & ~long_gaps
                if short_gaps.any():
                    user_data.loc[~long_gaps, 'value'] = \
                        user_data.loc[~long_gaps, 'value'].interpolate(method='linear')
                    var_imputed += short_gaps.sum()
                
                # For long gaps: Use time-of-day patterns
                if long_gaps.any():
                    # Get historical patterns for this time of day
                    hour_means = user_data.groupby(user_data['time'].dt.hour)['value'].mean()
                    
                    # Fill long gaps with time-of-day means
                    for idx in user_data[long_gaps & user_data['value'].isnull()].index:
                        hour = user_data.loc[idx, 'time'].hour
                        if hour in hour_means:
                            user_data.loc[idx, 'value'] = hour_means[hour]
                        else:
                            # If no historical data for this hour, use overall mean
                            user_data.loc[idx, 'value'] = user_data['value'].mean()
                        var_imputed += 1
                
                # Update the main dataframe
                df_clean.loc[mask, 'value'] = user_data['value']
        
        if var_imputed > 0:
            logger.info(f"Imputed {var_imputed} missing values for {var}")
            total_imputed += var_imputed
    
    # Final check for any remaining missing values
    remaining_missing = df_clean['value'].isnull().sum()
    if remaining_missing > 0:
        logger.warning(f"Warning: {remaining_missing} missing values remain after imputation")
    
    logger.info(f"Total values imputed: {total_imputed} ({total_imputed/len(df)*100:.2f}% of data)")
    return df_clean

def clean_dataset(input_file, output_file=None):
    """Clean the dataset by removing outliers and handling missing data
    
    Args:
        input_file (str): Path to input CSV file
        output_file (str, optional): Path to save cleaned CSV file. If None, will use 'cleaned_' prefix
    
    Returns:
        pd.DataFrame: Cleaned dataset
    """
    logger.info(f"Loading dataset from {input_file}")
    df = pd.read_csv(input_file)
    df['time'] = pd.to_datetime(df['time'])
    
    # Remove outliers
    df_clean = remove_outliers(df)
    
    # Handle missing data
    df_clean = handle_missing_data(df_clean)
    
    # Save cleaned dataset
    if output_file is None:
        output_file = input_file.replace('.csv', '_cleaned.csv')
    df_clean.to_csv(output_file, index=False)
    logger.info(f"Saved cleaned dataset to {output_file}")
    
    return df_clean

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
    
    # 5. ML model recommendations
    suitable_models = []
    unsuitable_models = []
    
    # Time series models
    if sampling_stats['has_regular_sampling']:
        suitable_models.extend([
            'ARIMA/SARIMA',
            'Prophet',
            'LSTM/RNN',
            'Temporal Convolutional Networks'
        ])
    else:
        unsuitable_models.extend([
            'ARIMA/SARIMA',
            'Basic RNNs'
        ])
        suitable_models.extend([
            'Irregular time series models (e.g., Neural ODEs)',
            'GRU/LSTM with time delta features'
        ])
    
    # Traditional ML models
    if analysis['balance']['is_balanced']:
        suitable_models.extend([
            'Random Forest',
            'Gradient Boosting (XGBoost, LightGBM)',
            'Linear/Logistic Regression',
            'SVM'
        ])
    else:
        suitable_models.extend([
            'Weighted Random Forest',
            'Balanced Gradient Boosting',
            'SMOTE + Traditional Models'
        ])
        unsuitable_models.extend([
            'Basic Linear/Logistic Regression',
            'Unweighted classifiers'
        ])
    
    # Deep Learning models
    if len(df) > 10000:  # Sufficient data for deep learning
        suitable_models.extend([
            'Deep Neural Networks',
            'Transformers with temporal encoding',
            'Multi-task Learning Models'
        ])
    else:
        unsuitable_models.append('Deep Neural Networks (insufficient data)')
    
    analysis['ml_recommendations'] = {
        'suitable_models': suitable_models,
        'unsuitable_models': unsuitable_models,
        'notes': []
    }
    
    # Add important notes based on analysis
    if not analysis['balance']['is_balanced']:
        analysis['ml_recommendations']['notes'].append(
            'Data is imbalanced - consider using class weights or SMOTE'
        )
    
    if time_span.days < 30:
        analysis['ml_recommendations']['notes'].append(
            'Short time span - may not capture long-term patterns'
        )
    
    if any(stats['missing_pct'] > 20 for stats in feature_stats.values()):
        analysis['ml_recommendations']['notes'].append(
            'High missing data percentage - consider advanced imputation techniques'
        )
    
    logger.info("Dataset structure analysis complete")
    return analysis

def print_analysis_report(analysis):
    """Print a formatted report of the dataset analysis.
    
    Args:
        analysis (dict): Analysis results from analyze_dataset_structure
    """
    print("\n=== Dataset Structure Analysis Report ===")
    
    print("\nBasic Statistics:")
    print(f"- Number of users: {analysis['basic_stats']['n_users']}")
    print(f"- Number of variables: {analysis['basic_stats']['n_variables']}")
    print(f"- Time span: {analysis['basic_stats']['time_span_days']} days")
    print(f"- Total records: {analysis['basic_stats']['total_records']:,}")
    
    print("\nData Balance:")
    balance = analysis['balance']['records_per_user']
    print(f"- Records per user: {balance['mean']:.1f} ± {balance['std']:.1f}")
    print(f"- Min records: {balance['min']}, Max records: {balance['max']}")
    print(f"- Data is {'balanced' if analysis['balance']['is_balanced'] else 'imbalanced'}")
    
    print("\nTemporal Characteristics:")
    print(f"- Median sampling interval: {analysis['temporal']['median_sampling_interval']}")
    print(f"- Sampling is {'regular' if analysis['temporal']['has_regular_sampling'] else 'irregular'}")
    
    print("\nFeature Statistics:")
    for var, stats in analysis['features'].items():
        print(f"\n{var}:")
        print(f"  - Mean: {stats['mean']:.2f} ± {stats['std']:.2f}")
        print(f"  - Missing: {stats['missing_pct']:.1f}%")
        print(f"  - Unique values: {stats['unique_values']}")
    
    print("\nRecommended ML Models:")
    for model in analysis['ml_recommendations']['suitable_models']:
        print(f"✓ {model}")
    
    print("\nNot Recommended ML Models:")
    for model in analysis['ml_recommendations']['unsuitable_models']:
        print(f"✗ {model}")
    
    if analysis['ml_recommendations']['notes']:
        print("\nImportant Notes:")
        for note in analysis['ml_recommendations']['notes']:
            print(f"! {note}")

if __name__ == "__main__":
    # Example usage
    input_file = "data/dataset_mood_smartphone.csv"
    df = pd.read_csv(input_file)
    df['time'] = pd.to_datetime(df['time'])
    
    # Clean the dataset
    df_clean = clean_dataset(input_file)
    
    # Analyze the cleaned dataset
    analysis = analyze_dataset_structure(df_clean)
    print_analysis_report(analysis)
