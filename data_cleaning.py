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

if __name__ == "__main__":
    # Example usage
    input_file = "data/dataset_mood_smartphone.csv"
    clean_dataset(input_file)
