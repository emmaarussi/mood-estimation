import pandas as pd
import numpy as np

# Load raw data
print("Loading raw dataset...")
df = pd.read_csv('data/dataset_mood_smartphone.csv')
df['time'] = pd.to_datetime(df['time'], format='mixed')

# Focus on mood data
mood_data = df[df['variable'] == 'mood'].copy()
mood_data = mood_data.sort_values(['id', 'time'])

print("\nMood data overview:")
print(f"Total mood records: {len(mood_data)}")
print(f"Missing values: {mood_data['value'].isnull().sum()}")
print(f"Unique users: {mood_data['id'].nunique()}")

# Check time gaps
print("\nAnalyzing time gaps between mood measurements per user:")
time_diffs = mood_data.groupby('id')['time'].diff()
print("\nTime gap statistics (in hours):")
print(time_diffs.dt.total_seconds().div(3600).describe())

# Count large gaps (> 12 hours)
large_gaps = time_diffs.dt.total_seconds().div(3600) > 12
print(f"\nNumber of gaps > 12 hours: {large_gaps.sum()}")

# Check distribution of mood values
print("\nMood value distribution:")
print(mood_data['value'].describe())

# Check if there are any unexpected values
print("\nUnique mood values:")
print(sorted(mood_data['value'].unique()))

# Check completeness per user
print("\nAnalyzing records per user:")
records_per_user = mood_data.groupby('id').agg({
    'time': ['count', 'min', 'max']
}).reset_index()
records_per_user.columns = ['id', 'n_records', 'first_record', 'last_record']
records_per_user['days_span'] = (records_per_user['last_record'] - records_per_user['first_record']).dt.total_seconds() / (24*3600)
records_per_user['avg_records_per_day'] = records_per_user['n_records'] / records_per_user['days_span']

print("\nRecords per user statistics:")
print(records_per_user[['n_records', 'days_span', 'avg_records_per_day']].describe())
