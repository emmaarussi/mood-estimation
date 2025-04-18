# Feature Engineering Process Documentation

## Data Preparation and Leakage Prevention

### 1. Data Format Transformation
- **Initial Format**: Long format with columns [id, time, variable, value]
- **Transformation**: Convert to wide format using `pivot_long_to_wide`
- **Duplicate Handling**: 
  - Multiple measurements at same timestamp are averaged
  - Ensures one record per (id, time) combination
  - Prevents information leakage from future duplicate measurements

### 2. Temporal Ordering
- **Strict Ordering**: Data is sorted by ['id', 'time'] before any feature creation
- **Purpose**: Ensures temporal causality in feature calculations
- **Impact**: Prevents accidental use of future data in rolling calculations

### 3. Feature Creation Order
Features are created in a specific order to prevent data leakage:

1. **Temporal Features** (No leakage risk)
   - hour, day_of_week, month, time_of_day
   - Cyclical encoding: hour_sin, hour_cos, day_sin, day_cos
   - These are point-in-time features with no temporal dependencies

2. **Lag Features** (Leakage prevention critical)
   - All calculations use `.shift(1)` to access only past data
   - Rolling windows use `closed='left'` to exclude current observation
   - Windows: 24h, 72h, 168h (1 day, 3 days, 1 week)
   - Features:
     - mood_lag_Xh: Previous mood value
     - mood_rolling_mean_Xh: Mean of past moods
     - mood_rolling_std_Xh: Standard deviation of past moods

3. **Activity Features** (Historical windows)
   - All calculations use past data only via `closed='left'`
   - Features calculated over 24h, 72h, 168h windows:
     - activity_intensity_Xh
     - activity_variability_Xh
     - screen_time_Xh

4. **Communication Features** (Historical windows)
   - Past-only calculations for call and SMS data
   - Features over 24h, 72h, 168h windows:
     - call_frequency_Xh
     - sms_frequency_Xh
     - total_communication_Xh

5. **App Usage Features** (Historical windows)
   - Rolling calculations on past app usage data
   - Features over 24h, 72h, 168h windows:
     - productive_ratio_Xh
     - app_diversity_Xh

6. **Circumplex Features** (Historical windows)
   - Calculations on past arousal and valence data
   - Features over 24h, 72h, 168h windows:
     - arousal_std_Xh
     - valence_std_Xh
     - affect_intensity_Xh
     - affect_angle_Xh

### 4. Gap Features (From Data Cleaning)
- Gap features from cleaned dataset are preserved:
  - gap_category (normal, 12-24h, 24-48h, 48h+)
  - avg_gap_hours
  - gap_std_hours
  - max_gap_hours

## Key Data Leakage Prevention Techniques

1. **Shifting Operations**
   - Use of `.shift(1)` ensures we only access past values
   - Example: `mood_lag_24h = mood.shift(1)`

2. **Rolling Windows**
   - All rolling operations use `closed='left'`
   - Excludes current observation from calculations
   - Example: `rolling(window=24, min_periods=1, closed='left')`

3. **Temporal Ordering**
   - Strict sorting by time before any calculations
   - Ensures temporal causality is maintained

4. **Window Boundaries**
   - All windowed calculations respect temporal boundaries
   - No future data used in any rolling statistics

5. **Missing Value Handling**
   - Missing values filled with 0 only after all temporal features are created
   - Prevents future information from influencing past predictions

## Feature Dependencies

Features are created in order of dependency to prevent leakage:
1. Independent features (temporal) first
2. Simple lag features second
3. Complex rolling features last

This ordering ensures that no feature accidentally incorporates information from the future or from the current prediction point.
