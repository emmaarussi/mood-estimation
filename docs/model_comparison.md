# Mood Prediction Model Comparison

## Model Performance Comparison

### Simple Features Model
- Training: R² = 0.52, RMSE = 0.67
- Validation: R² = 0.35, RMSE = 0.61
- Test: R² = 0.21, RMSE = 0.82

### Full Features Model
- Training: R² = 0.73, RMSE = 0.51
- Validation: R² = 0.40, RMSE = 0.46
- Test: R² = 0.14, RMSE = 0.70

## Feature Engineering Methodology

### Simple Features Model
The simple features model uses a 7-day rolling window approach to create the following features:
1. Basic Statistics:
   - Mean mood over window
   - Standard deviation of mood
   - Mood trend (difference between last and first day)
2. Activity Metrics:
   - Mean activity level
   - Mean screen time
   - Mean communication time
3. Emotional Features:
   - Mean arousal
   - Mean valence
   - Mean emotion intensity
4. Temporal Features:
   - Day of week
   - Is weekend
5. Data Quality:
   - Number of measurements

### Full Features Model
The full features model creates a more comprehensive set of features (119 total):
1. Temporal Features:
   - Hour of day (raw and cyclical encoding)
   - Day of week (raw and cyclical encoding)
   - Month
   - Time of day categories (night, morning, afternoon, evening)

2. Lag Features:
   - Previous mood values at different time points (8h, 16h, 24h, 48h, 72h, 168h)
   - Rolling statistics for each lag window:
     - Mean
     - Standard deviation
     - Minimum
     - Maximum

3. Activity Features:
   - Activity intensity over multiple windows
   - Activity variability
   - Screen time aggregations

4. Communication Features:
   - Call frequency and duration
   - SMS frequency
   - Overall communication patterns

5. App Usage Features:
   - Usage patterns by category
   - App switching frequency
   - Screen time distribution

6. Circumplex Features:
   - Arousal and valence means
   - Arousal and valence variability
   - Combined metrics:
     - Affect intensity
     - Affect angle

7. User-Specific Features:
   - User average mood
   - User mood variability
   - Time since last mood report

## Model Architecture and Training

### Model Architecture
Both models use XGBoost (eXtreme Gradient Boosting) for regression with the following key parameters:

Simple Features Model:
- n_estimators: 100
- learning_rate: 0.1
- max_depth: 3

Full Features Model:
- n_estimators: 200
- learning_rate: 0.05
- max_depth: 4
- min_child_weight: 2
- subsample: 0.8
- colsample_bytree: 0.8
- gamma: 1

### Training Methodology
1. Data Split:
   - Train: Data up to 2014-05-08
   - Validation: 2014-05-08 to 2014-05-23
   - Test: After 2014-05-23
   - Buffer period: 7 days between splits to ensure independence

2. Cross-Validation:
   - Early stopping using validation set
   - Monitoring mean squared error

3. Feature Importance:
Simple Features Top 5:
- Mood mean
- Valence mean
- Arousal mean
- Mood trend
- Activity mean

Full Features Top 5:
- Circumplex valence (25.5%)
- 168h mood rolling mean (6.3%)
- 168h mood rolling min (5.6%)
- User average mood (3.9%)
- Circumplex arousal (2.6%)

## Model Comparison Analysis

1. Performance:
   - The full features model shows better training performance (R² 0.73 vs 0.52)
   - Validation performance is similar (R² 0.40 vs 0.35)
   - Test performance is slightly worse for the full model (R² 0.14 vs 0.21)

2. Overfitting:
   - The full features model shows more overfitting (larger gap between train and test performance)
   - The simple model maintains more consistent performance across splits

3. Feature Importance:
   - Both models identify emotional valence as a key predictor
   - The full model relies more heavily on historical mood data
   - The simple model gives more weight to current emotional state

4. Recommendations:
   - The simple model might be more practical for deployment due to:
     - Less overfitting
     - More stable performance
     - Simpler feature engineering pipeline
   - The full model might be improved by:
     - Stronger regularization
     - Feature selection
     - Reducing model complexity
