# Feature Optimization Results - 2025-04-17

## Feature Set Overview
Total features: 64 features across 7 categories

### Feature Categories
1. Temporal Features (8)
   - hour, day_of_week, month, time_of_day
   - hour_sin, hour_cos, day_sin, day_cos

2. Mood Features (9) - Windows: 24h, 72h, 168h
   - mood_lag_Xh
   - mood_rolling_mean_Xh
   - mood_rolling_std_Xh

3. Activity Features (9) - Windows: 24h, 72h, 168h
   - activity_intensity_Xh
   - activity_variability_Xh
   - screen_time_Xh

4. Communication Features (9) - Windows: 24h, 72h, 168h
   - call_frequency_Xh
   - sms_frequency_Xh
   - total_communication_Xh

5. App Usage Features (6) - Windows: 24h, 72h, 168h
   - productive_ratio_Xh
   - app_diversity_Xh

6. Circumplex Features (12) - Windows: 24h, 72h, 168h
   - arousal_std_Xh
   - valence_std_Xh
   - affect_intensity_Xh
   - affect_angle_Xh

7. Other Features (1)
   - time_since_last_mood

## Model Performance
- Training R²: 0.8724 (RMSE: 0.4447)
- Validation R²: 0.8051 (RMSE: 0.9969)
- Test R²: 0.3092 (RMSE: 1.3290)

## Top 20 Features by Importance
1. mood_lag_24h (0.321)
2. activity_variability_72h (0.141)
3. sms_frequency_168h (0.048)
4. mood_rolling_std_72h (0.047)
5. call_frequency_168h (0.039)
6. mood_rolling_std_24h (0.032)
7. screen_time_168h (0.028)
8. mood_rolling_mean_24h (0.026)
9. sms_frequency_24h (0.026)
10. total_communication_168h (0.025)
11. activity_intensity_24h (0.024)
12. mood_rolling_std_168h (0.024)
13. hour (0.020)
14. total_communication_72h (0.019)
15. call_frequency_24h (0.017)
16. day_cos (0.016)
17. month (0.015)
18. sms_frequency_72h (0.014)
19. mood_rolling_mean_72h (0.014)
20. activity_variability_168h (0.013)

## Feature Category Distribution in Top 20
- Mood features: 6 features
- Activity features: 3 features
- Communication features: 6 features
- Temporal features: 3 features

## Key Findings
1. Most Important Features:
   - Short-term mood (24h lag) is the strongest predictor
   - Activity variability over 3 days is the second strongest
   - Communication patterns (SMS, calls) over 1 week are important
   - Mood variability (std) is more important than mean mood

2. Window Size Impact:
   - 24h windows appear 7 times in top 20
   - 72h windows appear 4 times
   - 168h windows appear 6 times
   - All window sizes contribute meaningful features

3. Data Leakage Prevention:
   - All rolling windows use closed='left'
   - All lag features use shift(1)
   - Temporal features only use current timestamp
   - No cross-user information sharing

## Current Challenges
1. Overfitting:
   - Large gap between training (R²: 0.87) and test (R²: 0.31) performance
   - Model performs well on validation but poorly on test data

## Recommendations
1. Model Improvements:
   - Increase XGBoost regularization
   - Try feature selection using importance threshold
   - Consider ensemble methods or stacking
   - Experiment with simpler models for base predictions

2. Feature Engineering:
   - Focus on top 20 features for a simpler model
   - Consider interaction features between top predictors
   - Investigate why activity variability is so important

3. Data Collection:
   - Collect more test data if possible
   - Investigate if test period has different characteristics
   - Consider if external factors affect mood during test period
