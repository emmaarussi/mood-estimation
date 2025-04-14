# Dataset Structure Analysis Report

## Basic Statistics
* **Number of users:** 27
* **Number of variables:** 19
* **Time span:** 111 days
* **Total records:** 363,212

## Data Balance
* **Records per user:** 13452.3 ± 4903.9
* **Min records:** 2590
* **Max records:** 21142
* **Status:** ✅ Balanced

## Temporal Characteristics
* **Median sampling interval:** 0 days 00:03:48.707000
* **Sampling status:** ⚠️ Irregular

## Feature Statistics

### call
* **Mean:** 1.00 ± 0.00
* **Missing:** 0.0%
* **Unique values:** 1

### sms
* **Mean:** 1.00 ± 0.00
* **Missing:** 0.0%
* **Unique values:** 1

### mood
* **Mean:** 7.02 ± 0.97
* **Missing:** 0.0%
* **Unique values:** 7

### circumplex.arousal
* **Mean:** -0.10 ± 1.05
* **Missing:** 0.0%
* **Unique values:** 20

### circumplex.valence
* **Mean:** 0.69 ± 0.66
* **Missing:** 0.0%
* **Unique values:** 19

### activity
* **Mean:** 0.10 ± 0.15
* **Missing:** 0.0%
* **Unique values:** 1621

### appCat.other
* **Mean:** 12.96 ± 11.61
* **Missing:** 0.0%
* **Unique values:** 4582

### appCat.builtin
* **Mean:** 7.11 ± 8.35
* **Missing:** 0.0%
* **Unique values:** 19427

### appCat.communication
* **Mean:** 32.17 ± 41.07
* **Missing:** 0.0%
* **Unique values:** 38276

### appCat.entertainment
* **Mean:** 10.00 ± 16.57
* **Missing:** 0.0%
* **Unique values:** 10304

### appCat.social
* **Mean:** 54.11 ± 70.27
* **Missing:** 0.0%
* **Unique values:** 14182

### screen
* **Mean:** 42.94 ± 59.70
* **Missing:** 0.0%
* **Unique values:** 60974

### appCat.utilities
* **Mean:** 13.29 ± 15.02
* **Missing:** 0.0%
* **Unique values:** 1685

### appCat.unknown
* **Mean:** 29.71 ± 38.03
* **Missing:** 0.0%
* **Unique values:** 817

### appCat.finance
* **Mean:** 14.80 ± 18.08
* **Missing:** 0.0%
* **Unique values:** 687

### appCat.travel
* **Mean:** 31.82 ± 37.25
* **Missing:** 0.0%
* **Unique values:** 2557

### appCat.office
* **Mean:** 5.89 ± 7.32
* **Missing:** 0.0%
* **Unique values:** 2773

### appCat.weather
* **Mean:** 18.87 ± 14.36
* **Missing:** 0.0%
* **Unique values:** 249

### appCat.game
* **Mean:** 88.07 ± 119.73
* **Missing:** 0.0%
* **Unique values:** 767

## Machine Learning Model Recommendations

### Recommended Models
✅ Irregular time series models (e.g., Neural ODEs)
✅ GRU/LSTM with time delta features
✅ Random Forest
✅ Gradient Boosting (XGBoost, LightGBM)
✅ Linear/Logistic Regression
✅ SVM
✅ Deep Neural Networks
✅ Transformers with temporal encoding
✅ Multi-task Learning Models

### Not Recommended Models
❌ ARIMA/SARIMA
❌ Basic RNNs
