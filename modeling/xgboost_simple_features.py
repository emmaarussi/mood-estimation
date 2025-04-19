import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from modeling.metrics import evaluate_model_pipeline
from feature_engineering.simple_feature_engineering import prepare_rolling_window_data

def split_user_data(user_df, train_frac=0.6, val_frac=0.2):
    user_df = user_df.sort_values("date").reset_index(drop=True)
    n = len(user_df)
    train_end = int(n * train_frac)
    val_end   = int(n * (train_frac + val_frac))
    user_df["split"] = "test"
    user_df.loc[:train_end-1, "split"] = "train"
    user_df.loc[train_end:val_end-1, "split"] = "val"
    return user_df

def main(input_file=None):
    
    # Data loading
    input_file = 'data/rolling_features_4d.parquet' # NOTE: model worked best when using rolling window of length 4
    df = pd.read_parquet(input_file)
    
    # Split user data
    df = df.groupby("id", group_keys=False).apply(split_user_data)
    
    # Categorical cols handling
    categorical_cols = ['day_of_week', 'is_weekend']
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True, dummy_na=False)
    
    # 3 Missing‐mood flag & fill
    df["mood_missing"] = df["mood"].isna().astype(int)
    df["mood"] = df["mood"].fillna(df["mood"].mean())

    # 4 Define features & target
    target       = "target_mood"
    exclude_cols = ["date","id","split",target]
    features     = [c for c in df.columns if c not in exclude_cols]

    # 5 Build train/val/test sets
    X_train = df[df["split"]=="train"][["id","date"]+features];  y_train = df[df["split"]=="train"][target]
    X_val   = df[df["split"]=="val"][["id","date"]+features];    y_val   = df[df["split"]=="val"][target]
    X_test  = df[df["split"]=="test"][["id","date"]+features];   y_test  = df[df["split"]=="test"][target]

    # drop id/date for fitting
    X_train_f = X_train.drop(["id","date"],axis=1)
    X_val_f   = X_val.drop(["id","date"],axis=1)
    X_test_f  = X_test.drop(["id","date"],axis=1)

    # 6 Baseline: untuned XGBoost
    baseline = xgb.XGBRegressor(
        random_state=42,
        eval_metric="rmse",
        n_estimators=100,
        max_depth=3
    )
    baseline.fit(X_train_f, y_train)

    def report(name, y_true, y_pred):
        print(f"\n{name}")
        print(" MAE: ", mean_absolute_error(y_true, y_pred))
        print(" RMSE:", np.sqrt(mean_squared_error(y_true, y_pred)))
        print(" R²:  ", r2_score(y_true, y_pred))

    y_val_base  = baseline.predict(X_val_f)
    y_test_base = baseline.predict(X_test_f)
    report("Baseline XGB  — Validation", y_val,  y_val_base)
    report("Baseline XGB  — Test",       y_test, y_test_base)

    # ————————————————————————————————
    # Tuned XGBoost via time‑series CV
    tscv = TimeSeriesSplit(n_splits=5)
    param_dist = {
        "n_estimators":     [100, 200, 300],
        "max_depth":        [3, 5, 7],
        "learning_rate":    [0.01, 0.1, 0.2],
        "subsample":        [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0]
    }
    search = RandomizedSearchCV(
        xgb.XGBRegressor(random_state=42, eval_metric="rmse"),
        param_distributions=param_dist,
        n_iter=10,
        scoring="neg_mean_absolute_error",
        cv=tscv,
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
    search.fit(X_train_f, y_train)

    best_xgb = search.best_estimator_
    print("\nBest hyperparameters:", search.best_params_)

    # ————————————————————————————————
    # 8️⃣ Evaluate & plot only the tuned model
    # (this will use your predict_evaluate_models function,
    # which does all the scatter/TS plots + feature importances)
    evaluate_model_pipeline(
        best_xgb,
        X_train, y_train,
        X_val,   y_val,
        X_test,  y_test,
        model_name="Tuned XGBoost Classifier"
    )
    
    return


if __name__ == "__main__":
    main()
