import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from datetime import timedelta
import sys, os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from metrics import evaluate_model_pipeline
from feature_engineering.simple_feature_engineering import prepare_rolling_window_data


def split_user_data(user_df, train_frac=0.6, val_frac=0.2):
    user_df = user_df.sort_values("date").reset_index(drop=True)
    n = len(user_df)
    te = int(n * train_frac)
    ve = int(n * (train_frac + val_frac))
    user_df["split"] = "test"
    user_df.loc[:te-1, "split"] = "train"
    user_df.loc[te:ve-1, "split"] = "val"
    return user_df

def report_clf(name, y_true, y_pred):
    print(f"\n--- {name} ---")
    print("Accuracy :", accuracy_score(y_true, y_pred))
    print("Precision:", precision_score(y_true, y_pred))
    print("Recall   :", recall_score(y_true, y_pred))
    print("F1       :", f1_score(y_true, y_pred))

def main(input_file=None):
    # 1 Load data
    input_file = 'data/rolling_features_4d.parquet' # 4 days is the best!
    df = pd.read_parquet(input_file)

    # 2 Make splits
    df = df.groupby("id", group_keys=False).apply(split_user_data)
    
    # Categorical cols handling
    categorical_cols = ['day_of_week', 'is_weekend']
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True, dummy_na=False)

    # 3 Create binary target: high mood = 1 if mood > 6.5 else 0
    df["target_class"] = (df["target_mood"] > 6.5).astype(int)

    # 4 Define features & target
    target       = "target_class"
    exclude_cols = ["date","id","split","target_mood", target]
    features     = [c for c in df.columns if c not in exclude_cols]

    # 5 Build train/val/test sets
    X_train = df[df["split"]=="train"][["id","date"]+features]; y_train = df[df["split"]=="train"][target]
    X_val   = df[df["split"]=="val"]  [["id","date"]+features]; y_val   = df[df["split"]=="val"]  [target]
    X_test  = df[df["split"]=="test"] [["id","date"]+features]; y_test  = df[df["split"]=="test"] [target]

    # Drop id/date for fitting
    X_train_f = X_train.drop(["id","date"],axis=1)
    X_val_f   = X_val.drop(["id","date"],axis=1)
    X_test_f  = X_test.drop(["id","date"],axis=1)

    # — Baseline: untuned XGBClassifier
    baseline = xgb.XGBClassifier(
        random_state=42,
        eval_metric="logloss",
        n_estimators=100,
        max_depth=3
    )
    baseline.fit(X_train_f, y_train)

    # Evaluate baseline
    y_val_base  = baseline.predict(X_val_f)
    y_test_base = baseline.predict(X_test_f)
    report_clf("Baseline XGBClassifier — Validation", y_val,  y_val_base)
    report_clf("Baseline XGBClassifier — Test",       y_test, y_test_base)

    # — Tuned XGBClassifier via time‑series CV
    tscv = TimeSeriesSplit(n_splits=5)
    param_dist = {
        "n_estimators":     [100, 200, 300],
        "max_depth":        [3, 5, 7],
        "learning_rate":    [0.01, 0.1, 0.2],
        "subsample":        [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0]
    }
    search = RandomizedSearchCV(
        estimator=xgb.XGBClassifier(random_state=42, eval_metric="logloss"),
        param_distributions=param_dist,
        n_iter=10,
        scoring="f1",      # optimize for F1‐score
        cv=tscv,
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
    search.fit(X_train_f, y_train)

    best_xgb = search.best_estimator_
    print("\nBest hyperparameters:", search.best_params_)

    # — Evaluate & plot only the tuned model
    evaluate_model_pipeline(
    best_xgb,
    X_train, y_train,
    X_val,   y_val,
    X_test,  y_test,
    model_name="Tuned XGBoost Classifier"
    )

if __name__ == "__main__":
    main()
