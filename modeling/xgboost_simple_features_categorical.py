import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
import matplotlib.pyplot as plt
import sys
import seaborn as sns
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from feature_engineering.simple_feature_engineering import prepare_rolling_window_data


def main(input_file=None):
    # Handle input file path
    input_file = input_file or 'data/mood_prediction_simple_features.csv'

    # Create necessary directories
    os.makedirs('data_analysis/plots/modeling', exist_ok=True)
    os.makedirs('models', exist_ok=True)

    # Load data
    print("Loading data...")
    df = pd.read_csv(input_file)
    df['time'] = pd.to_datetime(df['time'])

    # Prepare features (categorical outcome)
    print("Preparing features (categorical)...")
    X, y, dates, user_ids, encoder = prepare_rolling_window_data(
        df, window_size=7, categorical=True
    )
    dates = pd.to_datetime(dates)

    # Combine for splitting
    full = pd.concat([X, pd.Series(y, name='target')], axis=1)

    train_list, val_list, test_list = [], [], []
    for user, grp in full.groupby('user_id'):
        grp = grp.sort_values('date')
        n = len(grp)
        if n < 10:
            continue
        i1 = int(n * 0.8)
        i2 = int(n * 0.9)
        train_list.append(grp.iloc[:i1])
        val_list.append(grp.iloc[i1:i2])
        test_list.append(grp.iloc[i2:])

    train_df = pd.concat(train_list).reset_index(drop=True)
    val_df   = pd.concat(val_list).reset_index(drop=True)
    test_df  = pd.concat(test_list).reset_index(drop=True)

    X_train, y_train = train_df.drop('target', axis=1), train_df['target']
    X_val,   y_val   = val_df.drop('target', axis=1),   val_df['target']
    X_test,  y_test  = test_df.drop('target', axis=1),  test_df['target']
    
    # Set up classes
    n_classes = len(np.unique(y_train))
    if n_classes < 2:
        raise ValueError(f"Not enough classes in training set: found {n_classes}.")
    if n_classes == 2:
        objective = 'binary:logistic'
        eval_metric = 'logloss' # Use logloss for binary
    else:
        objective = 'multi:softprob'
        eval_metric = 'mlogloss' # Use mlogloss for multi-class

    print(f"Using objective: {objective}, eval_metric: {eval_metric}, num_class: {n_classes}")
    
    # Baseline model
    print("Training baseline classifier...")
    clf = XGBClassifier(
        objective=objective,
        num_class=n_classes if n_classes > 2 else None,
        eval_metric=eval_metric,
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    )
    clf.fit(
        X_train.drop(['user_id','date'], axis=1), y_train,
        eval_set=[(X_val.drop(['user_id','date'], axis=1), y_val)],
        verbose=False
    )
    clf.save_model('models/baseline_xgb_classifier.model')
    print("Baseline model saved.")
    evaluate_and_plot(
        clf, X_train, y_train, X_val, y_val, X_test, y_test, encoder, 'Baseline'
    )

    # Hyperparameter tuning
    print("\nStarting hyperparameter tuning...")
    param_grid = {
        'learning_rate':    [0.01, 0.05, 0.1],
        'n_estimators':     [100, 300],
        'max_depth':        [3, 5, 7],
        'min_child_weight': [1, 3],
        'subsample':        [0.7, 0.9],
        'colsample_bytree': [0.7, 0.9],
        'gamma':            [0, 0.2, 0.5],
    }
    
    xgb_base = XGBClassifier(
        objective=objective,
        eval_metric=eval_metric,
        num_class=n_classes if n_classes > 2 else None,
        random_state=42
    )
    # Define the GroupKFold cross-validator
    n_splits_cv = 3  # Or more, depending on data size
    time_split = TimeSeriesSplit(n_splits=n_splits_cv)

    # Sort X_train and y_train by date to ensure correct temporal order
    train_sorted = X_train.sort_values('date').reset_index(drop=True)
    y_train_sorted = y_train.loc[train_sorted.index]

    grid_search = GridSearchCV(
        estimator=xgb_base,
        param_grid=param_grid,
        scoring='f1_weighted',
        cv=time_split,  # Use TimeSeriesSplit here
        verbose=0,
        n_jobs=-1
    )

    print(f"Performing {n_splits_cv}-Fold TimeSeriesSplit Cross-Validation for Tuning...")
    grid_search.fit(
        train_sorted.drop(['user_id', 'date'], axis=1),  # Features only
        y_train_sorted                                   # Target
    )
    print("Best params:", grid_search.best_params_)
    print("Best CV acc:", grid_search.best_score_)

    best = grid_search.best_estimator_
    best.save_model('models/xgb_classifier_tuned.model')
    print("Tuned model saved.")
    evaluate_and_plot(
        best, X_train, y_train, X_val, y_val, X_test, y_test, encoder, 'Tuned'
    )


def evaluate_and_plot(model, X_train, y_train, X_val, y_val, X_test, y_test, encoder, label):
    """
    Evaluate classifier, print metrics, and plot confusion matrix.
    """
    def _eval(split, X, y):
        Xf = X.drop(['user_id','date'], axis=1)
        preds = model.predict(Xf)
        acc = accuracy_score(y, preds)
        print(f"{label} {split} Accuracy: {acc:.4f}")
        print(classification_report(y, preds, target_names=encoder.classes_, zero_division=0))
        cm = confusion_matrix(y, preds)
        plt.figure(figsize=(6,6))
        sns.heatmap(cm, annot=True, fmt='d',
                    xticklabels=encoder.classes_,
                    yticklabels=encoder.classes_)
        plt.title(f"{label} {split} Confusion Matrix")
        plt.ylabel('True')
        plt.xlabel('Predicted')
        plt.tight_layout()
        plt.savefig(f"data_analysis/plots/modeling/{label.lower()}_{split.lower()}_cm.png")
        plt.close()

    print(f"\n--- {label} Evaluation ---")
    _eval('Train', X_train, y_train)
    _eval('Val',   X_val,   y_val)
    _eval('Test',  X_test,  y_test)


if __name__ == "__main__":
    main()