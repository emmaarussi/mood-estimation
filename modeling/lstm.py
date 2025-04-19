# -------------------------------------------------------------
# Updated LSTM pipeline with original plotting, tuning, and summaries
# -------------------------------------------------------------
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from metrics import evaluate_predictions, plot_results_scatter, plot_results_ts
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Embedding, Flatten, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping as KerasES
from scikeras.wrappers import KerasRegressor
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV

# -------------------------------------------------------------
# 1. Load and preprocess data
# -------------------------------------------------------------
FILE = "data/basic_features.parquet"
if FILE.endswith(".parquet"):
    df = pd.read_parquet(FILE)
else:
    df = pd.read_csv(FILE)

base_features = [
    "activity", "call", "circumplex_arousal", "circumplex_valence",
    "mood", "screen", "sms", "social_communication",
    "entertainment_leisure", "productivity_work", "miscellaneous",
    "emotion_intensity"
]

df = pd.get_dummies(df, columns=["day_of_week", "is_weekend"], drop_first=True)

date_dummy_features = [c for c in df.columns if c.startswith("day_of_week_") or c.startswith("is_weekend_")]
imputation_cols = base_features + date_dummy_features

# Impute per user
imputed = []
for _, group in df.groupby("id", group_keys=False):
    g = group.copy()
    g[imputation_cols] = g[imputation_cols].ffill().bfill().fillna(g[imputation_cols].mean())
    imputed.append(g)
df = pd.concat(imputed, ignore_index=True)

# Drop missing target
df = df.dropna(subset=["target_mood"]).reset_index(drop=True)

# Map users to integer IDs for embedding
user_lookup = {uid: idx for idx, uid in enumerate(df["id"].astype(str).unique())}
df["uid_int"] = df["id"].astype(str).map(user_lookup)

feature_cols = base_features + date_dummy_features  # no one-hot user dummies

# -------------------------------------------------------------
# 2. Build sequences (no cross-user leakage)
# -------------------------------------------------------------
SEQ_LEN = 14
X_seq, uid_arr, y, dates = [], [], [], []
for uid, group in df.groupby("id", sort=False):
    g = group.sort_values("date").reset_index(drop=True)
    for i in range(SEQ_LEN, len(g)):
        window = g.loc[i-SEQ_LEN:i-1, feature_cols]
        if window.isna().any().any():
            continue
        X_seq.append(window.values)
        uid_arr.append(g.loc[i, "uid_int"])
        y.append(g.loc[i, "target_mood"])
        dates.append(g.loc[i, "date"])

X_seq = np.array(X_seq)
uid_arr = np.array(uid_arr, dtype="int32")
y = np.array(y)
dates = np.array(dates, dtype="datetime64[ns]")

# Chronological split 60/20/20
order = np.argsort(dates)
X_seq, uid_arr, y, dates = X_seq[order], uid_arr[order], y[order], dates[order]
N = len(X_seq)
tr_end, val_end = int(0.6 * N), int(0.8 * N)
X_tr, uid_tr, y_tr = X_seq[:tr_end], uid_arr[:tr_end], y[:tr_end]
X_val, uid_val, y_val = X_seq[tr_end:val_end], uid_arr[tr_end:val_end], y[tr_end:val_end]
X_te, uid_te, y_te = X_seq[val_end:], uid_arr[val_end:], y[val_end:]

# -------------------------------------------------------------
# 3. Scale features and target
# -------------------------------------------------------------
n_features = len(feature_cols)
scaler_X = StandardScaler()
X_tr_s = scaler_X.fit_transform(X_tr.reshape(-1, n_features)).reshape(X_tr.shape)
X_val_s = scaler_X.transform(X_val.reshape(-1, n_features)).reshape(X_val.shape)
X_te_s  = scaler_X.transform(X_te.reshape(-1, n_features)).reshape(X_te.shape)

scaler_y = StandardScaler()
y_tr_s = scaler_y.fit_transform(y_tr.reshape(-1, 1)).ravel()
y_val_s = scaler_y.transform(y_val.reshape(-1, 1)).ravel()

# -------------------------------------------------------------
# 4. Baseline model: build, summary, train, evaluate
# -------------------------------------------------------------
NUM_USERS = len(user_lookup)
DROPOUT = 0.2

def build_model(units=32, dense_units=16, learning_rate=3e-4):
    seq_in = Input((SEQ_LEN, n_features), name="seq")
    uid_in = Input((1,), dtype="int32", name="uid")
    emb = Embedding(NUM_USERS, 8)(uid_in)
    emb = Flatten()(emb)
    x = LSTM(units, dropout=DROPOUT, recurrent_dropout=0.0)(seq_in)
    x = Concatenate()([x, emb])
    x = Dense(dense_units, activation="relu")(x)
    x = Dropout(DROPOUT)(x)
    out = Dense(1)(x)
    model = Model([seq_in, uid_in], out)
    model.compile(optimizer=Adam(learning_rate), loss="mae", metrics=["mse"])
    return model

# Baseline training
baseline = build_model()
print("\n=== Baseline Model Summary ===")
baseline.summary()
es = KerasES(monitor="val_loss", patience=4, restore_best_weights=True)
baseline.fit(
    [X_tr_s, uid_tr], y_tr_s,
    validation_data=([X_val_s, uid_val], y_val_s),
    epochs=60, batch_size=32, callbacks=[es], verbose=1
)

# Evaluate baseline
eval_splits = [
    ("Train", X_tr_s, uid_tr, y_tr, y_tr_s),
    ("Val",   X_val_s, uid_val, y_val, y_val_s),
    ("Test",  X_te_s,  uid_te, y_te,  None)
]
preds_baseline = {}
for name, X_, uid_, y_true, y_scaled in eval_splits:
    y_pred_s = baseline.predict([X_, uid_]).ravel()
    y_pred = scaler_y.inverse_transform(y_pred_s.reshape(-1,1)).ravel()
    if name != "Test":
        evaluate_predictions(y_scaled, y_pred_s, f"Baseline {name}")
    else:
        evaluate_predictions(y_true, y_pred, "Baseline Test")
    preds_baseline[name] = y_pred

plot_results_scatter(y_te, preds_baseline["Test"], pd.DataFrame({"date": dates[val_end:]}), "Baseline Test Scatter")
plot_results_ts(y_te, preds_baseline["Test"], pd.DataFrame({"date": dates[val_end:]}), "Baseline Test TS")

# -------------------------------------------------------------
# 5. Manual hyperparameter tuning
# -------------------------------------------------------------
print("\n=== Manual Hyperparameter Tuning ===")

# Define parameter grid
param_grid = {
    "units": [8, 16, 24],
    "dense_units": [4, 8, 12],
    "learning_rate": [5e-4, 1e-3, 5e-3]
}
batch_sizes = [16, 32, 64]
epochs_list = [20, 30, 40]

best_val_mae = float('inf')
best_params = {}

# Use TimeSeriesSplit to create validation sets
tscv = TimeSeriesSplit(n_splits=3)

# Generate combinations for testing
import itertools
import random
all_combinations = list(itertools.product(
    param_grid["units"], 
    param_grid["dense_units"],
    param_grid["learning_rate"],
    batch_sizes,
    epochs_list
))

# Randomly select combinations to test
random.seed(42)
n_iter = 10  # Number of combinations to try
selected_combinations = random.sample(all_combinations, min(n_iter, len(all_combinations)))

for units, dense_units, lr, batch_size, epochs in selected_combinations:
    cv_maes = []
    print(f"\nTesting: units={units}, dense={dense_units}, lr={lr}, batch={batch_size}, epochs={epochs}")
    
    # Perform CV using TimeSeriesSplit
    for train_idx, val_idx in tscv.split(X_tr_s):
        # Get the fold's data
        X_fold_train = X_tr_s[train_idx]
        uid_fold_train = uid_tr[train_idx]
        y_fold_train = y_tr_s[train_idx]
        
        X_fold_val = X_tr_s[val_idx]
        uid_fold_val = uid_tr[val_idx]
        y_fold_val = y_tr_s[val_idx]
        
        # Build and train a model for this fold
        model = build_model(units=units, dense_units=dense_units, learning_rate=lr)
        es = KerasES(monitor="val_loss", patience=3, restore_best_weights=True)
        
        model.fit(
            [X_fold_train, uid_fold_train], y_fold_train,
            validation_data=([X_fold_val, uid_fold_val], y_fold_val),
            epochs=epochs, batch_size=batch_size,
            callbacks=[es], verbose=0  # Quiet output
        )
        
        # Calculate validation MAE
        val_preds = model.predict([X_fold_val, uid_fold_val]).ravel()
        mae = np.mean(np.abs(val_preds - y_fold_val))
        cv_maes.append(mae)
    
    # Calculate average MAE across CV folds
    mean_mae = np.mean(cv_maes)
    print(f"  Mean CV MAE: {mean_mae:.4f}")
    
    # Update best parameters if better
    if mean_mae < best_val_mae:
        best_val_mae = mean_mae
        best_params = {
            'model__units': units,
            'model__dense_units': dense_units,
            'optimizer__learning_rate': lr,
            'fit__batch_size': batch_size,
            'fit__epochs': epochs
        }

print(f"\nBest params found: {best_params}")
print(f"Best validation MAE: {best_val_mae:.4f}")
best_params_found = True

# If you want to be safe, add a fallback (though this shouldn't be needed now)
if not best_params:
    print("Using default parameters as fallback.")
    best_params = {
        'fit__batch_size': 32,
        'fit__epochs': 30,
        'model__dense_units': 16,
        'model__units': 32,
        'optimizer__learning_rate': 3e-4
    }
    
# -------------------------------------------------------------
# 6. Build, retrain, and evaluate the best model
# -------------------------------------------------------------
print("\n=== Building and Retraining Best Model ===")

# Extract best parameters for model building and fitting
# Provide defaults in case tuning failed or best_params is empty
best_model_params = {
    'units': best_params.get('model__units', 32),
    'dense_units': best_params.get('model__dense_units', 16),
    'learning_rate': best_params.get('optimizer__learning_rate', 3e-4)
}
best_fit_params = {
    'epochs': best_params.get('fit__epochs', 30),
    'batch_size': best_params.get('fit__batch_size', 32)
}

# Build the best model using the found parameters
tuned_model = build_model(**best_model_params)
print("\n=== Best Model Summary ===")
tuned_model.summary()

# Retrain the best model on the full training data, using validation for early stopping
print(f"\nRetraining with epochs={best_fit_params['epochs']}, batch_size={best_fit_params['batch_size']}")
es_final = KerasES(monitor="val_loss", patience=5, restore_best_weights=True, verbose=1)
history = tuned_model.fit(
    [X_tr_s, uid_tr], y_tr_s,
    validation_data=([X_val_s, uid_val], y_val_s),
    epochs=best_fit_params['epochs'],
    batch_size=best_fit_params['batch_size'],
    callbacks=[es_final],
    verbose=1
)

# Evaluate the final tuned model on all splits
print("\n=== Evaluating Best Model ===")
eval_splits = [
    ("Train", X_tr_s, uid_tr, y_tr, y_tr_s),
    ("Val",   X_val_s, uid_val, y_val, y_val_s),
    ("Test",  X_te_s,  uid_te, y_te,  None) # y_scaled is None for test
]

preds_tuned = {}
for name, X_, uid_, y_true, y_scaled in eval_splits:
    # Predict using the final retrained model
    y_pred_s = tuned_model.predict([X_, uid_]).ravel()
    # Inverse transform predictions to original scale
    y_pred = scaler_y.inverse_transform(y_pred_s.reshape(-1,1)).ravel()
    preds_tuned[name] = y_pred # Store original scale predictions

    print(f"\n--- Tuned {name} ---")
    if name != "Test":
        # Evaluate Train/Val on scaled data as the model predicts scaled values
        # and early stopping was based on scaled validation loss
        evaluate_predictions(y_scaled, y_pred_s, f"Tuned {name} (Scaled)")
    else:
        # Evaluate Test on the original scale for final performance assessment
        evaluate_predictions(y_true, y_pred, f"Tuned {name} (Original Scale)")

# Optional: Plot results for the test set of the tuned model
plot_results_scatter(y_te, preds_tuned["Test"], pd.DataFrame({"date": dates[val_end:]}), "Tuned Test Scatter")
plot_results_ts(y_te, preds_tuned["Test"], pd.DataFrame({"date": dates[val_end:]}), "Tuned Test TS")

print("\n=== LSTM Pipeline Finished ===")