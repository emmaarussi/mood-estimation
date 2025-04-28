# -------------------------------------------------------------
# LSTM Classification Pipeline for High Mood Prediction
# -------------------------------------------------------------
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Embedding, Flatten, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping as KerasES
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
import sys, os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from metrics import evaluate_model_pipeline

# -------------------------------------------------------------
# 1. Load and preprocess raw data
# -------------------------------------------------------------
FILE = "data/basic_features.parquet"
if FILE.endswith(".parquet"):
    df = pd.read_parquet(FILE)
else:
    df = pd.read_csv(FILE)

# One-hot date features
df = pd.get_dummies(df, columns=["day_of_week", "is_weekend"], drop_first=True)

# Impute numeric features per user
base_features = [
    "activity", "call", "circumplex_arousal", "circumplex_valence",
    "mood", "screen", "sms", "social_communication",
    "entertainment_leisure", "productivity_work", "miscellaneous",
    "emotion_intensity"
]
date_dummy_features = [c for c in df.columns if c.startswith("day_of_week_") or c.startswith("is_weekend_")]
impute_cols = base_features + date_dummy_features
imputed = []
for _, grp in df.groupby("id", group_keys=False):
    g = grp.copy()
    g[impute_cols] = g[impute_cols].ffill().bfill().fillna(g[impute_cols].mean())
    imputed.append(g)
df = pd.concat(imputed, ignore_index=True)
# drop missing mood records
df = df.dropna(subset=["target_mood"]).reset_index(drop=True)

# Create binary target: high mood > 6.5
df['target_class'] = (df['target_mood'] > 6.5).astype(int)

# Map users to integer IDs for embedding
user_lookup = {u: i for i, u in enumerate(df['id'].astype(str).unique())}
df['uid_int'] = df['id'].astype(str).map(user_lookup)

# -------------------------------------------------------------
# 2. Build sequences (SEQ_LEN days) with no user leakage
# -------------------------------------------------------------
SEQ_LEN = 14
feature_cols = base_features + date_dummy_features
X_seq, uid_arr, y_cls, dates = [], [], [], []
for uid, grp in df.groupby('id', sort=False):
    g = grp.sort_values('date').reset_index(drop=True)
    for i in range(SEQ_LEN, len(g)):
        window = g.loc[i-SEQ_LEN:i-1, feature_cols]
        if window.isna().any().any():
            continue
        X_seq.append(window.values)
        uid_arr.append(g.loc[i, 'uid_int'])
        y_cls.append(g.loc[i, 'target_class'])
        dates.append(g.loc[i, 'date'])
X = np.array(X_seq)
uids = np.array(uid_arr, dtype='int32')
y = np.array(y_cls)
dates = np.array(dates, dtype='datetime64[ns]')

# Chronological split 60/20/20
i = np.argsort(dates)
X, uids, y, dates = X[i], uids[i], y[i], dates[i]
N = len(X)
tr, ve = int(0.6 * N), int(0.8 * N)
X_tr, u_tr, y_tr = X[:tr], uids[:tr], y[:tr]
X_val, u_val, y_val = X[tr:ve], uids[tr:ve], y[tr:ve]
X_test, u_test, y_test = X[ve:], uids[ve:], y[ve:]

# Scale features (per-time-step)
n_feat = len(feature_cols)
scaler = StandardScaler()
X_tr_s = scaler.fit_transform(X_tr.reshape(-1, n_feat)).reshape(X_tr.shape)
X_val_s = scaler.transform(X_val.reshape(-1, n_feat)).reshape(X_val.shape)
X_test_s = scaler.transform(X_test.reshape(-1, n_feat)).reshape(X_test.shape)

# -------------------------------------------------------------
# 3. Define LSTM classifier model
# -------------------------------------------------------------
NUM_USERS = len(user_lookup)
DROPOUT = 0.2

def build_clf(units=32, dense_units=16, lr=1e-3):
    seq_in = Input((SEQ_LEN, n_feat), name='seq')
    uid_in = Input((1,), dtype='int32', name='uid')
    emb = Embedding(NUM_USERS, 8)(uid_in)
    emb = Flatten()(emb)
    x = LSTM(units, dropout=DROPOUT, recurrent_dropout=0.0)(seq_in)
    x = Concatenate()([x, emb])
    x = Dense(dense_units, activation='relu')(x)
    x = Dropout(DROPOUT)(x)
    out = Dense(1, activation='sigmoid')(x)
    m = Model([seq_in, uid_in], out)
    m.compile(optimizer=Adam(lr), loss='binary_crossentropy', metrics=['accuracy'])
    return m

# -------------------------------------------------------------
# 4. Baseline LSTM classifier
# -------------------------------------------------------------
baseline_clf = build_clf()
print("\n=== Baseline LSTM Classifier Summary ===")
baseline_clf.summary()

es = KerasES(monitor='val_loss', patience=5, restore_best_weights=True)
baseline_clf.fit(
    [X_tr_s, u_tr], y_tr,
    validation_data=([X_val_s, u_val], y_val),
    epochs=30, batch_size=32,
    callbacks=[es], verbose=1
)

# Evaluate baseline classifier with explicit metrics
def report_clf(name, y_true, y_pred):
    print(f"--- {name} ---")
    print("Accuracy :", accuracy_score(y_true, y_pred))
    print("Precision:", precision_score(y_true, y_pred))
    print("Recall   :", recall_score(y_true, y_pred))
    print("F1       :", f1_score(y_true, y_pred))

splits = [("Train", X_tr_s, u_tr, y_tr),
          ("Validation", X_val_s, u_val, y_val),
          ("Test", X_test_s, u_test, y_test)]
for name, X_, u_, y_true in splits:
    y_pred = (baseline_clf.predict([X_, u_]) > 0.5).astype(int)
    report_clf(f"Baseline LSTM {name}", y_true, y_pred)

# -------------------------------------------------------------
# 5. Hyperparameter tuning with simpler approach
# -------------------------------------------------------------

# Instead of using CV with multiple inputs, let's simplify by:
# 1. Creating a single validation split
# 2. Manually evaluating different hyperparameters
# 3. Tracking the best configuration

print("\n=== Manual Hyperparameter Tuning ===")
param_grid = {
    'units': [16, 32, 48],
    'dense_units': [8, 16, 24],
    'lr': [1e-3, 5e-4, 5e-3]
}
batch_sizes = [16, 32]
epochs_list = [20, 30]

best_val_acc = 0
best_config = {}
best_model = None

# Use a subset of possible combinations to avoid excessive training
from itertools import product
configs = list(product(
    param_grid['units'], 
    param_grid['dense_units'],
    param_grid['lr'],
    batch_sizes,
    epochs_list
))
# Sample a subset of configurations randomly
import random
random.seed(42)
selected_configs = random.sample(configs, min(8, len(configs)))

for units, dense_units, lr, batch_size, epochs in selected_configs:
    print(f"\nTrying: units={units}, dense={dense_units}, lr={lr}, batch={batch_size}, epochs={epochs}")
    
    # Build model with these hyperparameters
    model = build_clf(units=units, dense_units=dense_units, lr=lr)
    
    # Train with early stopping
    es = KerasES(monitor='val_loss', patience=3, restore_best_weights=True)
    history = model.fit(
        [X_tr_s, u_tr], y_tr,
        validation_data=([X_val_s, u_val], y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[es],
        verbose=0  # Suppress per-epoch output
    )
    
    # Evaluate on validation set
    y_val_pred = (model.predict([X_val_s, u_val]) > 0.5).astype(int)
    val_acc = accuracy_score(y_val, y_val_pred)
    print(f"Validation accuracy: {val_acc:.4f}")
    
    # Update best configuration if this is better
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_config = {
            'units': units,
            'dense_units': dense_units,
            'lr': lr,
            'batch_size': batch_size,
            'epochs': epochs
        }
        # Save model weights
        best_model = model

print("\nBest hyperparameters:", best_config)
print("Best validation accuracy:", best_val_acc)

# -------------------------------------------------------------
# 6. Evaluate the best model
# -------------------------------------------------------------
print("\n=== Best Model Summary ===")
best_model.summary()

# Evaluate tuned classifier with explicit metrics
for name, X_, u_, y_true in splits:
    y_pred = (best_model.predict([X_, u_]) > 0.5).astype(int)
    report_clf(f"Best LSTM {name}", y_true, y_pred)
    
# Evaluate baseline classifier with explicit metrics
for name, X_, u_, y_true in splits:
    y_pred = (baseline_clf.predict([X_, u_]) > 0.5).astype(int)
    report_clf(f"Best LSTM {name}", y_true, y_pred)
    
from sklearn.metrics import RocCurveDisplay

# ROC curve for best LSTM on test set
y_test_proba = best_model.predict([X_test_s, u_test]).flatten()
RocCurveDisplay.from_predictions(
    y_test, y_test_proba, name="Best LSTM"
)
import matplotlib.pyplot as plt
#plt.title("ROC Curve - Best LSTM (Test Set)")
plt.savefig("data_analysis/plots/modeling/roc_curve_best_lstm.png")
plt.close()
print("ROC curve saved to: data_analysis/plots/modeling/roc_curve_best_lstm.png")