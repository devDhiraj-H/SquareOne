"""
experiments/simulation/train_detector_model.py

Trains and exports the production Cyber-Physical IDS checkpoints:
1. Layer 1 (Physics Invariant Engine):
   - Calibrates benign flight aerodynamic & kinematic residuals (mu, inv_cov, tau_chi2).
2. Layer 2 (Temporal Multi-Modal Classifier):
   - Builds rolling-window temporal features (W=10) across physical & cyber domains.
   - Trains multi-class Temporal Rolling XGBoost model.
3. Exports artifacts to experiments/simulation/models/detector_checkpoint.pkl
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
import xgboost as xgb

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXPERIMENTS_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
sys.path.append(EXPERIMENTS_DIR)

from utils.data_loader import (
    load_physical_dataset, 
    load_cyber_dataset, 
    load_multimodal_dataset, 
    CANONICAL_CLASSES
)


def compute_aerodynamic_residuals(df_phys, y_phys, dt=0.52, g=9.81):
    """Computes first-principles quadrotor kinematic and aerodynamic residuals."""
    res_dfs = []
    for c in CANONICAL_CLASSES:
        sub = df_phys[y_phys == c].copy()
        
        diff_h = sub['height'].diff().fillna(0)
        r_h = (diff_h - sub['z_speed'] * dt)
        
        ax = (sub['x_speed'].diff().fillna(0) / dt)
        ay = (sub['y_speed'].diff().fillna(0) / dt)
        az = (sub['z_speed'].diff().fillna(0) / dt)
        a_mag = np.sqrt(ax**2 + ay**2 + az**2)
        
        r_pitch = ax - g * np.sin(np.radians(sub['pitch']))
        r_roll = ay + g * np.sin(np.radians(sub['roll']))
        
        r_dist_y = sub['mp_distance_y'].diff().fillna(0) - sub['y_speed'] * dt
        r_dist_z = sub['mp_distance_z'].diff().fillna(0) - sub['z_speed'] * dt
        
        df_r = pd.DataFrame({
            'res_height': r_h,
            'accel_x': ax,
            'accel_y': ay,
            'accel_z': az,
            'accel_mag': a_mag,
            'res_pitch_aero': r_pitch,
            'res_roll_aero': r_roll,
            'res_dist_y': r_dist_y,
            'res_dist_z': r_dist_z
        }, index=sub.index)
        res_dfs.append(df_r)
        
    return pd.concat(res_dfs).sort_index()


def train_and_export():
    print("=" * 80)
    print("UAV CYBER-PHYSICAL IDS: TRAINING PRODUCTION SIMULATION MODELS")
    print("=" * 80)

    models_dir = os.path.join(SCRIPT_DIR, "models")
    os.makedirs(models_dir, exist_ok=True)
    checkpoint_path = os.path.join(models_dir, "detector_checkpoint.pkl")

    # -------------------------------------------------------------------------
    # 1. LAYER 1: CALIBRATE PHYSICS INVARIANTS (MAHALANOBIS RESIDUALS)
    # -------------------------------------------------------------------------
    print("\n[+] 1/3 Calibrating Layer 1: Physics Kinematic Invariants...")
    phys_csv = os.path.join(EXPERIMENTS_DIR, "data", "Hardened_Physical_UAV_Dataset.csv")
    if not os.path.exists(phys_csv):
        phys_csv = os.path.join(EXPERIMENTS_DIR, "..", "Physical_UAV_Dataset.csv")

    X_p, y_p, phys_cols = load_physical_dataset(phys_csv)
    residuals = compute_aerodynamic_residuals(X_p, y_p)
    residual_cols = list(residuals.columns)

    # Benign baseline calibration
    benign_residuals = residuals[y_p == 'Benign']
    mu = benign_residuals.mean().values
    cov = np.cov(benign_residuals.values, rowvar=False) + np.eye(residuals.shape[1]) * 1e-4
    inv_cov = np.linalg.pinv(cov)

    diff = residuals.values - mu
    chi2_scores = np.sum(np.dot(diff, inv_cov) * diff, axis=1)
    tau_chi2 = float(np.percentile(chi2_scores[y_p == 'Benign'], 95))

    print(f"    - Benign residual features calibrated: {len(residual_cols)}")
    print(f"    - Chi-Square 95th Percentile Threshold (5% FAR): tau = {tau_chi2:.3f}")

    # -------------------------------------------------------------------------
    # 2. LAYER 2: TRAIN TEMPORAL ROLLING MULTI-MODAL XGBOOST
    # -------------------------------------------------------------------------
    print("\n[+] 2/3 Training Layer 2: Temporal Rolling Multimodal Classifier (W=10)...")
    multi_csv = os.path.join(EXPERIMENTS_DIR, "data", "Hardened_Multimodal_UAV_Dataset.csv")
    if os.path.exists(multi_csv):
        df_multi = pd.read_csv(multi_csv)
        y_raw = df_multi['Target_Label'] if 'Target_Label' in df_multi.columns else df_multi['class']
        X_multi = df_multi.drop(columns=[c for c in ['Target_Label', 'class', 'timestamp_p', 'timestamp_c'] if c in df_multi.columns])
        feature_names = list(X_multi.columns)
    else:
        X_multi, y_raw, feature_names = load_multimodal_dataset()

    le = LabelEncoder().fit(CANONICAL_CLASSES)
    W = 10

    # Build rolling features
    X_roll_list, y_roll_list = [], []
    for c in CANONICAL_CLASSES:
        mask = (y_raw == c).values
        df_c = X_multi[mask].reset_index(drop=True)
        if len(df_c) <= W:
            continue
        
        df_roll_mean = df_c.rolling(W).mean().dropna()
        df_roll_std = df_c.rolling(W).std().fillna(0).iloc[W-1:]
        df_diff = df_c.diff(1).fillna(0).iloc[W-1:]
        
        df_temp = pd.concat([
            df_c.iloc[W-1:].reset_index(drop=True),
            df_roll_mean.reset_index(drop=True).add_suffix('_roll_mean'),
            df_roll_std.reset_index(drop=True).add_suffix('_roll_std'),
            df_diff.reset_index(drop=True).add_suffix('_diff')
        ], axis=1)
        
        labels = np.full(len(df_temp), le.transform([c])[0])
        X_roll_list.append(df_temp)
        y_roll_list.append(labels)

    X_roll = pd.concat(X_roll_list, ignore_index=True)
    y_roll = np.concatenate(y_roll_list, axis=0)
    rolling_feature_names = list(X_roll.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X_roll, y_roll, test_size=0.25, random_state=42, stratify=y_roll
    )

    clf = xgb.XGBClassifier(
        n_estimators=120,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='mlogloss'
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred) * 100
    macro_f1 = f1_score(y_test, y_pred, average='macro') * 100

    print(f"    - Trained Temporal XGBoost on {X_train.shape[1]} temporal features")
    print(f"    - Holdout Test Accuracy: {acc:.2f}% | Macro F1: {macro_f1:.2f}%")
    print("\n" + classification_report(y_test, y_pred, target_names=CANONICAL_CLASSES, digits=3))

    # -------------------------------------------------------------------------
    # 3. EXPORT UNIFIED CHECKPOINT
    # -------------------------------------------------------------------------
    print("\n[+] 3/3 Serializing checkpoint to disk...")
    checkpoint = {
        'classes': CANONICAL_CLASSES,
        'label_encoder': le,
        'physics_layer': {
            'residual_cols': residual_cols,
            'mu': mu,
            'inv_cov': inv_cov,
            'tau_chi2': tau_chi2,
            'max_accel_bound': 15.0,  # Physical drone limit (m/s^2)
            'dt_nominal': 0.52
        },
        'temporal_ml_layer': {
            'model': clf,
            'window_size': W,
            'raw_feature_names': feature_names,
            'rolling_feature_names': rolling_feature_names
        },
        'metadata': {
            'accuracy': acc,
            'macro_f1': macro_f1
        }
    }

    with open(checkpoint_path, 'wb') as f:
        pickle.dump(checkpoint, f)

    print(f"[SUCCESS] Saved Production Checkpoint -> {checkpoint_path}")
    print("=" * 80)


if __name__ == '__main__':
    train_and_export()
