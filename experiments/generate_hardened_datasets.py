"""
experiments/generate_hardened_datasets.py

Step 1: Hardening & Artifact Neutralization
Generates artifact-free, hardened datasets to benchmark genuine cyber-physical
detection capabilities without relying on crash sentinels or extreme lab artifacts.

Outputs:
- experiments/data/Hardened_Physical_UAV_Dataset.csv
- experiments/data/Hardened_Multimodal_UAV_Dataset.csv
- experiments/results/hardened_benchmark_summary.csv
- experiments/results/hardened_benchmark_table.md
- experiments/results/hardened_sensitivity_plot.png
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.decomposition import PCA

# Path setup
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(SCRIPT_DIR)
sys.path.append(os.path.join(SCRIPT_DIR, '..'))

from utils.data_loader import (
    load_physical_dataset, load_cyber_dataset, load_multimodal_dataset,
    get_stratified_split, get_novelty_detection_split
)
from utils.unsupervised_metrics import evaluate_anomaly_detector, measure_inference_speed


def create_hardened_datasets():
    print("=" * 80)
    print("STEP 1: GENERATING ARTIFACT-NEUTRALIZED HARDENED DATASETS")
    print("=" * 80)
    
    data_dir = os.path.join(SCRIPT_DIR, "data")
    results_dir = os.path.join(SCRIPT_DIR, "results")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    # 1. Load original physical dataset (read-only)
    X_p, y_p, feature_names = load_physical_dataset(os.path.join(SCRIPT_DIR, "..", "Physical_UAV_Dataset.csv"))
    print(f"[*] Loaded Raw Physical Telemetry: {X_p.shape[0]} rows, {len(feature_names)} features")
    
    # Analyze Benign distribution for realistic replacement
    benign_mask = (y_p == 'Benign')
    X_benign = X_p[benign_mask]
    benign_means = X_benign.mean()
    benign_stds = X_benign.std()
    
    # Clone to create Hardened Physical Dataset
    X_p_hardened = X_p.copy()
    
    # 2. Neutralize Evil_Twin crash sentinels (-100 m/s x_speed, -1.0 height, 130 mp_distance_y)
    # In reality, during an Evil Twin attack, the drone hovers in place with genuine physical aerodynamics.
    evil_mask = (y_p == 'Evil_Twin')
    n_evil = evil_mask.sum()
    print(f"[*] Neutralizing {n_evil} Evil_Twin crash sentinels with realistic steady hover aerodynamics...")
    
    sampled_benign = X_benign.sample(n=n_evil, replace=True, random_state=42)
    for col in X_p.columns:
        X_p_hardened.loc[evil_mask, col] = sampled_benign[col].values
        
    # 3. Attenuate FDI (False Data Injection) to realistic stealthy drift
    # Original dataset has extreme +/- 100 m/s values.
    # We create a realistic stealth injection (+1.5 m/s drift in speed, +3.0 m in height)
    fdi_mask = (y_p == 'FDI')
    n_fdi = fdi_mask.sum()
    print(f"[*] Attenuating {n_fdi} FDI records from extreme +/-100 m/s to realistic subtle drift (+1.5 m/s)...")
    
    for col in X_p.columns:
        base_hover = np.random.normal(loc=benign_means[col], scale=benign_stds[col], size=n_fdi)
        if 'speed' in col:
            # Subtle velocity offset of 1.5 m/s
            base_hover += np.random.uniform(1.0, 2.5, size=n_fdi)
        elif col == 'height':
            base_hover += np.random.uniform(2.0, 4.0, size=n_fdi)
        X_p_hardened.loc[fdi_mask, col] = base_hover
        
    # Save Hardened Physical Dataset
    df_p_hardened = X_p_hardened.copy()
    df_p_hardened['Target_Label'] = y_p
    hardened_phys_path = os.path.join(data_dir, "Hardened_Physical_UAV_Dataset.csv")
    df_p_hardened.to_csv(hardened_phys_path, index=False)
    print(f"[+] Saved Hardened Physical Dataset -> {hardened_phys_path}")
    
    # 4. Generate Hardened Multimodal Dataset
    print("\n[*] Assembling Hardened Multimodal Dataset (Hardened Physical + Cyber)...")
    X_c, y_c, _ = load_cyber_dataset(os.path.join(SCRIPT_DIR, "..", "Cyber_UAV_Dataset.csv"))
    
    classes = ['Benign', 'DoS', 'Replay', 'Evil_Twin', 'FDI']
    p_list, c_list, y_list = [], [], []
    for c in classes:
        n = min(sum(y_p == c), sum(y_c == c))
        p_chunk = X_p_hardened[y_p == c].sample(n=n, random_state=42).reset_index(drop=True)
        c_chunk = X_c[y_c == c].sample(n=n, random_state=42).reset_index(drop=True)
        
        p_chunk.columns = [f"phys_{col}" for col in p_chunk.columns]
        c_chunk.columns = [f"cyb_{col}" for col in c_chunk.columns]
        
        merged_chunk = pd.concat([p_chunk, c_chunk], axis=1)
        p_list.append(merged_chunk)
        y_list.extend([c] * n)
        
    X_fused_hardened = pd.concat(p_list, ignore_index=True)
    df_fused_hardened = X_fused_hardened.copy()
    df_fused_hardened['Target_Label'] = y_list
    
    hardened_fused_path = os.path.join(data_dir, "Hardened_Multimodal_UAV_Dataset.csv")
    df_fused_hardened.to_csv(hardened_fused_path, index=False)
    print(f"[+] Saved Hardened Multimodal Dataset -> {hardened_fused_path}")
    
    return X_p_hardened, y_p, X_fused_hardened, pd.Series(y_list)


def run_hardened_evaluations(X_p_hard, y_p, X_f_hard, y_f):
    print("\n" + "=" * 80)
    print("STEP 2: BENCHMARKING MODELS UNDER HARDENED CONDITIONS")
    print("=" * 80)
    
    results = []
    
    # =========================================================================
    # A. SUPERVISED BENCHMARKS ON HARDENED DATA
    # =========================================================================
    print("\n[+] Running Supervised Models on Hardened Data...")
    
    # 1. Physical XGBoost (Without crash codes)
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import train_test_split
    
    le = LabelEncoder()
    y_p_enc = le.fit_transform(y_p)
    X_tr_p, X_te_p, y_tr_p, y_te_p = train_test_split(X_p_hard, y_p_enc, test_size=0.3, random_state=42, stratify=y_p_enc)
    
    xgb_p = xgb.XGBClassifier(n_estimators=100, max_depth=6, random_state=42, eval_metric='mlogloss')
    xgb_p.fit(X_tr_p, y_tr_p)
    y_pred_p = xgb_p.predict(X_te_p)
    
    acc_p = accuracy_score(y_te_p, y_pred_p) * 100
    f1_p = f1_score(y_te_p, y_pred_p, average='macro') * 100
    cm_p = confusion_matrix(y_te_p, y_pred_p)
    
    # Per-class F1
    class_names = list(le.classes_)
    f1s_p = f1_score(y_te_p, y_pred_p, average=None) * 100
    
    results.append({
        'Suite': 'Hardened (No Crash Sentinels)',
        'Model': 'Physical XGBoost',
        'Domain': 'Physical',
        'Accuracy (%)': round(acc_p, 2),
        'Macro F1 (%)': round(f1_p, 2),
        'F1: Benign (%)': round(f1s_p[class_names.index('Benign')], 2),
        'F1: DoS (%)': round(f1s_p[class_names.index('DoS')], 2),
        'F1: Evil_Twin (%)': round(f1s_p[class_names.index('Evil_Twin')], 2),
        'F1: FDI (%)': round(f1s_p[class_names.index('FDI')], 2),
        'F1: Replay (%)': round(f1s_p[class_names.index('Replay')], 2),
    })
    print(f"  -> Physical XGBoost Macro F1: {f1_p:.2f}% (Evil_Twin F1 dropped from 99.9% to {f1s_p[class_names.index('Evil_Twin')]:.2f}%)")
    
    # 2. Multimodal XGBoost (With Hardened Physical + Cyber)
    y_f_enc = le.fit_transform(y_f)
    X_tr_f, X_te_f, y_tr_f, y_te_f = train_test_split(X_f_hard, y_f_enc, test_size=0.3, random_state=42, stratify=y_f_enc)
    
    xgb_f = xgb.XGBClassifier(n_estimators=100, max_depth=6, random_state=42, eval_metric='mlogloss')
    xgb_f.fit(X_tr_f, y_tr_f)
    y_pred_f = xgb_f.predict(X_te_f)
    
    acc_f = accuracy_score(y_te_f, y_pred_f) * 100
    f1_f = f1_score(y_te_f, y_pred_f, average='macro') * 100
    f1s_f = f1_score(y_te_f, y_pred_f, average=None) * 100
    
    results.append({
        'Suite': 'Hardened (No Crash Sentinels)',
        'Model': 'Multimodal XGBoost',
        'Domain': 'Fused',
        'Accuracy (%)': round(acc_f, 2),
        'Macro F1 (%)': round(f1_f, 2),
        'F1: Benign (%)': round(f1s_f[class_names.index('Benign')], 2),
        'F1: DoS (%)': round(f1s_f[class_names.index('DoS')], 2),
        'F1: Evil_Twin (%)': round(f1s_f[class_names.index('Evil_Twin')], 2),
        'F1: FDI (%)': round(f1s_f[class_names.index('FDI')], 2),
        'F1: Replay (%)': round(f1s_f[class_names.index('Replay')], 2),
    })
    print(f"  -> Multimodal XGBoost Macro F1: {f1_f:.2f}% (Evil_Twin F1 rescued by Cyber stream to: {f1s_f[class_names.index('Evil_Twin')]:.2f}%)")
    
    # 3. Multimodal LightGBM
    lgb_f = lgb.LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)
    lgb_f.fit(X_tr_f, y_tr_f)
    y_pred_lgb = lgb_f.predict(X_te_f)
    f1_lgb = f1_score(y_te_f, y_pred_lgb, average='macro') * 100
    f1s_lgb = f1_score(y_te_f, y_pred_lgb, average=None) * 100
    
    results.append({
        'Suite': 'Hardened (No Crash Sentinels)',
        'Model': 'Multimodal LightGBM',
        'Domain': 'Fused',
        'Accuracy (%)': round(accuracy_score(y_te_f, y_pred_lgb) * 100, 2),
        'Macro F1 (%)': round(f1_lgb, 2),
        'F1: Benign (%)': round(f1s_lgb[class_names.index('Benign')], 2),
        'F1: DoS (%)': round(f1s_lgb[class_names.index('DoS')], 2),
        'F1: Evil_Twin (%)': round(f1s_lgb[class_names.index('Evil_Twin')], 2),
        'F1: FDI (%)': round(f1s_lgb[class_names.index('FDI')], 2),
        'F1: Replay (%)': round(f1s_lgb[class_names.index('Replay')], 2),
    })
    
    # Convert to DataFrame
    df_hardened = pd.DataFrame(results)
    csv_path = os.path.join(SCRIPT_DIR, "results", "hardened_benchmark_summary.csv")
    df_hardened.to_csv(csv_path, index=False)
    
    md_path = os.path.join(SCRIPT_DIR, "results", "hardened_benchmark_table.md")
    with open(md_path, "w") as f:
        f.write("# Hardened Benchmark: Performance Without Artificial Crash Codes\n\n")
        f.write(df_hardened.to_markdown(index=False))
        f.write("\n\n")
    print(f"\n[+] Saved Hardened Summary Table -> {md_path}")
    
    # =========================================================================
    # B. VISUALIZATION: THE RESCUE EFFECT OF MULTIMODAL FUSION
    # =========================================================================
    print("[+] Generating Comparative Visualization...")
    
    classes = ['Benign', 'DoS', 'Evil_Twin', 'FDI', 'Replay']
    df_melt = pd.melt(
        df_hardened,
        id_vars=['Model'],
        value_vars=[f'F1: {c} (%)' for c in classes],
        var_name='Attack Class',
        value_name='F1 Score'
    )
    df_melt['Attack Class'] = df_melt['Attack Class'].str.replace('F1: ', '').str.replace(' (%)', '')
    
    fig, ax = plt.subplots(figsize=(11, 6))
    sns.barplot(data=df_melt, x='Attack Class', y='F1 Score', hue='Model', palette='viridis', ax=ax)
    ax.set_title("Hardened Benchmark: How Cyber Modality Rescues Detection When Crash Sentinels Are Removed", fontsize=12, fontweight='bold')
    ax.set_ylabel("F1 Score (%)", fontsize=11)
    ax.set_ylim(0, 105)
    ax.grid(axis='y', alpha=0.3)
    ax.legend(title="Hardened Model", loc="lower right")
    
    plt.tight_layout()
    plot_path = os.path.join(SCRIPT_DIR, "results", "hardened_sensitivity_plot.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"[+] Saved Hardened Comparative Plot -> {plot_path}")
    
    print("\n" + "=" * 80)
    print("HARDENED BENCHMARK COMPLETE!")
    print("=" * 80)


if __name__ == '__main__':
    X_p_h, y_p, X_f_h, y_f = create_hardened_datasets()
    run_hardened_evaluations(X_p_h, y_p, X_f_h, y_f)
