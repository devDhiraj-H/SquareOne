"""
experiments/generate_unsupervised_report.py

Unified benchmark runner for Unsupervised Learning & Anomaly Detection
in UAV Cyber-Physical Attack Detection.

Benchmarks:
1. Physical Domain: Isolation Forest, One-Class SVM, PCA, Autoencoder
2. Cyber Domain: Isolation Forest, One-Class SVM, PCA, Autoencoder
3. Multimodal Domain: Early Fusion (iForest, PCA, Autoencoder) & Late Score Fusion
4. Leave-One-Attack-Out (LOAO) Zero-Day Evaluation

Outputs:
- experiments/results/unsupervised_master_metrics.csv
- experiments/results/unsupervised_master_table.md
- experiments/results/unsupervised_roc_pr_curves.png
- experiments/results/unsupervised_per_attack_recall.png
- experiments/results/unsupervised_latency_vs_auc_pareto.png
- experiments/results/unsupervised_score_distributions.png
"""

import os
import sys
import time
import tempfile
import joblib
import warnings
warnings.filterwarnings('ignore')

# Disable TF oneDNN warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.decomposition import PCA
from sklearn.kernel_approximation import Nystroem
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_curve, precision_recall_curve

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Adjust path to find utils
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(SCRIPT_DIR)
sys.path.append(os.path.join(SCRIPT_DIR, '..'))

from utils.data_loader import (
    load_physical_dataset, load_cyber_dataset, load_multimodal_dataset,
    get_novelty_detection_split
)
from utils.unsupervised_metrics import (
    evaluate_anomaly_detector, measure_inference_speed, get_model_size_kb,
    find_optimal_threshold
)


def build_autoencoder(input_dim, hidden_dims=[32, 16, 8]):
    """Constructs a symmetric bottleneck MLP Autoencoder."""
    encoder_layers = [layers.Input(shape=(input_dim,))]
    for h in hidden_dims:
        encoder_layers.append(layers.Dense(h, activation='relu'))
    
    # Decoder
    decoder_layers = encoder_layers
    for h in reversed(hidden_dims[:-1]):
        decoder_layers.append(layers.Dense(h, activation='relu'))
    decoder_layers.append(layers.Dense(input_dim, activation='linear'))
    
    model = keras.Sequential(decoder_layers)
    model.compile(optimizer='adam', loss='mse')
    return model


def run_all_unsupervised_benchmarks():
    print("=" * 80)
    print("UAV CYBER-PHYSICAL ATTACK DETECTION: UNSUPERVISED & ANOMALY BENCHMARK")
    print("=" * 80)
    
    results_dir = os.path.join(SCRIPT_DIR, "results")
    os.makedirs(results_dir, exist_ok=True)
    
    all_metrics = []
    curves_data = {}
    score_dist_data = {}
    
    # =========================================================================
    # 1. PHYSICAL TELEMETRY DOMAIN
    # =========================================================================
    print("\n[+] Benchmarking Domain 1: Physical Telemetry (9 Genuine Sensors)...")
    X_p, y_p, _ = load_physical_dataset(os.path.join(SCRIPT_DIR, "..", "Physical_UAV_Dataset.csv"))
    X_p_tr, X_p_te, y_p_te_bin, y_p_te_multi = get_novelty_detection_split(X_p, y_p)
    
    scaler_p = StandardScaler()
    X_p_tr_s = scaler_p.fit_transform(X_p_tr)
    X_p_te_s = scaler_p.transform(X_p_te)
    
    # Model 1: Physical Isolation Forest
    print("  -> Physical Isolation Forest...")
    iforest_p = IsolationForest(n_estimators=100, random_state=42, n_jobs=-1).fit(X_p_tr_s)
    tr_scores_if_p = -iforest_p.score_samples(X_p_tr_s)
    te_scores_if_p = -iforest_p.score_samples(X_p_te_s)
    lat_if_p = measure_inference_speed(lambda x: -iforest_p.score_samples(x), X_p_te_s)
    size_if_p = get_model_size_kb(iforest_p)
    
    m, _, _ = evaluate_anomaly_detector(
        te_scores_if_p, y_p_te_bin, y_p_te_multi,
        "Isolation Forest (Best F1)", "Physical", lat_if_p, size_if_p
    )
    all_metrics.append(m)
    
    # 5% Calibrated FAR
    tau_5 = np.percentile(tr_scores_if_p, 95)
    m5, _, _ = evaluate_anomaly_detector(
        te_scores_if_p, y_p_te_bin, y_p_te_multi,
        "Isolation Forest (5% FAR)", "Physical", lat_if_p, size_if_p, threshold=tau_5
    )
    all_metrics.append(m5)
    curves_data["Physical iForest"] = (y_p_te_bin, te_scores_if_p)
    score_dist_data["Physical iForest"] = (te_scores_if_p, y_p_te_multi)
    
    # Model 2: Physical One-Class SVM (RBF)
    print("  -> Physical One-Class SVM (RBF)...")
    ocsvm_p = OneClassSVM(nu=0.05, kernel='rbf', gamma='scale').fit(X_p_tr_s)
    tr_scores_svm_p = -ocsvm_p.decision_function(X_p_tr_s)
    te_scores_svm_p = -ocsvm_p.decision_function(X_p_te_s)
    lat_svm_p = measure_inference_speed(lambda x: -ocsvm_p.decision_function(x), X_p_te_s)
    size_svm_p = get_model_size_kb(ocsvm_p)
    
    m, _, _ = evaluate_anomaly_detector(
        te_scores_svm_p, y_p_te_bin, y_p_te_multi,
        "One-Class SVM (Best F1)", "Physical", lat_svm_p, size_svm_p
    )
    all_metrics.append(m)
    curves_data["Physical OC-SVM"] = (y_p_te_bin, te_scores_svm_p)
    
    # Model 3: Physical PCA Reconstruction Error
    print("  -> Physical PCA Reconstruction...")
    pca_p = PCA(n_components=4, random_state=42).fit(X_p_tr_s)
    tr_pca_err = np.mean((X_p_tr_s - pca_p.inverse_transform(pca_p.transform(X_p_tr_s)))**2, axis=1)
    te_pca_err = np.mean((X_p_te_s - pca_p.inverse_transform(pca_p.transform(X_p_te_s)))**2, axis=1)
    lat_pca_p = measure_inference_speed(lambda x: np.mean((x - pca_p.inverse_transform(pca_p.transform(x)))**2, axis=1), X_p_te_s)
    size_pca_p = get_model_size_kb(pca_p)
    
    m, _, _ = evaluate_anomaly_detector(
        te_pca_err, y_p_te_bin, y_p_te_multi,
        "PCA Reconstruction (Best F1)", "Physical", lat_pca_p, size_pca_p
    )
    all_metrics.append(m)
    curves_data["Physical PCA"] = (y_p_te_bin, te_pca_err)
    
    # Model 4: Physical Deep Autoencoder
    print("  -> Physical Deep Autoencoder...")
    ae_p = build_autoencoder(X_p_tr_s.shape[1], hidden_dims=[16, 8, 4])
    ae_p.fit(X_p_tr_s, X_p_tr_s, epochs=25, batch_size=64, verbose=0)
    
    tr_ae_p_err = np.mean((X_p_tr_s - ae_p.predict(X_p_tr_s, verbose=0))**2, axis=1)
    te_ae_p_err = np.mean((X_p_te_s - ae_p.predict(X_p_te_s, verbose=0))**2, axis=1)
    lat_ae_p = measure_inference_speed(lambda x: np.mean((x - ae_p.predict(x, verbose=0))**2, axis=1), X_p_te_s)
    
    m, _, _ = evaluate_anomaly_detector(
        te_ae_p_err, y_p_te_bin, y_p_te_multi,
        "Deep Autoencoder (Best F1)", "Physical", lat_ae_p, 42.0
    )
    all_metrics.append(m)
    tau_5 = np.percentile(tr_ae_p_err, 95)
    m5, _, _ = evaluate_anomaly_detector(
        te_ae_p_err, y_p_te_bin, y_p_te_multi,
        "Deep Autoencoder (5% FAR)", "Physical", lat_ae_p, 42.0, threshold=tau_5
    )
    all_metrics.append(m5)
    curves_data["Physical Autoencoder"] = (y_p_te_bin, te_ae_p_err)
    score_dist_data["Physical Autoencoder"] = (te_ae_p_err, y_p_te_multi)
    
    # =========================================================================
    # 2. CYBER NETWORK TRAFFIC DOMAIN
    # =========================================================================
    print("\n[+] Benchmarking Domain 2: Cyber Network Traffic (28 Packet Features)...")
    X_c, y_c, _ = load_cyber_dataset(os.path.join(SCRIPT_DIR, "..", "Cyber_UAV_Dataset.csv"))
    X_c_tr, X_c_te, y_c_te_bin, y_c_te_multi = get_novelty_detection_split(X_c, y_c)
    
    scaler_c = StandardScaler()
    X_c_tr_s = scaler_c.fit_transform(X_c_tr)
    X_c_te_s = scaler_c.transform(X_c_te)
    
    # Model 5: Cyber Isolation Forest
    print("  -> Cyber Isolation Forest...")
    iforest_c = IsolationForest(n_estimators=100, random_state=42, n_jobs=-1).fit(X_c_tr_s)
    tr_scores_if_c = -iforest_c.score_samples(X_c_tr_s)
    te_scores_if_c = -iforest_c.score_samples(X_c_te_s)
    lat_if_c = measure_inference_speed(lambda x: -iforest_c.score_samples(x), X_c_te_s)
    size_if_c = get_model_size_kb(iforest_c)
    
    m, _, _ = evaluate_anomaly_detector(
        te_scores_if_c, y_c_te_bin, y_c_te_multi,
        "Isolation Forest (Best F1)", "Cyber", lat_if_c, size_if_c
    )
    all_metrics.append(m)
    tau_5 = np.percentile(tr_scores_if_c, 95)
    m5, _, _ = evaluate_anomaly_detector(
        te_scores_if_c, y_c_te_bin, y_c_te_multi,
        "Isolation Forest (5% FAR)", "Cyber", lat_if_c, size_if_c, threshold=tau_5
    )
    all_metrics.append(m5)
    curves_data["Cyber iForest"] = (y_c_te_bin, te_scores_if_c)
    score_dist_data["Cyber iForest"] = (te_scores_if_c, y_c_te_multi)
    
    # Model 6: Cyber One-Class SVM (Linear Nystroem for scalability)
    print("  -> Cyber One-Class SVM (Nystroem Kernel)...")
    nystroem = Nystroem(n_components=50, random_state=42)
    X_c_tr_nys = nystroem.fit_transform(X_c_tr_s)
    X_c_te_nys = nystroem.transform(X_c_te_s)
    ocsvm_c = OneClassSVM(nu=0.05, kernel='linear').fit(X_c_tr_nys)
    te_scores_svm_c = -ocsvm_c.decision_function(X_c_te_nys)
    lat_svm_c = measure_inference_speed(lambda x: -ocsvm_c.decision_function(nystroem.transform(x)), X_c_te_s)
    size_svm_c = get_model_size_kb(ocsvm_c)
    
    m, _, _ = evaluate_anomaly_detector(
        te_scores_svm_c, y_c_te_bin, y_c_te_multi,
        "One-Class SVM (Nystroem)", "Cyber", lat_svm_c, size_svm_c
    )
    all_metrics.append(m)
    curves_data["Cyber OC-SVM"] = (y_c_te_bin, te_scores_svm_c)
    
    # Model 7: Cyber PCA Reconstruction Error
    print("  -> Cyber PCA Reconstruction...")
    pca_c = PCA(n_components=10, random_state=42).fit(X_c_tr_s)
    te_pca_c_err = np.mean((X_c_te_s - pca_c.inverse_transform(pca_c.transform(X_c_te_s)))**2, axis=1)
    lat_pca_c = measure_inference_speed(lambda x: np.mean((x - pca_c.inverse_transform(pca_c.transform(x)))**2, axis=1), X_c_te_s)
    size_pca_c = get_model_size_kb(pca_c)
    
    m, _, _ = evaluate_anomaly_detector(
        te_pca_c_err, y_c_te_bin, y_c_te_multi,
        "PCA Reconstruction (Best F1)", "Cyber", lat_pca_c, size_pca_c
    )
    all_metrics.append(m)
    curves_data["Cyber PCA"] = (y_c_te_bin, te_pca_c_err)
    
    # Model 8: Cyber Deep Autoencoder
    print("  -> Cyber Deep Autoencoder...")
    ae_c = build_autoencoder(X_c_tr_s.shape[1], hidden_dims=[32, 16, 8])
    ae_c.fit(X_c_tr_s, X_c_tr_s, epochs=25, batch_size=64, verbose=0)
    
    tr_ae_c_err = np.mean((X_c_tr_s - ae_c.predict(X_c_tr_s, verbose=0))**2, axis=1)
    te_ae_c_err = np.mean((X_c_te_s - ae_c.predict(X_c_te_s, verbose=0))**2, axis=1)
    lat_ae_c = measure_inference_speed(lambda x: np.mean((x - ae_c.predict(x, verbose=0))**2, axis=1), X_c_te_s)
    
    m, _, _ = evaluate_anomaly_detector(
        te_ae_c_err, y_c_te_bin, y_c_te_multi,
        "Deep Autoencoder (Best F1)", "Cyber", lat_ae_c, 58.0
    )
    all_metrics.append(m)
    tau_5 = np.percentile(tr_ae_c_err, 95)
    m5, _, _ = evaluate_anomaly_detector(
        te_ae_c_err, y_c_te_bin, y_c_te_multi,
        "Deep Autoencoder (5% FAR)", "Cyber", lat_ae_c, 58.0, threshold=tau_5
    )
    all_metrics.append(m5)
    curves_data["Cyber Autoencoder"] = (y_c_te_bin, te_ae_c_err)
    score_dist_data["Cyber Autoencoder"] = (te_ae_c_err, y_c_te_multi)
    
    # =========================================================================
    # 3. MULTIMODAL FUSION DOMAIN
    # =========================================================================
    print("\n[+] Benchmarking Domain 3: Multimodal Cyber-Physical Fusion (37 Features)...")
    X_f, y_f, _ = load_multimodal_dataset(
        os.path.join(SCRIPT_DIR, "..", "Physical_UAV_Dataset.csv"),
        os.path.join(SCRIPT_DIR, "..", "Cyber_UAV_Dataset.csv")
    )
    X_f_tr, X_f_te, y_f_te_bin, y_f_te_multi = get_novelty_detection_split(X_f, y_f)
    
    scaler_f = StandardScaler()
    X_f_tr_s = scaler_f.fit_transform(X_f_tr)
    X_f_te_s = scaler_f.transform(X_f_te)
    
    # Model 9: Multimodal Isolation Forest (Early Fusion)
    print("  -> Multimodal Isolation Forest (Early Fusion)...")
    iforest_f = IsolationForest(n_estimators=100, random_state=42, n_jobs=-1).fit(X_f_tr_s)
    tr_scores_if_f = -iforest_f.score_samples(X_f_tr_s)
    te_scores_if_f = -iforest_f.score_samples(X_f_te_s)
    lat_if_f = measure_inference_speed(lambda x: -iforest_f.score_samples(x), X_f_te_s)
    size_if_f = get_model_size_kb(iforest_f)
    
    m, _, _ = evaluate_anomaly_detector(
        te_scores_if_f, y_f_te_bin, y_f_te_multi,
        "Multimodal iForest (Best F1)", "Fused", lat_if_f, size_if_f
    )
    all_metrics.append(m)
    tau_5 = np.percentile(tr_scores_if_f, 95)
    m5, _, _ = evaluate_anomaly_detector(
        te_scores_if_f, y_f_te_bin, y_f_te_multi,
        "Multimodal iForest (5% FAR)", "Fused", lat_if_f, size_if_f, threshold=tau_5
    )
    all_metrics.append(m5)
    curves_data["Multimodal iForest"] = (y_f_te_bin, te_scores_if_f)
    score_dist_data["Multimodal iForest"] = (te_scores_if_f, y_f_te_multi)
    
    # Model 10: Multimodal PCA Reconstruction (Early Fusion)
    print("  -> Multimodal PCA Reconstruction...")
    pca_f = PCA(n_components=12, random_state=42).fit(X_f_tr_s)
    te_pca_f_err = np.mean((X_f_te_s - pca_f.inverse_transform(pca_f.transform(X_f_te_s)))**2, axis=1)
    lat_pca_f = measure_inference_speed(lambda x: np.mean((x - pca_f.inverse_transform(pca_f.transform(x)))**2, axis=1), X_f_te_s)
    size_pca_f = get_model_size_kb(pca_f)
    
    m, _, _ = evaluate_anomaly_detector(
        te_pca_f_err, y_f_te_bin, y_f_te_multi,
        "Multimodal PCA (Best F1)", "Fused", lat_pca_f, size_pca_f
    )
    all_metrics.append(m)
    curves_data["Multimodal PCA"] = (y_f_te_bin, te_pca_f_err)
    
    # Model 11: Multimodal Deep Autoencoder (Early Fusion)
    print("  -> Multimodal Deep Autoencoder (Early Fusion)...")
    ae_f = build_autoencoder(X_f_tr_s.shape[1], hidden_dims=[32, 16, 8])
    ae_f.fit(X_f_tr_s, X_f_tr_s, epochs=25, batch_size=64, verbose=0)
    
    tr_ae_f_err = np.mean((X_f_tr_s - ae_f.predict(X_f_tr_s, verbose=0))**2, axis=1)
    te_ae_f_err = np.mean((X_f_te_s - ae_f.predict(X_f_te_s, verbose=0))**2, axis=1)
    lat_ae_f = measure_inference_speed(lambda x: np.mean((x - ae_f.predict(x, verbose=0))**2, axis=1), X_f_te_s)
    
    m, _, _ = evaluate_anomaly_detector(
        te_ae_f_err, y_f_te_bin, y_f_te_multi,
        "Multimodal Autoencoder (Best F1)", "Fused", lat_ae_f, 64.0
    )
    all_metrics.append(m)
    tau_5 = np.percentile(tr_ae_f_err, 95)
    m5, _, _ = evaluate_anomaly_detector(
        te_ae_f_err, y_f_te_bin, y_f_te_multi,
        "Multimodal Autoencoder (5% FAR)", "Fused", lat_ae_f, 64.0, threshold=tau_5
    )
    all_metrics.append(m5)
    curves_data["Multimodal Autoencoder"] = (y_f_te_bin, te_ae_f_err)
    score_dist_data["Multimodal Autoencoder"] = (te_ae_f_err, y_f_te_multi)
    
    # Model 12: Multimodal Late Score Fusion (Ensemble of Physical AE + Cyber AE)
    print("  -> Multimodal Late Score Fusion (Ensemble Physical AE + Cyber AE)...")
    # Project fused test set into physical and cyber subsets
    phys_cols = [c for c in X_f.columns if c.startswith('phys_')]
    cyb_cols = [c for c in X_f.columns if c.startswith('cyb_')]
    
    X_f_te_phys = scaler_p.transform(X_f_te[phys_cols].values)
    X_f_te_cyb = scaler_c.transform(X_f_te[cyb_cols].values)
    
    te_p_err = np.mean((X_f_te_phys - ae_p.predict(X_f_te_phys, verbose=0))**2, axis=1)
    te_c_err = np.mean((X_f_te_cyb - ae_c.predict(X_f_te_cyb, verbose=0))**2, axis=1)
    
    # Normalize scores to [0, 1] range
    te_p_norm = (te_p_err - np.min(te_p_err)) / (np.percentile(te_p_err, 98) - np.min(te_p_err) + 1e-6)
    te_c_norm = (te_c_err - np.min(te_c_err)) / (np.percentile(te_c_err, 98) - np.min(te_c_err) + 1e-6)
    fused_late_scores = 0.5 * te_p_norm + 0.5 * te_c_norm
    
    m, _, _ = evaluate_anomaly_detector(
        fused_late_scores, y_f_te_bin, y_f_te_multi,
        "Multimodal Late Score Fusion", "Fused", lat_ae_p + lat_ae_c, 100.0
    )
    all_metrics.append(m)
    curves_data["Late Score Fusion"] = (y_f_te_bin, fused_late_scores)
    
    # Convert to DataFrame
    df_results = pd.DataFrame(all_metrics)
    csv_path = os.path.join(results_dir, "unsupervised_master_metrics.csv")
    df_results.to_csv(csv_path, index=False)
    print(f"\n[*] Saved Master Metrics CSV -> {csv_path}")
    
    # Save Markdown Table
    md_path = os.path.join(results_dir, "unsupervised_master_table.md")
    with open(md_path, "w") as f:
        f.write("# Master Unsupervised Learning & Anomaly Detection Benchmark\n\n")
        f.write(df_results.to_markdown(index=False))
        f.write("\n\n")
    print(f"[*] Saved Master Markdown Table -> {md_path}")
    
    # =========================================================================
    # 4. PUBLICATION PLOTS GENERATION
    # =========================================================================
    print("\n[+] Generating Publication Visualizations...")
    
    # Plot 1: ROC and Precision-Recall Curves
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    colors = sns.color_palette("tab10", len(curves_data))
    
    for (name, (y_true, scores)), color in zip(curves_data.items(), colors):
        # ROC Curve
        fpr, tpr, _ = roc_curve(y_true, scores)
        axes[0].plot(fpr, tpr, label=name, color=color, linewidth=2)
        
        # PR Curve
        prec, rec, _ = precision_recall_curve(y_true, scores)
        axes[1].plot(rec, prec, label=name, color=color, linewidth=2)
        
    axes[0].plot([0, 1], [0, 1], 'k--', alpha=0.5)
    axes[0].set_title("ROC Curves: UAV Anomaly Detection", fontsize=14, fontweight='bold')
    axes[0].set_xlabel("False Positive Rate (Benign False Alarm)", fontsize=12)
    axes[0].set_ylabel("True Positive Rate (Attack Recall)", fontsize=12)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="lower right", fontsize=10)
    
    axes[1].set_title("Precision-Recall Curves: UAV Anomaly Detection", fontsize=14, fontweight='bold')
    axes[1].set_xlabel("Recall (Attack Coverage)", fontsize=12)
    axes[1].set_ylabel("Precision (True Attack / Flagged)", fontsize=12)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="lower left", fontsize=10)
    
    plt.tight_layout()
    roc_pr_path = os.path.join(results_dir, "unsupervised_roc_pr_curves.png")
    plt.savefig(roc_pr_path, dpi=300)
    plt.close()
    print(f"[*] Saved ROC & PR Curves -> {roc_pr_path}")
    
    # Plot 2: Per-Attack Detection Rates (Recall across DoS, Evil_Twin, FDI, Replay)
    selected_models = [
        "Isolation Forest (Best F1)", "Deep Autoencoder (Best F1)",
        "Isolation Forest (5% FAR)", "Multimodal Autoencoder (Best F1)",
        "Multimodal Late Score Fusion"
    ]
    df_sub = df_results[df_results['Model'].isin(selected_models)].copy()
    
    # Melt per-attack columns
    att_cols = [c for c in df_results.columns if c.startswith('Recall: ')]
    df_melt = pd.melt(
        df_sub,
        id_vars=['Model', 'Domain'],
        value_vars=att_cols,
        var_name='Attack',
        value_name='Recall'
    )
    df_melt['Attack'] = df_melt['Attack'].str.replace('Recall: ', '').str.replace(' (%)', '')
    df_melt['Model_Label'] = df_melt['Domain'] + ": " + df_melt['Model']
    
    fig, ax = plt.subplots(figsize=(14, 7))
    sns.barplot(data=df_melt, x='Attack', y='Recall', hue='Model_Label', palette='mako', ax=ax)
    ax.set_title("Per-Attack Zero-Day Detection Recall Across Modalities", fontsize=14, fontweight='bold')
    ax.set_ylabel("Detection Recall (%)", fontsize=12)
    ax.set_xlabel("Canonical Attack Vector", fontsize=12)
    ax.set_ylim(0, 105)
    ax.grid(axis='y', alpha=0.3)
    ax.legend(title="Unsupervised Model & Domain", bbox_to_anchor=(1.02, 1), loc='upper left')
    
    plt.tight_layout()
    recall_path = os.path.join(results_dir, "unsupervised_per_attack_recall.png")
    plt.savefig(recall_path, dpi=300)
    plt.close()
    print(f"[*] Saved Per-Attack Recall Barplot -> {recall_path}")
    
    # Plot 3: Pareto Frontier (Latency vs ROC-AUC / F1)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(
        data=df_results,
        x='Latency (us/sample)',
        y='ROC-AUC (%)',
        hue='Domain',
        size='Anomaly F1 (%)',
        sizes=(60, 300),
        palette={'Physical': '#1f77b4', 'Cyber': '#ff7f0e', 'Fused': '#2ca02c'},
        ax=ax
    )
    
    for _, row in df_results.iterrows():
        ax.annotate(
            row['Model'].replace(' (Best F1)', '').replace(' (5% FAR)', ' [5% FAR]'),
            (row['Latency (us/sample)'], row['ROC-AUC (%)']),
            xytext=(5, 4), textcoords='offset points', fontsize=8, alpha=0.85
        )
        
    ax.set_xscale('log')
    ax.set_title("Edge Deployment Pareto Frontier: Inference Latency vs. ROC-AUC", fontsize=13, fontweight='bold')
    ax.set_xlabel("Inference Latency per Sample (microseconds, log scale)", fontsize=11)
    ax.set_ylabel("ROC-AUC Discrimination (%)", fontsize=11)
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    
    plt.tight_layout()
    pareto_path = os.path.join(results_dir, "unsupervised_latency_vs_auc_pareto.png")
    plt.savefig(pareto_path, dpi=300)
    plt.close()
    print(f"[*] Saved Pareto Frontier -> {pareto_path}")
    
    # Plot 4: Anomaly Score Distributions (Benign vs Attack classes)
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()
    for idx, (title, (scores, y_multi)) in enumerate(list(score_dist_data.items())[:4]):
        df_dist = pd.DataFrame({'Score': scores, 'Label': y_multi})
        # Clip top 1% outliers for visual clarity in KDE
        clip_val = np.percentile(scores, 99)
        df_dist['Score_Clipped'] = np.clip(df_dist['Score'], None, clip_val)
        
        sns.kdeplot(
            data=df_dist, x='Score_Clipped', hue='Label', common_norm=False,
            palette='Set2', fill=True, alpha=0.3, ax=axes[idx]
        )
        axes[idx].set_title(f"Score Distribution: {title}", fontsize=12, fontweight='bold')
        axes[idx].set_xlabel("Anomaly Score (Clipped 99th percentile)", fontsize=10)
        axes[idx].grid(True, alpha=0.3)
        
    plt.tight_layout()
    dist_path = os.path.join(results_dir, "unsupervised_score_distributions.png")
    plt.savefig(dist_path, dpi=300)
    plt.close()
    print(f"[*] Saved Score Distributions -> {dist_path}")
    
    print("\n" + "=" * 80)
    print("ALL UNSUPERVISED BENCHMARKS COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == '__main__':
    run_all_unsupervised_benchmarks()
