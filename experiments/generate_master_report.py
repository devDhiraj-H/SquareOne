"""
experiments/generate_master_report.py

Runs and aggregates benchmarks across all supervised models (Random Forest, Extra Trees,
Linear/RBF SVM, MLP, LightGBM, XGBoost, and Multimodal Cyber-Physical Fusion).
Generates publication-ready summary tables, CSVs, and edge-efficiency Pareto frontier plots.
"""

import os
import sys
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import SGDClassifier
import lightgbm as lgb
import xgboost as xgb

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.data_loader import load_physical_dataset, load_cyber_dataset, get_stratified_split
from utils.metrics import compute_comprehensive_metrics


def run_all_benchmarks():
    print("==================================================")
    print("[*] STARTING COMPREHENSIVE UAV BENCHMARK SUITE")
    print("==================================================")
    
    results = []
    
    # -------------------------------------------------------------
    # 1. PHYSICAL TELEMETRY BENCHMARKS
    # -------------------------------------------------------------
    print("\n[+] Benchmarking PHYSICAL Domain Models...")
    X_p, y_p, feats_p = load_physical_dataset()
    X_tr_p, X_te_p, y_tr_p, y_te_p, enc_p = get_stratified_split(X_p, y_p, test_size=0.3, random_state=42)
    
    physical_models = [
        ("Random Forest (Default)", RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)),
        ("Random Forest (Balanced)", RandomForestClassifier(n_estimators=120, max_depth=18, class_weight="balanced", random_state=42, n_jobs=-1)),
        ("Extra Trees", ExtraTreesClassifier(n_estimators=120, max_features="sqrt", random_state=42, n_jobs=-1)),
        ("Linear SVM (Scaled)", Pipeline([('scaler', StandardScaler()), ('clf', LinearSVC(C=1.0, max_iter=3000, random_state=42))])),
        ("RBF SVM (Balanced)", Pipeline([('scaler', StandardScaler()), ('clf', SVC(C=10.0, gamma=0.1, class_weight='balanced', kernel='rbf', random_state=42))])),
        ("MLP Neural Net", Pipeline([('scaler', StandardScaler()), ('clf', MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300, alpha=1e-4, early_stopping=True, random_state=42))])),
        ("LightGBM (Default)", lgb.LGBMClassifier(n_estimators=100, learning_rate=0.08, num_leaves=31, random_state=42, n_jobs=-1, verbose=-1)),
        ("LightGBM (Balanced)", lgb.LGBMClassifier(n_estimators=120, learning_rate=0.05, num_leaves=31, class_weight='balanced', random_state=42, n_jobs=-1, verbose=-1)),
        ("XGBoost (Default)", xgb.XGBClassifier(n_estimators=100, learning_rate=0.08, max_depth=6, random_state=42, n_jobs=-1, eval_metric='mlogloss'))
    ]
    
    for name, model in physical_models:
        print(f"    -> Training {name}...")
        model.fit(X_tr_p, y_tr_p)
        m, _, _ = compute_comprehensive_metrics(model, X_te_p, y_te_p, enc_p, model_name=name, domain="Physical")
        results.append(m)
        
    # -------------------------------------------------------------
    # 2. CYBER NETWORK PACKET BENCHMARKS
    # -------------------------------------------------------------
    print("\n[+] Benchmarking CYBER Domain Models (Full 5 Classes)...")
    X_c, y_c, feats_c = load_cyber_dataset()
    X_tr_c, X_te_c, y_tr_c, y_te_c, enc_c = get_stratified_split(X_c, y_c, test_size=0.3, random_state=42)
    
    cyber_models = [
        ("Random Forest (Default)", RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)),
        ("Random Forest (Low FAR)", RandomForestClassifier(n_estimators=100, max_depth=18, random_state=42, n_jobs=-1)),
        ("Extra Trees", ExtraTreesClassifier(n_estimators=100, max_depth=18, random_state=42, n_jobs=-1)),
        ("Linear SVM (Scaled)", Pipeline([('scaler', StandardScaler()), ('clf', LinearSVC(C=1.0, max_iter=3000, random_state=42))])),
        ("Linear SVM (Balanced)", Pipeline([('scaler', StandardScaler()), ('clf', LinearSVC(C=1.0, class_weight='balanced', max_iter=3000, random_state=42))])),
        ("Nystroem RBF SVM", Pipeline([
            ('scaler', StandardScaler()),
            ('nystroem', Nystroem(kernel='rbf', gamma=0.05, n_components=150, random_state=42)),
            ('clf', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-4, max_iter=2000, random_state=42))
        ])),
        ("MLP Neural Net", Pipeline([('scaler', StandardScaler()), ('clf', MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=250, alpha=1e-4, early_stopping=True, random_state=42))])),
        ("LightGBM (Default)", lgb.LGBMClassifier(n_estimators=100, learning_rate=0.08, num_leaves=31, random_state=42, n_jobs=-1, verbose=-1)),
        ("LightGBM (Low FAR)", lgb.LGBMClassifier(n_estimators=120, learning_rate=0.05, num_leaves=25, subsample=0.85, random_state=42, n_jobs=-1, verbose=-1)),
        ("XGBoost (Default)", xgb.XGBClassifier(n_estimators=100, learning_rate=0.08, max_depth=6, random_state=42, n_jobs=-1, eval_metric='mlogloss'))
    ]
    
    for name, model in cyber_models:
        print(f"    -> Training {name}...")
        model.fit(X_tr_c, y_tr_c)
        m, _, _ = compute_comprehensive_metrics(model, X_te_c, y_te_c, enc_c, model_name=name, domain="Cyber")
        results.append(m)

    # -------------------------------------------------------------
    # 3. CYBER-PHYSICAL MULTIMODAL FUSION BENCHMARKS
    # -------------------------------------------------------------
    print("\n[+] Benchmarking CYBER-PHYSICAL MULTIMODAL FUSION Models...")
    classes = ['Benign', 'DoS', 'Replay', 'Evil_Twin', 'FDI']
    min_samples = {c: min(sum(y_p == c), sum(y_c == c)) for c in classes}
    
    p_list, c_list, y_list = [], [], []
    for c in classes:
        n = min_samples[c]
        p_chunk = X_p[y_p == c].sample(n=n, random_state=42).reset_index(drop=True)
        c_chunk = X_c[y_c == c].sample(n=n, random_state=42).reset_index(drop=True)
        p_chunk.columns = [f"phys_{col}" for col in p_chunk.columns]
        c_chunk.columns = [f"cyb_{col}" for col in c_chunk.columns]
        merged_chunk = pd.concat([p_chunk, c_chunk], axis=1)
        p_list.append(merged_chunk)
        y_list.extend([c] * n)
        
    X_fused = pd.concat(p_list, ignore_index=True)
    y_fused = pd.Series(y_list)
    X_tr_f, X_te_f, y_tr_f, y_te_f, enc_f = get_stratified_split(X_fused, y_fused, test_size=0.3, random_state=42)
    
    fusion_models = [
        ("Multimodal LightGBM", lgb.LGBMClassifier(n_estimators=120, learning_rate=0.06, random_state=42, n_jobs=-1, verbose=-1)),
        ("Multimodal XGBoost", xgb.XGBClassifier(n_estimators=120, learning_rate=0.06, max_depth=6, random_state=42, n_jobs=-1, eval_metric='mlogloss')),
        ("Multimodal Random Forest", RandomForestClassifier(n_estimators=120, random_state=42, n_jobs=-1)),
        ("Multimodal Extra Trees", ExtraTreesClassifier(n_estimators=120, random_state=42, n_jobs=-1))
    ]
    
    for name, model in fusion_models:
        print(f"    -> Training {name}...")
        model.fit(X_tr_f, y_tr_f)
        m, _, _ = compute_comprehensive_metrics(model, X_te_f, y_te_f, enc_f, model_name=name, domain="Fused")
        results.append(m)

    # -------------------------------------------------------------
    # 4. SAVE METRICS & GENERATE ARTIFACTS
    # -------------------------------------------------------------
    df = pd.DataFrame(results)
    
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(out_dir, exist_ok=True)
    
    csv_path = os.path.join(out_dir, "master_metrics_summary.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n[+] Saved master metrics to: {csv_path}")
    
    # Save Markdown Table
    md_path = os.path.join(out_dir, "master_metrics_table.md")
    cols = list(df.columns)
    header_str = "| " + " | ".join(cols) + " |\n"
    sep_str = "| " + " | ".join(["---"] * len(cols)) + " |\n"
    rows_str = ""
    for _, r in df.iterrows():
        rows_str += "| " + " | ".join(str(r[c]) for c in cols) + " |\n"
    with open(md_path, "w") as f:
        f.write("# Master Supervised Learning Benchmark: UAV Cyber Attack Detection\n\n")
        f.write(header_str + sep_str + rows_str)
        f.write("\n")
    print(f"[+] Saved Markdown table to: {md_path}")
    
    # -------------------------------------------------------------
    # 5. PLOT COMPARISONS & PARETO FRONTIER
    # -------------------------------------------------------------
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    
    # Plot 1: Domain Macro F1 Comparison
    domain_order = ["Physical", "Cyber", "Fused"]
    sns.boxplot(data=df, x="Domain", y="Macro F1 (%)", order=domain_order, ax=axes[0, 0], palette="Blues")
    sns.stripplot(data=df, x="Domain", y="Macro F1 (%)", order=domain_order, ax=axes[0, 0], color="black", size=6, jitter=0.2)
    axes[0, 0].set_title("Macro F1-Score Distribution across Domains (%)", fontsize=13)
    axes[0, 0].set_ylim(50, 100)
    
    # Plot 2: Top Models per Domain
    top_models = df.sort_values(by="Macro F1 (%)", ascending=False).groupby("Domain").head(3)
    sns.barplot(data=top_models, x="Model", y="Macro F1 (%)", hue="Domain", dodge=False, ax=axes[0, 1], palette="crest")
    axes[0, 1].set_title("Top 3 Models per Domain - Macro F1-Score (%)", fontsize=13)
    axes[0, 1].tick_params(axis='x', rotation=35)
    axes[0, 1].set_ylim(70, 100)
    
    # Plot 3: False Alarm Rates
    sns.barplot(data=df, x="Model", y="False Alarm Rate (%)", hue="Domain", ax=axes[1, 0], palette="Set2")
    axes[1, 0].set_title("False Alarm Rate (FAR on Benign) [%]", fontsize=13)
    axes[1, 0].tick_params(axis='x', rotation=55)
    
    # Plot 4: Latency Comparison (Log Scale)
    sns.barplot(data=df, x="Model", y="Latency (us/sample)", hue="Domain", ax=axes[1, 1], palette="Set1")
    axes[1, 1].set_title("Inference Latency (μs per sample, Log Scale)", fontsize=13)
    axes[1, 1].set_yscale('log')
    axes[1, 1].tick_params(axis='x', rotation=55)
    
    plt.tight_layout()
    chart_path = os.path.join(out_dir, "model_comparison_plots.png")
    plt.savefig(chart_path, dpi=300)
    plt.close()
    print(f"[+] Saved comparison charts to: {chart_path}")
    
    # Plot 5: Latency vs Macro F1 Pareto Frontier
    plt.figure(figsize=(10, 6.5))
    domain_styles = [("Physical", "o", "#1f77b4"), ("Cyber", "s", "#d62728"), ("Fused", "^", "#2ca02c")]
    for domain, marker, color in domain_styles:
        subset = df[df["Domain"] == domain]
        plt.scatter(subset["Latency (us/sample)"], subset["Macro F1 (%)"], label=domain, s=140, alpha=0.85, c=color, marker=marker, edgecolors='black')
        for _, row in subset.iterrows():
            short_name = row["Model"].replace(" (Scaled)", "").replace(" (Default)", "").replace("Multimodal ", "Fused ")
            plt.annotate(
                short_name,
                (row["Latency (us/sample)"], row["Macro F1 (%)"]),
                fontsize=7.5, xytext=(4, 2), textcoords="offset points"
            )
            
    plt.xscale('log')
    plt.xlabel("Inference Latency (μs / sample, Log Scale)", fontsize=12)
    plt.ylabel("Macro F1-Score (%)", fontsize=12)
    plt.title("UAV Edge Deployment Pareto Frontier: Macro-F1 vs. Inference Latency", fontsize=14)
    plt.legend(title="Domain", loc="lower left", fontsize=11)
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.tight_layout()
    pareto_path = os.path.join(out_dir, "latency_vs_f1_pareto.png")
    plt.savefig(pareto_path, dpi=300)
    plt.close()
    print(f"[+] Saved Pareto frontier plot to: {pareto_path}")
    
    print("\n==================================================")
    print("[*] COMPLETE BENCHMARK FINISHED!")
    print("==================================================")

if __name__ == "__main__":
    run_all_benchmarks()
