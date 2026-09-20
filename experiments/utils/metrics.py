"""
experiments/utils/metrics.py

Comprehensive evaluation utilities for UAV attack detection models.
Calculates academic research metrics:
- Accuracy, Balanced Accuracy, Macro F1, Weighted F1
- False Alarm Rate (FAR / FPR on Benign class)
- Inference Latency (microseconds per sample)
- Model Storage Footprint (KB)
"""

import time
import os
import sys
import tempfile
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, classification_report,
    confusion_matrix, f1_score, precision_score, recall_score
)


def compute_comprehensive_metrics(model, X_test, y_test, encoder, model_name="Model", domain="Physical"):
    """
    Evaluates a trained model across research, edge-efficiency, and security metrics.
    """
    # 1. Measure Inference Latency
    # Warm up
    _ = model.predict(X_test.iloc[:10] if hasattr(X_test, 'iloc') else X_test[:10])
    
    start_time = time.perf_counter()
    y_pred = model.predict(X_test)
    elapsed = time.perf_counter() - start_time
    latency_us_per_sample = (elapsed / len(X_test)) * 1e6
    
    # 2. Model Size on disk
    with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as tmp:
        joblib.dump(model, tmp.name)
        model_size_kb = os.path.getsize(tmp.name) / 1024.0
        os.unlink(tmp.name)

    # 3. Standard Classification Metrics
    acc = accuracy_score(y_test, y_pred)
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average='macro')
    weighted_f1 = f1_score(y_test, y_pred, average='weighted')
    macro_prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
    macro_rec = recall_score(y_test, y_pred, average='macro', zero_division=0)
    
    # 4. False Alarm Rate (FAR) for Benign
    class_names = [str(c) for c in encoder.classes_]
    cm = confusion_matrix(y_test, y_pred)
    
    # Identify Benign index
    far = np.nan
    if 'Benign' in class_names:
        benign_idx = class_names.index('Benign')
        # False Positives for Benign = True Benign predicted as Attack
        fp_benign = np.sum(cm[benign_idx, :]) - cm[benign_idx, benign_idx]
        total_benign = np.sum(cm[benign_idx, :])
        far = (fp_benign / total_benign) * 100.0 if total_benign > 0 else 0.0

    # 5. Build Results Dictionary
    metrics = {
        "Model": model_name,
        "Domain": domain,
        "Accuracy (%)": round(acc * 100.0, 2),
        "Balanced Acc (%)": round(bal_acc * 100.0, 2),
        "Macro F1 (%)": round(macro_f1 * 100.0, 2),
        "Weighted F1 (%)": round(weighted_f1 * 100.0, 2),
        "Macro Precision (%)": round(macro_prec * 100.0, 2),
        "Macro Recall (%)": round(macro_rec * 100.0, 2),
        "False Alarm Rate (%)": round(far, 2) if not np.isnan(far) else "N/A",
        "Latency (us/sample)": round(latency_us_per_sample, 2),
        "Model Size (KB)": round(model_size_kb, 1),
    }
    
    # Per-class F1 dictionary
    per_class_f1 = f1_score(y_test, y_pred, average=None, zero_division=0)
    for cls_name, f1_val in zip(class_names, per_class_f1):
        metrics[f"F1: {cls_name} (%)"] = round(f1_val * 100.0, 2)
        
    return metrics, y_pred, cm


def plot_confusion_matrix(cm, class_names, title="Confusion Matrix", save_path=None):
    """
    Plots a normalized confusion matrix heatmap.
    """
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_norm = np.nan_to_num(cm_norm)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm_norm, annot=True, fmt=".2%", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names, ax=ax
    )
    ax.set_title(title, fontsize=14, pad=12)
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
    plt.show()
    return fig
