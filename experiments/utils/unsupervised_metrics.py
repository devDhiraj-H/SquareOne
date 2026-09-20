"""
experiments/utils/unsupervised_metrics.py

Evaluation utilities for Unsupervised Learning & Anomaly Detection models
in UAV Cyber-Physical Attack Detection.

Metrics evaluated:
- ROC-AUC & PR-AUC (Threshold-independent discrimination)
- Calibrated Binary Metrics at optimal threshold (F1, Precision, Recall)
- False Alarm Rate (FAR = FPR on Benign test samples)
- Per-Attack Detection Rate / Recall (DoS, Evil_Twin, FDI, Replay)
- Edge Inference Latency (microseconds / sample)
- Model Footprint (KB)
"""

import os
import time
import tempfile
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score, average_precision_score, roc_curve, precision_recall_curve,
    f1_score, precision_score, recall_score, balanced_accuracy_score,
    confusion_matrix
)


def find_optimal_threshold(y_true_binary, anomaly_scores, metric="f1"):
    """
    Finds the score threshold that maximizes F1 score or balances TPR/FPR.
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true_binary, anomaly_scores)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    best_idx = np.argmax(f1_scores)
    
    if best_idx < len(thresholds):
        return thresholds[best_idx]
    return np.median(anomaly_scores)


def evaluate_anomaly_detector(
    scores,
    y_test_binary,
    y_test_multiclass,
    model_name="Model",
    domain="Physical",
    latency_us=0.0,
    model_size_kb=0.0,
    threshold=None
):
    """
    Evaluates an unsupervised anomaly detector given raw continuous anomaly scores.
    Higher score = more anomalous.
    """
    scores = np.asarray(scores, dtype=float)
    y_test_binary = np.asarray(y_test_binary, dtype=int)
    
    # 1. Threshold-independent metrics
    roc_auc = roc_auc_score(y_test_binary, scores)
    pr_auc = average_precision_score(y_test_binary, scores)
    
    # 2. Optimal / Calibrated threshold
    if threshold is None:
        threshold = find_optimal_threshold(y_test_binary, scores, metric="f1")
        
    y_pred_binary = (scores >= threshold).astype(int)
    
    # 3. Binary classification metrics
    f1 = f1_score(y_test_binary, y_pred_binary, zero_division=0)
    precision = precision_score(y_test_binary, y_pred_binary, zero_division=0)
    recall = recall_score(y_test_binary, y_pred_binary, zero_division=0)
    bal_acc = balanced_accuracy_score(y_test_binary, y_pred_binary)
    
    # 4. False Alarm Rate (FAR): FPR on Benign class
    benign_mask = (y_test_binary == 0)
    total_benign = np.sum(benign_mask)
    false_alarms = np.sum((y_pred_binary == 1) & benign_mask)
    far = (false_alarms / total_benign) * 100.0 if total_benign > 0 else 0.0
    
    # 5. Per-Attack Detection Rates (Recall for each attack)
    metrics = {
        "Model": model_name,
        "Domain": domain,
        "ROC-AUC (%)": round(roc_auc * 100.0, 2),
        "PR-AUC (%)": round(pr_auc * 100.0, 2),
        "Anomaly F1 (%)": round(f1 * 100.0, 2),
        "Balanced Acc (%)": round(bal_acc * 100.0, 2),
        "Precision (%)": round(precision * 100.0, 2),
        "Recall (%)": round(recall * 100.0, 2),
        "False Alarm Rate (%)": round(far, 2),
        "Latency (us/sample)": round(latency_us, 2),
        "Model Size (KB)": round(model_size_kb, 1),
        "Threshold": round(float(threshold), 4)
    }
    
    attack_classes = ['DoS', 'Evil_Twin', 'FDI', 'Replay']
    for attack in attack_classes:
        att_mask = (y_test_multiclass == attack).values if hasattr(y_test_multiclass, 'values') else (y_test_multiclass == attack)
        total_att = np.sum(att_mask)
        if total_att > 0:
            detected = np.sum((y_pred_binary == 1) & att_mask)
            att_recall = (detected / total_att) * 100.0
            metrics[f"Recall: {attack} (%)"] = round(att_recall, 2)
        else:
            metrics[f"Recall: {attack} (%)"] = np.nan
            
    return metrics, y_pred_binary, threshold


def measure_inference_speed(scorer_func, X_test, n_warmup=10):
    """
    Measures latency per sample in microseconds.
    """
    sample = X_test[:n_warmup]
    _ = scorer_func(sample)
    
    start = time.perf_counter()
    _ = scorer_func(X_test)
    elapsed = time.perf_counter() - start
    return (elapsed / len(X_test)) * 1e6


def get_model_size_kb(model):
    """
    Calculates serialized model size on disk in KB.
    """
    try:
        with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as tmp:
            joblib.dump(model, tmp.name)
            size_kb = os.path.getsize(tmp.name) / 1024.0
            os.unlink(tmp.name)
            return size_kb
    except Exception:
        return 0.0
