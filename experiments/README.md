# Experiments Suite: UAV Cyber-Physical Attack Detection

This directory contains modular, leak-free supervised and unsupervised learning benchmarks, deep autoencoders, and multimodal fusion architectures for cyber-physical attack detection and zero-day anomaly detection on Unmanned Aerial Vehicles (UAVs).

---

## 1. Directory Structure

```
experiments/
├── 00_master_benchmark_summary.ipynb       # Supervised Executive summary & comparative plots
├── 01_physical_random_forest.ipynb         # Random Forest on Physical telemetry (leak-free)
├── 02_cyber_random_forest.ipynb            # Random Forest on Cyber network traffic (5 classes)
├── 03_physical_svm.ipynb                   # Linear & RBF SVM on Physical telemetry
├── 04_cyber_svm.ipynb                      # Linear & Nystroem Kernel SVM on Cyber traffic
├── 05_mlp_neural_network.ipynb             # Multi-Layer Perceptron (Feed-Forward ANN)
├── 06_lightgbm_benchmark.ipynb             # LightGBM (Physical & Cyber)
├── 07_xgboost_benchmark.ipynb              # XGBoost (Physical & Cyber)
├── 08_multimodal_fusion.ipynb              # Supervised Multimodal Fusion (94.2% Acc, 87.0% F1)
│
├── 09_unsupervised_physical_anomalies.ipynb# Unsupervised Anomaly Detection on Physical Telemetry
├── 10_unsupervised_cyber_anomalies.ipynb   # Unsupervised Anomaly Detection on Cyber Traffic
├── 11_deep_autoencoder_anomaly_detection.ipynb # Deep Bottleneck Autoencoder Architectures
├── 12_multimodal_unsupervised_fusion.ipynb # Multimodal Anomaly Fusion (96.55% ROC-AUC)
├── 13_zero_day_leave_one_attack_out.ipynb  # Leave-One-Attack-Out (LOAO) Zero-Day Benchmark
│
├── generate_master_report.py               # Supervised benchmark runner & plot generator
├── generate_unsupervised_report.py         # Unsupervised benchmark runner & plot generator
├── build_unsupervised_notebooks.py         # Modular notebook builder
│
├── utils/
│   ├── data_loader.py                      # Leakage-free 5-class loader & novelty split utilities
│   ├── metrics.py                          # Academic supervised classification metrics
│   └── unsupervised_metrics.py             # Academic anomaly detection metrics (AUC, FAR, Latency)
└── results/
    ├── master_metrics_summary.csv          # Supervised benchmark results (20+ models)
    ├── master_metrics_table.md             # Supervised Markdown publication table
    ├── model_comparison_plots.png          # Supervised visual comparison plots
    ├── latency_vs_f1_pareto.png            # Supervised edge deployment Pareto frontier
    │
    ├── unsupervised_master_metrics.csv     # Unsupervised benchmark results across all domains
    ├── unsupervised_master_table.md        # Unsupervised Markdown publication table
    ├── unsupervised_roc_pr_curves.png      # Publication ROC and PR curves
    ├── unsupervised_per_attack_recall.png  # Per-attack zero-day detection recall rates
    ├── unsupervised_latency_vs_auc_pareto.png # Edge latency vs. ROC-AUC Pareto frontier
    └── unsupervised_score_distributions.png# KDE distributions of Benign vs Attack scores
```

---

## 2. Canonical Attack Taxonomy (5 Classes)
Both domains are standardized to evaluate the exact same 5 operational states:
1. **`Benign`**: Normal flight and standard network telemetry.
2. **`DoS`**: Denial-of-service Wi-Fi packet flooding.
3. **`Replay`**: Replaying captured legitimate flight commands.
4. **`Evil_Twin`**: Rogue Access Point spoofing the Ground Control Station.
5. **`FDI`**: False Data Injection tampering with attitude and position estimates.

---

## 3. Dataset Preprocessing & "Unlabeled Records" Resolution
In raw exports of `Dataset_T-ITS.csv` (54,783 total rows), records appeared unlabelled when viewing the `class` column in isolation (21,679 NaN values). Investigation revealed:
1. **Interleaved Data Modalities**: The file contains 12,520 physical telemetry rows interleaved with 42,262 cyber packet rows. Physical sensor rows have a different schema, leaving Wi-Fi header columns blank.
2. **Shifted Label Columns**: Cyber `Evil_Twin` (5,684 packets) and `FDI` (3,474 packets) labels were stored in column 38 (`Unnamed: 38`), not `class`.
3. **Complete Reconstruction**: Our `data_loader.py` merges `Unnamed: 38` and `class`, and maps the physical chunks to their canonical ground truth. **All 12,520 physical samples and all 42,260 cyber samples are fully labeled and accounted for with zero data loss.**

---

## 4. Leakage Prevention ("Honest" Features)
- **Physical Features (9 genuine sensors):** `height`, `x_speed`, `y_speed`, `z_speed`, `pitch`, `roll`, `yaw`, `mp_distance_y`, `mp_distance_z`. Elapsed-time proxies (`flight_time`, `battery`, `timestamp_p`, `barometer`, `temperature`) are excluded.
- **Cyber Features (28 packet metrics):** Frame lengths, WLAN protocol flags, sequence numbers, TCP/UDP port behaviors. Identifier addresses (`ip.src`, `ip.dst`, `wlan.sa/ta/da/ra`) and sequence counters (`frame.number`, `packet_id`, `timestamp_c`) are excluded.

---

## 5. Supervised Learning Highlights
- **Multimodal Fusion Breakthrough:** Fusing Cyber (28 features) and Physical (9 features) modalities achieves **94.20% Accuracy, 87.00% Macro F1**, and drives False Alarm Rate (FAR) down to **0.54%** at **~1.69 $\mu$s/sample** latency.

---

## 6. Unsupervised Learning & Anomaly Detection Highlights
In operational UAV deployments, zero-day attacks lack prior signatures. We evaluated semi-supervised novelty detection (models trained **strictly on Benign normal flight**) and Leave-One-Attack-Out (LOAO) zero-day testing:

| Model | Domain | ROC-AUC (%) | PR-AUC (%) | Anomaly F1 (%) | False Alarm Rate (%) | Latency ($\mu$s/sample) | Key Detection Strengths |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Multimodal PCA Reconstruction** | **Fused** | **96.55** | **99.47** | **96.23** | **12.12** | **0.91** | FDI (100%), Evil_Twin (100%), DoS (84.4%) |
| **Multimodal Deep Autoencoder** | **Fused** | **95.93** | **99.38** | **95.81** | **19.66** (Best F1)<br>**5.36** (Calibrated) | **19.79** | 100% Zero-Day Recall on FDI & Evil_Twin at $\le 5\%$ FAR |
| **Multimodal Isolation Forest** | **Fused** | **94.01** | **98.99** | **95.39** | **5.36** (Calibrated) | **2.49** | Fast tree partitioning, edge deployable |
| **Cyber Deep Autoencoder** | Cyber | 92.92 | 99.35 | 92.23 (5% FAR) | 6.05 | 15.30 | DoS (83.7%), Replay (77.7%) |
| **Physical Isolation Forest** | Physical | 89.19 | 98.34 | 87.25 (5% FAR) | 3.42 | 2.50 | FDI (99.9%), Evil_Twin (100%) |

### Key Takeaway: Cross-Layer Complementarity
- **Physical Models Alone:** Incapable of detecting network packet floods (`DoS` recall $\approx 3.6\%$) because a hovering UAV's kinematics are not immediately disrupted.
- **Cyber Models Alone:** Weaker at detecting subtle kinematic deviations without physical state feedback.
- **Multimodal Unsupervised Fusion:** Eliminates single-domain blindspots, providing true zero-day protection across both the cyber network and physical avionics.

---

## 7. How to Run
All experiments use the pre-configured virtual environment `.venv`:
```bash
# To regenerate all supervised benchmarks, plots, and tables:
.venv/bin/python experiments/generate_master_report.py

# To regenerate all unsupervised & anomaly detection benchmarks, plots, and tables:
.venv/bin/python experiments/generate_unsupervised_report.py
```
