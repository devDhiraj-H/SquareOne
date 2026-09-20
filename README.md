# Autonomous UAV Cyber-Physical Intrusion Detection System (UAV-CPS-IDS)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![ArduPilot SITL](https://img.shields.io/badge/ArduPilot-SITL%204.x-orange.svg)](https://ardupilot.org/)
[![MAVLink 2.0](https://img.shields.io/badge/MAVLink-2.0-green.svg)](https://mavlink.io/)
[![XGBoost](https://img.shields.io/badge/ML-XGBoost%20%7C%20LightGBM-purple.svg)](https://xgboost.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A real-time, dual-layer cyber-physical intrusion detection and mitigation framework for autonomous Unmanned Aerial Vehicles (UAVs). The system couples microsecond-level deterministic Newtonian physics invariants with temporal gradient-boosted machine learning to detect multi-vector cyber-physical attacks on live MAVLink telemetry streams.

---

## 1. Overview & Research Motivation

Autonomous UAVs operate at the intersection of physical avionics (GPS, IMU, barometer) and cyber telemetry links (Wi-Fi, MAVLink, UDP). Existing intrusion detection systems (IDS) suffer from single-domain blindspots:

- **Physical-Only Detectors:** Blind to network packet floods during static hover (DoS recall on physical telemetry is ~3.6%).
- **Cyber-Only Detectors:** Blind to subtle kinematic spoofing if packet headers remain syntactically valid.

### The Dual-Layer Solution

```
                                  +---------------------------------------+
                                  |       ArduPilot SITL Simulator        |
                                  |    (Quad Copter-4.x Dynamics & EKF)   |
                                  +-------------------+-------------------+
                                                      | MAVLink UDP Telemetry
                                                      v
                                  +---------------------------------------+
                                  | 120 Hz Non-Blocking Ingestion Worker  |
                                  +-------------------+-------------------+
                                                      |
                               +----------------------+----------------------+
                               |                                             |
                               v                                             v
  +-----------------------------------------+   +-----------------------------------------+
  |        LAYER 1: PHYSICS REFLEX          |   |        LAYER 2: CYBER & PROTOCOL        |
  |  - Latency: < 1 microsecond             |   |  - Latency: ~150 microseconds           |
  |  - Zero-ML Deterministic Newtonian Laws |   |  - Per-SysID Protocol State Machine     |
  |  - Features:                            |   |  - Sliding Window (W=10) XGBoost        |
  |    * Vertical integration error (r_h)   |   |  - Features:                            |
  |    * Dynamic acceleration (a_mag)       |   |    * Packet rate & bad rate             |
  |    * Mahalanobis Chi-Square (chi^2)     |   |    * Timestamp & sequence regressions   |
  |  - Intercepts: False Data Injection     |   |    * Rogue GCS sysid & unauthorized cmd |
  +--------------------+--------------------+   |  - Intercepts: DoS, Replay, Evil Twin   |
                       |                        +--------------------+--------------------+
                       |                                             |
                       +----------------------+----------------------+
                                              |
                                              v
                               +-------------------------------+
                               |  Hierarchical Decision Gate   |
                               |  FDI -> DoS -> ET -> Replay   |
                               +--------------+----------------+
                                              |
                                              v
                               +-------------------------------+
                               | Live ANSI Heads-Up Display    |
                               | (7 Hz Decoupled Dashboard)    |
                               +-------------------------------+
```

1. **Layer 1 (Physics Invariant Reflex, $< 1\ \mu\text{s}$):** Instantaneous Newtonian kinematic consistency checks that evaluate vertical integration errors and dynamic acceleration limits, intercepting False Data Injection (FDI) with **0.0 ms alert latency**.
2. **Layer 2 (Cyber Protocols & Temporal ML, $\sim 150\ \mu\text{s}$):** Per-SysID sequence/timestamp monotonicity state machines combined with a sliding-window ($W=10$) multimodal XGBoost model, intercepting Denial-of-Service (DoS), Replay, and Rogue Ground Control Station (Evil Twin) attacks.

---

## 2. Canonical 5-Class Attack Taxonomy

All datasets, models, and real-time simulations adhere to a standardized 5-class operational taxonomy:

| Class | Domain | Attack Mechanism | Detection Strategy |
|---|---|---|---|
| **`Benign`** | Nominal | Legitimate autonomous waypoint navigation and nominal telemetry (115–125 pkts/s). | Baseline invariant envelope |
| **`DoS`** | Cyber | UDP packet floods ($\sim 900\text{ pkts/s}$) aimed at exhausting telemetry buffers. | Calibrated rate gate ($\ge 260\text{ pkts/s}$) |
| **`Replay`** | Cyber / Protocol | Sniffing authentic telemetry and replaying stale packets into the stream. | Timestamp & sequence monotonicity checks |
| **`Evil_Twin`** | Cyber / Protocol | Rogue GCS (`sysid=254`) emitting conflicting heartbeats and flight overrides. | Per-SysID isolation & authorization gate |
| **`FDI`** | Physical | Injecting false GPS coordinates ($25\text{ m}$ jumps, $30\text{ m/s}$ vertical climb). | Newtonian kinematic invariants ($r_h, a_{\text{mag}}, \chi^2$) |

---

## 3. Real-Time Performance & Verified Benchmarks

Evaluated over **12,739 live flight telemetry samples** on an autonomous 3D waypoint patrol mission:

### Multi-Class Performance (Live Simulation)

| Class | Precision | Recall | F1-Score | True Positives / Total |
|---|---|---|---|---|
| **Benign** | 99.16% | 99.17% | **99.17%** | 11,044 / 11,136 |
| **DoS** | 91.28% | 98.99% | **94.98%** | 492 / 497 |
| **Replay** | 96.26% | 83.23% | **89.27%** | 412 / 495 |
| **Evil_Twin** | 96.77% | 92.02% | **94.34%** | 150 / 163 |
| **FDI** | 91.46% | 97.99% | **94.61%** | 439 / 448 |

- **Overall Simulation Accuracy:** **98.41%**
- **Macro Average F1-Score:** **94.48%**
- **False Alarm Rate (FAR):** **0.83%** (industry standard target $< 1.0\%$)

### Detection Latency Profile

| Attack | First-Alert Latency | Detection Rate | Primary Mechanism |
|---|---|---|---|
| **FDI** | **0.0 ms** | 98.0% | Layer 1 Deterministic Physics Reflex |
| **DoS** | **152.0 ms** | 99.0% | Layer 2 Rolling Packet Rate Gate |
| **Replay** | **199.0 ms** | 83.2% | Layer 2 Protocol Monotonicity Gate |
| **Evil Twin** | **242.0 ms** | 92.0% | Layer 2 Per-SysID Authorization Gate |

---

## 4. Repository Structure

```
SquareOne/
├── README.md                                  # This documentation
├── requirements.txt                           # Exact Python dependencies
├── .gitignore                                 # Gitignore for ML, SITL & logs
├── professor.txt                              # Complete academic & thesis defense compendium
├── placement.txt                              # Technical interview & placement cheat sheet
│
├── Dataset_T-ITS.csv                          # Full canonical dataset (54,783 samples)
├── Physical_UAV_Dataset.csv                   # Leakage-free physical sensor dataset (12,520 rows)
├── Cyber_UAV_Dataset.csv                      # Leakage-free cyber network dataset (42,260 rows)
│
└── experiments/
    ├── 00_master_benchmark_summary.ipynb       # Supervised executive summary
    ├── 01_physical_random_forest.ipynb         # Random Forest on Physical telemetry
    ├── 02_cyber_random_forest.ipynb            # Random Forest on Cyber traffic
    ├── 03_physical_svm.ipynb                   # Linear & RBF SVM on Physical telemetry
    ├── 04_cyber_svm.ipynb                      # Linear & Nystroem Kernel SVM on Cyber traffic
    ├── 05_mlp_neural_network.ipynb             # Multi-Layer Perceptron (ANN)
    ├── 06_lightgbm_benchmark.ipynb             # LightGBM benchmarks
    ├── 07_xgboost_benchmark.ipynb              # XGBoost benchmarks
    ├── 08_multimodal_fusion.ipynb              # Supervised Multimodal Cyber-Physical Fusion
    ├── 09_unsupervised_physical_anomalies.ipynb# Unsupervised Physical Anomaly Detection
    ├── 10_unsupervised_cyber_anomalies.ipynb   # Unsupervised Cyber Anomaly Detection
    ├── 11_deep_autoencoder_anomaly_detection.ipynb # Deep Bottleneck Autoencoder
    ├── 12_multimodal_unsupervised_fusion.ipynb # Multimodal Anomaly Fusion
    ├── 13_zero_day_leave_one_attack_out.ipynb  # Leave-One-Attack-Out (LOAO) Zero-Day testing
    ├── 14_hardened_adversarial_benchmarks.ipynb# Hardened benchmarks without crash sentinels
    ├── 15_temporal_deep_sequence_modeling.ipynb# Temporal 1D-CNN, GRU, and Rolling XGBoost
    ├── 16_physics_informed_residual_monitoring.ipynb # Chi-Square kinematic invariant & PI-XGBoost
    │
    ├── utils/
    │   ├── data_loader.py                      # Leakage-free loader & novelty split utilities
    │   ├── metrics.py                          # Academic supervised metrics
    │   └── unsupervised_metrics.py             # Academic anomaly detection metrics (AUC, FAR)
    │
    ├── data/
    │   ├── Hardened_Multimodal_UAV_Dataset.csv
    │   └── Hardened_Physical_UAV_Dataset.csv
    │
    ├── results/                                # Benchmark summary CSVs, tables, and plots
    │   ├── master_metrics_summary.csv
    │   ├── master_metrics_table.md
    │   ├── temporal_benchmark_table.md
    │   ├── physics_informed_benchmark_table.md
    │   └── unsupervised_master_table.md
    │
    └── simulation/                             # Live closed-loop SITL simulation suite
        ├── SIMULATION_SETUP_AND_COMMANDS.md    # Multi-terminal step-by-step execution guide
        ├── fly_mission.py                      # Autonomous 3D waypoint patrol mission controller
        ├── live_ids_detector.py                # Real-time dual-layer MAVLink IDS detector HUD
        ├── record_flight_data.py               # Headless telemetry logger
        ├── evaluate_simulation_run.py          # Multi-class evaluation & timeline plot generator
        ├── stress_test.py                      # 11-scenario automated attack gauntlet
        ├── train_detector_model.py             # Dual-layer checkpoint retrainer
        ├── simulation_telemetry_dataset.csv    # Live flight telemetry log (12,739 samples)
        ├── simulation_attack_detection_timeline.png # 4-panel publication verification plot
        │
        ├── attacks/                            # UDP attack injectors
        │   ├── attack_fdi.py                   # False Data Injection (clock-synced)
        │   ├── attack_dos.py                   # UDP packet flood injector
        │   ├── attack_replay.py                # Sniff & Replay injector
        │   └── attack_evil_twin.py             # Rogue GCS & unauthorized command injector
        │
        └── models/
            └── detector_checkpoint.pkl         # Production model checkpoint
```

---

## 5. Quickstart & Installation

### 1. Environment Setup

Clone the repository and install dependencies:

```bash
git clone https://github.com/<your-username>/uav-cyber-physical-ids.git
cd uav-cyber-physical-ids

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install required packages
pip install -r requirements.txt
```

### 2. Run Offline Benchmarks

To regenerate benchmark tables and Pareto plots:

```bash
# Supervised benchmark (RF, SVM, MLP, LightGBM, XGBoost, Multimodal)
python experiments/generate_master_report.py

# Unsupervised & Zero-Day anomaly detection benchmark
python experiments/generate_unsupervised_report.py

# Temporal sequence modeling benchmark (1D-CNN, GRU, Rolling XGBoost)
python experiments/generate_temporal_report.py

# Physics-informed residual monitoring benchmark
python experiments/generate_physics_report.py
```

---

## 6. Real-Time Flight Simulation Guide

To run closed-loop simulation on ArduPilot SITL, open **4 terminal windows**:

### Terminal 1: Launch ArduPilot SITL
```bash
cd ~/ardupilot/ArduCopter
source .venv/bin/activate
../Tools/autotest/sim_vehicle.py -v ArduCopter -f quad \
  --out=udp:127.0.0.1:14550 \
  --out=udp:127.0.0.1:14552 \
  --out=udp:127.0.0.1:14553 \
  --console
```
*Wait until MAVProxy displays `STABILIZE>`.*

### Terminal 2: Start Autonomous Patrol Mission
```bash
python experiments/simulation/fly_mission.py --laps 3 --speed 5.0
```
*The drone automatically arms, takes off to 15m, and executes a 30m x 30m perimeter patrol.*

### Terminal 3: Launch Live Dual-Layer IDS HUD
```bash
python experiments/simulation/live_ids_detector.py --port 14552
```
*Renders live flight kinematics, Layer 1 invariants, Layer 2 packet rates, and active alert status.*

### Terminal 4: Inject On-Demand Attacks
```bash
# 1. False Data Injection (Physical Layer)
python experiments/simulation/attacks/attack_fdi.py --duration 15

# 2. Denial-of-Service Flood (Network Layer)
python experiments/simulation/attacks/attack_dos.py --duration 15

# 3. Telemetry Replay (Protocol Layer)
python experiments/simulation/attacks/attack_replay.py --sniff 5 --duration 15

# 4. Rogue GCS / Evil Twin (Authorization Layer)
python experiments/simulation/attacks/attack_evil_twin.py --duration 15

# Or run the automated 11-scenario gauntlet:
python experiments/simulation/stress_test.py
```

### Evaluate Post-Flight Results
```bash
python experiments/simulation/evaluate_simulation_run.py
```
*Generates precision/recall tables and saves `experiments/simulation/simulation_attack_detection_timeline.png`.*

---

## 7. Academic & Technical Interview Guides

This repository includes two comprehensive reference compendiums:

- **[`professor.txt`](professor.txt):** Complete academic thesis defense guide covering mathematical formulations of physics invariants, data leakage eradication, zero-day generalization, 8 engineering failures and their resolutions, and 10+ rigorous professor Q&As with follow-ups.
- **[`placement.txt`](placement.txt):** Technical interview cheat sheet containing the 30-second pitch, 2-minute deep dive, STAR-method engineering stories (socket bottlenecks, per-sysid tracking, `seq=0` protocol edge cases), and technical interview Q&As across System Design, ML Engineering, and Embedded Systems.

---

## 8. License & Citation

This project is licensed under the MIT License.

```bibtex
@article{uav_cyber_physical_ids_2026,
  title   = {Real-Time Dual-Layer Cyber-Physical Intrusion Detection System for Autonomous UAVs},
  author  = {Research Team},
  journal = {Autonomous Cyber-Physical Systems Security},
  year    = {2026}
}
```
