# UAV Cyber-Physical Simulation & Live Intrusion Detection (IDS)

This directory contains the real-time simulation testbed connecting our trained dual-layer detection engine to live MAVLink flight streams from **ArduPilot SITL** and **QGroundControl**.

---

## 1. Multi-Layer Architecture

```
                  ┌─────────────────────────────────────────┐
                  │   ArduPilot SITL Drone Telemetry        │
                  │   (MAVLink UDP Stream :14552)           │
                  └──────────────────┬──────────────────────┘
                                     │
                    ┌────────────────┴────────────────┐
                    ▼                                 ▼
       ┌─────────────────────────┐       ┌─────────────────────────┐
       │   LAYER 1: PHYSICS      │       │   LAYER 2: TEMPORAL ML  │
       │   Kinematic Invariants  │       │   Rolling Window (W=10) │
       │   (Mahalanobis / Chi²)  │       │   Multimodal XGBoost    │
       ├─────────────────────────┤       ├─────────────────────────┤
       │ • Latency: < 1 µs       │       │ • Sliding window W=10   │
       │ • Zero ML parameters    │       │ • Latency: ~150 µs      │
       │ • Invariants monitored: │       │ • Multimodal features:  │
       │   - Δh vs vz*Δt         │       │   - Packet rate / jitter│
       │   - a_mag ≤ 15 m/s²     │       │   - MAVLink packet sizes│
       │   - Tilt vs acc coupling│       │   - Kinematic rolling   │
       │ • Catches:              │       │ • Catches:              │
       │   - Instant FDI jumps   │       │   - DoS packet floods   │
       │   - Sensor spoofing     │       │   - Replay attacks      │
       │   - Coordinate hacking  │       │   - Evil Twin rogue APs │
       └────────────┬────────────┘       └────────────┬────────────┘
                    │                                 │
                    └────────────────┬────────────────┘
                                     ▼
                  ┌─────────────────────────────────────────┐
                  │     LIVE TERMINAL HUD & ALERT ENGINE    │
                  │ (Real-Time Kinematics & Incident Log)   │
                  └─────────────────────────────────────────┘
```

---

## 2. Directory Structure

```
experiments/simulation/
├── models/
│   └── detector_checkpoint.pkl     # Trained weights, scaler & physics invariants
├── train_detector_model.py         # Production training & calibration script
├── live_ids_detector.py            # Real-time MAVLink listener & Terminal HUD
├── mock_uav_telemetry.py           # Standalone mock flight generator (for quick tests)
├── detector_events.log             # Persistent incident audit trail
├── attacks/
│   ├── attack_fdi.py               # False Data Injection (high-g coordinate jumps)
│   ├── attack_dos.py               # Denial-of-Service packet flood
│   ├── attack_replay.py            # Authentic telemetry sniff & playback
│   └── attack_evil_twin.py         # Rogue GCS heartbeat & command spoofing
```

---

## 3. Environments & Prerequisites

There are **two distinct virtual environments** used in this workflow:

1. **ArduPilot SITL Environment (`~/ardupilot/ArduCopter/.venv`):**
   * Manages ArduPilot build tools, MAVProxy, and SITL vehicle simulation.
   * **Crucial Requirement:** Must have `empy==3.3.4` (ArduPilot's WAF build system is incompatible with `empy` 4.x).
   ```bash
   cd ~/ardupilot/ArduCopter
   source .venv/bin/activate
   pip install "empy==3.3.4"
   ```

2. **Project ML & Detector Environment (`/home/average_sapien/Documents/ml_cyber_attack_dection/.venv`):**
   * Manages the Dual-Layer IDS detector, mission flight controller, attack injection suite, and evaluation engine.
   * Requires: `pymavlink`, `xgboost`, `scikit-learn`, `pandas`, `numpy`, `matplotlib`.
   ```bash
   cd /home/average_sapien/Documents/ml_cyber_attack_dection
   .venv/bin/pip install pymavlink xgboost scikit-learn pandas numpy matplotlib
   ```

---

## 4. Multi-Terminal Execution Guide

Open **4 separate terminals** in Linux:

### Terminal 1: ArduPilot SITL Flight Simulator
Launches the ArduCopter physics simulation and forwards MAVLink telemetry to QGroundControl (port 14550), our IDS Detector (port 14552), and the Autonomous Mission Controller (port 14553):
```bash
cd ~/ardupilot/ArduCopter
source .venv/bin/activate
../Tools/autotest/sim_vehicle.py -v ArduCopter -f quad --out=udp:127.0.0.1:14550 --out=udp:127.0.0.1:14552 --out=udp:127.0.0.1:14553 --console
```
*Wait until the MAVProxy console finishes initializing and displays the `STABILIZE>` prompt.*
*(Optional manual takeoff in SITL console if not using `fly_mission.py`:)*
```text
mode guided
arm throttle
takeoff 15
```

### Terminal 2: QGroundControl (Visual Ground Station)
Opens the 3D map and telemetry HUD:
```bash
qgroundcontrol
```

### Terminal 3: Live Cyber-Physical Detector HUD
Listens on UDP port 14552 and renders the dual-layer detection dashboard:
```bash
cd /home/average_sapien/Documents/ml_cyber_attack_dection
.venv/bin/python experiments/simulation/live_ids_detector.py
```

### Terminal 4: Attack Injection Suite (On Demand)
Inject attacks to test detection live:

**Test 1: False Data Injection (FDI)**
```bash
cd /home/average_sapien/Documents/ml_cyber_attack_dection
.venv/bin/python experiments/simulation/attacks/attack_fdi.py
```
*Expected HUD Response:* Immediate **RED ALERT** in **Layer 1 (Physics Kinematic Invariant)** due to impossible vertical velocity / high-g acceleration.

**Test 2: Denial-of-Service (DoS) Packet Flood**
```bash
cd /home/average_sapien/Documents/ml_cyber_attack_dection
.venv/bin/python experiments/simulation/attacks/attack_dos.py --rate 400
```
*Expected HUD Response:* **RED ALERT** in **Layer 2 (Multimodal ML)** as packet frequencies spike and inter-arrival times collapse.

**Test 3: Telemetry Replay Attack**
```bash
cd /home/average_sapien/Documents/ml_cyber_attack_dection
.venv/bin/python experiments/simulation/attacks/attack_replay.py
```
*Expected HUD Response:* **RED ALERT** in **Layer 2 (Multimodal ML)** as the temporal sequence mismatch registers replayed loops.

**Test 4: Evil Twin / Rogue GCS Spoofing**
```bash
cd /home/average_sapien/Documents/ml_cyber_attack_dection
.venv/bin/python experiments/simulation/attacks/attack_evil_twin.py
```
*Expected HUD Response:* **RED ALERT** in **Layer 2 (Cyber Protocol & ML)** flagging conflicting rogue GCS heartbeats and unauthorized command injection.

---

## 5. Quick Verification (Without ArduPilot SITL)
If you want to test the detector HUD and attack scripts right away without launching ArduPilot:
1. In Terminal 1:
   ```bash
   .venv/bin/python experiments/simulation/mock_uav_telemetry.py
   ```
2. In Terminal 2:
   ```bash
   .venv/bin/python experiments/simulation/live_ids_detector.py
   ```
3. In Terminal 3: Run any attack script from `experiments/simulation/attacks/`.
