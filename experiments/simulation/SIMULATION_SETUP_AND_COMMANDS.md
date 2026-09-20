# UAV Cyber-Physical Simulation Setup & Manual Execution Guide

This document provides complete, copy-paste terminal instructions to run the autonomous flight simulation, live dual-layer Intrusion Detection System (IDS), attack injections, and evaluation engine.

---

## 1. Network Topology & Port Mapping

The simulation communicates using UDP-based MAVLink packets routed to 3 dedicated endpoints:

```
                      ┌─────────────────────────────────────────────────────────────┐
                      │                 ArduPilot SITL Simulator                    │
                      │   (sim_vehicle.py -v ArduCopter -f quad --console ...)      │
                      └──────────────────────────────┬──────────────────────────────┘
                                                     │ MAVLink UDP Broadcasts
                     ┌───────────────────────────────┼───────────────────────────────┐
                     ▼                               ▼                               ▼
            udp:127.0.0.1:14550             udp:127.0.0.1:14552             udp:127.0.0.1:14553
     ┌─────────────────────────────┐ ┌─────────────────────────────┐ ┌─────────────────────────────┐
     │  QGroundControl / Sniffers  │ │    Live IDS Detector HUD    │ │ Autonomous Mission Patrol   │
     │     (Port 14550)            │ │   (live_ids_detector.py)    │ │      (fly_mission.py)       │
     └─────────────────────────────┘ └─────────────────────────────┘ └─────────────────────────────┘
```

* **Port 14550:** Primary telemetry stream used by Ground Control Stations (QGroundControl) and attack sniffing tools (`attack_replay.py`).
* **Port 14552:** Dedicated telemetry stream continuously monitored and evaluated in real time by the dual-layer IDS detector HUD.
* **Port 14553:** Dedicated control link for the autonomous waypoint patrol controller (`fly_mission.py`).

---

## 2. Prerequisites & Environment Check

Notice that **two distinct virtual environments** are used:

### Environment 1: ArduPilot SITL (`~/ardupilot/ArduCopter/.venv`)
Used exclusively to compile and run ArduPilot SITL and MAVProxy.
* **Critical Requirement:** ArduPilot's WAF build system requires `empy==3.3.4` (version 4.x breaks template generators).
```bash
cd ~/ardupilot/ArduCopter
source .venv/bin/activate
pip install "empy==3.3.4"
```

### Environment 2: Project ML & Detector (`/home/average_sapien/Documents/ml_cyber_attack_dection/.venv`)
Used for all project code: `live_ids_detector.py`, `fly_mission.py`, attack injectors, and evaluation.
```bash
cd /home/average_sapien/Documents/ml_cyber_attack_dection
.venv/bin/python -c "import pymavlink, xgboost, sklearn, pandas, numpy, matplotlib; print('[+] All simulation & ML dependencies OK')"
```

---

## 3. Multi-Terminal Execution (Step-by-Step)

Open **4 separate Linux terminal windows or tabs**. Run each command in its designated terminal.

### Terminal 1: ArduPilot SITL Flight Simulator
Launches the ArduCopter physics simulation with multi-port telemetry routing:

```bash
cd ~/ardupilot/ArduCopter

# 1. Activate the ArduCopter virtual environment
source .venv/bin/activate

# 2. Launch SITL with all 3 output ports (14550 for GCS, 14552 for IDS, 14553 for Mission Controller)
../Tools/autotest/sim_vehicle.py -v ArduCopter -f quad --out=udp:127.0.0.1:14550 --out=udp:127.0.0.1:14552 --out=udp:127.0.0.1:14553 --console
```

*Wait until the MAVProxy console finishes initializing and displays the `STABILIZE>` prompt.*
*(Takeoff will be handled automatically by Terminal 2's `fly_mission.py`, or you can manually test takeoff with: `mode guided` -> `arm throttle` -> `takeoff 15`).*

---

### Terminal 2: Autonomous 3D Waypoint Patrol Mission
Launches the autonomous waypoint flight patrol around a 30m x 30m perimeter at 5 m/s:

```bash
cd /home/average_sapien/Documents/ml_cyber_attack_dection
.venv/bin/python experiments/simulation/fly_mission.py --laps 3 --speed 5.0
```

*Key Mission Features:*
* Detects if the drone was already manually taken off in Terminal 1 and takes over immediately.
* Performs banking turns and kinematic 3D maneuvers to realistically test physics and cyber invariants.
* Automatically initiates Return-To-Launch (RTL) and lands after completing the specified laps.

---

### Terminal 3: Live Dual-Layer IDS Detector HUD
Runs the real-time detection engine listening on UDP port 14552:

```bash
cd /home/average_sapien/Documents/ml_cyber_attack_dection
.venv/bin/python experiments/simulation/live_ids_detector.py --port 14552
```

*Dashboard Display Overview:*
* **Header Banner:** `[OK] ALL SYSTEMS NOMINAL` (Green) during legitimate flight, switching to `[!] ACTIVE ATTACK DETECTED` (Red) upon intrusion.
* **Kinematics:** Altitude, speed $(v_x, v_y, v_z)$, roll, pitch, yaw, and dynamic acceleration magnitude ($a_{\text{mag}}$).
* **Layer 1 (Physics Reflex, < 1 µs):** Instantaneous residual $\chi^2$, vertical integration error $r_h$, and acceleration bound ($15\text{ m/s}^2$).
* **Layer 2 (Cyber & Temporal ML, ~150 µs):** Telemetry stream rate (pkts/s), rolling temporal classifier confidence, and sequence/timestamp invariants.
* **Incident Log:** Automatically logs all telemetry and alert states to `experiments/simulation/simulation_telemetry_dataset.csv`.

---

### Terminal 4: On-Demand Cyber-Physical Attack Injections
While the drone is actively flying its patrol mission, inject attacks from this terminal:

#### 1. False Data Injection (FDI) Attack (Physical Layer)
Injects severe coordinate jumps ($25\text{ m}$ altitude oscillations and $30\text{ m/s}$ vertical climbs):
```bash
cd /home/average_sapien/Documents/ml_cyber_attack_dection
.venv/bin/python experiments/simulation/attacks/attack_fdi.py --duration 15
```
* **Expected Result:** Instant Layer 1 RED ALERT (`Excessive Dynamic Acceleration` or `Kinematic Discontinuity`, latency $0\text{ ms}$).

#### 2. Denial-of-Service (DoS) Attack (Network Layer)
Blasts high-rate UDP packet floods ($\sim 900\text{ pkts/s}$ vs nominal $\sim 115\text{ pkts/s}$):
```bash
cd /home/average_sapien/Documents/ml_cyber_attack_dection
.venv/bin/python experiments/simulation/attacks/attack_dos.py --duration 15
```
* **Expected Result:** Instant Layer 2 RED ALERT (`Packet Flood > 260 pkts/s`, latency $\approx 150\text{ ms}$).

#### 3. Telemetry Replay Attack (Protocol Layer)
Sniffs authentic flight packets for 5 seconds, then replays them back into the telemetry stream:
```bash
cd /home/average_sapien/Documents/ml_cyber_attack_dection
.venv/bin/python experiments/simulation/attacks/attack_replay.py --sniff 5 --duration 15
```
* **Expected Result:** Instant Layer 2 RED ALERT (`Timestamp Regression` & `Sequence Regression`, cross-validated by kinematic divergence).

#### 4. Evil Twin / Rogue GCS Spoofing Attack (Cyber Protocol Layer)
Spoofs a rogue Ground Control Station (`sysid=254`) emitting conflicting heartbeats and unauthorized flight commands:
```bash
cd /home/average_sapien/Documents/ml_cyber_attack_dection
.venv/bin/python experiments/simulation/attacks/attack_evil_twin.py --duration 15
```
* **Expected Result:** Instant Layer 2 RED ALERT (`Rogue GCS Heartbeat (Unauthorized sysid 254 != 255)` or `Unauthorized Flight Command`).

---

## 4. Post-Flight Evaluation & Verification Report

After the flight mission completes and the drone lands, run the evaluation engine to generate performance tables and timeline plots:

```bash
cd /home/average_sapien/Documents/ml_cyber_attack_dection
.venv/bin/python experiments/simulation/evaluate_simulation_run.py
```

*Output Artifacts Produced:*
1. **Terminal Report:** Precision, Recall, F1-Scores, False Alarm Rate (FAR), and First-Alert Latencies.
2. **Timeline Verification Plot:** Saved to `experiments/simulation/simulation_attack_detection_timeline.png`.

---

## 5. Optional / Standalone Tools

* **Headless Telemetry Recorder:** If you want to record flight data without rendering the HUD:
  ```bash
  .venv/bin/python experiments/simulation/record_flight_data.py --duration 300
  ```
* **Retrain / Re-calibrate Checkpoint:** If models or invariant parameters are updated:
  ```bash
  .venv/bin/python experiments/simulation/train_detector_model.py
  ```
* **Optional 3D Visualization (QGroundControl):**
  ```bash
  qgroundcontrol
  ```

---

## 6. Troubleshooting & Common Situations

| Issue | Cause | Solution |
| :--- | :--- | :--- |
| `WAF build failed / empy syntax error` | `empy` version is 4.x instead of 3.3.4 | In `~/ardupilot/ArduCopter/.venv`, run `pip install "empy==3.3.4"`. |
| `fly_mission.py` hangs on `Waiting for UAV heartbeat...` | SITL did not output to port 14553 | In Terminal 1 (MAVProxy), type `output add 127.0.0.1:14553`, or launch SITL with `--out=udp:127.0.0.1:14553`. |
| `ValueError: UDP ports must be specified as host:port` | Typo in `--out` parameter (e.g., `upd:` instead of `udp:`) | Verify exact syntax: `--out=udp:127.0.0.1:<PORT>`. |
| `Address already in use` on port 14552 | An earlier detector or recorder process is still listening | Run `kill -9 $(lsof -t -i:14552)` to clear the port. |
| SITL says `WAF build` or takes time to start | SITL builds binary on first launch | Wait for `STABILIZE>` prompt before running other scripts. |
| Detector shows Replay after SITL reboot | Autopilot reset its boot clock to 0 ms | The updated detector recognizes reboots within 3 packets and resets the baseline automatically. |
| Drone does not respond to `fly_mission.py` | Drone is not armed or not in GUIDED mode | Run `mode guided` and `arm throttle` in Terminal 1 (SITL prompt). |
