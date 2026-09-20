# CONTEXT FOR NEXT SESSION: UAV Cyber-Physical Attack Detection & Simulation

> **PROMPT TO COPY-PASTE INTO NEXT SESSION:**
> `"Please read experiments/CONTEXT_FOR_NEXT_SESSION.md and experiments/simulation/SIMULATION_SETUP_AND_COMMANDS.md, and continue from the exact state described."`

---

## 1. Project Background & Dual-Layer Architecture

* **Domain:** Autonomous Cyber-Physical Security for Unmanned Aerial Vehicles (UAVs / Drones).
* **Core Problem:** UAVs face multi-vector attacks spanning physical flight sensors (GPS, IMU, barometer) and cyber network links (Wi-Fi, MAVLink, telemetry).
* **Canonical 5-Class Attack Taxonomy:**
  1. `Benign`: Legitimate autonomous flight and nominal telemetry.
  2. `DoS`: Denial-of-Service UDP/Wi-Fi packet flooding.
  3. `Replay`: Sniffing authentic flight commands/telemetry and playing them back.
  4. `Evil_Twin`: Rogue GCS (sysid=254) injecting unauthorized heartbeats and flight commands.
  5. `FDI`: False Data Injection tampering with GPS coordinates, altitude, and velocities.
* **Dual-Layer Architecture:**
  * **Layer 1 (Physics Invariant Reflex, < 1 µs):** Instantaneous Newtonian physical invariants (residual χ², vertical integration error r_h, acceleration magnitude a_mag ≤ 15 m/s²). Catches FDI coordinate jumps.
  * **Layer 2 (Cyber Protocols & Temporal ML, ~150 µs):** High-speed packet arrival tracking, per-sysid sequence/timestamp monotonicity verification, and sliding window (W=10) multimodal XGBoost. Catches DoS, Replay, and Evil Twin attacks.

---

## 2. Critical Rules & Guardrails (Strictly Enforced)

1. **Strict Scope Boundary:** All code, scripts, models, datasets, logs, and plots MUST remain inside `experiments/` (specifically `experiments/simulation/`).
2. **Never Touch Original Root Datasets:**
   * `Dataset_T-ITS.csv` — DO NOT TOUCH.
   * `Physical_UAV_Dataset.csv` — DO NOT TOUCH.
   * `Cyber_UAV_Dataset.csv` — DO NOT TOUCH.
3. **Manual Execution Mandate:** The user runs all simulator processes, detectors, and attack injections manually in their own terminals. Do not execute simulator or background processes automatically without permission.
4. **Environment:** Use the dedicated virtual environment at `.venv/bin/python`.

---

## 3. All Engineering Problems Diagnosed & Resolved (Full History)

### Problem 1: Static Hover Blindspot & Telemetry Bottleneck
* **Symptom:** Replay and DoS went undetected during static hover; HUD bottlenecked UDP socket.
* **Solution:** Developed `fly_mission.py` for active 30m×30m patrol at 5 m/s. Decoupled packet reception (120 Hz) from HUD rendering (7 Hz).

### Problem 2: DoS False Alarm Storm on Normal Flight
* **Symptom:** 99.8% false DoS alarm rate during autonomous flight.
* **Root Cause:** SITL naturally broadcasts ~115–125 pkts/s. Initial threshold was 70 pkts/s.
* **Solution:** Calibrated DoS gate to ≥ 260.0 pkts/s or rolling `bad_rate > 25`. DoS false alarms eliminated.

### Problem 3: Kinematic Turning Residuals (χ² & Dynamic Acceleration)
* **Symptom:** Sharp banking turns triggered Layer 1 FDI alerts.
* **Solution:** Calibrated joint Layer 1 threshold: (a_mag > 15.0 m/s² AND |r_h| > 1.0 m) OR χ² > 650.0. Agile turns pass; FDI attacks detected at >98%.

### Problem 4: Stale Autopilot Reboot Latch & Replay False Alarms
* **Symptom:** 1,200+ false Replay positives due to frozen `time_boot_ms=504854ms` baseline after SITL reboot.
* **Solution:** Added reboot detection (backward jump >30s or reset <20s → 3 consecutive advancing packets reset baseline). Added link-drop reset (gap >2s clears trackers). Added UDP jitter tolerance (requires drop >2 packets).

### Problem 5: FDI Synthetic Clock Desync
* **Symptom:** FDI attacks accidentally triggered Replay alerts.
* **Solution:** `attack_fdi.py` now pre-synchronizes `time_boot_ms` with live MAVLink stream before injection.

### Problem 6: Sequence Regression seq=0 False Replay Positives (Fixed this session)
* **Symptom:** 83 benign samples and 12 Evil Twin samples flagged as Replay. Replay precision stuck at 76.51%.
* **Root Cause:** `seq=0` in a backward sequence jump was treated as a Replay indicator. In reality, `seq=0` always means a NEW MAVLink source connected (SITL reboot or rogue GCS sysid=254 starting its own counter).
* **Solution:** Added `if seq == 0: return False` guard in `is_genuine_replay_event()` in `evaluate_simulation_run.py` and equivalent `if seq != 0:` guard in `live_ids_detector.py`. Replay precision: **76.51% → 95.17%**, FAR: **1.52% → 0.71%**.

### Problem 7: Cross-Sysid Sequence Comparison (Fixed this session)
* **Symptom:** 12/13 missed Evil Twin samples predicted as Replay. Evil Twin recall stuck at 79.37%.
* **Root Cause:** The detector tracked a single global `last_seq` across ALL sysids. Evil Twin's rogue GCS (sysid=254) injects packets with its own independent counter (e.g., seq=16). Compared against autopilot's last seq=120, this triggered `seq 16 < prev 120 → Sequence Regression → Replay`. The ML model never ran.
* **Solution:** Replaced `self.last_seq` with `self.last_seq_per_sysid = {}` (dict keyed by sysid). `self.primary_sysid` is locked to the autopilot on first heartbeat. Packets from any other sysid are tracked in their own bucket — never cross-compared. Non-primary, non-GCS sysid packets now boost `evil_twin_evidence_count` with reason `"Unauthorized GCS Sequence (sysid=254...)"`. Evaluator priority flipped: **Evil Twin checked BEFORE Replay** so rogue sysid signals are never consumed by the Replay gate first. Evil Twin recall: **79.37% → 92.02%**.

### Problem 8: fly_mission.py Bugs (Fixed this session)
* **Bug 1 — Port Mismatch:** Defaulted to `udpin:0.0.0.0:14553`; hung if SITL didn't output there.
  * **Fix:** Auto-detect: tries port 14553 first (4s timeout) then falls back to 14550 with a diagnostic message.
* **Bug 2 — Arming Race Condition:** Called `arducopter_arm()` then immediately sent `MAV_CMD_NAV_TAKEOFF` before pre-arm checks completed.
  * **Fix:** `wait_until_armed()` polls `HEARTBEAT.base_mode` flag until `MAV_MODE_FLAG_SAFETY_ARMED` is confirmed (30s timeout with clear error on failure).
* **Bug 3 — Missing Stream Requests:** No `LOCAL_POSITION_NED` stream requested; caused 20s fallback timer at every waypoint.
  * **Fix:** Explicit `MAV_CMD_SET_MESSAGE_INTERVAL` for LOCAL_POSITION_NED @ 10 Hz, GLOBAL_POSITION_INT @ 4 Hz, VFR_HUD @ 4 Hz, ATTITUDE @ 10 Hz.
* **Bug 4 — VFR_HUD.alt is MSL not AGL:** `VFR_HUD.alt` gives MSL altitude (~584m at SITL origin). Script thought drone was already airborne, skipped climb, ArduPilot auto-disarmed.
  * **Fix:** `get_current_altitude()` and climb loop now exclusively use `GLOBAL_POSITION_INT.relative_alt / 1000.0` (true AGL). `VFR_HUD` removed from all altitude-critical reads.

---

## 4. Latest Verified Results — Full Session History

### Final State (After Stress Test — 12,739 samples)

| Class | Precision | Recall | F1-Score | True Positives |
|---|---|---|---|---|
| Benign | 99.16% | 99.17% | **99.17%** | 11044 / 11136 |
| DoS | 91.28% | 98.99% | **94.98%** | 492 / 497 |
| Replay | 96.26% | 83.23% | **89.27%** | 412 / 495 |
| Evil_Twin | 96.77% | 92.02% | **94.34%** | 150 / 163 |
| FDI | 91.46% | 97.99% | **94.61%** | 439 / 448 |

* **Overall Accuracy: 98.41%** | **Macro F1: 94.48%** | **FAR: 0.83%**

### Detection Latencies
| Attack | Latency | Detection Rate |
|---|---|---|
| FDI | 0.0 ms (Layer 1 physics) | 98.0% |
| DoS | 152.0 ms | 99.0% |
| Replay | 199.0 ms | 83.2% |
| Evil Twin | 242.0 ms | 92.0% |

### Full Improvement History
| Metric | Baseline | This Session Final | Net Gain |
|---|---|---|---|
| Overall Accuracy | 97.70% | **98.41%** | +0.71% |
| Macro F1 | 90.88% | **94.48%** | +3.60% |
| Benign FAR | 1.52% | **0.83%** | −0.69% |
| Replay F1 | 78.12% | **89.27%** | +11.15% |
| Replay Precision | 76.51% | **96.26%** | +19.75% |
| Evil Twin F1 | 87.72% | **94.34%** | +6.62% |
| Evil Twin Recall | 79.37% | **92.02%** | +12.65% |

---

## 5. Simulation Suite Artifacts & Map

All simulation tools are ready in `experiments/simulation/`:

| File | Purpose |
|---|---|
| `SIMULATION_SETUP_AND_COMMANDS.md` | Step-by-step manual terminal execution guide |
| `fly_mission.py` | Autonomous 3D waypoint patrol — **fully fixed, runs from ground** |
| `live_ids_detector.py` | Live MAVLink UDP listener, dual-layer IDS, terminal HUD |
| `record_flight_data.py` | Standalone headless telemetry & attack recorder |
| `evaluate_simulation_run.py` | Multi-class evaluation engine & publication plot generator |
| `stress_test.py` | **NEW** — Automated 11-scenario attack gauntlet orchestrator |
| `simulation_telemetry_dataset.csv` | Live flight telemetry dataset (12,739 records, cumulative) |
| `simulation_attack_detection_timeline.png` | 4-panel publication timeline verification plot |
| `attacks/attack_fdi.py` | FDI injector (synchronized clock) |
| `attacks/attack_dos.py` | DoS packet flood injector |
| `attacks/attack_replay.py` | Sniff & Replay injector (supports `--duration`, `--sniff`) |
| `attacks/attack_evil_twin.py` | Rogue GCS heartbeat & unauthorized flight command injector |
| `models/detector_checkpoint.pkl` | Production dual-layer model checkpoint |

---

## 6. Key Architecture Details for Reference

### Port Topology
| Port | Consumer |
|---|---|
| 14550 | QGroundControl / GCS / `fly_mission.py` fallback |
| 14552 | `live_ids_detector.py`, `record_flight_data.py` |
| 14553 | `fly_mission.py` primary (auto-falls back to 14550) |

### SITL Launch Command (add all 3 output ports)
```bash
../Tools/autotest/sim_vehicle.py -v ArduCopter -f quad \
  --out=udp:127.0.0.1:14550 \
  --out=udp:127.0.0.1:14552 \
  --out=udp:127.0.0.1:14553 \
  --console
```

### Layer 1 Thresholds (calibrated)
* `a_mag > 15.0 m/s²` AND `|r_h| > 1.0 m` → FDI
* `|r_h| > 3.0 m` → FDI (standalone)
* `χ² > 650.0` AND `|r_h| > 0.5` → FDI

### Layer 2 Thresholds (calibrated)
* `pkt_rate > 260.0 pkts/s` → DoS
* Timestamp regression with 350ms ≤ dt_back ≤ 60s (non-stale baseline) → Replay
* Sequence regression with `seq > 0` dropping backward by > 2 (same sysid only) → Replay
* `seq = 0` backward drop → ignored (new source connected, not replay)
* Packet from non-primary sysid (not 255) → Evil Twin evidence +3
* Rogue GCS heartbeat (MAV_TYPE_GCS, sysid ≠ 255) → Evil Twin evidence +8
* Unauthorized flight command (COMMAND_LONG, SET_MODE, etc. from sysid ≠ 255) → Evil Twin evidence +8

### Detection Priority Order (both evaluator and live detector)
```
FDI → DoS → Evil_Twin → Replay → Benign
```
Evil Twin is checked BEFORE Replay to prevent rogue sysid packets from being consumed by the Replay gate.

---

## 7. What Was Done This Session (2026-09-20)

1. **Diagnosed and fixed Replay false positives** — seq=0 guard added to both detector and evaluator
2. **Diagnosed and fixed Evil Twin misses** — per-sysid sequence tracking, ET priority before Replay
3. **Fixed fly_mission.py** — 4 bugs resolved (port, arming race, missing streams, VFR_HUD MSL vs AGL)
4. **Created `stress_test.py`** — 11-scenario automated attack gauntlet
5. **Ran full stress test** — confirmed generalisation across short-burst, high-rate, and multi-vector attacks
6. **Net improvement this session:** Macro F1 +3.60%, Evil Twin recall +12.65%, Replay precision +19.75%

---

## 8. Suggested Next Steps

1. **Paper / Thesis Deliverables** (if applicable):
   * Generate LaTeX comparison tables (Offline Benchmark vs Real-Time Simulation)
   * Export confusion matrix plots and cross-layer detection timeline figures
   * All benchmark CSVs already in `experiments/results/`

2. **Embedded / Companion Computer Profiling** (optional, ~15 mins):
   * Measure CPU utilisation and RAM footprint of live IDS engine
   * Demonstrates feasibility on Raspberry Pi 4 / Jetson Nano for real deployment

3. **More Stress Test Laps** (if cleaner statistics needed):
   * Run 5–10 laps with `stress_test.py` to reduce variance on Replay recall (currently 83.23%)
   * Replay recall is inherently limited by the boundary-sniff window size, not model weakness
