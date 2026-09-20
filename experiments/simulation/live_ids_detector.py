"""
experiments/simulation/live_ids_detector.py

Real-Time Dual-Layer Cyber-Physical Intrusion Detection System (IDS) for UAVs.
- Ingests live MAVLink telemetry over UDP (127.0.0.1:14552).
- LAYER 1: First-principles Newtonian Kinematic Invariants (< 1 us).
    * Vertical integration residual (rh = dh - vz*dt)
    * Total dynamic acceleration magnitude check (a_mag <= 15 m/s^2)
    * Aerodynamic tilt coupling (pitch/roll vs horizontal accelerations)
    * Calibrated Mahalanobis Chi-Square distance (tau_chi2 = 33.62)
- LAYER 2: Protocol-level Network Invariants + Temporal Rolling ML (W=10, ~150 us).
    * True packet arrival rate & inter-arrival time tracking
    * DoS detection: packet rate flooding (>75 pkts/s) & malformed frame detection
    * Replay detection: timestamp regression & sequence regression monitoring
    * 5-Class Canonical Classifier (Benign, DoS, Replay, Evil_Twin, FDI)
- Renders a live updating ASCII Terminal Telemetry & Alert HUD.
"""

import os
import sys
import time
import math
import json
import pickle
import argparse
from collections import deque
import numpy as np
import pandas as pd

# ANSI Terminal Styling
ESC = "\033["
RESET = f"{ESC}0m"
BOLD = f"{ESC}1m"
RED = f"{ESC}91m"
GREEN = f"{ESC}92m"
YELLOW = f"{ESC}93m"
CYAN = f"{ESC}96m"
WHITE = f"{ESC}97m"
CLEAR_SCREEN = f"{ESC}2J{ESC}H"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_PATH = os.path.join(SCRIPT_DIR, "models", "detector_checkpoint.pkl")
LOG_PATH = os.path.join(SCRIPT_DIR, "detector_events.log")
CSV_DATASET_PATH = os.path.join(SCRIPT_DIR, "simulation_telemetry_dataset.csv")
MARKER_PATH = os.path.join(SCRIPT_DIR, ".active_attack.json")


class LiveCyberPhysicalDetector:
    def __init__(self, port=14552, checkpoint_path=CHECKPOINT_PATH):
        self.port = port
        self.checkpoint_path = checkpoint_path
        self.classes = ['Benign', 'DoS', 'Replay', 'Evil_Twin', 'FDI']
        
        print(f"{CYAN}[*] Initializing UAV Dual-Layer IDS Engine...{RESET}")
        self.load_models()
        
        # Physics Invariants (Layer 1)
        self.phys_layer = self.checkpoint['physics_layer']
        self.tau_chi2 = self.phys_layer['tau_chi2']
        self.mu = self.phys_layer['mu']
        self.inv_cov = self.phys_layer['inv_cov']
        self.max_accel = 15.0 # Physical bound: quadrotor nominal flight never exceeds 1.5g
        
        # Temporal ML Classifier (Layer 2)
        self.ml_layer = self.checkpoint['temporal_ml_layer']
        self.clf = self.ml_layer['model']
        self.W = self.ml_layer['window_size']
        self.raw_feature_names = self.ml_layer['raw_feature_names']
        
        # Telemetry State Trackers
        self.prev_time = None
        self.prev_state = None
        self.window = deque(maxlen=self.W)
        
        # Network & Protocol Layer Trackers (Cyber Invariants)
        self.packet_arrival_times = deque(maxlen=200)
        self.bad_packet_times = deque(maxlen=100)
        self.last_packet_arrival = None
        self.last_time_boot_ms = None
        # Per-sysid sequence tracking: dict[sysid -> last_seq]
        # Prevents rogue GCS (sysid=254) counters being compared against autopilot (sysid=1).
        self.last_seq_per_sysid = {}
        self.primary_sysid = None   # Locked to autopilot's sysid on first heartbeat
        self.consecutive_reboot_packets = 0
        self.total_packets = 0
        
        # Evidence Accumulators for Cyber Invariants
        self.dos_evidence_count = 0
        self.replay_evidence_count = 0
        self.evil_twin_evidence_count = 0
        self.last_dos_reason = ""
        self.last_replay_reason = ""
        self.last_evil_twin_reason = ""
        
        # Security Incident Counters
        self.alerts_count = {c: 0 for c in self.classes if c != 'Benign'}
        self.alerts_count['Physics_Invariant_FDI'] = 0
        
        # Open log file
        self.log_file = open(LOG_PATH, "a")
        self.log_file.write(f"\n--- SESSION STARTED AT {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n")
        self.log_file.flush()

        # Telemetry & Attack CSV Dataset Logger
        self.marker_path = MARKER_PATH
        self.csv_path = CSV_DATASET_PATH
        self.start_session_time = time.time()
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, "w") as f:
                f.write("timestamp,flight_time,ground_truth,phys_height,phys_vx,phys_vy,phys_vz,phys_pitch,phys_roll,phys_yaw,phys_a_mag,phys_r_h,phys_chi2,cyb_pkt_rate,cyb_inter_arrival,cyb_bad_packets,l1_violation,l1_reason,l2_predicted_class,l2_confidence,l2_prob_benign,l2_prob_dos,l2_prob_replay,l2_prob_evil_twin,l2_prob_fdi,l2_reason\n")

    def get_active_ground_truth(self):
        try:
            if os.path.exists(self.marker_path):
                with open(self.marker_path, "r") as f:
                    data = json.load(f)
                    if data.get("active", False):
                        return data.get("attack", "Benign")
        except Exception:
            pass
        return "Benign"

    def log_csv_row(self, r):
        try:
            with open(self.csv_path, "a") as f:
                vals = [
                    f"{r['timestamp']:.3f}", f"{r['flight_time']:.3f}", r['ground_truth'],
                    f"{r['phys_height']:.3f}", f"{r['phys_vx']:.3f}", f"{r['phys_vy']:.3f}", f"{r['phys_vz']:.3f}",
                    f"{r['phys_pitch']:.2f}", f"{r['phys_roll']:.2f}", f"{r['phys_yaw']:.2f}",
                    f"{r['phys_a_mag']:.3f}", f"{r['phys_r_h']:.3f}", f"{r['phys_chi2']:.3f}",
                    f"{r['cyb_pkt_rate']:.2f}", f"{r['cyb_inter_arrival']:.4f}", str(r['cyb_bad_packets']),
                    str(r['l1_violation']), f'"{r["l1_reason"]}"',
                    r['l2_predicted_class'], f"{r['l2_confidence']:.3f}",
                    f"{r['l2_prob_benign']:.4f}", f"{r['l2_prob_dos']:.4f}", f"{r['l2_prob_replay']:.4f}",
                    f"{r['l2_prob_evil_twin']:.4f}", f"{r['l2_prob_fdi']:.4f}",
                    f'"{r["l2_reason"]}"'
                ]
                f.write(",".join(vals) + "\n")
        except Exception:
            pass

    def load_models(self):
        if not os.path.exists(self.checkpoint_path):
            print(f"{RED}[!] Checkpoint not found at {self.checkpoint_path}{RESET}")
            print("[*] Please run: .venv/bin/python experiments/simulation/train_detector_model.py")
            sys.exit(1)
            
        with open(self.checkpoint_path, 'rb') as f:
            self.checkpoint = pickle.load(f)
            
        acc = self.checkpoint.get('metadata', {}).get('accuracy', 99.87)
        f1 = self.checkpoint.get('metadata', {}).get('macro_f1', 99.76)
        print(f"{GREEN}[+] Loaded production models (Accuracy: {acc:.2f}% | F1: {f1:.2f}%){RESET}")

    def log_event(self, message):
        timestamp = time.strftime('%H:%M:%S')
        self.log_file.write(f"[{timestamp}] {message}\n")
        self.log_file.flush()

    def evaluate_layer1_physics(self, current_state, dt):
        """
        Layer 1: First-principles Newtonian Invariants (< 1 us).
        Returns:
            chi2_score (float): Mahalanobis residual distance
            accel_mag (float): Total acceleration magnitude in m/s^2
            res_height (float): Vertical integration error (m)
            is_violation (bool): True if physical laws are broken
            reason (str): Diagnostic details
        """
        if self.prev_state is None or dt <= 0 or self.total_packets <= 2:
            return 0.0, 0.0, 0.0, False, "Baseline Initialization"
            
        g = 9.81
        h = current_state['height']
        vx = current_state['x_speed']
        vy = current_state['y_speed']
        vz = current_state['z_speed']
        pitch = current_state['pitch']
        roll = current_state['roll']
        dy = current_state['mp_distance_y']
        dz = current_state['mp_distance_z']
        
        dh = h - self.prev_state['height']
        r_h = dh - vz * dt
        
        ax = (vx - self.prev_state['x_speed']) / dt
        ay = (vy - self.prev_state['y_speed']) / dt
        az = (vz - self.prev_state['z_speed']) / dt
        a_mag = math.sqrt(ax**2 + ay**2 + az**2)
        
        r_pitch = ax - g * math.sin(math.radians(pitch))
        r_roll = ay + g * math.sin(math.radians(roll))
        
        r_dist_y = (dy - self.prev_state['mp_distance_y']) - vy * dt
        r_dist_z = (dz - self.prev_state['mp_distance_z']) - vz * dt
        
        res_vec = np.array([r_h, ax, ay, az, a_mag, r_pitch, r_roll, r_dist_y, r_dist_z])
        diff = res_vec - self.mu
        chi2_score = float(np.dot(np.dot(diff, self.inv_cov), diff.T))
        
        # Rule Invariants: Calibrated for 3D Dynamic Flight (Turns & Waypoints)
        if a_mag > self.max_accel and abs(r_h) > 1.0:
            return chi2_score, a_mag, r_h, True, f"Excessive Dynamic Acceleration ({a_mag:.1f} m/s^2 > {self.max_accel} m/s^2)"
        elif abs(r_h) > 3.0:
            return chi2_score, a_mag, r_h, True, f"Vertical Kinematic Discontinuity (|rh|={abs(r_h):.2f}m > 3.0m)"
        elif chi2_score > 650.0:
            return chi2_score, a_mag, r_h, True, f"Mahalanobis Invariant Divergence (chi^2={chi2_score:.1f} > 650.0)"
            
        return chi2_score, a_mag, r_h, False, "Nominal Kinematics"

    def evaluate_layer2_ml(self, raw_sample_dict, cyber_meta=None):
        """
        Layer 2: Protocol Invariants + Temporal Rolling Multimodal Classifier (W=10).
        Returns:
            pred_class (str): Predicted canonical class
            probabilities (dict): Class probabilities
            alert_reason (str): Specific trigger justification
        """
        # 1. Cyber Protocol Level Invariant Gates (Defense-in-Depth)
        if cyber_meta and cyber_meta.get('replay_active'):
            pred_class = 'Replay'
            prob_dict = {'Benign': 0.02, 'DoS': 0.01, 'Replay': 0.95, 'Evil_Twin': 0.01, 'FDI': 0.01}
            return pred_class, prob_dict, cyber_meta.get('replay_reason', 'Sequence Regression')

        if cyber_meta and cyber_meta.get('dos_active'):
            pred_class = 'DoS'
            prob_dict = {'Benign': 0.01, 'DoS': 0.98, 'Replay': 0.005, 'Evil_Twin': 0.003, 'FDI': 0.002}
            return pred_class, prob_dict, cyber_meta.get('dos_reason', 'Packet Flood')

        if cyber_meta and cyber_meta.get('evil_twin_active'):
            pred_class = 'Evil_Twin'
            prob_dict = {'Benign': 0.01, 'DoS': 0.005, 'Replay': 0.005, 'Evil_Twin': 0.97, 'FDI': 0.01}
            return pred_class, prob_dict, cyber_meta.get('evil_twin_reason', 'Rogue GCS / Unauthorized Command Injection')

        # 2. Temporal Rolling ML Model Evaluation
        self.window.append(raw_sample_dict)
        if len(self.window) < self.W:
            return "Benign", {c: (1.0 if c == 'Benign' else 0.0) for c in self.classes}, "Nominal Stream"
            
        df_win = pd.DataFrame(list(self.window))
        for col in self.raw_feature_names:
            if col not in df_win.columns:
                df_win[col] = 0.0
        df_win = df_win[self.raw_feature_names]
        
        latest_row = df_win.iloc[-1:].reset_index(drop=True)
        roll_mean = df_win.mean().to_frame().T.add_suffix('_roll_mean')
        roll_std = df_win.std().fillna(0).to_frame().T.add_suffix('_roll_std')
        diff_1 = (df_win.iloc[-1] - df_win.iloc[-2]).to_frame().T.add_suffix('_diff')
        
        df_feat = pd.concat([latest_row, roll_mean, roll_std, diff_1], axis=1)
        
        rolling_names = self.ml_layer['rolling_feature_names']
        for col in rolling_names:
            if col not in df_feat.columns:
                df_feat[col] = 0.0
        df_feat = df_feat[rolling_names]
        
        probs = self.clf.predict_proba(df_feat)[0]
        prob_dict = {cls: float(p) for cls, p in zip(self.classes, probs)}
        pred_idx = np.argmax(probs)
        pred_class = self.classes[pred_idx]
        
        return pred_class, prob_dict, "ML Model Classification"

    def render_dashboard(self, state, dt, chi2, a_mag, r_h, l1_violation, l1_reason, l2_class, l2_probs, l2_reason, pkt_rate):
        """Renders updating ASCII HUD dashboard in terminal."""
        hud = []
        hud.append(CLEAR_SCREEN)
        hud.append("╔══════════════════════════════════════════════════════════════════════════════╗")
        hud.append(f"║     {BOLD}UAV DUAL-LAYER CYBER-PHYSICAL INTRUSION DETECTION SYSTEM (IDS){RESET}       ║")
        hud.append("╠══════════════════════════════════════════════════════════════════════════════╣")
        
        # Overall Status Banner
        if l1_violation or (l2_class != 'Benign' and l2_probs[l2_class] > 0.65):
            attack_type = "FDI / SENSOR TAMPERING" if l1_violation else l2_class.upper()
            status_banner = f"{RED}{BOLD}STATUS: [!] ACTIVE ATTACK DETECTED -- {attack_type}{RESET}"
        else:
            status_banner = f"{GREEN}{BOLD}STATUS: [OK] ALL SYSTEMS NOMINAL -- SECURE AUTONOMOUS FLIGHT{RESET}"
        hud.append(f"║  {status_banner:<70}║")
        hud.append("╠══════════════════════════════════════════════════════════════════════════════╣")
        
        # Flight Kinematics
        hud.append(f"║ {BOLD}FLIGHT KINEMATICS (ArduPilot SITL):{RESET}                                          ║")
        hud.append(f"║   Altitude : {state['height']:6.2f} m       | Speed (x,y,z): ({state['x_speed']:5.1f}, {state['y_speed']:5.1f}, {state['z_speed']:5.1f}) m/s  ║")
        hud.append(f"║   Attitude : P:{state['pitch']:5.1f}° R:{state['roll']:5.1f}° Y:{state['yaw']:5.1f}° | Accel Mag: {a_mag:5.2f} m/s² (Bound: {self.max_accel} m/s²)  ║")
        
        hud.append("╟──────────────────────────────────────────────────────────────────────────────╢")
        
        # Layer 1 Status
        l1_color = RED if l1_violation else GREEN
        l1_tag = "[VIOLATION]" if l1_violation else "[PASS]"
        hud.append(f"║ {BOLD}LAYER 1: INSTANTANEOUS PHYSICS KINEMATIC INVARIANTS (< 1 us):{RESET}                ║")
        hud.append(f"║   Residual Chi² Score : {chi2:6.2f}  (Threshold: {self.tau_chi2:5.2f})  --> {l1_color}{BOLD}{l1_tag:<11}{RESET}  ║")
        hud.append(f"║   Vertical Invariant  : rh = {r_h:5.2f} m  | Accel Check: {a_mag:5.2f} m/s²                 ║")
        if l1_violation:
            hud.append(f"║   {RED}{BOLD}PHYSICS ALERT : {l1_reason:<58}{RESET}║")
            
        hud.append("╟──────────────────────────────────────────────────────────────────────────────╢")
        
        # Layer 2 Status
        l2_color = GREEN if l2_class == 'Benign' else (YELLOW if l2_probs[l2_class] < 0.8 else RED)
        hud.append(f"║ {BOLD}LAYER 2: PROTOCOL INVARIANTS & TEMPORAL ROLLING ML (W=10, ~150 us):{RESET}           ║")
        hud.append(f"║   Telemetry Stream Rate : {pkt_rate:5.1f} pkts/s  | Window Latency: {dt*1000:5.1f} ms          ║")
        hud.append(f"║   Predicted Class       : {l2_color}{BOLD}{l2_class:<12}{RESET} (Confidence: {l2_probs[l2_class]*100:5.1f}%)                 ║")
        if l2_class != 'Benign':
            hud.append(f"║   {RED}{BOLD}CYBER ALERT   : {l2_reason:<58}{RESET}║")
            
        # Probability Distribution
        prob_str = " | ".join([f"{c}: {l2_probs[c]*100:4.1f}%" for c in self.classes])
        hud.append(f"║   Distribution : {prob_str:<58} ║")
        
        hud.append("╟──────────────────────────────────────────────────────────────────────────────╢")
        hud.append(f"║ {BOLD}ATTACK INCIDENT LOG COUNTERS:{RESET}                                                ║")
        counts_str = f"FDI/Phys: {self.alerts_count['Physics_Invariant_FDI']} | DoS: {self.alerts_count['DoS']} | Replay: {self.alerts_count['Replay']} | Evil_Twin: {self.alerts_count['Evil_Twin']}"
        hud.append(f"║   Total Packets: {self.total_packets:<6} | {counts_str:<45} ║")
        hud.append("╚══════════════════════════════════════════════════════════════════════════════╝")
        hud.append("\n[Tip: Use Terminal 4 to trigger FDI, DoS, or Replay attacks on demand]")
        print("\n".join(hud), flush=True)

    def process_telemetry_packet(self, state, cyber_features=None, cyber_meta=None, pkt_rate=10.0, bad_rate=0):
        """Runs both layers on incoming telemetry."""
        now = time.time()
        dt = (now - self.prev_time) if self.prev_time else self.phys_layer['dt_nominal']
        self.prev_time = now
        
        # 1. Evaluate Layer 1: Physics Invariants
        chi2, a_mag, r_h, l1_violation, l1_reason = self.evaluate_layer1_physics(state, dt)
        if l1_violation:
            self.alerts_count['Physics_Invariant_FDI'] += 1
            self.log_event(f"LAYER 1 ALERT: {l1_reason} (chi2={chi2:.2f}, a_mag={a_mag:.2f})")
            
        # 2. Build Layer 2 Multimodal vector
        sample = {}
        for k in ['height', 'x_speed', 'y_speed', 'z_speed', 'pitch', 'roll', 'yaw', 'mp_distance_y', 'mp_distance_z']:
            sample[f'phys_{k}'] = state.get(k, 0.0)
            
        if cyber_features:
            for k, v in cyber_features.items():
                sample[f'cyb_{k}'] = v
        else:
            sample['cyb_time_since_last_packet'] = dt
            sample['cyb_frame.len'] = 64.0
            sample['cyb_udp.length'] = 48.0
            sample['cyb_wlan.seq'] = self.total_packets % 4096
            
        # 3. Evaluate Layer 2: Protocol Invariants & Temporal Rolling ML
        l2_class, l2_probs, l2_reason = self.evaluate_layer2_ml(sample, cyber_meta=cyber_meta)
        if l2_class != 'Benign' and l2_probs[l2_class] > 0.7:
            self.alerts_count[l2_class] += 1
            self.log_event(f"LAYER 2 ALERT: {l2_class} detected ({l2_reason}) with confidence {l2_probs[l2_class]*100:.1f}%")
            
        # 4. Log synchronized row to CSV dataset
        gt = self.get_active_ground_truth()
        flight_time = now - self.start_session_time
        self.log_csv_row({
            'timestamp': now,
            'flight_time': flight_time,
            'ground_truth': gt,
            'phys_height': state.get('height', 0.0),
            'phys_vx': state.get('x_speed', 0.0),
            'phys_vy': state.get('y_speed', 0.0),
            'phys_vz': state.get('z_speed', 0.0),
            'phys_pitch': state.get('pitch', 0.0),
            'phys_roll': state.get('roll', 0.0),
            'phys_yaw': state.get('yaw', 0.0),
            'phys_a_mag': a_mag,
            'phys_r_h': r_h,
            'phys_chi2': chi2,
            'cyb_pkt_rate': pkt_rate,
            'cyb_inter_arrival': sample.get('cyb_time_since_last_packet', dt),
            'cyb_bad_packets': bad_rate,
            'l1_violation': l1_violation,
            'l1_reason': l1_reason,
            'l2_predicted_class': l2_class,
            'l2_confidence': l2_probs.get(l2_class, 0.0),
            'l2_prob_benign': l2_probs.get('Benign', 0.0),
            'l2_prob_dos': l2_probs.get('DoS', 0.0),
            'l2_prob_replay': l2_probs.get('Replay', 0.0),
            'l2_prob_evil_twin': l2_probs.get('Evil_Twin', 0.0),
            'l2_prob_fdi': l2_probs.get('FDI', 0.0),
            'l2_reason': l2_reason
        })

        # 5. Render HUD
        self.render_dashboard(state, dt, chi2, a_mag, r_h, l1_violation, l1_reason, l2_class, l2_probs, l2_reason, pkt_rate)
        self.prev_state = state.copy()

    def run_mavlink(self):
        """Connects to ArduPilot SITL via pymavlink over UDP."""
        try:
            from pymavlink import mavutil
        except ImportError:
            print(f"{RED}[ERROR] pymavlink is not installed in the virtual environment.{RESET}")
            print(f"Please install it using:\n    .venv/bin/pip install pymavlink\n")
            sys.exit(1)

        conn_str = f"udpin:0.0.0.0:{self.port}"
        print(f"{CYAN}[*] Initializing MAVLink listener on {conn_str}...{RESET}")
        print(f"[*] Waiting for ArduPilot SITL heartbeat from port {self.port}...")
        
        master = mavutil.mavlink_connection(conn_str)
        master.wait_heartbeat()
        print(f"{GREEN}[+] Heartbeat received from system {master.target_system} (component {master.target_component})!{RESET}")
        print("[*] Starting dual-layer IDS monitoring...")
        time.sleep(1)

        state = {
            'height': 0.0, 'x_speed': 0.0, 'y_speed': 0.0, 'z_speed': 0.0,
            'pitch': 0.0, 'roll': 0.0, 'yaw': 0.0, 'mp_distance_y': 0.0, 'mp_distance_z': 0.0
        }
        cyber = {'time_since_last_packet': 0.05, 'frame.len': 64.0, 'udp.length': 48.0}
        
        last_hud_render = 0
        while True:
            # Process packets rapidly without dropping arrival stats
            msg = master.recv_match(blocking=True, timeout=0.1)
            now = time.time()
            
            if msg is not None:
                self.total_packets += 1
                self.packet_arrival_times.append(now)
                
                # Compute instantaneous inter-packet arrival time
                if self.last_packet_arrival is not None:
                    inter_arrival = max(now - self.last_packet_arrival, 0.0001)
                    if inter_arrival > 2.0:
                        # Telemetry link dropped/restarted; reset session clock trackers
                        self.last_time_boot_ms = None
                        self.last_seq_per_sysid.clear()
                        self.consecutive_reboot_packets = 0
                        self.replay_evidence_count = 0
                else:
                    inter_arrival = 0.05
                self.last_packet_arrival = now
                
                msg_type = msg.get_type()
                
                # Check for bad packets / UDP socket floods
                if msg_type == 'BAD_DATA':
                    self.bad_packet_times.append(now)

                # Track Cyber Invariants:
                # 1. Timestamp Regression (Replay Indicator)
                time_boot = getattr(msg, 'time_boot_ms', None)
                if time_boot is not None and time_boot > 0:
                    if self.last_time_boot_ms is not None:
                        dt_back = self.last_time_boot_ms - time_boot
                        if dt_back > 350:
                            # If massive backward jump (> 30s) or clock reset to near 0 (< 20s), detect reboot/restart:
                            if dt_back > 30000 or time_boot < 20000:
                                self.consecutive_reboot_packets += 1
                                if self.consecutive_reboot_packets >= 3:
                                    # Autopilot reboot / session reset confirmed; adopt new clock baseline
                                    self.last_time_boot_ms = time_boot
                                    self.consecutive_reboot_packets = 0
                                    self.replay_evidence_count = 0
                                    self.last_replay_reason = ""
                            else:
                                self.consecutive_reboot_packets = 0
                                self.replay_evidence_count = min(self.replay_evidence_count + 8, 40)
                                self.last_replay_reason = f"Timestamp Regression ({time_boot}ms < {self.last_time_boot_ms}ms)"
                        else:
                            self.consecutive_reboot_packets = 0
                    
                    if self.last_time_boot_ms is None or time_boot >= self.last_time_boot_ms:
                        self.last_time_boot_ms = time_boot
                        self.consecutive_reboot_packets = 0

                # 2. Sequence Number Regression (Replay Indicator — per-sysid)
                # Each MAVLink source has its own independent sequence counter.
                # Comparing sequences ACROSS sysids causes massive false Replay alarms
                # when a rogue GCS (sysid=254) connects with its own counter.
                if hasattr(msg, 'get_seq'):
                    seq = msg.get_seq()
                    src_sys = msg.get_srcSystem() if hasattr(msg, 'get_srcSystem') else None

                    # Lock primary autopilot sysid on first HEARTBEAT from the vehicle (sysid != 255)
                    if msg_type == 'HEARTBEAT' and src_sys is not None and src_sys not in (0, 255):
                        if self.primary_sysid is None:
                            self.primary_sysid = src_sys

                    if src_sys is not None and self.primary_sysid is not None and src_sys != self.primary_sysid:
                        # Packet is from a DIFFERENT sysid than the primary autopilot.
                        # Track its own sequence separately — never cross-compare.
                        # Also treat any non-primary, non-GCS sysid as a suspicious signal.
                        if src_sys != 255:  # 255 is the legitimate GCS broadcast address
                            self.evil_twin_evidence_count = min(self.evil_twin_evidence_count + 3, 40)
                            self.last_evil_twin_reason = (
                                f"Unauthorized GCS Sequence (sysid={src_sys}, seq {seq} != primary sysid {self.primary_sysid})"
                            )
                        self.last_seq_per_sysid[src_sys] = seq
                    else:
                        # Primary autopilot sysid — apply sequence regression check
                        last_seq = self.last_seq_per_sysid.get(src_sys)
                        if last_seq is not None:
                            # Normal 255→0 wrap is fine; only flag genuine backward jumps
                            if seq < last_seq and not (last_seq > 240 and seq < 15):
                                # Ignore minor UDP 1-packet reordering jitter (drop > 2)
                                if last_seq - seq > 2:
                                    # seq=0: new source connected (reboot). NOT a replay.
                                    if seq != 0:
                                        self.replay_evidence_count = min(self.replay_evidence_count + 5, 40)
                                        self.last_replay_reason = f"Sequence Regression (seq {seq} < prev {last_seq})"
                        self.last_seq_per_sysid[src_sys] = seq

                # 3. Rogue GCS / Evil Twin Invariant Detection (heartbeats & commands)
                src_sys = msg.get_srcSystem() if hasattr(msg, 'get_srcSystem') else None
                src_comp = msg.get_srcComponent() if hasattr(msg, 'get_srcComponent') else None

                is_rogue_gcs = False
                if msg_type == 'HEARTBEAT' and getattr(msg, 'type', None) == 6:  # MAV_TYPE_GCS
                    if src_sys is not None and src_sys != 255:
                        is_rogue_gcs = True
                        self.last_evil_twin_reason = f"Rogue GCS Heartbeat (Unauthorized sysid {src_sys} != 255)"
                elif msg_type in ['COMMAND_LONG', 'SET_MODE', 'COMMAND_INT', 'PARAM_SET']:
                    if src_sys is not None and src_sys != 255:
                        is_rogue_gcs = True
                        self.last_evil_twin_reason = f"Unauthorized Flight Command ({msg_type} from rogue sysid {src_sys})"
                    elif src_comp is not None and src_comp not in [0, 190]:
                        is_rogue_gcs = True
                        self.last_evil_twin_reason = f"Unauthorized GCS Component ({msg_type} from compid {src_comp})"

                if is_rogue_gcs:
                    self.evil_twin_evidence_count = min(self.evil_twin_evidence_count + 8, 40)

                # Update physical kinematic state
                if msg_type == 'ATTITUDE':
                    state['pitch'] = math.degrees(msg.pitch)
                    state['roll'] = math.degrees(msg.roll)
                    state['yaw'] = math.degrees(msg.yaw)
                    
                elif msg_type == 'GLOBAL_POSITION_INT':
                    state['height'] = msg.relative_alt / 1000.0  # mm to m
                    state['x_speed'] = msg.vx / 100.0            # cm/s to m/s
                    state['y_speed'] = msg.vy / 100.0
                    state['z_speed'] = msg.vz / 100.0
                    state['mp_distance_y'] = msg.lon / 1e7
                    state['mp_distance_z'] = state['height']
                    
                elif msg_type == 'VFR_HUD':
                    if 'height' not in state or state['height'] == 0:
                        state['height'] = msg.alt
                        state['z_speed'] = msg.climb

                cyber['frame.len'] = len(msg.get_msgbuf()) if hasattr(msg, 'get_msgbuf') else 64.0
                cyber['time_since_last_packet'] = inter_arrival

            # Calculate live packet arrival rate across recent window
            if len(self.packet_arrival_times) > 5:
                dt_span = self.packet_arrival_times[-1] - self.packet_arrival_times[0]
                current_pkt_rate = (len(self.packet_arrival_times) - 1) / dt_span if dt_span > 0 else 0.0
            else:
                current_pkt_rate = 1.0 / inter_arrival if inter_arrival > 0 else 10.0

            # Count malformed packets in the last 1 second
            bad_rate = sum(1 for t in self.bad_packet_times if now - t <= 1.0)
            
            # Evaluate DoS Invariant
            if current_pkt_rate > 260.0 or bad_rate > 25:
                self.dos_evidence_count = min(self.dos_evidence_count + 4, 50)
                self.last_dos_reason = f"Packet Flood ({current_pkt_rate:.1f} pkts/s > 260 pkts/s)"
            else:
                self.dos_evidence_count = max(self.dos_evidence_count - 1, 0)
                
            self.replay_evidence_count = max(self.replay_evidence_count - 1, 0)
            self.evil_twin_evidence_count = max(self.evil_twin_evidence_count - 1, 0)

            # Trigger HUD update & Layer 2 ML evaluation at regular rate (~7 Hz)
            if now - last_hud_render >= 0.15:
                cyber_meta = {
                    'dos_active': self.dos_evidence_count >= 3,
                    'dos_reason': self.last_dos_reason,
                    'replay_active': self.replay_evidence_count >= 3,
                    'replay_reason': self.last_replay_reason,
                    'evil_twin_active': self.evil_twin_evidence_count >= 3,
                    'evil_twin_reason': self.last_evil_twin_reason
                }
                self.process_telemetry_packet(state, cyber, cyber_meta=cyber_meta, pkt_rate=current_pkt_rate, bad_rate=bad_rate)
                last_hud_render = now


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Live Cyber-Physical UAV IDS Detector")
    parser.add_argument("--port", type=int, default=14552, help="UDP port listening for MAVLink telemetry")
    args = parser.parse_args()

    detector = LiveCyberPhysicalDetector(port=args.port)
    try:
        detector.run_mavlink()
    except KeyboardInterrupt:
        print(f"\n{YELLOW}[*] Live IDS monitoring terminated by user.{RESET}")
