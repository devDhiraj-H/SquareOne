"""
experiments/simulation/record_flight_data.py

Standalone Headless Telemetry & Attack Data Gatherer:
- Connects to MAVLink UDP telemetry stream (127.0.0.1:14552 or custom).
- Logs UAV physical kinematics, cyber network metrics, and active attack ground truth.
- Saves high-frequency synchronized time-series to:
  experiments/simulation/simulation_telemetry_dataset.csv
"""

import os
import sys
import time
import json
import math
import argparse
from collections import deque

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(SCRIPT_DIR, "simulation_telemetry_dataset.csv")
MARKER_PATH = os.path.join(SCRIPT_DIR, ".active_attack.json")

def get_current_ground_truth():
    try:
        if os.path.exists(MARKER_PATH):
            with open(MARKER_PATH, "r") as f:
                data = json.load(f)
                if data.get("active", False):
                    return data.get("attack", "Benign")
    except Exception:
        pass
    return "Benign"

def record_data(port=14552, output_csv=CSV_PATH, rate_hz=10):
    try:
        from pymavlink import mavutil
    except ImportError:
        print("[!] pymavlink not installed. Please run: .venv/bin/pip install pymavlink")
        sys.exit(1)

    print("=" * 75)
    print("       UAV FLIGHT & ATTACK TELEMETRY DATA GATHERER")
    print("=" * 75)
    print(f"[*] Listening on port      : udp:0.0.0.0:{port}")
    print(f"[*] Logging destination    : {output_csv}")
    print(f"[*] Sampling frequency     : ~{rate_hz} Hz")
    print(f"[*] Ground truth tracker   : {MARKER_PATH}")
    print("=" * 75)

    master = mavutil.mavlink_connection(f"udpin:0.0.0.0:{port}")
    print("[*] Waiting for MAVLink heartbeat...")
    master.wait_heartbeat()
    print(f"[+] Heartbeat received from system {master.target_system}! Recording started.")

    # Initialize CSV header if needed
    header = (
        "timestamp,flight_time,ground_truth,"
        "phys_height,phys_vx,phys_vy,phys_vz,"
        "phys_pitch,phys_roll,phys_yaw,phys_a_mag,"
        "cyb_pkt_rate,cyb_inter_arrival,cyb_bad_packets\n"
    )
    if not os.path.exists(output_csv):
        with open(output_csv, "w") as f:
            f.write(header)

    state = {
        'height': 0.0, 'vx': 0.0, 'vy': 0.0, 'vz': 0.0,
        'pitch': 0.0, 'roll': 0.0, 'yaw': 0.0, 'a_mag': 0.0
    }
    prev_state = None
    prev_time = None
    last_arrival = None
    arrival_times = deque(maxlen=200)
    bad_packet_times = deque(maxlen=100)

    start_time = time.time()
    last_log_time = 0.0
    sample_interval = 1.0 / rate_hz
    recorded_rows = 0

    try:
        while True:
            msg = master.recv_match(blocking=True, timeout=0.1)
            now = time.time()

            if msg is not None:
                arrival_times.append(now)
                dt_packet = (now - last_arrival) if last_arrival else 0.05
                last_arrival = now

                msg_type = msg.get_type()
                if msg_type == 'BAD_DATA':
                    bad_packet_times.append(now)
                elif msg_type == 'ATTITUDE':
                    state['pitch'] = math.degrees(msg.pitch)
                    state['roll'] = math.degrees(msg.roll)
                    state['yaw'] = math.degrees(msg.yaw)
                elif msg_type == 'GLOBAL_POSITION_INT':
                    state['height'] = msg.relative_alt / 1000.0
                    state['vx'] = msg.vx / 100.0
                    state['vy'] = msg.vy / 100.0
                    state['vz'] = msg.vz / 100.0
                elif msg_type == 'VFR_HUD':
                    if state['height'] == 0.0:
                        state['height'] = msg.alt
                        state['vz'] = msg.climb

            # Periodic CSV logging
            if now - last_log_time >= sample_interval:
                dt_sample = (now - prev_time) if prev_time else sample_interval
                prev_time = now

                # Compute acceleration magnitude
                if prev_state and dt_sample > 0:
                    ax = (state['vx'] - prev_state['vx']) / dt_sample
                    ay = (state['vy'] - prev_state['vy']) / dt_sample
                    az = (state['vz'] - prev_state['vz']) / dt_sample
                    state['a_mag'] = math.sqrt(ax**2 + ay**2 + az**2)
                else:
                    state['a_mag'] = 0.0

                # Compute network metrics
                if len(arrival_times) > 5:
                    span = arrival_times[-1] - arrival_times[0]
                    pkt_rate = (len(arrival_times) - 1) / span if span > 0 else 0.0
                else:
                    pkt_rate = 1.0 / dt_packet if dt_packet > 0 else 10.0

                bad_count = sum(1 for t in bad_packet_times if now - t <= 1.0)
                gt = get_current_ground_truth()
                flight_time = now - start_time

                # Append to CSV
                with open(output_csv, "a") as f:
                    row = [
                        f"{now:.3f}", f"{flight_time:.3f}", gt,
                        f"{state['height']:.3f}", f"{state['vx']:.3f}", f"{state['vy']:.3f}", f"{state['vz']:.3f}",
                        f"{state['pitch']:.2f}", f"{state['roll']:.2f}", f"{state['yaw']:.2f}", f"{state['a_mag']:.3f}",
                        f"{pkt_rate:.2f}", f"{dt_packet:.4f}", str(bad_count)
                    ]
                    f.write(",".join(row) + "\n")

                recorded_rows += 1
                prev_state = state.copy()
                last_log_time = now

                # Status line in terminal
                gt_tag = f"[{gt}]" if gt == "Benign" else f"\033[91m[{gt} ATTACK ACTIVE]\033[0m"
                print(f"  [RECORDING] t={flight_time:5.1f}s | Status: {gt_tag:<25} | Alt: {state['height']:5.1f}m | Rate: {pkt_rate:5.1f} pkts/s | Samples: {recorded_rows}", end='\r')

    except KeyboardInterrupt:
        print(f"\n\n[+] Recording stopped. Saved {recorded_rows} synchronized telemetry samples to {output_csv}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="UAV Flight Telemetry Data Gatherer")
    parser.add_argument("--port", type=int, default=14552, help="UDP port listening for MAVLink telemetry")
    parser.add_argument("--out", type=str, default=CSV_PATH, help="Output CSV file path")
    parser.add_argument("--rate", type=int, default=10, help="Sampling frequency (Hz)")
    args = parser.parse_args()

    record_data(port=args.port, output_csv=args.out, rate_hz=args.rate)
